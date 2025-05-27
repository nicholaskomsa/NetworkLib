#include "Gpt2.h"

#include <fstream>

#include <boost/json.hpp>


using namespace NetworkLib;


void GPT2::Forward::setup() {

	mParallelInput.setup({}, mTestInputSize, 256);
	mParallelI.setup(Floats{}, mTestInputSize, 8);

	load();
}
void GPT2::Forward::load() {

	//the gpt2 model is found on huggingface website: https://huggingface.co/openai-community/gpt2?show_file_info=model.safetensors
	//we need to load gpt2 from file and create the activation space
	//the order of the activation space affects speed same as the gpt model space itself
	//so the layout follows forward process specifically
	//the gpt2 model is loaded from file - in a memory layout openai saved to file
	//but it could possibly be made faster still by moving from spanT to span?

	constexpr float floatSize = sizeof(float);

	auto readFile = [&]() {

		//the gpt file consists of a header segment, a json string, and a floatspace segment which is the remaining of the file
		auto fileName = std::format("{}model.safeTensors", mFilePath);

		std::println("Reading file: {}", fileName);

		std::ifstream fin(fileName, std::ios::in | std::ios::binary);

		if (!fin)
			Error::fileNotFound(fileName);

		std::uint64_t headerSize;
		fin.read(reinterpret_cast<char*>(&headerSize), sizeof(headerSize));

		std::string header; header.resize(headerSize);
		fin.read(header.data(), header.size());

		std::streampos current = fin.tellg()
			, end = fin.seekg(0, std::ios::end).tellg(); //get length of file
		fin.seekg(current);

		std::streamoff floatsSize = static_cast<std::streamoff>(end - current) / floatSize;
		mTensorSpace.resize(floatsSize);

		fin.read(reinterpret_cast<char*>(mTensorSpace.data()), floatsSize * floatSize);

		constexpr std::size_t knownFileFloatSize = 548090880 / floatSize;

		assert(knownFileFloatSize == floatsSize);

		fin.close();
		std::puts("file read done");

		return header;
		};

	auto header = readFile();

	boost::json::value j = boost::json::parse(header);
	std::size_t floatsUsed = 0;

	mActivationSpace.resize(mSeqModel + (mSeqModel * 7 + mSeqModel3 + mSeqSeqHead * 2 + mSeqModel4 * 3) * mAttnLayers.size() + mSeqModel + mSeqVocab * 2);
	auto activationSpace = mActivationSpace.begin();

	auto readTensorByName = [&](std::string_view name) {

		auto& obj = j.at(name);
		auto& offsets = obj.at("data_offsets").as_array();
		auto a = offsets.front().as_int64() / floatSize, b = offsets.back().as_int64() / floatSize;

		auto start = std::next(mTensorSpace.begin(), a);
		auto end = std::next(mTensorSpace.begin(), b);
		auto size = std::distance(start, end);

		Tensor::View tensorView(start, size);

		floatsUsed += size;

		auto& shape = obj.at("shape").as_array();
		auto dimensions = shape.size();

		auto getDim = [&](auto i) {
			return std::size_t(shape[i].as_int64());
			};

		Tensor tensor;

		auto expectedSize = 0ULL;
		switch (dimensions) {
		case 1:
			tensor = { tensorView, getDim(0) };
			expectedSize = tensor.size();
			break;
		case 2:
			tensor = { tensorView, getDim(0), getDim(1) };
			expectedSize = tensor.size2D();
			break;
		case 3:
			tensor = { tensorView, getDim(0), getDim(1), getDim(2) };
			expectedSize = tensor.size3D();
			break;
		case 4:
			tensor = { tensorView, getDim(0), getDim(1), getDim(2), getDim(3) };
			expectedSize = tensor.size4D();
			break;
		}
		assert(expectedSize == size);

		return tensor;
		};

	mWpeWeight = readTensorByName("wpe.weight");
	mWteWeight = readTensorByName("wte.weight");

	mWActivations = { activationSpace, mDSeq, mDModel };

	for (auto i : std::views::iota(0ULL, mAttnLayers.size()))
		mAttnLayers[i].load(readTensorByName, i, activationSpace);

	mFinalLayer.load(readTensorByName("ln_f.bias"), readTensorByName("ln_f.weight"), activationSpace);

	mUnembedActivations = { activationSpace, mDSeq, mDVocab };
	mUnembedActivationsSoftmax = { activationSpace, mDSeq, mDVocab };

	assert(floatsUsed == mTensorSpace.size());
	assert(activationSpace == mActivationSpace.end());

	std::puts("Tensors read successfully");
}

void GPT2::Forward::embedInput(std::size_t i, Token token) {

	Tensor::View wte, wpe, wActivations;
	//weight token embed, weight position embed
	//generates an activation combining token and position

	wte = mWteWeight.view(token);
	wpe = mWpeWeight.view(i);
	wActivations = mWActivations.view(i);

	for (const auto& [a, t, p] : std::views::zip(wActivations, wte, wpe))
		a = t + p;
}
void GPT2::Forward::embedInputs(TokensView tokens) {

	//for each token, generate a position+token embedding
	//as a setup step before forward

	mParallelInput([&](auto& section) {

		for (auto i : section.mIotaView)
			embedInput(i, tokens[i]);

		});
}
void GPT2::Forward::unEmbedOutput(std::size_t i) {

	//after forward, generate the probability of a specific token

	Tensor::View input, wte, output;
	//weight token embed seen in embedInput
	//input is the output of earlier forward process

	input = mFinalLayer.getActivations().view(i);
	output = mUnembedActivations.view(i);

	mParallelI.section(mUnembedActivations.mY);
	mParallelI([&](Parallel::Section& section) {

		for (auto m : section.mIotaView) {

			wte = mWteWeight.view(m);

			float sum = 0.0f;

			for (const auto& [in, w] : std::views::zip(input, wte))
				sum += in * w;

			output[m] = sum;
		}

		});
}
void GPT2::Forward::unEmbedOutputs() {

	//after forward, generate each token probability
	mUnembedTime.accumulateTime([&]() {

		mParallelInput([&](auto& section) {

			Tensor::ConstView input, wte;
			Tensor::View output;

			for (auto i : section.mIotaView) {

				//weight token embed seen in embedInput
				//input is the output of earlier forward process

				input = mFinalLayer.getActivations().constView(i);
				output = mUnembedActivations.view(i);

				for (auto m : std::views::iota(0ULL, output.size())) {

					wte = mWteWeight.constView(m);

					float dot = 0.0f;

					for (const auto& [in, w] : std::views::zip(input, wte))
						dot += in * w;

					output[m] = dot;
				}
			}

			});
		});

}

GPT2::Token GPT2::Forward::feedForward(TokensView tokens) {

	//feedForward will feed all tokens fresh into the network, up to dseq number of tokens
	//tokens max size == mTestInputSize
	//if larger than maxsize, should scroll to tail-mTestInputSize, it is assumed you did this earlier

	//the parallel process operates over all tokens simultaneously
	//for each input, there is large matrix and vector work, and this is futher parallelised
	//the best case performance is with fewer tokens, such as a short english sentence
	//and worst case performace when tokens size = mDSeq, the maximum model size

	mParallelInput.section(tokens.size());
	std::size_t m = tokens.size() - 1;

	embedInputs(tokens);

	Tensor* input = &mWActivations;

	mLayersTime.accumulateTime([&]() {

		for (auto& layer : mAttnLayers)
			input = &layer.forward(*input, mParallelInput);

		});

	mFinalLayer.normalise(*input, mParallelInput);

	unEmbedOutputs();

	Token predicted = getPrediction(m);

	return predicted;
}
GPT2::Token GPT2::Forward::feedMore(TokensView tokens) {

	//because many short english sentences are small, they are way smaller than the maximum model size of mDSeq
	//and in fact predictions fit inside basically "unallocated" "more" model space, so making predictions is very fast
	//until the model space is filled, at which feedMore becomes unfunctional
	//the model has run out of "more" prediction space. 
	//At this point you need to make external decisions about your input data, such as scrolling it to create "more" space.
	//feedMore acts like all previous tokens are valid, and the back token, needs processed only
	//identical to feedForward except for parallel processing which is instead oriented toward a single sample

	int i = tokens.size() - 1;
	embedInput(i, tokens.back());

	Tensor* input = &mWActivations;

	for (auto& layer : mAttnLayers)
		input = &layer.forward(i, *input, mParallelI);

	mFinalLayer.normalise(i, *input);

	unEmbedOutput(i);

	Token predicted = getPrediction(i);

	return predicted;
}
float GPT2::Forward::crossEntropyLoss(TokensView nextTokens) {

	mParallelInput.section(nextTokens.size());

	mParallelInput([&](auto& section) {

		Tensor::ConstView unembed;
		Tensor::View unembedSoftmax;

		for (auto i : section.mIotaView) {

			unembed = mUnembedActivations.view(i);
			unembedSoftmax = mUnembedActivationsSoftmax.view(i);

			GPT2::softmax(mUnembedActivations.mY - 1, unembed, unembedSoftmax);
		}

		});

	Tensor::ConstView unembedSoftmax;
	float loss = 0.0f;
	for (auto i : std::views::iota(0ULL, nextTokens.size())) {

		Token expected = nextTokens[i];
		unembedSoftmax = mUnembedActivationsSoftmax.view(i);

		float expectedSoftmax = unembedSoftmax[expected];

		loss += -std::logf(expectedSoftmax);
	}

	loss /= nextTokens.size();

	return loss;
}
GPT2::Token GPT2::Forward::getPrediction(std::size_t i) const {

	//unembed activations is the entire sequence of all tokens, each a prediction of its probability 
	//the highest probability is the predicted token here, but other tokens may also have some lower possibility

	auto unembedActivations = mUnembedActivations.constView(i);
	auto selected = std::max_element(unembedActivations.begin(), unembedActivations.end());
	Token predicted = std::distance(unembedActivations.begin(), selected);

	return predicted;
}