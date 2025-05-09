#include "Gpt2.h"

#include <system_error>
#include <sstream>
#include <fstream>
#include <cassert>
#include <ranges>
#include <map>
#include <set>

#include <boost/json.hpp>

using namespace NetworkLib;

Parallel GPT2::AttnLayer::mParallelHeads( mHeadNum, mHeadNum);

const float GPT2::AttnLayer::r_sqrtHeadsPerDModel = 1.0f / std::sqrtf(GPT2::mHeadsPerDModel);
const float GPT2::MLP::r_sqrt2Pi = 1.0f / std::sqrtf(2.0f * std::numbers::pi);

GPT2::Error::Error(std::errc code, const std::string& message) : std::system_error(int(code), std::generic_category(), message) {}

void GPT2::Error::fileNotFound(std::string_view fileName) {
	throw Error(std::errc::no_such_file_or_directory, std::format("File Not Found: {}", fileName));
}
void GPT2::Error::wordNotFound(std::string_view word) {
	throw Error(std::errc::invalid_argument, std::format("Translator: Word Not Found: {}", word));
}
void GPT2::Error::tokenNotFound(Token token) {
	throw Error(std::errc::invalid_argument, std::format("Translator: Token Not Found: {}", token));
}

void GPT2::Translator::load() {

	//we need to load the translator file, written by raffK project.
	//in raffK project, the gpt2 python code exported all of its vocabulary to file
	//we read this file to generate our vocabulary knowledge, "enc" file from raffK project

	auto readFile = [&]() {

		//enc file https://github.com/rkaehn/gpt-2/blob/main/assets/enc
		auto fileName = std::format("{}enc", mFilePath);

		std::println("Reading file: {}", fileName);

		std::ifstream fin(fileName, std::ios::in | std::ios::binary);

		if (!fin)
			GPT2::Error::fileNotFound(fileName);

		using Offset = std::pair<std::uint32_t, std::uint32_t>;
		std::vector<Offset> offsets;

		offsets.resize(mDVocab);

		constexpr auto mDenseWordsSize = 321428;
		mDenseWords.resize(mDenseWordsSize);

		fin.read(reinterpret_cast<char*>(offsets.data()), mDVocab * sizeof(Offset));
		fin.read(mDenseWords.data(), mDenseWordsSize);

		fin.close();
		std::puts("file read done");

		return offsets;
		};

	auto offsets = readFile();
	
	for (Token token : std::views::iota(0ULL, offsets.size())) {

		auto& [offset, size] = offsets[token];

		std::string_view word(mDenseWords.data() + offset, size);

		mWordMap.insert({ word, token });
	}
}
std::string GPT2::Translator::decode( TokensView tokens) const {

	//concat a series of tokens into a string

	std::stringstream sstr;

	for (auto token : tokens) 
		sstr << getWord(token);
	
	return sstr.str();
}
std::string GPT2::Translator::decode(Token token) const {
	return std::string( getWord(token) );
}
GPT2::Translator::Word GPT2::Translator::getWord(Token token) const {

	//this function will take a token and convert it into a gpt word
	//this would only fail if token is larger than vocab size

	auto found = mWordMap.right.find(token);
	if (found == mWordMap.right.end())
		Error::tokenNotFound(token);

	return found->get_left();
}
GPT2::Tokens GPT2::Translator::encode(std::string_view remaining) const {

	//take a string potentially containing many tokens, and generate all tokens
	//many gpt "words" begin with a white space " ", 
	//and this is the pause in vocabulary rather than delimiters of " " token between words
	//therefore, " Hello" is one token and " World" is another, and the sentence " Hello World" converts to two tokens only.
	//"Hello World" also converts to two tokens, many words have a " " variant and others do not
	//auto tokens = gpt2.mTranslator.encode(" Hello World");
	//std::println("Tokens: {}", tokens.size()); == 2

	//determine the size categories of the words in the vocabulary
	static std::set<std::size_t > wordSizes;
	if(wordSizes.empty() )
		for (auto& [word, token] : mWordMap.left)
			wordSizes.insert(word.size());

	Tokens tokens;

	auto getToken = [&]() {

		auto matchVocabWord = [&]() {

			for (auto size : wordSizes | std::views::reverse) {

				if (size > remaining.size()) continue;

				Word testWord = remaining | std::views::take(size);

				auto wordFound = mWordMap.left.find(testWord);

				if (wordFound != mWordMap.left.end()) 
					return *wordFound;
			}

			};

		const auto [word, token] = matchVocabWord();

		tokens.push_back(token);
		remaining = remaining.substr(word.size());

		return remaining.size();
		};

	while (getToken());

	return tokens;
}
GPT2::Token GPT2::Translator::getToken(std::string_view word) const {

	//convert a word to a token
	//this will only fail if for some reason word is not a true GPT "word" found in vocabulary

	auto found = mWordMap.left.find(word);
	if (found == mWordMap.left.end())
		Error::wordNotFound(word);

	return found->get_right();
}

void GPT2::TestData::load() {

	//this file loads the concerning first citizen test data set, the first 64 tokens is used by Diagnostics
	//this file is found at raffK project, "data"

	auto readFile = [&]() {

		//data file https://github.com/rkaehn/gpt-2/blob/main/assets/data
		auto fileName = std::format("{}data", mFilePath);

		std::println("Reading file: {}", fileName);

		std::ifstream fin(fileName, std::ios::in | std::ios::binary);

		if (!fin)
			Error::fileNotFound(fileName);

		std::streampos end = fin.seekg(0, std::ios::end).tellg();

		constexpr auto tokenSize = sizeof(Token);
		std::streamoff tokensSize = static_cast<std::streamoff>(end) / tokenSize;
		mTokens.resize(tokensSize);

		fin.seekg(0);
		fin.read(reinterpret_cast<char*>(mTokens.data()), tokensSize * tokenSize);

		fin.close();
		std::puts("file read done");

		};

	readFile();

	std::println(std::cout, "Data Tokens size: {}", mTokens.size());
}

void GPT2::Forward::setup() {

	mParallelInput.setup({}, mTestInputSize, 64);
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

	for (auto i : std::views::iota( 0ULL, mAttnLayers.size()))
		mAttnLayers[i].load(readTensorByName, i, activationSpace);

	mFinalLayer.load(readTensorByName("ln_f.bias"), readTensorByName("ln_f.weight"), activationSpace);

	mUnembedActivations = {activationSpace, mDSeq, mDVocab };
	mUnembedActivationsSoftmax = { activationSpace, mDSeq, mDVocab };

	assert(floatsUsed == mTensorSpace.size());
	assert(activationSpace == mActivationSpace.end());

	std::puts("Tensors read successfully");
}
void GPT2::forward(std::size_t i, const Tensor& inputTensor, Tensor& outputTensor, const Tensor& weightTensor, const Tensor& biasTensor, Parallel& parallel) {
	
	//this is a "matrix * vector + vector" or "fully connected" aka "forward" o += w * i + b

	Tensor::ConstView input = inputTensor.constView(i)
		, b = biasTensor.constView();
	Tensor::View output = outputTensor.view(i);

	//a fully connected input and output with a bias
	//identical to forward except for paralleled for i sample

	std::copy(b.begin(), b.end(), output.begin());

	parallel([&](Parallel::Section& section) {

		auto& outputs = std::any_cast<Floats&>(section.mAny);
		outputs.clear();
		outputs.resize(output.size(), 0.0f);

		auto& [first, second] = section.mOffsets;
		for (auto m : std::views::iota(first, second)) {

			const auto& in = input[m];

			for (const auto& [o, w] : std::views::zip(outputs, weightTensor.constView(m)))
				o += w * in;
		}

		}, [&](Parallel::Section& section) {

			auto& outputs = std::any_cast<Floats&>(section.mAny);

			for (auto m : std::views::iota(0ULL, output.size()))
				output[m] += outputs[m];
				
			});
}
void GPT2::forward(const Tensor& inputTensor, Tensor& outputTensor, const Tensor& weightTensor, const Tensor& biasTensor, Parallel& parallel) {
	
	//each input is doing "matrix * vector + vector" or is "fully connected"

	parallel([&](Parallel::Section& section) {

		Tensor::ConstView input, b = biasTensor.constView();
		Tensor::View output;

		auto& [first, second] = section.mOffsets;
		for (auto i : std::views::iota( first, second)) {

			//a fully connected input and output with a bias

			input = inputTensor.constView(i);
			output = outputTensor.view(i);

			std::copy(b.begin(), b.end(), output.begin());

			float in;
			Tensor::ConstView weights;

			for (auto m : std::views::iota(0ULL, input.size())) {

				in = input[m];
				weights = weightTensor.constView(m);

				for (const auto& [o, w] : std::views::zip(output, weights))
					o += w * in;
			}
		}

		});
}
void GPT2::softmax(std::size_t i, Tensor::ConstView input, Tensor::View output) {

	const auto ibegin = input.begin(), iend = ibegin + 1 + i;
	const auto obegin = output.begin(), oend = obegin + 1 + i;

	const auto softmaxMax = *std::max_element(ibegin, iend);

	std::transform(std::execution::seq, ibegin, iend, obegin, [&](auto& in) {
		return std::expf(in - softmaxMax);
		});

	const auto softmaxSum = std::reduce(obegin, oend)
		, r_softmaxSum = 1.0f / softmaxSum;

	std::transform(std::execution::seq, obegin, oend, obegin, [&](auto& o) {
		return o * r_softmaxSum;
		});
}

void GPT2::MLP::forward(const Tensor& input, Parallel& parallel) {

	GPT2::forward(input, mCFCActivations, mCFCWeight, mCFCBias, parallel);

	//an activation function, gelu, is applied here and cdf is cached

	auto gelu = [&]() {
		parallel([&](Parallel::Section& section) {

			const auto& [first, second] = section.mOffsets;
			for (auto i : std::views::iota(first, second)) {

				auto cfcActivations = mCFCActivations.view(i);
				auto gelu = mGeluActivations.view(i);
				auto cdf = mGeluCDF.view(i);

				for (const auto& [cfc, gelu, cdf] : std::views::zip(cfcActivations, gelu, cdf)) {

					cdf = 0.5f * (1.0f + std::erff(cfc * r_sqrt2));
					gelu = cfc * cdf;
				}
			}

			});
		};
	gelu();

	GPT2::forward(mGeluActivations, mCProjActivations, mCProjWeight, mCProjBias, parallel);
}
void GPT2::MLP::forward(std::size_t i, const Tensor& input, Parallel& parallel) {

	//should be pre-sectioned earlier
	GPT2::forward(i, input, mCFCActivations, mCFCWeight, mCFCBias, parallel);

	Tensor::View mlpActivations = mCFCActivations.view(i), geluActivations = mGeluActivations.view(i);

	std::transform(std::execution::par_unseq, mlpActivations.begin(), mlpActivations.end(), geluActivations.begin(),
		[&](auto x) {
			return x * 0.5f * (1.0f + std::erff(x * r_sqrt2));
		});

	parallel.section(mGeluActivations.mY);
	GPT2::forward(i, mGeluActivations, mCProjActivations, mCProjWeight, mCProjBias, parallel);
}
const Tensor& GPT2::MLP::getCProjActivations() const {
	return mCProjActivations;
}
void GPT2::MLP::load(auto&& cfcBias, auto&& cfcWeight, auto&& cProjBias, auto&& cProjWeight, Floats::iterator& activationSpace) {

	mCFCBias = std::move(cfcBias);
	mCFCWeight = std::move(cfcWeight);
	mCProjBias = std::move(cProjBias);
	mCProjWeight = std::move(cProjWeight);
	mCFCActivations = { activationSpace, mDSeq, mDModel4 };
	mGeluActivations = { activationSpace, mDSeq, mDModel4 };
	mGeluCDF = { activationSpace, mDSeq, mDModel4 };
	mCProjActivations = { activationSpace, mDSeq, mDModel };
}

Tensor& GPT2::LinearLayer::getActivations() {
	return mActivations;
}
void GPT2::LinearLayer::load(Tensor&& bias, Tensor&& weight, Floats::iterator& activationSpace) {

	mBias = std::move(bias);
	mWeight = std::move(weight);
	mActivations = { activationSpace, mDSeq, mDModel };

	mMean.resize(mTestInputSize);
	mRStdDev.resize(mTestInputSize);
}
void GPT2::LinearLayer::normalise(std::size_t m, const Tensor& input) {

	//normalise is a case of linear-forward 
	//where a normalisation preprocess of input followed by a linear connect
	//the connect is linear rather than full-connect (+=), between normalized input and output
	//o = norm * w + b

	Tensor::ConstView in = input.constView(m);
	Tensor::View out = mActivations.view(m);

	const auto mean = std::reduce(in.begin(), in.end()) / in.size();
	mMean[m] = mean;

	auto meanDiffSq = std::reduce(in.begin(), in.end(), 0.0f,
		[&](auto sum, auto x) {
			auto diff = x - mean;
			return sum + diff * diff;
		}) / in.size();

	const auto r_stdDev = 1.0f / std::sqrt(meanDiffSq);
	mRStdDev[m] = r_stdDev;

	Tensor::ConstView bias = mBias.constView(), weight = mWeight.constView();

	float norm = 0;
	for (const auto& [i, w, b, o] : std::views::zip(in, weight, bias, out)) {
		norm = (i - mean) * r_stdDev;
		o = norm * w + b;
	}
}
void GPT2::LinearLayer::normalise(const Tensor& input, Parallel& parallel) {

	parallel([&](auto& sections) {

		auto& [first, second] = sections.mOffsets;
		for( auto m : std::views::iota(first,second))
			normalise(m, input);

		});
}

void GPT2::AttnLayer::calculateQKAtten(std::size_t headOffset, std::size_t i, Tensor::View attnOut) {

	//q-head is taken from the q-tensor, Q is for query
	//k-head is taken from the k-tensor, K is for key
	//q and k are multiplied together and summed and scaled

	const auto qOffset = mQOffset + headOffset;
	Tensor::ConstView kh, qh = Tensor::constField( mCAttnActivations.view(i), qOffset, mHeadsPerDModel );

	const auto kOffset = mKOffset + headOffset;
	float dot;

	for (auto m : std::views::iota(0ULL, i+1)) {

		kh = Tensor::constField( mCAttnActivations.view(m), kOffset, mHeadsPerDModel );
		dot = 0.0f;

		for (const auto& [q, k] : std::views::zip(qh, kh))
			dot += q * k;

		attnOut[m] = dot * r_sqrtHeadsPerDModel;
	};
}
void GPT2::AttnLayer::calculateVAtten(std::size_t headOffset, std::size_t i, Tensor::ConstView attnOutSoftmax) {

	//z-head is taken from the z-tensor, Z is for Activation
	//v-head is taken from the v-tensor, V is for Value
	//for each word, accumulate the v-attention into the z-head
	//v-attention is based on attention-softmax and v-head

	Tensor::View zh = Tensor::field( mAttnZ.view(i), headOffset, mHeadsPerDModel );
	const auto vOffset = mVOffset + headOffset;
	Tensor::ConstView vh;
	float factor;

	for (auto m : std::views::iota(0ULL, i+1)) {

		vh = Tensor::constField(mCAttnActivations.view(m), vOffset, mHeadsPerDModel );
		factor = attnOutSoftmax[m];

		for (const auto& [z, v] : std::views::zip(zh, vh))
			z += v * factor;
	}
}
void GPT2::AttnLayer::multiHeadedAttn(std::size_t m) {

	//attention is a "multi-headed" process, and the same "Attention" operations are repeated for each head.
	//on each head, over the series of input, perform an accumulative attention process
	//this process generates a sort of diagonal matrix where there are lengths of n, n+1, n+2, n+3 opposing zeros
	//for each head, there are three types of segments of data, Q=query, K=key, and V=value
	//all of these segments and heads are located adjacent to each other in a vector
	//and referenced using offsets mQOffset, mKOffset, mVOffset, and headOffset
	//q and k and v are generated by the previous forward

	mParallelHeads([&](Parallel::Section& section) {

		const auto& [first, second] = section.mOffsets;
		for (auto h : std::views::iota( first, second)) {

			const auto headOffset = h * mHeadsPerDModel;

			for(auto i : std::views::iota(0ULL, m + 1) ) {

				Tensor::View attnOut = mAttnActivations.view(h, i)
					, attnOutSoftmax = mAttnSoftmaxActivations.view(h, i);

				calculateQKAtten(headOffset, i, attnOut);
				GPT2::softmax(i, attnOut, attnOutSoftmax);
				calculateVAtten(headOffset, i, attnOutSoftmax);

			}
		}

		});
}
void GPT2::AttnLayer::attention(std::size_t m ) {

	Tensor::View z = mAttnZ.view(m);
	std::fill(z.begin(), z.end(), 0.0f);

	multiHeadedAttn(m);
}
void GPT2::AttnLayer::attention(Parallel& parallel) {

	auto m = parallel.mSize - 1;

	auto activations = mAttnZ.viewBlock(m);

	//activations z cleared here, attention generates mAttnZ (activations)
	std::fill(activations.begin(), activations.end(), 0.0f);

	multiHeadedAttn(m);
}
void GPT2::AttnLayer::residual(std::size_t i, const Tensor& inputTensor, const Tensor& projectionTensor, Tensor& residualTensor) {
	
	//a residual is the sum of the input and the projection

	for (const auto& [out, p, in] : std::views::zip(residualTensor.view(i), projectionTensor.constView(i), inputTensor.constView(i)))
		out = p + in;
}
void GPT2::AttnLayer::residual(const Tensor& inputTensor, const Tensor& projectionTensor, Tensor& residualTensor, Parallel& parallel) {

	parallel([&](auto& sections) {

		auto& [first, second] = sections.mOffsets;
		for( auto i : std::views::iota( first, second))
			residual(i, inputTensor, projectionTensor, residualTensor);

		});
}
Tensor& GPT2::AttnLayer::forward(const Tensor& inputTensor, Parallel& parallel) {

	mL1.normalise(inputTensor, parallel);
	GPT2::forward(mL1.getActivations(), mCAttnActivations, mCAttnWeight, mCAttnBias, parallel);

	attention(parallel);

	GPT2::forward(mAttnZ, mCProjActivations, mCProjWeight, mCProjBias, parallel);

	residual(inputTensor, mCProjActivations, mResidualActivation1, parallel);

	mL2.normalise(mResidualActivation1, parallel);

	mMLP.forward(mL2.getActivations(), parallel);

	residual(mResidualActivation1, mMLP.getCProjActivations(), mResidualActivation2, parallel);

	return mResidualActivation2;
}
Tensor& GPT2::AttnLayer::forward(std::size_t i, const Tensor& inputTensor, Parallel& parallel) {

	//a specific attention layer is a series of "forward", normalise, residual, MLP, and "attention" process.
	//where forward is aka a "large matrix" or "fully connected" operation

	parallel.section(mDModel);

	mL1.normalise(i, inputTensor);
	GPT2::forward(i, mL1.getActivations(), mCAttnActivations, mCAttnWeight, mCAttnBias, parallel);

	attention(i);

	GPT2::forward(i, mAttnZ, mCProjActivations, mCProjWeight, mCProjBias, parallel);

	residual(i, inputTensor, mCProjActivations, mResidualActivation1);

	mL2.normalise(i, mResidualActivation1);

	mMLP.forward(i, mL2.getActivations(), parallel);

	residual(i, mResidualActivation1, mMLP.getCProjActivations(), mResidualActivation2);

	return mResidualActivation2;
}
void GPT2::AttnLayer::load(ReadTensorFunctor&& readTensorByName, std::size_t layerIdx, Floats::iterator& activationSpace) {

	auto layer = std::format("h.{}.", layerIdx);

	auto attnTensor = [&](const auto& name) {
		return readTensorByName(std::format("{}attn.{}", layer, name));
		};

	mBias = attnTensor("bias");
	mCAttnBias = attnTensor("c_attn.bias");
	mCAttnWeight = attnTensor("c_attn.weight");
	mCProjBias = attnTensor("c_proj.bias");
	mCProjWeight = attnTensor("c_proj.weight");

	mCAttnActivations = { activationSpace, mDSeq, mDModel3 };
	mAttnActivations = { activationSpace, mDSeq, mDSeq, mHeadNum };

	mAttnSoftmaxActivations = { activationSpace, mDSeq, mDSeq, mHeadNum };
	mAttnZ = { activationSpace, mDSeq, mDModel };

	auto linearTensor = [&](auto idx, const auto& name) {
		return readTensorByName(std::format("{}ln_{}.{}", layer, idx, name));
		};

	mL1.load(linearTensor(1, "bias"), linearTensor(1, "weight"), activationSpace);

	mCProjActivations = { activationSpace, mDSeq, mDModel };
	mResidualActivation1 = { activationSpace, mDSeq, mDModel };

	mL2.load(linearTensor(2, "bias"), linearTensor(2, "weight"), activationSpace);

	auto mlpTensor = [&](const auto& name) {
		return readTensorByName(std::format("{}mlp.{}", layer, name));
		};

	mMLP.load(mlpTensor("c_fc.bias"), mlpTensor("c_fc.weight"), mlpTensor("c_proj.bias"), mlpTensor("c_proj.weight"), activationSpace);

	mResidualActivation2 = { activationSpace, mDSeq, mDModel };

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

	mParallelInput([&](auto& sections) {

		auto& [first, second] = sections.mOffsets;
		for( auto i : std::views::iota(first, second))
			embedInput(i, tokens[i]);

		});
}
void GPT2::Forward::unEmbedOutput(std::size_t i ) {

	//after forward, generate the probability of a specific token

	Tensor::View input, wte, output;
	//weight token embed seen in embedInput
	//input is the output of earlier forward process

	input = mFinalLayer.getActivations().view(i);
	output = mUnembedActivations.view(i);

	mParallelI.section(mUnembedActivations.mY);
	mParallelI([&](Parallel::Section& section) {

		auto [first, second] = section.mOffsets;
		for( auto m : std::views::iota(first, second)){

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

	mParallelInput([&](auto& section) {

		Tensor::ConstView input, wte;
		Tensor::View output;

		auto& [first, second] = section.mOffsets;
		for (auto i : std::views::iota(first, second)) {

			//weight token embed seen in embedInput
			//input is the output of earlier forward process

			input = mFinalLayer.getActivations().view(i);
			output = mUnembedActivations.view(i);

			for (auto m : std::views::iota(0ULL, output.size())) {

				wte = mWteWeight.view(m);

				float dot = 0.0f;

				for (const auto& [in, w] : std::views::zip(input, wte))
					dot += in * w;

				output[m] = dot;
			}
		}

		});
}
void GPT2::setup() {

	//we need to load our gpt data from file and also load the token translator file

	mForward.setup();

	mTranslator.load();
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

	for (auto& layer : mAttnLayers)
		input = &layer.forward(*input, mParallelInput);

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
		
		Tensor::View unembed, unembedSoftmax;

		auto& [first, second] = section.mOffsets;
		for (auto i : std::views::iota(first, second)) {

			unembed = mUnembedActivations.view(i);
			unembedSoftmax = mUnembedActivationsSoftmax.view(i);

			GPT2::softmax(mUnembedActivations.mY - 1, unembed, unembedSoftmax);
		}

		});

	Tensor::View unembedSoftmax;
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
void GPT2::Diagnostics::firstCitizenTest64() {

	//this test is used to check the specific values of feed forward for correctness

	auto test = [&](auto& gpt2, Token predicted) {
		//when concerning first citizen test data, mData,  this checksum tests the test size which is 64 tokens
		assert(64 == mTestInputSize);

		auto getSum = [&](const auto& tensor) {
			return std::int64_t(std::reduce(tensor.mTensor.begin(), tensor.mTensor.end(), double(0.0)));
			};

		auto& forward = gpt2.mForward;

		auto testEmbed = [&] {
			assert(-30 == getSum(forward.mWActivations));
			};

		auto testFrontLayer = [&]() {

			const auto& layer = forward.mAttnLayers.front();
			assert(-334 == getSum(layer.mL1.mActivations));
			assert(-3325 == getSum(layer.mCAttnActivations));
			assert(454 == getSum(layer.mAttnZ));
			assert(389 == getSum(layer.mCProjActivations));
			assert(358 == getSum(layer.mResidualActivation1));
			assert(280 == getSum(layer.mL2.mActivations));
			assert(-235461 == getSum(layer.mMLP.mCFCActivations));
			assert(-10345 == getSum(layer.mMLP.mGeluActivations));
			assert(-155 == getSum(layer.mResidualActivation2));

			};

		auto testBackLayer = [&]() {

			const auto& layer = forward.mAttnLayers.back();

			assert(-3859 == getSum(layer.mResidualActivation2));

			};

		auto testOutput = [&]() {

			auto testSum = getSum(forward.mFinalLayer.getActivations());
			constexpr auto finalSum = 16654;

			//std::println("{} == {} is {}", finalSum, testSum, finalSum == testSum);

			assert(finalSum == testSum);

			};


		auto testUnEmbed = [&]() {

			//inaccuracies/difference from reduce?
			auto sum = getSum(forward.mUnembedActivations);
			assert(-353845318 == sum);

			};
		auto testPrediction = [&]() {
			//385 == us
			assert(385 == predicted);
			};

		testEmbed();
		testFrontLayer();
		testBackLayer();
		testOutput();
		testUnEmbed();
		testPrediction();
		};

	run([&](auto& gpt2) {
		auto& data = gpt2.mTestData;
		data.load();
		TokensView tokens = { data.mTokens.begin(), GPT2::mTestInputSize };

		Token predicted = gpt2.mForward.feedForward(tokens);

		test(gpt2, predicted);

		});
}

void GPT2::Diagnostics::feedForwardSpeed1024() {

	//this test is used to examine feedforward speed

	run([&](auto& gpt2) {

		auto& data = gpt2.mTestData;
		data.load();
		TokensView dataView(data.mTokens.begin(), GPT2::mTestInputSize);
		auto preText = gpt2.mTranslator.decode(dataView);
		std::println("{}", preText);

		Token predicted;
		Tokens tokens(dataView.begin(), dataView.end());

		TimeAverage<milliseconds> ffAvg;
		
		for (auto i : std::views::iota( 0, 200)) {

			auto elapsed = ffAvg.accumulateTime([&]() {
				predicted = gpt2.mForward.feedForward(tokens);
				});

			auto word = gpt2.mTranslator.decode(predicted);
			std::print("{}({}:{})", word, elapsed.count(), ffAvg.average());

			std::shift_left(tokens.begin(), tokens.end(), 1);
			tokens.back() = predicted;
		}

		});
}
void GPT2::Diagnostics::crossEntropyTest64() {

	run([&](auto& gpt2) {

		auto& data = gpt2.mTestData;
		data.load();

		TokensView tokens(data.mTokens.begin(), GPT2::mTestInputSize)
			, nextTokens(data.mTokens.begin() + 1, GPT2::mTestInputSize);

		auto preText = gpt2.mTranslator.decode(tokens);
		std::println("{}", preText);

		Token predicted, expected = nextTokens.back();

		float crossEntropyLoss;

		TimeAverage<milliseconds> ffAvg;

		auto elapsed = ffAvg.accumulateTime([&]() {

			predicted = gpt2.mForward.feedForward(tokens);

			crossEntropyLoss = gpt2.mForward.crossEntropyLoss(nextTokens );

			});

		auto predictedWord = gpt2.mTranslator.decode(predicted);
		auto expectedWord = gpt2.mTranslator.decode(expected);

		std::println("{}=={}; Cross Entropy Loss: {} == 4.133143", predictedWord, expectedWord, crossEntropyLoss );
	
		});

}
void GPT2::Diagnostics::backwardTest64() {

	run([&](auto& gpt2) {

		auto& data = gpt2.mTestData;
		data.load();

		TokensView tokens(data.mTokens.begin(), GPT2::mTestInputSize)
			, nextTokens(data.mTokens.begin() + 1, GPT2::mTestInputSize);

		auto preText = gpt2.mTranslator.decode(tokens);
		std::println("{}", preText);

		Token predicted, expected = nextTokens.back();

		float crossEntropyLoss;
		TimeAverage<milliseconds> ffAvg;

		auto& forward = gpt2.mForward;

		auto elapsed = ffAvg.accumulateTime([&]() {

			predicted = forward.feedForward(tokens);

			crossEntropyLoss = forward.crossEntropyLoss(nextTokens);

			});

		auto predictedWord = gpt2.mTranslator.decode(predicted);
		auto expectedWord = gpt2.mTranslator.decode(expected);

		std::println("{}=={}; Cross Entropy Loss: {} == 4.133143", predictedWord, expectedWord, crossEntropyLoss);


		auto& backward = gpt2.mBackward;

		backward.setup(&gpt2.mForward);

		backward.backward(nextTokens);

		std::println("results:");

		sumf(backward.mUnembed, "0.008");//re source 0008
		sumf(forward.mUnembedActivationsSoftmax, "64");
		sumf(backward.mFinalLayer.mActivations, "-0.0403");
		sumf(backward.mFinalLayer.mBias, "-0.0403");
		sumf(backward.mFinalLayer.mWeight, "-0.5371");

		auto& attnBack = backward.mAttnLayers.back();

		sumf(attnBack.getOutput(), "-1.0-e08 on debug");
		sumAbsf(attnBack.mMLP.mCProjBias, "0.4879f");
		sumAbsf(attnBack.mMLP.mCProjWeight, "348");
		sumAbsf(attnBack.mMLP.mGeluActivations, "58.9");
		sumAbsf(attnBack.mMLP.mCFCActivations, "14.5");
		sumAbsf(attnBack.mL2.mActivations, "54.6");
		sumAbsf(attnBack.mMLP.mCFCWeight, "523.4");
		sumAbsf(attnBack.mMLP.mCFCBias, "3.66");
		sumAbsf(attnBack.mL2.mWeight, "5.93");
		sumAbsf(attnBack.mL2.mBias, "11.73");
		sumAbsf(attnBack.mResidualActivation1, "3.26");
		sumAbsf(attnBack.mAttnZ, "10.85");
		//sumAbsf(attnBack.mCAttnActivations.viewBlock(), "3.116");
		attnSumAbsf(attnBack.mCAttnActivations, mVOffset, "3.116");
		attnSumAbsf(attnBack.mCAttnActivations, mKOffset, "--");
		attnSumAbsf(attnBack.mCAttnActivations, mQOffset, "--");
		sumAbsf(attnBack.mCAttnActivations.viewBlock(), "6.96");
		});

}


void GPT2::Diagnostics::simpleChat() {

	run([&](auto& gpt2) {
		gpt2.chat();
		});
}

void GPT2::Diagnostics::run(TestFunction&& test) {

	try {

		auto gpt2 = std::make_unique<GPT2>(); //gpt2 is large and offsourced to heap

		gpt2->setup();

		test(*gpt2);

	}catch (const GPT2::Error& e) {
		std::println(std::cerr, "{}", e.what());
	}catch (const std::exception& e) {
		std::println(std::cerr, "{}", e.what());
	}catch (...) {
		std::println(std::cerr, "Unknown error");
	}
}