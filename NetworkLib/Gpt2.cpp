#include "Gpt2.h"

#include <system_error>
#include <sstream>
#include <fstream>
#include <cassert>
#include <ranges>

#include <boost/json.hpp>

using namespace NetworkLib;

Parallel GPT2::mParallelInput(GPT2::mTestInputSize)
	, GPT2::mParallelHeads(GPT2::mHeadNum, GPT2::mHeadNum)
	, GPT2::mParallelI(GPT2::mTestInputSize);

const float GPT2::AttnLayer::r_sqrtHeadsPerDModel = 1.0f / std::sqrtf(GPT2::mHeadsPerDModel);

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

	for (Token token = 0; token < offsets.size(); ++token) {

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

	Tokens tokens;

	auto getToken = [&]() {

		const std::string delims = " \n\t\r";

		auto getWordSize = [&](auto remaining) {

			std::size_t endOfWord = remaining.find_first_of(delims);

			if (endOfWord == std::string::npos)
				endOfWord = remaining.size();

			return endOfWord;
			};

		auto wordSize = 0;
		if (remaining.front() == ' ')
			wordSize = 1 + getWordSize(remaining.substr(1));
		else
			wordSize = getWordSize(remaining);

		std::string_view testWord;
		auto wordExists = mWordMap.left.end();

		for (std::size_t size = wordSize; size >= 1; --size) {

			testWord = remaining.substr(0, size);

			wordExists = mWordMap.left.find(testWord);
			if (wordExists != mWordMap.left.end())
				break;
		}

		auto& [word, token] = *wordExists;

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

void GPT2::Data::load() {

	//this file loads the concerning first citizen test data set, the first 64 tokens is used by checksum
	//this file is found at raffK project, "data"

	auto readFile = [&]() {

		//enc file https://github.com/rkaehn/gpt-2/blob/main/assets/data
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

void GPT2::load() {
	//the gpt2 model is found on huggingface website: https://huggingface.co/openai-community/gpt2?show_file_info=model.safetensors
	//we need to load gpt2 from file and create the activation space
	//the order of the activation space affects speed same as the gpt model space itself
	//so the layout follows forward process specifically
	//the gpt2 model is loaded from file - in a memory layout openai saved to file
	//but it could possibly be made faster still by moving from spanT to span?

	constexpr float floatSize = sizeof(float);
	using Header = std::string;

	auto readFile = [&]() -> Header {

		//gpt2 tensors https://huggingface.co/openai-community/gpt2?show_file_info=model.safetensors
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

	mActivationSpace.resize(mSeqModel + (mSeqModel * 7 + mSeqModel3 + mSeqSeqHead * 2 + mSeqModel4 * 2) * mAttnLayers.size() + mSeqModel + mSeqVocab);
	auto activationSpace = mActivationSpace.begin();

	auto readTensorByName = [&](std::string_view name) {

		auto& obj = j.at(name);
		auto& offsets = obj.at("data_offsets").as_array();
		auto a = offsets.front().as_int64() / floatSize, b = offsets.back().as_int64() / floatSize;

		auto start = std::next(mTensorSpace.begin(), a);
		auto end = std::next(mTensorSpace.begin(), b);
		auto size = std::distance(start, end);

		Tensor::TensorView tensorView(start, size);

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

	mWActivations = { {activationSpace, mSeqModel}, mDSeq, mDModel };
	std::advance(activationSpace, mSeqModel);

	for (std::size_t i = 0; i < mAttnLayers.size(); ++i)
		mAttnLayers[i].load(readTensorByName, i, activationSpace);

	mFinalLayer.load(readTensorByName("ln_f.bias"), readTensorByName("ln_f.weight"), activationSpace);

	mUnembedActivations = { {activationSpace, mSeqVocab}, mDSeq, mDVocab };
	std::advance(activationSpace, mSeqVocab);

	assert(floatsUsed == mTensorSpace.size());
	assert(activationSpace == mActivationSpace.end());

	std::puts("Tensors read successfully");
}
void GPT2::forward(std::size_t i, const Tensor& inputTensor, const Tensor& outputTensor, const Tensor& weightTensor, const Tensor& biasTensor, Parallel& parallel) {
	//this is a matrix multiply and add - "forward" o = w * i + b
	Tensor::TensorView input, output, b = biasTensor.span();

	input = inputTensor.spanT(i);
	output = outputTensor.spanT(i);

	std::copy(b.begin(), b.end(), output.begin());

	parallel([&](Parallel::SectionsView sections) {

		for (auto& section : sections) {
			if (section.mAny.has_value() == false)
				section.mAny = Floats(output.size(), 0.0f);

			auto& floats = std::any_cast<Floats&>(section.mAny);
			floats.clear();
			floats.resize(output.size(), 0.0f);
		}

		}, [&](Parallel::Section& section) {

			auto& outputs = std::any_cast<Floats&>(section.mAny);

			auto& [first, second] = section.mOffsets;

			for (std::size_t m = first; m < second; ++m) {

				const auto& in = input[m];

				for (const auto& [o, w] : std::views::zip(outputs, weightTensor.spanT(m)))
					o += w * in;
			}

			}, [&](Parallel::SectionsView sections) {

				for (auto& section : sections) {

					auto& sOutputs = std::any_cast<Floats&>(section.mAny);

					for (std::size_t m = 0; m < output.size(); ++m)
						output[m] += sOutputs[m];
				}
				});
}
void GPT2::forward(const Tensor& inputTensor, const Tensor& outputTensor, const Tensor& weightTensor, const Tensor& biasTensor, Parallel& parallel) {

	parallel([&](Parallel::SectionsView sections) {

		for (auto& section : sections) {

			if (section.mAny.has_value() == false)
				section.mAny = Parallel();

			auto& parallel = std::any_cast<Parallel&>(section.mAny);
			parallel.section(inputTensor.mY, Parallel::mLargeHardwareThreads);
		}

		}, [&](Parallel::Section& section) {

			auto& [first, second] = section.mOffsets;
			auto& parallel = std::any_cast<Parallel&>(section.mAny);

			for (std::size_t i = first; i < second; ++i)
				forward(i, inputTensor, outputTensor, weightTensor, biasTensor, parallel);

			});
}

void GPT2::MLP::forward(const Tensor& input) {

	GPT2::forward(input, mCFCActivations, mCFCWeight, mCFCBias, mParallelInput);

	std::transform(std::execution::par_unseq, mCFCActivations.mTensor.begin(), mCFCActivations.spanT(mParallelInput.mSize - 1).end(), mGeluActivations.mTensor.begin(),
		[&](auto x) {
			return x * 0.5f * (1.0f + std::erff(x * r_sqrt2));
		});

	GPT2::forward(mGeluActivations, mCProjActivations, mCProjWeight, mCProjBias, mParallelInput);
}
void GPT2::MLP::forward(std::size_t i, const Tensor& input) {

	GPT2::forward(i, input, mCFCActivations, mCFCWeight, mCFCBias, mParallelI);

	Tensor::TensorView mlpActivations = mCFCActivations.spanT(i), geluActivations = mGeluActivations.spanT(i);

	std::transform(std::execution::par_unseq, mlpActivations.begin(), mlpActivations.end(), geluActivations.begin(),
		[&](auto x) {
			return x * 0.5f * (1.0f + std::erff(x * r_sqrt2));
		});

	mParallelI.section(mDModel4);
	GPT2::forward(i, mGeluActivations, mCProjActivations, mCProjWeight, mCProjBias, mParallelI);
}
const Tensor& GPT2::MLP::getCProjActivations() const {
	return mCProjActivations;
}
void GPT2::MLP::load(auto&& cfcBias, auto&& cfcWeight, auto&& cProjBias, auto&& cProjWeight, Floats::iterator& activationSpace) {

	mCFCBias = std::move(cfcBias);
	mCFCWeight = std::move(cfcWeight);
	mCProjBias = std::move(cProjBias);
	mCProjWeight = std::move(cProjWeight);

	mCFCActivations = { {activationSpace, mSeqModel4}, mDSeq, mDModel4 };
	std::advance(activationSpace, mSeqModel4);

	mGeluActivations = { {activationSpace, mSeqModel4}, mDSeq, mDModel4 };
	std::advance(activationSpace, mSeqModel4);

	mCProjActivations = { {activationSpace, mSeqModel}, mDSeq, mDModel };
	std::advance(activationSpace, mSeqModel);
}

const Tensor& GPT2::LinearLayer::getActivations() const {
	return mActivations;
}
void GPT2::LinearLayer::load(Tensor&& bias, Tensor&& weight, Floats::iterator& activationSpace) {
	mBias = std::move(bias);
	mWeight = std::move(weight);
	mActivations = { {activationSpace, mSeqModel}, mDSeq, mDModel };
	std::advance(activationSpace, mSeqModel);
}
void GPT2::LinearLayer::normalise(std::size_t m, const Tensor& input) {

	Tensor::TensorView in = input.spanT(m), out = mActivations.spanT(m);

	const auto mean = std::reduce(in.begin(), in.end()) / in.size();

	auto meanDiffSq = std::reduce(in.begin(), in.end(), 0.0f,
		[&](auto sum, auto x) {
			auto diff = x - mean;
			return sum + diff * diff;
		}) / in.size();

	auto r_stdDev = 1.0f / std::sqrt(meanDiffSq);

	Tensor::TensorView bias = mBias.span(), weight = mWeight.span();

	float norm = 0;
	for (const auto& [i, w, b, o] : std::views::zip(in, weight, bias, out)) {
		norm = (i - mean) * r_stdDev;
		o = norm * w + b;
	}
}
void GPT2::LinearLayer::normalise(const Tensor& input) {

	mParallelInput([&](auto& sections) {

		auto& [first, second] = sections.mOffsets;

		for (std::size_t m = first; m < second; ++m)
			normalise(m, input);

		});
}

void GPT2::AttnLayer::calculateQKAtten(std::size_t headOffset, std::size_t i, Tensor::TensorView attnOut) {

	const auto qOffset = mQOffset + headOffset;
	Tensor::TensorView qh = { mCAttnActivations.spanT(i).data() + qOffset, mHeadsPerDModel };

	const auto kOffset = mKOffset + headOffset;

	for (std::size_t m = 0; m <= i; ++m) {

		Tensor::TensorView kh = { mCAttnActivations.spanT(m).data() + kOffset, mHeadsPerDModel };
		float dot = 0.0f;

		for (const auto& [q, k] : std::views::zip(qh, kh))
			dot += q * k;

		attnOut[m] = dot * r_sqrtHeadsPerDModel;
	};
}
void GPT2::AttnLayer::softmax(std::size_t i, Tensor::TensorView input, Tensor::TensorView output) {

	const auto ibegin = input.begin(), iend = ibegin + 1 + i, obegin = output.begin(), oend = obegin + 1 + i;

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
void GPT2::AttnLayer::calculateVAtten(std::size_t headOffset, std::size_t i, Tensor::TensorView attnOutSoftmax) {

	Tensor::TensorView zh = { mAttnZ.spanT(i).data() + headOffset, mHeadsPerDModel };
	const auto vOffset = mVOffset + headOffset;

	for (std::size_t m = 0; m <= i; ++m) {

		Tensor::TensorView vh = { mCAttnActivations.spanT(m).data() + vOffset, mHeadsPerDModel };
		float factor = attnOutSoftmax[m];

		for (const auto& [z, v] : std::views::zip(zh, vh))
			z += v * factor;
	}
}
void GPT2::AttnLayer::multiHeadedAttn(std::size_t m) {

	mParallelHeads([&](auto& section) {

		auto& [first, second] = section.mOffsets;

		for (std::size_t h = first; h < second; ++h) {

			const auto headOffset = h * mHeadsPerDModel;

			for (std::size_t i = 0; i <= m; ++i) {

				Tensor::TensorView attnOut = mAttnActivations.spanT(h, i)
					, attnOutSoftmax = mAttnSoftmaxActivations.spanT(h, i);

				calculateQKAtten(headOffset, i, attnOut);
				softmax(i, attnOut, attnOutSoftmax);
				calculateVAtten(headOffset, i, attnOutSoftmax);
			}
		}

		});
}
void GPT2::AttnLayer::attention(std::size_t m) {

	Tensor::TensorView z = mAttnZ.spanT(m);
	std::fill(z.begin(), z.end(), 0.0f);

	multiHeadedAttn(m);
}
void GPT2::AttnLayer::attention() {

	auto m = mParallelInput.mSize - 1;

	//activations z cleared here
	std::fill(mAttnZ.mTensor.begin(), mAttnZ.spanT(m).end(), 0.0f);

	multiHeadedAttn(m);
}
void GPT2::AttnLayer::residual(std::size_t i, const Tensor& inputTensor, const Tensor& projectionTensor, const Tensor& residualTensor) {

	for (const auto& [out, p, in] : std::views::zip(residualTensor.spanT(i), projectionTensor.spanT(i), inputTensor.spanT(i)))
		out = p + in;
}
void GPT2::AttnLayer::residual(const Tensor& inputTensor, const Tensor& projectionTensor, const Tensor& residualTensor) {

	mParallelInput([&](auto& sections) {

		auto& [first, second] = sections.mOffsets;

		for (std::size_t i = first; i < second; ++i)
			residual(i, inputTensor, projectionTensor, residualTensor);

		});
}
Tensor& GPT2::AttnLayer::forward(Tensor& inputTensor) {

	mL1.normalise(inputTensor);
	GPT2::forward(mL1.getActivations(), mCAttnActivations, mCAttnWeight, mCAttnBias, mParallelInput);

	attention();

	GPT2::forward(mAttnZ, mCProjActivations, mCProjWeight, mCProjBias, mParallelInput);

	residual(inputTensor, mCProjActivations, mResidualActivation1);

	mL2.normalise(mResidualActivation1);

	mMLP.forward(mL2.getActivations());

	residual(mResidualActivation1, mMLP.getCProjActivations(), mResidualActivation2);

	return mResidualActivation2;
}
Tensor& GPT2::AttnLayer::forward(std::size_t i, const Tensor& inputTensor) {

	mParallelI.section(mDModel);

	mL1.normalise(i, inputTensor);
	GPT2::forward(i, mL1.getActivations(), mCAttnActivations, mCAttnWeight, mCAttnBias, mParallelI);

	attention(i);

	GPT2::forward(i, mAttnZ, mCProjActivations, mCProjWeight, mCProjBias, mParallelI);

	residual(i, inputTensor, mCProjActivations, mResidualActivation1);

	mL2.normalise(i, mResidualActivation1);

	mMLP.forward(i, mL2.getActivations());

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

	mCAttnActivations = { {activationSpace, mSeqModel3}, mDSeq, mDModel3 };
	std::advance(activationSpace, mSeqModel3);

	mAttnActivations = { {activationSpace, mSeqSeqHead}, mDSeq, mDSeq, mHeadNum };
	std::advance(activationSpace, mSeqSeqHead);

	mAttnSoftmaxActivations = { {activationSpace, mSeqSeqHead}, mDSeq, mDSeq, mHeadNum };
	std::advance(activationSpace, mSeqSeqHead);

	mAttnZ = { {activationSpace, mSeqModel}, mDSeq, mDModel };
	std::advance(activationSpace, mSeqModel);

	auto linearTensor = [&](auto idx, const auto& name) {
		return readTensorByName(std::format("{}ln_{}.{}", layer, idx, name));
		};

	mL1.load(linearTensor(1, "bias"), linearTensor(1, "weight"), activationSpace);

	mCProjActivations = { {activationSpace, mSeqModel}, mDSeq, mDModel };
	std::advance(activationSpace, mSeqModel);

	mResidualActivation1 = { {activationSpace, mSeqModel}, mDSeq, mDModel };
	std::advance(activationSpace, mSeqModel);

	mL2.load(linearTensor(2, "bias"), linearTensor(2, "weight"), activationSpace);

	auto mlpTensor = [&](const auto& name) {
		return readTensorByName(std::format("{}mlp.{}", layer, name));
		};

	mMLP.load(mlpTensor("c_fc.bias"), mlpTensor("c_fc.weight"), mlpTensor("c_proj.bias"), mlpTensor("c_proj.weight"), activationSpace);

	mResidualActivation2 = { {activationSpace, mSeqModel}, mDSeq, mDModel };
	std::advance(activationSpace, mSeqModel);
}

void GPT2::embedInput(std::size_t i, Token token) {
	
	Tensor::TensorView wte, wpe, wActivations;
	//weight token embed, weight position embed
	//generates an activation combining token and position

	wte = mWteWeight.spanT(token);
	wpe = mWpeWeight.spanT(i);
	wActivations = mWActivations.spanT(i);

	for (const auto& [a, t, p] : std::views::zip(wActivations, wte, wpe))
		a = t + p;
}
void GPT2::embedInputs(TokensView tokens) {

	//for each token, generate a position+token embedding
	//as a setup step before forward

	mParallelInput([&](auto& sections) {

		auto& [first, second] = sections.mOffsets;

		for (std::size_t i = first; i < second; ++i)
			embedInput(i, tokens[i]);

		});
}
void GPT2::unEmbedOutput(std::size_t i) {

	//after forward, generate the probability of a specific token

	Tensor::TensorView input, wte, output;
	//weight token embed seen in embedInput
	//input is the output of earlier forward process

	input = mFinalLayer.getActivations().spanT(i);
	output = mUnembedActivations.spanT(i);

	//this is a full-connect between input, output (no bias) and wte
	for (std::size_t m = 0; m < output.size(); ++m) {

		wte = mWteWeight.spanT(m);

		float sum = 0.0f;

		for (const auto& [in, w] : std::views::zip(input, wte))
			sum += in * w;

		output[m] = sum;
	}
}
void GPT2::unEmbedOutputs() {

	//after forward, generate each token probability

	mParallelInput([&](auto& sections) {

		auto& [first, second] = sections.mOffsets;

		for (std::size_t i = first; i < second; ++i)
			unEmbedOutput(i);

		});
}
void GPT2::setup() {

	//we need to load our gpt data from file and also load the token translator file

	load();
	mTranslator.load();
}
GPT2::Token GPT2::getPrediction(std::size_t m) {

	//unembed activations is the entire sequence of all tokens, each a prediction of its probability 
	//the highest probability is the predicted token here, but other tokens may also have some lower possibility

	auto unembedActivations = mUnembedActivations.spanT(m);
	auto selected = std::max_element(unembedActivations.begin(), unembedActivations.end());
	Token predicted = std::distance(unembedActivations.begin(), selected);

	return predicted;
}
GPT2::Token GPT2::feedForward(TokensView tokens) {
	
	//feedForward will feed all tokens fresh into the network, up to dseq number of tokens
	//tokens max size == mTestInputSize
	//if larger than maxsize, should scroll to tail-mTestInputSize
	
	//the parallel process operates over all tokens simultaneously

	mParallelInput.section(tokens.size(), Parallel::mLargeHardwareThreads);

	embedInputs(tokens); 

	Tensor* input = &mWActivations;

	for (auto& layer : mAttnLayers)
		input = &layer.forward(*input);

	mFinalLayer.normalise(*input);

	unEmbedOutputs();

	Token predicted = getPrediction(tokens.size() - 1);

	auto checkSum64 = [&]() {

		//when concerning first citizen test data, this checksum tests the test size which is 64 tokens
		assert(64 == mTestInputSize);

		auto getSum = [&](const auto& tensor) {
			return std::int64_t(std::reduce(tensor.mTensor.begin(), tensor.mTensor.end(), double(0.0)));
			};

		auto testEmbed = [&] {
			assert(-30 == getSum(mWActivations));
			};

		auto testFrontLayer = [&]() {

			const auto& layer = mAttnLayers.front();
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

			const auto& layer = mAttnLayers.back();

			assert(-3859 == getSum(layer.mResidualActivation2));

			};

		auto testOutput = [&]() {

			auto testSum = getSum(mFinalLayer.getActivations());
			constexpr auto finalSum = 16654;

			//std::println("{} == {} is {}", finalSum, testSum, finalSum == testSum);

			assert(finalSum == testSum);

			};

		auto testUnEmbed = [&]() {

			//inaccuracies/difference from reduce?
		//	std::println("-353845315 == {}", getSum(mUnembedActivations));
			assert(-353845315 == getSum(mUnembedActivations));

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
	//		checkSum64();

	return predicted;
}
GPT2::Token GPT2::feedMore(TokensView tokens) {

	//feedMore acts like all previous tokens are valid, and the back token, needs processed only
	//identical to feedForward except for parallel processing which is instead oriented toward a single sample

	mParallelInput.section(tokens.size(), Parallel::mLargeHardwareThreads);

	int i = tokens.size() - 1;
	embedInput(i, tokens.back());

	Tensor* input = &mWActivations;

	for (auto& layer : mAttnLayers)
		input = &layer.forward(i, *input);

	mFinalLayer.normalise(*input);

	unEmbedOutput(i);

	Token predicted = getPrediction(i);

	return predicted;
}