#include "Gpt2.h"

#include <system_error>

using namespace NetworkLib;

Parallel GPT2::mParallelInput(GPT2::mTestInputSize);
Parallel GPT2::mParallelHeads(GPT2::mHeadNum, GPT2::mHeadNum);

const float GPT2::AttnLayer::r_sqrtHeadsPerDModel = 1.0f / std::sqrtf(GPT2::mHeadsPerDModel);

GPT2::Error::Error(std::errc code, const std::string& message) : std::system_error(int(code), std::generic_category(), message) {}

void GPT2::Error::fileNotFound(const std::string& fileName) {
	throw Error(std::errc::no_such_file_or_directory, std::format("File Not Found: {}", fileName));
}

void GPT2::Decoder::readEnc() {

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
		mWords.resize(mDVocab);
		mDenseWords.resize(mDenseWordsSize);

		fin.read(reinterpret_cast<char*>(offsets.data()), mDVocab * sizeof(Offset));
		fin.read(mDenseWords.data(), mDenseWordsSize);

		fin.close();
		std::puts("file read done");

		return offsets;
		};

	auto offsets = readFile();

	for (std::size_t i = 0; i < mWords.size(); ++i) {

		auto& [offset, size] = offsets[i];
		auto& word = mWords[i];

		word = { mDenseWords.data() + offset, size };

		mWordMap[word] = i;
	}
}


std::string GPT2::Decoder::decode( TokensView tokens) {

	std::string text;
	text.reserve(tokens.size() * 5); //avg word size == 5?

	for (auto token : tokens) {
		text += mWords[token];
	}

	return text;
}
std::string GPT2::Decoder::decode(Token token) {
	return std::string( mWords[token] );
}
void GPT2::Data::readData() {

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

void GPT2::readSafeTensors() {

	constexpr auto floatSize = sizeof(float);
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

	auto readTensorByName = [&](const auto& name) {

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

	auto createAttentionLayer = [&](auto& attnLayer) {

		auto layerIdx = &attnLayer - mAttnLayers.data();
		auto layer = std::format("h.{}.", layerIdx);

		auto attnName = [layer](const auto& postFix) {
			return std::format("{}attn.{}", layer, postFix);
			};

		attnLayer.mBias = readTensorByName(attnName("bias"));
		attnLayer.mCAttnBias = readTensorByName(attnName("c_attn.bias"));
		attnLayer.mCAttnWeight = readTensorByName(attnName("c_attn.weight"));
		attnLayer.mCProjBias = readTensorByName(attnName("c_proj.bias"));
		attnLayer.mCProjWeight = readTensorByName(attnName("c_proj.weight"));

		auto linearName = [&](auto idx, const auto& postFix) {
			return std::format("{}ln_{}.{}", layer, idx, postFix);
			};

		attnLayer.mL1.mBias = readTensorByName(linearName(1, "bias"));
		attnLayer.mL1.mWeight = readTensorByName(linearName(1, "weight"));
		attnLayer.mL2.mBias = readTensorByName(linearName(2, "bias"));
		attnLayer.mL2.mWeight = readTensorByName(linearName(2, "weight"));

		auto mlpName = [&](const auto& postFix) {
			return std::format("{}mlp.{}", layer, postFix);
			};

		attnLayer.mMLP.mCFCBias = readTensorByName(mlpName("c_fc.bias"));
		attnLayer.mMLP.mCFCWeight = readTensorByName(mlpName("c_fc.weight"));
		attnLayer.mMLP.mCProjBias = readTensorByName(mlpName("c_proj.bias"));
		attnLayer.mMLP.mCProjWeight = readTensorByName(mlpName("c_proj.weight"));

		};

	std::for_each(mAttnLayers.begin(), mAttnLayers.end(), createAttentionLayer);

	mFinalLayer.mBias = readTensorByName("ln_f.bias");
	mFinalLayer.mWeight = readTensorByName("ln_f.weight");

	assert(floatsUsed == mTensorSpace.size());

	std::puts("Tensors read successfully");

	auto createActivationSpace = [&]() {

		auto seqModel = mDSeq * mDModel;
		auto seqModel3 = mDSeq * mDModel3;
		auto seqModel4 = mDSeq * mDModel4;
		auto seqSeqHead = mDSeq * mDSeq * mHeadNum;
		auto seqVocab = mDSeq * mDVocab;

		mActivationSpace.resize(seqModel + (seqModel * 7 + seqModel3 + seqSeqHead * 2 + seqModel4 * 2) * mAttnLayers.size() + seqModel + seqVocab);

		auto begin = mActivationSpace.begin();

		mWActivations = { {begin, seqModel}, mDSeq, mDModel };
		std::advance(begin, seqModel);

		for (auto& layer : mAttnLayers) {

			layer.mL1.mActivations = { {begin, seqModel}, mDSeq, mDModel };
			std::advance(begin, seqModel);

			layer.mCAttnActivations = { {begin, seqModel3}, mDSeq, mDModel3 };
			std::advance(begin, seqModel3);

			layer.mAttnActivations = { {begin, seqSeqHead}, mDSeq, mDSeq, mHeadNum };
			std::advance(begin, seqSeqHead);

			layer.mAttnSoftmaxActivations = { {begin, seqSeqHead}, mDSeq, mDSeq, mHeadNum };
			std::advance(begin, seqSeqHead);

			layer.mAttnZ = { {begin, seqModel}, mDSeq, mDModel };
			std::advance(begin, seqModel);

			layer.mCProjActivations = { {begin, seqModel}, mDSeq, mDModel };
			std::advance(begin, seqModel);

			layer.mResidualActivation1 = { {begin, seqModel}, mDSeq, mDModel };
			std::advance(begin, seqModel);

			layer.mL2.mActivations = { {begin, seqModel}, mDSeq, mDModel };
			std::advance(begin, seqModel);

			layer.mMLP.mCFCActivations = { {begin, seqModel4}, mDSeq, mDModel4 };
			std::advance(begin, seqModel4);

			layer.mMLP.mGeluActivations = { {begin, seqModel4}, mDSeq, mDModel4 };
			std::advance(begin, seqModel4);

			layer.mMLP.mCProjActivations = { {begin, seqModel}, mDSeq, mDModel };
			std::advance(begin, seqModel);

			layer.mResidualActivation2 = { {begin, seqModel}, mDSeq, mDModel };
			std::advance(begin, seqModel);

		}

		mFinalLayer.mActivations = { {begin, seqModel}, mDSeq, mDModel };
		std::advance(begin, seqModel);

		mUnembedActivations = { {begin, seqVocab}, mDSeq, mDVocab };
		std::advance(begin, seqVocab);
		};

	createActivationSpace();
}
