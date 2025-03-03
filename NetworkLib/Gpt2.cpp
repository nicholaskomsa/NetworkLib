#include "Gpt2.h"

#include <system_error>

using namespace NetworkLib;

Parallel GPT2::mParallelInput(GPT2::mTestInputSize)
, GPT2::mParallelHeads(GPT2::mHeadNum, GPT2::mHeadNum)
, GPT2::mParallelI(GPT2::mTestInputSize);

const float GPT2::AttnLayer::r_sqrtHeadsPerDModel = 1.0f / std::sqrtf(GPT2::mHeadsPerDModel);

GPT2::Error::Error(std::errc code, const std::string& message) : std::system_error(int(code), std::generic_category(), message) {}

void GPT2::Error::fileNotFound(const std::string& fileName) {
	throw Error(std::errc::no_such_file_or_directory, std::format("File Not Found: {}", fileName));
}

void GPT2::Translator::readEnc() {

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


std::string GPT2::Translator::decode( TokensView tokens) {

	std::string text;
	text.reserve(tokens.size() * 5); //avg word size == 5?

	for (auto token : tokens) {
		text += mWords[token];
	}

	return text;
}
std::string GPT2::Translator::decode(Token token) {
	return std::string( mWords[token] );
}

GPT2::Tokens GPT2::Translator::encode(std::string_view remaining) {

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
		auto wordExists = mWordMap.end();

		for (std::size_t size = wordSize; size >= 1; --size) {

			testWord = remaining.substr(0, size);

			wordExists = mWordMap.find(testWord);
			if (wordExists != mWordMap.end())
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

	mActivationSpace.resize(mSeqModel + (mSeqModel * 7 + mSeqModel3 + mSeqSeqHead * 2 + mSeqModel4 * 2) * mAttnLayers.size() + mSeqModel + mSeqVocab);
	auto begin = mActivationSpace.begin();

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

	mWActivations = { {begin, mSeqModel}, mDSeq, mDModel };
	std::advance(begin, mSeqModel);

	for (std::size_t i = 0; i < mAttnLayers.size(); ++i)
		mAttnLayers[i].load(readTensorByName, i, begin);

	mFinalLayer.load(readTensorByName("ln_f.bias"), readTensorByName("ln_f.weight"), begin);

	mUnembedActivations = { {begin, mSeqVocab}, mDSeq, mDVocab };
	std::advance(begin, mSeqVocab);

	assert(floatsUsed == mTensorSpace.size());
	assert( begin == mActivationSpace.end());

	std::puts("Tensors read successfully");
}
