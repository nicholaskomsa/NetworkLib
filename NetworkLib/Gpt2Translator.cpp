#include "Gpt2.h"

#include <fstream>
#include <set>

using namespace NetworkLib;

using Translator = GPT2::Translator;

void Translator::load() {

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
std::string Translator::decode(TokensView tokens) const {

	//concat a series of tokens into a string

	std::stringstream sstr;

	for (auto token : tokens)
		sstr << getWord(token);

	return sstr.str();
}
std::string Translator::decode(Token token) const {
	return std::string(getWord(token));
}
Translator::Word Translator::getWord(Token token) const {

	//this function will take a token and convert it into a gpt word
	//this would only fail if token is larger than vocab size

	auto found = mWordMap.right.find(token);
	if (found == mWordMap.right.end())
		Error::tokenNotFound(token);

	return found->get_left();
}
GPT2::Tokens Translator::encode(std::string_view remaining) const {

	//take a string potentially containing many tokens, and generate all tokens
	//many gpt "words" begin with a white space " ", 
	//and this is the pause in vocabulary rather than delimiters of " " token between words
	//therefore, " Hello" is one token and " World" is another, and the sentence " Hello World" converts to two tokens only.
	//"Hello World" also converts to two tokens, many words have a " " variant and others do not
	//auto tokens = gpt2.mTranslator.encode(" Hello World");
	//std::println("Tokens: {}", tokens.size()); == 2

	//determine the size categories of the words in the vocabulary
	static std::set<std::size_t > wordSizes;
	if (wordSizes.empty())
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

		remaining = remaining | std::views::drop(word.size());

		return remaining.size();
		};

	while (getToken());

	return tokens;
}
GPT2::Token Translator::getToken(std::string_view word) const {

	//convert a word to a token
	//this will only fail if for some reason word is not a true GPT "word" found in vocabulary

	auto found = mWordMap.left.find(word);
	if (found == mWordMap.left.end())
		Error::wordNotFound(word);

	return found->get_right();
}