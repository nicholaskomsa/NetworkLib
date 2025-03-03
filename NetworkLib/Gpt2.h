#pragma once

#include <array>
#include <iostream>
#include <print>
#include <map>
#include <numbers>

#include "Algorithms.h"
#include "Parallel.h";
#include "Tensor.h"

namespace NetworkLib {

	class GPT2 {
	public:
		
		//configuration chat gpt2 model here
		static constexpr auto mFilePath = "F:/software dev/programming2025/downloads/";
		static constexpr std::size_t mDVocab = 50257
			, mDModel = 768, mDModel3 = mDModel * 3, mDModel4 = mDModel * 4
			, mDSeq = 1024
			, mHeadNum = 12, mAttnLayersNum = 12
			, mHeadsPerDModel = mDModel / mHeadNum
			, mQOffset = 0, mKOffset = mDModel, mVOffset = mDModel * 2
			, mTestInputSize = mDSeq;	//vs dSeq for full size or 64 for test size


		static constexpr auto mSeqModel = mDSeq * mDModel
			, mSeqModel3 = mDSeq * mDModel3
			, mSeqModel4 = mDSeq * mDModel4
			, mSeqSeqHead = mDSeq * mDSeq * mHeadNum
			, mSeqVocab = mDSeq * mDVocab;

		struct Error : public std::system_error {

			Error(std::errc code, const std::string& message);

			static void fileNotFound(const std::string& fileName);
		};

		static Parallel mParallelInput, mParallelHeads, mParallelI;

		using Floats = std::vector<float>;

		using Token = std::uint16_t;
		using Tokens = std::vector<Token>;
		using TokensView = std::span<Token>;

		struct Translator {

			using Word = std::string_view;
			using Words = std::vector<Word>;
			using WordMap = std::map<Word, Token>;

			Words mWords;
			WordMap mWordMap;//map words to their index

			static constexpr auto mDenseWordsSize = 321428;
			std::string mDenseWords;

			void readEnc();
			std::string decode(TokensView tokens);
			std::string decode(Token token);

			Tokens encode(std::string_view remaining);

		} mTranslator;

		struct Data {

			Tokens mTokens;

			void readData();

		} mData;

	private:

		static void forward(std::size_t i, const Tensor& inputTensor, const Tensor& outputTensor, const Tensor& weightTensor, const Tensor& biasTensor, Parallel& parallel);
		static void forward(const Tensor& inputTensor, const Tensor& outputTensor, const Tensor& weightTensor, const Tensor& biasTensor, Parallel& parallel);
		
		class MLP {

			Tensor mCFCBias, mCFCWeight, mCProjBias, mCProjWeight;
			Tensor mCFCActivations, mCProjActivations, mGeluActivations;

			static constexpr float r_sqrt2 = 1.0f / std::numbers::sqrt2;

		public:
			void forward(const Tensor& input);
			void forward(std::size_t i, const Tensor& input);

			const Tensor& getCProjActivations() const;

			void load(auto&& cfcBias, auto&& cfcWeight, auto&& cProjBias, auto&& cProjWeight, Floats::iterator& activationSpace);
		};
		class LinearLayer {

			Tensor mBias, mWeight;
			Tensor mActivations;
		public:

			const Tensor& getActivations() const;

			void load(Tensor&& bias, Tensor&& weight, Floats::iterator& activationSpace);

			void normalise(std::size_t m, const Tensor& input);
			void normalise(const Tensor& input);
		};
		class AttnLayer {

			LinearLayer mL1, mL2;

			Tensor mBias, mCAttnBias, mCAttnWeight, mCProjBias, mCProjWeight;
			Tensor mCAttnActivations, mAttnActivations, mAttnSoftmaxActivations, mAttnZ, mCProjActivations;
			Tensor mResidualActivation1, mResidualActivation2;

			MLP mMLP;

			static const float r_sqrtHeadsPerDModel;

			void calculateQKAtten(std::size_t headOffset, std::size_t i, Tensor::TensorView attnOut);
			void softmax(std::size_t i, Tensor::TensorView input, Tensor::TensorView output);
			void calculateVAtten(std::size_t headOffset, std::size_t i, Tensor::TensorView attnOutSoftmax);
			
			void multiHeadedAttn(std::size_t m);

			void attention(std::size_t m);
			void attention();

			void residual(std::size_t i, const Tensor& inputTensor, const Tensor& projectionTensor, const Tensor& residualTensor);
			void residual(const Tensor& inputTensor, const Tensor& projectionTensor, const Tensor& residualTensor);
		
		public:

			Tensor& forward(Tensor& inputTensor);
			Tensor& forward(std::size_t i, const Tensor& inputTensor);
			
			using ReadTensorFunc = std::function<Tensor(std::string_view)>;
			void load(ReadTensorFunc&& readTensorByName, std::size_t layerIdx, Floats::iterator& activationSpace);
		};

		Floats mTensorSpace, mActivationSpace;

		Tensor mWpeWeight, mWteWeight, mWActivations, mUnembedActivations;
		
		std::array<AttnLayer, mAttnLayersNum> mAttnLayers;
		LinearLayer mFinalLayer;

		void readSafeTensors();
		void embedInput(std::size_t i, Token token);
		void embedInputs(TokensView tokens);
		void unEmbedOutput(std::size_t i);
		void unEmbedOutputs();

		Token feedForward(TokensView tokens);
		Token feedMore(TokensView tokens);

	public:

		GPT2() = default;

		void setup();

		Token getPrediction(std::size_t m);

		void chat() {

			bool chatting = true;
			Tokens scrollingTokens;
			const Token endl = mTranslator.mWordMap["\n"];
			std::string line = "What color is the Sky?";
			
			do {
			
				scrollingTokens.clear();

				//std::getline(std::cin, line);
				if (line == "exit") break;
				std::cout << std::endl;

				auto userTokens = mTranslator.encode(line);
				userTokens.push_back(endl);

				scrollingTokens.insert(scrollingTokens.end(), userTokens.begin(), userTokens.end());
				slide(scrollingTokens);
				scrollingTokens.push_back(endl);

				std::cout << mTranslator.decode(scrollingTokens);
				std::cout << std::endl;

			} while (chatting);
		}
		void slide(Tokens& tokens, std::size_t distance = 50) {

			//first ensure that tokens is at most mTestInputSize
			if (tokens.size() > mTestInputSize) {
				//get tail of tokens
				tokens.erase(tokens.begin(), tokens.end() - mTestInputSize);
			}

			bool endOfSentence = false;

			auto putWord = [&](Token token) {
				auto word = mTranslator.decode(token);
				//	std::print("{}", decode);
				auto end = word.back();
				if (end == '.' || end == '?' || end == '!') endOfSentence = true;
				};

			auto addToken = [&](Token token) {

				bool scrolled = false;

				constexpr auto scrollDistance = mTestInputSize * 0.9f;

				if (tokens.size() == mTestInputSize) {

					std::shift_left(tokens.begin(), tokens.end(), scrollDistance);
					tokens.resize(mTestInputSize - scrollDistance);

					tokens.back() = token;

					scrolled = true;

				}
				else
					tokens.push_back(token);

				putWord(token);

				return scrolled;
				};

			bool scrolled = true;
			Token newToken = 0;
			TimeAverage<milliseconds> ffAvg, fmAvg;

			for (std::size_t s = 0; s < distance && !endOfSentence; ++s) {
			
				if (scrolled)
					ffAvg.accumulateTime([&]() {
						newToken = feedForward(tokens);
						});
				else
					fmAvg.accumulateTime([&]() {
						newToken = feedMore(tokens);
						});
			
				auto printAvgTime = [&]() {

					auto& updated = scrolled ? ffAvg : fmAvg;
					std::print("{},", updated.average() );
					};
				printAvgTime();

				scrolled = addToken(newToken);
			}
		}
	};
}