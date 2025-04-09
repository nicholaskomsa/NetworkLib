#pragma once

#include <array>
#include <iostream>
#include <print>
#include <numbers>

#include "Algorithms.h"
#include "Tensor.h"
#include "Parallel.h";

#include <boost/bimap.hpp>

namespace NetworkLib {

	class GPT2 {
	public:
		
		//configuration chat gpt2 model here
		static constexpr auto mFilePath = "F:/software dev/programming2025/downloads/";
		static constexpr std::size_t mDVocab = 50257
			, mDModel = 768
			, mDSeq = 1024
			, mHeadNum = 12, mAttnLayersNum = 12
			, mHeadsPerDModel = mDModel / mHeadNum
			, mQOffset = 0, mKOffset = mDModel, mVOffset = mDModel * 2
			, mTestInputSize = 64;// mDSeq;	//vs dSeq for full size or 64 for test size



		static constexpr auto  mDModel3 = mDModel * 3, mDModel4 = mDModel * 4
			, mSeqModel = mDSeq * mDModel
			, mSeqModel3 = mDSeq * mDModel3
			, mSeqModel4 = mDSeq * mDModel4
			, mSeqSeqHead = mDSeq * mDSeq * mHeadNum
			, mSeqVocab = mDSeq * mDVocab;

		using Floats = std::vector<float>;

		using Token = std::uint16_t;
		using Tokens = std::vector<Token>;
		using TokensView = std::span<Token>;

		struct Error : public std::system_error {

			Error(std::errc code, const std::string& message);

			static void fileNotFound(std::string_view fileName);
			static void wordNotFound(std::string_view word);
			static void tokenNotFound(Token token);
		};

		class Translator {
		public:
			using Word = std::string_view;
			using WordMap = boost::bimap<Word, Token>;

			void load();
			std::string decode(TokensView tokens) const;
			std::string decode(Token token) const;

			Tokens encode(std::string_view remaining) const;
			Token getToken(std::string_view word) const;
			Word getWord(Token token) const;
		private:

			WordMap mWordMap;
			std::string mDenseWords;

		} mTranslator;

		struct TestData {

			Tokens mTokens;

			void load();

		} mTestData;

		GPT2() = default;

		void setup();

		class Diagnostics {

			using TestFunction = std::function<void(GPT2& gpt2)>;
			void run(TestFunction&& test);

		public:

			static double sumf(Tensor::TensorView tensorView, std::string_view expected) {

				double sum = std::reduce(tensorView.begin(), tensorView.end(), double(0.0));
				std::print("{}=={}\n", expected, sum);
				return sum;
			}
			static double sumf(const Tensor& tensor, std::string_view expected) {
				return sumf(tensor.mTensor, expected);
			}

			void firstCitizenTest64();
			void feedForwardSpeed1024();
			void simpleChat();
			void crossEntropyTest64();
			void backwardTest64();
		};
		friend class Diagnostics;

	private:

		static void forward(std::size_t i, const Tensor& inputTensor, const Tensor& outputTensor, const Tensor& weightTensor, const Tensor& biasTensor, Parallel& parallel);
		static void forward(const Tensor& inputTensor, const Tensor& outputTensor, const Tensor& weightTensor, const Tensor& biasTensor, Parallel& parallel);
		static void softmax(std::size_t i, Tensor::TensorView input, Tensor::TensorView output);

		class MLP {

			friend class Diagnostics;

			Tensor mCFCBias, mCFCWeight, mCProjBias, mCProjWeight;
			Tensor mCFCActivations, mCProjActivations, mGeluActivations;

			static constexpr float r_sqrt2 = 1.0f / std::numbers::sqrt2;
		public:

			void load(auto&& cfcBias, auto&& cfcWeight, auto&& cProjBias, auto&& cProjWeight, Floats::iterator& activationSpace);
			const Tensor& getCProjActivations() const;

			void forward(const Tensor& input, Parallel& parallel);
			void forward(std::size_t i, const Tensor& input, Parallel& parallel);
		};
		class LinearLayer {

			friend class Diagnostics;

			Tensor mBias, mWeight;
			Tensor mActivations;
		public:

			void load(Tensor&& bias, Tensor&& weight, Floats::iterator& activationSpace);
			const Tensor& getActivations() const;

			void normalise(std::size_t i, const Tensor& input);
			void normalise(const Tensor& input, Parallel& parallel);
		};
		class AttnLayer {

			friend class Diagnostics;

			static Parallel mParallelHeads;

			LinearLayer mL1, mL2;

			Tensor mBias, mCAttnBias, mCAttnWeight, mCProjBias, mCProjWeight;
			Tensor mCAttnActivations, mAttnActivations, mAttnSoftmaxActivations, mAttnZ, mCProjActivations;
			Tensor mResidualActivation1, mResidualActivation2;

			MLP mMLP;

			static const float r_sqrtHeadsPerDModel;

			void calculateQKAtten(std::size_t headOffset, std::size_t i, Tensor::TensorView attnOut);
			void calculateVAtten(std::size_t headOffset, std::size_t i, Tensor::TensorView attnOutSoftmax);
			
			void multiHeadedAttn(std::size_t m);

			void attention(std::size_t m);
			void attention(Parallel& parallel);

			void residual(std::size_t i, const Tensor& inputTensor, const Tensor& projectionTensor, const Tensor& residualTensor);
			void residual(const Tensor& inputTensor, const Tensor& projectionTensor, const Tensor& residualTensor, Parallel& parallel);
		
		public:

			using ReadTensorFunctor = std::function<Tensor(std::string_view)>;
			void load(ReadTensorFunctor&& readTensorByName, std::size_t layerIdx, Floats::iterator& activationSpace);

			Tensor& forward(Tensor& inputTensor, Parallel& parallel);
			Tensor& forward(std::size_t i, const Tensor& inputTensor, Parallel& parallel);
		};

		class Backward;

		class Forward {

			friend class Diagnostics;
			friend class Backward;

			Parallel mParallelInput, mParallelI;

			Floats mTensorSpace, mActivationSpace;

			Tensor mWpeWeight, mWteWeight, mWActivations, mUnembedActivations, mUnembedActivationsSoftmax;

			std::array<AttnLayer, mAttnLayersNum> mAttnLayers;
			LinearLayer mFinalLayer;

			void load();
			void embedInput(std::size_t i, Token token);
			void embedInputs(TokensView tokens);
			void unEmbedOutput(std::size_t i);
			void unEmbedOutputs();

		public:

			void setup();

			Token feedForward(TokensView tokens);
			Token feedMore(TokensView tokens);

			Token getPrediction(std::size_t i) const;
			float crossEntropyLoss(TokensView tokens, Token expected);

		} mForward;

		struct Backward {

			friend class Diagnostics;

			Floats mBackwardSpace;

			Tensor mUnembed, mFinalLayer;
			Tensor mWteWeight;

			Forward* mForward;

			static constexpr std::size_t mVocabModel = mDVocab * mDModel;

		public:
			
			void setup(Forward* forward) {

				mForward = forward;

				mBackwardSpace.resize(mSeqVocab + mSeqModel + mVocabModel);
				auto backwardSpace = mBackwardSpace.begin();
				
				mUnembed = { {backwardSpace, mSeqVocab}, mDSeq, mDVocab };
				std::advance(backwardSpace, mSeqVocab);

				mFinalLayer = { {backwardSpace, mSeqModel}, mDSeq, mDModel };
				std::advance(backwardSpace, mSeqModel);

				mWteWeight = { {backwardSpace, mVocabModel}, mDVocab, mDModel };
				std::advance(backwardSpace, mVocabModel);
			}

			void backward(TokensView nextTokens) {

				auto& forward = *mForward;
				auto& parallel = forward.mParallelInput;
				parallel.section(nextTokens.size());

				auto& forwardSoftmax = forward.mUnembedActivationsSoftmax;
				Diagnostics::sumf(forwardSoftmax, "64");

				auto softmaxSpan = forwardSoftmax.spanTEnd(nextTokens.size() - 1);
				std::copy(softmaxSpan.begin(), softmaxSpan.end(), mUnembed.mTensor.begin());

				Token token;
				Tensor::TensorView unembed;

				for (auto i : std::views::iota( 0ULL, nextTokens.size())) {

					unembed = mUnembed.spanT(i);
					token = nextTokens[i];

					unembed[token] -= 1.0f;
				};

				Diagnostics::sumf(mUnembed, "0.0009");

				auto inputs = forward.mFinalLayer.getActivations();
				auto inputsSpanEnd = inputs.spanTEnd(nextTokens.size() - 1);
				std::fill(inputsSpanEnd.begin(), inputsSpanEnd.end(), 0.0f);
				
				auto dInputs = mFinalLayer;
				auto dInputsSpanEnd = dInputs.spanTEnd(nextTokens.size() - 1);

				Tensor& wte = forward.mWteWeight;
				Tensor& dWte = mWteWeight;

				const float r_tokens = 1.0f / nextTokens.size();

				parallel([&](auto& section) {

					Tensor::TensorView input, dInput, output, weight, dWeight;

					auto& [first, second] = section.mOffsets;
					for (auto i : std::views::iota(first, second)) {

						output = mUnembed.spanT(i);
						dInput = dInputs.spanT(i);
						input = inputs.spanT(i);
					
						for( auto m : std::views::iota(0ULL, output.size())){
				
							float o = output[m];
							weight = wte.spanT(m);
							dWeight = dWte.spanT(m);
							
							for (const auto& [din, in, w, dw] : std::views::zip(dInput, input, weight, dWeight)) {
								din += o * w;
								dw = o * in * r_tokens;
							}
						}
					}

					});

				auto dInputsSpanEnd = mFinalLayer.spanTEnd(nextTokens.size() - 1);
				std::transform(std::execution::par_unseq, dInputsSpanEnd.begin(), dInputsSpanEnd.end(), dInputsSpanEnd.begin(), [&](auto f) {return f * r_tokens; });

				Diagnostics::sumf(dInputsSpanEnd, "-0.0403");

			}

		} mBackward;

	public:

		void chat() {

			//this function prompts chatgpt repeatedly for short single "sentences"

			bool chatting = true;
			Tokens scrollingTokens;
			const Token endl = mTranslator.getToken("\n");
			std::string line = "What color is the Sky?";
			
			do {
			
				scrollingTokens.clear();

				std::getline(std::cin, line);
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

			//this function takes input tokens, up to dseq in number
			//and continues to predict until end of sentence or distance is reached
			//end of sentence is "." or "?" or "!"
			
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

			for( auto s : std::views::iota(0ULL, distance )){
			
				if (scrolled)
					ffAvg.accumulateTime([&]() {
						newToken = mForward.feedForward(tokens);
						});
				else
					fmAvg.accumulateTime([&]() {
						newToken = mForward.feedMore(tokens);
						});
			
				auto printAvgTime = [&]() {

					auto& updated = scrolled ? ffAvg : fmAvg;
					std::print("{},", updated.average() );
					};
				printAvgTime();

				scrolled = addToken(newToken);

				if (endOfSentence) break;
			}
		}
	};
}