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
		public:
			friend class Diagnostics;

			Tensor mBias, mWeight;
			Tensor mActivations;

			//for backward
			using PartialBiasWeight = std::pair<Floats, Floats>;
			Floats mMean, mRStdDev;
		public:

			void load(Floats::iterator& backwardSpace) {

				mActivations = { {backwardSpace, mSeqModel}, mDSeq, mDModel };
				std::advance(backwardSpace, mSeqModel);

				mBias = { {backwardSpace, mDModel}, mDModel };
				std::advance(backwardSpace, mDModel);

				mWeight = { {backwardSpace, mDModel}, mDModel  };
				std::advance(backwardSpace, mDModel);

				mMean.resize(mTestInputSize);
				mRStdDev.resize(mTestInputSize);
			}

			void load(Tensor&& bias, Tensor&& weight, Floats::iterator& activationSpace);
			Tensor& getActivations();

			void normalise(std::size_t i, const Tensor& input);
			void normalise(const Tensor& input, Parallel& parallel);

			void backward(const LinearLayer& inputLayer, Tensor& inputs, Tensor& dInputs, Parallel& parallel) {

				Tensor::TensorView dBias = mBias.span();
				Tensor& dActivations = mActivations;
				Tensor::TensorView dWeight = mWeight.span();

				Tensor::TensorView weight = inputLayer.mWeight.span();
				auto& means = inputLayer.mMean;
				auto& rStdDevs = inputLayer.mRStdDev;

				parallel([&](auto& section) {

					auto& [dBias, dWeight] = std::any_cast<LinearLayer::PartialBiasWeight&>(section.mAny);

					dBias.clear();
					dBias.resize(inputs.mY, 0.0f);

					dWeight.clear();
					dWeight.resize(inputs.mY, 0.0f);

					Tensor::TensorView dOut, input, dInput;

					auto [first, second] = section.mOffsets;
					for (auto i : std::views::iota(first, second)) {

						dOut = dActivations.spanT(i);
						input = inputs.spanT(i);

						float mean = means[i]
							, rStdDev = rStdDevs[i]
							, meanPartial = 0.0f
							, stdDevPartial = 0.0f;

						for (const auto& [g, o, i, dW, w] : std::views::zip(dBias, dOut, input, dWeight, weight)) {

							float dInNorm = o * w
								, inNorm = (i - mean) * rStdDev;

							g += o;
							dW += o * inNorm;

							meanPartial += dInNorm;
							stdDevPartial += dInNorm * inNorm;
						}

						meanPartial /= dBias.size();
						stdDevPartial /= dBias.size();

						dInput = dInputs.spanT(i);

						for (const auto& [o, i, w, dI] : std::views::zip(dOut, input, weight, dInput)) {

							float dInNorm = o * w
								, inNorm = (i - mean) * rStdDev;

							dI = dInNorm - meanPartial - inNorm * stdDevPartial;
							dI *= rStdDev;
						}
					}

					}, [&](Parallel::Section& section) {

						auto& [partialBias, partialWeight] = std::any_cast<LinearLayer::PartialBiasWeight&>(section.mAny);

						for (const auto& [b, w, pb, pw] : std::views::zip(dBias, dWeight, partialBias, partialWeight)) {
							b += pb;
							w += pw;
						}

						});

			}
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
			
			void load(Floats::iterator& backwardSpace) {

				mResidualActivation2 = { {backwardSpace, mSeqModel}, mDSeq, mDModel };
				std::advance(backwardSpace, mSeqModel);



			}

			Tensor& forward(Tensor& inputTensor, Parallel& parallel);
			Tensor& forward(std::size_t i, const Tensor& inputTensor, Parallel& parallel);

			Tensor& getOutput() { return mResidualActivation2; }
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

			Parallel mParallelInput;

			Floats mBackwardSpace;

			Tensor mUnembed, mWteWeight;

			LinearLayer mFinalLayer;

			std::array<AttnLayer, mAttnLayersNum> mAttnLayers;

			Forward* mForward;

			static constexpr std::size_t mVocabModel = mDVocab * mDModel;

		public:
			
			void setup(Forward* forward) {

				mForward = forward;

				mBackwardSpace.resize(mSeqVocab + mVocabModel + (mSeqModel + mDModel*2) + (mSeqModel) * mAttnLayersNum);
				auto backwardSpace = mBackwardSpace.begin();
				
				mUnembed = { {backwardSpace, mSeqVocab}, mDSeq, mDVocab };
				std::advance(backwardSpace, mSeqVocab);

				mWteWeight = { {backwardSpace, mVocabModel}, mDVocab, mDModel };
				std::advance(backwardSpace, mVocabModel);

				mFinalLayer.load(backwardSpace);

				for( auto& attnLayer : mAttnLayers ) {

					attnLayer.load(backwardSpace);
				}

				mParallelInput.setup(std::pair<Floats, Floats>{}, mTestInputSize, 32);
			}

			void unEmbedOutputs(TokensView nextTokens) {

				auto& forward = *mForward;
				auto& parallel = forward.mParallelInput;
				parallel.section(nextTokens.size());

				auto& forwardSoftmax = forward.mUnembedActivationsSoftmax;
				Diagnostics::sumf(forwardSoftmax, "64");

				auto softmaxSpan = forwardSoftmax.spanTEnd(nextTokens.size() - 1);
				std::copy(softmaxSpan.begin(), softmaxSpan.end(), mUnembed.mTensor.begin());

				Token token;
				Tensor::TensorView unembed;

				for (auto i : std::views::iota(0ULL, nextTokens.size())) {

					unembed = mUnembed.spanT(i);
					token = nextTokens[i];

					unembed[token] -= 1.0f;
				};

				Diagnostics::sumf(mUnembed, "0.0009");

				Tensor& inputs = forward.mFinalLayer.getActivations();
				Tensor& dInputs = mFinalLayer.getActivations();
				Tensor::TensorView dInputsSpanEnd = dInputs.spanTEnd(nextTokens.size() - 1);

				std::fill(dInputsSpanEnd.begin(), dInputsSpanEnd.end(), 0.0f);

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

						for (auto m : std::views::iota(0ULL, output.size())) {

							float o = output[m];
							weight = wte.spanT(m);
							dWeight = dWte.spanT(m);

							for (const auto& [din, in, w, dw] : std::views::zip(dInput, input, weight, dWeight)) {
								din += o * w;
								dw += o * in * r_tokens;
							}
						}
					}

					});

				std::transform(std::execution::par_unseq, dInputsSpanEnd.begin(), dInputsSpanEnd.end(), dInputsSpanEnd.begin(), [&](auto f) {return f * r_tokens; });
			
				Diagnostics::sumf(dInputsSpanEnd, "-0.0403");
			}

			void backward(TokensView nextTokens) {

				auto& forward = *mForward;
				auto& parallel = mParallelInput;
				parallel.section(nextTokens.size());

				unEmbedOutputs(nextTokens);
				
				Tensor& inputs = forward.mAttnLayers.back().getOutput();
				Tensor& dInputs = mAttnLayers.back().getOutput();

				mFinalLayer.backward(forward.mFinalLayer, inputs, dInputs, parallel);

				Diagnostics::sumf(mFinalLayer.mBias, "-0.0403");
				Diagnostics::sumf(mFinalLayer.mWeight, "-0.5371");
				Diagnostics::sumf(dInputs, "-.e8 on debug");

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