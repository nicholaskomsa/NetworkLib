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
			, mSeqVocab = mDSeq * mDVocab
			, mModel4Model = mDModel4 * mDModel;

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

			static double sumf(Tensor::ConstView tensorView, std::string_view expected) {

				double sum = std::reduce(tensorView.begin(), tensorView.end(), double(0.0));
				std::print("{}=={}\n", expected, sum);
				return sum;
			}
			static double sumf(const Tensor& tensor, std::string_view expected) {
				return sumf(tensor.mTensor, expected);
			}
			static double sumAbsf(const Tensor& tensor, std::string_view expected) {
				double sum = std::reduce(tensor.mTensor.begin(), tensor.mTensor.end(), double(0.0), [](auto a, auto b) {return a + std::abs(b); });
				std::print("{}=={}\n", expected, sum);
				return sum;
			}

			void firstCitizenTest64();
			void feedForwardSpeed1024();
			void simpleChat();
			void crossEntropyTest64();
			void backwardTest64();
		};
		friend class Diagnostics;

	private:

		static void forward(std::size_t i, const Tensor& inputTensor, Tensor& outputTensor, const Tensor& weightTensor, const Tensor& biasTensor, Parallel& parallel);
		static void forward(const Tensor& inputTensor, Tensor& outputTensor, const Tensor& weightTensor, const Tensor& biasTensor, Parallel& parallel);
		static void softmax(std::size_t i, Tensor::ConstView input, Tensor::View output);

		static void backward(const Tensor& dOutputs, const Tensor& weights, Tensor& dWeights, Tensor& dBias, const Tensor& inActivations, Tensor& outActivations, Parallel& parallel) {

			Tensor::ConstView dOutput, activations, weight;
			Tensor::View dWeight, dBias1, dActivations;

			dBias1 = dBias.view();

			for (auto i : std::views::iota(0ULL, parallel.mSize)) {

				dOutput = dOutputs.constViewT(i);

				for (const auto& [b, o] : std::views::zip(dBias1, dOutput))
					b += o;
			
				activations = inActivations.constViewT(i);
				dActivations = outActivations.viewT(i);

				for (auto m : std::views::iota(0ULL, activations.size())) {

					weight = weights.constViewT(m);
					dWeight = dWeights.viewT(m);

					float in = activations[m];

					float dot = 0.0f;

					for (const auto& [dW, o, w] : std::views::zip(dWeight, dOutput, weight)) {
						dW += in * o;
						dot += w * o;
					}

					dActivations[m] = dot;
				}
			}
		}

		class MLP {

			friend class Diagnostics;

			Tensor mCFCBias, mCFCWeight, mCProjBias, mCProjWeight;
			Tensor mCFCActivations, mCProjActivations, mGeluActivations;

			static constexpr float r_sqrt2 = 1.0f / std::numbers::sqrt2;
		public:

			void load(Floats::iterator& backwardSpace) {

				mCProjWeight = { { backwardSpace, mModel4Model }, mDModel4, mDModel };
				std::advance(backwardSpace, mModel4Model);

				mCProjBias = { { backwardSpace, mDModel }, mDModel };
				std::advance(backwardSpace, mDModel);

				mCFCBias = { { backwardSpace, mDModel4 }, mDModel4 };
				std::advance(backwardSpace, mDModel4);

				mCFCWeight = { { backwardSpace, mModel4Model }, mDModel, mDModel4 };
				std::advance(backwardSpace, mModel4Model);
				

				mGeluActivations = { { backwardSpace, mSeqModel4 }, mDSeq, mDModel4 };
				std::advance(backwardSpace, mSeqModel4);

			}
			void load(auto&& cfcBias, auto&& cfcWeight, auto&& cProjBias, auto&& cProjWeight, Floats::iterator& activationSpace);
			const Tensor& getCProjActivations() const;

			void forward(const Tensor& input, Parallel& parallel);
			void forward(std::size_t i, const Tensor& input, Parallel& parallel);

			void backward(const MLP& mlp, const Tensor& dOutputs, Parallel& parallel) {

				GPT2::backward(dOutputs, mlp.mCProjWeight, mCProjWeight, mCProjBias, mlp.mGeluActivations, mGeluActivations, parallel);

			}
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
			}

			void load(Tensor&& bias, Tensor&& weight, Floats::iterator& activationSpace);
			Tensor& getActivations();

			void normalise(std::size_t i, const Tensor& input);
			void normalise(const Tensor& input, Parallel& parallel);

			void backward(const LinearLayer& inputLayer, const Tensor& inputs, Tensor& dInputs, Parallel& parallel) {
				
				const Tensor& dActivations = mActivations;
				Tensor::View dBias = mBias.view()
					, dWeight = mWeight.view();

				std::fill(dBias.begin(), dBias.end(), 0.0f);
				std::fill(dWeight.begin(), dWeight.end(), 0.0f);

				Tensor::ConstView weight = inputLayer.mWeight.constView();
				const Floats& means = inputLayer.mMean
					, &rStdDevs = inputLayer.mRStdDev;

				parallel([&](auto& section) {

					auto& [dBias, dWeight] = std::any_cast<LinearLayer::PartialBiasWeight&>(section.mAny);

					dBias.clear();
					dBias.resize(inputs.mY, 0.0f);

					dWeight.clear();
					dWeight.resize(inputs.mY, 0.0f);

					Tensor::ConstView dOut, input;
					Tensor::View dInput;

					auto [first, second] = section.mOffsets;
					for (auto i : std::views::iota(first, second)) {

						dOut = dActivations.constViewT(i);
						input = inputs.constViewT(i);
						dInput = dInputs.viewT(i);

						float mean = means[i]
							, rStdDev = rStdDevs[i]
							, meanPartial = 0.0f
							, stdDevPartial = 0.0f;

						float dInNorm;

						for (const auto& [g, o, i, dW, w, dI] : std::views::zip(dBias, dOut, input, dWeight, weight, dInput)) {

							dI = (i - mean) * rStdDev; //==inNorm will pass through as dI

							dW += o * dI;
							g += o;

							dInNorm = o * w;
							meanPartial += dInNorm;
							stdDevPartial += dInNorm * dI;
						}

						meanPartial /= dBias.size();
						stdDevPartial /= dBias.size();

						for (const auto& [o, i, w, dI] : std::views::zip(dOut, input, weight, dInput)) {

							dInNorm = o * w;
							
							dI = dInNorm - meanPartial - dI * stdDevPartial;
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

			void calculateQKAtten(std::size_t headOffset, std::size_t i, Tensor::View attnOut);
			void calculateVAtten(std::size_t headOffset, std::size_t i, Tensor::View attnOutSoftmax);
			
			void multiHeadedAttn(std::size_t m);

			void attention(std::size_t m);
			void attention(Parallel& parallel);

			void residual(std::size_t i, const Tensor& inputTensor, const Tensor& projectionTensor, Tensor& residualTensor);
			void residual(const Tensor& inputTensor, const Tensor& projectionTensor, Tensor& residualTensor, Parallel& parallel);
		
		public:

			using ReadTensorFunctor = std::function<Tensor(std::string_view)>;
			void load(ReadTensorFunctor&& readTensorByName, std::size_t layerIdx, Floats::iterator& activationSpace);
			
			Tensor mUnembedOut;

			void load(Floats::iterator& backwardSpace) {

				mResidualActivation2 = { {backwardSpace, mSeqModel}, mDSeq, mDModel };
				std::advance(backwardSpace, mSeqModel);


				mMLP.load(backwardSpace);
			}

			Tensor& forward(const Tensor& inputTensor, Parallel& parallel);
			Tensor& forward(std::size_t i, const Tensor& inputTensor, Parallel& parallel);

			Tensor& getOutput() { return mResidualActivation2; } 
		
			void backward( AttnLayer& attn, Parallel& parallel) {

				mMLP.backward(attn.mMLP, getOutput(), parallel);
				
			}
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
			float crossEntropyLoss(TokensView nextTokens);

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

				mBackwardSpace.resize(mSeqVocab + mVocabModel + (mSeqModel + mModel4Model*2 + mDModel4 + mDModel ) + (mSeqModel + mModel4Model * 2 + mSeqModel4) * mAttnLayersNum);
				auto backwardSpace = mBackwardSpace.begin();
				
				mUnembed = { {backwardSpace, mSeqVocab}, mDSeq, mDVocab };
				std::advance(backwardSpace, mSeqVocab);

				mWteWeight = { {backwardSpace, mVocabModel}, mDVocab, mDModel };
				std::advance(backwardSpace, mVocabModel);

				mFinalLayer.load(backwardSpace);

				for( auto& attnLayer : mAttnLayers ) {

					attnLayer.load(backwardSpace);
				}

				mParallelInput.setup(LinearLayer::PartialBiasWeight{}, mTestInputSize, 32);
			}

			void unEmbedOutputs(TokensView nextTokens) {

				auto& forward = *mForward;
				auto& parallel = forward.mParallelInput;
				parallel.section(nextTokens.size());

				Tensor& forwardSoftmax = forward.mUnembedActivationsSoftmax;

				auto softmaxBlock = forwardSoftmax.viewTBlock(nextTokens.size() - 1);
				std::copy(softmaxBlock.begin(), softmaxBlock.end(), mUnembed.mTensor.begin());

				Token token;
				Tensor::View unembed;

				for (auto i : std::views::iota(0ULL, nextTokens.size())) {

					unembed = mUnembed.viewT(i);
					token = nextTokens[i];

					unembed[token] -= 1.0f;
				};


				Tensor& inputs = forward.mFinalLayer.getActivations();
				Tensor& dInputs = mFinalLayer.getActivations();
				Tensor::View dInputsBlock = dInputs.viewTBlock(nextTokens.size() - 1);

				std::fill(dInputsBlock.begin(), dInputsBlock.end(), 0.0f);

				Tensor& wte = forward.mWteWeight;
				Tensor& dWte = mWteWeight;

				Tensor::View dWteBlock = dWte.viewTBlock(dWte.mY - 1);
				std::fill(dWteBlock.begin(), dWteBlock.end(), 0.0f);

				const float r_tokens = 1.0f / nextTokens.size();

				parallel([&](auto& section) {

					Tensor::View input, dInput, output, weight, dWeight;

					auto& [first, second] = section.mOffsets;
					for (auto i : std::views::iota(first, second)) {

						output = mUnembed.viewT(i);
						dInput = dInputs.viewT(i);
						input = inputs.viewT(i);

						for (auto m : std::views::iota(0ULL, output.size())) {

							float o = output[m];
							weight = wte.viewT(m);
							dWeight = dWte.viewT(m);

							for (const auto& [din, in, w, dw] : std::views::zip(dInput, input, weight, dWeight)) {
								din += o * w;
								dw += o * in * r_tokens;
							}
						}
					}

					});

				std::transform(std::execution::par_unseq, dInputsBlock.begin(), dInputsBlock.end(), dInputsBlock.begin(), [&](auto f) {return f * r_tokens; });
			}

			void backward(TokensView nextTokens) {

				auto& forward = *mForward;
				auto& parallel = mParallelInput;
				parallel.section(nextTokens.size());

				unEmbedOutputs(nextTokens);

				const Tensor& inputs = forward.mAttnLayers.back().getOutput();
				Tensor& dInputs = mAttnLayers.back().getOutput();

				mFinalLayer.backward(forward.mFinalLayer, inputs, dInputs, parallel);

				mAttnLayers.back().backward(forward.mAttnLayers.back(), parallel);

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

				std::cout << mTranslator.decode(scrollingTokens)
					<< std::endl;

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