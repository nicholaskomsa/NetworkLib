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
			, mTestInputSize = 64;	//vs dSeq for full size or 64 for test size

		using Floats = Tensor::Floats;
		using Token = std::uint16_t;
		using Tokens = std::vector<Token>;
		using TokensView = std::span<Token>;
		using IotaView = std::ranges::iota_view<std::size_t, std::size_t>;

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
			static double sumAbsf(Tensor::ConstView tensor, std::string_view expected) {
				double sum = std::reduce(tensor.begin(), tensor.end(), double(0.0), [](double a, float b) {return a + std::abs(b); });
				std::print("{}=={}\n", expected, sum);
				return sum;
			}
			static double sumAbsf(const Tensor& tensor, std::string_view expected) {
				return sumAbsf(tensor.mTensor, expected);
			}
			static double attnSumAbsf(const Tensor& tensor, std::size_t offset, std::string_view expected) {

				auto inputs = std::views::iota(0ULL, mTestInputSize);

				double fieldSum = std::reduce( inputs.begin(), inputs.end(), 0.0, [&](double sum, auto i) {
					
					auto tensorView = tensor.constView(i);

					for( auto h : std::views::iota(0ULL, mHeadNum) ) {

						auto headOffset = h * mHeadsPerDModel;

						auto fieldView = Tensor::constField(tensorView, headOffset + offset, mHeadsPerDModel);

						sum = std::reduce(fieldView.begin(), fieldView.end(), sum, [](double sum, float f) {return sum + std::abs(f); });
					}

					return sum;
				});

				std::print("{}=={}\n", expected, fieldSum);
				return fieldSum;
			}
		
			void firstCitizenTest64();
			void feedForwardSpeed1024();
			void simpleChat();
			void crossEntropyTest64();
			void backwardTest64();
			void SGDTest64();
		};
		friend class Diagnostics;

	private:

		static constexpr auto  mDModel3 = mDModel * 3, mDModel4 = mDModel * 4
			, mSeqModel = mDSeq * mDModel
			, mSeqModel3 = mDSeq * mDModel3
			, mSeqModel4 = mDSeq * mDModel4
			, mSeqSeqHead = mDSeq * mDSeq * mHeadNum
			, mSeqVocab = mDSeq * mDVocab
			, mModel4Model = mDModel4 * mDModel
			, mModel3Model = mDModel3 * mDModel
			, mModelModel = mDModel * mDModel
			, mVocabModel = mDVocab * mDModel;

		static void forward(std::size_t i, const Tensor& inputTensor, Tensor& outputTensor, const Tensor& weightTensor, const Tensor& biasTensor, Parallel& parallel);
		static void forward(const Tensor& inputTensor, Tensor& outputTensor, const Tensor& weightTensor, const Tensor& biasTensor, Parallel& parallel);
		static void softmax(std::size_t i, Tensor::ConstView input, Tensor::View output);

		static TimeAverage<milliseconds> mBackwardTime;

		using PartialBiasWeight = std::pair<Floats, Floats>;
		static void backward(const Tensor& dOutputs, const Tensor& weights, Tensor& dWeights, Tensor& dBias, const Tensor& inActivations, Tensor& outActivations, Parallel& parallel) {

			mBackwardTime.accumulateTime([&](){

				auto dWeightsBlock = dWeights.viewBlock();
				auto dBiasView = dBias.view();

				parallel([&](auto& section) {

					auto& [pdBias, pdWeightsFloats] = std::any_cast<PartialBiasWeight&>(section.mAny);

					pdBias.clear();
					pdBias.resize(dBias.mX, 0.0f);

					pdWeightsFloats.clear();
					pdWeightsFloats.resize(dWeights.size2D(), 0.0f);
					Tensor pdWeights = { pdWeightsFloats, dWeights.mX, dWeights.mY };

					Tensor::ConstView dOutput, activations, weight;
					Tensor::View pdWeight, dActivations;

					for (auto i : section.mIotaView) {

						dOutput = dOutputs.constView(i);

						for (const auto& [b, o] : std::views::zip(pdBias, dOutput))
							b += o;

						activations = inActivations.constView(i);
						dActivations = outActivations.view(i);

						for (auto m : std::views::iota(0ULL, activations.size())) {

							weight = weights.constView(m);
							pdWeight = pdWeights.view(m);

							float in = activations[m];

							float dot = 0.0f;

							for (const auto& [pdW, o, w] : std::views::zip(pdWeight, dOutput, weight)) {
								pdW += in * o;
								dot += w * o;
							}

							dActivations[m] = dot;
						}
					}

					}, [&](auto& section) {

						auto& [pdBias, pdWeightsFloats] = std::any_cast<PartialBiasWeight&>(section.mAny);
						Tensor pdWeights = { pdWeightsFloats, dWeights.mX, dWeights.mY };

						for (const auto& [b, pb] : std::views::zip(dBiasView, pdBias))
							b += pb;

						for (const auto& [w, pw] : std::views::zip(dWeightsBlock, pdWeights.viewBlock()))
							w += pw;


						});
				});
		}
		static void softmaxBack(const IotaView& iotaView, Tensor::ConstView input, Tensor::ConstView output, Tensor::View dSoftmax) {
			
			float softmaxSum = std::reduce(iotaView.begin(), iotaView.end(), 0.0f, [&](auto sum, auto m) {
				return sum + input[m] * output[m];
				});

			for (auto m : iotaView ) 
				dSoftmax[m] = input[m] * (output[m] - softmaxSum);
			
		}
		static void sgd(Tensor::View weights, Tensor::ConstView gradients, float learnRate) {

			std::transform(std::execution::seq, weights.begin(), weights.end(), gradients.begin(), weights.begin(),
				[&](auto& w, auto& g) {
					return w - g * learnRate;
				});
		}

		class MLP {

			friend class Diagnostics;

			Tensor mCFCBias, mCFCWeight, mCProjBias, mCProjWeight;
			Tensor mCFCActivations, mCProjActivations, mGeluActivations;

			Tensor mGeluCDF;

			static constexpr float r_sqrt2 = 1.0f / std::numbers::sqrt2;
			static const float r_sqrt2Pi;

			static TimeAverage<milliseconds> mBackwardGeluTime;
		public:

			static std::size_t getBackwardSize() {
				return mModel4Model * 2 + mSeqModel4 * 2 + mDModel4 + mDModel;
			}
			void load(Floats::iterator& backwardSpace) {

				mCProjWeight = { backwardSpace, mDModel4, mDModel };
				mCProjBias = { backwardSpace, mDModel };
				mCFCBias = { backwardSpace, mDModel4 };
				mCFCWeight = { backwardSpace, mDModel, mDModel4 };
				mGeluActivations = { backwardSpace, mDSeq, mDModel4 };
				mCFCActivations = { backwardSpace, mDSeq, mDModel4 };
			}
			void load(auto&& cfcBias, auto&& cfcWeight, auto&& cProjBias, auto&& cProjWeight, Floats::iterator& activationSpace);
			const Tensor& getCProjActivations() const;

			void forward(const Tensor& input, Parallel& parallel);
			void forward(std::size_t i, const Tensor& input, Parallel& parallel);

			void backward(const MLP& mlp, const Tensor& linear, const Tensor& dResidual, Tensor& dLinear, Parallel& parallel) {

				GPT2::backward(dResidual, mlp.mCProjWeight, mCProjWeight, mCProjBias, mlp.mGeluActivations, mGeluActivations, parallel);

				auto backwardGelu = [&]() {

					auto& forwardCFCs = mlp.mCFCActivations;
					auto& forwardCDFs = mlp.mGeluCDF;
					auto& dGelus = mGeluActivations;
					auto& dCFCs = mCFCActivations;

					parallel([&](auto& section) {

						Tensor::ConstView inputs, dGelu, cdfs;
						Tensor::View dInputs;

						for (auto i : section.mIotaView) {

							inputs = forwardCFCs.constView(i);
							cdfs = forwardCDFs.constView(i);
							dGelu = dGelus.constView(i);
							dInputs = dCFCs.view(i);

							for (const auto& [dout, in, din, cdf] : std::views::zip(dGelu, inputs, dInputs, cdfs)) {

								float dGeluDIn = cdf + in * (r_sqrt2Pi * std::exp(-0.5f * in * in));

								din = dout * dGeluDIn;
							}
						}

						});

					};

				mBackwardGeluTime.accumulateTime([&]() {
					backwardGelu();
					});

				GPT2::backward(mCFCActivations, mlp.mCFCWeight, mCFCWeight, mCFCBias, linear, dLinear, parallel);
			}

			void sgd(const MLP& gradients, float learnRate) {

				GPT2::sgd(mCFCWeight.viewBlock(), gradients.mCFCWeight.constViewBlock(), learnRate);
				GPT2::sgd(mCFCBias.view(), gradients.mCFCBias.constView(), learnRate);

				GPT2::sgd(mCProjWeight.viewBlock(), gradients.mCProjWeight.constViewBlock(), learnRate);
				GPT2::sgd(mCProjBias.view(), gradients.mCProjBias.constView(), learnRate);
			}
		};

		class LinearLayer {
		public:
			friend class Diagnostics;

			Tensor mBias, mWeight;
			Tensor mActivations;

			//for backward
			Floats mMean, mRStdDev;

			static TimeAverage<milliseconds> mBackwardTime;

		public:

			static std::size_t getBackwardSize() {
				return mSeqModel + mDModel * 2;
			}
			void load(Floats::iterator& backwardSpace) {

				mActivations = { backwardSpace, mDSeq, mDModel };
				mBias = { backwardSpace, mDModel };
				mWeight = { backwardSpace, mDModel  };
			}

			void load(Tensor&& bias, Tensor&& weight, Floats::iterator& activationSpace);
			Tensor& getActivations();

			void normalise(std::size_t i, const Tensor& input);
			void normalise(const Tensor& input, Parallel& parallel);

			void backward(const LinearLayer& inputLayer, const Tensor& inputs, Tensor& dInputs, Parallel& parallel){
				
				mBackwardTime.accumulateTime([&]{

					const Tensor& dActivations = mActivations;
					Tensor::View dBias = mBias.view()
						, dWeight = mWeight.view();

					Tensor::ConstView weight = inputLayer.mWeight.constView();
					const Floats& means = inputLayer.mMean
						, &rStdDevs = inputLayer.mRStdDev;

					parallel([&](auto& section) {

						auto& [pdBias, pdWeight] = std::any_cast<PartialBiasWeight&>(section.mAny);

						pdBias.clear();
						pdBias.resize(inputs.mY, 0.0f);

						pdWeight.clear();
						pdWeight.resize(inputs.mY, 0.0f);

						Tensor::ConstView dOut, input;
						Tensor::View dInput;

						for (auto i : section.mIotaView) {

							dOut = dActivations.constView(i);
							input = inputs.constView(i);
							dInput = dInputs.view(i);

							float mean = means[i]
								, rStdDev = rStdDevs[i]
								, meanPartial = 0.0f
								, stdDevPartial = 0.0f;

							float dInNorm;

							for (const auto& [dB, o, i, dW, w, dI] : std::views::zip(pdBias, dOut, input, pdWeight, weight, dInput)) {

								dI = (i - mean) * rStdDev; //==inNorm will pass through as dI

								dW += o * dI;
								dB += o;

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

						}, [&](auto& section) {

							auto& [partialBias, partialWeight] = std::any_cast<PartialBiasWeight&>(section.mAny);

							for (const auto& [b, w, pb, pw] : std::views::zip(dBias, dWeight, partialBias, partialWeight)) {
								b += pb;
								w += pw;
							}

							});

					});
			}
	
			void sgd(const LinearLayer& gradients, float learnRate) {

				GPT2::sgd(mWeight.view(), gradients.mWeight.constView(), learnRate);
				GPT2::sgd(mBias.view(), gradients.mBias.constView(), learnRate);
			}

		};
		class AttnLayer {

			friend class Diagnostics;

			static Parallel mParallelHeads;

			LinearLayer mL1, mL2;

			Tensor mBias, mCAttnBias, mCAttnWeight, mCProjBias, mCProjWeight;
			Tensor mCAttnActivations, mAttnActivations, mAttnSoftmaxActivations, mAttnZ, mCProjActivations;
			Tensor mResidualActivation1, mResidualActivation2
				, mResidualActivation1Out; //for backward

			MLP mMLP;

			static TimeAverage<milliseconds> mBackwardAttnTime;

			static const float r_sqrtHeadsPerDModel;

			void calculateQKAtten(std::size_t headOffset, std::size_t i, Tensor::View attnOut);
			void calculateVAtten(std::size_t headOffset, std::size_t i, Tensor::ConstView attnOutSoftmax);
			
			void backwardVAtten(const AttnLayer& attn, std::size_t headOffset, const IotaView& qView, std::size_t i, Tensor::ConstView inputAttnOutSoftmax, Tensor::View outputAttnOutSoftmax) {

				const auto vOffset = mVOffset + headOffset;
				Tensor::ConstView vh, dzh = Tensor::constField(mAttnZ.view(i), headOffset, mHeadsPerDModel );
				Tensor::View dvh;
				float factor, dot;

				for (auto m : qView) {

					vh = Tensor::constField(attn.mCAttnActivations.constView(m), vOffset, mHeadsPerDModel);
					dvh = Tensor::field(mCAttnActivations.view(m), vOffset, mHeadsPerDModel);

					factor = inputAttnOutSoftmax[m];
					dot = 0.0f;

					for (const auto& [dv, dz, v] : std::views::zip(dvh, dzh, vh)) {
						dv += factor * dz;
						dot += v * dz;
					}

					outputAttnOutSoftmax[m] += dot;
				}
			}
			void backwardQKAtten(const AttnLayer& attn, std::size_t headOffset, const IotaView& qView, std::size_t i, Tensor::ConstView attnActivations) {

				const auto qOffset = mQOffset + headOffset;
				Tensor::ConstView kh, qh = Tensor::constField(attn.mCAttnActivations.constView(i), qOffset, mHeadsPerDModel);
				Tensor::View dkh, dqh = Tensor::field(mCAttnActivations.view(i), qOffset, mHeadsPerDModel);

				float o;
				const auto kOffset = mKOffset + headOffset;

				for (auto m : qView) {

					kh = Tensor::constField(attn.mCAttnActivations.constView(m), kOffset, mHeadsPerDModel);
					dkh = Tensor::field(mCAttnActivations.view(m), kOffset, mHeadsPerDModel);
					o = attnActivations[m] * r_sqrtHeadsPerDModel;

					for (const auto& [q, dq, k, dk] : std::views::zip(qh, dqh, kh, dkh)) {

						dq += o * k;
						dk += o * q;
					}
				}
			}
			void multiHeadedAttn(std::size_t m);

			void attention(std::size_t m);
			void attention(Parallel& parallel);

			void multiHeadedAttnBack(AttnLayer& attn, Parallel& parallel) {


				auto iView = std::views::iota(0ULL, mTestInputSize);
				
				mParallelHeads([&](auto& section) {

					IotaView qView;

					for (auto h : section.mIotaView) {

						const auto headOffset = h * mHeadsPerDModel;

						for (auto i : iView ) {

							Tensor::ConstView inputAttnOutSoftmax = attn.mAttnSoftmaxActivations.constView(h, i);
							Tensor::View outputAttnOutSoftmax = mAttnSoftmaxActivations.view(h, i)
								, attnActivations = mAttnActivations.view(h, i);

							qView = std::views::iota(0ULL, i + 1);

							backwardVAtten( attn, headOffset, qView, i, inputAttnOutSoftmax, outputAttnOutSoftmax);
							
							softmaxBack(qView, inputAttnOutSoftmax, outputAttnOutSoftmax, attnActivations);

							backwardQKAtten(attn, headOffset, qView, i, attnActivations);
						}
					}
					});
			}

			void residual(std::size_t i, const Tensor& inputTensor, const Tensor& projectionTensor, Tensor& residualTensor);
			void residual(const Tensor& inputTensor, const Tensor& projectionTensor, Tensor& residualTensor, Parallel& parallel);
		
			void residualBack(const Tensor& a, const Tensor& b, Tensor& outputTensor) {

				Tensor::ConstView aBlock = a.constViewBlock()
					, bBlock = b.constViewBlock();
				Tensor::View outputBlock = outputTensor.viewBlock();

				std::transform(std::execution::par_unseq, aBlock.begin(), aBlock.end(), bBlock.begin(), outputBlock.begin(), [](auto a, auto b) {return a + b; });
			}

		public:

			using ReadTensorFunctor = std::function<Tensor(std::string_view)>;
			void load(ReadTensorFunctor&& readTensorByName, std::size_t layerIdx, Floats::iterator& activationSpace);

			static std::size_t getBackwardSize(){

				return mSeqModel * 3
					+ LinearLayer::getBackwardSize() * 2
					+ MLP::getBackwardSize()
					+ mDModel3 + mModel3Model
					+ mDModel + mModelModel
					+ mSeqModel
					+ mSeqSeqHead*2
					+ mSeqModel3;
			}
			void load(Floats::iterator& backwardSpace) {

				mResidualActivation2 = { backwardSpace, mDSeq, mDModel };
				mMLP.load(backwardSpace);
				mL2.load(backwardSpace);
				mResidualActivation1Out = { backwardSpace, mDSeq, mDModel };
				mResidualActivation1 = { backwardSpace, mDSeq, mDModel };

				//mBias = { backwardSpace, mDSeq, mDModel };
				mCAttnBias = { backwardSpace, mDModel3 };
				mCAttnWeight = { backwardSpace, mDModel, mDModel3 }; 
				mCProjBias = { backwardSpace, mDModel }; 
				mCProjWeight = { backwardSpace, mDModel, mDModel };
				
				mAttnZ = { backwardSpace, mDSeq, mDModel };
				mAttnSoftmaxActivations = { backwardSpace, mDSeq, mDSeq, mHeadNum };
				mCAttnActivations = { backwardSpace, mDSeq, mDModel3 };

				mAttnActivations = { backwardSpace, mDSeq, mDSeq, mHeadNum };

				mL1.load(backwardSpace);
			}

			Tensor& forward(const Tensor& inputTensor, Parallel& parallel);
			Tensor& forward(std::size_t i, const Tensor& inputTensor, Parallel& parallel);

			Tensor& getOutput() { return mResidualActivation2; } 
		
			void backward( AttnLayer& attn, const Tensor& forwardResidual2, Tensor& residual2, Parallel& parallel) {

				mMLP.backward(attn.mMLP, attn.mL2.getActivations(), mResidualActivation2, mL2.mActivations, parallel);
				
				mL2.backward(attn.mL2, attn.mResidualActivation1, mResidualActivation1Out, parallel);

				residualBack(mResidualActivation2, mResidualActivation1Out, mResidualActivation1);
				
				GPT2::backward(mResidualActivation1, attn.mCProjWeight, mCProjWeight, mCProjBias, attn.mAttnZ, mAttnZ, parallel);

				mBackwardAttnTime.accumulateTime([&] {

					multiHeadedAttnBack(attn, parallel);

					});
				
				GPT2::backward(mCAttnActivations, attn.mCAttnWeight, mCAttnWeight, mCAttnBias, attn.mL1.mActivations, mL1.mActivations, parallel);

				mL1.backward(attn.mL1, forwardResidual2, mResidualActivation1Out, parallel);

				residualBack(mResidualActivation1Out, mResidualActivation1, residual2);
			}

			void sgd(const AttnLayer& gradients, float learnRate) {

				mL1.sgd(gradients.mL1, learnRate);
				mL2.sgd(gradients.mL2, learnRate);

				GPT2::sgd(mCAttnWeight.viewBlock(), gradients.mCAttnWeight.constViewBlock(), learnRate);
				GPT2::sgd(mCAttnBias.view(), gradients.mCAttnBias.constView(), learnRate);

				GPT2::sgd(mCProjWeight.viewBlock(), gradients.mCProjWeight.constViewBlock(), learnRate);
				GPT2::sgd(mCProjBias.view(), gradients.mCProjBias.constView(), learnRate);

				mMLP.sgd(gradients.mMLP, learnRate);
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

			Tensor mUnembed, mWteWeight, mEmbed, mWpeWeight;

			LinearLayer mFinalLayer;

			std::array<AttnLayer, mAttnLayersNum> mAttnLayers;

			Forward* mForward;

			static TimeAverage<milliseconds> mEmbedTime, mUnembedTime, mLayersTime;

		public:
			
			void setup(Forward* forward) {

				mForward = forward;

				mBackwardSpace.resize(mSeqVocab + mVocabModel 
					+ LinearLayer::getBackwardSize()
					+ AttnLayer::getBackwardSize() * mAttnLayersNum
					+ mSeqModel*2);

				auto backwardSpace = mBackwardSpace.begin();
				
				mUnembed = { backwardSpace, mDSeq, mDVocab };
				mWteWeight = { backwardSpace, mDVocab, mDModel };
				mFinalLayer.load(backwardSpace);

				for( auto& attnLayer : mAttnLayers ) 
					attnLayer.load(backwardSpace);
				
				mEmbed = { backwardSpace, mDSeq, mDModel };
				mWpeWeight = { backwardSpace, mDSeq, mDModel };

				mParallelInput.setup(PartialBiasWeight{}, mTestInputSize, 32);
			}

			void unEmbedOutputs(TokensView nextTokens, Parallel& parallel) {

				auto& forward = *mForward;

				Tensor& forwardSoftmax = forward.mUnembedActivationsSoftmax;

				auto softmaxBlock = forwardSoftmax.viewBlock(nextTokens.size() - 1);
				std::copy(softmaxBlock.begin(), softmaxBlock.end(), mUnembed.mTensor.begin());

				Token token;
				Tensor::View unembed;

				for (auto i : std::views::iota(0ULL, nextTokens.size())) {

					unembed = mUnembed.view(i);
					token = nextTokens[i];

					unembed[token] -= 1.0f;
				};


				Tensor& inputs = forward.mFinalLayer.getActivations();
				Tensor& dInputs = mFinalLayer.getActivations();

				Tensor& wte = forward.mWteWeight;
				Tensor& dWte = mWteWeight;

				const float r_tokens = 1.0f / nextTokens.size();

				parallel([&](auto& section) {

					Tensor::View input, dInput, output, weight, dWeight;

					auto& [pdBias, pdWeightsFloats] = std::any_cast<PartialBiasWeight&>(section.mAny);
					pdWeightsFloats.clear();
					pdWeightsFloats.resize(dWte.size2D(), 0.0f);
					Tensor pdWeights = { pdWeightsFloats, dWte.mX, dWte.mY };

					float o, o2;

					for (auto i : section.mIotaView) {

						output = mUnembed.view(i);
						dInput = dInputs.view(i);
						input = inputs.view(i);

						for (auto m : std::views::iota(0ULL, output.size())) {

							o = output[m];
							o2 = o * r_tokens;
							
							weight = wte.view(m);
							dWeight = pdWeights.view(m);

							for (const auto& [din, in, w, dw] : std::views::zip(dInput, input, weight, dWeight)) {
								din += o * w;
								dw += o2 * in;
							}
						}
					}

					}, [&](auto& section) {

						auto& [pdBias, pdWeightsFloats] = std::any_cast<PartialBiasWeight&>(section.mAny);

						Tensor::View dWteBlock = dWte.viewBlock();

						std::transform(std::execution::par_unseq, dWteBlock.begin(), dWteBlock.end()
							, pdWeightsFloats.begin(), dWteBlock.begin(), [&](auto& w, auto& pw) {
								return w + pw;
							});

						//for (const auto& [w, pdw] : std::views::zip(dWteBlock, pdWeightsFloats))
						//	w += pdw;
						
						});
				Tensor::View dInputsBlock = dInputs.viewBlock();

				std::transform(std::execution::par_unseq, dInputsBlock.begin(), dInputsBlock.end(), dInputsBlock.begin(), [&](auto f) {return f * r_tokens; });
			}

			void embedOutputs(TokensView tokens, Parallel& parallel) {

				parallel([&](auto& section) {

					Tensor::ConstView dout;
					Tensor::View wte, wpe;

					for (auto i : section.mIotaView) {

						dout = mEmbed.constView(i);
						wte = mWteWeight.view(tokens[i]);
						wpe = mWpeWeight.view(i);

						for (const auto& [o, t, p] : std::views::zip(dout, wte, wpe)) {
							p += o;
							t += o;
						}
					}

					});
			}
			void backward(TokensView tokens, TokensView nextTokens) {

				auto& forward = *mForward;
				mParallelInput.section(tokens.size());

				std::fill(mBackwardSpace.begin(), mBackwardSpace.end(), 0.0f);

				mUnembedTime.accumulateTime([&]() {
					unEmbedOutputs(nextTokens, mParallelInput);
					});

				mFinalLayer.backward(forward.mFinalLayer, forward.mAttnLayers.back().getOutput()
					, mAttnLayers.back().getOutput(), mParallelInput);
				

				auto& forwardLayers = forward.mAttnLayers;
				auto& layers = mAttnLayers;

				mLayersTime.accumulateTime([&]() {

					for (auto l : std::views::iota(1ULL, mAttnLayers.size()) | std::views::reverse) {

						Tensor& forwardOutput = forwardLayers[l - 1].getOutput()
							, & output = layers[l - 1].getOutput();

						AttnLayer& forwardLayer = forwardLayers[l]
							, & layer = layers[l];

						layer.backward(forwardLayer, forwardOutput, output, mParallelInput);
					}

					layers.front().backward(forwardLayers.front(), forward.mWActivations
						, mEmbed, mParallelInput);
					});

				mEmbedTime.accumulateTime([&]() {
					embedOutputs(tokens, mParallelInput);
					});
			}


			void sgd(float learnRate = 0.0002) {

				auto& forward = *mForward;
				auto& forwardLayers = forward.mAttnLayers;
				auto& layers = mAttnLayers;

				auto n = mTestInputSize - 1;






				GPT2::sgd(forward.mWpeWeight.viewBlock(n), mWpeWeight.viewBlock(n), learnRate);
				GPT2::sgd(forward.mWteWeight.viewBlock(n), mWteWeight.viewBlock(n), learnRate);

				GPT2::sgd(forward.mFinalLayer.mBias.view(n), mFinalLayer.mBias.view(n), learnRate);
				GPT2::sgd(forward.mFinalLayer.mWeight.viewBlock(), mFinalLayer.mWeight.viewBlock(), learnRate);

				auto iotaView = std::ranges::iota_view(0ULL, mAttnLayers.size());

				std::for_each(std::execution::par, iotaView.begin(), iotaView.end(), [&](auto i) {

					auto& layer = forwardLayers[i];	
					auto& gradient = layers[i];

					layer.sgd(gradient, learnRate);
					});
	
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