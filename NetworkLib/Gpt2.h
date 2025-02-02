#pragma once

#include <array>
#include <iostream>
#include <print>
#include <fstream>
#include <span>
#include <sstream>
#include <execution>
#include <boost/json.hpp>
#include <map>

#include "Tensor.h"

namespace NetworkLib {


	struct GPT2 {

		static constexpr auto mFilePath = "F:/software dev/programming2025/downloads/";
		static constexpr std::size_t mDVocab = 50257
			, mDModel = 768, mDModel3 = mDModel * 3, mDModel4 = mDModel * 4
			, mDSeq = 1024
			, mHeadNum = 12, mAttnLayersNum = 12
			, mHeadsPerDModel = mDModel / mHeadNum
			, mQOffset = 0, mKOffset = mDModel, mVOffset = mDModel * 2
			, mTestInputSize = 64;	//vs dSeq for full size or 64 for test size

		struct Error : public std::system_error {

			Error(std::errc code, const std::string& message);

			static void fileNotFound(const std::string& fileName);
		};

		using Floats = std::vector<float>;


		struct MLP {
			Tensor mCFCBias, mCFCWeight, mCProjBias, mCProjWeight;
			Tensor mCFCActivations, mCProjActivations, mGeluActivations;
		};
		struct LinearLayer {

			Tensor mBias, mWeight;
			Tensor mActivations;

			void normalise(auto& input) {

				Tensor::TensorView i, o, b, w;

				for (std::size_t m = 0; m < mTestInputSize; ++m) { //inputSize vs dseq

					i = input.spanT(m);
					o = mActivations.spanT(m);

					const auto mean = std::reduce(i.begin(), i.end()) / i.size();

					auto meanDiffSq = std::reduce(i.begin(), i.end(), 0.0f,
						[&](auto sum, auto x) {
							auto diff = x - mean;
							return sum + diff * diff;
						}) / i.size();

					auto r_stdDev = 1.0f / std::sqrt(meanDiffSq);

					b = mBias.span();
					w = mWeight.span();

					float inNorm = 0.0f;
					for (std::size_t n = 0; n < i.size(); ++n) {
						inNorm = (i[n] - mean) * r_stdDev;
						o[n] = inNorm * w[n] + b[n];
					}
				}
			}
		};
		struct AttnLayer {

			LinearLayer mL1, mL2;

			Tensor mBias, mCAttnBias, mCAttnWeight, mCProjBias, mCProjWeight;
			Tensor mCAttnActivations, mAttnActivations, mAttnSoftmaxActivations, mAttnZ, mCProjActivations;
			Tensor mResidualActivation1, mResidualActivation2;

			MLP mMLP;

			void attention() {

				//activations z cleared here
				std::fill(mAttnZ.mTensor.begin(), mAttnZ.mTensor.end(), 0.0f);

				const auto r_sqrtHeadsPerDModel = 1.0f / std::sqrtf(mHeadsPerDModel);

				for (std::size_t q_i = 0; q_i < mTestInputSize; ++q_i) {

					Tensor::TensorView q = mCAttnActivations.spanT(q_i)
						, z = mAttnZ.spanT(q_i);

					for (std::size_t h = 0; h < mHeadNum; ++h) {

						const auto headOffset = h * mHeadsPerDModel;
						Tensor::TensorView attnOut = mAttnActivations.spanT(h, q_i)
							, attnOutSoftmax = mAttnSoftmaxActivations.spanT(h, q_i);

						auto calculateQKAtten = [&]() {

							const auto qOffset = mQOffset + headOffset;
							Tensor::TensorView k, kh, qh = { q.data() + qOffset, mHeadsPerDModel };

							const auto kOffset = mKOffset + headOffset;
							for (std::size_t m = 0; m <= q_i; ++m) {

								k = mCAttnActivations.spanT(m);
								kh = { k.data() + kOffset, mHeadsPerDModel };

								float dot = 0.0f;

								for (std::size_t n = 0; n < qh.size(); ++n)
									dot += qh[n] * kh[n];

								attnOut[m] = dot * r_sqrtHeadsPerDModel;
							}
							};

						auto softmaxQ = [&](const auto& input, const auto& output) {

							const auto softmaxMax = *std::max_element(input.begin(), input.begin() + q_i);

							float softmaxSum = 0.0f;
							float softmaxExp = 0.0f;

							for (std::size_t m = 0; m <= q_i; ++m) {

								softmaxExp = std::expf(input[m] - softmaxMax);

								output[m] = softmaxExp;

								softmaxSum += softmaxExp;
							}

							const auto r_softmaxSum = 1.0f / softmaxSum;

							for (std::size_t m = 0; m <= q_i; ++m)
								output[m] *= r_softmaxSum;
							};

						auto calculateVAtten = [&]() {

							Tensor::TensorView v, vh, zh = { z.data() + headOffset, mHeadsPerDModel };
							auto factor = 0.0f;

							const auto vOffset = mVOffset + headOffset;
							for (std::size_t m = 0; m <= q_i; ++m) {

								v = mCAttnActivations.spanT(m);
								vh = { v.data() + vOffset, mHeadsPerDModel };

								factor = attnOutSoftmax[m];

								for (std::size_t n = 0; n < vh.size(); ++n)
									zh[n] += vh[n] * factor;

							}
							};

						calculateQKAtten();
						softmaxQ(attnOut, attnOutSoftmax);
						calculateVAtten();
					}
				}
			};
			void residual(const Tensor& inputTensor, const auto& projectionTensor, const auto& residualTensor) {

				Tensor::TensorView p, input, o;

				for (std::size_t i = 0; i < mTestInputSize; ++i) {

					p = projectionTensor.spanT(i);
					input = inputTensor.spanT(i);
					o = residualTensor.spanT(i);

					for (std::size_t m = 0; m < o.size(); ++m)
						o[m] = p[m] + input[m];
				}
			}


			void forward(const auto& inputTensor, const auto& outputTensor, const auto& weightTensor, const auto& biasTensor) {

				const auto b = biasTensor.span();

				Tensor::TensorView input, output, w;

				for (std::size_t i = 0; i < mTestInputSize; ++i) {

					input = inputTensor.spanT(i);
					output = outputTensor.spanT(i);

					std::copy(b.begin(), b.end(), output.begin());

					for (std::size_t m = 0; m < input.size(); ++m) {

						const auto& in = input[m];
						w = weightTensor.spanT(m);

						for (std::size_t n = 0; n < output.size(); ++n)
							output[n] += w[n] * in;
					}
				}
			}
			Tensor& forward(const auto& inputTensor) {

				mL1.normalise(inputTensor);
				forward(mL1.mActivations, mCAttnActivations, mCAttnWeight, mCAttnBias);

				attention();

				forward(mAttnZ, mCProjActivations, mCProjWeight, mCProjBias);

				residual(inputTensor, mCProjActivations, mResidualActivation1);

				mL2.normalise(mResidualActivation1);
				forward(mL2.mActivations, mMLP.mCFCActivations, mMLP.mCFCWeight, mMLP.mCFCBias);

				std::transform(mMLP.mCFCActivations.mTensor.begin(), mMLP.mCFCActivations.mTensor.end(), mMLP.mGeluActivations.mTensor.begin(),
					[](auto x) {
						return x * 0.5f * (1.0f + std::erff(x / std::sqrt(2.0f)));
					});

				forward(mMLP.mGeluActivations, mMLP.mCProjActivations, mMLP.mCProjWeight, mMLP.mCProjBias);

				residual(mResidualActivation1, mMLP.mCProjActivations, mResidualActivation2);

				return mResidualActivation2;
			}
		};

		Floats mTensorSpace, mActivationSpace;

		Tensor mWpeWeight, mWteWeight, mWActivations;
		LinearLayer mFinalLayer;
		std::array<AttnLayer, mAttnLayersNum> mAttnLayers;

		using Token = std::uint16_t;

		struct Decoder {

			using Word = std::string_view;
			using Words = std::vector<Word>;
			using WordMap = std::map<Word, Token>;

			Words mWords;
			WordMap mWordMap;//map words to their index

			static constexpr auto mDenseWordsSize = 321428;
			std::string mDenseWords;

			void readEnc();
			std::string decode(std::span<Token> tokens);

		} mDecoder;

		struct Data {

			std::vector<Token> mTokens;

			void readData();

		} mData;

	public:

		void readSafeTensors();

		void feedForward() {

			auto embedInput = [&]() {

				Tensor::TensorView wte, wpe, wActivations;
				Token token;

				for (std::size_t i = 0; i < mTestInputSize; ++i) { //inputSize vs dseq

					token = mData.mTokens[i];

					wte = mWteWeight.spanT(token);
					wpe = mWpeWeight.spanT(i);

					wActivations = mWActivations.spanT(i);

					for (std::size_t w = 0; w < mDModel; ++w)
						wActivations[w] = wte[w] + wpe[w];
				}


				};

			embedInput();

			Tensor* input = &mWActivations;
			std::for_each(mAttnLayers.begin(), mAttnLayers.end(), [&](auto& layer) {

				std::puts(".");
				input = &layer.forward(*input);

				});

			mFinalLayer.normalise(*input);

			auto checkSum64 = [&]() {

				assert(64 == mTestInputSize);

				auto getSum = [&](const auto& tensor) {
					return std::int64_t(std::reduce(tensor.mTensor.begin(), tensor.mTensor.end()));
					};

				assert(-30 == getSum(mWActivations));

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

					assert(16654 == getSum(mFinalLayer.mActivations));

					std::println("{}", getSum(mFinalLayer.mActivations));

					};


				testFrontLayer();
				testBackLayer();
				testOutput();

				};

			checkSum64();
		}
	};

}