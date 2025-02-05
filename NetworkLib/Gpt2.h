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
#include <numbers>
#include <deque>
#include <random>

#include "Tensor.h"

namespace NetworkLib {

	template<typename T>
	T time(const std::string& caption, auto&& functor) {
		std::println("timing {}", caption);
		auto start = std::chrono::high_resolution_clock::now();
		functor();
		auto end = std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<T>(end - start);
		std::println("\t{} took {}", caption, elapsed.count());
		
		return elapsed;
	};

	struct Parallel {

		static constexpr auto mHardwareThreads = 32;
		using Offsets = std::pair<std::size_t, std::size_t>;
		using Sections = std::vector<Offsets>;

		using Functor = std::function<void(Offsets&)>;

		Sections mSections;
		std::size_t mSize{ 0 };

		Parallel() = default;
		Parallel(std::size_t size, std::size_t hardwareSections = 0) {

			section(size, hardwareSections);
		};
		void section(std::size_t size, std::size_t hardwareSections = 0 ){
			mSize = size;

			auto sectionSize = size / (hardwareSections == 0 ? mHardwareThreads: hardwareSections);
			if (sectionSize == 0) sectionSize = 1;
			auto numSections = size / sectionSize;

			std::size_t start = 0, end = 0;

			mSections.resize(numSections);

			for (auto s = 0; s < numSections; ++s) {

				end = start + sectionSize;

				mSections[s] = { start, end };

				start = end;
			}
			mSections.back().second = size;
		}

		void operator()(Functor&& functor, bool single=false) {

			if(single)
				std::for_each(std::execution::seq, mSections.begin(), mSections.end(), [&](auto& section) {
					functor(section);
					});
			else
				std::for_each(std::execution::par_unseq, mSections.begin(), mSections.end(), [&](auto& section) {
					functor(section);
					});
		}
	};

	struct GPT2 {

		static constexpr auto mFilePath = "F:/software dev/programming2025/downloads/";
		static constexpr std::size_t mDVocab = 50257
			, mDModel = 768, mDModel3 = mDModel * 3, mDModel4 = mDModel * 4
			, mDSeq = 1024
			, mHeadNum = 12, mAttnLayersNum = 12
			, mHeadsPerDModel = mDModel / mHeadNum
			, mQOffset = 0, mKOffset = mDModel, mVOffset = mDModel * 2
			, mTestInputSize = mDSeq;	//vs dSeq for full size or 64 for test size

		struct Error : public std::system_error {

			Error(std::errc code, const std::string& message);

			static void fileNotFound(const std::string& fileName);
		};

		static Parallel mParallelInput, mParallelHeads;

		using Floats = std::vector<float>;

		struct MLP {
			Tensor mCFCBias, mCFCWeight, mCProjBias, mCProjWeight;
			Tensor mCFCActivations, mCProjActivations, mGeluActivations;
		};
		struct LinearLayer {

			Tensor mBias, mWeight;
			Tensor mActivations;

			void normalise(auto& input) {

				mParallelInput([&](auto& offsets) {

					Tensor::TensorView i, o, b, w;

					for (std::size_t m = offsets.first; m < offsets.second; ++m) { //inputSize vs dseq

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
					});
			}
		};
		struct AttnLayer {

			LinearLayer mL1, mL2;

			Tensor mBias, mCAttnBias, mCAttnWeight, mCProjBias, mCProjWeight;
			Tensor mCAttnActivations, mAttnActivations, mAttnSoftmaxActivations, mAttnZ, mCProjActivations;
			Tensor mResidualActivation1, mResidualActivation2;

			MLP mMLP;

			static const float r_sqrtHeadsPerDModel;

			void attention() {

				//activations z cleared here
				std::fill(mAttnZ.mTensor.begin(), mAttnZ.mTensor.end(), 0.0f);

				auto calculateQKAtten = [&](std::size_t headOffset, auto i, const auto& attnOut) {

					const auto qOffset = mQOffset + headOffset;
					Tensor::TensorView q = mCAttnActivations.spanT(i)
						, qh = { q.data() + qOffset, mHeadsPerDModel };

					const auto kOffset = mKOffset + headOffset;

					Tensor::TensorView k, kh;

					for( std::size_t m = 0; m <= i; ++m) {

						k = mCAttnActivations.spanT(m);
						kh = { k.data() + kOffset, mHeadsPerDModel };

						float dot = 0.0f;

						for (std::size_t n = 0; n < qh.size(); ++n)
							dot += qh[n] * kh[n];

						attnOut[m] =  dot * r_sqrtHeadsPerDModel;
						
					}

					};

				auto softmax = [&](std::size_t i, const auto& input, const auto& output) {
					
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

					};

				auto calculateVAtten = [&](std::size_t headOffset, std::size_t i, auto& attnOutSoftmax) {

					Tensor::TensorView z = mAttnZ.spanT(i)
						, zh = { z.data() + headOffset, mHeadsPerDModel };

					const auto vOffset = mVOffset + headOffset;

					Tensor::TensorView v, vh;
					auto factor = 0.0f;
		
					for( std::size_t m =0; m <= i; ++m){

						v = mCAttnActivations.spanT(m);
						vh = { v.data() + vOffset, mHeadsPerDModel };

						factor = attnOutSoftmax[m];

						for (std::size_t n = 0; n < vh.size(); ++n)
							zh[n] += vh[n] * factor;
					}

					};

				mParallelHeads([&](auto& offsets) {

					for (std::size_t h = offsets.first; h < offsets.second; ++h) {

						const auto headOffset = h * mHeadsPerDModel;

						for (std::size_t i = 0; i < mParallelInput.mSize; ++i) {
							
							Tensor::TensorView attnOut = mAttnActivations.spanT(h, i)
								, attnOutSoftmax = mAttnSoftmaxActivations.spanT(h, i);
							
							calculateQKAtten(headOffset, i, attnOut);
							softmax(i, attnOut, attnOutSoftmax);
							calculateVAtten(headOffset, i, attnOutSoftmax);
						}
					}
					});
			};
			void residual(const Tensor& inputTensor, const auto& projectionTensor, const auto& residualTensor) {

				mParallelInput([&](auto& offsets) {

					Tensor::TensorView p, input, o;

					for (std::size_t i = offsets.first; i < offsets.second; ++i) {

						p = projectionTensor.spanT(i);
						input = inputTensor.spanT(i);
						o = residualTensor.spanT(i);

						for (std::size_t m = 0; m < o.size(); ++m)
							o[m] = p[m] + input[m];
					}

					});
			}

			void forward(const auto& inputTensor, const auto& outputTensor, const auto& weightTensor, const auto& biasTensor) {

				const auto b = biasTensor.span();

				mParallelInput([&](auto& offsets) {

					Tensor::TensorView input, output, w;

					for (std::size_t i = offsets.first; i < offsets.second; ++i) {

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
					});
			}
			Tensor& forward(const auto& inputTensor) {

				mL1.normalise(inputTensor);
				forward(mL1.mActivations, mCAttnActivations, mCAttnWeight, mCAttnBias);

				attention();

				forward(mAttnZ, mCProjActivations, mCProjWeight, mCProjBias);

				residual(inputTensor, mCProjActivations, mResidualActivation1);

				mL2.normalise(mResidualActivation1);
				forward(mL2.mActivations, mMLP.mCFCActivations, mMLP.mCFCWeight, mMLP.mCFCBias);

				constexpr auto r_sqrt2 = 1.0f / std::numbers::sqrt2;

				std::transform(std::execution::par_unseq, mMLP.mCFCActivations.mTensor.begin(), mMLP.mCFCActivations.mTensor.end(), mMLP.mGeluActivations.mTensor.begin(),
					[&](auto x) {
						return x * 0.5f * (1.0f + std::erff(x * r_sqrt2));
					});

				forward(mMLP.mGeluActivations, mMLP.mCProjActivations, mMLP.mCProjWeight, mMLP.mCProjBias);

				residual(mResidualActivation1, mMLP.mCProjActivations, mResidualActivation2);

				return mResidualActivation2;
			}
		};

		Floats mTensorSpace, mActivationSpace;

		Tensor mWpeWeight, mWteWeight, mWActivations, mUnembedActivations;
		LinearLayer mFinalLayer;
		std::array<AttnLayer, mAttnLayersNum> mAttnLayers;

		using Token = std::uint16_t;
		using Tokens = std::vector<Token>;
		using TokensView = std::span<Token>;
		using TokenDeque = std::deque<Token>;

		struct Decoder {

			using Word = std::string_view;
			using Words = std::vector<Word>;
			using WordMap = std::map<Word, Token>;

			Words mWords;
			WordMap mWordMap;//map words to their index

			static constexpr auto mDenseWordsSize = 321428;
			std::string mDenseWords;

			void readEnc();
			std::string decode( TokensView tokens);
			std::string decode(Token token);

		} mDecoder;

		struct Data {

			Tokens mTokens;

			void readData();

		} mData;

	public:

		void readSafeTensors();

		Token feedForward(TokensView tokens) {

			//std::println("feedForward input size = {}", tokens.size());

			mParallelInput.section(tokens.size());

			auto embedInput = [&]() {

				mParallelInput([&](auto& offsets) {

					Tensor::TensorView wte, wpe, wActivations;
					Token token;

					for (std::size_t i = offsets.first; i < offsets.second; ++i) {

						token = tokens[i]; //mData.mTokens[i]

						wte = mWteWeight.spanT(token);
						wpe = mWpeWeight.spanT(i);

						wActivations = mWActivations.spanT(i);

						for (std::size_t m = 0; m < wActivations.size(); ++m)
							wActivations[m] = wte[m] + wpe[m];
					}
				});

				};

			auto unEmbedOutput = [&]() {

				mParallelInput([&](auto& offsets) {

					Tensor::TensorView input, wte, output;

					for (std::size_t i = offsets.first; i < offsets.second; ++i) { 

						input = mFinalLayer.mActivations.spanT(i);
						output = mUnembedActivations.spanT(i);

						for (std::size_t m = 0; m < output.size(); ++m) {
							
							wte = mWteWeight.spanT(m);

							float dot = 0.0f;

							for (std::size_t n = 0; n < input.size(); ++n)
								dot += input[n] * wte[n];

							output[m] = dot;
						}
					}
					});

				};

		//	time<std::chrono::milliseconds>("ff", [&]() {

				embedInput();

				Tensor* input = &mWActivations;
				std::for_each(mAttnLayers.begin(), mAttnLayers.end(), [&](auto& layer) {

					//std::cout << ".";
					input = &layer.forward(*input);

					});

				mFinalLayer.normalise(*input);

				unEmbedOutput();

			//	});

			auto getPrediction = [&]() {

				auto unembedActivations = mUnembedActivations.spanT(mParallelInput.mSize - 1);
				auto selected = std::max_element(unembedActivations.begin(), unembedActivations.end());
				Token predicted = std::distance(unembedActivations.begin(), selected);

				return predicted;
				};

			Token predicted = getPrediction();

			auto writeCompletion = [&]() {

				auto outputText = mDecoder.decode({ mData.mTokens.begin(), mData.mTokens.begin() + mParallelInput.mSize });
				std::println("{}{}", outputText, mDecoder.decode(predicted));
				};

			//writeCompletion();

			auto checkSum64 = [&]() {

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

					auto testSum = getSum(mFinalLayer.mActivations);
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

		void slide(TokensView tokens,  std::size_t distance = 20) {

			GPT2::Tokens testTokens(tokens.begin(), tokens.end());

			for (std::size_t s = 0; s < distance; ++s) {

				auto newToken = feedForward(testTokens);

				std::shift_left(testTokens.begin(), testTokens.end(), 1);

				testTokens.back() = newToken;

				std::print("{}", mDecoder.decode(testTokens));
			}
		}
	};
}