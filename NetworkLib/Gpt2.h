#pragma once

#include <array>
#include <iostream>
#include <print>
#include <fstream>
#include <boost/json.hpp>
#include <map>
#include <numbers>
#include <deque>
#include <random>
#include <sstream>
#include <ranges>

#include "Parallel.h";
#include "Tensor.h"

namespace NetworkLib {

	using namespace std::chrono;
	template<typename TimeType>
	TimeType time(const std::string& caption, auto&& functor) {

		if (caption.size()) std::println("timing {}", caption);

		auto start = high_resolution_clock::now();
		functor();
		auto end = high_resolution_clock::now();
		auto elapsed = duration_cast<TimeType>(end - start);

		if (caption.size()) std::println("\t{} took {}", caption, elapsed.count());

		return elapsed;
	}

	template<typename TimeType>
	struct TimeAverage {

		TimeType sum = TimeType(0);
		std::size_t count = 0;

		void accumulateTime(const std::string& caption, auto&& functor) {
			sum = sum + time<TimeType>(caption, std::move(functor));
			++count;
		}
		std::size_t average() const {
			return sum.count() / count;
		}
	};

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

	private:

		static void forward(std::size_t i, const Tensor& inputTensor, const Tensor& outputTensor, const Tensor& weightTensor, const Tensor& biasTensor, Parallel& parallel, bool single = false) {
			//this is a matrix multiply and add - "forward" o = w * i + b
			Tensor::TensorView input, output, b = biasTensor.span();

			input = inputTensor.spanT(i);
			output = outputTensor.spanT(i);

			std::copy(b.begin(), b.end(), output.begin());

			parallel([&](Parallel::SectionsView sections) {

				for (auto& section : sections) {
					if (section.mAny.has_value() == false)
						section.mAny = Floats(output.size(), 0.0f);

					auto& floats = std::any_cast<Floats&>(section.mAny);
					floats.clear();
					floats.resize(output.size(), 0.0f);
				}

			}, [&](Parallel::Section& section) {

				auto& outputs = std::any_cast<Floats&>(section.mAny);

				auto& [first, second] = section.mOffsets;

				for (std::size_t m = first; m < second; ++m) {

					const auto& in = input[m];

					for (const auto& [o, w] : std::views::zip(outputs, weightTensor.spanT(m)))
						o += w * in;
				}

			}, [&](Parallel::SectionsView sections) {

				for (auto& section : sections) {

					auto& sOutputs = std::any_cast<Floats&>(section.mAny);

					for (std::size_t m = 0; m < output.size(); ++m)
						output[m] += sOutputs[m];
				}
			}, single);
		}
		static void forward(const Tensor& inputTensor, const Tensor& outputTensor, const Tensor& weightTensor, const Tensor& biasTensor, Parallel& parallel) {
			
			parallel([&](Parallel::SectionsView sections) {

				for (auto& section : sections) {

					if (section.mAny.has_value() == false)
						section.mAny = Parallel();

					auto& parallel = std::any_cast<Parallel&>(section.mAny);
					parallel.section(inputTensor.mY, Parallel::mLargeHardwareThreads);
				}

				}, [&](auto& section) {

					auto& [first, second] = section.mOffsets;
					auto& parallel = std::any_cast<Parallel&>(section.mAny);

					for (std::size_t i = first; i < second; ++i)
						forward(i, inputTensor, outputTensor, weightTensor, biasTensor, parallel);

					});
		}

		struct MLP {
			Tensor mCFCBias, mCFCWeight, mCProjBias, mCProjWeight;
			Tensor mCFCActivations, mCProjActivations, mGeluActivations;

			static constexpr float r_sqrt2 = 1.0f / std::numbers::sqrt2;

			const Tensor& forward(const Tensor& input) {

				GPT2::forward(input, mCFCActivations, mCFCWeight, mCFCBias, mParallelInput);

				std::transform(std::execution::par_unseq, mCFCActivations.mTensor.begin(), mCFCActivations.spanT(mParallelInput.mSize-1).end(), mGeluActivations.mTensor.begin(),
					[&](auto x) {
						return x * 0.5f * (1.0f + std::erff(x * r_sqrt2));
					});

				GPT2::forward(mGeluActivations, mCProjActivations, mCProjWeight, mCProjBias, mParallelInput);

				return mCProjActivations;
			}
			const Tensor& forward(std::size_t i, const Tensor& input) {

				GPT2::forward(i, input, mCFCActivations, mCFCWeight, mCFCBias, mParallelI);

				Tensor::TensorView mlpActivations = mCFCActivations.spanT(i), geluActivations = mGeluActivations.spanT(i);

				std::transform(std::execution::par_unseq, mlpActivations.begin(), mlpActivations.end(), geluActivations.begin(),
					[&](auto x) {
						return x * 0.5f * (1.0f + std::erff(x * r_sqrt2));
					});

				mParallelI.section(mDModel4);
				GPT2::forward(i, mGeluActivations, mCProjActivations, mCProjWeight, mCProjBias, mParallelI);

				return mCProjActivations;
			}


			void load(auto&& cfcBias, auto&& cfcWeight, auto&& cProjBias, auto&& cProjWeight, Floats::iterator& begin) {

				mCFCBias = std::move(cfcBias);
				mCFCWeight = std::move(cfcWeight);
				mCProjBias = std::move(cProjBias);
				mCProjWeight = std::move(cProjWeight);
			
				mCFCActivations = { {begin, mSeqModel4}, mDSeq, mDModel4 };
				std::advance(begin, mSeqModel4);

				mGeluActivations = { {begin, mSeqModel4}, mDSeq, mDModel4 };
				std::advance(begin, mSeqModel4);

				mCProjActivations = { {begin, mSeqModel}, mDSeq, mDModel };
				std::advance(begin, mSeqModel);
			}
		};
		struct LinearLayer {

			Tensor mBias, mWeight;
			Tensor mActivations;

			void load(Tensor&& bias, Tensor&& weight, Floats::iterator& begin){
				mBias = std::move(bias);
				mWeight = std::move(weight);
				mActivations = { {begin, mSeqModel}, mDSeq, mDModel };
				std::advance(begin, mSeqModel);
			}

			void normalise( std::size_t m, auto& input) {

				Tensor::TensorView in = input.spanT(m), out = mActivations.spanT(m);

				const auto mean = std::reduce(in.begin(), in.end()) / in.size();

				auto meanDiffSq = std::reduce(in.begin(), in.end(), 0.0f,
					[&](auto sum, auto x) {
						auto diff = x - mean;
						return sum + diff * diff;
					}) / in.size();

				auto r_stdDev = 1.0f / std::sqrt(meanDiffSq);

				Tensor::TensorView bias = mBias.span(), weight = mWeight.span();

				float norm = 0;
				for (const auto& [i, w, b, o] : std::views::zip(in, weight, bias, out)) {
					norm = (i - mean) * r_stdDev;
					o = norm * w + b;
				}
			}
			void normalise(auto& input) {

				mParallelInput([&](auto& sections) {

					auto& [first, second] = sections.mOffsets;

					for (std::size_t m = first; m < second; ++m)
						normalise(m, input);

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

			void calculateQKAtten(std::size_t headOffset, auto i, Tensor::TensorView attnOut) {

				const auto qOffset = mQOffset + headOffset;
				Tensor::TensorView qh = { mCAttnActivations.spanT(i).data() + qOffset, mHeadsPerDModel };

				const auto kOffset = mKOffset + headOffset;
				
				for (std::size_t m = 0; m <= i; ++m) {

					Tensor::TensorView kh = { mCAttnActivations.spanT(m).data() + kOffset, mHeadsPerDModel };
					float dot = 0.0f;

					for( const auto& [q, k] : std::views::zip(qh, kh))
						dot += q * k;

					attnOut[m] = dot * r_sqrtHeadsPerDModel;
				};
			}
			void softmax(std::size_t i, Tensor::TensorView input, Tensor::TensorView output) {

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
			}
			void calculateVAtten(std::size_t headOffset, std::size_t i, auto& attnOutSoftmax) {

				Tensor::TensorView zh = { mAttnZ.spanT(i).data() + headOffset, mHeadsPerDModel };
				const auto vOffset = mVOffset + headOffset;

				for (std::size_t m = 0; m <= i; ++m) {

					Tensor::TensorView vh = { mCAttnActivations.spanT(m).data() + vOffset, mHeadsPerDModel };
					float factor = attnOutSoftmax[m];

					for( const auto& [z, v] : std::views::zip(zh, vh))
						z += v * factor;
				}
			}
			
			void multiHeadedAttn(std::size_t m){

				mParallelHeads([&](auto& section) {

					auto& [first, second] = section.mOffsets;

					for (std::size_t h = first; h < second; ++h) {

						const auto headOffset = h * mHeadsPerDModel;

						for (std::size_t i = 0; i <= m; ++i) {

							Tensor::TensorView attnOut = mAttnActivations.spanT(h, i)
								, attnOutSoftmax = mAttnSoftmaxActivations.spanT(h, i);

							calculateQKAtten(headOffset, i, attnOut);
							softmax(i, attnOut, attnOutSoftmax);
							calculateVAtten(headOffset, i, attnOutSoftmax);
						}
					}

					});
			}

			void attention(std::size_t m) {

				Tensor::TensorView z = mAttnZ.spanT(m);
				std::fill(z.begin(), z.end(), 0.0f);

				multiHeadedAttn(m);
			}
			void attention() {

				auto m = mParallelInput.mSize - 1;

				//activations z cleared here
				std::fill(mAttnZ.mTensor.begin(), mAttnZ.spanT(m).end(), 0.0f);

				multiHeadedAttn(m);
			};

			void residual(std::size_t i, const Tensor& inputTensor, const auto& projectionTensor, const auto& residualTensor) {

				for( const auto& [out, p, in ] : std::views::zip(residualTensor.spanT(i), projectionTensor.spanT(i), inputTensor.spanT(i)))
					out = p + in;
			}
			void residual( const Tensor& inputTensor, const auto& projectionTensor, const auto& residualTensor) {

				mParallelInput([&](auto& sections) {

					auto& [first, second] = sections.mOffsets;

					for (std::size_t i = first; i < second; ++i)
						residual(i, inputTensor, projectionTensor, residualTensor);
					
					});
			}
		
			Tensor& forward(Tensor& inputTensor) {

				mL1.normalise(inputTensor);
				GPT2::forward(mL1.mActivations, mCAttnActivations, mCAttnWeight, mCAttnBias, mParallelInput);

				attention();

				GPT2::forward(mAttnZ, mCProjActivations, mCProjWeight, mCProjBias, mParallelInput);

				residual(inputTensor, mCProjActivations, mResidualActivation1);

				mL2.normalise(mResidualActivation1);

				mMLP.forward(mL2.mActivations);	

				residual(mResidualActivation1, mMLP.mCProjActivations, mResidualActivation2);

				return mResidualActivation2;
			}
			Tensor& forward(std::size_t i, const Tensor& inputTensor) {

				mParallelI.section(mDModel);

				mL1.normalise(i, inputTensor); 
				GPT2::forward(i, mL1.mActivations, mCAttnActivations, mCAttnWeight, mCAttnBias, mParallelI);

				attention(i);

				GPT2::forward(i, mAttnZ, mCProjActivations, mCProjWeight, mCProjBias, mParallelI);

				residual(i, inputTensor, mCProjActivations, mResidualActivation1);

				mL2.normalise(i, mResidualActivation1);

				mMLP.forward(i, mL2.mActivations);

				residual(i, mResidualActivation1, mMLP.mCProjActivations, mResidualActivation2);

				return mResidualActivation2;
			}
			

			void load(auto&& readTensorByName, std::size_t layerIdx, Floats::iterator& begin) {
				
				auto layer = std::format("h.{}.", layerIdx);

				auto attnTensor = [&](const auto& name) {
					return readTensorByName(std::format("{}attn.{}", layer, name));
					};

				mBias = attnTensor("bias");
				mCAttnBias = attnTensor("c_attn.bias");
				mCAttnWeight = attnTensor("c_attn.weight");
				mCProjBias = attnTensor("c_proj.bias");
				mCProjWeight = attnTensor("c_proj.weight");

				mCAttnActivations = { {begin, mSeqModel3}, mDSeq, mDModel3 };
				std::advance(begin, mSeqModel3);

				mAttnActivations = { {begin, mSeqSeqHead}, mDSeq, mDSeq, mHeadNum };
				std::advance(begin, mSeqSeqHead);

				mAttnSoftmaxActivations = { {begin, mSeqSeqHead}, mDSeq, mDSeq, mHeadNum };
				std::advance(begin, mSeqSeqHead);

				mAttnZ = { {begin, mSeqModel}, mDSeq, mDModel };
				std::advance(begin, mSeqModel);

				auto linearTensor = [&](auto idx, const auto& name) {
					return readTensorByName(std::format("{}ln_{}.{}", layer, idx, name));
					};

				mL1.load(linearTensor(1, "bias"), linearTensor(1, "weight"), begin);

				mCProjActivations = { {begin, mSeqModel}, mDSeq, mDModel };
				std::advance(begin, mSeqModel);

				mResidualActivation1 = { {begin, mSeqModel}, mDSeq, mDModel };
				std::advance(begin, mSeqModel);

				mL2.load(linearTensor(2, "bias"), linearTensor(2, "weight"), begin);

				auto mlpTensor = [&](const auto& name) {
					return readTensorByName(std::format("{}mlp.{}", layer, name));
					};

				mMLP.load(mlpTensor("c_fc.bias"), mlpTensor("c_fc.weight"), mlpTensor("c_proj.bias"), mlpTensor("c_proj.weight"), begin);

				mResidualActivation2 = { {begin, mSeqModel}, mDSeq, mDModel };
				std::advance(begin, mSeqModel);
			}
		};

		Floats mTensorSpace, mActivationSpace;

		Tensor mWpeWeight, mWteWeight, mWActivations, mUnembedActivations;
		LinearLayer mFinalLayer;
		std::array<AttnLayer, mAttnLayersNum> mAttnLayers;

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
			std::string decode( TokensView tokens);
			std::string decode( Token token);

			Tokens encode(std::string_view remaining);

		} mTranslator;

		struct Data {

			Tokens mTokens;

			void readData();

		} mData;

		void readSafeTensors();
		void embedInput(std::size_t i, Token token) {

			Tensor::TensorView wte, wpe, wActivations;

			wte = mWteWeight.spanT(token);
			wpe = mWpeWeight.spanT(i);

			wActivations = mWActivations.spanT(i);
			for (const auto& [a, t, p] : std::views::zip(wActivations, wte, wpe))
				a = t + p;
		}
		void embedInputs(TokensView tokens) {

			mParallelInput([&](auto& sections) {

				auto& [first, second] = sections.mOffsets;

				for (std::size_t i = first; i < second; ++i)
					embedInput(i, tokens[i]);

				});
		}
		void unEmbedOutput(std::size_t i) {

			Tensor::TensorView input, wte, output;

			input = mFinalLayer.mActivations.spanT(i);
			output = mUnembedActivations.spanT(i);

			for (std::size_t m = 0; m < output.size(); ++m) {

				wte = mWteWeight.spanT(m);

				float dot = 0.0f;

				for (const auto& [in, w] : std::views::zip(input, wte))
					dot += in * w;

				output[m] = dot;
			}
		}
		void unEmbedOutputs() {

			mParallelInput([&](auto& sections) {

				auto& [first, second] = sections.mOffsets;

				for (std::size_t i = first; i < second; ++i)
					unEmbedOutput(i);

				});
		}

	public:

		GPT2() = default;

		void setup() {

			readSafeTensors();
			//FloatSpaceConvert::colorizeFloatSpace("gpt2", mFloatSpace);

			mTranslator.readEnc();
		}
		Token getPrediction(std::size_t m){

			auto unembedActivations = mUnembedActivations.spanT(m);
			auto selected = std::max_element(unembedActivations.begin(), unembedActivations.end());
			Token predicted = std::distance(unembedActivations.begin(), selected);

			return predicted;
		}

		Token feedForward( TokensView tokens) {
			//feedForward will feed all tokens fresh into the network, up to dseq number of tokens
			//tokens max size == mTestInputSize
			//if larger than maxsize, should scroll to tail-mTestInputSize

			mParallelInput.section(tokens.size(), Parallel::mLargeHardwareThreads);

			embedInputs(tokens);

			Tensor* input = &mWActivations;
			std::for_each(mAttnLayers.begin(), mAttnLayers.end(), [&](auto& layer) {
				input = &layer.forward(*input);
				});

			mFinalLayer.normalise(*input);

			unEmbedOutputs();

			Token predicted = getPrediction(tokens.size() - 1);

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
		Token feedMore( TokensView tokens ) {
			//feedMore acts like all previous tokens are valid, and the back token, needs processed only
			mParallelInput.section(tokens.size(), Parallel::mLargeHardwareThreads);
			
			int i = tokens.size() - 1;
			embedInput( i, tokens.back());

			Tensor* input = &mWActivations;
			std::for_each(mAttnLayers.begin(), mAttnLayers.end(), [&](auto& layer) {
				input = &layer.forward(i, *input);
				});

			mFinalLayer.normalise(*input);

			unEmbedOutput(i);

			Token predicted = getPrediction(i);

			return predicted;
		}

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
					ffAvg.accumulateTime("", [&]() {
						newToken = feedForward(tokens);
						});
				else
					fmAvg.accumulateTime("", [&]() {
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