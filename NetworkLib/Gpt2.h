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

#include "Parallel.h";
#include "Tensor.h"

namespace NetworkLib {

	using namespace std::chrono;
	template<typename T>
	T time(const std::string& caption, auto&& functor) {

		if (caption.size()) std::println("timing {}", caption);

		auto start = std::chrono::high_resolution_clock::now();
		functor();
		auto end = std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<T>(end - start);

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

		static Parallel mParallelInput;
		static Parallel mParallelHeads;

		using Floats = std::vector<float>;

		struct MLP {
			Tensor mCFCBias, mCFCWeight, mCProjBias, mCProjWeight;
			Tensor mCFCActivations, mCProjActivations, mGeluActivations;
		};
		struct LinearLayer {

			Tensor mBias, mWeight;
			Tensor mActivations;

			void normalise( std::size_t m, auto& input) {

				Tensor::TensorView i, o, b, w;

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

			void normalise(auto& input, bool more = false) {

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
				Tensor::TensorView q = mCAttnActivations.spanT(i)
					, qh = { q.data() + qOffset, mHeadsPerDModel };

				const auto kOffset = mKOffset + headOffset;
				
				std::for_each(std::execution::seq, attnOut.begin(), attnOut.begin() + 1 + i, [&](auto& out) {
					
					std::size_t m = &out - attnOut.data();

					Tensor::TensorView kh = { mCAttnActivations.spanT(m).data() + kOffset, mHeadsPerDModel };

					float dot = 0.0f;

					for (std::size_t n = 0; n < qh.size(); ++n)
						dot += qh[n] * kh[n];

					out = dot * r_sqrtHeadsPerDModel;
					
					});
			}
			void softmax(std::size_t i, const auto& input, const auto& output) {

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

				Tensor::TensorView z = mAttnZ.spanT(i)
					, zh = { z.data() + headOffset, mHeadsPerDModel };

				const auto vOffset = mVOffset + headOffset;
					
				//mAttnZ zh has += in regards to parallel
				for (std::size_t m = 0; m <= i; ++m) {

					Tensor::TensorView v, vh;
					v = mCAttnActivations.spanT(m);
					vh = { v.data() + vOffset, mHeadsPerDModel };

					float factor = attnOutSoftmax[m];

					for (std::size_t n = 0; n < vh.size(); ++n)
						zh[n] += vh[n] * factor;
				}
			}
			
			void attention(std::size_t m) {

				Tensor::TensorView z = mAttnZ.spanT(m);
				std::fill(z.begin(),z.end(), 0.0f);

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
			void attention() {

				//activations z cleared here
				std::fill(mAttnZ.mTensor.begin(), mAttnZ.mTensor.end(), 0.0f);

				mParallelHeads([&](auto& section) {

					auto& [first, second] = section.mOffsets;

					for (std::size_t h = first; h < second; ++h) {

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

			void residual(std::size_t i, const Tensor& inputTensor, const auto& projectionTensor, const auto& residualTensor) {

				Tensor::TensorView p, input, o;

				p = projectionTensor.spanT(i);
				input = inputTensor.spanT(i);
				o = residualTensor.spanT(i);

				for (std::size_t m = 0; m < o.size(); ++m)
					o[m] = p[m] + input[m];
				
			}
			void residual( const Tensor& inputTensor, const auto& projectionTensor, const auto& residualTensor) {

				mParallelInput([&](auto& sections) {

					auto& [first, second] = sections.mOffsets;

					for (std::size_t i = first; i < second; ++i)
						residual(i, inputTensor, projectionTensor, residualTensor);
					
					});
			}
		
			void forward(std::size_t i, const auto& inputTensor, const auto& outputTensor, const auto& weightTensor, const auto& biasTensor, Parallel& parallel, bool single=false) {
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

					}, [&](auto& sections) {

					auto& outputs = std::any_cast<Floats&>(sections.mAny);

					auto& [first, second] = sections.mOffsets;

					for (std::size_t m = first; m < second; ++m) {

						const auto& in = input[m];
						auto w = weightTensor.spanT(m);
						auto o = outputs.begin();

						for (std::size_t n = 0; n < output.size(); ++n)
							o[n] += w[n] * in;
					}

					}, single );

				for (auto& section : parallel.mSections) {

					auto& sOutputs = std::any_cast<Floats&>(section.mAny);

					for(std::size_t m = 0; m < output.size(); ++m)
						output[m] += sOutputs[m];
				}
			}
			void forward(const auto& inputTensor, const auto& outputTensor, const auto& weightTensor, const auto& biasTensor) {
				
				mParallelInput([&](Parallel::SectionsView sections) {

					for (auto& section : sections) {

						if (section.mAny.has_value() == false)
							section.mAny = Parallel();

						auto& parallel = std::any_cast<Parallel&>(section.mAny);
						parallel.section(inputTensor.mY, Parallel::mLargeHardwareThreads);
					}

					}, [&](auto& sections) {

					auto& [first, second] = sections.mOffsets;
					auto& parallel = std::any_cast<Parallel&>(sections.mAny);

					for (std::size_t i = first; i < second; ++i)
						forward(i, inputTensor, outputTensor, weightTensor, biasTensor, parallel);

					});
			}

			Tensor& forward(Tensor& inputTensor) {

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
			Tensor& forward(std::size_t i, const Tensor& inputTensor, Parallel& parallel) {

				parallel.section(mDModel);

				mL1.normalise(i, inputTensor); 
				forward(i, mL1.mActivations, mCAttnActivations, mCAttnWeight, mCAttnBias, parallel);

				attention(i);

				forward(i, mAttnZ, mCProjActivations, mCProjWeight, mCProjBias, parallel);

				residual(i, inputTensor, mCProjActivations, mResidualActivation1);

				mL2.normalise(i, mResidualActivation1);
				forward(i, mL2.mActivations, mMLP.mCFCActivations, mMLP.mCFCWeight, mMLP.mCFCBias, parallel);

				constexpr auto r_sqrt2 = 1.0f / std::numbers::sqrt2;

				Tensor::TensorView mlpActivations = mMLP.mCFCActivations.spanT(i), geluActivations = mMLP.mGeluActivations.spanT(i);

				std::transform(std::execution::par_unseq, mlpActivations.begin(), mlpActivations.end(), geluActivations.begin(),
					[&](auto x) {
						return x * 0.5f * (1.0f + std::erff(x * r_sqrt2));
					});

				parallel.section(mDModel4);
				forward(i, mMLP.mGeluActivations, mMLP.mCProjActivations, mMLP.mCProjWeight, mMLP.mCProjBias, parallel);

				residual(i, mResidualActivation1, mMLP.mCProjActivations, mResidualActivation2);

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

			Tokens encode(const std::string& text) {

				std::string_view remaining(text);

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

		} mTranslator;

		struct Data {

			Tokens mTokens;

			void readData();

		} mData;

	public:

		GPT2() = default;

		void setup() {

			readSafeTensors();
			//FloatSpaceConvert::colorizeFloatSpace("gpt2", mFloatSpace);

			mTranslator.readEnc();
		}

		void readSafeTensors();

		//tokens max size == dseq
		void embedInput(std::size_t i, Token token) {

			Tensor::TensorView wte, wpe, wActivations;

			wte = mWteWeight.spanT(token);
			wpe = mWpeWeight.spanT(i);

			wActivations = mWActivations.spanT(i);

			for (std::size_t m = 0; m < wActivations.size(); ++m)
				wActivations[m] = wte[m] + wpe[m];
		}
		void embedInputs(TokensView tokens) {

			mParallelInput([&](auto& sections) {

				auto& [first, second] = sections.mOffsets;

				for (std::size_t i = first; i < second; ++i) 
					embedInput( i, tokens[i] );
				
				});
		}

		void unEmbedOutput(std::size_t i) {

			Tensor::TensorView input, wte, output;

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

		void unEmbedOutputs() {

			mParallelInput([&](auto& sections) {

				auto& [first, second] = sections.mOffsets;

				for (std::size_t i = first; i < second; ++i)
					unEmbedOutput(i);
		
				});
		}

		Token feedForward( TokensView tokens) {
			//feedForward will feed all tokens fresh into the network, up to dseq number of tokens
			//tokens max size == mTestInputSize
			//if larger than maxsize, should scroll to tail-mTestInputSize

			mParallelInput.section(tokens.size(), Parallel::mLargeHardwareThreads);

		//	time<std::chrono::milliseconds>("ff", [&]() {

				embedInputs(tokens);

				Tensor* input = &mWActivations;
				std::for_each(mAttnLayers.begin(), mAttnLayers.end(), [&](auto& layer) {

					//std::cout << ".";
					input = &layer.forward(*input);

					});

				mFinalLayer.normalise(*input);

				unEmbedOutputs();

			//	});

			auto getPrediction = [&]() {

				auto unembedActivations = mUnembedActivations.spanT(mParallelInput.mSize - 1);
				auto selected = std::max_element(unembedActivations.begin(), unembedActivations.end());
				Token predicted = std::distance(unembedActivations.begin(), selected);

				return predicted;
				};

			Token predicted = getPrediction();

			auto writeCompletion = [&]() {

				auto outputText = mTranslator.decode({ mData.mTokens.begin(), mData.mTokens.begin() + mParallelInput.mSize });
				std::println("{}{}", outputText, mTranslator.decode(predicted));
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
		Token feedMore( TokensView tokens ) {
			//feedMore acts like all previous tokens are valid, and the back token, needs processed only
			mParallelInput.section(tokens.size(), Parallel::mHardwareThreads);
			
			int i = tokens.size() - 1;
			embedInput( i, tokens.back());

			Parallel parallel;
			Tensor* input = &mWActivations;
			std::for_each(mAttnLayers.begin(), mAttnLayers.end(), [&](auto& layer) {

				input = &layer.forward(i, *input, parallel);
					
				});

			mFinalLayer.normalise(*input);

			unEmbedOutput(i);

			auto getPrediction = [&]() {

				auto unembedActivations = mUnembedActivations.spanT(tokens.size()-1);
				auto selected = std::max_element(unembedActivations.begin(), unembedActivations.end());
				Token predicted = std::distance(unembedActivations.begin(), selected);

				return predicted;
				};

			Token predicted = getPrediction();

			return predicted;
		}

		void chat() {

			bool chatting = true;
			Tokens scrollingTokens;
			const Token endl = mTranslator.mWordMap["\n"];
			std::string line;
			
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