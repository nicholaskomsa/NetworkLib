#include "Gpt2.h"

#include <Serializer.h>
#include <random>

using namespace NetworkLib;
using Diagnostics = GPT2::Diagnostics;

double Diagnostics::sumf(Tensor::ConstView tensorView, std::string_view expected) {

	double sum = std::reduce(tensorView.begin(), tensorView.end(), double(0.0));
	std::print("{}=={}\n", expected, sum);
	return sum;
}
double Diagnostics::sumf(const Tensor& tensor, std::string_view expected) {
	return sumf(tensor.mTensor, expected);
}
double Diagnostics::sumAbsf(Tensor::ConstView tensor, std::string_view expected) {
	double sum = std::reduce(tensor.begin(), tensor.end(), double(0.0), [](double a, float b) {return a + std::abs(b); });
	std::print("{}=={}\n", expected, sum);
	return sum;
}
double Diagnostics::sumAbsf(const Tensor& tensor, std::string_view expected) {
	return sumAbsf(tensor.mTensor, expected);
}
double Diagnostics::attnSumAbsf(const Tensor& tensor, std::size_t offset, std::string_view expected) {

	auto inputs = std::views::iota(0ULL, mTestInputSize);

	double fieldSum = std::reduce(inputs.begin(), inputs.end(), 0.0, [&](double sum, auto i) {

		auto tensorView = tensor.constView(i);

		for (auto h : std::views::iota(0ULL, mHeadNum)) {

			auto headOffset = h * mHeadsPerDModel;

			auto fieldView = Tensor::constField(tensorView, headOffset + offset, mHeadsPerDModel);

			sum = std::reduce(fieldView.begin(), fieldView.end(), sum, [](double sum, float f) {return sum + std::abs(f); });
		}

		return sum;
		});

	std::print("{}=={}\n", expected, fieldSum);
	return fieldSum;
}

void Diagnostics::firstCitizenTest64() {

	//this test is used to check the specific values of feed forward for correctness

	auto test = [&](auto& gpt2, Token predicted) {
		//when concerning first citizen test data, mData,  this checksum tests the test size which is 64 tokens
		assert(64 == mTestInputSize);

		auto getSum = [&](const auto& tensor) {
			return std::int64_t(std::reduce(tensor.mTensor.begin(), tensor.mTensor.end(), double(0.0)));
			};

		auto& forward = gpt2.mForward;

		auto testEmbed = [&] {
			assert(-30 == getSum(forward.mWActivations));
			};

		auto testFrontLayer = [&]() {

			const auto& layer = forward.mAttnLayers.front();
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

			const auto& layer = forward.mAttnLayers.back();

			assert(-3859 == getSum(layer.mResidualActivation2));

			sumf(layer.mResidualActivation2, "-3859");
			};

		auto testOutput = [&]() {

			auto testSum = getSum(forward.mFinalLayer.getActivations());
			constexpr auto finalSum = 16654;

			//std::println("{} == {} is {}", finalSum, testSum, finalSum == testSum);

			assert(finalSum == testSum);
			sumf(forward.mFinalLayer.getActivations(), "16654");
			};


		auto testUnEmbed = [&]() {

			//inaccuracies/difference from reduce?
			auto sum = getSum(forward.mUnembedActivations);
			assert(-353845318 == sum);

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

	run([&](auto& gpt2) {
		auto& data = gpt2.mTestData;
		data.load();
		TokensView tokens = { data.mTokens.begin(), GPT2::mTestInputSize };

		Token predicted = gpt2.mForward.feedForward(tokens);

		test(gpt2, predicted);

		});
}
void Diagnostics::feedForwardSpeed1024() {

	//this test is used to examine feedforward speed

	run([&](auto& gpt2) {

		auto& data = gpt2.mTestData;
		data.load();
		TokensView dataView(data.mTokens.begin(), GPT2::mTestInputSize);
		auto preText = gpt2.mTranslator.decode(dataView);
		std::println("{}", preText);

		Token predicted;
		Tokens tokens(dataView.begin(), dataView.end());

		TimeAverage<milliseconds> ffAvg;

		for (auto i : std::views::iota(0, 200)) {

			auto elapsed = ffAvg.accumulateTime([&]() {
				predicted = gpt2.mForward.feedForward(tokens);
				});

			auto word = gpt2.mTranslator.decode(predicted);
			std::print("{}({}:{})", word, elapsed.count(), ffAvg.average());

			std::shift_left(tokens.begin(), tokens.end(), 1);
			tokens.back() = predicted;
		}

		});
}
void Diagnostics::crossEntropyTest64() {

	run([&](auto& gpt2) {

		auto& data = gpt2.mTestData;
		data.load();

		TokensView tokens(data.mTokens.begin(), GPT2::mTestInputSize)
			, nextTokens(data.mTokens.begin() + 1, GPT2::mTestInputSize);

		auto preText = gpt2.mTranslator.decode(tokens);
		std::println("{}", preText);

		Token predicted, expected = nextTokens.back();

		float crossEntropyLoss;

		TimeAverage<milliseconds> ffAvg;

		auto elapsed = ffAvg.accumulateTime([&]() {

			predicted = gpt2.mForward.feedForward(tokens);

			crossEntropyLoss = gpt2.mForward.crossEntropyLoss(nextTokens);

			});

		auto predictedWord = gpt2.mTranslator.decode(predicted);
		auto expectedWord = gpt2.mTranslator.decode(expected);

		std::println("{}=={}; Cross Entropy Loss: {} == 4.133143", predictedWord, expectedWord, crossEntropyLoss);

		});

}
void Diagnostics::backwardTest64() {

	run([&](auto& gpt2) {

		auto& data = gpt2.mTestData;
		data.load();

		TokensView tokens(data.mTokens.begin(), GPT2::mTestInputSize)
			, nextTokens(data.mTokens.begin() + 1, GPT2::mTestInputSize);

		auto preText = gpt2.mTranslator.decode(tokens);
		std::println("{}", preText);

		Token predicted, expected = nextTokens.back();

		float crossEntropyLoss;
		TimeAverage<milliseconds> ffAvg;

		auto& forward = gpt2.mForward;
		auto& backward = gpt2.mBackward;

		auto elapsed = ffAvg.accumulateTime([&]() {

			predicted = forward.feedForward(tokens);

			crossEntropyLoss = forward.crossEntropyLoss(nextTokens);

			});

		auto predictedWord = gpt2.mTranslator.decode(predicted);
		auto expectedWord = gpt2.mTranslator.decode(expected);

		std::println("{}=={}; Cross Entropy Loss: {} == 4.133143", predictedWord, expectedWord, crossEntropyLoss);

		backward.setup(&gpt2.mForward);

		backward.backward(tokens, nextTokens);

		std::println("results:");

		sumf(backward.mUnembed, "0.008");//re source 0008
		sumf(forward.mUnembedActivationsSoftmax, "64");
		sumf(backward.mFinalLayer.mActivations, "-0.0403");
		sumf(backward.mFinalLayer.mBias, "-0.0403");
		sumf(backward.mFinalLayer.mWeight, "-0.5371");

		auto& attnBack = backward.mAttnLayers.back();

		sumf(attnBack.getOutput(), "-1.0-e08 on debug");
		sumAbsf(attnBack.mMLP.mCProjBias, "0.4879f");
		sumAbsf(attnBack.mMLP.mCProjWeight, "348");
		sumAbsf(attnBack.mMLP.mGeluActivations, "58.9");
		sumAbsf(attnBack.mMLP.mCFCActivations, "14.5");
		sumAbsf(attnBack.mL2.mActivations, "54.6");
		sumAbsf(attnBack.mMLP.mCFCWeight, "523.4");
		sumAbsf(attnBack.mMLP.mCFCBias, "3.66");
		sumAbsf(attnBack.mL2.mWeight, "5.93");
		sumAbsf(attnBack.mL2.mBias, "11.73");
		sumAbsf(attnBack.mResidualActivation1, "3.26");
		sumAbsf(attnBack.mAttnZ, "10.85");
		attnSumAbsf(attnBack.mCAttnActivations, mVOffset, "3.116");
		attnSumAbsf(attnBack.mCAttnActivations, mKOffset, "--");
		attnSumAbsf(attnBack.mCAttnActivations, mQOffset, "--");
		sumAbsf(attnBack.mCAttnActivations, "6.96");
		sumAbsf(attnBack.mCAttnWeight, "297");
		sumAbsf(attnBack.mCAttnBias, "2.53");
		sumAbsf(attnBack.mL1.mActivations, "24.27");
		sumAbsf(attnBack.mL1.mWeight, "2.54");
		sumAbsf(attnBack.mL1.mBias, "8.4");
		sumAbsf(attnBack.mResidualActivation1Out, "1.06");


		auto& attnPrev = *(backward.mAttnLayers.rbegin() + 1);
		sumAbsf(attnPrev.getOutput(), "3.6");

		sumAbsf(backward.mEmbed, "262.7");
		sumAbsf(backward.mWteWeight, "922");
		sumAbsf(backward.mWpeWeight, "262");
		});

}
void Diagnostics::SGDTest() {

	run([&](auto& gpt2) {

		auto& data = gpt2.mTestData;
		data.load();

		TokensView tokens(data.mTokens.begin(), GPT2::mTestInputSize)
			, nextTokens(data.mTokens.begin() + 1, GPT2::mTestInputSize);

		auto& translator = gpt2.mTranslator;
		auto preText = translator.decode(tokens);
		std::println("{}", preText);

		Token predicted = InvalidToken, expected = nextTokens.back();

		float crossEntropyLoss;

		auto& forward = gpt2.mForward;
		auto& backward = gpt2.mBackward;

		backward.setup(&gpt2.mForward);

		std::size_t generation = 0;

		auto print = [&]() {

			auto predictedWord = translator.decode(predicted);
			auto expectedWord = translator.decode(expected);
			auto equality = (predicted == expected ? "==" : "!=");

			std::println("{}; ce: {}, {}{}{}; {}; {}"
				, generation, crossEntropyLoss, predictedWord, equality, expectedWord
				, forward.getTimeAverages()
				, backward.getTimeAverages());
			};

		do{

			predicted = forward.feedForward(tokens);

			if (predicted == expected){
				print();
				break;
			}

			crossEntropyLoss = forward.crossEntropyLoss(nextTokens);

			backward.backward(tokens, nextTokens);

			backward.sgd();
			
			print();

			++generation;

		} while (true);

		});
}

void Diagnostics::serializeTest() {
	run([&](auto& gpt2) {

		auto& translator = gpt2.mTranslator;
		auto& data = gpt2.mTestData;
		data.load();
		TokensView completeTrainTokens = data.mTokens;
		auto& forward = gpt2.mForward;
		auto& backward = gpt2.mBackward;
		backward.setup(&gpt2.mForward);

		Serializer serializer;
		auto setupSerializer = [&]() {
			//only record a frameRect vs full chatgpt2
			Tensor::View tensorView = forward.getTensorSpace();
			const auto [w, h] = FloatSpaceConvert::getDimensions(tensorView.size());
			float x = 0, y = 0, scale = std::pow(2, 4);
			auto frameRect = FloatSpaceConvert::getFloatSubSpaceDimensions(x, y, scale, w, h);
			serializer.createOutputStream(tensorView, frameRect, w);
			};

		setupSerializer();

		Token predicted, expected;
		float crossEntropyLoss;
		std::size_t generation = 0;

		auto print = [&]() {

			auto predictedWord = translator.decode(predicted);
			auto expectedWord = translator.decode(expected);
			auto equality = (predicted == expected ? "==" : "!=");

			std::println("{}; ce: {}, {}{}{}; {}; {}"
				, generation, crossEntropyLoss, predictedWord, equality, expectedWord
				, forward.getTimeAverages()
				, backward.getTimeAverages());
			};

		auto setTrainingTokens = [&]() {

			static std::mt19937 random;
			static std::size_t currentOffset = 0;

			auto begin = completeTrainTokens.begin();

			auto prediction = GPT2::mTestInputSize + 1;

			if (currentOffset + prediction < completeTrainTokens.size()){ 

				std::advance(begin, currentOffset);

				//slide random amount
				auto min = 1ULL;
				auto max = GPT2::mTestInputSize * .33f;
				std::uniform_int_distribution<std::size_t> slideDistance(min, max);

				currentOffset += slideDistance(random);

			}else {

				//adjust to allow training of last prediction which may overlap with earlier	
				currentOffset = completeTrainTokens.size() - prediction;

				std::advance(begin, currentOffset);

				currentOffset = 0;
			}

			TokensView trainTokens( begin, GPT2::mTestInputSize)
				, nextTokens( begin + 1, GPT2::mTestInputSize );

			return std::make_pair( trainTokens, nextTokens );
			};

		for( auto i : std::views::iota(0, 1000)){
			++generation;

			const auto [ trainTokens, nextTokens]  = setTrainingTokens();

			expected = nextTokens.back();
			predicted = forward.feedForward(trainTokens);

			crossEntropyLoss = forward.crossEntropyLoss(nextTokens);

			backward.backward(trainTokens, nextTokens);

			backward.sgd();
			serializer.writeToFile();


			print();
		}

		});
}
void Diagnostics::simpleChat() {

	run([&](auto& gpt2) {
		gpt2.chat();
		});
}
void Diagnostics::run(TestFunction&& test) {

	try {

		auto gpt2 = std::make_unique<GPT2>(); //gpt2 is large and offsourced to heap

		gpt2->setup();

		test(*gpt2);

	}
	catch (const GPT2::Error& e) {
		std::println(std::cerr, "{}", e.what());
	}
	catch (const std::exception& e) {
		std::println(std::cerr, "{}", e.what());
	}
	catch (...) {
		std::println(std::cerr, "Unknown error");
	}
}