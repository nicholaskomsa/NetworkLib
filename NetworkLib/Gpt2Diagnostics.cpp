#include "Gpt2.h"

using namespace NetworkLib;

double GPT2::Diagnostics::sumf(Tensor::ConstView tensorView, std::string_view expected) {

	double sum = std::reduce(tensorView.begin(), tensorView.end(), double(0.0));
	std::print("{}=={}\n", expected, sum);
	return sum;
}
double GPT2::Diagnostics::sumf(const Tensor& tensor, std::string_view expected) {
	return sumf(tensor.mTensor, expected);
}
double GPT2::Diagnostics::sumAbsf(Tensor::ConstView tensor, std::string_view expected) {
	double sum = std::reduce(tensor.begin(), tensor.end(), double(0.0), [](double a, float b) {return a + std::abs(b); });
	std::print("{}=={}\n", expected, sum);
	return sum;
}
double GPT2::Diagnostics::sumAbsf(const Tensor& tensor, std::string_view expected) {
	return sumAbsf(tensor.mTensor, expected);
}
double GPT2::Diagnostics::attnSumAbsf(const Tensor& tensor, std::size_t offset, std::string_view expected) {

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
void GPT2::Diagnostics::firstCitizenTest64() {

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

			};

		auto testOutput = [&]() {

			auto testSum = getSum(forward.mFinalLayer.getActivations());
			constexpr auto finalSum = 16654;

			//std::println("{} == {} is {}", finalSum, testSum, finalSum == testSum);

			assert(finalSum == testSum);

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

void GPT2::Diagnostics::feedForwardSpeed1024() {

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
void GPT2::Diagnostics::crossEntropyTest64() {

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
void GPT2::Diagnostics::backwardTest64() {

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
void GPT2::Diagnostics::SGDTest64() {

	run([&](auto& gpt2) {

		auto& data = gpt2.mTestData;
		data.load();

		TokensView tokens(data.mTokens.begin(), GPT2::mTestInputSize)
			, nextTokens(data.mTokens.begin() + 1, GPT2::mTestInputSize);

		auto preText = gpt2.mTranslator.decode(tokens);
		std::println("{}", preText);

		Token predicted, expected = nextTokens.back();

		float crossEntropyLoss;
		TimeAverage<milliseconds> ffAvg, bAvg, sgdAvg;

		auto& forward = gpt2.mForward;
		auto& backward = gpt2.mBackward;

		backward.setup(&gpt2.mForward);

		std::size_t generation = 0;
		do {

			auto forwardElapsed = ffAvg.accumulateTime([&]() {

				predicted = forward.feedForward(tokens);

				crossEntropyLoss = forward.crossEntropyLoss(nextTokens);

				});

			auto predictedWord = gpt2.mTranslator.decode(predicted);
			auto expectedWord = gpt2.mTranslator.decode(expected);

			auto backwardElapsed = bAvg.accumulateTime([&]() {

				backward.backward(tokens, nextTokens);

				});

			auto sgdElapsed = bAvg.accumulateTime([&]() {

				backward.sgd();

				});

			std::println("{}:\t cel: {}, predicted/expected: {}/{}; took: forward: {}, back: {}, sgd: {}"
				", battn: {}, bgelu : {}, blin : {}"
				", backward: {}"
				", bembed: {}, bunembed: {}, blayers: {}"

				, generation, crossEntropyLoss, predictedWord, expectedWord, forwardElapsed, backwardElapsed, sgdElapsed
				, AttnLayer::mBackwardAttnTime.getString()
				, MLP::mBackwardGeluTime.getString()
				, LinearLayer::mBackwardTime.getString()
				, mBackwardTime.getString()
				, Backward::mEmbedTime.getString()
				, Backward::mUnembedTime.getString()
				, Backward::mLayersTime.getString());

			++generation;

		} while (predicted != expected);

		});

}
void GPT2::Diagnostics::simpleChat() {

	run([&](auto& gpt2) {
		gpt2.chat();
		});
}

void GPT2::Diagnostics::run(TestFunction&& test) {

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