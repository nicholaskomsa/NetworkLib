#include "Gpt2.h"

using namespace NetworkLib;

void GPT2::Backward::setup(Forward* forward) {

	mForward = forward;

	mBackwardSpace.resize(mSeqVocab + mVocabModel
		+ LinearLayer::getBackwardSize()
		+ AttnLayer::getBackwardSize() * mAttnLayersNum
		+ mSeqModel * 2);

	auto backwardSpace = mBackwardSpace.begin();

	mUnembed = { backwardSpace, mDSeq, mDVocab };
	mWteWeight = { backwardSpace, mDVocab, mDModel };
	mFinalLayer.load(backwardSpace);

	for (auto& attnLayer : mAttnLayers)
		attnLayer.load(backwardSpace);

	mEmbed = { backwardSpace, mDSeq, mDModel };
	mWpeWeight = { backwardSpace, mDSeq, mDModel };

	mParallelInput.setup(PartialBiasWeight{}, mTestInputSize, 128);
	mParallelUnembed.setup({}, mUnembed.mY, 128);
}
void GPT2::Backward::unEmbedOutputs(TokensView nextTokens) {

	auto& forward = *mForward;

	Tensor& forwardSoftmax = forward.mUnembedActivationsSoftmax;

	auto softmaxBlock = forwardSoftmax.constViewBlock();
	std::copy(softmaxBlock.begin(), softmaxBlock.end(), mUnembed.mTensor.begin());

	Token token;
	Tensor::View unembed;

	for (auto i : std::views::iota(0ULL, nextTokens.size())) {

		unembed = mUnembed.view(i);
		token = nextTokens[i];

		unembed[token] -= 1.0f;
	};

	const Tensor& inputs = forward.mFinalLayer.getActivations();
	Tensor& dInputs = mFinalLayer.getActivations();

	const Tensor& wte = forward.mWteWeight;
	Tensor& dWte = mWteWeight;

	const float r_tokens = 1.0f / nextTokens.size();

	IotaView outputsIotaView = std::views::iota(0ULL, mUnembed.mY);

	mParallelInput([&](auto& section) {

		for (auto i : section.mIotaView) {
		
			auto dInput = dInputs.view(i);
			auto output = mUnembed.constView(i);

			for (auto m : outputsIotaView) {
				
				float o = output[m];
				auto weight = wte.constView(m);

				for (const auto& [din, w] : std::views::zip(dInput, weight))
					din += o * w;
			}
		}
		});

	mParallelUnembed([&](auto& section) {

		for (auto m : section.mIotaView) {

			auto dWeight = dWte.view(m);

			for (auto i : std::views::iota(0ULL, mParallelInput.mSize)) {

				auto output = mUnembed.constView(i);

				auto input = inputs.constView(i);

				float o = output[m] * r_tokens;

				for (const auto& [dw, in] : std::views::zip(dWeight, input))
					dw += o * in;
			}
		}

		});

		Tensor::View dInputsBlock = dInputs.viewBlock();

		std::transform(std::execution::par_unseq, dInputsBlock.begin(), dInputsBlock.end(), dInputsBlock.begin(), [&](auto f) {return f * r_tokens; });
}
void GPT2::Backward::embedOutputs(TokensView tokens, Parallel& parallel) {

	//this is a race condition: "tokens[i]" do not execute in parallel

	Tensor::ConstView dout;
	Tensor::View wte, wpe;

	for (auto i : std::views::iota(0ULL, parallel.mSize)) {

		dout = mEmbed.constView(i);
		wte = mWteWeight.view(tokens[i]);
		wpe = mWpeWeight.view(i);

		for (const auto& [o, t, p] : std::views::zip(dout, wte, wpe)) {
			p += o;
			t += o;
		}
	}

}
void GPT2::Backward::backward(TokensView tokens, TokensView nextTokens) {

	auto& forward = *mForward;
	mParallelInput.section(tokens.size());

	std::fill(mBackwardSpace.begin(), mBackwardSpace.end(), 0.0f);

	mUnembedTime.accumulateTime([&]() {
		unEmbedOutputs(nextTokens);
		});

	auto& forwardLayers = forward.mAttnLayers;
	auto& layers = mAttnLayers;

	mFinalLayer.backward(forward.mFinalLayer, forwardLayers.back().getOutput()
		, layers.back().getOutput(), mParallelInput);

	mLayersTime.accumulateTime([&]() {

		for (auto l : std::views::iota(1ULL, layers.size()) | std::views::reverse) {

			Tensor& forwardOutput = forwardLayers[l - 1].getOutput()
				, & output = layers[l - 1].getOutput();

			AttnLayer& forwardLayer = forwardLayers[l]
				, & layer = layers[l];

			layer.backward(forwardLayer, forwardOutput, output, mParallelInput);
		}

		layers.front().backward(forwardLayers.front(), forward.mWActivations
			, mEmbed, mParallelInput);
		});

	
	embedOutputs(tokens, mParallelInput);
		
}
void GPT2::Backward::sgd(float learnRate) {

	auto& forward = *mForward;
	auto& forwardLayers = forward.mAttnLayers;
	auto& layers = mAttnLayers;

	GPT2::sgd(forward.mWpeWeight.viewBlock(), mWpeWeight.viewBlock(), learnRate);
	GPT2::sgd(forward.mWteWeight.viewBlock(), mWteWeight.viewBlock(), learnRate);

	forward.mFinalLayer.sgd(mFinalLayer, learnRate);

	auto iotaView = std::ranges::iota_view(0ULL, layers.size());

	std::for_each(std::execution::par, iotaView.begin(), iotaView.end(), [&](auto i) {

		auto& layer = forwardLayers[i];
		auto& gradient = layers[i];

		layer.sgd(gradient, learnRate);
		});

}
