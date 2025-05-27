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

	mParallelInput.setup(PartialBiasWeight{}, mTestInputSize, 32);
}
void GPT2::Backward::unEmbedOutputs(TokensView nextTokens, Parallel& parallel) {

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

		IotaView outputsIotaView = std::views::iota(0ULL, mUnembed.mY);

		for (auto i : section.mIotaView) {

			output = mUnembed.view(i);
			dInput = dInputs.view(i);
			input = inputs.view(i);

			for (auto m : outputsIotaView) {

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
		unEmbedOutputs(nextTokens, mParallelInput);
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

	GPT2::sgd(forward.mFinalLayer.mBias.view(), mFinalLayer.mBias.view(), learnRate);
	GPT2::sgd(forward.mFinalLayer.mWeight.viewBlock(), mFinalLayer.mWeight.viewBlock(), learnRate);

	auto iotaView = std::ranges::iota_view(0ULL, layers.size());

	std::for_each(std::execution::par, iotaView.begin(), iotaView.end(), [&](auto i) {

		auto& layer = forwardLayers[i];
		auto& gradient = layers[i];

		layer.sgd(gradient, learnRate);
		});

}
