#include "Gpt2.h"

#include <system_error>
#include <sstream>
#include <fstream>
#include <cassert>
#include <ranges>
#include <map>

#include "Algorithms.h"
#include <Gpt2Forward.cpp>
using namespace NetworkLib;

Parallel GPT2::AttnLayer::mParallelHeads( mHeadNum, mHeadNum);

TimeAverage<milliseconds> GPT2::LinearLayer::mBackwardTime
, GPT2::AttnLayer::mForwardAttnTime, GPT2::AttnLayer::mBackwardAttnTime
, GPT2::MLP::mBackwardGeluTime
, GPT2::mBackwardTime, GPT2::mForwardTime
, GPT2::Backward::mUnembedTime, GPT2::Backward::mLayersTime
, GPT2::Forward::mUnembedTime, GPT2::Forward::mLayersTime;

const float GPT2::AttnLayer::r_sqrtHeadsPerDModel = 1.0f / std::sqrtf(GPT2::mHeadsPerDModel);
const float GPT2::MLP::r_sqrt2Pi = 1.0f / std::sqrtf(2.0f * std::numbers::pi);

GPT2::Error::Error(std::errc code, const std::string& message) : std::system_error(int(code), std::generic_category(), message) {}

void GPT2::Error::fileNotFound(std::string_view fileName) {
	throw Error(std::errc::no_such_file_or_directory, std::format("File Not Found: {}", fileName));
}
void GPT2::Error::wordNotFound(std::string_view word) {
	throw Error(std::errc::invalid_argument, std::format("Translator: Word Not Found: {}", word));
}
void GPT2::Error::tokenNotFound(Token token) {
	throw Error(std::errc::invalid_argument, std::format("Translator: Token Not Found: {}", token));
}


void GPT2::TestData::load() {

	//this file loads the concerning first citizen test data set, the first 64 tokens is used by Diagnostics
	//this file is found at raffK project, "data"

	auto readFile = [&]() {

		//data file https://github.com/rkaehn/gpt-2/blob/main/assets/data
		auto fileName = std::format("{}data", mFilePath);

		std::println("Reading file: {}", fileName);

		std::ifstream fin(fileName, std::ios::in | std::ios::binary);

		if (!fin)
			Error::fileNotFound(fileName);

		std::streampos end = fin.seekg(0, std::ios::end).tellg();

		constexpr auto tokenSize = sizeof(Token);
		std::streamoff tokensSize = static_cast<std::streamoff>(end) / tokenSize;
		mTokens.resize(tokensSize);

		fin.seekg(0);
		fin.read(reinterpret_cast<char*>(mTokens.data()), tokensSize * tokenSize);

		fin.close();
		std::puts("file read done");

		};

	readFile();

	std::println(std::cout, "Data Tokens size: {}", mTokens.size());
}

void GPT2::forward(std::size_t i, const Tensor& inputTensor, Tensor& outputTensor, const Tensor& weightTensor, const Tensor& biasTensor, Parallel& parallel) {
	
	//this is a "matrix * vector + vector" or "fully connected" aka "forward" o += w * i + b

	Tensor::ConstView inputs = inputTensor.constView(i)
		, b = biasTensor.constView();
	Tensor::View outputs = outputTensor.view(i);

	//a fully connected input and output with a bias
	//identical to forward except for paralleled for i sample

	std::copy(b.begin(), b.end(), outputs.begin());

	Parallel parallel2(outputs.size(), 64);
	
	auto inputIota = std::views::iota(0ULL, parallel.mSize);

	auto weights = weightTensor.constView(i);

	parallel2([&](auto& section) {

		for (auto o : section.mIotaView)
			for (auto i : inputIota)
				outputs[o] += weights[o] * inputs[i];
		
		});
}
void GPT2::forward(const Tensor& inputTensor, Tensor& outputTensor, const Tensor& weightTensor, const Tensor& biasTensor, Parallel& parallel) {
	
	//each input is doing "matrix * vector + vector" or is "fully connected"

	mForwardTime.accumulateTime([&]() {

		parallel([&](Parallel::Section& section) {

			Tensor::ConstView input, b = biasTensor.constView();
			Tensor::View output;

			for (auto i : section.mIotaView) {

				//a fully connected input and output with a bias

				input = inputTensor.constView(i);
				output = outputTensor.view(i);

				std::copy(b.begin(), b.end(), output.begin());

				float in;
				Tensor::ConstView weights;

				for (auto m : std::views::iota(0ULL, input.size())) {

					in = input[m];
					weights = weightTensor.constView(m);

					for (const auto& [o, w] : std::views::zip(output, weights))
						o += w * in;

					//std::transform(std::execution::seq, output.begin(), output.end(), weights.begin(), output.begin()
					//	, [&](auto o, auto w) {

					//		return o + w * in;
					//	});
				}
			}

			});

		});
}
void GPT2::softmax(std::size_t i, Tensor::ConstView input, Tensor::View output) {

	const auto ibegin = input.begin(), iend = ibegin + 1 + i;
	const auto obegin = output.begin(), oend = obegin + 1 + i;

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
void GPT2::backward(const Tensor& dOutputs, const Tensor& weights, Tensor& dWeights, Tensor& dBias, const Tensor& inActivations, Tensor& outActivations, Parallel& parallel) {

	mBackwardTime.accumulateTime([&]() {

		auto inputIota = std::views::iota(0ULL, parallel.mSize);
		Parallel parallel2(dBias.size(), 64);

		auto dWeightsBlock = dWeights.viewBlock();
		auto dBiasView = dBias.view();

		auto biasIota = std::views::iota(0ULL, dBias.size());

		parallel2([&](auto& section) {
			
			for( auto b : section.mIotaView )
				for (auto i : inputIota) {

					auto dOutput = dOutputs.constView(i);

					dBiasView[b] += dOutput[b];
				}
			});

		parallel2.section(inActivations.mY, 128);

		parallel2([&](auto& section) {

			for (auto m : section.mIotaView) {

				auto weight = weights.constView(m);
				auto dWeight = dWeights.view(m);

				for (auto i : inputIota) {

					auto dOutput = dOutputs.constView(i);
					auto activations = inActivations.constView(i);
					auto dActivations = outActivations.view(i);
					float in = activations[m];

					float dot = 0.0f;

					for (const auto& [pdW, o, w] : std::views::zip(dWeight, dOutput, weight)) {
						pdW += in * o;
						dot += w * o;
					}

					dActivations[m] = dot;
				}
			}
			});

		});
}
void GPT2::softmaxBack(const IotaView& iotaView, Tensor::ConstView input, Tensor::ConstView output, Tensor::View dSoftmax) {

	float softmaxSum = std::reduce(iotaView.begin(), iotaView.end(), 0.0f, [&](auto sum, auto m) {
		return sum + input[m] * output[m];
		});

	for (auto m : iotaView)
		dSoftmax[m] = input[m] * (output[m] - softmaxSum);

}
void GPT2::sgd(Tensor::View weights, Tensor::ConstView gradients, float learnRate) {

	std::transform(std::execution::par_unseq, weights.begin(), weights.end(), gradients.begin(), weights.begin(),
		[&](auto& w, auto& g) {
			return w - g * learnRate;
		});
}

void GPT2::MLP::forward(const Tensor& input, Parallel& parallel) {

	GPT2::forward(input, mCFCActivations, mCFCWeight, mCFCBias, parallel);

	//an activation function, gelu, is applied here and cdf is cached

	auto gelu = [&]() {

		parallel([&](auto& section) {

			Tensor::ConstView cfcActivations;
			Tensor::View cdf, gelu;

			for (auto i : section.mIotaView) {

				cfcActivations = mCFCActivations.view(i);
				gelu = mGeluActivations.view(i);
				cdf = mGeluCDF.view(i);

				for (const auto& [cfc, gelu, cdf] : std::views::zip(cfcActivations, gelu, cdf)) {

					cdf = 0.5f * (1.0f + std::erff(cfc * r_sqrt2));
					gelu = cfc * cdf;
				}
			}

			});
		};
	gelu();

	GPT2::forward(mGeluActivations, mCProjActivations, mCProjWeight, mCProjBias, parallel);
}
void GPT2::MLP::forward(std::size_t i, const Tensor& input, Parallel& parallel) {

	//should be pre-sectioned earlier
	GPT2::forward(i, input, mCFCActivations, mCFCWeight, mCFCBias, parallel);

	Tensor::View mlpActivations = mCFCActivations.view(i), geluActivations = mGeluActivations.view(i);

	std::transform(std::execution::par_unseq, mlpActivations.begin(), mlpActivations.end(), geluActivations.begin(),
		[&](auto x) {
			return x * 0.5f * (1.0f + std::erff(x * r_sqrt2));
		});

	parallel.section(mGeluActivations.mY);
	GPT2::forward(i, mGeluActivations, mCProjActivations, mCProjWeight, mCProjBias, parallel);
}
const Tensor& GPT2::MLP::getCProjActivations() const {
	return mCProjActivations;
}
void GPT2::MLP::backward(const MLP& mlp, const Tensor& linear, const Tensor& dResidual, Tensor& dLinear, Parallel& parallel) {

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

void GPT2::MLP::sgd(const MLP& gradients, float learnRate) {

	GPT2::sgd(mCFCWeight.viewBlock(), gradients.mCFCWeight.constViewBlock(), learnRate);
	GPT2::sgd(mCFCBias.view(), gradients.mCFCBias.constView(), learnRate);

	GPT2::sgd(mCProjWeight.viewBlock(), gradients.mCProjWeight.constViewBlock(), learnRate);
	GPT2::sgd(mCProjBias.view(), gradients.mCProjBias.constView(), learnRate);
}

void GPT2::MLP::load(auto&& cfcBias, auto&& cfcWeight, auto&& cProjBias, auto&& cProjWeight, Floats::iterator& activationSpace) {

	mCFCBias = std::move(cfcBias);
	mCFCWeight = std::move(cfcWeight);
	mCProjBias = std::move(cProjBias);
	mCProjWeight = std::move(cProjWeight);
	mCFCActivations = { activationSpace, mDSeq, mDModel4 };
	mGeluActivations = { activationSpace, mDSeq, mDModel4 };
	mGeluCDF = { activationSpace, mDSeq, mDModel4 };
	mCProjActivations = { activationSpace, mDSeq, mDModel };
}
std::size_t GPT2::MLP::getBackwardSize() {
	return mModel4Model * 2 + mSeqModel4 * 2 + mDModel4 + mDModel;
}
void GPT2::MLP::load(Floats::iterator& backwardSpace) {

	mCProjWeight = { backwardSpace, mDModel4, mDModel };
	mCProjBias = { backwardSpace, mDModel };
	mCFCBias = { backwardSpace, mDModel4 };
	mCFCWeight = { backwardSpace, mDModel, mDModel4 };
	mGeluActivations = { backwardSpace, mDSeq, mDModel4 };
	mCFCActivations = { backwardSpace, mDSeq, mDModel4 };
}

Tensor& GPT2::LinearLayer::getActivations() {
	return mActivations;
}
std::size_t GPT2::LinearLayer::getBackwardSize() {
	return mSeqModel + mDModel * 2;
}
void GPT2::LinearLayer::load(Floats::iterator& backwardSpace) {

	mActivations = { backwardSpace, mDSeq, mDModel };
	mBias = { backwardSpace, mDModel };
	mWeight = { backwardSpace, mDModel };
}
void GPT2::LinearLayer::load(Tensor&& bias, Tensor&& weight, Floats::iterator& activationSpace) {

	mBias = std::move(bias);
	mWeight = std::move(weight);
	mActivations = { activationSpace, mDSeq, mDModel };

	mMean.resize(mTestInputSize);
	mRStdDev.resize(mTestInputSize);
}
void GPT2::LinearLayer::normalise(std::size_t m, const Tensor& input) {

	//normalise is a case of linear-forward 
	//where a normalisation preprocess of input followed by a linear connect
	//the connect is linear rather than full-connect (+=), between normalized input and output
	//o = norm * w + b

	Tensor::ConstView in = input.constView(m);
	Tensor::View out = mActivations.view(m);

	const auto mean = std::reduce(in.begin(), in.end()) / in.size();
	mMean[m] = mean;

	auto meanDiffSq = std::reduce(in.begin(), in.end(), 0.0f,
		[&](auto sum, auto x) {
			auto diff = x - mean;
			return sum + diff * diff;
		}) / in.size();

	const auto r_stdDev = 1.0f / std::sqrt(meanDiffSq);
	mRStdDev[m] = r_stdDev;

	Tensor::ConstView bias = mBias.constView(), weight = mWeight.constView();

	float norm = 0;
	for (const auto& [i, w, b, o] : std::views::zip(in, weight, bias, out)) {
		norm = (i - mean) * r_stdDev;
		o = norm * w + b;
	}
}
void GPT2::LinearLayer::normalise(const Tensor& input, Parallel& parallel) {

	parallel([&](auto& section) {

		for (auto m : section.mIotaView) 
			normalise(m, input);

		});
}
void GPT2::LinearLayer::backward(const LinearLayer& inputLayer, const Tensor& inputs, Tensor& dInputs, Parallel& parallel) {

	mBackwardTime.accumulateTime([&] {

		const Tensor& dActivations = mActivations;
		Tensor::View dBias = mBias.view()
			, dWeight = mWeight.view();

		Tensor::ConstView weight = inputLayer.mWeight.constView();
		const Floats& means = inputLayer.mMean
			, & rStdDevs = inputLayer.mRStdDev;

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

void GPT2::LinearLayer::sgd(const LinearLayer& gradients, float learnRate) {

	GPT2::sgd(mWeight.view(), gradients.mWeight.constView(), learnRate);
	GPT2::sgd(mBias.view(), gradients.mBias.constView(), learnRate);
}

void GPT2::AttnLayer::calculateQKAtten(std::size_t headOffset, std::size_t i, Tensor::View attnOut) {

	//q-head is taken from the q-tensor, Q is for query
	//k-head is taken from the k-tensor, K is for key
	//q and k are multiplied together and summed and scaled

	const auto qOffset = mQOffset + headOffset;
	Tensor::ConstView kh, qh = Tensor::constField( mCAttnActivations.view(i), qOffset, mHeadsPerDModel );

	const auto kOffset = mKOffset + headOffset;
	float dot;

	for (auto m : std::views::iota(0ULL, i+1)) {

		kh = Tensor::constField( mCAttnActivations.view(m), kOffset, mHeadsPerDModel );
		dot = 0.0f;

		for (const auto& [q, k] : std::views::zip(qh, kh))
			dot += q * k;

		attnOut[m] = dot * r_sqrtHeadsPerDModel;
	};
}
void GPT2::AttnLayer::calculateVAtten(std::size_t headOffset, std::size_t i, Tensor::ConstView attnOutSoftmax) {

	//z-head is taken from the z-tensor, Z is for Activation
	//v-head is taken from the v-tensor, V is for Value
	//for each word, accumulate the v-attention into the z-head
	//v-attention is based on attention-softmax and v-head

	Tensor::View zh = Tensor::field( mAttnZ.view(i), headOffset, mHeadsPerDModel );
	const auto vOffset = mVOffset + headOffset;
	Tensor::ConstView vh;
	float factor;

	for (auto m : std::views::iota(0ULL, i+1)) {

		vh = Tensor::constField(mCAttnActivations.view(m), vOffset, mHeadsPerDModel );
		factor = attnOutSoftmax[m];

		for (const auto& [z, v] : std::views::zip(zh, vh))
			z += v * factor;
	}
}

void GPT2::AttnLayer::backwardVAtten(const AttnLayer& attn, std::size_t headOffset, const IotaView& qIotaView, std::size_t i, Tensor::ConstView inputAttnOutSoftmax, Tensor::View outputAttnOutSoftmax) {

	const auto vOffset = mVOffset + headOffset;
	Tensor::ConstView vh, dzh = Tensor::constField(mAttnZ.view(i), headOffset, mHeadsPerDModel);
	Tensor::View dvh;
	float factor, dot;

	for (auto m : qIotaView) {

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
void GPT2::AttnLayer::backwardQKAtten(const AttnLayer& attn, std::size_t headOffset, const IotaView& qIotaView, std::size_t i, Tensor::ConstView attnActivations) {

	const auto qOffset = mQOffset + headOffset;
	Tensor::ConstView kh, qh = Tensor::constField(attn.mCAttnActivations.constView(i), qOffset, mHeadsPerDModel);
	Tensor::View dkh, dqh = Tensor::field(mCAttnActivations.view(i), qOffset, mHeadsPerDModel);

	float o;
	const auto kOffset = mKOffset + headOffset;

	for (auto m : qIotaView) {

		kh = Tensor::constField(attn.mCAttnActivations.constView(m), kOffset, mHeadsPerDModel);
		dkh = Tensor::field(mCAttnActivations.view(m), kOffset, mHeadsPerDModel);
		o = attnActivations[m] * r_sqrtHeadsPerDModel;

		for (const auto& [q, dq, k, dk] : std::views::zip(qh, dqh, kh, dkh)) {

			dq += o * k;
			dk += o * q;
		}
	}
}
void GPT2::AttnLayer::multiHeadedAttn(std::size_t m) {

	//attention is a "multi-headed" process, and the same "Attention" operations are repeated for each head.
	//on each head, over the series of input, perform an accumulative attention process
	//this process generates a sort of diagonal matrix where there are lengths of n, n+1, n+2, n+3 opposing zeros
	//for each head, there are three types of segments of data, Q=query, K=key, and V=value
	//all of these segments and heads are located adjacent to each other in a vector
	//and referenced using offsets mQOffset, mKOffset, mVOffset, and headOffset
	//q and k and v are generated by the previous forward

	mParallelHeads([&](auto& section) {

		for (auto h : section.mIotaView ) {

			const auto headOffset = h * mHeadsPerDModel;

			for(auto i : std::views::iota(0ULL, m + 1) ) {

				Tensor::View attnOut = mAttnActivations.view(h, i)
					, attnOutSoftmax = mAttnSoftmaxActivations.view(h, i);

				calculateQKAtten(headOffset, i, attnOut);
				GPT2::softmax(i, attnOut, attnOutSoftmax);
				calculateVAtten(headOffset, i, attnOutSoftmax);
			}
		}

		});
}
void GPT2::AttnLayer::attention(std::size_t m ) {

	Tensor::View z = mAttnZ.view(m);
	std::fill(z.begin(), z.end(), 0.0f);

	multiHeadedAttn(m);
}
void GPT2::AttnLayer::attention(Parallel& parallel) {

	auto m = parallel.mSize - 1;

	auto activations = mAttnZ.viewBlock(m);

	//activations z cleared here, attention generates mAttnZ (activations)
	std::fill(activations.begin(), activations.end(), 0.0f);

	mForwardAttnTime.accumulateTime([&]() {
		multiHeadedAttn(m);
		});
}
void GPT2::AttnLayer::multiHeadedAttnBack(AttnLayer& attn, Parallel& parallel) {


	auto inputIotaView = std::views::iota(0ULL, mTestInputSize);

	mParallelHeads([&](auto& section) {

		IotaView qIotaView;
		Tensor::ConstView inputAttnOutSoftmax;
		Tensor::View outputAttnOutSoftmax, attnActivations;

		for (auto h : section.mIotaView) {

			const auto headOffset = h * mHeadsPerDModel;

			for (auto i : inputIotaView) {

				inputAttnOutSoftmax = attn.mAttnSoftmaxActivations.constView(h, i);
				outputAttnOutSoftmax = mAttnSoftmaxActivations.view(h, i);
				attnActivations = mAttnActivations.view(h, i);

				qIotaView = std::views::iota(0ULL, i + 1);

				backwardVAtten(attn, headOffset, qIotaView, i, inputAttnOutSoftmax, outputAttnOutSoftmax);

				softmaxBack(qIotaView, inputAttnOutSoftmax, outputAttnOutSoftmax, attnActivations);

				backwardQKAtten(attn, headOffset, qIotaView, i, attnActivations);
			}
		}
		});
}
void GPT2::AttnLayer::residual(std::size_t i, const Tensor& inputTensor, const Tensor& projectionTensor, Tensor& residualTensor) {
	
	//a residual is the sum of the input and the projection

	for (const auto& [out, p, in] : std::views::zip(residualTensor.view(i), projectionTensor.constView(i), inputTensor.constView(i)))
		out = p + in;
}
void GPT2::AttnLayer::residual(const Tensor& inputTensor, const Tensor& projectionTensor, Tensor& residualTensor, Parallel& parallel) {

	parallel([&](auto& section) {

		for (auto i : section.mIotaView)
			residual(i, inputTensor, projectionTensor, residualTensor);

		});
}
void  GPT2::AttnLayer::residualBack(const Tensor& a, const Tensor& b, Tensor& outputTensor) {

	Tensor::ConstView aBlock = a.constViewBlock()
		, bBlock = b.constViewBlock();
	Tensor::View outputBlock = outputTensor.viewBlock();

	std::transform(std::execution::par_unseq, aBlock.begin(), aBlock.end(), bBlock.begin(), outputBlock.begin(), [](auto a, auto b) {return a + b; });
}
Tensor& GPT2::AttnLayer::forward(const Tensor& inputTensor, Parallel& parallel) {

	mL1.normalise(inputTensor, parallel);
	GPT2::forward(mL1.getActivations(), mCAttnActivations, mCAttnWeight, mCAttnBias, parallel);

	attention(parallel);

	GPT2::forward(mAttnZ, mCProjActivations, mCProjWeight, mCProjBias, parallel);

	residual(inputTensor, mCProjActivations, mResidualActivation1, parallel);

	mL2.normalise(mResidualActivation1, parallel);

	mMLP.forward(mL2.getActivations(), parallel);

	residual(mResidualActivation1, mMLP.getCProjActivations(), mResidualActivation2, parallel);

	return mResidualActivation2;
}
Tensor& GPT2::AttnLayer::forward(std::size_t i, const Tensor& inputTensor, Parallel& parallel) {

	//a specific attention layer is a series of "forward", normalise, residual, MLP, and "attention" process.
	//where forward is aka a "large matrix" or "fully connected" operation

	parallel.section(mDModel);

	mL1.normalise(i, inputTensor);
	GPT2::forward(i, mL1.getActivations(), mCAttnActivations, mCAttnWeight, mCAttnBias, parallel);

	attention(i);

	GPT2::forward(i, mAttnZ, mCProjActivations, mCProjWeight, mCProjBias, parallel);

	residual(i, inputTensor, mCProjActivations, mResidualActivation1);

	mL2.normalise(i, mResidualActivation1);

	mMLP.forward(i, mL2.getActivations(), parallel);

	residual(i, mResidualActivation1, mMLP.getCProjActivations(), mResidualActivation2);

	return mResidualActivation2;
}
Tensor& GPT2::AttnLayer::getOutput() { return mResidualActivation2; }
void GPT2::AttnLayer::backward(AttnLayer& attn, const Tensor& forwardResidual2, Tensor& residual2, Parallel& parallel) {

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

void GPT2::AttnLayer::sgd(const AttnLayer& gradients, float learnRate) {

	mL1.sgd(gradients.mL1, learnRate);
	mL2.sgd(gradients.mL2, learnRate);

	GPT2::sgd(mCAttnWeight.viewBlock(), gradients.mCAttnWeight.constViewBlock(), learnRate);
	GPT2::sgd(mCAttnBias.view(), gradients.mCAttnBias.constView(), learnRate);

	GPT2::sgd(mCProjWeight.viewBlock(), gradients.mCProjWeight.constViewBlock(), learnRate);
	GPT2::sgd(mCProjBias.view(), gradients.mCProjBias.constView(), learnRate);

	mMLP.sgd(gradients.mMLP, learnRate);
}
void GPT2::AttnLayer::load(ReadTensorFunctor&& readTensorByName, std::size_t layerIdx, Floats::iterator& activationSpace) {

	auto layer = std::format("h.{}.", layerIdx);

	auto attnTensor = [&](const auto& name) {
		return readTensorByName(std::format("{}attn.{}", layer, name));
		};

	mBias = attnTensor("bias");
	mCAttnBias = attnTensor("c_attn.bias");
	mCAttnWeight = attnTensor("c_attn.weight");
	mCProjBias = attnTensor("c_proj.bias");
	mCProjWeight = attnTensor("c_proj.weight");

	mCAttnActivations = { activationSpace, mDSeq, mDModel3 };
	mAttnActivations = { activationSpace, mDSeq, mDSeq, mHeadNum };

	mAttnSoftmaxActivations = { activationSpace, mDSeq, mDSeq, mHeadNum };
	mAttnZ = { activationSpace, mDSeq, mDModel };

	auto linearTensor = [&](auto idx, const auto& name) {
		return readTensorByName(std::format("{}ln_{}.{}", layer, idx, name));
		};

	mL1.load(linearTensor(1, "bias"), linearTensor(1, "weight"), activationSpace);

	mCProjActivations = { activationSpace, mDSeq, mDModel };
	mResidualActivation1 = { activationSpace, mDSeq, mDModel };

	mL2.load(linearTensor(2, "bias"), linearTensor(2, "weight"), activationSpace);

	auto mlpTensor = [&](const auto& name) {
		return readTensorByName(std::format("{}mlp.{}", layer, name));
		};

	mMLP.load(mlpTensor("c_fc.bias"), mlpTensor("c_fc.weight"), mlpTensor("c_proj.bias"), mlpTensor("c_proj.weight"), activationSpace);

	mResidualActivation2 = { activationSpace, mDSeq, mDModel };

}
std::size_t GPT2::AttnLayer::getBackwardSize() {

	return mSeqModel * 3
		+ LinearLayer::getBackwardSize() * 2
		+ MLP::getBackwardSize()
		+ mDModel3 + mModel3Model
		+ mDModel + mModelModel
		+ mSeqModel
		+ mSeqSeqHead * 2
		+ mSeqModel3;
}
void GPT2::AttnLayer::load(Floats::iterator& backwardSpace) {

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

void GPT2::setup() {

	//we need to load our gpt data from file and also load the token translator file

	mForward.setup();

	mTranslator.load();
}
void GPT2::chat() {

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
void GPT2::slide(Tokens& tokens, std::size_t distance) {

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
	Token newToken;
	TimeAverage<milliseconds> ffAvg, fmAvg;

	for (auto s : std::views::iota(0ULL, distance)) {

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
			std::print("{},", updated.average());
			};
		printAvgTime();

		scrolled = addToken(newToken);

		if (endOfSentence) break;
	}
}