#include "Gpt2.h"

#include <system_error>
#include <sstream>
#include <fstream>
#include <cassert>
#include <ranges>
#include <map>

#include <boost/json.hpp>

using namespace NetworkLib;

Parallel GPT2::AttnLayer::mParallelHeads( mHeadNum, mHeadNum);

TimeAverage<milliseconds> GPT2::LinearLayer::mBackwardTime
, GPT2::AttnLayer::mBackwardAttnTime
, GPT2::MLP::mBackwardGeluTime
, GPT2::mBackwardTime
, GPT2::Backward::mEmbedTime, GPT2::Backward::mUnembedTime, GPT2::Backward::mLayersTime;

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

void GPT2::Forward::setup() {

	mParallelInput.setup({}, mTestInputSize, 64);
	mParallelI.setup(Floats{}, mTestInputSize, 8);

	load();
}
void GPT2::Forward::load() {

	//the gpt2 model is found on huggingface website: https://huggingface.co/openai-community/gpt2?show_file_info=model.safetensors
	//we need to load gpt2 from file and create the activation space
	//the order of the activation space affects speed same as the gpt model space itself
	//so the layout follows forward process specifically
	//the gpt2 model is loaded from file - in a memory layout openai saved to file
	//but it could possibly be made faster still by moving from spanT to span?

	constexpr float floatSize = sizeof(float);

	auto readFile = [&]() {

		//the gpt file consists of a header segment, a json string, and a floatspace segment which is the remaining of the file
		auto fileName = std::format("{}model.safeTensors", mFilePath);

		std::println("Reading file: {}", fileName);

		std::ifstream fin(fileName, std::ios::in | std::ios::binary);

		if (!fin)
			Error::fileNotFound(fileName);

		std::uint64_t headerSize;
		fin.read(reinterpret_cast<char*>(&headerSize), sizeof(headerSize));

		std::string header; header.resize(headerSize);
		fin.read(header.data(), header.size());

		std::streampos current = fin.tellg()
			, end = fin.seekg(0, std::ios::end).tellg(); //get length of file
		fin.seekg(current);

		std::streamoff floatsSize = static_cast<std::streamoff>(end - current) / floatSize;
		mTensorSpace.resize(floatsSize);

		fin.read(reinterpret_cast<char*>(mTensorSpace.data()), floatsSize * floatSize);

		constexpr std::size_t knownFileFloatSize = 548090880 / floatSize;

		assert(knownFileFloatSize == floatsSize);

		fin.close();
		std::puts("file read done");

		return header;
		};

	auto header = readFile();

	boost::json::value j = boost::json::parse(header);
	std::size_t floatsUsed = 0;

	mActivationSpace.resize(mSeqModel + (mSeqModel * 7 + mSeqModel3 + mSeqSeqHead * 2 + mSeqModel4 * 3) * mAttnLayers.size() + mSeqModel + mSeqVocab * 2);
	auto activationSpace = mActivationSpace.begin();

	auto readTensorByName = [&](std::string_view name) {

		auto& obj = j.at(name);
		auto& offsets = obj.at("data_offsets").as_array();
		auto a = offsets.front().as_int64() / floatSize, b = offsets.back().as_int64() / floatSize;

		auto start = std::next(mTensorSpace.begin(), a);
		auto end = std::next(mTensorSpace.begin(), b);
		auto size = std::distance(start, end);

		Tensor::View tensorView(start, size);

		floatsUsed += size;

		auto& shape = obj.at("shape").as_array();
		auto dimensions = shape.size();

		auto getDim = [&](auto i) {
			return std::size_t(shape[i].as_int64());
			};

		Tensor tensor;

		auto expectedSize = 0ULL;
		switch (dimensions) {
		case 1:
			tensor = { tensorView, getDim(0) };
			expectedSize = tensor.size();
			break;
		case 2:
			tensor = { tensorView, getDim(0), getDim(1) };
			expectedSize = tensor.size2D();
			break;
		case 3:
			tensor = { tensorView, getDim(0), getDim(1), getDim(2) };
			expectedSize = tensor.size3D();
			break;
		case 4:
			tensor = { tensorView, getDim(0), getDim(1), getDim(2), getDim(3) };
			expectedSize = tensor.size4D();
			break;
		}
		assert(expectedSize == size);

		return tensor;
		};

	mWpeWeight = readTensorByName("wpe.weight");
	mWteWeight = readTensorByName("wte.weight");

	mWActivations = { activationSpace, mDSeq, mDModel };

	for (auto i : std::views::iota( 0ULL, mAttnLayers.size()))
		mAttnLayers[i].load(readTensorByName, i, activationSpace);

	mFinalLayer.load(readTensorByName("ln_f.bias"), readTensorByName("ln_f.weight"), activationSpace);

	mUnembedActivations = {activationSpace, mDSeq, mDVocab };
	mUnembedActivationsSoftmax = { activationSpace, mDSeq, mDVocab };

	assert(floatsUsed == mTensorSpace.size());
	assert(activationSpace == mActivationSpace.end());

	std::puts("Tensors read successfully");
}
void GPT2::forward(std::size_t i, const Tensor& inputTensor, Tensor& outputTensor, const Tensor& weightTensor, const Tensor& biasTensor, Parallel& parallel) {
	
	//this is a "matrix * vector + vector" or "fully connected" aka "forward" o += w * i + b

	Tensor::ConstView input = inputTensor.constView(i)
		, b = biasTensor.constView();
	Tensor::View output = outputTensor.view(i);

	//a fully connected input and output with a bias
	//identical to forward except for paralleled for i sample

	std::copy(b.begin(), b.end(), output.begin());

	parallel([&](Parallel::Section& section) {

		auto& outputs = std::any_cast<Floats&>(section.mAny);
		outputs.clear();
		outputs.resize(output.size(), 0.0f);

		for (auto m : section.mIotaView) {

			const auto& in = input[m];

			for (const auto& [o, w] : std::views::zip(outputs, weightTensor.constView(m)))
				o += w * in;
		}

		}, [&](Parallel::Section& section) {

			auto& outputs = std::any_cast<Floats&>(section.mAny);

			for (auto m : std::views::iota(0ULL, output.size()))
				output[m] += outputs[m];
				
			});
}
void GPT2::forward(const Tensor& inputTensor, Tensor& outputTensor, const Tensor& weightTensor, const Tensor& biasTensor, Parallel& parallel) {
	
	//each input is doing "matrix * vector + vector" or is "fully connected"

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
			}
		}

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

			IotaView activationsIotaView = std::views::iota(0ULL, inActivations.mY);

			for (auto i : section.mIotaView) {

				dOutput = dOutputs.constView(i);

				for (const auto& [b, o] : std::views::zip(pdBias, dOutput))
					b += o;

				activations = inActivations.constView(i);
				dActivations = outActivations.view(i);

				for (auto m : activationsIotaView) {

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
void GPT2::softmaxBack(const IotaView& iotaView, Tensor::ConstView input, Tensor::ConstView output, Tensor::View dSoftmax) {

	float softmaxSum = std::reduce(iotaView.begin(), iotaView.end(), 0.0f, [&](auto sum, auto m) {
		return sum + input[m] * output[m];
		});

	for (auto m : iotaView)
		dSoftmax[m] = input[m] * (output[m] - softmaxSum);

}
void GPT2::sgd(Tensor::View weights, Tensor::ConstView gradients, float learnRate) {

	std::transform(std::execution::seq, weights.begin(), weights.end(), gradients.begin(), weights.begin(),
		[&](auto& w, auto& g) {
			return w - g * learnRate;
		});
}

void GPT2::MLP::forward(const Tensor& input, Parallel& parallel) {

	GPT2::forward(input, mCFCActivations, mCFCWeight, mCFCBias, parallel);

	//an activation function, gelu, is applied here and cdf is cached

	auto gelu = [&]() {
		parallel([&](Parallel::Section& section) {

			for (auto i : section.mIotaView) {

				auto cfcActivations = mCFCActivations.view(i);
				auto gelu = mGeluActivations.view(i);
				auto cdf = mGeluCDF.view(i);

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

	mParallelHeads([&](Parallel::Section& section) {

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

	multiHeadedAttn(m);
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
void GPT2::Forward::embedInput(std::size_t i, Token token) {
	
	Tensor::View wte, wpe, wActivations;
	//weight token embed, weight position embed
	//generates an activation combining token and position

	wte = mWteWeight.view(token);
	wpe = mWpeWeight.view(i);
	wActivations = mWActivations.view(i);

	for (const auto& [a, t, p] : std::views::zip(wActivations, wte, wpe))
		a = t + p;
}
void GPT2::Forward::embedInputs(TokensView tokens) {

	//for each token, generate a position+token embedding
	//as a setup step before forward

	mParallelInput([&](auto& section) {

		for (auto i : section.mIotaView)
			embedInput(i, tokens[i]);

		});
}
void GPT2::Forward::unEmbedOutput(std::size_t i ) {

	//after forward, generate the probability of a specific token

	Tensor::View input, wte, output;
	//weight token embed seen in embedInput
	//input is the output of earlier forward process

	input = mFinalLayer.getActivations().view(i);
	output = mUnembedActivations.view(i);

	mParallelI.section(mUnembedActivations.mY);
	mParallelI([&](Parallel::Section& section) {

		for (auto m : section.mIotaView) {

			wte = mWteWeight.view(m);

			float sum = 0.0f;

			for (const auto& [in, w] : std::views::zip(input, wte))
				sum += in * w;

			output[m] = sum;
		}

		});
}
void GPT2::Forward::unEmbedOutputs() {

	//after forward, generate each token probability

	mParallelInput([&](auto& section) {

		Tensor::ConstView input, wte;
		Tensor::View output;

		for (auto i : section.mIotaView) {

			//weight token embed seen in embedInput
			//input is the output of earlier forward process

			input = mFinalLayer.getActivations().view(i);
			output = mUnembedActivations.view(i);

			for (auto m : std::views::iota(0ULL, output.size())) {

				wte = mWteWeight.view(m);

				float dot = 0.0f;

				for (const auto& [in, w] : std::views::zip(input, wte))
					dot += in * w;

				output[m] = dot;
			}
		}

		});
}
void GPT2::setup() {

	//we need to load our gpt data from file and also load the token translator file

	mForward.setup();

	mTranslator.load();
}
GPT2::Token GPT2::Forward::feedForward(TokensView tokens) {
	
	//feedForward will feed all tokens fresh into the network, up to dseq number of tokens
	//tokens max size == mTestInputSize
	//if larger than maxsize, should scroll to tail-mTestInputSize, it is assumed you did this earlier
	
	//the parallel process operates over all tokens simultaneously
	//for each input, there is large matrix and vector work, and this is futher parallelised
	//the best case performance is with fewer tokens, such as a short english sentence
	//and worst case performace when tokens size = mDSeq, the maximum model size

	mParallelInput.section(tokens.size());
	std::size_t m = tokens.size() - 1;

	embedInputs(tokens);

	Tensor* input = &mWActivations;

	for (auto& layer : mAttnLayers)
		input = &layer.forward(*input, mParallelInput);

	mFinalLayer.normalise(*input, mParallelInput);

	unEmbedOutputs();

	Token predicted = getPrediction(m);

	return predicted;
}
GPT2::Token GPT2::Forward::feedMore(TokensView tokens) {

	//because many short english sentences are small, they are way smaller than the maximum model size of mDSeq
	//and in fact predictions fit inside basically "unallocated" "more" model space, so making predictions is very fast
	//until the model space is filled, at which feedMore becomes unfunctional
	//the model has run out of "more" prediction space. 
	//At this point you need to make external decisions about your input data, such as scrolling it to create "more" space.
	//feedMore acts like all previous tokens are valid, and the back token, needs processed only
	//identical to feedForward except for parallel processing which is instead oriented toward a single sample

	int i = tokens.size() - 1;
	embedInput(i, tokens.back());

	Tensor* input = &mWActivations;

	for (auto& layer : mAttnLayers)
		input = &layer.forward(i, *input, mParallelI);

	mFinalLayer.normalise(i, *input);

	unEmbedOutput(i);

	Token predicted = getPrediction(i);

	return predicted;
}
float GPT2::Forward::crossEntropyLoss(TokensView nextTokens) {

	mParallelInput.section(nextTokens.size());

	mParallelInput([&](auto& section) {
		
		Tensor::View unembed, unembedSoftmax;

		for (auto i : section.mIotaView) {

			unembed = mUnembedActivations.view(i);
			unembedSoftmax = mUnembedActivationsSoftmax.view(i);

			GPT2::softmax(mUnembedActivations.mY - 1, unembed, unembedSoftmax);
		}

		});

	Tensor::View unembedSoftmax;
	float loss = 0.0f;
	for (auto i : std::views::iota(0ULL, nextTokens.size())) {

		Token expected = nextTokens[i];
		unembedSoftmax = mUnembedActivationsSoftmax.view(i);

		float expectedSoftmax = unembedSoftmax[expected];

		loss += -std::logf(expectedSoftmax);
	}

	loss /= nextTokens.size();

	return loss;
}
GPT2::Token GPT2::Forward::getPrediction(std::size_t i) const {

	//unembed activations is the entire sequence of all tokens, each a prediction of its probability 
	//the highest probability is the predicted token here, but other tokens may also have some lower possibility

	auto unembedActivations = mUnembedActivations.constView(i);
	auto selected = std::max_element(unembedActivations.begin(), unembedActivations.end());
	Token predicted = std::distance(unembedActivations.begin(), selected);

	return predicted;
}

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
void GPT2::Backward::sgd(float learnRate) {

	auto& forward = *mForward;
	auto& forwardLayers = forward.mAttnLayers;
	auto& layers = mAttnLayers;

	GPT2::sgd(forward.mWpeWeight.viewBlock(), mWpeWeight.viewBlock(), learnRate);
	GPT2::sgd(forward.mWteWeight.viewBlock(), mWteWeight.viewBlock(), learnRate);

	GPT2::sgd(forward.mFinalLayer.mBias.view(), mFinalLayer.mBias.view(), learnRate);
	GPT2::sgd(forward.mFinalLayer.mWeight.viewBlock(), mFinalLayer.mWeight.viewBlock(), learnRate);

	auto iotaView = std::ranges::iota_view(0ULL, mAttnLayers.size());

	std::for_each(std::execution::par, iotaView.begin(), iotaView.end(), [&](auto i) {

		auto& layer = forwardLayers[i];
		auto& gradient = layers[i];

		layer.sgd(gradient, learnRate);
		});

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
	Token newToken = 0;
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