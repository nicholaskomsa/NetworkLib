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

		Error(std::errc code, const std::string& message) : std::system_error(int(code), std::generic_category(), message) {}

		static void fileNotFound(const auto& fileName) {
			throw Error(std::errc::no_such_file_or_directory, std::format("File Not Found: {}", fileName));
		}
	};

	using Floats = std::vector<float>;

	struct Tensor {

		using TensorView = std::span<float>;
		TensorView mTensor;

		std::size_t mX{ 0 }, mY{ 0 }, mZ{ 0 }, mW{ 0 };

		Tensor() = default;
		Tensor(TensorView floats, std::size_t x, std::size_t y=0, std::size_t z=0, std::size_t w=0) : mTensor(floats), mX(x), mY(y), mZ(z), mW(w) {}


		std::size_t size() const {
			return mX;
		}
		std::size_t size2D() const {
			return mX * mY;
		}
		std::size_t size3D() const {
			return mX * mY * mZ;
		}
		std::size_t size4D() const {
			return mX * mY * mZ * mW;
		}

		float& at(std::size_t col) const {
			return mTensor[col];
		}

		float& at(std::size_t row, std::size_t col) const {
			return  mTensor[row * mX + col];
		}
		float& atT(std::size_t row, std::size_t col) const {
			return  mTensor[col * mY + row];
		}

		float& at(std::size_t depth, std::size_t row, std::size_t col) const {
			return  mTensor[depth * (mY * mX) + row * mX + col];
		}
		float& atT(std::size_t depth, std::size_t row, std::size_t col) const {
			return  mTensor[depth * (mY * mX) + col * mY + row];
		}

		float& at(std::size_t w, std::size_t z, std::size_t y, std::size_t x) const {
			return  mTensor[w * (mZ * mY * mX) + z * (mY * mX) + y * mX + x];
		}

		TensorView span() const {
			return { &at(0), mX };
		}
		TensorView spanT() const {
			return { &at(0), mY };
		}

		TensorView span(std::size_t row) const {
			return { &at(row,0), mX };
		}
		TensorView spanT(size_t col) const {
 			return { &atT(0, col), mY };
		}

		TensorView span(std::size_t depth, std::size_t row) const {
			return { &at(depth, row, 0), mX };
		}
		TensorView spanT(std::size_t depth, std::size_t col) const {
			return { &at(depth, 0, col), mY };
		}

		TensorView span(std::size_t w, std::size_t z, std::size_t y) const {
			return { &at(w, z, y, 0), mX };
		}

		void forward(const auto& outputTensor, const auto& weightTensor, const auto& biasTensor) const {

			const auto b = biasTensor.span();

			Tensor::TensorView input, output, w;

			for (std::size_t i = 0; i < mTestInputSize; ++i) {

				input = spanT(i);
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
	};

	struct MLP {
		Tensor mCFCBias, mCFCWeight, mCProjBias, mCProjWeight;
		Tensor mCFCActivations, mCProjActivations, mGeluActivations;
	};
	struct LinearLayer {
		Tensor mBias, mWeight;
		Tensor mActivations;

		void normalise(auto& input) {

			Tensor::TensorView i, o, b, w;

			for (std::size_t m = 0; m < mTestInputSize; ++m) {

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

			for (std::size_t h = 0; h < mHeadNum; ++h) {

				const auto headOffset = h * mHeadsPerDModel;

				for (std::size_t q_i = 0; q_i < mTestInputSize; ++q_i) {

					Tensor::TensorView q = mCAttnActivations.spanT(q_i)
							, z = mAttnZ.spanT(q_i)

							, attnOut = mAttnActivations.spanT(h, q_i)
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
		void residual(const Tensor& inputTensor, const auto& projectionTensor, const auto& residualTensor ){

			Tensor::TensorView p, input, o;

			for (std::size_t i = 0; i < mTestInputSize; ++i) {

				p = projectionTensor.spanT(i);
				input = inputTensor.spanT(i);
				o = residualTensor.spanT(i);

				for (std::size_t m = 0; m < o.size(); ++m)
					o[m] = p[m] + input[m];
			}
		}
		Tensor& forward(const auto& inputTensor ) {

			mL1.normalise(inputTensor);
			mL1.mActivations.forward(mCAttnActivations, mCAttnWeight, mCAttnBias);

			attention();

			mAttnZ.forward(mCProjActivations, mCProjWeight, mCProjBias);

			residual(inputTensor, mCProjActivations, mResidualActivation1);

			mL2.normalise(mResidualActivation1);
			mL2.mActivations.forward(mMLP.mCFCActivations, mMLP.mCFCWeight, mMLP.mCFCBias );

			std::transform(mMLP.mCFCActivations.mTensor.begin(), mMLP.mCFCActivations.mTensor.end(), mMLP.mGeluActivations.mTensor.begin(),
				[](auto x) {
					return x * 0.5f * (1.0f + std::erff(x / std::sqrt(2.0f)));
				});
// 
			mMLP.mGeluActivations.forward(mMLP.mCProjActivations, mMLP.mCProjWeight, mMLP.mCProjBias);

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
		using WordMap= std::map<Word, Token>;

		Words mWords;
		WordMap mWordMap;//map words to their index

		static constexpr auto mDenseWordsSize = 321428;
		std::string mDenseWords;

		void readEnc() {

			auto readFile = [&]()  {

				//enc file https://github.com/rkaehn/gpt-2/blob/main/assets/enc
				auto fileName = std::format("{}enc", mFilePath);

				std::println("Reading file: {}", fileName);

				std::ifstream fin(fileName, std::ios::in | std::ios::binary);

				if (!fin)
					Error::fileNotFound(fileName);

				using Offset = std::pair<std::uint32_t, std::uint32_t>;
				std::vector<Offset> offsets; 

				offsets.resize(mDVocab);
				mWords.resize(mDVocab);
				mDenseWords.resize(mDenseWordsSize);

				fin.read(reinterpret_cast<char*>(offsets.data()), mDVocab * sizeof(Offset));
				fin.read(mDenseWords.data(), mDenseWordsSize);

				fin.close();
				std::puts("file read done");

				return offsets;		
				};

			auto offsets = readFile();

			for (std::size_t i = 0; i < mWords.size(); ++i) {

				auto& [offset, size] = offsets[i];
				auto& word = mWords[i];

				word = { mDenseWords.data() + offset, size };

				mWordMap[word] = i;
			}
		}

		std::string decode(std::span<Token> tokens) {

			std::string text;
			text.reserve(tokens.size()*3); //avg word size == 3?

			for (auto token : tokens) {
				text += mWords[token];
			}

			return text;
		}
	} mDecoder;

	struct Data {

		std::vector<Token> mTokens;

		void readData() {

			auto readFile = [&]() {

				//enc file https://github.com/rkaehn/gpt-2/blob/main/assets/data
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

	} mData;

public:

	void readSafeTensors() {

		constexpr auto floatSize = sizeof(float);
		using Header = std::string;

		auto readFile = [&]() -> Header {

			//gpt2 tensors https://huggingface.co/openai-community/gpt2?show_file_info=model.safetensors
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

		auto readTensorByName = [&](const auto& name) {

			auto& obj = j.at(name);
			auto& offsets = obj.at("data_offsets").as_array();
			auto a = offsets.front().as_int64() / floatSize, b = offsets.back().as_int64() / floatSize;

			auto start = std::next(mTensorSpace.begin(), a);
			auto end = std::next(mTensorSpace.begin(), b);
			auto size = std::distance(start, end);

			Tensor::TensorView tensorView(start, size);

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

		auto createAttentionLayer = [&](auto& attnLayer) {

			auto layerIdx = &attnLayer - mAttnLayers.data();
			auto layer = std::format("h.{}.", layerIdx);

			auto attnName = [layer](const auto& postFix) {
				return std::format("{}attn.{}", layer, postFix);
				};

			attnLayer.mBias = readTensorByName(attnName("bias"));
			attnLayer.mCAttnBias = readTensorByName(attnName("c_attn.bias"));
			attnLayer.mCAttnWeight = readTensorByName(attnName("c_attn.weight"));
			attnLayer.mCProjBias = readTensorByName(attnName("c_proj.bias"));
			attnLayer.mCProjWeight = readTensorByName(attnName("c_proj.weight"));

			auto linearName = [&](auto idx, const auto& postFix) {
				return std::format("{}ln_{}.{}", layer, idx, postFix);
				};

			attnLayer.mL1.mBias = readTensorByName(linearName(1, "bias"));
			attnLayer.mL1.mWeight = readTensorByName(linearName(1, "weight"));
			attnLayer.mL2.mBias = readTensorByName(linearName(2, "bias"));
			attnLayer.mL2.mWeight = readTensorByName(linearName(2, "weight"));

			auto mlpName = [&](const auto& postFix) {
				return std::format("{}mlp.{}", layer, postFix);
				};

			attnLayer.mMLP.mCFCBias = readTensorByName(mlpName("c_fc.bias"));
			attnLayer.mMLP.mCFCWeight = readTensorByName(mlpName("c_fc.weight"));
			attnLayer.mMLP.mCProjBias = readTensorByName(mlpName("c_proj.bias"));
			attnLayer.mMLP.mCProjWeight = readTensorByName(mlpName("c_proj.weight"));

			};

		std::for_each(mAttnLayers.begin(), mAttnLayers.end(), createAttentionLayer);

		mFinalLayer.mBias = readTensorByName("ln_f.bias");
		mFinalLayer.mWeight = readTensorByName("ln_f.weight");

		assert(floatsUsed == mTensorSpace.size());

		std::puts("Tensors read successfully");

		auto createActivationSpace = [&]() {

			auto seqModel = mDSeq * mDModel;
			auto seqModel3 = mDSeq * mDModel3;
			auto seqModel4 = mDSeq * mDModel4;
			auto seqSeqHead = mDSeq * mDSeq * mHeadNum;

			mActivationSpace.resize(seqModel + (seqModel*7 + seqModel3 + seqSeqHead*2 + seqModel4 * 2) * mAttnLayers.size());

			auto begin = mActivationSpace.begin();

			mWActivations = { {begin, seqModel}, mDSeq, mDModel };
			std::advance(begin, seqModel);

			for (auto& layer : mAttnLayers) {

				layer.mL1.mActivations = { {begin, seqModel}, mDSeq, mDModel };
				std::advance(begin, seqModel);

				layer.mCAttnActivations = { {begin, seqModel3}, mDSeq, mDModel3 };
				std::advance(begin, seqModel3);

				layer.mAttnActivations = { {begin, seqSeqHead}, mDSeq, mDSeq, mHeadNum };
				std::advance(begin, seqSeqHead);

				layer.mAttnSoftmaxActivations = { {begin, seqSeqHead}, mDSeq, mDSeq, mHeadNum };
				std::advance(begin, seqSeqHead);

				layer.mAttnZ = { {begin, seqModel}, mDSeq, mDModel };
				std::advance(begin, seqModel);
				
				layer.mCProjActivations = { {begin, seqModel}, mDSeq, mDModel };
				std::advance(begin, seqModel);

				layer.mResidualActivation1 = { {begin, seqModel}, mDSeq, mDModel };
				std::advance(begin, seqModel);

				layer.mL2.mActivations = { {begin, seqModel}, mDSeq, mDModel };
				std::advance(begin, seqModel);

				layer.mMLP.mCFCActivations = { {begin, seqModel4}, mDSeq, mDModel4 };
				std::advance(begin, seqModel4);

				layer.mMLP.mGeluActivations = { {begin, seqModel4}, mDSeq, mDModel4 };
				std::advance(begin, seqModel4);

				layer.mMLP.mCProjActivations = { {begin, seqModel}, mDSeq, mDModel };
				std::advance(begin, seqModel);

				layer.mResidualActivation2 = { {begin, seqModel}, mDSeq, mDModel };
				std::advance(begin, seqModel);

			}
			};
		createActivationSpace();
	}	

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

		auto checkSum = [&]() {

			assert(64 == mTestInputSize);

			auto getSum = [&](const auto& tensor) {
				return std::int64_t(std::reduce(tensor.mTensor.begin(), tensor.mTensor.end()));
			};

			assert( -30 == getSum(mWActivations));

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


			testFrontLayer();
			testBackLayer();
			};

		checkSum();
	}
};
