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
		, mDModel = 768, mDModel3 = mDModel * 3
		, mDSeq = 1024
		, mHeadNum = 12
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

		void forward(auto& outputTensor, auto& weightTensor, auto& biasTensor) const {

			for (std::size_t i = 0; i < mTestInputSize; ++i) {

				auto input = spanT(i);
				auto bias = biasTensor.span();
				auto output = outputTensor.spanT(i);

				std::copy(bias.begin(), bias.end(), output.begin());

				for (std::size_t m = 0; m < input.size(); ++m) {

					auto in = input[m];
					auto w = weightTensor.spanT(m);

					for (std::size_t n = 0; n < output.size(); ++n)
						output[n] += w[n] * in;
				}
			}
			};
	};

	struct MLP {
		Tensor mCFCBias, mCFCWeight, mCProjBias, mCProjWeight;
		Tensor mActivations;
	};
	struct LinearLayer {
		Tensor mBias, mWeight;
		Tensor mActivations;
	};
	struct AttnLayer {

		LinearLayer mL1, mL2;

		Tensor mBias, mCAttnBias, mCAttnWeight, mCProjBias, mCProjWeight;
		Tensor mCAttnActivations, mAttnActivations, mAttnSoftmaxActivations, mAttnZ, mCProjActivations;

		MLP mMLP;
	};

	Floats mTensorSpace, mActivationSpace;

	Tensor mWpeWeight, mWteWeight, mWActivations;
	LinearLayer mFinalLayer;
	std::array<AttnLayer,12> mAttnLayers;

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

				fin.read(reinterpret_cast<char*>(offsets.data()),  mDVocab * sizeof(Offset));
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

			auto embedSize = mDSeq * mDModel;
			auto cAttnSize = mDSeq * mDModel3;
			auto attnSize = mDSeq * mDSeq * mHeadNum;
			auto attnSoftMaxSize = attnSize;
			auto zSize = mDSeq * mDModel;

			mActivationSpace.resize(embedSize + (embedSize + cAttnSize + attnSize + attnSoftMaxSize + zSize + zSize) * mAttnLayers.size());

			auto begin = mActivationSpace.begin();

			mWActivations = { {begin, embedSize}, mDSeq, mDModel };
			std::advance(begin, embedSize);

			for (auto& layer : mAttnLayers) {

				layer.mL1.mActivations = { {begin, embedSize}, mDSeq, mDModel };
				std::advance(begin, embedSize);

				layer.mCAttnActivations = { {begin, cAttnSize}, mDSeq, mDModel3 };
				std::advance(begin, cAttnSize);

				layer.mAttnActivations = { {begin, attnSize}, mDSeq, mDSeq, mHeadNum };
				std::advance(begin, attnSize);

				layer.mAttnSoftmaxActivations = { {begin, attnSoftMaxSize}, mDSeq, mDSeq, mHeadNum };
				std::advance(begin, attnSoftMaxSize);

				layer.mAttnZ = { {begin, zSize}, mDSeq, mDModel };
				std::advance(begin, zSize);
				
				layer.mCProjActivations = { {begin, zSize}, mDSeq, mDModel };
				std::advance(begin, zSize);
			}
			};
		createActivationSpace();
	}	

	void feedForward() {

		const auto r_sqrtHeadsPerDModel = 1.0f / std::sqrtf(mHeadsPerDModel);

		for (std::size_t i = 0; i < mTestInputSize; ++i) { //inputSize vs dseq

			Token token = mData.mTokens[i];

			//wte dvocab * dmodel
			auto wte = mWteWeight.spanT(token);//this is transposed span over mDModel
			//wpe dseq * dmodel
			auto wpe = mWpeWeight.spanT(i);

			auto wActivations = mWActivations.spanT(i);

			for (std::size_t w = 0; w < mDModel; ++w)
				wActivations[w] = wte[w] + wpe[w];
		}

		for (std::size_t layer_i = 0; layer_i < 1/*mAttnLayers.size()*/; ++layer_i) {

			auto& layer = mAttnLayers[layer_i];
			for (std::size_t i = 0; i < mTestInputSize; ++i) {

				auto wActivations = mWActivations.spanT(i);

				auto mean = std::reduce(wActivations.begin(), wActivations.end()) / wActivations.size();

				auto meanDiffSq = std::reduce(wActivations.begin(), wActivations.end(), 0.0f,
					[&](auto sum, auto w) {
						auto diff = w - mean;
						return sum + diff * diff;
					}) / wActivations.size();

				auto r_stdDev = 1.0f / std::sqrt(meanDiffSq);

				auto ln1Bias = layer.mL1.mBias.span();
				auto ln1Weight = layer.mL1.mWeight.span();
				auto layerOut = layer.mL1.mActivations.spanT(i);

				for (std::size_t w = 0; w < wActivations.size(); ++w) {
					auto inNorm = (wActivations[w] - mean) * r_stdDev;
					layerOut[w] = inNorm * ln1Weight[w] + ln1Bias[w];
				}
			}

			layer.mL1.mActivations.forward(layer.mCAttnActivations, layer.mCAttnWeight, layer.mCAttnBias);

			//activations z cleared here
			auto zoutTensor = layer.mAttnZ.mTensor;
			std::fill(zoutTensor.begin(), zoutTensor.end(), 0.0f);

			for (std::size_t h = 0; h < mHeadNum; ++h) {

				const auto headOffset = h * mHeadsPerDModel;

				for (std::size_t q_i = 0; q_i < mTestInputSize; ++q_i) {

					auto qout = layer.mCAttnActivations.spanT(q_i);
					auto attnOut = layer.mAttnActivations.spanT(h, q_i);
					auto zout = layer.mAttnZ.spanT(q_i);
					auto softmaxOut = layer.mAttnSoftmaxActivations.spanT(h, q_i);

					auto calculateQKAtten = [&]() {
						Tensor::TensorView q = { qout.data() + mQOffset, mDModel };
						Tensor::TensorView qh = { q.data() + headOffset, mHeadsPerDModel };

						for (std::size_t k_i = 0; k_i <= q_i; ++k_i) {

							auto kout = layer.mCAttnActivations.spanT(k_i);
							Tensor::TensorView k = { kout.data() + mKOffset, mDModel };
							Tensor::TensorView kh = { k.data() + headOffset, mHeadsPerDModel };

							float dot = 0.0f;

							for (std::size_t n = 0; n < qh.size(); ++n) {
								dot += qh[n] * kh[n];
							}

							attnOut[k_i] = dot * r_sqrtHeadsPerDModel;
						}
						};
					auto qkAttnSoftmax = [&]() {

						const auto softmaxMax = *std::max_element(attnOut.begin(), attnOut.begin() + q_i);

						float softmaxSum = 0.0f;

						for (std::size_t k_i = 0; k_i <= q_i; ++k_i) {

							auto softmaxExp = std::expf(attnOut[k_i] - softmaxMax);

							softmaxOut[k_i] = softmaxExp;

							softmaxSum += softmaxExp;
						}

						auto r_softmaxSum = 1.0f / softmaxSum;

						for (std::size_t k_i = 0; k_i <= q_i; ++k_i)
							softmaxOut[k_i] *= r_softmaxSum;
						};
					auto calculateVAtten = [&]() {
						Tensor::TensorView zh = { zout.data() + headOffset, mHeadsPerDModel };

						for (std::size_t v_i = 0; v_i <= q_i; ++v_i) {

							auto vout = layer.mCAttnActivations.spanT(v_i);
							Tensor::TensorView v = { vout.data() + mVOffset, mDModel };
							Tensor::TensorView vh = { v.data() + headOffset, mHeadsPerDModel };

							auto factor = softmaxOut[v_i];

							for (std::size_t n = 0; n < vh.size(); ++n) {
								zh[n] += vh[n] * factor;
							}
						}
						};

					calculateQKAtten();
					qkAttnSoftmax();
					calculateVAtten();
				}
			}

			layer.mAttnZ.forward(layer.mCProjActivations, layer.mCProjWeight, layer.mCProjBias);
		}

		auto checkSum = [&](auto& layer) {

			auto getSum = [&](auto& tensor) {
				return std::int64_t(std::reduce(tensor.begin(), tensor.end()));
			};

			assert( -30 == getSum(mWActivations.mTensor));
			assert( -334 == getSum(layer.mL1.mActivations.mTensor));
			assert( -3325 == getSum(layer.mCAttnActivations.mTensor));
			assert( 454 == getSum(layer.mAttnZ.mTensor));
			assert( 389 == getSum(layer.mCProjActivations.mTensor));

			};

		checkSum(mAttnLayers.front());
	}
};
