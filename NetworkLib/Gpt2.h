#pragma once

#include <iostream>
#include <print>
#include <fstream>
#include <span>
#include <sstream>
#include <execution>
#include <boost/json.hpp>

struct GPT2 {

	using Floats = std::vector<float>;

	struct Tensor {

		using TensorView = std::span<float>;
		TensorView mTensor;

		std::size_t mX{ 0 }, mY{ 0 }, mZ{ 0 }, mW{ 0 };

		Tensor() = default;
		Tensor(TensorView floats, std::size_t x) : mTensor(floats), mX(x) {}
		Tensor(TensorView floats, std::size_t x, std::size_t y) : mTensor(floats), mX(x), mY(y) {}
		Tensor(TensorView floats, std::size_t x, std::size_t y, std::size_t z) : mTensor(floats), mX(x), mY(y), mZ(z) {}
		Tensor(TensorView floats, std::size_t x, std::size_t y, std::size_t z, std::size_t w) : mTensor(floats), mX(x), mY(y), mZ(z), mW(w) {}


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

		float& at(std::size_t col) {
			return mTensor[col];
		}
		float& at(std::size_t row, std::size_t col) {
			return  mTensor[row * mX + col];
		}
		float& at(std::size_t depth, std::size_t row, std::size_t col) {
			return  mTensor[depth * (mY * mX) + row * mX + col];
		}
		float& at(std::size_t w, std::size_t z, std::size_t y, std::size_t x) {
			return  mTensor[w * (mZ * mY * mX) + z * (mY * mX) + y * mX + x];
		}
		TensorView span(std::size_t col) {
			return { &at(col), mX };
		}
		TensorView span(std::size_t row, std::size_t col) {
			return { &at(row,col), mX };
		}
		TensorView span(std::size_t depth, std::size_t row, std::size_t col) {
			return { &at(depth, row, col), mX };
		}
	};

	struct MLP {
		Tensor mCFCBias, mCFCWeight, mCProjBias, mCProjWeight;
	};
	struct LinearLayer {
		Tensor mBias, mWeight;
	};
	struct AttnLayer {

		LinearLayer mL1, mL2;

		Tensor mBias, mCAttnBias, mCAttnWeight, mCProjBias, mCProjWeight;

		MLP mMLP;
	};

	Floats mFloatSpace;

	Tensor mWpeWeight, mWteWeight;
	LinearLayer mFinalLayer;

	static constexpr auto mAttentionLayersSize = 12;
	std::vector<AttnLayer> mAttnLayers;

	struct Error : public std::system_error {

		Error(std::errc code, const std::string& message) : std::system_error(int(code), std::generic_category(), message) {}

		static void fileNotFound() {
			throw Error(std::errc::no_such_file_or_directory, "file not found");
		}
	};

public:

	void readSafeTensors(const std::string& filePath = "F:/software dev/programming2025/downloads") {

		using Header = std::string;
		auto readFile = [&]() -> Header {

			//gpt2 tensors https://huggingface.co/openai-community/gpt2?show_file_info=model.safetensors
			auto fileName = std::format("{}/model.safeTensors", filePath);

			std::println("Reading file: {}", fileName);

			std::ifstream fin(fileName, std::ios::in | std::ios::binary);

			if (!fin)
				Error::fileNotFound();

			std::uint64_t headerSize;
			fin.read(reinterpret_cast<char*>(&headerSize), sizeof(headerSize));

			std::string header; header.resize(headerSize);
			fin.read(header.data(), header.size());

			std::streampos current = fin.tellg(), end = fin.seekg(0, std::ios::end).tellg();

			constexpr auto floatSize = sizeof(float);
			std::streamoff floatsSize = static_cast<std::streamoff>(end - current) / floatSize;
			mFloatSpace.resize(floatsSize);

			fin.seekg(current);
			fin.read(reinterpret_cast<char*>(mFloatSpace.data()), floatsSize * floatSize);

			constexpr std::size_t knownFileFloatSize = 548090880 / floatSize;

			assert(knownFileFloatSize == floatsSize);

			fin.close();
			std::puts("file read...");

			return header;
			};

		auto header = readFile();

		boost::json::value j = boost::json::parse(header);
		std::size_t floatsUsed = 0;

		auto readTensorByName = [&](const auto& name) {

			auto& obj = j.at(name);
			auto& offsets = obj.at("data_offsets").as_array();
			auto a = offsets.front().as_int64() / 4, b = offsets.back().as_int64() / 4;

			auto start = std::next(mFloatSpace.begin(), a);
			auto end = std::next(mFloatSpace.begin(), b);
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
				expectedSize = tensor.size();;
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

			auto attnName = [&](const auto& postFix) {
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

		mAttnLayers.resize(mAttentionLayersSize);
		std::for_each(mAttnLayers.begin(), mAttnLayers.end(), createAttentionLayer);

		mFinalLayer.mBias = readTensorByName("ln_f.bias");
		mFinalLayer.mWeight = readTensorByName("ln_f.weight");

		assert(floatsUsed == mFloatSpace.size());

		std::puts("Tensors read successfully");
	}

	void readPostEmbedText() {

	}

	void feedForward() {

	}
};
