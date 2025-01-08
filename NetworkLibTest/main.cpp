#include <iostream>
#include <print>
#include <fstream>
#include <span>
#include <sstream>
#include <boost/json.hpp>

class GPT2 {

	using Tensor = std::vector<float>;
	using TensorView = std::span<float>;

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

	void readSafeTensors(const std::string& filePath= "F:/software dev/programming2025/downloads") {

		using FilePair = std::pair<std::string, Tensor >;
		auto readFile = [&]() -> FilePair {

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

			std::stringstream sstr;
			sstr << fin.rdbuf();
			fin.close();
			std::puts("file read...");

			auto strData = sstr.str();
			const auto charSize = strData.size(), floatSize = charSize / 4;

			TensorView source(reinterpret_cast<float*>(strData.data()), floatSize);
			Tensor dest(floatSize);
			std::copy(source.begin(), source.end(), dest.begin());
		
			return { header, dest};
			};

		auto [header, floats] = readFile();

		boost::json::value j = boost::json::parse(header);
		std::size_t floatsUsed = 0;

		auto readTensorByName = [&](const auto& name) {

			auto obj = j.at(name);
			auto offsets = obj.at("data_offsets").as_array();
			auto a = offsets.front().as_int64()/4, b = offsets.back().as_int64()/4;
			
			auto start = std::next(floats.begin(), a);
			auto end = std::next(floats.begin(), b);
			auto size = std::distance(start, end);

			std::vector<float> v(size);
			std::copy(start, end, v.begin());

			floatsUsed += size;
			return v;
			};

		mWpeWeight = readTensorByName("wpe.weight");
		mWteWeight = readTensorByName("wte.weight");

		auto getAttentionLayer = [&](const auto& layer) {

			auto attnName = [&]( const auto& postFix) {
				return std::format("h.{}.attn.{}", layer, postFix);
				};

			AttnLayer attnLayer;

			attnLayer.mBias = readTensorByName(attnName("bias"));
			attnLayer.mCAttnBias = readTensorByName(attnName("c_attn.bias"));
			attnLayer.mCAttnWeight = readTensorByName(attnName("c_attn.weight"));
			attnLayer.mCProjBias = readTensorByName(attnName("c_proj.bias"));
			attnLayer.mCProjWeight	= readTensorByName(attnName("c_proj.weight"));

			auto linearName = [&](auto idx, const auto& postFix) {
				return std::format("h.{}.ln_{}.{}", layer, idx, postFix);
				};

			attnLayer.mL1.mBias = readTensorByName(linearName(1, "bias"));
			attnLayer.mL1.mWeight = readTensorByName(linearName(1, "weight"));
			attnLayer.mL2.mBias = readTensorByName(linearName(2, "bias"));
			attnLayer.mL2.mWeight = readTensorByName(linearName(2, "weight"));

			auto mlpName = [&](const auto& postFix) {
				return std::format("h.{}.mlp.{}", layer, postFix);
				};

			attnLayer.mMLP.mCFCBias = readTensorByName(mlpName("c_fc.bias"));
			attnLayer.mMLP.mCFCWeight = readTensorByName(mlpName("c_fc.weight"));
			attnLayer.mMLP.mCProjBias = readTensorByName(mlpName("c_proj.bias"));
			attnLayer.mMLP.mCProjWeight = readTensorByName(mlpName("c_proj.weight"));

			return attnLayer;
			};

		mAttnLayers.resize(mAttentionLayersSize);
		for (int i = 0; i < mAttentionLayersSize; i++)
			mAttnLayers[i] = getAttentionLayer(i);

		mFinalLayer.mBias = readTensorByName("ln_f.bias");
		mFinalLayer.mWeight = readTensorByName("ln_f.weight");

		assert( floatsUsed == data.size());

		std::puts("Tensors read successfully");
	}
};

int main() {

	GPT2 gpt2;
	gpt2.readSafeTensors();

	std::puts("Program Finished press enter to exit");
	std::cin.get();

	return 0;
}