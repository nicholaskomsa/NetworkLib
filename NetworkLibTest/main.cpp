#define FREEIMAGE_LIB
#include <FreeImage.h>


#include <iostream>
#include <print>
#include <fstream>
#include <span>
#include <sstream>
#include <execution>
#include <boost/json.hpp>


namespace FloatSpaceConvert {

	//converts fits FLOAT images to each colorize mode
	enum class ColorizeMode {
		NICKRGB,
		SHORTNRGB,
		ROYGBIV,
		GREYSCALE,
		BINARY
	};

	std::uint32_t rgb(std::uint8_t r, std::uint8_t g, std::uint8_t b) {

		std::uint32_t rgba = 0;
		std::uint8_t* bytes = reinterpret_cast<std::uint8_t*>(&rgba);
		bytes[0] = r;
		bytes[1] = g;
		bytes[2] = b;

		return rgba;
	}

	auto nrgb = [&](auto percent)->std::uint32_t {

		//produce a three bytes (rgb) max value
		constexpr std::uint32_t maxValue = { std::numeric_limits<std::uint32_t>::max() >> 8 };

		std::uint32_t value = maxValue * percent;
		return value;
		};

	auto snrgb = [&](auto percent)->std::uint32_t {

		//produce a three bytes (rgb) max value
		constexpr std::uint32_t maxValue = { std::numeric_limits<std::uint32_t>::max() >> 16 };

		return maxValue * percent;
		};
	auto roygbiv = [&](auto percent) {

		uint8_t r = 0, g = 0, b = 0;

		/*plot short rainbow RGB*/
		float a = (1.0 - percent) / 0.20;	//invert and group
		int X = std::floor(a);	//this is the integer part
		float Y = std::floor(255.0 * (a - X)); //fractional part from 0 to 255
		switch (X) {
		case 0: r = 255; g = Y; b = 0; break;
		case 1: r = 255 - Y; g = 255; b = 0; break;
		case 2: r = 0; g = 255; b = Y; break;
		case 3: r = 0; g = 255 - Y; b = 255; break;
		case 4: r = Y; g = 0; b = 255; break;
		case 5: r = 255; g = 0; b = 255; break;
		}

		return rgb(r, g, b);
		};

	auto grayScale = [&](auto percent) {

		constexpr std::uint8_t maxValue = { std::numeric_limits<std::uint8_t>::max() };
		std::uint8_t gray = maxValue * percent;
		return rgb(gray, gray, gray);

		};

	auto binary = [&](auto percent) {

		constexpr std::uint8_t maxValue = { std::numeric_limits<std::uint8_t>::max() };
		//perrcent is between 0 and 1 so round to 0 or 1 and multiply by max value for either 0 or 255
		std::uint8_t bit = std::round(percent);
		std::uint8_t gray = maxValue * bit;
		return rgb(gray, gray, gray);

		};

	void floatSpaceConvert(std::span<const float> data, std::span<uint32_t> converted, ColorizeMode colorMode = ColorizeMode::NICKRGB, double vMin = 0.0, double vMax = 1.0, double stripeNum = 1) {

		auto getViewWindow = [&](double startPercent = 0.0, double endPercent = 1.0) ->std::tuple<double, double, double> {

			auto minmax = std::minmax_element(data.begin(), data.end());
			auto min = *minmax.first, max = *minmax.second;

			double distance = max - min;

			double viewMin = min + distance * startPercent;
			double viewMax = min + distance * endPercent;
			double viewDistance = viewMax - viewMin;

			if (viewDistance == 0) viewDistance = 1;

			return { viewMin, viewMax, viewDistance };
			};

		auto [viewMin, viewMax, viewDistance] = getViewWindow(vMin, vMax);	//0,1 is full view window of data

		double stripeDistance = viewDistance / stripeNum;

		auto convertToGreyScale = [&](double f)->double {

			double percent = 1.0;

			f -= viewMin;

			if (f < viewDistance) {
				f -= stripeDistance * std::floor(f / stripeDistance);

				percent = f / stripeDistance;
			}

			//percent is between 0 and 1
			return percent;
			};

		auto setOpaque = [&](std::uint32_t& p) {
			reinterpret_cast<uint8_t*>(&p)[3] = 255;
			};

		auto forEachPixel = [&](auto&& colorize) {

			std::transform(std::execution::seq, data.begin(), data.end(), converted.begin(), [&](auto& f) {

				auto percent = convertToGreyScale(f);

				auto rgba = colorize(percent);

				//we want these pixels to be defined as completly non-transparent
				setOpaque(rgba);

				return rgba;

				});
			};

		switch (colorMode) {
		case ColorizeMode::NICKRGB: {

			forEachPixel(nrgb);

		}break;

		case ColorizeMode::ROYGBIV: {

			forEachPixel(roygbiv);

		} break;

		case ColorizeMode::GREYSCALE: {

			forEachPixel(grayScale);

		} break;

		case ColorizeMode::BINARY: {

			forEachPixel(binary);

		} break;

		case ColorizeMode::SHORTNRGB: {

			forEachPixel(snrgb);

		} break;
		}
	}

	void colorizeFloatSpace(const std::string& fileName, std::span<const float> floats, bool rectangular = true) {

		std::println("Colorizing float space: {}", fileName);

		auto writeColorizedImages = [&](auto idx, auto& image, auto width, auto height) {

			if (image.size() == 0) return;

			auto saveToBmpFile = [&](std::string fileName, std::span<uint32_t> image) {

				uint8_t* bytes = reinterpret_cast<uint8_t*>(image.data());
				//converted data are four byte type (int32)
				//r g b a

				int pitch = width * (32 / 8);

				//freeimage is writing in bgra format
				auto bgra = [&](std::uint32_t rgba) {

					std::uint32_t tmp = rgba;
					std::uint8_t* bytes = reinterpret_cast<std::uint8_t*>(&tmp);
					std::swap(bytes[0], bytes[2]);

					return tmp;
					};

				std::transform(image.begin(), image.end(), image.begin(), bgra);

				//correct byte order for free image write
				FIBITMAP* convertedImage = FreeImage_ConvertFromRawBits(bytes, width, height, pitch, 32, FI_RGBA_RED_MASK, FI_RGBA_GREEN_MASK, FI_RGBA_BLUE_MASK);

				FreeImage_Save(FIF_BMP, convertedImage, fileName.c_str(), 0);

				FreeImage_Unload(convertedImage);
				};

			auto stripes = { 1,2,10,20,50,100 };
			auto colorizeModes = { ColorizeMode::GREYSCALE, ColorizeMode::ROYGBIV, ColorizeMode::NICKRGB, ColorizeMode::BINARY, ColorizeMode::SHORTNRGB };

			auto colorizeModeStr = [&](auto colorizeMode) {
				switch (colorizeMode) {
				case ColorizeMode::NICKRGB: return "nickrgb";
				case ColorizeMode::ROYGBIV: return "roygbiv";
				case ColorizeMode::GREYSCALE: return "greyscale";
				case ColorizeMode::BINARY: return "binary";
				case ColorizeMode::SHORTNRGB: return "snrgb";
				}
				return "unknown";
				};

			FreeImage_Initialise();

			std::for_each(std::execution::par, stripes.begin(), stripes.end(), [&](int stripeNum) {

				std::vector<uint32_t> converted(width*height); //adjust to texture min size

				for (auto colorizeMode : colorizeModes) {

					floatSpaceConvert(image, converted, colorizeMode, 0.0f, 1.0f, stripeNum);

					auto completeFileNameWithColorMode = std::format("{}_{}_{}.bmp", fileName, colorizeModeStr(colorizeMode), stripeNum);
					saveToBmpFile(completeFileNameWithColorMode, converted);
				}

				});

			FreeImage_DeInitialise();

			};

		auto size = floats.size();

		auto getDimensions = [&](auto size, float aspectRatio = 3840.0f/ 2160.0f) ->std::pair<int, int> {

			int width = 0, height = 0;

			width = std::sqrt(size * aspectRatio) ;
			height = std::ceil(size / float(width));

			return { width, height };
			};

		auto [width, height] = getDimensions(size);

		writeColorizedImages(fileName, floats, width, height);

	}
};


struct GPT2 {

	using Tensor = std::vector<float>;
	using TensorView = std::span<float>;

	struct MLP {
		TensorView mCFCBias, mCFCWeight, mCProjBias, mCProjWeight;
	};
	struct LinearLayer {
		TensorView mBias, mWeight;
	};
	struct AttnLayer {

		LinearLayer mL1, mL2; 

		TensorView mBias, mCAttnBias, mCAttnWeight, mCProjBias, mCProjWeight;

		MLP mMLP;
	};

	Tensor mFloatSpace;

	TensorView mWpeWeight, mWteWeight;
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

			std::streampos current = fin.tellg(), end = fin.seekg(current, std::ios::end).tellg();

			constexpr auto floatSize = sizeof(float);
			std::streamoff floatsSize = static_cast<std::streamoff>(end - current) / floatSize;
			mFloatSpace.resize(floatsSize);

			fin.seekg(current);
			fin.read(reinterpret_cast<char*>(mFloatSpace.data()), floatsSize * floatSize);

			fin.close();
			std::puts("file read...");

			return header;
			};

		auto header = readFile();

		boost::json::value j = boost::json::parse(header);
		std::size_t floatsUsed = 0;

		auto readTensorByName = [&](const auto& name) {

			auto obj = j.at(name);
			auto offsets = obj.at("data_offsets").as_array();
			auto a = offsets.front().as_int64()/4, b = offsets.back().as_int64()/4;
			
			auto start = std::next(mFloatSpace.begin(), a);
			auto end = std::next(mFloatSpace.begin(), b);
			auto size = std::distance(start, end);

			TensorView tensor(start, size);

			floatsUsed += size;
			return tensor;
			};

		mWpeWeight = readTensorByName("wpe.weight");
		mWteWeight = readTensorByName("wte.weight");

		auto createAttentionLayer = [&](auto& attnLayer) {

			auto layerIdx = &attnLayer - mAttnLayers.data();
			auto layer = std::format("h.{}.", layerIdx);

			auto attnName = [&]( const auto& postFix) {
				return std::format("{}attn.{}", layer, postFix);
				};

			attnLayer.mBias = readTensorByName(attnName("bias"));
			attnLayer.mCAttnBias = readTensorByName(attnName("c_attn.bias"));
			attnLayer.mCAttnWeight = readTensorByName(attnName("c_attn.weight"));
			attnLayer.mCProjBias = readTensorByName(attnName("c_proj.bias"));
			attnLayer.mCProjWeight	= readTensorByName(attnName("c_proj.weight"));

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

			return attnLayer;
			};

		mAttnLayers.resize(mAttentionLayersSize);
		std::for_each(mAttnLayers.begin(), mAttnLayers.end(), createAttentionLayer);

		mFinalLayer.mBias = readTensorByName("ln_f.bias");
		mFinalLayer.mWeight = readTensorByName("ln_f.weight");

		assert( floatsUsed == mFloatSpace.size());

		std::puts("Tensors read successfully");
	}
};

int main() {

	GPT2 gpt2;
	gpt2.readSafeTensors();

	FloatSpaceConvert::colorizeFloatSpace("gpt2", gpt2.mFloatSpace);

	std::puts("Program Finished press enter to exit");
	std::cin.get();

	return 0;
}