#pragma once

#include <print>
#include <iostream>
#include <vector>
#include <span>

#include <execution>

#define FREEIMAGE_LIB
#include <FreeImage.h>

namespace FloatSpaceConvert {

	//converts fits FLOAT images to each colorize mode
	enum class ColorizeMode {
		NICKRGB,
		SHORTNRGB,
		ROYGBIV,
		GREYSCALE,
		BINARY
	};
	using ColorNames = std::map<ColorizeMode, std::string>;

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

		//produce a two bytes (rg) max value
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

	void colorizeFloatSpace(const std::string& baseFileName, std::span<const float> floats) {

		std::println("Colorizing float space: {}", baseFileName);

		auto writeColorizedImages = [&](auto& image, auto width, auto height) {

			if (image.size() == 0) return;

			auto saveToBmpFile = [&](std::string fileName, std::span<uint32_t> image) {

				uint8_t* bytes = reinterpret_cast<uint8_t*>(image.data());
				//converted data are four byte type (int32)
				//r g b a

				int pitch = width * (32 / 8);

				//freeimage is writing in bgra format depending if you are windows vs apple, check your free image file format
				auto bgra = [&](std::uint32_t rgba) {

					std::uint8_t* bytes = reinterpret_cast<std::uint8_t*>(&rgba);
					std::swap(bytes[0], bytes[2]);

					return rgba;
					};

				std::transform(image.begin(), image.end(), image.begin(), bgra);

				//correct byte order for free image write
				FIBITMAP* convertedImage = FreeImage_ConvertFromRawBits(bytes, width, height, pitch, 32, FI_RGBA_RED_MASK, FI_RGBA_GREEN_MASK, FI_RGBA_BLUE_MASK);

				FreeImage_Save(FIF_BMP, convertedImage, fileName.c_str(), 0);

				FreeImage_Unload(convertedImage);
				};

			ColorNames colorNames = {
				{ ColorizeMode::NICKRGB, "nickrgb"}
				,{ ColorizeMode::SHORTNRGB, "shortnrgb" }
				,{ ColorizeMode::ROYGBIV, "roygbiv" }
				,{ ColorizeMode::GREYSCALE, "greyscale" }
				,{ ColorizeMode::BINARY, "binary" }
			};

			auto stripes = { 1,2,10,20,50,100 };

			FreeImage_Initialise();

			std::for_each(std::execution::par, stripes.begin(), stripes.end(), [&](int stripeNum) {

				std::vector<uint32_t> converted(width * height); //adjust to texture min size

				for (auto& [mode, name] : colorNames) {

					floatSpaceConvert(image, converted, mode, 0.0f, 1.0f, stripeNum);

					auto completeFileNameWithColorMode = std::format("{}_{}_{}.bmp", baseFileName, name, stripeNum);
					saveToBmpFile(completeFileNameWithColorMode, converted);
				}

				});

			FreeImage_DeInitialise();

			};

		auto getDimensions = [&](auto size, float aspectRatio = 3840.0f / 2160.0f) ->std::pair<int, int> {

			int width = 0, height = 0;

			width = std::sqrt(size * aspectRatio);
			height = std::ceil(size / float(width));

			return { width, height };
			};

		auto [width, height] = getDimensions(floats.size());

		writeColorizedImages(floats, width, height);

	}
};	