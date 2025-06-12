#include "FloatSpaceConvert.h"

#define FREEIMAGE_LIB
#include <FreeImage.h>

#include <print>
#include <execution>
#include <algorithm>

void FloatSpaceConvert::setOpaque(std::uint32_t& p) {
	reinterpret_cast<uint8_t*>(&p)[3] = 255;
}
std::uint32_t FloatSpaceConvert::rgb(std::uint8_t r, std::uint8_t g, std::uint8_t b) {

	std::uint32_t rgba = 0;
	std::uint8_t* bytes = reinterpret_cast<std::uint8_t*>(&rgba);
	bytes[0] = r;
	bytes[1] = g;
	bytes[2] = b;

	return rgba;
}

std::uint32_t FloatSpaceConvert::nrgb(double percent) {

	//produce a three bytes (rgb) max value
	constexpr std::uint32_t maxValue = { std::numeric_limits<std::uint32_t>::max() >> 8 };

	return maxValue * percent;
}

std::uint32_t FloatSpaceConvert::snrgb(double percent) {

	//produce a two bytes (rg) max value
	constexpr std::uint32_t maxValue = { std::numeric_limits<std::uint32_t>::max() >> 16 };

	return maxValue * percent;
}

std::uint32_t FloatSpaceConvert::roygbiv(double percent) {

	//https://www.particleincell.com/2014/colormap/
	uint8_t r = 0, g = 0, b = 0;

	/*plot long rainbow RGB*/
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
}

std::uint32_t FloatSpaceConvert::grayScale(double percent) {

	constexpr std::uint8_t maxValue = { std::numeric_limits<std::uint8_t>::max() };
	std::uint8_t gray = maxValue * percent;
	return rgb(gray, gray, gray);

}

std::uint32_t FloatSpaceConvert::binary(double percent) {

	constexpr std::uint8_t maxValue = { std::numeric_limits<std::uint8_t>::max() };
	//perrcent is between 0 and 1 so round to 0 or 1 and multiply by max value for either 0 or 255
	std::uint8_t bit = std::round(percent);
	std::uint8_t gray = maxValue * bit;
	return rgb(gray, gray, gray);

}

void FloatSpaceConvert::floatSpaceConvert(std::span<const float> data, std::span<uint32_t> converted, ColorizeMode colorMode, double vMin, double vMax, double stripeNum) {

	//floatSpaceConvert(data, converted, nullptr, colorMode, vMin, vMax, stripeNum);
	auto [width, height] = getDimensions(data.size(), 1.0f);
	floatSubSpaceConvert(data, converted, 0, 0, width, height, width
		, colorMode, vMin, vMax, stripeNum);
}
std::pair<std::size_t, std::size_t> FloatSpaceConvert::getDimensions(std::size_t size, float aspectRatio) {

	int width = 0, height = 0;

	width = std::sqrt(size * aspectRatio);
	height = std::ceil(size / float(width));

	return { width, height };
}

void FloatSpaceConvert::colorizeFloatSpace(const std::string& baseFileName, std::span<const float> floats) {

	std::println("Colorizing float space: {}", baseFileName);

	auto writeColorizedImages = [&](auto& image, auto width, auto height) {

		if (image.size() == 0) return;

		auto saveToBmpFile = [&](std::string fileName, std::span<uint32_t> image) {
			//image is int32 r g b a

			//freeimage is writing in bgra format depending if you are windows vs apple, check your free image file format
			auto bgra = [&](std::uint32_t rgba) {

				std::uint8_t* bytes = reinterpret_cast<std::uint8_t*>(&rgba);
				std::swap(bytes[0], bytes[2]);

				return rgba;
				};

			//given windows do this transform else consider free image file format
			std::transform(image.begin(), image.end(), image.begin(), bgra);

			uint8_t* bytes = reinterpret_cast<uint8_t*>(image.data());
			int pitch = width * (32 / 8);

			//converted data are four byte type (int32)
			//consider byte order for free image write unless windows
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

		auto stripes = { 1,2,10,100 };

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

	auto [width, height] = getDimensions(floats.size());

	writeColorizedImages(floats, width, height);

}

FloatSpaceConvert::ColorNames FloatSpaceConvert::getColorNames() {
	return {
		{ ColorizeMode::NICKRGB, "nickrgb"}
		,{ ColorizeMode::SHORTNRGB, "shortnrgb" }
		,{ ColorizeMode::ROYGBIV, "roygbiv" }
		,{ ColorizeMode::GREYSCALE, "greyscale" }
		,{ ColorizeMode::BINARY, "binary" }
	};
}

void FloatSpaceConvert::floatSubSpaceConvert(std::span<const float> data, std::span<uint32_t> converted
	, std::size_t x, std::size_t y, std::size_t w, std::size_t h
	, std::size_t textureWidth
	, ColorizeMode colorMode, double vMin, double vMax, double stripeNum) {

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

	auto getColorizeMethod = [&]() ->ColorizeMethod {

		switch (colorMode) {
		case ColorizeMode::NICKRGB: return &nrgb;
		case ColorizeMode::SHORTNRGB: return &snrgb;
		case ColorizeMode::ROYGBIV: return &roygbiv;
		case ColorizeMode::GREYSCALE: return &grayScale;
		case ColorizeMode::BINARY: return &binary;
		}
		};
	ColorizeMethod colorizeMethod = getColorizeMethod();

	auto convertMethod = [&](float f) {

		auto percent = convertToGreyScale(f);

		auto rgba = colorizeMethod(percent);

		//we want these pixels to be defined as completly non-transparent
		setOpaque(rgba);

		return rgba;

		};

	//x y w h are received in texture-space ( 0, textureWidth ) and ( 0, textureHeight )
	//converted may be larger than data and has unused pixels

	auto forSubPixels = [&]() {

		auto hIota = std::views::iota(y, y + h);

		std::for_each(std::execution::par, hIota.begin(), hIota.end(), [&](auto iy) {

			auto offset = iy * textureWidth;

			for (auto ix : std::views::iota(x, x + w)) {

				auto index = offset + ix;

				if (index < data.size())
					converted[index] = convertMethod(data[index]);
			}

			});
		};
	
	forSubPixels();
}