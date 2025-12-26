#include "FloatSpaceConvert.h"

#define FREEIMAGE_LIB
#include <FreeImage.h>

#include <ranges>
#include <print>
#include <execution>
#include <algorithm>

using namespace std;

using UInt32Limits = std::numeric_limits<uint32_t>;
//UInt32 is the size of a pixel ( RGBA )
using UInt8Limits = std::numeric_limits<uint8_t>;
//UInt8 is the size of a color channel ( R or G or B or A ) or one pixel byte

void FloatSpaceConvert::setOpaque(uint32_t& p) {
	reinterpret_cast<uint8_t*>(&p)[3] = 255;
}
uint32_t FloatSpaceConvert::rgb(uint8_t r, uint8_t g, uint8_t b) {

	uint32_t rgba = 0;
	uint8_t* bytes = reinterpret_cast<uint8_t*>(&rgba);
	bytes[0] = r;
	bytes[1] = g;
	bytes[2] = b;

	return rgba;
}

uint32_t FloatSpaceConvert::nrgb(double percent) {

	//produce a three bytes (rgb) max value
	constexpr uint32_t maxValue = UInt32Limits::max() >> 8;

	return maxValue * percent;
}

uint32_t FloatSpaceConvert::snrgb(double percent) {

	//produce a two bytes (rg) max value
	constexpr uint32_t maxValue = UInt32Limits::max() >> 16;

	return maxValue * percent;
}

uint32_t FloatSpaceConvert::roygbiv(double percent) {

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

uint32_t FloatSpaceConvert::grayScale(double percent) {

	constexpr std::uint8_t maxValue = UInt8Limits::max();
	uint8_t gray = maxValue * percent;
	return rgb(gray, gray, gray);
}

uint32_t FloatSpaceConvert::binary(double percent) {

	constexpr uint8_t maxValue = UInt8Limits::max();
	//perrcent is between 0 and 1 so round to 0 or 1 and multiply by max value for either 0 or 255
	uint8_t bit = std::round(percent);
	uint8_t gray = maxValue * bit;
	return rgb(gray, gray, gray);
}

void FloatSpaceConvert::floatSpaceConvert(DataView data, PixelView converted, ColorizeMode colorMode, double vMin, double vMax, double stripeNum) {

	auto dimensions = getDimensions(data.size(), 1.0f);

	floatSubSpaceConvert(data, converted, { {0,0}, dimensions }, dimensions.mWidth
		, colorMode, vMin, vMax, stripeNum);
}
FloatSpaceConvert::Dimensions FloatSpaceConvert::getDimensions(size_t size, float aspectRatio) {

	Dimensions dimensions;

	dimensions.mWidth = sqrt(size * aspectRatio);
	dimensions.mHeight = ceil(size / float(dimensions.mWidth));

	return dimensions;
}

void FloatSpaceConvert::colorizeFloatSpace(const std::string_view baseFileName, DataView floats) {

	println("Colorizing float space: {}", baseFileName);

	auto writeColorizedImages = [&](auto& image, auto width, auto height) {

		if (image.size() == 0) return;

		auto saveToBmpFile = [&](const string& fileName, PixelView image) {
			//image is int32 r g b a

			//freeimage is writing in bgra format depending if you are windows vs apple, check your free image file format
			auto bgra = [&](uint32_t rgba) {

				std::uint8_t* bytes = reinterpret_cast<uint8_t*>(&rgba);
				std::swap(bytes[0], bytes[2]);

				return rgba;
				};

			//given windows do this transform else consider free image file format
			transform(image.begin(), image.end(), image.begin(), bgra);

			uint8_t* bytes = reinterpret_cast<uint8_t*>(image.data());
			int pitch = width * (32 / 8);

			//converted data are four byte type (uint32)
			//consider byte order for free image write unless windows
			FIBITMAP* convertedImage = FreeImage_ConvertFromRawBits(bytes, width, height, pitch, 32, FI_RGBA_RED_MASK, FI_RGBA_GREEN_MASK, FI_RGBA_BLUE_MASK);

			FreeImage_Save(FIF_BMP, convertedImage, fileName.c_str(), 0);

			FreeImage_Unload(convertedImage);
			};

		ColorNames colorNames = getColorNames();

		auto stripes = array{ 1,2,10,100 };

		FreeImage_Initialise();

		for_each(execution::par, stripes.begin(), stripes.end(), [&](auto stripeNum) {

			vector<uint32_t> converted(width * height); //adjust to texture min size

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

void FloatSpaceConvert::floatSubSpaceConvert(DataView data, PixelView converted
	, const Rect& subFrame
	, size_t frameWidth
	, ColorizeMode colorMode, double vMin, double vMax, double stripeNum) {

	auto getViewWindow = [&](double startPercent = 0.0, double endPercent = 1.0) -> tuple<double, double, double> {

		auto minmax = std::minmax_element(execution::par_unseq, data.begin(), data.end());
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
			f -= stripeDistance * floor(f / stripeDistance);

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

	//x y w h are received in float space ( 0, dataWidth ) and ( 0, dataHeight )
	//converted may be larger than data and would have unwritten pixels

	auto forSubPixels = [&]() {

		auto& [x, y] = subFrame.mOrigin;
		auto& [w, h] = subFrame.mDimensions;

		auto hIota = views::iota(y, y + h);
		auto wIota = views::iota(x, x + w);

		for_each(execution::par, hIota.begin(), hIota.end(), [&](auto iy) {

			for (auto ix : wIota) {

				size_t index = iy * frameWidth + ix;
				size_t pxIndex = (iy - y) * w + (ix - x);

				if (pxIndex < converted.size() && index < data.size())
					converted[pxIndex] = convertMethod(data[index]);
			}

			});
		};
	
	forSubPixels();
}


FloatSpaceConvert::Rect FloatSpaceConvert::getFloatSpaceRect(float& x, float& y, float& scale, size_t frameWidth, size_t frameHeight) {

	if (scale < 1.0f) scale = 1.0f;

	float rScale = 1.0f / scale;

	x = clamp(x, -1.0f + rScale, 1.0f - rScale);
	y = clamp(y, -1.0f + rScale, 1.0f - rScale);

	float x1 = x - rScale, x2 = x + rScale
		, y1 = y + rScale, y2 = y - rScale;

	x1 = (x1 + 1.0f) / 2.0f;
	x2 = (x2 + 1.0f) / 2.0f;

	y1 = (-y1 + 1.0f) / 2.0f;
	y2 = (-y2 + 1.0f) / 2.0f;

	size_t px1 = floor(x1 * frameWidth)
		, px2 = ceil(x2 * frameWidth)
		, py1 = floor(y1 * frameHeight)
		, py2 = ceil(y2 * frameHeight);

	size_t pw = px2 - px1
		, ph = py2 - py1;

	return { {px1, py1}, {pw, ph} };
}
