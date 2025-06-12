#pragma once

#include <string>
#include <span>
#include <map>
#include <functional>

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
	using ColorizeMethod = std::function<std::uint32_t(double)>;
	using ConvertFunction = std::function<void(std::size_t)>;
	using SpaceFunction = std::function<void(ConvertFunction&&)>;

	void setOpaque(std::uint32_t& p);
	std::uint32_t rgb(std::uint8_t r, std::uint8_t g, std::uint8_t b);

	std::uint32_t nrgb(double percent);
	std::uint32_t snrgb(double percent);
	std::uint32_t roygbiv(double percent);
	std::uint32_t grayScale(double percent);
	std::uint32_t binary(double percent);

	void floatSpaceConvert(std::span<const float> data, std::span<uint32_t> converted, ColorizeMode colorMode = ColorizeMode::NICKRGB, double vMin = 0.0, double vMax = 1.0, double stripeNum = 1);
	void floatSubSpaceConvert(std::span<const float> data, std::span<uint32_t> converted, std::size_t x, std::size_t y, std::size_t w, std::size_t h, std::size_t texW, ColorizeMode colorMode = ColorizeMode::NICKRGB, double vMin = 0.0, double vMax = 1.0, double stripeNum = 1);
	
	std::pair<std::size_t, std::size_t> getDimensions(std::size_t size, float aspectRatio = 3840.0f / 2160.0f);

	void colorizeFloatSpace(const std::string& baseFileName, std::span<const float> floats);

	ColorNames getColorNames();

};	