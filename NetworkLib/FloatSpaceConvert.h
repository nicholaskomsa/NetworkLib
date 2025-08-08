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
 
	void setOpaque(std::uint32_t& p);
	std::uint32_t rgb(std::uint8_t r, std::uint8_t g, std::uint8_t b);

	std::uint32_t nrgb(double percent);
	std::uint32_t snrgb(double percent);
	std::uint32_t roygbiv(double percent);
	std::uint32_t grayScale(double percent);
	std::uint32_t binary(double percent);
	
	struct Coord {
		std::size_t mX=0, mY=0;
		bool operator==(const Coord& d) const = default;
		auto operator<=>(const Coord& d) const = default;
	};
	struct Dimensions {
		std::size_t mWidth=0, mHeight=0;
		bool operator==(const Dimensions& d) const = default;
		auto operator<=>(const Dimensions& d) const = default;
	};
	struct Rect {
		Coord mOrigin;
		Dimensions mDimensions;
		bool operator==(const Rect& d) const = default;
		auto operator<=>(const Rect& d) const = default;
	};

	using DataView = std::span<const float>;
	using PixelView = std::span<std::uint32_t>;

	void floatSpaceConvert(DataView data, PixelView converted, ColorizeMode colorMode = ColorizeMode::NICKRGB, double vMin = 0.0, double vMax = 1.0, double stripeNum = 1);
	void floatSubSpaceConvert(DataView data, PixelView converted
		, const Rect& subFrame
		, std::size_t frameWidth, ColorizeMode colorMode = ColorizeMode::NICKRGB, double vMin = 0.0, double vMax = 1.0, double stripeNum = 1);
	
	Dimensions getDimensions(std::size_t size, float aspectRatio = 3840.0f / 2160.0f);

	void colorizeFloatSpace(const std::string_view baseFileName, DataView floats);

	ColorNames getColorNames();


	Rect getFloatSpaceRect(float& x, float& y, float& scale, std::size_t frameWidth, std::size_t frameHeight);

};	