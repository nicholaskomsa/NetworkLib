#pragma once

#include <execution>

#include "Animator.h"

class EquationAnimator : public Animator {
public:
	EquationAnimator() = default;
	
	//color z is determined by a function of x and y; z = f(x,y)
	using EquationFunction = std::function<float(float, float)>;

	void run() {

		std::size_t width = mWindowWidth, height = mWindowHeight;
		std::vector<float> floats(width * height);

		mFrameWidth = width;
		mFrameHeight = height;

		//to avoid division by zero in equations, replace zero with smallest float
		auto safeDenom = [&](float x) {
			float safe = (x == 0.0f) ? std::numeric_limits<float>::min() : x;
			return safe;
			};

		auto drawEquation = [&](EquationFunction&& equation) {

			float scale = 1.0f;

			std::size_t halfWidth = width / 2.0f, halfHeight = height / 2.0f;

			auto horizontalPixels = std::views::iota(0ULL, width);
			std::for_each(std::execution::par_unseq, horizontalPixels.begin(), horizontalPixels.end(), [&](auto px) {
					for (auto py : std::views::iota(0ULL, height)) {

						float x = scale* safeDenom(float(px) - halfWidth);
						float y = scale* safeDenom(float(py) - halfHeight);
						 
						floats[px + py * width] = equation(x, y);
					}
				});
			};

		auto equationA = [&](float x, float y) -> float {
			float r = std::sqrt(x * x + y * y);
			return std::sin(r) / safeDenom(r);
			};

		auto equationB = [&](float x, float y) -> float {
			return std::sin(0.1f * x) * std::cos(0.1f * y);
			};

		auto equationCircle = [&](float x, float y) -> float {
			float r = std::sqrt(x * x + y * y);
			return r;
			};

		auto equationC = [&](float x, float y) -> float {
			return std::pow(y, 2) - std::pow(x, 2);
			};

		drawEquation(equationA);

		std::array<FloatSpaceConvert::ColorizeMode, 5> colorModes = {
			FloatSpaceConvert::ColorizeMode::BINARY
			, FloatSpaceConvert::ColorizeMode::GREYSCALE
			, FloatSpaceConvert::ColorizeMode::ROYGBIV
			, FloatSpaceConvert::ColorizeMode::SHORTNRGB
			, FloatSpaceConvert::ColorizeMode::NICKRGB
			};

		auto colorizeMode = colorModes.begin();
		mColorizeMode = *colorizeMode;

		setup(floats);

		std::size_t startStripes = 1, endStripes = 1000
			, currentStripes = startStripes
			, stripeStride = 3;

		mPaused = true;
		
		auto animateStripesAndColorModes = [&]() {

			mOptionalStripeNum = currentStripes;

			currentStripes += stripeStride;

			if (currentStripes >= endStripes) {
				currentStripes = startStripes;
				std::advance(colorizeMode, 1);
				//mPaused = true;
				if (colorizeMode == colorModes.end())
					colorizeMode = colorModes.begin();
				mColorizeMode = *colorizeMode;

				mOptionalStripeNum = currentStripes;
				floatSpaceConvert();
			}
			};

		auto step = [&](auto floats) {

			animateStripesAndColorModes();
			
			return true;
			};

		Animator::run(step);

		shutdown();
	}
};