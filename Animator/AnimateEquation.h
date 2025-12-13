#pragma once

#include <execution>

#include "Animator.h"

class EquationAnimator : public Animator {
public:
	EquationAnimator() = default;
	
	//color z is determined by a function of x and y; z = f(x,y)
	using EquationFunction = std::function<float(float, float)>;

	void run() {

		using namespace std;

		size_t width = mWindowWidth, height = mWindowHeight;
		vector<float> floats(width * height);

		mFrameWidth = width;
		mFrameHeight = height;

		//to avoid division by zero in equations, replace zero with smallest float
		auto safeDenom = [&](float x) {
			float safe = (x == 0.0f) ? numeric_limits<float>::min() : x;
			return safe;
			};

		auto drawEquation = [&](EquationFunction&& equation) {

			float scale = 1.0f;

			size_t halfWidth = width / 2.0f, halfHeight = height / 2.0f;

			auto horizontalPixels = views::iota(0ULL, width);
			for_each(execution::par_unseq, horizontalPixels.begin(), horizontalPixels.end(), [&](auto px) {
					for (auto py : views::iota(0ULL, height)) {

						float x = scale* safeDenom(float(px) - halfWidth)
							, y = scale* safeDenom(float(py) - halfHeight);
						 
						floats[px + py * width] = equation(x, y);
					}
				});
			};

		auto equationCircle = [&](float x, float y) -> float {
			float r = sqrt(x * x + y * y);
			return r;
			};

		auto equationA = [&](float x, float y) -> float {
			float r = equationCircle(x,y);
			return sin(r) / safeDenom(r);
			};

		auto equationStandingWave = [&](float x, float y) -> float {
			//vertical line instead of horizontal
			return sin( 0.1f * x) * cos( 0.1f * y);
			};

		auto equationHyperbola = [&](float x, float y) -> float {
			return pow(y, 2) - pow(x, 2);
			};

		auto equationRippledSine = [&](float x, float y) -> float {
			return sin(x * x + y * y);
			};

		auto equationMonkeySaddle = [&](float x, float y) -> float {
			return x * x * x - 3.0f * x * y * y;
			};

		auto equationChaoticPeaksA = [&](float x, float y) -> float {
			return sin(x) * cos(y) + sin(y * 0.5f) * cos(x * 0.5f);
			};

		auto equationChaoticPeaksB = [&](float x, float y) -> float {
			return sin(x) + sin(sqrt(2.0f) * y) + sin(1.5f * x + 0.5f * y);
			};

		constexpr auto expScale = 1.0f / 10'000.0f;

		auto equationGaussianMountain = [&](float x, float y) -> float {
			float r2 = x * x + y * y;
			return exp(-r2 * expScale );
			};

		auto equationGaussianPeeksTwo = [&](float x, float y) -> float {
			float a = x * x + y * y;
			float b = pow(x - 1.0, 2.0) + pow(y + 1.0, 2.0);
			
			return exp(-a * expScale ) - exp( -b* expScale );
			};
		auto equationMexicanHat = [&](float x, float y) -> float {
			float r2 = x * x + y * y;
			return (1.0f - r2) * exp(-r2 * expScale / 2.0f);
			};

		auto equationMonkeySaddleHat = [&](float x, float y) -> float {

			float mh = equationMexicanHat(x,y);
			float ms = equationMonkeySaddle(x,y);
			
			return mh * ms;
			};

		drawEquation(equationMonkeySaddleHat);

		using ColorizeMode = FloatSpaceConvert::ColorizeMode;
		auto colorModes = array{
			ColorizeMode::BINARY
			, ColorizeMode::GREYSCALE
			, ColorizeMode::ROYGBIV
			, ColorizeMode::SHORTNRGB
			, ColorizeMode::NICKRGB
			};

		auto colorizeMode = colorModes.begin();
		mColorizeMode = *colorizeMode;

		setup(floats);

		size_t startStripes = 1, endStripes = 1000
			, currentStripes = startStripes
			, stripeStride = 3;

		mPaused = true;
		
		auto animateStripesAndColorModes = [&]() {

			mOptionalStripeNum = currentStripes;

			currentStripes += stripeStride;

			if (currentStripes >= endStripes) {
				currentStripes = startStripes;
				advance(colorizeMode, 1);
				if (colorizeMode == colorModes.end())
					colorizeMode = colorModes.begin();
				mColorizeMode = *colorizeMode;

				mOptionalStripeNum = currentStripes;
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