#pragma once

#include <execution>

#include "Animator.h"

class EquationAnimator : public Animator {
public:
	EquationAnimator() = default;
	
	//color z is determined by a function of x and y; z = f(x,y)
	using EquationFunction = std::function<float(float, float)>;
	using FloatLimits = std::numeric_limits<float>;
	void run() {

		using namespace std;

		size_t width = mWindowWidth, height = mWindowHeight;
		vector<float> floats(width * height);

		mFrameWidth = width;
		mFrameHeight = height;

		//to avoid division by zero in equations, replace zero with smallest float
		auto safeDenom = [&](float x) {
			float safe = (x == 0.0f) ? FloatLimits::min() : x;
			return safe;
			};
		constexpr auto downScale = 1.0f / 5'000.0f;
		constexpr auto bigFloat = 100'000.0f;
		constexpr auto expScale = 1.0f / 10'000.0f;

		auto drawEquation = [&](EquationFunction&& equation, float scale = 1.0f) {

			auto halfWidth = width / 2.0f, halfHeight = height / 2.0f;

			auto horizontalPixels = views::iota(0ULL, width);
			for_each(execution::par_unseq, horizontalPixels.begin(), horizontalPixels.end(), [&](auto px) {
					for (auto py : views::iota(0ULL, height)) {

						float x = safeDenom(scale * (float(px) - halfWidth))
							, y = safeDenom(scale * (float(py) - halfHeight));
						 
						floats[px + py * width] = equation(x, y);
					}
				});

			auto handleInfs = [&]() {
				//replace +/- inf it will break FloatSpaceConvert
				//seach for the min and maxes excluding infs
				//replace infs with max or min rather than FLOAT max/lowest to minimize floatspace distortion during FSC
				auto max = *max_element(floats.begin(), floats.end(), [](auto a, auto b) {
					if( isfinite(a) && isfinite(b) )
						return a < b;

					return false;
					});

				auto min = *min_element(floats.begin(), floats.end(), [](auto a, auto b) {
					if (isfinite(a) && isfinite(b))
						return a < b;

					return false;
					});

				std::replace(floats.begin(), floats.end(), FloatLimits::infinity(), max);
				std::replace(floats.begin(), floats.end(), -FloatLimits::infinity(), min);

				};
			handleInfs();
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
 
		auto equationMexicanHatDiff = [&](float x, float y) -> float {

			float mexicanHats = equationMexicanHat(x, y) - equationMexicanHat(x, y + 1);
			return mexicanHats;
			};

		auto equationMexicanHatSky= [&](float x, float y) -> float {

			auto mh = equationMexicanHat(x, y);
			auto r = equationCircle(x+1, y-1);
			return mh * r;
			};
		auto equationMexicanHatSkyDifference = [&](float x, float y) -> float {

			auto mh = equationMexicanHat(x, y);
			auto mh2 = equationMexicanHat(x, y + 1);
			auto r = equationCircle(x + 1, y - 1);
			return (mh - mh2) * r;

			};


		auto equationLogarmithm = [&](float x, float y) -> float {
			float r = sqrt(x * x + y * y);
			return log(safeDenom(r*r));

			};
		auto equationLogarithmicSpiral = [&](float x, float y) -> float {

			float theta = atan2(x, y); //vertical line
			float r = sqrt(x * x + y * y);
			return log(safeDenom(r)) - theta;
			};
		auto equationLogarithmicSpiralWithWaves = [&](float x, float y) -> float {
			
			float r = sqrt(x * x + y * y);
			float theta = atan2(x, y); //vertical line

			return log(safeDenom(r)) - theta + sin(5.0f * log(safeDenom(r*r)));
			};

		auto equationIntensityOfLight = [&](float x, float y) -> float {
			float r = sqrt(x * x + y * y);
			constexpr auto intensity = FloatLimits::max() / bigFloat / 100;

			return intensity / safeDenom(r * r);
			};

		drawEquation(equationLogarithmicSpiralWithWaves);

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