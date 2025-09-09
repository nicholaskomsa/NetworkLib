#pragma once

#include <random>

#include "Parallel.h"
#include "GpuTensor.h"
#include "NetworkTemplate.h"

namespace NetworkLib {
	namespace Gpu {

		class Network {
		public:

			Network(NetworkTemplate* networkTemplate)
				: mNetworkTemplate(networkTemplate) {
			}

			void create(bool backwards=true) {

				auto& networkTemplate = *mNetworkTemplate;
				auto& layerTemplates = networkTemplate.mLayerTemplates;

				auto& firstInputSize = networkTemplate.mInputSize;
				std::size_t size = 0, inputSize = firstInputSize;
				for (auto& layerTemplate : layerTemplates) {
					const auto n = layerTemplate.mNodeCount;
					//weights + bias + outputs + activations
					size += n * inputSize + n * 3;
					inputSize = n;
				}

				if (backwards)
					for (auto& layerTemplate : layerTemplates) {
						const auto n = layerTemplate.mNodeCount;
						size += n; //primes
					}

				mGpuFloats.create(size);

				mLayers.resize(layerTemplates.size());

				auto begin = mGpuFloats.begin();
				inputSize = firstInputSize;
				for (const auto& [layer, layerTemplate] : std::views::zip(mLayers, layerTemplates)) {

					const auto n = layerTemplate.mNodeCount;
					
					layer.advance(mGpuFloats, begin, n, inputSize, layerTemplate.mActivationFunction, backwards);

					inputSize = n;
				}
			}
			void destroy() {
				mGpuFloats.destroy();
			}

			void initialize(std::mt19937& random) {

				for (auto& layer : mLayers) 
					layer.generate(random);
				
				applyKHScales();
			}

			void upload() {
				mGpuFloats.mView.upload();
			}

			void applyKHScales() {

				Parallel parallel(mLayers.size());
				parallel([&](auto& section) {

					for (auto l : section.mIotaView) {

						auto& layer = mLayers[l];

						std::size_t inputSize;

						if (l == 0)
							inputSize = mNetworkTemplate->mInputSize;
						else
							inputSize = mLayers[l - 1].mBias.mSize;

						auto& w = layer.mWeights;

						layer.applyKHScaleUniform(inputSize);
					}
					});
			}

			const GpuView1& forward(Environment& env, const GpuView1& seen) {
				
				const GpuView1* i = &seen;

				for( auto& layer : mLayers ) 
					i = &layer.forward(env, *i);
				
				return *i;
			}

			void backward(Environment& env, const GpuView1& seen, const GpuView1& desired, float learnRate) {

				auto backLayer = [&]() {
					auto& back = mLayers.back();

					auto af = back.mActivationFunction;

					const auto& sought = back.mActivations;
					auto& p1 = back.mPrimes;

					//output layer makes a comparison between desired and sought
					env.errorFunction(af, desired, sought, p1);
					};

				auto hiddenLayers = [&]() {

					for (auto l : std::views::iota(0ULL, mLayers.size() - 1) | std::views::reverse) {

						auto& layer = mLayers[l];
						auto& nextLayer = mLayers[l + 1];

						auto& p1 = layer.mPrimes;
						auto& o1 = layer.mOutputs;
						auto& np1 = nextLayer.mPrimes;
						auto& nw2 = nextLayer.mWeights;

						auto af = layer.mActivationFunction;

						env.matTMulVec(nw2, np1, p1);
						env.activationFunctionPrime(af, o1, p1);
					}
					};

				auto updateWeights = [&]() {
					for (auto l : std::views::iota(0ULL, mLayers.size())) {

						auto& layer = mLayers[l];

						const GpuView1* input = nullptr;
						if (l == 0)
							input = &seen;
						else
							input = &mLayers[l - 1].mActivations;

						env.updateWeights(*input, layer.mWeights, layer.mPrimes, learnRate);
					}
					};

				backLayer();
				hiddenLayers();
				updateWeights();
			}
			const GpuView1& getSought() {
				return mLayers.back().mActivations;
			}
			class Layer {
			public:
				void generate(std::mt19937& random) {

					std::uniform_real_distribution<float> reals(-1.0f, 1.0f);

					std::generate(mWeights.begin(), mWeights.end(), [&]() {
						return reals(random);
						});

					std::generate(mBias.begin(), mBias.end(), [&]() {
						return reals(random);
						});
				}

				void applyKHScaleUniform(std::size_t inputSize) {

					auto scale = std::sqrtf(6.0f / (inputSize + mBias.mSize));

					for (auto& w : mWeights)
						w *= scale;
				}

				GpuView1& forward(Environment& env, const GpuView1& input) {

					env.matMulVec(mWeights, input, mOutputs);
					env.vecAddVec(mBias, mOutputs);
					env.activationFunction(mActivationFunction, mOutputs, mActivations);

					return mActivations;
				}

				void advance(FloatSpace1& gpuFloats, float*& begin
					, std::size_t n, std::size_t inputSize
					, LayerTemplate::ActivationFunction af, bool backwards) {
					
					gpuFloats.advance(mWeights, begin, n, inputSize);
					gpuFloats.advance(mBias, begin, n);
					gpuFloats.advance(mOutputs, begin, n);
					gpuFloats.advance(mActivations, begin, n);

					mActivationFunction = af;

					if (backwards)
						gpuFloats.advance(mPrimes, begin, n);
				}
				GpuView2 mWeights;
				GpuView1 mBias,mOutputs,mActivations, mPrimes;
				LayerTemplate::ActivationFunction mActivationFunction = LayerTemplate::ActivationFunction::None;

			};

			using Layers = std::vector<Layer>;

			Layer& getLayer(std::size_t i) {
				return mLayers[i];
			}

		private:
			NetworkTemplate* mNetworkTemplate = nullptr;

			FloatSpace1 mGpuFloats;


			Layers mLayers;
		};

	}
}