#pragma once

#include <random>

#include "Parallel.h"
#include "GpuTensor.h"
#include "NetworkTemplate.h"

namespace NetworkLib {
	namespace Gpu {

		class Network {
		public:
			static void applyKHScaleUniform(std::size_t inputSize, std::size_t size, GpuView2 weights) {

				auto scale = std::sqrtf(6.0f / (inputSize + size));

				for (auto& w : weights)
					w *= scale;
			}

			Network(NetworkTemplate* networkTemplate)
				: mNetworkTemplate(networkTemplate) {
			}

			void create(bool backwards=true) {

				auto& networkTemplate = *mNetworkTemplate;
				std::span<LayerTemplate> layerTemplates = networkTemplate.mLayerTemplates;

				auto& firstInputSize = networkTemplate.mInputSize;
				std::size_t size = 0, inputSize = firstInputSize;
				for (auto& layerTemplate : layerTemplates) {
					const auto n = layerTemplate.mNodeCount;
					//weights + bias + outputs + activations
					size += inputSize * n + n * 3;
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
					auto& [w, b, o, a, af, p] = layer;

					mGpuFloats.advance(w, begin, n, inputSize);
					mGpuFloats.advance(b, begin, n);
					mGpuFloats.advance(o, begin, n);
					mGpuFloats.advance(a, begin, n);

					af = layerTemplate.mActivationFunction;

					if (backwards) 
						mGpuFloats.advance(p, begin, n);

					inputSize = n;
				}
			}
			void destroy() {
				mGpuFloats.destroy();
			}

			void initialize(std::mt19937& random) {

				std::uniform_real_distribution<float> reals(-1.0f, 1.0f);

				for (auto& layer : mLayers) {

					auto& w = layer.mWeights;
					auto& b = layer.mBias;

					std::generate(w.begin(), w.end(), [&]() {
						return reals(random);
						});

					std::generate(b.begin(), b.end(), [&]() {
						return reals(random);
						});
				}

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

						std::size_t inputSize
							, size = layer.mBias.mSize;

						if (l == 0)
							inputSize = mNetworkTemplate->mInputSize;
						else
							inputSize = mLayers[l - 1].mBias.mSize;

						auto& w = layer.mWeights;

						applyKHScaleUniform(inputSize, size, w);
					}
					});
			}

			GpuView1& forward(Environment& env, GpuView1& seen) {
				
				GpuView1* i = &seen;

				for( auto& layer : mLayers ) 
					i = &layer.forward(env, *i);
				
				return *i;
			}
			void backward(Environment& env, GpuView1& seen, GpuView1& desired, float learnRate) {
;
				auto& back = mLayers.back();

				auto af = back.mActivationFunction;

				auto& sought =(af == LayerTemplate::ActivationFunction::None)? 
					back.mOutputs: back.mActivations;
				auto& p1 = back.mPrimes;
				
				//output layer makes a comparison between desired and sought
				env.errorFunction(af, desired, sought, p1);

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

				for (auto l : std::views::iota(0ULL, mLayers.size())) {

					auto& layer = mLayers[l];

					GpuView1* input = nullptr;
					if( l == 0)
						input = &seen;
					else
						input = &mLayers[l - 1].mActivations;

					env.updateWeights(env, *input, layer.mWeights, layer.mPrimes, learnRate);
				}
			}
	
			struct Layer {

				GpuView1& forward(Environment& env, const GpuView1& input) {

					env.matMulVec(mWeights, input, mOutputs);
					//env.vecAddVec(mBias, mOutputs);
					if (env.activationFunction(mActivationFunction, mOutputs, mActivations))
						return mActivations;
					else
						return mOutputs;
				}

				GpuView<Cpu::Tensor::View2> mWeights;
				GpuView<Cpu::Tensor::View1> mBias;
				GpuView<Cpu::Tensor::View1> mOutputs;
				GpuView<Cpu::Tensor::View1> mActivations;
				LayerTemplate::ActivationFunction mActivationFunction = LayerTemplate::ActivationFunction::None;


				GpuView<Cpu::Tensor::View1> mPrimes;


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