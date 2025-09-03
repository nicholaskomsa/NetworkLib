#pragma once

#include <random>

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

			void create() {

				auto& networkTemplate = *mNetworkTemplate;
				std::span<LayerTemplate> layerTemplates = networkTemplate.mLayerTemplates;

				auto& firstInputSize = networkTemplate.mInputSize;
				std::size_t size = 0, inputSize = firstInputSize;
				for (auto [n] : layerTemplates) {
					size += inputSize * n + n * 3;
					inputSize = n;
				}

				mGpuFloats.create(size);

				mLayers.resize(layerTemplates.size());

				auto begin = mGpuFloats.begin();
				inputSize = firstInputSize;
				for (const auto& [layer, layerTemplate] : std::views::zip(mLayers, layerTemplates)) {

					const auto& [n] = layerTemplate;
					auto& [w, b, o, a] = layer;

					mGpuFloats.advance(w, begin, inputSize, n);
					mGpuFloats.advance(b, begin, n);
					mGpuFloats.advance(o, begin, n);
					mGpuFloats.advance(a, begin, n);

					inputSize = n;
				}
			}
			void destroy() {
				mGpuFloats.destroy();
			}

			void initialize(std::mt19937& random) {

				std::uniform_real_distribution<float> reals(-1.0f, 1.0f);

				for (auto& [w, b, o, a] : mLayers) {

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
				auto layersIota = std::views::iota(0ULL, mLayers.size());
				std::for_each(std::execution::par, layersIota.begin(), layersIota.end(), [&](auto l) {

					auto& layer = mLayers[l];

					std::size_t inputSize
						, size = layer.mBias.mSize;

					if (l == 0)
						inputSize = mNetworkTemplate->mInputSize;
					else
						inputSize = mLayers[l - 1].mBias.mSize;

					auto& w = layer.mWeights;

					applyKHScaleUniform(inputSize, size, w);

					});
			}

			void forward(Environment& env, const GpuView1& input) {
				
				const GpuView1* i = &input;

				for( auto& layer : mLayers) {
					
					auto& o = layer.mOutputs;
					auto& w = layer.mWeights;
					auto& b = layer.mBias;
					auto& a = layer.mActivations;

					env.matMulVec(w, *i, o);
					env.vecAddVec(b, o);
					env.relu(o, a);

					i = &a;
				}
			}

		private:
			NetworkTemplate* mNetworkTemplate = nullptr;

			FloatSpace1 mGpuFloats;

			struct Layer {
				GpuView<Cpu::Tensor::View2> mWeights;
				GpuView<Cpu::Tensor::View1> mBias;
				GpuView<Cpu::Tensor::View1> mOutputs;
				GpuView<Cpu::Tensor::View1> mActivations;
			};

			std::vector<Layer> mLayers;
		};

	}
}