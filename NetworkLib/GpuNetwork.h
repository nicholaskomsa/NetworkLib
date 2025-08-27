#pragma once

#include "GpuTensor.h"

#include <random>

namespace NetworkLib {
	namespace Gpu {

		class Network {
		public:
			static void applyKHScale(std::size_t inputSize, std::size_t size, GpuView2 weights) {

				auto scale = std::sqrtf(6.0f / inputSize + size);

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
					size += inputSize * n + n;
					inputSize = n;
				}

				mGpuFloats.allocate(size);

				mLayers.resize(layerTemplates.size());

				auto begin = mGpuFloats.begin();
				inputSize = firstInputSize;
				for (const auto& [layer, layerTemplate] : std::views::zip(mLayers, layerTemplates)) {

					const auto& [n] = layerTemplate;
					auto& [w, b] = layer;

					mGpuFloats.advance(w, begin, inputSize, n);
					mGpuFloats.advance(b, begin, n);

					inputSize = n;
				}
			}
			void destroy() {
				mGpuFloats.free();
			}

			void initialize(std::mt19937& random) {

				std::uniform_real_distribution<float> reals(-1.0f, 1.0f);

				for (auto& [w, b] : mLayers) {

					std::generate(w.begin(), w.end(), [&]() {
						return reals(random);
						});

					std::generate(b.begin(), b.end(), [&]() {
						return reals(random);
						});
				}

				auto layersIota = std::views::iota(0ULL, mLayers.size());
				std::for_each(std::execution::par, layersIota.begin(), layersIota.end(), [&](auto l) {

					auto& layer = mLayers[l];

					std::size_t inputSize = (l == 0) ?
						mNetworkTemplate->mInputSize : mLayers[l - 1].mBias.mSize
						, size = layer.mBias.mSize;

					auto& w = layer.mWeights;

					applyKHScale(inputSize, size, w);

					});
			}

			void upload() {
				mGpuFloats.mView.upload();
			}
		private:
			NetworkTemplate* mNetworkTemplate = nullptr;

			FloatSpace1 mGpuFloats;

			struct Layer {
				GpuView<Cpu::Tensor::View2> mWeights;
				GpuView<Cpu::Tensor::View1> mBias;
			};

			std::vector<Layer> mLayers;
		};

	}
}