#pragma once

#include "GpuTensor.h"

namespace NetworkLib {
	namespace Gpu {

		class Network {

			NetworkTemplate* mNetworkTemplate = nullptr;

			FloatSpace1 mGpuFloats;

			struct Layer {
				GpuView<Cpu::Tensor::View2> mWeights;
				GpuView<Cpu::Tensor::View1> mBias;
			};

			std::vector<Layer> mLayers;

		public:

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

		};

	}
}