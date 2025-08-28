#pragma once

#include "CpuTensor.h"
#include "NetworkTemplate.h"

namespace NetworkLib {

	namespace Cpu {

		class Network {

			NetworkTemplate* mNetworkTemplate = nullptr;

			Tensor::FloatSpace1 mFloats;
			
			struct Layer {
				Tensor::View2 mWeights;
				Tensor::View1 mBias;
			};

			std::vector<Layer> mLayers;

		public:

			Network() = default;

			Network(NetworkTemplate* networkTemplate)
			: mNetworkTemplate(networkTemplate) {}

			void create() {
				
				auto& networkTemplate = *mNetworkTemplate;
				std::span<LayerTemplate> layerTemplates = networkTemplate.mLayerTemplates;

				const auto firstInputSize = networkTemplate.mInputSize;
				std::size_t size = 0, inputSize = firstInputSize;
				for (auto [n] : layerTemplates ) {
					size += inputSize * n + n;
					inputSize = n;
				}
				
				mFloats.resize(size);

				mLayers.resize(layerTemplates.size());

				auto begin = mFloats.mFloats.begin();
				inputSize = firstInputSize;
				for (const auto& [layer, layerTemplate] : std::views::zip(mLayers, layerTemplates)) {

					const auto& [n] = layerTemplate;
					auto& [w, b] = layer;

					Tensor::advance(w, begin, inputSize, n);
					Tensor::advance(b, begin, n);

					inputSize = n;
				}
					
			}

		};
	};
}