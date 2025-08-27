#pragma once

#include "CpuTensor.h"

namespace NetworkLib {

	namespace Cpu {

		struct LayerTemplate {
			std::size_t mNodeCount = 0;

			LayerTemplate(std::size_t nodeCount) : mNodeCount(nodeCount) {}
		};
		using LayerTemplates = std::vector<LayerTemplate>;

		struct NetworkTemplate {
			std::size_t mInputSize = 0;
			LayerTemplates mLayerTemplates;
		};
		
		class Network {

			NetworkTemplate* mNetworkTemplate = nullptr;

			Tensor::FloatSpace1 mFloats;
			
			struct Layer {
				Tensor::View2 mWeights;
				Tensor::View1 mBias;
			};

			Tensor::FloatSpace1 mWeightsAndBias;
			std::vector<Layer> mLayers;

		public:

			Network() = default;

			Network(NetworkTemplate* networkTemplate)
			: mNetworkTemplate(networkTemplate) {}

			void create() {
				
				auto& networkTemplate = *mNetworkTemplate;
				std::span<LayerTemplate> layerTemplates = networkTemplate.mLayerTemplates;

				mLayers.resize(layerTemplates.size());

				std::size_t size = 0, inputSize = networkTemplate.mInputSize;
				for (auto [n] : layerTemplates ) {
					size += inputSize * n + n;
					inputSize = n;
				}
				
				mFloats.resize(size);
				auto begin = mFloats.mFloats.begin();
				inputSize = networkTemplate.mInputSize;

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