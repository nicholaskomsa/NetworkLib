#pragma once

#include "CpuTensor.h"

namespace NetworkLib {

	namespace Cpu {


		struct LayerTemplate {

			std::size_t mInputSize = 0, mNodeCount = 0;
		};
		using LayersTemplate = std::vector<LayerTemplate>;

		class Network {

			LayersTemplate* mLayerTemplates = nullptr;

			Tensor::FloatSpace1 mFloats;
			
			struct Layer {
				Tensor::View2 mWeights;
				Tensor::View1 mBias;

			};

			Tensor::FloatSpace1 mWeightsAndBias;
			std::vector<Layer> mLayers;

		public:

			Network() = default;

			Network(LayersTemplate* layersTemplate)
			: mLayerTemplates(layersTemplate) {}

			void create() {
				
				auto& layerTemplates = *mLayerTemplates;

				mLayers.resize(layerTemplates.size());

				std::size_t size = 0;
				for (auto& [i, n] : layerTemplates)
					size += i * n + n;

				mFloats.resize(size);
				auto begin = mFloats.mFloats.begin();

				for (const auto& [layer, layerTemplate] : std::views::zip(mLayers, layerTemplates)) {

					const auto& [i, n] = layerTemplate;
					auto& [w, b] = layer;

					Tensor::advance(w, begin, i, n);
					Tensor::advance(b, begin, n);
				}
					

			}


		};
	};
}