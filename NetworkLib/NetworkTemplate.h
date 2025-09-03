#pragma once

#include <vector>
#include <functional>

namespace NetworkLib {

	struct LayerTemplate {
		std::size_t mNodeCount = 0;

		enum class ActivationFunction {
			None,
			ReLU,
			Sigmoid,
			Tanh,
			Softmax
		} mActivationFunction = ActivationFunction::ReLU;

		LayerTemplate(std::size_t nodeCount) : mNodeCount(nodeCount) {}
	};
	using LayerTemplates = std::vector<LayerTemplate>;

	struct NetworkTemplate {
		std::size_t mInputSize = 0;
		LayerTemplates mLayerTemplates;
	};
}