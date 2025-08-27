#pragma once

#include <vector>

namespace NetworkLib {

	struct LayerTemplate {
		std::size_t mNodeCount = 0;

		LayerTemplate(std::size_t nodeCount) : mNodeCount(nodeCount) {}
	};
	using LayerTemplates = std::vector<LayerTemplate>;

	struct NetworkTemplate {
		std::size_t mInputSize = 0;
		LayerTemplates mLayerTemplates;
	};
}