#pragma once

#include "CpuNetwork.h"

namespace NetworkLib {

	namespace Cpu {


		void exampleModel() {

			NetworkTemplate networkTemplate{ 748, LayerTemplates{ 10, 10, 1} };

			Network nn(&networkTemplate);

			nn.create();


		}
	}
}