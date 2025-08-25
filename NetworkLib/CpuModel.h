#pragma once

#include "CpuNetwork.h"

namespace NetworkLib {

	namespace Cpu {


		void exampleModel() {

			LayersTemplate layersTemplate{ {748, 10}, {10, 10}, { 10, 1} };

			Network nn(&layersTemplate);

			nn.create();


		}
	}
}