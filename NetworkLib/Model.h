#pragma once

#include "CpuNetwork.h"
#include "GpuNetwork.h"

namespace NetworkLib {


	void exampleModel() {

		std::mt19937 random;

		NetworkTemplate networkTemplate{ 748, { 100, 50, 10} };

		Cpu::Network nn(&networkTemplate);

		nn.create();

		Gpu::Environment gpu;
		gpu.create();

		Gpu::Network gnn(&networkTemplate);
		gnn.create();

		gnn.initialize(random);
		gnn.upload();


		gnn.destroy();
		
		gpu.destroy();
	}
	
}