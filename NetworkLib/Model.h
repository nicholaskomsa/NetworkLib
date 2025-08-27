#pragma once

#include "CpuNetwork.h"
#include "GpuNetwork.h"

namespace NetworkLib {


	void exampleModel() {

		NetworkTemplate networkTemplate{ 748, { 10, 10, 1} };

		Cpu::Network nn(&networkTemplate);

		nn.create();

		Gpu::Environment env;
		env.create();

		Gpu::Network gnn(&networkTemplate);
		gnn.create();





		gnn.destroy();
		
		env.destroy();

	}
	
}