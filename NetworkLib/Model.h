#pragma once

#include "CpuNetwork.h"
#include "GpuNetwork.h"

namespace NetworkLib {


	void exampleModel() {

		Gpu::Environment gpu;
		gpu.example();

		/*
		gpu.create();

		Gpu::Network gnn(&networkTemplate);
		gnn.create();

		gnn.initialize(random);
		gnn.upload();

		gnn.destroy();
		
		gpu.destroy();
		*/
	}
	
}