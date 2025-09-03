#pragma once

#include "CpuNetwork.h"
#include "GpuNetwork.h"

namespace NetworkLib {


	static void exampleModel() {

		std::mt19937 random;
		Gpu::Environment gpu;
		//gpu.example();
		gpu.create();

		NetworkTemplate networkTemplate = { 784, {100, 50, 10} };

		Gpu::Network gnn(&networkTemplate);
		gnn.create();

		gnn.initialize(random);
		gnn.upload();

		Gpu::FloatSpace1 trainingData;
		trainingData.create(784);
		std::fill(trainingData.begin(), trainingData.end(), 1);

		gnn.forward(gpu, trainingData.mView);

		gnn.destroy();
		
		gpu.destroy();
		
	}
	
}