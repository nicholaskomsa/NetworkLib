#pragma once

#include "Algorithms.h"

#include "CpuNetwork.h"
#include "GpuNetwork.h"

namespace NetworkLib {

	static void modelMapXOR() {

		std::mt19937 random;
		Gpu::Environment gpu;

		gpu.create();
		
		const auto inputSize = 2, outputSize = 2;

		using ActivationFunction = LayerTemplate::ActivationFunction;
		NetworkTemplate networkTemplate = { inputSize
			, {{ 100, ActivationFunction::ReLU}
			, { outputSize, ActivationFunction::None}}
		};

		Gpu::Network gnn(&networkTemplate);
		gnn.create();

		gnn.initialize(random);
		gnn.upload();

		using GpuSample = std::pair<Gpu::GpuView1, Gpu::GpuView1>;
		Gpu::FloatSpace1 sampleSpace;

		auto createSamples = [&]() {

			using Sample = std::pair<std::vector<float>, std::vector<float>>;

			std::vector<Sample> samples = {
				  {{0.0f, 0.0f}, {1.0f, 0.0f}}
				, {{1.0f, 0.0f}, {0.0f, 1.0f}}
				, {{0.0f, 1.0f}, {0.0f, 1.0f}}
				, {{1.0f, 1.0f}, {1.0f, 0.0f}}
			};

			std::vector<GpuSample> gpuSamples(samples.size());

			sampleSpace.create(gpuSamples.size() * (inputSize + outputSize));
			auto begin = sampleSpace.begin();
			for (auto& [seen, desired] : gpuSamples) {
				sampleSpace.advance(seen, begin, inputSize);
				sampleSpace.advance(desired, begin, outputSize);
			}

			auto generateSample = [&](const Sample& sample, GpuSample& gpuSample) {

				const auto& [seen, desired] = sample;

				std::copy(seen.begin(), seen.end(), gpuSample.first.begin());
				std::copy(desired.begin(), desired.end(), gpuSample.second.begin());
				};

			for (const auto& [sample, gpuSample] : std::views::zip(samples, gpuSamples))
				generateSample(sample, gpuSample);

			return gpuSamples;
			};



		auto trainingSamples = createSamples();
		sampleSpace.mView.upload();

		auto calculateConvergence = [&]() {

			for (const auto& [seen, desired] : trainingSamples) {

				const auto& sought = gnn.forward(gpu, seen);
				sought.downloadAsync(gpu);
				gpu.sync();

				std::print("\nseen: {}"
					"\ndesired: {}"
					"\nsought: {}"
					, seen
					, desired
					, sought
				);
			}
			};

		gpu.sync();

		calculateConvergence();

		TimeAverage<milliseconds> trainTime;

		for (auto generation : std::views::iota(0ULL, 2000ULL)) 
			trainTime.accumulateTime([&]() {

				const auto& [seen, desired] = trainingSamples[generation % trainingSamples.size()];

				gnn.forward(gpu, seen);
				gnn.backward(gpu, seen, desired, 0.002f);

				std::print(".");
					
				});
		
		calculateConvergence();

		sampleSpace.destroy();

		gnn.destroy();

		gpu.destroy();

	}

	static void exampleModel() {

		modelMap();
	}
	
}