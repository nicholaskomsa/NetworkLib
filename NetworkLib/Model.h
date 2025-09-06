#pragma once

#include "Algorithms.h"

#include "CpuNetwork.h"
#include "GpuNetwork.h"

namespace NetworkLib {

	static void modelMap() {

		std::mt19937 random;
		Gpu::Environment gpu;

		gpu.create();

		using ActivationFunction = LayerTemplate::ActivationFunction;
		NetworkTemplate networkTemplate = { 2
			, {{ 100, ActivationFunction::ReLU}
			, { 2, ActivationFunction::None}}
		};

		Gpu::Network gnn(&networkTemplate);
		gnn.create();

		gnn.initialize(random);
		gnn.upload();

		using GpuSample = std::pair<Gpu::GpuView1, Gpu::GpuView1>;
		Gpu::FloatSpace1 sampleSpace;

		auto createSamples = [&]() {

			std::vector<GpuSample> gpuSamples(2);

			sampleSpace.create(gpuSamples.size() * (2 + 2));
			auto begin = sampleSpace.begin();
			for (auto& [seen, desired] : gpuSamples) {
				sampleSpace.advance(seen, begin, 2);
				sampleSpace.advance(desired, begin, 2);
			}

			//values to map x, y = x2, y2 == {x ,y, x2, y2 }
			std::vector<float> sample1 = { 5, 5, 0.5, 0.5 }
			, sample2 = { 1, 7, 7, 1 };

			auto generateSample = [&](const std::vector<float>& sample, GpuSample& gpuSample) {

				std::span<const float> seen(sample.cbegin(), 2)
					, desired(sample.cbegin() + 2, 2);

				std::copy(seen.begin(), seen.end(), gpuSample.first.begin());
				std::copy(desired.begin(), desired.end(), gpuSample.second.begin());
				};

			generateSample(sample1, gpuSamples.front());
			generateSample(sample2, gpuSamples.back());

			return gpuSamples;
			};

		auto trainingSamples = createSamples();
		sampleSpace.mView.upload();

		gpu.sync();

		Gpu::GpuView1* gnnOutput = nullptr;

		for (auto generation : std::views::iota(0ULL, 1000ULL)) {

			TimeAverage<milliseconds> trainTime;
			trainTime.accumulateTime([&]() {

				auto& sample = trainingSamples[generation % trainingSamples.size()];
				gnnOutput = &gnn.forward(gpu, sample.first);
				gnn.backward(gpu, sample.first, sample.second, 0.0020f);

				gnnOutput->downloadAsync(gpu.getStream());
				gpu.sync();

				std::print("\nseen: ");
				for (auto f : sample.first)
					std::print("{}, ", f);
				std::print("\ndesired: ");
				for (auto f : sample.second)
					std::print("{}, ", f);
				std::print("\nsought: ");
				for (auto f : *gnnOutput)
					std::print("{}, ", f);


				});
		}

		sampleSpace.destroy();

		gnn.destroy();

		gpu.destroy();

	}

	static void exampleModel() {

		modelMap();
	}
	
}