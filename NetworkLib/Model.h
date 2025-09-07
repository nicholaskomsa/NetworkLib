#pragma once

#include "Algorithms.h"

#include "CpuNetwork.h"
#include "GpuNetwork.h"

namespace NetworkLib {

	namespace Model {

		static void modelXOR() {

			std::mt19937 random;
			Gpu::Environment gpu;

			gpu.create();

			const auto inputSize = 2, outputSize = 2;
			const auto trainNum = 5000;
			const float learnRate = 0.002f;

			using ActivationFunction = LayerTemplate::ActivationFunction;
			NetworkTemplate networkTemplate = { inputSize
				, {{ 7, ActivationFunction::ReLU}
				, { 4, ActivationFunction::ReLU}
				, { outputSize, ActivationFunction::None}}
			};

			Gpu::Network gnn(&networkTemplate);
			gnn.create();

			gnn.initialize(random);
			gnn.upload();

			using GpuSample = std::pair<Gpu::GpuView1, Gpu::GpuView1>;
			using Sample = std::pair<std::vector<float>, std::vector<float>>;

			auto createSamples = [&](auto& sampleSpace, const auto& samples) {

				std::vector<GpuSample> gpuSamples(samples.size());

				sampleSpace.create(gpuSamples.size() * (inputSize + outputSize));
				auto begin = sampleSpace.begin();
				for (auto& [seen, desired] : gpuSamples) {
					sampleSpace.advance(seen, begin, inputSize);
					sampleSpace.advance(desired, begin, outputSize);
				}

				auto generateGPUSample = [&](const Sample& sample, GpuSample& gpuSample) {
					const auto& [seen, desired] = sample;
					auto& [gpuSeen, gpuDesired] = gpuSample;
					std::copy(seen.begin(), seen.end(), gpuSeen.begin());
					std::copy(desired.begin(), desired.end(), gpuDesired.begin());
					};

				for (const auto& [sample, gpuSample] : std::views::zip(samples, gpuSamples))
					generateGPUSample(sample, gpuSample);

				return gpuSamples;
				};

			auto createXORSamples = [&](auto& sampleSpace) {

				const std::vector<Sample> samples = {
					{{0,0}, {1,0}},
					{{0,1}, {0,1}},
					{{1,0}, {0,1}},
					{{1,1}, {1,0}},
				};

				return createSamples(sampleSpace, samples);
				};

			Gpu::FloatSpace1 sampleSpace;
			auto trainingSamples = createXORSamples(sampleSpace);
			sampleSpace.upload();

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

			for (auto generation : std::views::iota(0, trainNum))
				trainTime.accumulateTime([&]() {

				const auto& [seen, desired] = trainingSamples[generation % trainingSamples.size()];

				gnn.forward(gpu, seen);
				gnn.backward(gpu, seen, desired, learnRate);

				std::print(".");

					});

			calculateConvergence();

			sampleSpace.destroy();

			gnn.destroy();

			gpu.destroy();

		}
	};

}