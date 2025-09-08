#pragma once

#include "Algorithms.h"

#include "CpuNetwork.h"
#include "GpuNetwork.h"

namespace NetworkLib {

	namespace Model {

		using GpuSample = std::pair<Gpu::GpuView1, Gpu::GpuView1>;
		using Sample = std::pair<std::vector<float>, std::vector<float>>;
		using GpuSamples = std::vector<GpuSample>;
		using Samples = std::vector<Sample>;

		GpuSamples createGPUSamples(Gpu::FloatSpace1& gpuSampleSpace
			, const Samples& samples
			, std::size_t inputSize, std::size_t outputSize) {

			GpuSamples gpuSamples(samples.size());

			gpuSampleSpace.create(gpuSamples.size() * (inputSize + outputSize));
			auto begin = gpuSampleSpace.begin();
			for (auto& [seen, desired] : gpuSamples) {
				gpuSampleSpace.advance(seen, begin, inputSize);
				gpuSampleSpace.advance(desired, begin, outputSize);
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
		}

		GpuSamples createXORSamples(Gpu::FloatSpace1& gpuSampleSpace) {

			const Samples samples = {
				{{0,0}, {1,0}},
				{{0,1}, {0,1}},
				{{1,0}, {0,1}},
				{{1,1}, {1,0}},
			};

			return createGPUSamples(gpuSampleSpace, samples, 2, 2);
			};

		static void modelXOR() {

			std::mt19937 random;
			Gpu::Environment gpu;

			gpu.create();

			constexpr std::size_t inputSize = 2, outputSize = 2
				, trainNum = 5000;
			constexpr float learnRate = 0.002f;

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

			Gpu::FloatSpace1 sampleSpace;
			GpuSamples trainingSamples = createXORSamples(sampleSpace);
			sampleSpace.upload();

			auto calculateConvergence = [&]() {

				for (const auto& [seen, desired] : trainingSamples) {

					const auto& sought = gnn.forward(gpu, seen);
					sought.downloadAsync(gpu);
					gpu.sync();

					std::println("\nseen: {}"
						"\ndesired: {}"
						"\nsought: {}"
						, seen
						, desired
						, sought
					);
				}
				};

			calculateConvergence();

			TimeAverage<milliseconds> trainTime;

			std::print("Training: ");

			for (auto generation : std::views::iota(0ULL, trainNum))
				trainTime.accumulateTime([&]() {

					const auto& [seen, desired] = trainingSamples[generation % trainingSamples.size()];

					gnn.forward(gpu, seen);
					gnn.backward(gpu, seen, desired, learnRate);

					printProgress(generation, trainNum);

					});

			calculateConvergence();

			sampleSpace.destroy();

			gnn.destroy();

			gpu.destroy();
		}
	};

}