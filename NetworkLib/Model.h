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

		using GpuBatchedSample = std::pair<Gpu::GpuView2, Gpu::GpuView2>;
		using GpuBatchedSamples = std::vector<GpuBatchedSample>;

		GpuBatchedSamples createGPUBatchedSamples(Gpu::FloatSpace1& gpuSampleSpace
			, const Samples& samples
			, std::size_t inputSize, std::size_t outputSize, std::size_t batchSize = 1) {

			GpuBatchedSamples gpuBatchedSamples;

			auto batchNum = std::ceil(samples.size() / float(batchSize));

			gpuBatchedSamples.resize(batchNum);

			gpuSampleSpace.create( batchNum * batchSize * (inputSize + outputSize));
			auto begin = gpuSampleSpace.begin();
			auto currentSample = samples.begin();

			for (auto batch : std::views::iota(0ULL, batchNum)) {

				auto& [seenBatch, desiredBatch] = gpuBatchedSamples[batch];

				gpuSampleSpace.advance(seenBatch, begin, inputSize, batchSize);
				gpuSampleSpace.advance(desiredBatch, begin, outputSize, batchSize);

				for (auto b : std::views::iota(0ULL, batchSize)) {

					if (currentSample == samples.end()) break;
					auto& [seen, desired] = *currentSample;
					++currentSample;

					auto gpuSeen = seenBatch.viewColumn(b);
					auto gpuDesired = desiredBatch.viewColumn(b);

					std::copy(seen.begin(), seen.end(), gpuSeen.begin());
					std::copy(desired.begin(), desired.end(), gpuDesired.begin());
				}
			}

			return gpuBatchedSamples;
		}

		GpuBatchedSamples createXORBatchedSamples(Gpu::FloatSpace1& gpuSampleSpace, std::size_t batchSize) {

			const Samples samples = {
				{{0,0}, {1,0}},
				{{0,1}, {0,1}},
				{{1,0}, {0,1}},
				{{1,1}, {1,0}},
			};
			auto inputSize = samples.front().first.size()
				, outputSize = samples.front().second.size();
			return createGPUBatchedSamples(gpuSampleSpace, samples, inputSize, outputSize, batchSize);
		}
		static void modelXOR() {

			std::mt19937_64 random;
			Gpu::Environment gpu;

			gpu.create();

			constexpr std::size_t inputSize = 2, outputSize = 2
				, trainNum = 5000;

			std::size_t batchSize = 4;
			float learnRate = 0.002f;

			using ActivationFunction = LayerTemplate::ActivationFunction;
			NetworkTemplate networkTemplate = { inputSize, batchSize
				, {{ 7, ActivationFunction::ReLU}
				, { 4, ActivationFunction::ReLU}
				, { outputSize, ActivationFunction::None}}
			};

			Gpu::Network gnn(&networkTemplate);
			gnn.create();

			gnn.initialize(random);
			gnn.upload();

			Gpu::FloatSpace1 sampleSpace;
			GpuBatchedSamples trainingBatchedSamples = createXORBatchedSamples(sampleSpace, batchSize);

			sampleSpace.upload();

			auto calculateConvergence = [&]() {

				for (const auto& [seen, desired] : trainingBatchedSamples) {

					auto sought = gnn.forward(gpu, seen);
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

					const auto& [seen, desired] = trainingBatchedSamples[generation % trainingBatchedSamples.size()];

					gnn.forward(gpu, seen);
					gnn.backward(gpu, seen, desired, learnRate);

					printProgress(generation, trainNum);

					});

			calculateConvergence();
			std::println("\nTraining Took: {}", trainTime.getString());



			sampleSpace.destroy();
			gnn.destroy();
			gpu.destroy();
		}

		static void modelMNIST() {

		}
	};

}