#pragma once

#include "GpuTensor.h"

namespace NetworkLib {

	class TrainingManager {
	public:
		using GpuSample = std::pair<Gpu::GpuView1, Gpu::GpuView1>;
		using Sample = std::pair<std::vector<float>, std::vector<float>>;
		using Samples = std::vector<Sample>;

		using GpuBatchedSample = std::pair<Gpu::GpuView2, Gpu::GpuView2>;
		using GpuBatchedSamples = std::vector<GpuBatchedSample>;

		void destroy() {
			mXORSampleSpace.destroy();
		}

		GpuBatchedSamples createGPUBatchedSamples(Gpu::FloatSpace1& gpuSampleSpace
			, const Samples& samples
			, std::size_t inputSize, std::size_t outputSize, std::size_t batchSize = 1) {

			GpuBatchedSamples gpuBatchedSamples;

			auto batchNum = std::ceil(samples.size() / float(batchSize));

			gpuBatchedSamples.resize(batchNum);

			gpuSampleSpace.create(batchNum * batchSize * (inputSize + outputSize));
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
			
			gpuSampleSpace.upload();

			return gpuBatchedSamples;
		}

		GpuBatchedSamples createXORBatchedSamples( std::size_t batchSize) {

			const Samples samples = {
				{{0,0}, {1,0}},
				{{0,1}, {0,1}},
				{{1,0}, {0,1}},
				{{1,1}, {1,0}},
			};
			auto inputSize = samples.front().first.size()
				, outputSize = samples.front().second.size();
			
			return createGPUBatchedSamples(mXORSampleSpace, samples, inputSize, outputSize, batchSize);
		}

		Gpu::FloatSpace1 mXORSampleSpace;
	};
	
}