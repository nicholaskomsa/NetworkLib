#pragma once

#include "Algorithms.h"

#include "CpuNetwork.h"
#include "GpuNetwork.h"

namespace NetworkLib {


	static void exampleModel() {

		std::mt19937 random;
		Gpu::Environment gpu;
		//gpu.example();
	//	return;
		gpu.create();

		using ActivationFunction = LayerTemplate::ActivationFunction;
		//NetworkTemplate networkTemplate = { 784
		////	, {{100, ActivationFunction::ReLU}
		//	, { 50, ActivationFunction::ReLU}
		//	, { 10, ActivationFunction::SoftmaxCrossEntropy}}
		//};
		NetworkTemplate networkTemplate = { 2
			, {{ 100, ActivationFunction::ReLU}
			, { 2, ActivationFunction::None}}
		};

		Gpu::Network gnn(&networkTemplate);
		gnn.create();

		gnn.initialize(random);
		gnn.upload();

		using GpuSample = std::pair<Gpu::GpuView1, Gpu::GpuView1>;
		auto createXORSamples = [&]() {
			//an xor dataset ( 4 samples ) + off/on one hot for each sample
			std::vector<GpuSample> trainingSamples(4);

			Gpu::FloatSpace1 trainingDataFloats;
			trainingDataFloats.create(trainingSamples.size() * (2 + 2) );
			auto begin = trainingDataFloats.begin();
			for (auto& [seen, desired] : trainingSamples) {
				trainingDataFloats.advance(seen, begin, 2);
				trainingDataFloats.advance(desired, begin, 2);
			}
			constexpr auto low = 0.001f
				, high = 1.0f;

			std::vector<float> sample1 = { low,low, 1,0 }
				, sample2 = { low,high, 0,1 }
				, sample3 = { high,low, 0,1 }
				, sample4 = { high,high, 1,0 };

			auto generateSample = [&](const std::vector<float>& sample, GpuSample& gpuSample) {
				
				std::span<const float> seen(sample.cbegin(), 2)
					, desired(sample.cbegin()+2, 2);

				std::copy(seen.begin(), seen.end(), gpuSample.first.begin());
				std::copy(desired.begin(), desired.end(), gpuSample.second.begin());
				};

			generateSample(sample1, trainingSamples[0]);
			generateSample(sample2, trainingSamples[1]);
			generateSample(sample3, trainingSamples[2]);
			generateSample(sample4, trainingSamples[3]);

			trainingDataFloats.mView.upload();

			return trainingSamples;
			};
		Gpu::FloatSpace1 sampleSpace;

		auto createOneSample = [&]() {

			std::vector<GpuSample> gpuSamples(1);

			sampleSpace.create(2 + 2);
			auto begin = sampleSpace.begin();
			auto& [seen, desired] = gpuSamples.front();
			sampleSpace.advance(seen, begin, 2);
			sampleSpace.advance(desired, begin, 2);

			std::vector<float> sample1 = { 5, 5, 0.5, 0.5 };

			auto generateSample = [&](const std::vector<float>& sample, GpuSample& gpuSample) {

				std::span<const float> seen(sample.cbegin(), 2)
					, desired(sample.cbegin() + 2, 2);

				std::copy(seen.begin(), seen.end(), gpuSample.first.begin());
				std::copy(desired.begin(), desired.end(), gpuSample.second.begin());
				};

			generateSample(sample1, gpuSamples.front());

			return gpuSamples;
			};

		auto trainingSamples = createOneSample();
		sampleSpace.mView.upload();

		gpu.sync();

		Gpu::GpuView1* gnnOutput = nullptr;

		for(auto generation : std::views::iota(0ULL, 1000ULL)) {

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

		gnn.destroy();
		
		gpu.destroy();
		
	}
	
}