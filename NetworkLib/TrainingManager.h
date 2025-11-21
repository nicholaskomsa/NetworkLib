#pragma once

#include "GpuTensor.h"
#include "GpuNetwork.h"
#include "NetworkSorter.h"
#include "Algorithms.h"

#include <fstream>
#include <map>
#include <numeric>

namespace NetworkLib {

	class TrainingManager {
	public:


		using CpuSample = std::pair<std::vector<float>, std::vector<float>>;
		using CpuSamples = std::vector<CpuSample>;

		using CpuBatchedSample = std::pair<Cpu::View2, Cpu::View2>;
		using CpuBatchedSamples = std::vector<CpuBatchedSample>;
		using CpuBatchedSamplesView = std::span<CpuBatchedSample>;

		//one input one output not batched
		using GpuSample = std::pair<Gpu::GpuView1, Gpu::GpuView1>;
		using GpuSamples = std::vector<GpuSample>;
		using GpuSamplesView = std::span<GpuSample>;

		//all batched samples have different output
		using GpuBatchedSample = std::pair<Gpu::GpuView2, Gpu::GpuView2>;
		using GpuBatchedSamples = std::vector<GpuBatchedSample>;
		using GpuBatchedSamplesView = std::span<GpuBatchedSample>;

		//all batched2 samples share the same output
		using GpuBatched2Sample = std::pair<Gpu::GpuView2, Gpu::GpuView1>;
		using GpuBatched2Samples = std::vector<GpuBatched2Sample>;
		using GpuBatched2SamplesView = std::span<GpuBatched2Sample>;

		//all batched3 samples have unique ouputs like batched except they are output idx instead of (2D/inline output)
		using GpuBatched3Sample = std::pair<Gpu::GpuView2, Gpu::GpuIntView2>;
		using GpuBatched3Samples = std::vector<GpuBatched3Sample>;
		using GpuBatched3SamplesView = std::span<GpuBatched3Sample>;


		Cpu::NetworksMap mNetworksMap;
		Cpu::Network::Id mNetworksIdCounter = 0;

		std::mutex mNetworksMapMutex;

		struct GpuTask {
			Gpu::Environment mGpu;
			Gpu::Network mGpuNetwork;
		};
		using GpuTasks = std::vector<GpuTask>;
		Parallel mParallelGpuTasks;
		std::size_t mGpuNum = 0;

		TrainingManager& operator=(const TrainingManager&) = delete;

		void create(std::size_t gpuNum);
		void destroy();

		void addNetwork();
		void addNetwork(std::size_t id);
		Cpu::Network& getNetwork(std::size_t id);
		GpuTask& getGpuTask(std::size_t idx = 0);

		using ForEachFunctor = std::function<void(GpuTask& gpuTask, Cpu::Network& cpuNetwork)>;

		void forEachNetwork(Cpu::NetworksMap& networks, ForEachFunctor&& functor);

		template<typename SamplesViewType>
		void train(GpuTask& gpuTask, std::size_t trainNum, const SamplesViewType& samples, float learnRate, std::size_t offset = 0, bool download=true, bool print = false) {
			
			auto& [gpu, gpuNetwork] = gpuTask;

			time<seconds>("Training Network", [&]() {

				for (auto generation : std::views::iota(offset, offset + trainNum)) {

					const auto& [seen, desired] = samples[generation % samples.size()];

					gpuNetwork.forward(gpu, seen);
					gpuNetwork.backward(gpu, seen, desired, learnRate);

					if (print)
						printProgress(generation, trainNum);
				}
				if( download)
					gpuNetwork.download(gpu);

				}, print);
		}
		template<typename SamplesViewType>
		void trainNetworks(std::size_t trainNum, const SamplesViewType& samples, float learnRate, std::size_t offset, bool print = false) {

			time<seconds>("Training Networks", [&]() {

				std::atomic<std::size_t> progress = 0;

				forEachNetwork(mNetworksMap, [&](GpuTask& gpuTask, Cpu::Network& cpuNetwork) {

					auto& [gpu, gpuNetwork] = gpuTask;
					gpuNetwork.mirror(cpuNetwork);

					train(gpuTask, trainNum, samples, learnRate, offset, true, false);

					if (print)
						printProgress(++progress, mNetworksMap.size());

					});

				}, print);
		}
		
		void calculateConvergence(GpuTask& gpuTask, Cpu::Network& cpuNetwork, const GpuBatched3SamplesView& samples, std::size_t trueSampleNum, const Gpu::GpuView2& desiredGroup, bool print = false) {

			auto& [gpu, gpuNetwork] = gpuTask;

			gpuNetwork.mirror(cpuNetwork);

			gpu.resetSqeResult();
			gpu.resetMissesResult();

			Gpu::GpuView2 sought;

			for (const auto& [seen, desired] : samples) {

				sought = gpuNetwork.forward(gpu, seen);

				gpu.sqe(sought, desired, desiredGroup);
				gpu.score(sought, desired, desiredGroup);
			}

			auto calculateVariables = [&]() {
				
				gpu.downloadConvergenceResults();

				auto soughtSize = sought.mView.extent(0);

				//normalise to get sqe -> mse 
				cpuNetwork.mMse = gpu.getSqeResult() / (soughtSize * trueSampleNum);
				cpuNetwork.mMisses = gpu.getMissesResult();
				cpuNetwork.mAccuracy = (trueSampleNum - cpuNetwork.mMisses) / float(trueSampleNum) * 100.0f;

				};

			calculateVariables();
			
			if (print)
				std::println("Mse: {}; Misses: {}; Accuracy: {};"
					, cpuNetwork.mMse, cpuNetwork.mMisses, cpuNetwork.mAccuracy
				);
		}


		template<typename SamplesViewType>
		void calculateConvergence(GpuTask& gpuTask, Cpu::Network& cpuNetwork, const SamplesViewType& samples, std::size_t trueSampleNum, bool print = false) {

			auto& [gpu, gpuNetwork] = gpuTask;

			gpuNetwork.mirror(cpuNetwork);

			gpu.resetSqeResult();
			gpu.resetMissesResult();

			for (const auto& [seen, desired] : samples) {

				auto sought = gpuNetwork.forward(gpu, seen);

				gpu.sqe(sought, desired);
				gpu.score(sought, desired);
			}

			auto calculateVariables = [&]() {

				gpu.downloadConvergenceResults();

				auto soughtSize = cpuNetwork.getSought().extent(0);

				//normalise to get sqe -> mse 
				cpuNetwork.mMse = gpu.getSqeResult() / (soughtSize * trueSampleNum);
				cpuNetwork.mMisses = gpu.getMissesResult();
				cpuNetwork.mAccuracy = (trueSampleNum - cpuNetwork.mMisses) / float(trueSampleNum) * 100.0f;

				};


			calculateVariables();

			if (print) 
				std::println("Mse: {}; Misses: {}; Accuracy: {};"
					, cpuNetwork.mMse, cpuNetwork.mMisses, cpuNetwork.mAccuracy
				);
		}
 
		template<typename SamplesViewType>
		void calculateNetworksConvergence(Cpu::NetworksMap& networks, const SamplesViewType& samples, std::size_t trueSampleNum, bool print = false) {

			time<seconds>("Calculate Networks Convergence", [&]() {
				forEachNetwork(networks, [&](GpuTask& gpuTask, Cpu::Network& cpuNetwork) {
					calculateConvergence(gpuTask, cpuNetwork, samples, trueSampleNum, print);
					});
				}, print);
		}
		template<typename SamplesViewType>
		void calculateNetworksConvergence(const SamplesViewType& samples, std::size_t trueSampleNum, bool print = false) {
			calculateNetworksConvergence(mNetworksMap, samples, trueSampleNum, print);
		}

		void calculateNetworksConvergence(Cpu::NetworksMap& networks, const GpuBatched3SamplesView& samples, std::size_t trueSampleNum, const Gpu::GpuView2& desiredGroup, bool print = false) {

			time<seconds>("Calculate Networks Convergence", [&]() {
				forEachNetwork(networks, [&](GpuTask& gpuTask, Cpu::Network& cpuNetwork) {
					calculateConvergence(gpuTask, cpuNetwork, samples, trueSampleNum, desiredGroup, print);
					});
				}, print);
		}
		
		void calculateNetworksConvergence(const GpuBatched3SamplesView& samples, std::size_t trueSampleNum, const Gpu::GpuView2& desiredGroup, bool print = false) {
			calculateNetworksConvergence(mNetworksMap, samples, trueSampleNum, desiredGroup, print);
		}
		static GpuBatchedSamples createGpuBatchedSamplesSpace(Gpu::LinkedFloatSpace& linkedSampleSpace
			, std::size_t inputSize, std::size_t outputSize, std::size_t sampleNum, std::size_t batchSize = 1);

		static GpuBatchedSamplesView advanceGpuBatchedSamples(Gpu::LinkedFloatSpace& linkedSampleSpace
			, float*& begin, GpuBatchedSamples& gpuSamples, const CpuSamples& cpuSamples
			, std::size_t batchSize = 1);

		static Gpu::GpuView1 advanceGpuViews(Gpu::LinkedFloatSpace& linkedSampleSpace
			, float*& begin, Gpu::GpuViews1& gpuViews, std::size_t outputSize);
	};
	
}