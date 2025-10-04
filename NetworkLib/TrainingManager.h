#pragma once

#include "GpuTensor.h"
#include "GpuNetwork.h"
#include "NetworkSorter.h"

namespace NetworkLib {

	class TrainingManager {
	public:
		using GpuSample = std::pair<Gpu::GpuView1, Gpu::GpuView1>;
		using CpuSample = std::pair<std::vector<float>, std::vector<float>>;
		using CpuSamples = std::vector<CpuSample>;

		using CpuBatchedSample = std::pair<Cpu::View2, Cpu::View2>;
		using CpuBatchedSamples = std::vector<CpuBatchedSample>;
		using CpuBatchedSamplesView = std::span<CpuBatchedSample>;

		using GpuBatchedSample = std::pair<Gpu::GpuView2, Gpu::GpuView2>;
		using GpuBatchedSamples = std::vector<GpuBatchedSample>;
		using GpuBatchedSamplesView = std::span<GpuBatchedSample>;

		Cpu::NetworksView mNetworks;

		struct GpuTask {
			Gpu::Environment mGpu;
			Gpu::Network mGpuNetwork;
		};
		using GpuTasks = std::vector<GpuTask>;
		Parallel mParallelGpuTasks;

		void create(std::size_t gpuNum, NetworkTemplate& networkTemplate, Cpu::NetworksView networks) {
			        
			mParallelGpuTasks.setup(GpuTask{}, networks.size(), gpuNum);

			mNetworks = networks;

			mParallelGpuTasks([&](Parallel::Section& section) {

				auto& gpuTask = std::any_cast<GpuTask&>(section.mAny);

				auto& [gpu, gpuNetwork] = gpuTask;
				gpu.create();
				gpuNetwork.mirror(networks.front());
				
 				});
		}

		void destroy() {

			auto destroyLinkedSpace = [](Gpu::LinkedFloatSpace& linkedSpace) {
				auto& [cpuSpace, gpuSpace] = linkedSpace;
				gpuSpace.destroy();
				cpuSpace.destroy();
				};


			mParallelGpuTasks([&](auto& section) {

				auto& [gpu, gpuNetwork] = std::any_cast<GpuTask&>(section.mAny);

				gpu.destroy();
				gpuNetwork.destroy();
				
				});

			mLogicSamples.destroy();
		}

		GpuTask& getGpuTask(std::size_t idx = 0) {
		
			auto& sectionsView = mParallelGpuTasks.mSectionsView;
			auto& section = sectionsView[idx % sectionsView.size()];
			return std::any_cast<GpuTask&>(section.mAny);
		}
		void train(Cpu::Network& cpuNetwork, GpuTask& gpuTask, std::size_t trainNum, GpuBatchedSamplesView samples, float learnRate, bool print = false) {
			
			train(gpuTask, trainNum, samples, learnRate, print);
		}
		void train(GpuTask& gpuTask, std::size_t trainNum, GpuBatchedSamplesView samples, float learnRate, bool print = false) {
			
			if( print)
				std::print("Training: ");

			auto& [gpu, gpuNetwork] = gpuTask;

 			TimeAverage<milliseconds> trainTime;

			trainTime.accumulateTime([&]() {
				for (auto generation : std::views::iota(0ULL, trainNum)) {
					const auto& [seen, desired] = samples[generation % samples.size()];

					gpuNetwork.forward(gpu, seen);
					gpuNetwork.backward(gpu, seen, desired, learnRate);
					
					if (print)
						printProgress(generation, trainNum);
				}

				gpuNetwork.mWeights.downloadAsync(gpu);
				gpuNetwork.mBias.downloadAsync(gpu);

				gpu.sync();
				});

			if (print)
				std::println("Training took: {}", trainTime.getString<seconds>());
		}
		void trainNetworks(std::size_t trainNum, GpuBatchedSamplesView samples, float learnRate, bool print=false) {

			if (print)
				std::print("Training Networks: ");

			TimeAverage<seconds> trainTime;

			trainTime.accumulateTime([&]() {

				std::atomic<std::size_t> progress = 0;
				mParallelGpuTasks([&](Parallel::Section& section) {

					auto& gpuTask = std::any_cast<GpuTask&>(section.mAny);

					auto& [gpu, gpuNetwork] = gpuTask;

					for (auto idx : section.mIotaView){

						auto& cpuNetwork = mNetworks[idx];

						gpuNetwork.mirror(cpuNetwork);

						train(gpuTask, trainNum, samples, learnRate);

						if (print)
							printProgress(++progress, mNetworks.size());
					}
					
					});

				});

			if (print)
				std::println("Training took: {}", trainTime.getString<seconds>());
		}

		void calculateConvergence(GpuTask& gpuTask, Cpu::Network& cpuNetwork, const TrainingManager::GpuBatchedSamplesView samples, bool print = false) {

			auto& [gpu, gpuNetwork] = gpuTask;

			gpuNetwork.mirror(cpuNetwork);

			gpu.resetMseResult();
			gpu.resetMissesResult();

			auto batchSize = samples.front().first.mView.extent(1);

			for (const auto& [seen, desired] : samples) {

				auto sought = gpuNetwork.forward(gpu, seen);
				auto output = gpuNetwork.getOutput();

				gpu.mse(sought, desired);
				gpu.score(sought, desired);

				if (print) {

					sought.downloadAsync(gpu);
					output.downloadAsync(gpu);
					gpu.sync();

					std::println("\nseen: {}"
						"\ndesired: {}"
						"\nsought: {}"
						"\noutput: {}"
						, seen
						, desired
						, sought
						, output
					);
				}
			}

			gpu.downloadConvergenceResults();
			cpuNetwork.mMse = gpu.getMseResult();
			cpuNetwork.mMisses = gpu.getMissesResult();

			if (print) {

				auto sampleNum = samples.size() * batchSize;

				std::println("\nMse: {}"
					"\nMisses: {}"
					"\nAccuracy: {}"
					, cpuNetwork.mMse
					, cpuNetwork.mMisses
					, (sampleNum - cpuNetwork.mMisses) / float(sampleNum) * 100.0f
				);
			}
		}
		void calculateNetworkConvergence(GpuTask& gpuTask, NetworksSorter::Idx idx, const TrainingManager::GpuBatchedSamplesView samples, bool print = false) {
		
			if (print)
				std::println("Calculate Convergence");

			auto& cpuNetwork = mNetworks[idx];

			TimeAverage<seconds> convergenceTime;
			convergenceTime.accumulateTime([&]() {

				calculateConvergence(gpuTask, cpuNetwork, samples, print);
 
				});

			if (print)
				std::println("Took: {}", convergenceTime.getString<seconds>());
		}
		void calculateNetworksConvergence(const TrainingManager::GpuBatchedSamplesView samples, bool print=false) {

			if (print)
				std::println("Calculate Convergence");

			TimeAverage<seconds> convergenceTime;

			convergenceTime.accumulateTime([&]() {

				mParallelGpuTasks([&](auto& section) {

					auto& gpuTask = std::any_cast<GpuTask&>(section.mAny);
	
					for (auto idx : section.mIotaView) {
						
						auto& cpuNetwork = mNetworks[idx];

						calculateConvergence(gpuTask, cpuNetwork, samples);
					}
					});
				});

			if (print)
				std::println("Took: {}", convergenceTime.getString<seconds>());

		}

		static GpuBatchedSamples createGpuBatchedSamplesSpace(Gpu::LinkedFloatSpace& linkedSampleSpace
			, std::size_t inputSize, std::size_t outputSize, std::size_t sampleNum, std::size_t batchSize = 1) {

			GpuBatchedSamples gpuBatchedSamples;
			auto batchNum = std::ceil(sampleNum / float(batchSize));
			gpuBatchedSamples.reserve(batchNum);

			linkedSampleSpace.create(batchNum * batchSize * (inputSize + outputSize));
			return gpuBatchedSamples;
		}

		static GpuBatchedSamplesView advanceGpuBatchedSamples(Gpu::LinkedFloatSpace& linkedSampleSpace
			, float*& begin, GpuBatchedSamples& gpuSamples, const CpuSamples& cpuSamples
			, std::size_t batchSize = 1) {
			
			auto inputSize = cpuSamples.front().first.size()
				, outputSize = cpuSamples.front().second.size();

			auto& gpuSampleSpace = linkedSampleSpace.mGpuSpace;

			auto currentSample = cpuSamples.begin();

			auto batchNum = std::ceil(cpuSamples.size() / float(batchSize));
			auto viewStart = gpuSamples.size();
			for (auto batch : std::views::iota(0ULL, batchNum )) {

				GpuBatchedSample gpuSample;

				auto& [seenBatch, desiredBatch] = gpuSample;

				gpuSampleSpace.advance(seenBatch, begin, inputSize, batchSize);
				gpuSampleSpace.advance(desiredBatch, begin, outputSize, batchSize);

				for (auto b : std::views::iota(0ULL, batchSize)) {

					if (currentSample == cpuSamples.end()) break;
					auto& [seen, desired] = *currentSample;
					++currentSample;

					auto gpuSeen = seenBatch.viewColumn(b);
					auto gpuDesired = desiredBatch.viewColumn(b);

					std::copy(seen.begin(), seen.end(), gpuSeen.begin());
					std::copy(desired.begin(), desired.end(), gpuDesired.begin());
				}

				gpuSamples.push_back(gpuSample);
			}
			return { gpuSamples.begin() + viewStart, gpuSamples.size() - viewStart };
		}


		struct LogicSamples {

			Gpu::LinkedFloatSpace mFloatSpace;
			GpuBatchedSamples mGpuBatchedSamples;

			GpuBatchedSamplesView mXORSamples;
			GpuBatchedSamplesView mANDSamples;
			GpuBatchedSamplesView mORSamples;

			void create(NetworkTemplate& networkTemplate) {

				auto inputSize = networkTemplate.mInputSize
					, outputSize = networkTemplate.mLayerTemplates.back().mNodeCount;
				auto batchSize = networkTemplate.mBatchSize;
				auto sampleNum =  4 * 3 ; //XOR, AND, OR all have 4 samples
				mGpuBatchedSamples = TrainingManager::createGpuBatchedSamplesSpace(mFloatSpace
					, inputSize, outputSize, sampleNum, batchSize);

				auto begin = mFloatSpace.mGpuSpace.begin();

				CpuSamples andSamples = {
					{{ 0,0 }, {1,0}},
					{{ 0,1 }, {1,0}},
					{{ 1,0 }, {1,0}},
					{{ 1,1 }, {0,1}}
					};
				const CpuSamples xorSamples = {
					{{0,0}, {1,0}},
					{{0,1}, {0,1}},
					{{1,0}, {0,1}},
					{{1,1}, {1,0}}
					};
				const CpuSamples orSamples = {
					{{0,0}, {1,0}},
					{{0,1}, {0,1}},
					{{1,0}, {0,1}},
					{{1,1}, {0,1}}
					};

				mANDSamples = TrainingManager::advanceGpuBatchedSamples(mFloatSpace, begin, mGpuBatchedSamples, andSamples, batchSize);
				mXORSamples = TrainingManager::advanceGpuBatchedSamples(mFloatSpace, begin, mGpuBatchedSamples, xorSamples, batchSize);
				mORSamples = TrainingManager::advanceGpuBatchedSamples(mFloatSpace, begin, mGpuBatchedSamples, orSamples, batchSize);

				mFloatSpace.mGpuSpace.upload();
			}
			void destroy() {
				mFloatSpace.destroy();
			}
		} mLogicSamples;

	};
	
}