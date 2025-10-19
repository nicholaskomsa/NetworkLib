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


		void create(std::size_t gpuNum) {
			
			mGpuNum = gpuNum;
			mParallelGpuTasks.setup(GpuTask{}, gpuNum, gpuNum);

			auto& defaultNetwork = mNetworksMap.begin()->second;
			mParallelGpuTasks([&](Parallel::Section& section) {

				auto& gpuTask = std::any_cast<GpuTask&>(section.mAny);

				auto& [gpu, gpuNetwork] = gpuTask;
				gpu.create();
				gpuNetwork.mirror(defaultNetwork);
 				});
		}

		void destroy() {

			for( auto& [id, network] : mNetworksMap )
				network.destroy();

			mParallelGpuTasks([&](auto& section) {

				auto& [gpu, gpuNetwork] = std::any_cast<GpuTask&>(section.mAny);

				gpu.destroy();
				gpuNetwork.destroy();
				
				});

			mLogicSamples.destroy();
		}
		void addNetwork() {
			std::size_t id = mNetworksIdCounter++;
			addNetwork(id);
		}
		void addNetwork(std::size_t id) {
			
			auto found = mNetworksMap.find(id);
			if (found != mNetworksMap.end())
				throw std::runtime_error("network already exists");

			auto network = mNetworksMap[id];
		}
		Cpu::Network& getNetwork(std::size_t id) {
			
			auto found = mNetworksMap.find(id);
			if (found == mNetworksMap.end())
				throw std::runtime_error("Network not found");

			return found->second;
		}
		GpuTask& getGpuTask(std::size_t idx = 0) {
		
			auto& sectionsView = mParallelGpuTasks.mSectionsView;
			auto& section = sectionsView[idx % sectionsView.size()];
			return std::any_cast<GpuTask&>(section.mAny);
		}

		void train(GpuTask& gpuTask, std::size_t trainNum, GpuBatchedSamplesView samples, float learnRate, bool print = false) {
			
			if( print)
				std::print("Training: ");

			auto& [gpu, gpuNetwork] = gpuTask;

 			TimeAverage<seconds> trainTime;

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

				forEachNetwork(mNetworksMap, [&](GpuTask& gpuTask, Cpu::Network& cpuNetwork) {

					auto& [gpu, gpuNetwork] = gpuTask;
					gpuNetwork.mirror(cpuNetwork);

					train(gpuTask, trainNum, samples, learnRate, false);

					if (print)
						printProgress(++progress, mNetworksMap.size());

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
		void calculateNetworkConvergence(GpuTask& gpuTask, Cpu::Network& cpuNetwork, const TrainingManager::GpuBatchedSamplesView samples, bool print = false) {
		
			if (print)
				std::println("Calculate Convergence");

			TimeAverage<seconds> convergenceTime;
			convergenceTime.accumulateTime([&]() {

				calculateConvergence(gpuTask, cpuNetwork, samples, print);
 
				});

			if (print)
				std::println("Took: {}", convergenceTime.getString<seconds>());
		}

		
		
		void forEachNetwork(Cpu::NetworksMap& networks, auto&& functor) {


			auto networksBegin = networks.begin();

			auto getNextNetwork = [&]()->Cpu::Network* {

				std::scoped_lock lock(mNetworksMapMutex);

				if (networksBegin == mNetworksMap.end())
					return nullptr;

				Cpu::Network* network = &networksBegin->second;

				std::advance(networksBegin, 1);
				return network;
				};

			mParallelGpuTasks.section(networks.size());

			mParallelGpuTasks([&](auto& section) {

				auto& gpuTask = std::any_cast<GpuTask&>(section.mAny);

				for (auto idx : section.mIotaView) {

					auto cpuNetwork = getNextNetwork();
					if (!cpuNetwork)
						return;

					functor(gpuTask, *cpuNetwork);
				}
				});
				
		}

		void calculateNetworksConvergence(Cpu::NetworksMap& networks, GpuBatchedSamplesView samples, bool print = false) {

			if (print)
				std::println("Calculate Convergence");

			TimeAverage<seconds> convergenceTime;

			convergenceTime.accumulateTime([&]() {
				forEachNetwork(networks, [&](GpuTask& gpuTask, Cpu::Network& cpuNetwork) {
					calculateConvergence(gpuTask, cpuNetwork, samples, print);
					});
				});

			if (print)
				std::println("Took: {}", convergenceTime.getString<seconds>());
		}

		void calculateNetworksConvergence(GpuBatchedSamplesView samples, bool print = false) {

			calculateNetworksConvergence(mNetworksMap, samples, print);
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
			constexpr std::size_t zero = 0;
			for (auto batch : std::views::iota(zero, batchNum )) {

				GpuBatchedSample gpuSample;

				auto& [seenBatch, desiredBatch] = gpuSample;

				gpuSampleSpace.advance(seenBatch, begin, inputSize, batchSize);
				gpuSampleSpace.advance(desiredBatch, begin, outputSize, batchSize);

				for (auto b : std::views::iota(zero, batchSize)) {

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
			GpuBatchedSamples mLogicSamples;

			GpuBatchedSamplesView mXORSamples;
			GpuBatchedSamplesView mANDSamples;
			GpuBatchedSamplesView mORSamples;

			void create(NetworkTemplate& networkTemplate) {

				auto inputSize = networkTemplate.mInputSize
					, outputSize = networkTemplate.mLayerTemplates.back().mNodeCount;
				auto batchSize = networkTemplate.mBatchSize;
				auto sampleNum =  4 * 3 ; //XOR, AND, OR all have 4 samples
				mLogicSamples = TrainingManager::createGpuBatchedSamplesSpace(mFloatSpace
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

				mANDSamples = TrainingManager::advanceGpuBatchedSamples(mFloatSpace, begin, mLogicSamples, andSamples, batchSize);
				mORSamples = TrainingManager::advanceGpuBatchedSamples(mFloatSpace, begin, mLogicSamples, orSamples, batchSize);
				mXORSamples = TrainingManager::advanceGpuBatchedSamples(mFloatSpace, begin, mLogicSamples, xorSamples, batchSize);

				mFloatSpace.mGpuSpace.upload();
			}
			void destroy() {
				mFloatSpace.destroy();
			}

			struct SamplesGroup {
				GpuBatchedSamplesView mXOR, mOR, mAND, mAll;
			};

			SamplesGroup getSamples() {
				return { mXORSamples, mORSamples, mANDSamples, mLogicSamples };
			}                 
			   
		} mLogicSamples;

	};
	
}