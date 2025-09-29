#pragma once
#include <random>

#include "Algorithms.h"

#include "Environment.h"
#include "TrainingManager.h"
#include "CpuNetwork.h"
#include "GpuNetwork.h"
#include "NetworkSorter.h"

namespace NetworkLib {

	namespace Model {

		class XOR {
		public:

			std::mt19937_64 mRandom;
			Gpu::Environment mGpu;
			TrainingManager::GpuBatchedSamples mBatchedSamples;
			Gpu::Network mGpuNetwork;
			Cpu::Network mCpuNetwork;
			NetworkTemplate mNetworkTemplate;
			TrainingManager mTrainingManager;

			std::size_t mInputSize = 2, mOutputSize = 2
				, mTrainNum = 5000;

			std::size_t mBatchSize = 4;
			float mLearnRate = 0.002f;

			bool mPrintConsole = false;

			void calculateConvergence() {

				mGpu.resetMseResult();

				for (const auto& [seen, desired] : mBatchedSamples) {

					auto sought = mGpuNetwork.forward(mGpu, seen);
					auto output = mGpuNetwork.getOutput();

					mGpu.mse(sought, desired);

					if (mPrintConsole) {

						sought.downloadAsync(mGpu);
						output.downloadAsync(mGpu);

						mGpu.sync();

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

				if(mPrintConsole)
					std::println("mse: {}\n", mGpu.getMseResult() );
			}
			void create() {

				mGpu.create();

				using ActivationFunction = LayerTemplate::ActivationFunction;
				mNetworkTemplate = { mInputSize, mBatchSize
					, {{ 8, ActivationFunction::ReLU}
					, { 4, ActivationFunction::ReLU}
					, { mOutputSize, ActivationFunction::Softmax}}
				};

				mCpuNetwork.create(&mNetworkTemplate);
				mCpuNetwork.initialize(mRandom);

				mGpuNetwork.mirror(mCpuNetwork);
				mGpuNetwork.upload();

				mBatchedSamples = mTrainingManager.createXORBatchedSamples(mBatchSize);
			}
			void destroy() {
				mTrainingManager.destroy();
				mGpuNetwork.destroy();
				mCpuNetwork.destroy();
				mGpu.destroy();
			}

			void trainOne(std::size_t generation) {

				TimeAverage<microseconds> trainTime;

				trainTime.accumulateTime([&]() {

					const auto& [seen, desired] = mBatchedSamples[generation % mBatchedSamples.size()];

					mGpuNetwork.forward(mGpu, seen);
					mGpuNetwork.backward(mGpu, seen, desired, mLearnRate);

					});
			}

			void train(bool print=false) {

				TimeAverage<microseconds> trainTime;

				if(mPrintConsole)
					std::print("Training: ");

				for (auto generation : std::views::iota(0ULL, mTrainNum))
					trainTime.accumulateTime([&]() {

						trainOne(generation);
						
						if(mPrintConsole)
							printProgress(generation, mTrainNum);

						});

			}

			void run(bool print=true) {

				mPrintConsole = print;

				create();
				calculateConvergence();
				train();
				calculateConvergence();
				destroy();
			}
		};
		
		class XORLottery {
		public:

			struct GpuTask {
				Gpu::Environment mGpu;
				Gpu::Network mGpuNetwork;
				Cpu::NetworksView mSourceNetworks;
			};

			std::vector<GpuTask> mGpuTasks;
			Parallel mParallelGpuTasks;

			Cpu::Networks mNetworks;
			NetworksSorter mNetworksSorter;

			TrainingManager::GpuBatchedSamples mBatchedSamples;

			NetworkTemplate mNetworkTemplate;
			TrainingManager mTrainingManager;

			std::size_t mInputSize = 2, mOutputSize = 2
				, mTrainNum = 5000;

			std::size_t mBatchSize = 4;
			float mLearnRate = 0.002f;

			std::size_t mMaxGpus = 2, mMaxNetworks = 100;
			bool mPrintConsole = false;
			
			void calculateConvergence(Gpu::Environment& gpu, Gpu::Network& gpuNetwork, Cpu::Network& cpuNetwork, bool print=false) {


				gpuNetwork.mirror(cpuNetwork);

				gpu.resetMseResult();
				gpu.resetMissesResult();

				for (const auto& [seen, desired] : mBatchedSamples) {

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
				gpu.sync();
				cpuNetwork.mMse = gpu.getMseResult();
				cpuNetwork.mMisses = gpu.getMissesResult();

				if (print) {

					std::println("\nMse: {}"
						"\nMisses: {}"
						, cpuNetwork.mMse
						, cpuNetwork.mMisses
					);
				}
			}
			void calculateConvergence() {

				if (mPrintConsole)
					std::println("Calculate Convergence");

				TimeAverage<microseconds> convergenceTime;

				convergenceTime.accumulateTime([&]() {

					mParallelGpuTasks([&](Parallel::Section& section) {

						for (auto gpuTaskId : section.mIotaView) {

							auto& [gpu, gpuNetwork, cpuNetworks] = mGpuTasks[gpuTaskId];

							for (auto& cpuNetwork : cpuNetworks) 
								calculateConvergence(gpu, gpuNetwork, cpuNetwork);
						}
						});
					});
				std::println("Took: {}", convergenceTime.getString<seconds>());

				mNetworksSorter.sortBySuperRadius();
				if (mPrintConsole) {
					
					std::println("Networks sorted by SuperRadius: ");

					for (std::size_t i = 0; auto networkIdx : mNetworksSorter.mNetworksIdx) {
						auto& network = mNetworks[networkIdx];
						std::println("Rank: {}; Network Id: {}; Misses: {}; Mse: {};", ++i, networkIdx, network.mMisses, network.mMse);
					}
				}
				
				auto& bestNetwork = mNetworksSorter.getBest();
				auto bestNetworkIdx = mNetworksSorter.getBestIdx();
				std::println("Rank 1 Network Id: {}; Misses: {}; Mse: {};", bestNetworkIdx, bestNetwork.mMisses, bestNetwork.mMse);

				auto& [gpu, gpuNetwork, cpuNetworks] = mGpuTasks.front();
				gpuNetwork.mirror(bestNetwork);
				calculateConvergence(gpu, gpuNetwork, bestNetwork, true);
			}
			void create() {

				if( mPrintConsole )
					std::println("Create XOR Lottery: ");

				using ActivationFunction = LayerTemplate::ActivationFunction;
				mNetworkTemplate = { mInputSize, mBatchSize
					, {{ 2, ActivationFunction::ReLU}
					, { mOutputSize, ActivationFunction::Softmax}}
				};

				//assign networks ids, but assign the last network the known convergent id
				std::mt19937_64 mRandom;
				auto knownId = mRandom.default_seed;

				mGpuTasks.resize(mMaxGpus);
				mParallelGpuTasks.section(mGpuTasks.size(), mGpuTasks.size());


				mNetworks.resize(mMaxNetworks);
				mNetworksSorter.create(mNetworks);

				Parallel parallelNetworks(mNetworks.size());
				
				parallelNetworks([&](auto& section) {

					for (auto id : section.mIotaView) {

						auto& network = mNetworks[id];

						network.create(&mNetworkTemplate, true);

						network.intializeId(id);
					}
					});

				auto setupGpuWork = [&]() {

					std::size_t start = 0, end = 0, size = mNetworks.size() / mGpuTasks.size();
					for (auto& [gpu, gpuNetwork, cpuNetworks] : mGpuTasks | std::views::take(mGpuTasks.size() - 1)) {
						gpu.create();

						end = start + size;
						cpuNetworks = { &mNetworks[start], size };
						start = end;
					}
					auto& [gpu, gpuNetwork, cpuNetworks] = mGpuTasks.back();
					gpu.create();	
					cpuNetworks = { &mNetworks[start], mNetworks.size() - start};
					
					};

				setupGpuWork();


				mBatchedSamples = mTrainingManager.createXORBatchedSamples(mBatchSize);
			}
			void destroy() {
				
				if( mPrintConsole )
					std::println("destroying model");

				mTrainingManager.destroy();
				for( auto& network: mNetworks)
					network.destroy();

				for (auto& [gpu, gpuNetworks, cpuNetworks] : mGpuTasks) {
					gpuNetworks.destroy();
					gpu.destroy();
				}
			}

			void trainOne( Gpu::Environment& gpu, Gpu::Network& network,std::size_t generation) {
				const auto& [seen, desired] = mBatchedSamples[generation % mBatchedSamples.size()];

				network.forward(gpu, seen);
				network.backward(gpu, seen, desired, mLearnRate);
			}

			void train(bool print = false) {

				if (mPrintConsole)
					std::print("Training Networks: ");

				TimeAverage<nanoseconds> trainTime;

				trainTime.accumulateTime([&]() {

					std::atomic<std::size_t> progress = 0;
					mParallelGpuTasks([&](Parallel::Section& section) {

						for (auto gpuTaskId : section.mIotaView) {

							auto& [gpu, gpuNetwork, cpuNetworks] = mGpuTasks[gpuTaskId];

							for (auto& cpuNetwork : cpuNetworks) {

								gpuNetwork.mirror(cpuNetwork);

								for (auto generation : std::views::iota(0ULL, mTrainNum))
									trainOne(gpu, gpuNetwork, generation);

								gpuNetwork.mWeights.downloadAsync(gpu);
								gpuNetwork.mBias.downloadAsync(gpu);

								gpu.sync();
								++progress;
								printProgress(progress, mNetworks.size());
							}
						}
						});
					
					});

				if (mPrintConsole)
					std::println("Training took: {}", trainTime.getString<seconds>());
			}

			void run(bool print = true) {

				mPrintConsole = print;

				create();
				train();
				calculateConvergence();

				destroy();
			}
		};
		
	}
}