#pragma once
#include <random>
#include <fstream>

#include "Algorithms.h"

#include "TrainingManager.h"

namespace NetworkLib {

	namespace Model {

		class Convolution1 {
		public:
			TrainingManager::GpuBatchedSamplesView mBatchedSamplesView;
			NetworkTemplate mNetworkTemplate;
			std::size_t mId = 981;
			TrainingManager mTrainingManager;
			TrainingManager::GpuTask* mGpuTask = nullptr;

			std::size_t mInputWidth = 2, mOutputSize = 2
				, mTrainNum = 5000;
			std::size_t kernelSize = 2;

			std::size_t mBatchSize = 4;
			float mLearnRate = 0.002f;

			bool mPrintConsole = false;

			void calculateConvergence() {

				auto& cpuNetwork = mTrainingManager.getNetwork(mId);
				mTrainingManager.calculateNetworkConvergence(*mGpuTask, cpuNetwork, mBatchedSamplesView, mPrintConsole);

			}
			void create() {

				using ConvolutionType = LayerTemplate::ConvolutionType;
				using ActivationFunction = LayerTemplate::ActivationFunction;

				mNetworkTemplate = { mInputWidth, mBatchSize
					, {{ ConvolutionType::Conv1, kernelSize, 2, ActivationFunction::ReLU }
					, { mOutputSize, ActivationFunction::Softmax}}
				};

				mTrainingManager.addNetwork(mId);
				auto& network = mTrainingManager.getNetwork(mId);
				network.create(&mNetworkTemplate, true);
				network.initializeId(mId);

				mTrainingManager.create(1);
				mTrainingManager.mLogicSamples.create(mNetworkTemplate);
				mBatchedSamplesView = mTrainingManager.mLogicSamples.mXORSamples;

				mGpuTask = &mTrainingManager.getGpuTask();
			}
			void destroy() {
				mTrainingManager.destroy();
			}

			void train(std::size_t trainNum = 1, bool print = false) {
				mTrainingManager.train(*mGpuTask, trainNum, mBatchedSamplesView, mLearnRate, print);
			}

			Cpu::Network& getNetwork() {
				return mTrainingManager.getNetwork(mId);
			}
			void run(bool print = true) {

				mPrintConsole = print;

				create();
				calculateConvergence();
				train(mTrainNum, true);

				calculateConvergence();
				destroy();
			}
		};

		class Convolution1Lottery {
		public:
			TrainingManager::GpuBatchedSamplesView mBatchedSamplesView;
			NetworkTemplate mNetworkTemplate;
			std::size_t mId = 981;
			TrainingManager mTrainingManager;
			NetworksSorter mNetworksSorter;

			std::size_t mInputWidth = 2, mOutputSize = 2
				, mTrainNum = 5000;
			std::size_t kernelSize = 2;

			std::size_t mBatchSize = 4;
			float mLearnRate = 0.002f;

			std::size_t mMaxGpus = 2, mMaxNetworks = 1000;

			bool mPrintConsole = false;
		
			void calculateConvergence() {

				mTrainingManager.calculateNetworksConvergence(mBatchedSamplesView);
				mNetworksSorter.sortBySuperRadius();

				if (mPrintConsole) {

					auto& networksMap = mTrainingManager.mNetworksMap;

					std::println("Networks sorted by SuperRadius: ");

					for (std::size_t rank = 1; auto networkId : mNetworksSorter.mNetworksIds) {
						auto& network = mTrainingManager.mNetworksMap[networkId];
						std::println("Rank: {}; Network Id: {}; Misses: {}; Mse: {};", rank++, networkId, network.mMisses, network.mMse);
					}
					auto& bestNetwork = mNetworksSorter.getBest();
					auto bestNetworkId = mNetworksSorter.getBestId();
					std::println("\nRank 1 Network Id: {}; Misses: {}; Mse: {};", bestNetworkId, bestNetwork.mMisses, bestNetwork.mMse);

					mTrainingManager.calculateNetworkConvergence(mTrainingManager.getGpuTask(), bestNetwork, mBatchedSamplesView, true);

					auto printZeroMisses = [&]() {

						auto zeroMissesCount = std::count_if(networksMap.begin(), networksMap.end(), [&](auto& networkPair) {

							return networkPair.second.mMisses == 0;

							});

						std::println("\nNetworks with zero misses: {}", zeroMissesCount);
						};
					printZeroMisses();
				}
			}
			void create() {

				using ConvolutionType = LayerTemplate::ConvolutionType;
				using ActivationFunction = LayerTemplate::ActivationFunction;

				mNetworkTemplate = { mInputWidth, mBatchSize
					, {{ ConvolutionType::Conv1, kernelSize, 2, ActivationFunction::ReLU }
					, { mOutputSize, ActivationFunction::Softmax}}
				};

				auto createNetworks = [&]() {

					for (auto id : std::views::iota(0ULL, mMaxNetworks))
						mTrainingManager.addNetwork(id);

					mNetworksSorter.create(mTrainingManager.mNetworksMap);

					auto& networkIds = mNetworksSorter.mNetworksIds;
					Parallel parallelNetworks(networkIds.size());
					parallelNetworks([&](auto& section) {

						for (auto idx : section.mIotaView) {

							auto id = networkIds[idx];
							auto& network = mTrainingManager.getNetwork(id);

							network.create(&mNetworkTemplate, true);
							network.initializeId(id);
						}
						});

					mTrainingManager.create(mMaxGpus);

					};

				createNetworks();

				mTrainingManager.mLogicSamples.create(mNetworkTemplate);
				mBatchedSamplesView = mTrainingManager.mLogicSamples.mXORSamples;
			}
			void destroy() {
				if (mPrintConsole)
					std::println("destroying model");

				mTrainingManager.destroy();
			}

			void train(std::size_t trainNum = 1, bool print = false) {
				mTrainingManager.trainNetworks(trainNum, mBatchedSamplesView, mLearnRate, print);
			}

			Cpu::Network& getNetwork() {
				return mTrainingManager.getNetwork(mId);
			}
			void run(bool print = true) {

				mPrintConsole = print;

				create();
				calculateConvergence();
				train(mTrainNum, true);

				calculateConvergence();
				destroy();
			}
		};
	}
}