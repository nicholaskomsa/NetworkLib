#pragma once
#include <random>
#include <fstream>

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
			TrainingManager::GpuBatchedSamplesView mBatchedSamplesView;
			NetworkTemplate mNetworkTemplate;
			std::size_t mId = 981;
			TrainingManager mTrainingManager;
			TrainingManager::GpuTask* mGpuTask = nullptr;

			std::size_t mInputSize = 2, mOutputSize = 2
				, mTrainNum = 5000;

			std::size_t mBatchSize = 4;
			float mLearnRate = 0.002f;

			bool mPrintConsole = false;

			void calculateConvergence() {

				auto& cpuNetwork = mTrainingManager.getNetwork(mId);
				mTrainingManager.calculateNetworkConvergence(*mGpuTask, cpuNetwork, mBatchedSamplesView, mPrintConsole);

			}
			void create() {

				using ActivationFunction = LayerTemplate::ActivationFunction;
				mNetworkTemplate = { mInputSize, mBatchSize
					, {{ 100, ActivationFunction::ReLU}
					, { 50, ActivationFunction::ReLU}
					, { 20, ActivationFunction::ReLU }
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

			void train(std::size_t trainNum=1, bool print=false) {
				mTrainingManager.train(*mGpuTask, trainNum, mBatchedSamplesView, mLearnRate, print);
			}

			Cpu::Network& getNetwork() {
				return mTrainingManager.getNetwork(mId);
			}
			void run(bool print=true) {

				mPrintConsole = print;

				create();
				calculateConvergence();
				train(mTrainNum, true);

				calculateConvergence();
				destroy();
			}
		};
		
		class XORLottery {
		public:

			NetworksSorter mNetworksSorter;

			TrainingManager::GpuBatchedSamplesView mBatchedSamplesView;

			NetworkTemplate mNetworkTemplate;
			TrainingManager mTrainingManager;

			std::size_t mInputSize = 2, mOutputSize = 2
				, mTrainNum = 5000;

			std::size_t mBatchSize = 4;
			float mLearnRate = 0.002f;

			std::size_t mMaxGpus = 2, mMaxNetworks = 1000;
			bool mPrintConsole = false;

			void calculateConvergence() {

				mTrainingManager.calculateNetworksConvergence(mBatchedSamplesView);
				mNetworksSorter.sortBySuperRadius();

				if (mPrintConsole) {

					std::println("Networks sorted by SuperRadius: ");

					for (std::size_t rank = 1; auto networkId : mNetworksSorter.mNetworksIds) {
						auto& network = mTrainingManager.mNetworksMap[networkId];
						std::println("Rank: {}; Network Id: {}; Misses: {}; Mse: {};", rank++, networkId, network.mMisses, network.mMse);
					}
					auto& bestNetwork = mNetworksSorter.getBest();
					auto bestNetworkId = mNetworksSorter.getBestId();
					std::println("\nRank 1 Network Id: {}; Misses: {}; Mse: {};", bestNetworkId, bestNetwork.mMisses, bestNetwork.mMse);

					mTrainingManager.calculateNetworkConvergence(mTrainingManager.getGpuTask(), bestNetwork, mBatchedSamplesView, true);
				}
			}
			void create() {

				if (mPrintConsole)
					std::println("Create XOR Lottery: ");

				using ActivationFunction = LayerTemplate::ActivationFunction;
				mNetworkTemplate = { mInputSize, mBatchSize
					, {{ 2, ActivationFunction::ReLU}
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
			void train(bool print = false) {
				mTrainingManager.trainNetworks(mTrainNum, mBatchedSamplesView, mLearnRate, print);
			}
			void run(bool print = true) {

				mPrintConsole = print;

				create();
				calculateConvergence();
				train(true);
				calculateConvergence();

				destroy();
			}
		};
	
		class LogicLottery {
		public:

			NetworksSorter mNetworksSorter;

			NetworkTemplate mNetworkTemplate;
			TrainingManager mTrainingManager;
			NetworksTracker mNetworksTracker;

			std::size_t mInputSize = 2, mOutputSize = 2
				, mTrainNum = 5000;

			std::size_t mBatchSize = 4;
			float mLearnRate = 0.002f;

			std::size_t mMaxGpus = 2, mMaxNetworks = 1000;
			bool mPrintConsole = false;
			std::string mRecordFileName = "./LogicLottery.txt";

			void clearRecord() {
				std::ofstream fout(mRecordFileName, std::ios::out);
			}
			template<typename ...Args>
			void record(const std::format_string<Args...>& format, Args&&... args) {

				std::string text;

				if constexpr (sizeof...(Args) == 0)
					text = format.get();
				else
					text = std::format(format, std::forward<Args>(args)...);
					
				std::cout << text << '\n';

				std::ofstream fout(mRecordFileName, std::ios::app);
				fout << text << '\n';
			}

			void calculateConvergence(TrainingManager::GpuBatchedSamplesView samples, const std::string& caption) {

				mTrainingManager.calculateNetworksConvergence(samples);
				mNetworksSorter.sortBySuperRadius();

				if (mPrintConsole) {
				
					record("\nConvergence Results for {}"
						"\nNetworks sorted by SuperRadius:" , caption);

					auto recordNetwork = [&]( std::size_t rank, auto& network) {
						record("Rank: {}; Id: {}; Misses: {}; Mse: {};", rank, network.mId, network.mMisses, network.mMse);
						};

					auto& networksMap = mTrainingManager.mNetworksMap;

					auto recordTopAndBottomNetworks = [&]() {

						std::size_t listSize = 5;

						auto best = mNetworksSorter.getTop(listSize);
						for (std::size_t rank = 0; auto idx : best) {
							auto& network = networksMap[idx];
							recordNetwork(++rank, network);
						}

						auto worst = mNetworksSorter.getBottom(listSize);
						for (std::size_t rank = networksMap.size() - listSize; auto idx : worst | std::views::reverse) {
							auto& network = networksMap[idx];
							recordNetwork(++rank, network);
						}
						record("\n");

						};

					auto recordBestNetwork = [&]() {

						auto& bestNetwork = mNetworksSorter.getBest();
						recordNetwork(1, bestNetwork);

						mTrainingManager.calculateNetworkConvergence(mTrainingManager.getGpuTask(), bestNetwork, samples, true);
					};
					
					auto recordZeroMisses = [&]() {

						auto zeroMissesCount = std::count_if(networksMap.begin(), networksMap.end(), [&](auto& networkPair) {

							return networkPair.second.mMisses == 0;

							});

						record("\nNetworks with zero misses: {}", zeroMissesCount);
						};


					recordTopAndBottomNetworks();
					recordBestNetwork();
					recordZeroMisses();
				}
			}
			void create() {

				if (mPrintConsole) 
					record("Create Logic Lottery:"
						"\nNetwork count: {}"
						"\nTrain Num: {}", mMaxNetworks, mTrainNum);
			
				using ActivationFunction = LayerTemplate::ActivationFunction;
				mNetworkTemplate = { mInputSize, mBatchSize
					, {{ 2, ActivationFunction::ReLU}
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

				mNetworksTracker.create(mMaxNetworks);
				mNetworksTracker.track(mTrainingManager.mNetworksMap);
			}
			void destroy() {

				if (mPrintConsole)
					std::println("destroying model");

				mTrainingManager.destroy();
			}

			void train(bool print = false) {

				auto [xorSamples, orSamples, andSamples, allSamples] = mTrainingManager.mLogicSamples.getSamples();

				mTrainingManager.trainNetworks(mTrainNum, xorSamples, mLearnRate, print);

				mNetworksTracker.track(mTrainingManager.mNetworksMap);
			}

			void calculateLogicConvergences() {

				auto [xorSamples, orSamples, andSamples, allSamples] = mTrainingManager.mLogicSamples.getSamples();

				calculateConvergence(allSamples, "All Logic Samples");
				calculateConvergence(orSamples, "OR Samples");
				calculateConvergence(andSamples, "AND Samples");
				calculateConvergence(xorSamples, "XOR Samples");
			}
			void run(bool print = true) {

				mPrintConsole = print;

				create();
				calculateLogicConvergences();
				train(true);
				calculateLogicConvergences();

				destroy();
			}
		};
	}
}