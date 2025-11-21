#pragma once
#include <random>
#include <fstream>

#include "Model.h"


namespace NetworkLib {

	namespace Model {

		struct LogicSamples {

			Gpu::LinkedFloatSpace mFloatSpace;

			TrainingManager::GpuBatchedSamples mLogicSamples;
			TrainingManager::GpuBatchedSamplesView mXORSamples, mANDSamples, mORSamples;

			std::size_t mTrueSampleNum = 0;

			struct SamplesViewGroup {
				TrainingManager::GpuBatchedSamplesView mXOR, mOR, mAND, mAll;
			};

			void create(NetworkTemplate& networkTemplate) {

				auto inputSize = networkTemplate.mInputSize
					, outputSize = networkTemplate.mLayerTemplates.back().mNodeCount;
				auto batchSize = networkTemplate.mBatchSize;
				auto sampleNum = 4 * 3; //XOR, AND, OR all have 4 samples

				mLogicSamples = TrainingManager::createGpuBatchedSamplesSpace(mFloatSpace
					, inputSize, outputSize, sampleNum, batchSize);

				mTrueSampleNum = outputSize * batchSize;

				auto begin = mFloatSpace.mGpuSpace.begin();

				TrainingManager::CpuSamples andSamples = {
					{{ 0,0 }, {1,0}},
					{{ 0,1 }, {1,0}},
					{{ 1,0 }, {1,0}},
					{{ 1,1 }, {0,1}}
				}, xorSamples = {
					{{0,0}, {1,0}},
					{{0,1}, {0,1}},
					{{1,0}, {0,1}},
					{{1,1}, {1,0}}
				}, orSamples = {
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

			SamplesViewGroup getSamples() {
				return { mXORSamples, mORSamples, mANDSamples, mLogicSamples };
			}
		};

		class XOR : public Model {
		public:
			LogicSamples mLogicSamples;
			
			TrainingManager::GpuBatchedSamplesView mBatchedSamplesView;

			std::size_t mId = 981;
			TrainingManager::GpuTask* mGpuTask = nullptr;

			XOR(): Model("mnist.txt", 2, 1, 2, 4, 1) {}

			void calculateConvergence(bool print=false) {
				auto trueSampleNum = mLogicSamples.mTrueSampleNum;
				auto& cpuNetwork = mTrainingManager.getNetwork(mId);
				mTrainingManager.calculateConvergence(*mGpuTask, cpuNetwork, mBatchedSamplesView, trueSampleNum, print);

			}
			void create() {

				using ActivationFunction = LayerTemplate::ActivationFunction;
				mNetworkTemplate = { mInputWidth, mBatchSize
					, {{ 1000, ActivationFunction::ReLU}
					, { 500, ActivationFunction::ReLU}
					, { mOutputSize, ActivationFunction::Softmax}
					}
				};

				auto createNetwork = [&]() {

					if (mPrintConsole)
						std::puts("Creating FC Network");

					mTrainingManager.addNetwork(mId);
					auto& network = mTrainingManager.getNetwork(mId);
					network.create(&mNetworkTemplate, true);
					network.initializeId(mId);

					mTrainingManager.create(1);

					mGpuTask = &mTrainingManager.getGpuTask();
					};
				createNetwork();

				mLogicSamples.create(mNetworkTemplate);
				mBatchedSamplesView = mLogicSamples.mXORSamples;

			}       
			void destroy() {
				mTrainingManager.destroy();

				mLogicSamples.destroy();
			}

			void train(std::size_t trainNum=1, bool print=false) {
				mTrainingManager.train(*mGpuTask, trainNum, mBatchedSamplesView, mLearnRate, 0, true, print);
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
			LogicSamples mLogicSamples;

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

				auto trueSampleNum = mLogicSamples.mTrueSampleNum;

				mTrainingManager.calculateNetworksConvergence(mBatchedSamplesView, trueSampleNum);
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

					auto trueSampleNum = mLogicSamples.mTrueSampleNum;

					mTrainingManager.calculateConvergence(mTrainingManager.getGpuTask(), bestNetwork, mBatchedSamplesView, trueSampleNum, true);
				
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
				
				mLogicSamples.create(mNetworkTemplate);
				mBatchedSamplesView = mLogicSamples.mXORSamples;
			}
			void destroy() {

				if (mPrintConsole)
					std::println("destroying model");

				mTrainingManager.destroy();

			}
			void train(bool print = false) {
				mTrainingManager.trainNetworks(mTrainNum, mBatchedSamplesView, mLearnRate, 0, print);
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
			LogicSamples mLogicSamples;

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

				auto trueSampleNum = mLogicSamples.mTrueSampleNum;

				mTrainingManager.calculateNetworksConvergence(samples, trueSampleNum);
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
						for (std::size_t rank = 0; auto id : best) {
							auto& network = networksMap[id];
							recordNetwork(++rank, network);
						}

						auto worst = mNetworksSorter.getBottom(listSize);
						for (std::size_t rank = networksMap.size() - listSize; auto id : worst | std::views::reverse) {
							auto& network = networksMap[id];
							recordNetwork(++rank, network);
						}
						record("\n");

						};

					auto recordBestNetwork = [&]() {

						auto& bestNetwork = mNetworksSorter.getBest();
						recordNetwork(1, bestNetwork);

						auto trueSampleNum = mLogicSamples.mTrueSampleNum;

						mTrainingManager.calculateConvergence(mTrainingManager.getGpuTask(), bestNetwork, samples, trueSampleNum, true);
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
					record("Create Logic Lottery Layout Left:"
						"\nNetwork count: {}"
						"\nTrain Num: {}", mMaxNetworks, mTrainNum);
			
				using ActivationFunction = LayerTemplate::ActivationFunction;
				mNetworkTemplate = { mInputSize, mBatchSize
					, {{ 2, ActivationFunction::ReLU}
					, { mOutputSize, ActivationFunction::Softmax}}
				};

				auto createNetworks = [&]() {

					for (auto n : std::views::iota(0ULL, mMaxNetworks))
						mTrainingManager.addNetwork();

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
				
				mLogicSamples.create(mNetworkTemplate);

				mNetworksTracker.create(mMaxNetworks);
				mNetworksTracker.track(mTrainingManager.mNetworksMap);
			}
			void destroy() {

				if (mPrintConsole)
					std::println("destroying model");

				mTrainingManager.destroy();
			}

			void train(bool print = false) {

				auto [xorSamples, orSamples, andSamples, allSamples] = mLogicSamples.getSamples();

				mTrainingManager.trainNetworks(mTrainNum, xorSamples, mLearnRate, 0, print);

				mNetworksTracker.track(mTrainingManager.mNetworksMap);
			}

			void calculateLogicConvergences() {

				auto [xorSamples, orSamples, andSamples, allSamples] = mLogicSamples.getSamples();

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