#pragma once

#include <fstream>
#include <iostream>

#include "TrainingManager.h"

namespace NetworkLib {

	namespace Model {

		class Model {
		public:
			std::string mRecordFileName = "./Model.txt";
			NetworkTemplate mNetworkTemplate;
			TrainingManager mTrainingManager;
			std::size_t mInputWidth = 0, mInputHeight = 0, mOutputSize = 0
				, mBatchSize = 0
				, mTrainNum = 0;
			float mLearnRate = 0.002f;

			Model(std::string_view recordFileName
				, std::size_t inputWidth=1, std::size_t inputHeight=1, std::size_t outputSize =1
				, std::size_t batchSize = 1, std::size_t trainNum=1)
				: mRecordFileName(recordFileName)
				, mInputWidth(inputWidth), mInputHeight(inputHeight), mOutputSize(outputSize)
				, mBatchSize( batchSize), mTrainNum(trainNum)
			{
				clearRecord();
			}

			void destroy() {
				mTrainingManager.destroy();
			}

			void createNetwork(std::string_view caption, Cpu::Network::Id id, bool backwards=true, bool initialize=true, bool print=false) {

				if (print)
					record("Creating {} Network: ID = {}", caption, id);

				mTrainingManager.addNetwork(id);
				auto& network = mTrainingManager.getNetwork(id);
				network.create(&mNetworkTemplate, backwards);

				if( initialize)
					network.initializeId(id);
			}

			void clearRecord() {
				std::ofstream{ mRecordFileName, std::ios::out };
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

			void recordNetwork(std::size_t rank, auto& network) {
				record("Rank: {}; Id: {}; Misses: {}; Mse: {};", rank, network.mId, network.mMisses, network.mMse);
			}
		};
		
		class LotteryModel : public Model {
		public:

			NetworksSorter mNetworksSorter;
			std::size_t mMaxGpus = 2, mMaxNetworks = 1000;

			LotteryModel(std::string_view recordFileName
				, std::size_t inputWidth, std::size_t inputHeight, std::size_t outputSize
				, std::size_t batchSize
				, std::size_t trainNum
				, std::size_t maxGpus, std::size_t maxNetworks)
				: Model(recordFileName, inputWidth, inputHeight, outputSize, batchSize, trainNum)
				, mMaxGpus(maxGpus), mMaxNetworks(maxNetworks)
			{}
			
			~LotteryModel() = default;

			void recordTopAndBottomNetworks(std::size_t listSize = 5) {

				auto& networksMap = mTrainingManager.mNetworksMap;

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
			}

			void recordZeroMisses(){

				auto& networksMap = mTrainingManager.mNetworksMap;

				auto zeroMissesCount = std::count_if(networksMap.begin(), networksMap.end(), [&](auto& networkPair) {
					return networkPair.second.mMisses == 0;
					});

				record("Networks with zero misses: {}", zeroMissesCount);
			}
			void createNetworks(std::string_view caption, bool print=false) {

				if (print)
					record("Create {} Lottery:\tNetwork count: {};\tGpu count: {};" , caption, mMaxNetworks, mMaxGpus);

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


			}

			void sort(std::string_view caption = "", bool print = false) {

				mNetworksSorter.sortBySuperRadius();

				if (!print) return;

				record("\nConvergence Results for {}"
					"\nNetworks sorted by SuperRadius:", caption);

				recordTopAndBottomNetworks();
				recordZeroMisses();
			}
		};
	}
}