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
			TrainingManager::GpuBatchedSamplesView mBatchedSamplesView;
			Cpu::Network mCpuNetwork;
			NetworkTemplate mNetworkTemplate;
			TrainingManager mTrainingManager;
			TrainingManager::GpuTask* mGpuTask = nullptr;

			std::size_t mInputSize = 2, mOutputSize = 2
				, mTrainNum = 5000;

			std::size_t mBatchSize = 4;
			float mLearnRate = 0.002f;

			bool mPrintConsole = false;

			void calculateConvergence() {

				mTrainingManager.calculateNetworkConvergence(*mGpuTask, 0, mBatchedSamplesView, mPrintConsole);

			}
			void create() {

				using ActivationFunction = LayerTemplate::ActivationFunction;
				mNetworkTemplate = { mInputSize, mBatchSize
					, {{ 1000, ActivationFunction::ReLU}
					, { 500, ActivationFunction::ReLU}
					, { 250, ActivationFunction::ReLU }
					, { mOutputSize, ActivationFunction::Softmax}}
				};

				mCpuNetwork.create(&mNetworkTemplate);
				mCpuNetwork.initializeId(981);
				
				mTrainingManager.create(1, mNetworkTemplate, { &mCpuNetwork, 1 });
				mTrainingManager.mLogicSamples.create(mNetworkTemplate);
				mBatchedSamplesView = mTrainingManager.mLogicSamples.mXORSamples;
				
				mGpuTask = &mTrainingManager.getGpuTask();
			}
			void destroy() {
				mTrainingManager.destroy();
				mCpuNetwork.destroy();
			}

			void train(std::size_t trainNum=1, bool print=false) {
				mTrainingManager.train(*mGpuTask, trainNum, mBatchedSamplesView, mLearnRate, print);
			}

			void run(bool print=true) {

				mPrintConsole = print;

				create();
				calculateConvergence();
				train(mTrainNum);

				calculateConvergence();
				destroy();
			}
		};
		
		class XORLottery {
		public:

			Cpu::Networks mNetworks;
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

					for (std::size_t i = 0; auto networkIdx : mNetworksSorter.mNetworksIdx) {
						auto& network = mNetworks[networkIdx];
						std::println("Rank: {}; Network Id: {}; Misses: {}; Mse: {};", ++i, networkIdx, network.mMisses, network.mMse);
					}
				}

				auto& bestNetwork = mNetworksSorter.getBest();
				auto bestNetworkIdx = mNetworksSorter.getBestIdx();
				std::println("\nRank 1 Network Id: {}; Misses: {}; Mse: {};", bestNetworkIdx, bestNetwork.mMisses, bestNetwork.mMse);

				mTrainingManager.calculateNetworkConvergence(mTrainingManager.getGpuTask(), bestNetworkIdx, mBatchedSamplesView, true);
			}
			void create() {

				if (mPrintConsole)
					std::println("Create XOR Lottery: ");

				using ActivationFunction = LayerTemplate::ActivationFunction;
				mNetworkTemplate = { mInputSize, mBatchSize
					, {{ 100, ActivationFunction::ReLU}
					, { mOutputSize, ActivationFunction::Softmax}}
				};

				mNetworks.resize(mMaxNetworks);
				mNetworksSorter.create(mNetworks);

				Parallel parallelNetworks(mNetworks.size());

				parallelNetworks([&](auto& section) {

					for (auto id : section.mIotaView) {

						auto& network = mNetworks[id];

						network.create(&mNetworkTemplate, true);

						network.initializeId(id);
					}
					});

				mTrainingManager.create(mMaxGpus, mNetworkTemplate, mNetworks);
				mTrainingManager.mLogicSamples.create(mNetworkTemplate);
				mBatchedSamplesView = mTrainingManager.mLogicSamples.mXORSamples;
			}
			void destroy() {

				if (mPrintConsole)
					std::println("destroying model");

				mTrainingManager.destroy();
				for (auto& network : mNetworks)
					network.destroy();

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
	}
}