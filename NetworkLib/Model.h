#pragma once
#include <random>
#include <fstream>

#include "Algorithms.h"

#include "TrainingManager.h"

#include "ModelLogic.h"

namespace NetworkLib {

	namespace Model {

		class MNIST {
		public:
			TrainingManager::GpuBatched2SamplesView mTrainBatched2SamplesView;
			TrainingManager::GpuBatched3SamplesView mTestBatched3SamplesView;
			
			Gpu::GpuView2 mTestBatched3DesiredGroup;

			NetworkTemplate mNetworkTemplate;
			std::size_t mId = 981;
			std::size_t mConvergenceNetworkId = 0;

			TrainingManager mTrainingManager;
			TrainingManager::GpuTask* mGpuTaskTrain = nullptr, * mGpuTaskConvergence = nullptr;
			std::mutex mNetworkMutex;

			std::size_t mInputWidth = 28, mInputHeight = 28, mOutputSize = 10
				, mTrainNum = 1;

			std::size_t mBatchSize = 1;
			float mLearnRate = 0.002f;

			bool mPrintConsole = false;

			MNIST() = default;
			MNIST& operator()(const MNIST&) = delete;

			void calculateConvergence(bool print=false) {

				auto& trainNetwork = mTrainingManager.getNetwork(mId);
				auto& ccNetwork = mTrainingManager.getNetwork(mConvergenceNetworkId);

				//we calculate convergence on a diff gpu so copy to it
				auto copyToConvergenceNetwork = [&]() {
					std::scoped_lock lock(mNetworkMutex);
					//copy train network to convergence network
					ccNetwork.mirror(trainNetwork);
					};

				time<milliseconds>("Calculating Convergence", [&]() {

					copyToConvergenceNetwork();


					auto trueSampleNum = mTrainingManager.mMNISTSamples.mTrueTestSamplesNum;

					mTrainingManager.calculateConvergence(*mGpuTaskConvergence, ccNetwork
						, mTestBatched3SamplesView, trueSampleNum, mTestBatched3DesiredGroup, print);

					}, print);
			}
			void create() {

				using ConvolutionType = LayerTemplate::ConvolutionType;
				using ActivationFunction = LayerTemplate::ActivationFunction;

				mNetworkTemplate = { mInputWidth*mInputHeight, mBatchSize
					, {{ 1000, ActivationFunction::ReLU }
					, { mOutputSize, ActivationFunction::Softmax }}
				};

				if (mPrintConsole) 
					std::puts("Creating MNIST Network");

				mTrainingManager.addNetwork(mId);
				auto& network = mTrainingManager.getNetwork(mId);
				network.create(&mNetworkTemplate, true);
				network.initializeId(mId);

				mTrainingManager.addNetwork(mConvergenceNetworkId);
				auto& ccnetwork = mTrainingManager.getNetwork(mConvergenceNetworkId);
				ccnetwork.create(&mNetworkTemplate, true);

				mTrainingManager.create(2);

				auto& mnistSamples = mTrainingManager.mMNISTSamples;

				mnistSamples.create(mNetworkTemplate);
				
				mTrainBatched2SamplesView = mnistSamples.mTrainBatched2Samples;
				mTestBatched3SamplesView = mnistSamples.mTestBatched3Samples;
				mTestBatched3DesiredGroup = mnistSamples.mOutputs;

				mGpuTaskTrain = &mTrainingManager.getGpuTask(0);
				mGpuTaskConvergence = &mTrainingManager.getGpuTask(1);
			}
			void destroy() {
				mTrainingManager.destroy();
			}

			void train(std::size_t trainNum = 1, std::size_t offset =0, bool print = false) {

				mTrainingManager.train(*mGpuTaskTrain, trainNum, mTrainBatched2SamplesView, mLearnRate, offset, false, print);

				//we need to synchronize with cc because that thread wants this data
				std::scoped_lock lock(mNetworkMutex);
				mGpuTaskTrain->mGpuNetwork.download(mGpuTaskTrain->mGpu);
			}

			Cpu::Network& getNetwork() {
				return mTrainingManager.getNetwork(mId);
			}
			Cpu::Network& getConvergenceNetwork() {
				return mTrainingManager.getNetwork(mConvergenceNetworkId);
			}
			void run(bool print = true) {

				mPrintConsole = print;

				create();
				calculateConvergence(true);

				auto totalTrainNum = mTrainNum * mBatchSize * mTrainBatched2SamplesView.size();
				train(totalTrainNum, 0, true);

				calculateConvergence(true);
				destroy();
			}
		};
	}
}