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
			TrainingManager::GpuBatched2SamplesView mTrainBatched2SamplesView, mTestBatched2SamplesView;
			NetworkTemplate mNetworkTemplate;
			std::size_t mId = 981;
			TrainingManager mTrainingManager;
			TrainingManager::GpuTask* mGpuTask = nullptr;

			std::size_t mInputWidth = 28, mInputHeight = 28, mOutputSize = 10
				, mTrainNum = 1;

			std::size_t mBatchSize = 4;
			float mLearnRate = 0.002f;

			bool mPrintConsole = false;

			void calculateConvergence(const auto& batched2SamplesView) {

				auto& cpuNetwork = mTrainingManager.getNetwork(mId);
				mTrainingManager.calculateNetworkConvergence(*mGpuTask, cpuNetwork, batched2SamplesView, false);

				auto samplesNum = batched2SamplesView.size()* mBatchSize;
				auto accuracy = (samplesNum - cpuNetwork.mMisses) / float(samplesNum) * 100.0f;

				std::println("Id: {}; Misses: {}; Accuracy: {:.2}%; Mse: {};", cpuNetwork.mId, cpuNetwork.mMisses, accuracy, cpuNetwork.mMse);
			}
			void create() {

				using ConvolutionType = LayerTemplate::ConvolutionType;
				using ActivationFunction = LayerTemplate::ActivationFunction;

				mNetworkTemplate = { mInputWidth*mInputHeight, mBatchSize
					, {{ 100, ActivationFunction::ReLU }
					, { 50, ActivationFunction::ReLU }
					, { mOutputSize, ActivationFunction::Softmax }}
				};

				if (mPrintConsole) 
					std::puts("Creating MNIST Network");

				mTrainingManager.addNetwork(mId);
				auto& network = mTrainingManager.getNetwork(mId);
				network.create(&mNetworkTemplate, true);
				network.initializeId(mId);

				mTrainingManager.create(1);

				auto& mnistSamples = mTrainingManager.mMNISTSamples;

				mnistSamples.create(mNetworkTemplate);
				
				mTrainBatched2SamplesView = mnistSamples.mTrainBatched2Samples;
				mTestBatched2SamplesView = mnistSamples.mTestBatched2Samples;

				mGpuTask = &mTrainingManager.getGpuTask();
			}
			void destroy() {
				mTrainingManager.destroy();
			}

			void train(std::size_t trainNum = 1, bool print = false) {
				mTrainingManager.train(*mGpuTask, trainNum, mTrainBatched2SamplesView, mLearnRate, 0, print);
			}

			Cpu::Network& getNetwork() {
				return mTrainingManager.getNetwork(mId);
			}
			void run(bool print = true) {

				mPrintConsole = print;

				create();
				calculateConvergence(mTestBatched2SamplesView);

				auto totalTrainNum = mTrainNum * mBatchSize * mTrainBatched2SamplesView.size();
				train(totalTrainNum, true);

				calculateConvergence(mTestBatched2SamplesView);
				destroy();
			}
		};
	}
}