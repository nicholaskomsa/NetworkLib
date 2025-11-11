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
			TrainingManager::GpuSamplesView mTestSamplesView;
			
			NetworkTemplate mNetworkTemplate;
			std::size_t mId = 981;
			TrainingManager mTrainingManager;
			TrainingManager::GpuTask* mGpuTask = nullptr;

			std::size_t mInputWidth = 28, mInputHeight = 28, mOutputSize = 10
				, mTrainNum = 1;

			std::size_t mBatchSize = 4;
			float mLearnRate = 0.002f;

			bool mPrintConsole = false;

			void calculateConvergence(const auto& testSamples) {

				auto& cpuNetwork = mTrainingManager.getNetwork(mId);
				mTrainingManager.calculateNetworkConvergence(*mGpuTask, cpuNetwork, testSamples, false);

				auto samplesNum = testSamples.size();
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
				mTestSamplesView = mnistSamples.mTestSamples;

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
				calculateConvergence(mTestSamplesView);

				auto totalTrainNum = mTrainNum * mBatchSize * mTrainBatched2SamplesView.size();
				train(totalTrainNum, true);

				calculateConvergence(mTestSamplesView);
				destroy();
			}
		};
	}
}