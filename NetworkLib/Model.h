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
			TrainingManager::GpuBatchedSamplesView mBatchedSamplesView;
			NetworkTemplate mNetworkTemplate;
			std::size_t mId = 981;
			TrainingManager mTrainingManager;
			TrainingManager::GpuTask* mGpuTask = nullptr;

			std::size_t mInputWidth = 28, mInputHeight = 28, mOutputSize = 10
				, mTrainNum = 5000;

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
					, {{ 100, ActivationFunction::ReLU }
					, { 50, ActivationFunction::ReLU }
					, { 10, ActivationFunction::Softmax }}
				};

				if (mPrintConsole) {
					std::println("{}", "Creating MNIST Network");
				}

				mTrainingManager.addNetwork(mId);
				auto& network = mTrainingManager.getNetwork(mId);
				network.create(&mNetworkTemplate, true);
				network.initializeId(mId);

				mTrainingManager.create(1);
				mTrainingManager.mMNISTSamples.create(mNetworkTemplate);
				mBatchedSamplesView = mTrainingManager.mMNISTSamples.mMNISTSamples;

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
			virtual void run(bool print = true) {

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