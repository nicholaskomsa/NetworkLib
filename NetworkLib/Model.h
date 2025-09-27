#pragma once
#include <random>

#include "Algorithms.h"

#include "Environment.h"
#include "TrainingManager.h"
#include "GpuNetwork.h"

namespace NetworkLib {

	namespace Model {

		class XOR {
		public:

			std::mt19937_64 mRandom;
			Gpu::Environment mGpu;
			TrainingManager::GpuBatchedSamples mBatchedSamples;
			Gpu::Network mNetwork;
			NetworkTemplate mNetworkTemplate;
			TrainingManager mTManager;

			std::size_t mInputSize = 2, mOutputSize = 2
				, mTrainNum = 5000;

			std::size_t mBatchSize = 4;
			float mLearnRate = 0.002f;

			bool mPrintConsole = false;

			void calculateConvergence() {

				mGpu.resetMseResult();

				for (const auto& [seen, desired] : mBatchedSamples) {

					auto sought = mNetwork.forward(mGpu, seen);
					auto output = mNetwork.getOutput();

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

				mNetwork.create(&mNetworkTemplate);
				mNetwork.initialize(mRandom);
				mNetwork.upload();

				mBatchedSamples = mTManager.createXORBatchedSamples(mBatchSize);
			}
			void destroy() {
				mTManager.destroy();
				mNetwork.destroy();
				mGpu.destroy();
			}

			void trainOne(std::size_t generation) {

				TimeAverage<microseconds> trainTime;

				trainTime.accumulateTime([&]() {

					const auto& [seen, desired] = mBatchedSamples[generation % mBatchedSamples.size()];

					mNetwork.forward(mGpu, seen);
					mNetwork.backward(mGpu, seen, desired, mLearnRate);

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

	}

}