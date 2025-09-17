#pragma once
#include <random>

#include "Algorithms.h"

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

			void calculateConvergence() {

				mGpu.resetMseResult();

				for (const auto& [seen, desired] : mBatchedSamples) {

					auto sought = mNetwork.forward(mGpu, seen);

					mGpu.mse(sought, desired);

					sought.downloadAsync(mGpu);
					
					mGpu.sync();

					std::println("\nseen: {}"
						"\ndesired: {}"
						"\nsought: {}"
						, seen
						, desired
						, sought
					);
				}
				std::println("mse: {}\n", mGpu.getMseResult() );
			}
			void create() {

				mGpu.create();

				using ActivationFunction = LayerTemplate::ActivationFunction;
				mNetworkTemplate = { mInputSize, mBatchSize
					, {{ 7, ActivationFunction::ReLU}
					, { 4, ActivationFunction::ReLU}
					, { mOutputSize, ActivationFunction::None}}
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
			void train() {

				TimeAverage<microseconds> trainTime;

				std::print("Training: ");

				for (auto generation : std::views::iota(0ULL, mTrainNum))
					trainTime.accumulateTime([&]() {

						const auto& [seen, desired] = mBatchedSamples[generation % mBatchedSamples.size()];

						mNetwork.forward(mGpu, seen);
						mNetwork.backward(mGpu, seen, desired, mLearnRate);

						printProgress(generation, mTrainNum);

						});

			}

			void run() {
				create();
				calculateConvergence();
				train();
				calculateConvergence();
				destroy();
			}
		};

	}

}