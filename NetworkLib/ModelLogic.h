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
					{{0,0}, {1,0}},
					{{0,1}, {1,0}},
					{{1,0}, {1,0}},
					{{1,1}, {0,1}}
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

			XOR(): Model("XOR.txt", 2, 1, 2, 4, 1) {}

			void calculateConvergence(bool print=false) {
				auto trueSampleNum = mLogicSamples.mTrueSampleNum;
				auto& cpuNetwork = mTrainingManager.getNetwork(mId);
				auto& gpuTask = mTrainingManager.getGpuTask();
				mTrainingManager.calculateConvergence(gpuTask, cpuNetwork, mBatchedSamplesView, trueSampleNum, false);
			}
			void create(bool print=false) {

				using ActivationFunction = LayerTemplate::ActivationFunction;
				mNetworkTemplate = { mInputWidth, mBatchSize
					, {{ 1000, ActivationFunction::ReLU}
					, { 500, ActivationFunction::ReLU}
					, { mOutputSize, ActivationFunction::Softmax}
					}
				};

				createNetwork("FC", mId, true, true, print);
				mTrainingManager.create(1);
					
				mLogicSamples.create(mNetworkTemplate);
				mBatchedSamplesView = mLogicSamples.mXORSamples;
			}       
			void destroy() {
				mLogicSamples.destroy();
				Model::destroy();
			}

			void train(std::size_t trainNum=1, bool print=false) {
				auto& gpuTask = mTrainingManager.getGpuTask();
				mTrainingManager.train(gpuTask, trainNum, mBatchedSamplesView, mLearnRate, 0, true, print);
			}

			Cpu::Network& getNetwork() {
				return mTrainingManager.getNetwork(mId);
			}

			void run(bool print=true) {

				create(print);
				calculateConvergence(print);
				train(mTrainNum, print);

				calculateConvergence(print);
				destroy();
			}
		};
		
		class XORLottery :	public LotteryModel {
		public:
			LogicSamples mSamples;

			TrainingManager::GpuBatchedSamplesView mBatchedSamplesView;

			XORLottery() : LotteryModel("XORLottery.txt", 2, 1, 2, 4, 1000, 2, 1000) {}

			void create(bool print=false) {

				using ActivationFunction = LayerTemplate::ActivationFunction;
				mNetworkTemplate = { mInputWidth, mBatchSize
					, {{ 2, ActivationFunction::ReLU}
					, { mOutputSize, ActivationFunction::Softmax}}
				};

				createNetworks("XOR", print);
				
				mSamples.create(mNetworkTemplate);
				mBatchedSamplesView = mSamples.mXORSamples;
			}
			void destroy(bool print=false) {

				if (print)
					std::println("destroying model");

				mSamples.destroy();
				LotteryModel::destroy();
			}
			void train(bool print = false) {
				mTrainingManager.trainNetworks(mTrainNum, mBatchedSamplesView, mLearnRate, 0, print);
			}
			void calculateTrainConvergence(bool print = false) {
				mTrainingManager.calculateNetworksConvergence(mBatchedSamplesView, mSamples.mTrueSampleNum, false);
				sort("Train", print);
			}

			void run(bool print = true) {

				create(print);

				calculateTrainConvergence(print);
				train(print);
				calculateTrainConvergence(print);

				destroy(print);
			}
		};
	
		class LogicLottery : LotteryModel{
		public:
			LogicSamples mSamples;

			LogicLottery() : LotteryModel("LogicLottery.txt", 2, 1, 2, 4, 5000, 2, 1000) {}

			void calculateConvergences(bool print) {

				auto trueSampleNum = mSamples.mTrueSampleNum;

				auto [xorSamples, orSamples, andSamples, allSamples] = mSamples.getSamples();

				auto sortSamples = [&](std::string_view caption, auto& samples) {
					mTrainingManager.calculateNetworksConvergence(samples, trueSampleNum, print);
					sort(caption, print);
					};

				sortSamples("XOR", xorSamples);
				sortSamples("OR", orSamples);
				sortSamples("AND", andSamples);
			}

			void create(bool print=false) {
			
				using ActivationFunction = LayerTemplate::ActivationFunction;
				mNetworkTemplate = { mInputWidth, mBatchSize
					, {{ 2, ActivationFunction::ReLU}
					, { mOutputSize, ActivationFunction::Softmax}}
				};

				createNetworks("Logic", print);
				mTrainingManager.create(mMaxGpus);

				mSamples.create(mNetworkTemplate);
			}
			void destroy(bool print=false) {

				if (print)
					std::println("destroying model");

				LotteryModel::destroy();
			}

			void train(bool print = false) {

				auto [xorSamples, orSamples, andSamples, allSamples] = mSamples.getSamples();
			
				mTrainingManager.trainNetworks(mTrainNum, xorSamples, mLearnRate, 0, print);
			}

			void run(bool print = true) {

				create(print);
				calculateConvergences(print);
				train(print);
				calculateConvergences(print);
				destroy(print);
			}
		};
	}
}