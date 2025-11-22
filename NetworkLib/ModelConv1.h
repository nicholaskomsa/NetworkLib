#pragma once
#include <random>
#include <fstream>

#include "Algorithms.h"

#include "TrainingManager.h"

#include "ModelLogic.h"

namespace NetworkLib {

	namespace Model {

		class Convolution1:	public Model {
		public:
			LogicSamples mLogicSamples;

			TrainingManager::GpuBatchedSamplesView mBatchedSamplesView;
		
			std::size_t mId = 981;

			Convolution1() : Model("Convolution1.txt", 2, 1, 2, 4, 1) {}

			void calculateConvergence(bool print=false) {
				auto trueSampleNum = mLogicSamples.mTrueSampleNum;
				auto& cpuNetwork = mTrainingManager.getNetwork(mId);
				mTrainingManager.calculateConvergence(mTrainingManager.getGpuTask(), cpuNetwork, mBatchedSamplesView, trueSampleNum, print);
			}

			void create(bool print=false) {

				auto createNetwork = [&]() {
					using ConvolutionType = LayerTemplate::ConvolutionType;
					using ActivationFunction = LayerTemplate::ActivationFunction;

					mNetworkTemplate = { mInputWidth, mBatchSize
						, {{ ConvolutionType::Conv1, 2, 2, ActivationFunction::ReLU }
						, { ConvolutionType::Conv1, 2, 2, ActivationFunction::Softmax }
						} };

					if (print)
						std::println("{}", "Creating Convolutional Network");

					mTrainingManager.addNetwork(mId);
					auto& network = mTrainingManager.getNetwork(mId);
					network.create(&mNetworkTemplate, true);
					network.initializeId(mId);

					mTrainingManager.create(1);
					};

				createNetwork();

				mLogicSamples.create(mNetworkTemplate);
				mBatchedSamplesView = mLogicSamples.mXORSamples;
			}
			void destroy() {
				mLogicSamples.destroy();
				Model::destroy();
			}

			void train(std::size_t trainNum = 1, bool print = false) {
				mTrainingManager.train(mTrainingManager.getGpuTask(), trainNum, mBatchedSamplesView, mLearnRate, 0, true, print);
			}

			Cpu::Network& getNetwork() {
				return mTrainingManager.getNetwork(mId);
			}
			void run(bool print = true) {

				create(print);
				calculateConvergence(print);
				train(mTrainNum, print);
				calculateConvergence(print);
				destroy();
			}
		};

		class Convolution1Comparison : public Convolution1 {
		public:

			void compare(XOR& fcModel) {
				
				auto& [fcGpu, fcGpuNetwork] = fcModel.mTrainingManager.getGpuTask();
				fcGpuNetwork.download(fcGpu);

				auto& [c1Gpu, c1GpuNetwork] = mTrainingManager.getGpuTask();
				c1GpuNetwork.download(c1Gpu);


				auto& fc = fcModel.mTrainingManager.getNetwork(mId);
				auto& c1 = mTrainingManager.getNetwork(mId);

				auto fcWeights1 = fc.mWeights;
				auto fcOutputs1 = fc.mOutputs;
				auto fcPrimes1 = Cpu::Tensor::flatten(fc.getPrimes());
				auto fcActivations1 = fc.mActivations;
	
				auto c1Weights1 = c1.mWeights;
				auto c1Outputs1 = c1.mOutputs;

				auto c1Primes1 = Cpu::Tensor::flatten(c1.getPrimes());
				auto c1Activations1 = c1.mActivations;

				auto weightsMse = Cpu::Network::mse(c1Weights1, fcWeights1);
				auto outputsMse = Cpu::Network::mse(c1Outputs1, fcOutputs1);
				auto primesMse = Cpu::Network::mse(c1Primes1, fcPrimes1);
				auto activationsMse = Cpu::Network::mse(c1Activations1, fcActivations1);

				auto sVec = [&](Cpu::Tensor::View1 view)->std::string {
					 std::stringstream sstr;
					 sstr << std::setprecision(std::numeric_limits<float>::max_digits10);
					 sstr << " [ ";

					 for (auto n : std::views::iota(0ULL, Cpu::Tensor::area(view)))
						sstr << view[n] << " ";

					sstr << "]";

					return sstr.str();
					};
				/*
				std::println("fc Weights: {}\nc1 Weights: {}\nMse : {}"
					"\nfc Activations: {}\nc1 Activations: {}"
					"\nfc Outputs: {}\nc1 Outputs: {}"
					"\nfc Primes: {}\nc1 Primes: {}"
					, sVec(fcWeights1), sVec(c1Weights1), weightsMse
					, sVec(fcActivations1), sVec(c1Activations1)
					, sVec(fcOutputs1), sVec(c1Outputs1)
					, sVec(fcPrimes1), sVec(c1Primes1));

					*/
				auto c1OutputsTop1 = Cpu::Tensor::flatten(c1.getLayer(0).mOutputs)
					, c1OutputsBot1 = Cpu::Tensor::flatten(c1.getLayer(1).mOutputs);

				auto c1WeightsTop1 = Cpu::Tensor::flatten(c1.getLayer(0).mWeights)
					, c1WeightsBot1 = Cpu::Tensor::flatten(c1.getLayer(1).mWeights);

				auto fcOutputsTop1 = Cpu::Tensor::flatten(fc.getLayer(0).mOutputs)
					, fcOutputsBot1 = Cpu::Tensor::flatten(fc.getLayer(1).mOutputs);

				auto fcWeightsTop1 = Cpu::Tensor::flatten(fc.getLayer(0).mWeights)
					, fcWeightsBot1 = Cpu::Tensor::flatten(fc.getLayer(1).mWeights);



				std::println("\nTop\nFc Outputs: {}\nC1 Outputs: {}\nFc Weights: {}\nC1 Weights: {}\n"
					"\nBot\nFc Outputs: {}\nC1 Outputs: {}\nFc Weights: {}\nC1 Weights: {}\n"

					, sVec(fcOutputsTop1), sVec(c1OutputsTop1), sVec(fcWeightsTop1), sVec(c1WeightsTop1)
					, sVec(fcOutputsBot1), sVec(c1OutputsBot1), sVec(fcWeightsBot1), sVec(c1WeightsBot1));
			}

			void run(bool print=true) {

				XOR xorModel;
				
				mTrainNum = 5000;

				xorModel.create(print); create(print);

				xorModel.train(mTrainNum, print); train(mTrainNum, print);

				xorModel.calculateConvergence(print); calculateConvergence(print);

				compare(xorModel);

				xorModel.destroy(); destroy();
			}
		};

		class Convolution1Lottery	:	public LotteryModel {
		public:
			LogicSamples mSamples;

			TrainingManager::GpuBatchedSamplesView mBatchedSamplesView;

			Convolution1Lottery() : LotteryModel("Conv1Lottery.txt", 2, 1, 2, 4, 1000, 2, 100) {}
			void create(bool print = false) {

				if (print)
					std::println("Create Conv1 XOR Lottery: ");

				using ConvolutionType = LayerTemplate::ConvolutionType;
				using ActivationFunction = LayerTemplate::ActivationFunction;
				mNetworkTemplate = { mInputWidth, mBatchSize
					, {{ ConvolutionType::Conv1, 2, 2, ActivationFunction::ReLU }
					, { ConvolutionType::Conv1, 2, 2, ActivationFunction::Softmax}
					}
				};

				createNetworks();

				mSamples.create(mNetworkTemplate);
				mBatchedSamplesView = mSamples.mXORSamples;
			}
			void destroy(bool print = false) {

				if (print)
					std::println("destroying model");

				mSamples.destroy();
				LotteryModel::destroy();
			}
			void train(bool print = false) {
				mTrainingManager.trainNetworks(mTrainNum, mBatchedSamplesView, mLearnRate, 0, print);
			}
			void calculateTrainConvergence(bool print = false) {

				mTrainingManager.calculateNetworksConvergence(mBatchedSamplesView
					, mSamples.mTrueSampleNum, print);

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
	}
}