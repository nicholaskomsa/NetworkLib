#pragma once
#include <random>
#include <fstream>

#include "Algorithms.h"

#include "TrainingManager.h"

#include "ModelLogic.h"

namespace NetworkLib {

	namespace Model {

		class Convolution1 {
		public:
			TrainingManager::GpuBatchedSamplesView mBatchedSamplesView;
			NetworkTemplate mNetworkTemplate;
			std::size_t mId = 981;
			TrainingManager mTrainingManager;
			TrainingManager::GpuTask* mGpuTask = nullptr;

			std::size_t mInputWidth = 2, mOutputSize = 2
				, mTrainNum = 5000;
			std::size_t kernelSize = 2;

			std::size_t mBatchSize = 4;
			float mLearnRate = 0.002f;

			bool mPrintConsole = false;

			virtual ~Convolution1() = default;

			void calculateConvergence() {

				auto& cpuNetwork = mTrainingManager.getNetwork(mId);
				mTrainingManager.calculateNetworkConvergence(*mGpuTask, cpuNetwork, mBatchedSamplesView, mPrintConsole);
			}
			void create() {

				using ConvolutionType = LayerTemplate::ConvolutionType;
				using ActivationFunction = LayerTemplate::ActivationFunction;

				//mNetworkTemplate = { mInputWidth, mBatchSize
				//	, {{ ConvolutionType::Conv1, kernelSize, 2, ActivationFunction::ReLU }
				///,  { mOutputSize, ActivationFunction::Softmax } 
				//	}};
				mNetworkTemplate = { mInputWidth, mBatchSize
					, {{ ConvolutionType::Conv1, kernelSize, 2, ActivationFunction::ReLU }
					,{ mOutputSize, ActivationFunction::Softmax }
					}};

				if (mPrintConsole) {
					std::println("{}","Creating Convolutional Network");
				}

				mTrainingManager.addNetwork(mId);
				auto& network = mTrainingManager.getNetwork(mId);
				network.create(&mNetworkTemplate, true);
				network.initializeId(mId);

				mTrainingManager.create(1);
				mTrainingManager.mLogicSamples.create(mNetworkTemplate);
				mBatchedSamplesView = mTrainingManager.mLogicSamples.mXORSamples;

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

		class Convolution1Comparison : public Convolution1 {
		public:

			void compare(XOR& fcModel) {
				
				auto& [fcGpu, fcGpuNetwork] = *fcModel.mGpuTask;
				fcGpuNetwork.download(fcGpu);

				auto& [c1Gpu, c1GpuNetwork] = *mGpuTask;
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

				std::println("fc Weights: {}\nc1 Weights: {}\nMse : {}"
					"\nfc Activations: {}\nc1 Activations: {}"
					"\nfc Outputs: {}\nc1 Outputs: {}"
					"\nfc Primes: {}\nc1 Primes: {}"
					, sVec(fcWeights1), sVec(c1Weights1), weightsMse
					, sVec(fcActivations1), sVec(c1Activations1)
					, sVec(fcOutputs1), sVec(c1Outputs1)
					, sVec(fcPrimes1), sVec(c1Primes1));
			}


			virtual ~Convolution1Comparison() = default;

			void run(bool print = true) override {
				

				mPrintConsole = print;

				XOR xorModel;
				
				xorModel.mPrintConsole = print;
				xorModel.create();
				xorModel.calculateConvergence();
				xorModel.train(1, true);
				xorModel.calculateConvergence();
				
				create();
				calculateConvergence();
				train(1, true);
				calculateConvergence();

				compare(xorModel);

				xorModel.destroy();
				destroy();
			}
		};

		class Convolution1Lottery {
		public:
			TrainingManager::GpuBatchedSamplesView mBatchedSamplesView;
			NetworkTemplate mNetworkTemplate;
			std::size_t mId = 981;
			TrainingManager mTrainingManager;
			NetworksSorter mNetworksSorter;

			std::size_t mInputWidth = 2, mOutputSize = 2
				, mTrainNum = 5000;
			std::size_t kernelSize = 2;

			std::size_t mBatchSize = 4;
			float mLearnRate = 0.002f;

			std::size_t mMaxGpus = 2, mMaxNetworks = 1000;

			bool mPrintConsole = false;
		
			void calculateConvergence() {

				mTrainingManager.calculateNetworksConvergence(mBatchedSamplesView);
				mNetworksSorter.sortBySuperRadius();

				if (mPrintConsole) {

					auto& networksMap = mTrainingManager.mNetworksMap;

					std::println("Networks sorted by SuperRadius: ");

					for (std::size_t rank = 1; auto networkId : mNetworksSorter.mNetworksIds) {
						auto& network = mTrainingManager.mNetworksMap[networkId];
						std::println("Rank: {}; Network Id: {}; Misses: {}; Mse: {};", rank++, networkId, network.mMisses, network.mMse);
					}
					auto& bestNetwork = mNetworksSorter.getBest();
					auto bestNetworkId = mNetworksSorter.getBestId();
					std::println("\nRank 1 Network Id: {}; Misses: {}; Mse: {};", bestNetworkId, bestNetwork.mMisses, bestNetwork.mMse);

					mTrainingManager.calculateNetworkConvergence(mTrainingManager.getGpuTask(), bestNetwork, mBatchedSamplesView, true);

					auto printZeroMisses = [&]() {

						auto zeroMissesCount = std::count_if(networksMap.begin(), networksMap.end(), [&](auto& networkPair) {

							return networkPair.second.mMisses == 0;

							});

						std::println("\nNetworks with zero misses: {}", zeroMissesCount);
						};
					printZeroMisses();
				}
			}
			void create() {

				using ConvolutionType = LayerTemplate::ConvolutionType;
				using ActivationFunction = LayerTemplate::ActivationFunction;

				mNetworkTemplate = { mInputWidth, mBatchSize
					, {{ ConvolutionType::Conv1, kernelSize, 2, ActivationFunction::ReLU }
					, { mOutputSize, ActivationFunction::Softmax}}
				};

				auto createNetworks = [&]() {

					for (auto id : std::views::iota(0ULL, mMaxNetworks))
						mTrainingManager.addNetwork(id);

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

					mTrainingManager.create(mMaxGpus);

					};

				createNetworks();

				mTrainingManager.mLogicSamples.create(mNetworkTemplate);
				mBatchedSamplesView = mTrainingManager.mLogicSamples.mXORSamples;
			}
			void destroy() {
				if (mPrintConsole)
					std::println("destroying model");

				mTrainingManager.destroy();
			}

			void train(std::size_t trainNum = 1, bool print = false) {
				mTrainingManager.trainNetworks(trainNum, mBatchedSamplesView, mLearnRate, print);
			}

			Cpu::Network& getNetwork() {
				return mTrainingManager.getNetwork(mId);
			}
			void run(bool print = true) {

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