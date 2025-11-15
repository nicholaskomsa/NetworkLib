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
				auto trueSampleNum = mTrainingManager.mLogicSamples.mTrueSampleNum;
				auto& cpuNetwork = mTrainingManager.getNetwork(mId);
				mTrainingManager.calculateConvergence(*mGpuTask, cpuNetwork, mBatchedSamplesView, trueSampleNum, mPrintConsole);
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
					, { ConvolutionType::Conv1, kernelSize, 2, ActivationFunction::Softmax }
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
				mTrainingManager.train(*mGpuTask, trainNum, mBatchedSamplesView, mLearnRate, 0, true, print);
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

			virtual ~Convolution1Comparison() = default;

			void run(bool print = true) override {


				mPrintConsole = print;
				XOR xorModel;
				xorModel.mPrintConsole = print;


				mTrainNum = 5000;

				xorModel.create(); create();

				xorModel.train(mTrainNum, true); train(mTrainNum, true);

				xorModel.calculateConvergence(); calculateConvergence();

				compare(xorModel);

				xorModel.destroy(); destroy();
			}
		};

		class Convolution1Lottery {
		public:
			TrainingManager::GpuBatchedSamplesView mBatchedSamplesView;
			NetworkTemplate mNetworkTemplate;
			std::size_t mId = 981;
			TrainingManager mTrainingManager;
			NetworksSorter mNetworksSorter;
			NetworksTracker mNetworksTracker;

			std::size_t mInputWidth = 2, mOutputSize = 2
				, mTrainNum = 5000;
			std::size_t kernelSize = 2;

			std::size_t mBatchSize = 4;
			float mLearnRate = 0.002f;

			std::size_t mMaxGpus = 2, mMaxNetworks = 1000;

			bool mPrintConsole = false;

			std::string mRecordFileName = "./LogicLottery.txt";

			void clearRecord() {
				std::ofstream fout(mRecordFileName, std::ios::out);
			}
			template<typename ...Args>
			void record(const std::format_string<Args...>& format, Args&&... args) {

				std::string text;

				if constexpr (sizeof...(Args) == 0)
					text = format.get();
				else
					text = std::format(format, std::forward<Args>(args)...);

				std::cout << text << '\n';

				std::ofstream fout(mRecordFileName, std::ios::app);
				fout << text << '\n';
			}

			void calculateConvergence(TrainingManager::GpuBatchedSamplesView samples, const std::string& caption) {

				auto trueSampleNum = mTrainingManager.mLogicSamples.mTrueSampleNum;
				mTrainingManager.calculateNetworksConvergence(samples, trueSampleNum);
				mNetworksSorter.sortBySuperRadius();

				if (mPrintConsole) {

					record("\nConvergence Results for {}"
						"\nNetworks sorted by SuperRadius:", caption);

					auto recordNetwork = [&](std::size_t rank, auto& network) {
						record("Rank: {}; Id: {}; Misses: {}; Mse: {};", rank, network.mId, network.mMisses, network.mMse);
						};

					auto& networksMap = mTrainingManager.mNetworksMap;

					auto recordTopAndBottomNetworks = [&]() {

						std::size_t listSize = 5;

						auto best = mNetworksSorter.getTop(listSize);
						for (std::size_t rank = 0; auto id : best) {
							auto& network = networksMap[id];
							recordNetwork(++rank, network);
						}

						auto worst = mNetworksSorter.getBottom(listSize);
						for (std::size_t rank = networksMap.size() - listSize; auto id : worst | std::views::reverse) {
							auto& network = networksMap[id];
							recordNetwork(++rank, network);
						}
						record("\n");

						};

					auto recordBestNetwork = [&]() {

						auto& bestNetwork = mNetworksSorter.getBest();
						recordNetwork(1, bestNetwork);

						auto trueSampleNum = mTrainingManager.mLogicSamples.mTrueSampleNum;

						mTrainingManager.calculateConvergence(mTrainingManager.getGpuTask(), bestNetwork, samples, trueSampleNum, true);
						};

					auto recordZeroMisses = [&]() {

						auto zeroMissesCount = std::count_if(networksMap.begin(), networksMap.end(), [&](auto& networkPair) {

							return networkPair.second.mMisses == 0;

							});

						record("\nNetworks with zero misses: {}", zeroMissesCount);
						};


					recordTopAndBottomNetworks();
					recordBestNetwork();
					recordZeroMisses();
				}
			}

			void create() {

				if (mPrintConsole)
					record("Create Logic Lottery:"
						"\nNetwork count: {}"
						"\nTrain Num: {}", mMaxNetworks, mTrainNum);

				using ConvolutionType = LayerTemplate::ConvolutionType;
				using ActivationFunction = LayerTemplate::ActivationFunction;

				mNetworkTemplate = { mInputWidth, mBatchSize
					, {{ ConvolutionType::Conv1, kernelSize, 2, ActivationFunction::ReLU }
					, { ConvolutionType::Conv1,kernelSize, 2, ActivationFunction::Softmax}
					}
				};

				auto createNetworks = [&]() {

					for (auto n : std::views::iota(0ULL, mMaxNetworks))
						mTrainingManager.addNetwork();

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

				mNetworksTracker.create(mMaxNetworks);
				mNetworksTracker.track(mTrainingManager.mNetworksMap);
			}
			void destroy() {

				if (mPrintConsole)
					std::println("destroying model");

				mTrainingManager.destroy();
			}

			void train(bool print = false) {

				auto [xorSamples, orSamples, andSamples, allSamples] = mTrainingManager.mLogicSamples.getSamples();

				mTrainingManager.trainNetworks(mTrainNum, xorSamples, mLearnRate, 0, print);

				mNetworksTracker.track(mTrainingManager.mNetworksMap);
			}

			void calculateLogicConvergences() {

				auto [xorSamples, orSamples, andSamples, allSamples] = mTrainingManager.mLogicSamples.getSamples();

				calculateConvergence(allSamples, "All Logic Samples");
				calculateConvergence(orSamples, "OR Samples");
				calculateConvergence(andSamples, "AND Samples");
				calculateConvergence(xorSamples, "XOR Samples");
			}
			void run(bool print = true) {

				mPrintConsole = print;

				create();
				calculateLogicConvergences();
				train(true);
				calculateLogicConvergences();

				destroy();
			}
		};
	}
}