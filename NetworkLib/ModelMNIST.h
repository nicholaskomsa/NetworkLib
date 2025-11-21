#pragma once

#include "Model.h"

namespace NetworkLib {

	namespace Model {

		struct MNISTSamples {

			std::string mMNISTFolder = "./mnist/";

			Gpu::LinkedFloatSpace mFloatSpace;

			TrainingManager::GpuBatched2Samples mTrainBatched2Samples;
			TrainingManager::GpuBatched3Samples mTestBatched3Samples;
			Gpu::GpuView2 mOutputs;
			std::size_t mTrueTrainSamplesNum, mTrueTestSamplesNum;

			using Images = std::vector<float>;
			using ImageView = std::span<float>;
			using DigitImagesMap = std::map<std::uint8_t, std::vector<ImageView>>;
			Images mTrainDigitsSamples, mTestDigitsSamples;
			DigitImagesMap mTestSamplesMap, mTrainSamplesMap;

			std::size_t getSamplesNum(const DigitImagesMap& samplesMap) {
				return std::accumulate(samplesMap.begin(), samplesMap.end(), 0UL
					, [](auto sum, const auto& pair) {
						return sum + pair.second.size();
					});
			}

			void loadAllDigitsSamples() {

				std::string testImagesFileName = "t10k-images.idx3-ubyte"
					, testLabelsFileName = "t10k-labels.idx1-ubyte"
					, trainImagesFileName = "train-images.idx3-ubyte"
					, trainLabelsFileName = "train-labels.idx1-ubyte";

				std::tie(mTrainSamplesMap, mTrainDigitsSamples) = loadDigitsSamples(trainImagesFileName, trainLabelsFileName);
				std::tie(mTestSamplesMap, mTestDigitsSamples) = loadDigitsSamples(testImagesFileName, testLabelsFileName);
			}
			std::pair<DigitImagesMap, Images> loadDigitsSamples(const std::string& imagesFileName, const std::string& labelsFileName) {

				using HeadingType = std::uint32_t;
				using LabelType = std::uint8_t;
				using Image1Type = std::uint8_t;
				using Image1 = std::vector<Image1Type>;

				auto readHeading = [](auto& filestream, HeadingType& h) {

					filestream.read(reinterpret_cast<char*>(&h), sizeof(h));
					h = std::byteswap(h);
					};

				auto checkMagic = [&](auto& filestream, HeadingType magic) {

					HeadingType magicHeading;
					readHeading(filestream, magicHeading);

					if (magicHeading != magic)
						throw std::logic_error("Incorrect magic heading");
					};

				constexpr auto binaryIn = std::ios::in | std::ios::binary;
				std::ifstream finImages(mMNISTFolder + imagesFileName, binaryIn)
					, finLabels(mMNISTFolder + labelsFileName, binaryIn);

				if (finImages.fail())
					throw std::runtime_error(std::format("failed to open {}", imagesFileName));
				if (finLabels.fail())
					throw std::runtime_error(std::format("failed to open {}", labelsFileName));

				HeadingType numImages = 0, numLabels = 0
					, rows = 0, cols = 0;

				checkMagic(finImages, 2051);
				checkMagic(finLabels, 2049);

				readHeading(finImages, numImages);
				readHeading(finLabels, numLabels);

				if (numImages != numLabels)
					throw std::logic_error("image num should equal label num");

				readHeading(finImages, rows);
				readHeading(finImages, cols);

				auto imageSize = rows * cols;

				LabelType label;
				Image1 image1(imageSize);
				DigitImagesMap digitsMap;
				Images images;
				constexpr float normalizeFactor = std::numeric_limits<Image1Type>::max();

				images.resize(imageSize * numImages);
				auto imagesIt = images.begin();

				std::println("Reading mnist x {} images", numImages);
				for (auto i : std::views::iota(0UL, numImages)) {

					finImages.read(reinterpret_cast<char*>(image1.data()), imageSize * sizeof(Image1Type));
					finLabels.read(reinterpret_cast<char*>(&label), sizeof(label));

					ImageView floatImage(imagesIt, imageSize);

					std::transform(image1.begin(), image1.end(), floatImage.begin()
						, [](auto i) { return i / normalizeFactor; });

					digitsMap[label].push_back(floatImage);

					std::advance(imagesIt, imageSize);
				}
				return { std::move(digitsMap), std::move(images) };
			}

			void create(NetworkTemplate& networkTemplate) {

				auto inputSize = networkTemplate.mInputSize
					, outputSize = networkTemplate.mLayerTemplates.back().mNodeCount;
				auto batchSize = networkTemplate.mBatchSize;

				constexpr std::size_t outputNum = 10; //digits 0-9

				loadAllDigitsSamples();

				auto copyToGpu = [&]() {

					//training samples need to be valid for an entire batch2 for training so
					//to complete all training data in this fashion (digit-stripe), some may be duplicated with %
					//until all images are included

					//test data has a batched3 type so that it fits with no duplication


					std::size_t trainSamplesNum = getSamplesNum(mTrainSamplesMap)
						, testSamplesNum = getSamplesNum(mTestSamplesMap)
						, trainBatchNum = std::ceil(trainSamplesNum / float(batchSize))
						, testBatchNum = std::ceil(testSamplesNum / float(batchSize))
						, outputClassesSize = outputSize * outputNum
						, trainInputsSize = trainBatchNum * batchSize * inputSize
						, testInputsSize = testBatchNum * batchSize * inputSize
						, testOutputBatchSize = testBatchNum * batchSize * 1;

					mTrainBatched2Samples.resize(trainBatchNum);
					mTestBatched3Samples.resize(testBatchNum);

					mFloatSpace.create(outputClassesSize + trainInputsSize + testInputsSize + testOutputBatchSize);
					auto& gpuSampleSpace = mFloatSpace.mGpuSpace;
					auto gpuSampleSpaceIt = gpuSampleSpace.begin();

					auto createOneHotOutputViews = [&]() {

						//first, setup the GpuOutputs for each digit as one hot output
						gpuSampleSpace.advance(mOutputs, gpuSampleSpaceIt, outputSize, outputNum);

						for (auto outputIdx : std::views::iota(0ULL, outputNum)) {

							auto desired = mOutputs.viewColumn(outputIdx);

							std::fill(desired.begin(), desired.end(), 0ULL);
							desired.mView[outputIdx] = 1.0f;
						}
						};

					auto createInputs = [&]() {

						auto setTrainingData = [&]() {

							std::size_t imageCounter = 0;
							std::size_t digit = 0;

							for (auto batch : std::views::iota(0ULL, mTrainBatched2Samples.size())) {

								auto& [seenBatch, desired] = mTrainBatched2Samples[batch];

								gpuSampleSpace.advance(seenBatch, gpuSampleSpaceIt, inputSize, batchSize);

								auto& cpuImages = mTrainSamplesMap.find(digit)->second;
								desired = mOutputs.viewColumn(digit);

								for (auto b : std::views::iota(0ULL, batchSize)) {

									auto seen = seenBatch.viewColumn(b);

									auto imageIdx = (imageCounter + b) % cpuImages.size();
									auto& image = cpuImages[imageIdx];

									auto seenSize = seen.mSize;
									auto imageSize = image.size();

									std::copy(image.begin(), image.end(), seen.begin());
								}

								if (++digit == outputNum) {
									imageCounter += batchSize;
									digit = 0;
								}

								mTrueTrainSamplesNum = getSamplesNum(mTrainSamplesMap);
							}
							};

						auto setTestData = [&]() {

							std::size_t sampleCounter = 0;

							for (auto& [seenBatch, desiredBatch] : mTestBatched3Samples) {
								gpuSampleSpace.advance(seenBatch, gpuSampleSpaceIt, inputSize, batchSize);
								gpuSampleSpace.advance(desiredBatch, gpuSampleSpaceIt, 1ULL, batchSize);
							}

							for (auto digit : std::views::iota(0, 10)) {

								auto& cpuImages = mTestSamplesMap.find(digit)->second;
								auto imagesSize = cpuImages.size();
								std::size_t imageCounter = 0;

								while (imageCounter < imagesSize) {

									auto batchIdx = sampleCounter / batchSize;

									auto& [seenBatch, desiredBatch] = mTestBatched3Samples[batchIdx];

									auto partialSize = std::min(batchSize, imagesSize - imageCounter);
									for (auto b : std::views::iota(0ULL, partialSize)) {

										Gpu::GpuView1 seen = seenBatch.viewColumn(b);
										Gpu::GpuIntView1 desired = desiredBatch.viewColumn(b);

										auto& image = cpuImages[imageCounter++];

										std::copy(image.begin(), image.end(), seen.begin());
										desired.mView[0] = digit;

										++sampleCounter;
									}
								}
							}

							mTrueTestSamplesNum = getSamplesNum(mTestSamplesMap);
							};

						setTrainingData();
						setTestData();
						};

					createOneHotOutputViews();
					createInputs();

					gpuSampleSpace.upload();
					};

				copyToGpu();
			}

			void destroy() {
				mFloatSpace.destroy();
			}
		};

		class MNIST:	public Model {
		public:
			
			MNISTSamples mSamples;

			std::size_t mId = 981;
			std::size_t mConvergenceNetworkId = 0;

			TrainingManager::GpuTask* mGpuTaskTrain = nullptr, * mGpuTaskConvergence = nullptr;
			std::mutex mNetworkMutex;

			MNIST() : Model("mnist.txt", 28, 28, 10, 4, 1) {}

			void calculateConvergence(bool print = false) {

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

					mTrainingManager.calculateConvergence(*mGpuTaskConvergence, ccNetwork
						, mSamples.mTestBatched3Samples, mSamples.mTrueTestSamplesNum, mSamples.mOutputs, print);

					}, print);
			}
			void create(bool print = false) {

				using ConvolutionType = LayerTemplate::ConvolutionType;
				using ActivationFunction = LayerTemplate::ActivationFunction;
				constexpr auto ReLU = ActivationFunction::ReLU;

				mNetworkTemplate = { mInputWidth * mInputHeight, mBatchSize
					, {{ ConvolutionType::Conv1, 3, 10, ReLU }
					, { mOutputSize, ActivationFunction::Softmax }}
				};

				if (print)
					std::puts("Creating MNIST Network");

				auto createNetworks = [&]() {
					mTrainingManager.addNetwork(mId);
					auto& network = mTrainingManager.getNetwork(mId);
					network.create(&mNetworkTemplate, true);
					network.initializeId(mId);

					mTrainingManager.addNetwork(mConvergenceNetworkId);
					auto& ccnetwork = mTrainingManager.getNetwork(mConvergenceNetworkId);
					ccnetwork.create(&mNetworkTemplate, false);

					mTrainingManager.create(2);

					mGpuTaskTrain = &mTrainingManager.getGpuTask(0);
					mGpuTaskConvergence = &mTrainingManager.getGpuTask(1);

					};
				createNetworks();

				mSamples.create(mNetworkTemplate);
			}
			void destroy() {
				mSamples.destroy();
				Model::destroy();
			}

			void train(std::size_t trainNum = 1, std::size_t offset = 0, bool print = false) {

				mTrainingManager.train(*mGpuTaskTrain, trainNum, mSamples.mTrainBatched2Samples, mLearnRate, offset, false, print);

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

				create(print);
				calculateConvergence(print);

				auto totalTrainNum = mTrainNum * mSamples.mTrainBatched2Samples.size();
				train(totalTrainNum, 0, print);

				calculateConvergence(print);
				destroy();
			}
		};

		class MNISTLottery : public LotteryModel {
		public:
			MNISTSamples mSamples;

			std::mutex mNetworkMutex;

			MNISTLottery() : LotteryModel("MNISTLottery.txt", 28, 28, 10, 4, 1, 2, 1000) {}

			void calculateTestConvergence(bool print = false) {

				mTrainingManager.calculateNetworksConvergence(mSamples.mTestBatched3Samples
					, mSamples.mTrueTestSamplesNum
					, mSamples.mOutputs, print);
			
				sort("Test", print);
			}
			void calculateTrainConvergence(bool print = false) {
				
				mTrainingManager.calculateNetworksConvergence(mSamples.mTrainBatched2Samples
					, mSamples.mTrueTrainSamplesNum, print);

				sort("Train", print);
			}

			void sort(std::string_view caption="", bool print = false ) {

				mNetworksSorter.sortBySuperRadius();

				if (!print) return;

				record("\nConvergence Results for MNIST {}"
					"\nNetworks sorted by SuperRadius:", caption);

				recordTopAndBottomNetworks();
				recordZeroMisses();
			}
			
			void create(bool print = false) {

				using ConvolutionType = LayerTemplate::ConvolutionType;
				using ActivationFunction = LayerTemplate::ActivationFunction;
				constexpr auto ReLU = ActivationFunction::ReLU;

				mNetworkTemplate = { mInputWidth * mInputHeight, mBatchSize
					, {{ 24, ReLU }
					, { mOutputSize, ActivationFunction::Softmax }}
				};

				if (print)
					std::puts("Creating MNIST Networks");

				createNetworks();

				mSamples.create(mNetworkTemplate);
			}
			void destroy() {
				
				mSamples.destroy();
				Model::destroy();
			}

			void train(std::size_t trainNum = 1, std::size_t offset = 0, bool print = false) {

				mTrainingManager.trainNetworks(trainNum, mSamples.mTrainBatched2Samples, mLearnRate, offset, print);
			}

			void run(bool print = true) {

				create(print);

				auto totalTrainNum = mTrainNum * mSamples.mTrainBatched2Samples.size();
				train(totalTrainNum, 0, print);

				calculateTestConvergence(print);

				destroy();
			}
		};
	}
}