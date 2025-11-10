#pragma once

#include "GpuTensor.h"
#include "GpuNetwork.h"
#include "NetworkSorter.h"
#include "Algorithms.h"

#include <fstream>
#include <map>
#include <numeric>

namespace NetworkLib {

	class TrainingManager {
	public:

		using GpuSample = std::pair<Gpu::GpuView1, Gpu::GpuView1>;
		using CpuSample = std::pair<std::vector<float>, std::vector<float>>;
		using CpuSample2 = std::pair< std::vector<float>, std::vector<std::size_t>>;

		using CpuSamples = std::vector<CpuSample>;
		using CpuSamples2 = std::vector<CpuSample2>;

		using CpuBatchedSample = std::pair<Cpu::View2, Cpu::View2>;
		using CpuBatchedSamples = std::vector<CpuBatchedSample>;
		using CpuBatchedSamplesView = std::span<CpuBatchedSample>;

		//all batched samples have different output
		using GpuBatchedSample = std::pair<Gpu::GpuView2, Gpu::GpuView2>;
		using GpuBatchedSamples = std::vector<GpuBatchedSample>;
		using GpuBatchedSamplesView = std::span<GpuBatchedSample>;

		//all batched2 samples share the same output
		using GpuBatched2Sample = std::pair<Gpu::GpuView2, Gpu::GpuView1>;
		using GpuBatched2Samples = std::vector<GpuBatched2Sample>;
		using GpuBatched2SamplesView = std::span<GpuBatched2Sample>;


		Cpu::NetworksMap mNetworksMap;
		Cpu::Network::Id mNetworksIdCounter = 0;

		std::mutex mNetworksMapMutex;

		struct GpuTask {
			Gpu::Environment mGpu;
			Gpu::Network mGpuNetwork;
		};
		using GpuTasks = std::vector<GpuTask>;
		Parallel mParallelGpuTasks;
		std::size_t mGpuNum = 0;

		TrainingManager& operator=(const TrainingManager&) = delete;

		void create(std::size_t gpuNum);
		void destroy();

		void addNetwork();
		void addNetwork(std::size_t id);
		Cpu::Network& getNetwork(std::size_t id);
		GpuTask& getGpuTask(std::size_t idx = 0);

		void forEachNetwork(Cpu::NetworksMap& networks, auto&& functor);

		template<typename SamplesViewType>
		void train(GpuTask& gpuTask, std::size_t trainNum, const SamplesViewType& samples, float learnRate, bool print = false) {
			
			auto& [gpu, gpuNetwork] = gpuTask;

			time<seconds>("Training Network", [&]() {

				for (auto generation : std::views::iota(0ULL, trainNum)) {

					const auto& [seen, desired] = samples[generation % samples.size()];

					gpuNetwork.forward(gpu, seen);
					gpuNetwork.backward(gpu, seen, desired, learnRate);

					if (print)
						printProgress(generation, trainNum);
				}

				gpuNetwork.download(gpu);

				}, print);
		}
		template<typename SamplesViewType>
		void trainNetworks(std::size_t trainNum, const SamplesViewType& samples, float learnRate, bool print = false){

			time<seconds>("Training Networks", [&]() {

				std::atomic<std::size_t> progress = 0;

				forEachNetwork(mNetworksMap, [&](GpuTask& gpuTask, Cpu::Network& cpuNetwork) {

					auto& [gpu, gpuNetwork] = gpuTask;
					gpuNetwork.mirror(cpuNetwork);

					train(gpuTask, trainNum, samples, learnRate, false);

					if (print)
						printProgress(++progress, mNetworksMap.size());

					});

				}, print);
		}

		template<typename SamplesViewType>
		void calculateConvergence(GpuTask& gpuTask, Cpu::Network& cpuNetwork, const SamplesViewType& samples, bool print = false) {

			auto& [gpu, gpuNetwork] = gpuTask;

			gpuNetwork.mirror(cpuNetwork);

			gpu.resetSqeResult();
			gpu.resetMissesResult();

			for (const auto& [seen, desired] : samples) {

				auto sought = gpuNetwork.forward(gpu, seen);

				gpu.sqe(sought, desired);
			//	gpu.score(sought, desired);

				if (print) {

					auto output = gpuNetwork.getOutput();

					sought.downloadAsync(gpu);
					output.downloadAsync(gpu);
					gpu.sync();

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

			auto batchSize = samples.front().first.mView.extent(1);
			auto desiredSize = samples.front().second.mView.extent(0);

			gpu.downloadConvergenceResults();
			//normalise to get sqe -> mse 
			cpuNetwork.mMse = gpu.getSqeResult() / (desiredSize * batchSize * samples.size());
			cpuNetwork.mMisses = gpu.getMissesResult();

			if (print) {

				auto sampleNum = samples.size() * batchSize;

				std::println("\nMse: {}"
					"\nMisses: {}"
					"\nAccuracy: {}"
					, cpuNetwork.mMse
					, cpuNetwork.mMisses
					, (sampleNum - cpuNetwork.mMisses) / float(sampleNum) * 100.0f
				);
			}
		}
		template<typename SamplesViewType>
		void calculateNetworkConvergence(GpuTask& gpuTask, Cpu::Network& cpuNetwork, const SamplesViewType& samples, bool print=false) {

			time<seconds>("Calculate Network Convergence", [&]() {

				calculateConvergence(gpuTask, cpuNetwork, samples, print);

				}, print);
		}
		template<typename SamplesViewType>
		void calculateNetworksConvergence(Cpu::NetworksMap& networks, const SamplesViewType& samples, bool print = false) {

			time<seconds>("Calculate Networks Convergence", [&]() {
				forEachNetwork(networks, [&](GpuTask& gpuTask, Cpu::Network& cpuNetwork) {
					calculateConvergence(gpuTask, cpuNetwork, samples, print);
					});
				}, print);
		}
		template<typename SamplesViewType>
		void calculateNetworksConvergence(const SamplesViewType& samples, bool print = false) {
			calculateNetworksConvergence(mNetworksMap, samples, print);
		}

		static GpuBatchedSamples createGpuBatchedSamplesSpace(Gpu::LinkedFloatSpace& linkedSampleSpace
			, std::size_t inputSize, std::size_t outputSize, std::size_t sampleNum, std::size_t batchSize = 1);

		static GpuBatchedSamplesView advanceGpuBatchedSamples(Gpu::LinkedFloatSpace& linkedSampleSpace
			, float*& begin, GpuBatchedSamples& gpuSamples, const CpuSamples& cpuSamples
			, std::size_t batchSize = 1);

		static Gpu::GpuView1 advanceGpuViews(Gpu::LinkedFloatSpace& linkedSampleSpace
			, float*& begin, Gpu::GpuViews1& gpuViews, std::size_t outputSize);
		
		struct LogicSamples {

			Gpu::LinkedFloatSpace mFloatSpace;
			GpuBatchedSamples mLogicSamples;

			GpuBatchedSamplesView mXORSamples;
			GpuBatchedSamplesView mANDSamples;
			GpuBatchedSamplesView mORSamples;

			void create(NetworkTemplate& networkTemplate);
			void destroy();
			struct SamplesGroup {
				GpuBatchedSamplesView mXOR, mOR, mAND, mAll;
			};

			SamplesGroup getSamples();
			   
		} mLogicSamples;

		struct MNISTSamples {

			std::string mMNISTFolder = "./mnist/";

			Gpu::LinkedFloatSpace mFloatSpace;
			GpuBatched2Samples mTrainBatched2Samples, mTestBatched2Samples;
			GpuBatched2SamplesView mTrainSamplesView, mTestSamplesView;
			Gpu::GpuView1 mOutputs1;
			Gpu::GpuViews1 mOutputs;

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
				using FloatImage1 = std::vector<float>;

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
				return { digitsMap, images };
			}
		
			void create(NetworkTemplate& networkTemplate) {
				
				auto inputSize = networkTemplate.mInputSize
					, outputSize = networkTemplate.mLayerTemplates.back().mNodeCount;
				auto batchSize = networkTemplate.mBatchSize;

				constexpr auto outputNum = 10; //digits 0-9

				loadAllDigitsSamples();   

				auto copyToGpu = [&]() {

					std::size_t trainSamplesNum = getSamplesNum(mTrainSamplesMap)
						, testSamplesNum = getSamplesNum(mTestSamplesMap)
						, trainBatchNum = std::ceil(trainSamplesNum / float(batchSize))
						, testBatchNum = std::ceil(testSamplesNum / float(batchSize))
						, outputClassesSize = outputSize * outputNum
						, trainInputsSize = trainBatchNum * batchSize * inputSize
						, testInputsSize = testBatchNum * batchSize * inputSize;

					mTrainBatched2Samples.resize(trainBatchNum);
					mTestBatched2Samples.resize(testBatchNum);

					mFloatSpace.create(outputClassesSize + trainInputsSize + testInputsSize);
					auto& gpuSampleSpace = mFloatSpace.mGpuSpace;
					auto gpuSampleSpaceIt = gpuSampleSpace.begin();
 
					auto createOneHotOutputViews = [&]() {

						mOutputs.resize(outputNum);

						//first, setup the GpuOutputs for each digit as one hot output
						mOutputs1 = TrainingManager::advanceGpuViews(mFloatSpace, gpuSampleSpaceIt, mOutputs, outputSize);

						std::vector<float> desired(outputSize);
						for (std::size_t digit = 0; auto output : mOutputs) {

							std::fill(desired.begin(), desired.end(), 0ULL);
							desired[digit++] = 1.0f;
							std::copy(desired.begin(), desired.end(), output.begin());
						}
						};

					auto createInputs = [&]() {

						auto setSamples = [&](const auto& samplesMap, GpuBatched2Samples& batchedSamples) {

							std::size_t imageCounter = 0;
							std::uint8_t digit = 0;

							for(auto batch: std::views::iota(0ULL, batchedSamples.size())){
								
								auto& [seenBatch, desired] = batchedSamples[batch];

								gpuSampleSpace.advance(seenBatch, gpuSampleSpaceIt, inputSize, batchSize);

								auto& cpuImages = samplesMap.find(digit)->second;
								desired = mOutputs[digit];

								for (auto b : std::views::iota(0ULL, batchSize)) {

									auto seen = seenBatch.viewColumn(b);

									auto imageIdx = (imageCounter + b) % cpuImages.size();
									auto& image = cpuImages[imageIdx];

									std::copy(image.begin(), image.end(), seen.begin());
								}
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
								if (++digit == outputNum) {
									imageCounter += batchSize;
									digit = 0;
								}
							}
							};
							
						setSamples(mTrainSamplesMap, mTrainBatched2Samples);
						setSamples(mTestSamplesMap, mTestBatched2Samples);
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


		} mMNISTSamples;

	};
	
}