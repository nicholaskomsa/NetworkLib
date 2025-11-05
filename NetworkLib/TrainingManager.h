#pragma once

#include "GpuTensor.h"
#include "GpuNetwork.h"
#include "NetworkSorter.h"

#include <fstream>
#include <map>
#include <numeric>

namespace NetworkLib {

	class TrainingManager {
	public:

		using SingleOutput = Gpu::GpuView1;
		using BatchedOutput = Gpu::GpuView2;

		using BatchedSingleOutput = std::vector<SingleOutput>;

		using GpuSample = std::pair<Gpu::GpuView1, Gpu::GpuView1>;
		using CpuSample = std::pair<std::vector<float>, std::vector<float>>;
		using CpuSamples = std::vector<CpuSample>;
		using GpuViews = std::vector<Gpu::GpuView1>;

		using CpuBatchedSample = std::pair<Cpu::View2, Cpu::View2>;
		using CpuBatchedSamples = std::vector<CpuBatchedSample>;
		using CpuBatchedSamplesView = std::span<CpuBatchedSample>;

		using GpuBatchedSample = std::pair<Gpu::GpuView2, Gpu::GpuView2>;
		using GpuBatchedSamples = std::vector<GpuBatchedSample>;
		using GpuBatchedSamplesView = std::span<GpuBatchedSample>;

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

		void train(GpuTask& gpuTask, std::size_t trainNum, GpuBatchedSamplesView samples, float learnRate, bool print = false);
		void trainNetworks(std::size_t trainNum, GpuBatchedSamplesView samples, float learnRate, bool print = false);

		void calculateConvergence(GpuTask& gpuTask, Cpu::Network& cpuNetwork, const TrainingManager::GpuBatchedSamplesView samples, bool print = false);
		void calculateNetworkConvergence(GpuTask& gpuTask, Cpu::Network& cpuNetwork, const TrainingManager::GpuBatchedSamplesView samples, bool print = false);
		
		void forEachNetwork(Cpu::NetworksMap& networks, auto&& functor);

		void calculateNetworksConvergence(Cpu::NetworksMap& networks, GpuBatchedSamplesView samples, bool print = false);
		void calculateNetworksConvergence(GpuBatchedSamplesView samples, bool print = false);

		static GpuBatchedSamples createGpuBatchedSamplesSpace(Gpu::LinkedFloatSpace& linkedSampleSpace
			, std::size_t inputSize, std::size_t outputSize, std::size_t sampleNum, std::size_t batchSize = 1);

		static GpuBatchedSamplesView advanceGpuBatchedSamples(Gpu::LinkedFloatSpace& linkedSampleSpace
			, float*& begin, GpuBatchedSamples& gpuSamples, const CpuSamples& cpuSamples
			, std::size_t batchSize = 1);

		static GpuBatchedSamples createGpuBatchedSamplesSpaceSharedOutput(Gpu::LinkedFloatSpace& linkedSampleSpace
			, std::size_t inputSize, std::size_t sampleNum, std::size_t outputSize, std::size_t outputNum, std::size_t batchSize = 1);

		static void advanceGpuViews(Gpu::LinkedFloatSpace& linkedSampleSpace
			, float*& begin, GpuViews& gpuViews, std::size_t outputSize);

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

			Gpu::LinkedFloatSpace mFloatSpace;
			GpuBatchedSamples mMNISTSamples;
			std::string mMNISTFolder = "./mnist/";

			GpuViews mOutputs;

			using DigitsSamplesMap = std::map<std::uint8_t, std::vector<std::vector<float>>>;
			DigitsSamplesMap mTestSamplesMap, mTrainSamplesMap;

			using GpuDigitSamplesMap = std::map<std::uint8_t, std::vector<GpuBatchedSamplesView>>;
			GpuDigitSamplesMap mGpuTrainSamplesMap, mGpuTestSamplesMap;
			
			std::size_t getSamplesNum(const DigitsSamplesMap& samplesMap) {
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

				mTrainSamplesMap = loadDigitsSamples(trainImagesFileName, trainLabelsFileName);
				mTestSamplesMap = loadDigitsSamples(testImagesFileName, testLabelsFileName);
			}
			DigitsSamplesMap loadDigitsSamples(const std::string& imagesFileName, const std::string& labelsFileName) {

				DigitsSamplesMap digitsMap;
					
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


				LabelType label;
				Image1 image(rows * cols);
				FloatImage1 floatImage(rows * cols);

				std::println("Reading mnist x {} images", numImages);

				constexpr float normalizeFactor = std::numeric_limits<Image1Type>::max();
				for (auto i : std::views::iota(0UL, numImages)) {

					finImages.read(reinterpret_cast<char*>(image.data()), image.size() * sizeof(Image1Type));
					finLabels.read(reinterpret_cast<char*>(&label), sizeof(label));
						
					std::transform(image.begin(), image.end(), floatImage.begin()
						, [](auto i) { return i / normalizeFactor; });

					digitsMap[label].push_back(floatImage);
				}
				return digitsMap;
			}
		
			void create(NetworkTemplate& networkTemplate) {
				
				auto inputSize = networkTemplate.mInputSize
					, outputSize = networkTemplate.mLayerTemplates.back().mNodeCount;
				auto batchSize = networkTemplate.mBatchSize;

				loadAllDigitsSamples();

				auto outputNum = 10; //digits 0-9
				mOutputs.resize(outputNum);

				auto sampleNum = getSamplesNum(mTrainSamplesMap) + getSamplesNum(mTestSamplesMap);

				mMNISTSamples = TrainingManager::createGpuBatchedSamplesSpaceSharedOutput(mFloatSpace
					, inputSize, sampleNum, outputSize, outputNum, batchSize);

				auto begin = mFloatSpace.mGpuSpace.begin();
				 
				auto createOutputViews = [&]() {
					//first, setup the GpuOutputs for each digit
					TrainingManager::advanceGpuViews(mFloatSpace, begin, mOutputs, outputSize);

					std::vector<float> desired(outputSize);
					for (std::size_t digit = 0; auto output : mOutputs) {

						std::fill(desired.begin(), desired.end(), 0ULL);

						desired[digit] = 1.0f;
						std::copy(desired.begin(), desired.end(), output.begin());
					}
					};
				createOutputViews();

				for( auto& [digit, samples] : mTrainSamplesMap ) {

					auto output = mOutputs[digit];

					//auto gpuBatchedSamplesView = TrainingManager::advanceGpuBatchedSamplesSharedOutput(mFloatSpace, begin
					//	, mMNISTSamples, output, samples, batchSize);

					//mGpuTrainSamplesMap[digit].push_back(gpuBatchedSamplesView);
				}


				//mANDSamples = TrainingManager::advanceGpuBatchedSamples(mFloatSpace, begin, mLogicSamples, andSamples, batchSize);
			//	mORSamples = TrainingManager::advanceGpuBatchedSamples(mFloatSpace, begin, mLogicSamples, orSamples, batchSize);
			//	mXORSamples = TrainingManager::advanceGpuBatchedSamples(mFloatSpace, begin, mLogicSamples, xorSamples, batchSize);
				

				mFloatSpace.mGpuSpace.upload();
			}
			void destroy() {
				mFloatSpace.destroy();
			}

		} mMNISTSamples;

	};
	
}