#pragma once

#include "TrainingManager.h"

#include "Algorithms.h"

using namespace NetworkLib;


using LogicSamples = TrainingManager::LogicSamples;

void TrainingManager::create(std::size_t gpuNum) {

	mGpuNum = gpuNum;
	mParallelGpuTasks.setup(GpuTask{}, gpuNum, gpuNum);

	auto& defaultNetwork = mNetworksMap.begin()->second;
	mParallelGpuTasks([&](Parallel::Section& section) {

		auto& gpuTask = std::any_cast<GpuTask&>(section.mAny);

		auto& [gpu, gpuNetwork] = gpuTask;
		gpu.create();
		gpuNetwork.mirror(defaultNetwork);
		});
}

void TrainingManager::destroy() {

	for (auto& [id, network] : mNetworksMap)
		network.destroy();

	mParallelGpuTasks([&](auto& section) {

		auto& [gpu, gpuNetwork] = std::any_cast<GpuTask&>(section.mAny);

		gpu.destroy();
		gpuNetwork.destroy();

		});

	mLogicSamples.destroy();
}
void TrainingManager::addNetwork() {
	std::size_t id = mNetworksIdCounter++;
	addNetwork(id);
}
void TrainingManager::addNetwork(std::size_t id) {

	auto found = mNetworksMap.find(id);
	if (found != mNetworksMap.end())
		throw std::runtime_error("network already exists");

	auto network = mNetworksMap[id];
}
Cpu::Network& TrainingManager::TrainingManager::getNetwork(std::size_t id) {

	auto found = mNetworksMap.find(id);
	if (found == mNetworksMap.end())
		throw std::runtime_error("Network not found");

	return found->second;
}
TrainingManager::GpuTask& TrainingManager::getGpuTask(std::size_t idx) {

	auto& sectionsView = mParallelGpuTasks.mSectionsView;
	auto& section = sectionsView[idx % sectionsView.size()];
	return std::any_cast<GpuTask&>(section.mAny);
}

void TrainingManager::train(GpuTask& gpuTask, std::size_t trainNum, GpuBatchedSamplesView samples, float learnRate, bool print) {

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
void TrainingManager::trainNetworks(std::size_t trainNum, GpuBatchedSamplesView samples, float learnRate, bool print) {

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

void TrainingManager::calculateConvergence(GpuTask& gpuTask, Cpu::Network& cpuNetwork, const TrainingManager::GpuBatchedSamplesView samples, bool print) {

	auto& [gpu, gpuNetwork] = gpuTask;

	gpuNetwork.mirror(cpuNetwork);
	
	gpu.resetMseResult();
	
	gpu.resetMissesResult();
	
	auto batchSize = samples.front().first.mView.extent(1);

	for (const auto& [seen, desired] : samples) {

		auto sought = gpuNetwork.forward(gpu, seen);
		auto output = gpuNetwork.getOutput();


		gpu.mse(sought, desired);
			gpu.score(sought, desired);

		if (print) {

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

	gpu.downloadConvergenceResults();
	cpuNetwork.mMse = gpu.getMseResult();
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
void TrainingManager::calculateNetworkConvergence(GpuTask& gpuTask, Cpu::Network& cpuNetwork, const TrainingManager::GpuBatchedSamplesView samples, bool print) {

	time<seconds>("Calculate Network Convergence", [&]() {

		calculateConvergence(gpuTask, cpuNetwork, samples, print);

		}, print);
}

void TrainingManager::forEachNetwork(Cpu::NetworksMap& networks, auto&& functor) {

	auto networksBegin = networks.begin();

	auto getNextNetwork = [&]()->Cpu::Network* {

		std::scoped_lock lock(mNetworksMapMutex);

		if (networksBegin == mNetworksMap.end())
			return nullptr;

		Cpu::Network* network = &networksBegin->second;

		std::advance(networksBegin, 1);
		return network;
		};

	mParallelGpuTasks.section(networks.size());

	mParallelGpuTasks([&](auto& section) {

		auto& gpuTask = std::any_cast<GpuTask&>(section.mAny);

		for (auto idx : section.mIotaView) {

			auto cpuNetwork = getNextNetwork();
			if (!cpuNetwork)
				return;

			functor(gpuTask, *cpuNetwork);
		}
		});
}

void TrainingManager::calculateNetworksConvergence(Cpu::NetworksMap& networks, GpuBatchedSamplesView samples, bool print) {

	time<seconds>("Calculate Networks Convergence", [&]() {
		forEachNetwork(networks, [&](GpuTask& gpuTask, Cpu::Network& cpuNetwork) {
			calculateConvergence(gpuTask, cpuNetwork, samples, print);
			});
		}, print);
}

void TrainingManager::calculateNetworksConvergence(GpuBatchedSamplesView samples, bool print) {

	calculateNetworksConvergence(mNetworksMap, samples, print);
}

TrainingManager::GpuBatchedSamples TrainingManager::createGpuBatchedSamplesSpace(Gpu::LinkedFloatSpace& linkedSampleSpace
	, std::size_t inputSize, std::size_t outputSize, std::size_t sampleNum, std::size_t batchSize) {

	GpuBatchedSamples gpuBatchedSamples;
	auto batchNum = std::ceil(sampleNum / float(batchSize));
	gpuBatchedSamples.reserve(batchNum);

	//auto outputIdx = 1; // 1 integer is the output which is the outputidx

	linkedSampleSpace.create(batchNum * batchSize * (inputSize + outputSize));
	return gpuBatchedSamples;
}

TrainingManager::GpuBatchedSamplesView TrainingManager::advanceGpuBatchedSamples(Gpu::LinkedFloatSpace& linkedSampleSpace
	, float*& begin, GpuBatchedSamples& gpuSamples, const CpuSamples& cpuSamples
	, std::size_t batchSize) {

	auto inputSize = cpuSamples.front().first.size()
		, outputSize = cpuSamples.front().second.size();

	auto& gpuSampleSpace = linkedSampleSpace.mGpuSpace;

	auto currentSample = cpuSamples.begin();

	auto batchNum = std::ceil(cpuSamples.size() / float(batchSize));
	auto viewStart = gpuSamples.size();
	constexpr std::size_t zero = 0;
	for (auto batch : std::views::iota(zero, batchNum)) {

		GpuBatchedSample gpuSample;

		auto& [seenBatch, desiredBatch] = gpuSample;

		gpuSampleSpace.advance(seenBatch, begin, inputSize, batchSize);
		gpuSampleSpace.advance(desiredBatch, begin, outputSize, batchSize);

		for (auto b : std::views::iota(zero, batchSize)) {

			if (currentSample == cpuSamples.end()) break;
			auto& [seen, desired] = *currentSample;
			++currentSample;

			auto gpuSeen = seenBatch.viewColumn(b);
			auto gpuDesired = desiredBatch.viewColumn(b);

			std::copy(seen.begin(), seen.end(), gpuSeen.begin());
			std::copy(desired.begin(), desired.end(), gpuDesired.begin());
		}

		gpuSamples.push_back(gpuSample);
	}
	return { gpuSamples.begin() + viewStart, gpuSamples.size() - viewStart };
}

TrainingManager::GpuBatchedSamples TrainingManager::createGpuBatchedSamplesSpaceSharedOutput(Gpu::LinkedFloatSpace& linkedSampleSpace
	, std::size_t inputSize, std::size_t sampleNum, std::size_t outputSize, std::size_t outputNum, std::size_t batchSize) {

	GpuBatchedSamples gpuBatchedSamples;
	auto batchNum = std::ceil(sampleNum / float(batchSize));
	gpuBatchedSamples.reserve(batchNum);
	linkedSampleSpace.create(batchNum * batchSize * inputSize
		+ outputSize * outputNum); //shared output space

	return gpuBatchedSamples;
}

void TrainingManager::advanceGpuViews(Gpu::LinkedFloatSpace& linkedSampleSpace
	, float*& begin, GpuViews& gpuViews, std::size_t outputSize) {

	auto& gpuSampleSpace = linkedSampleSpace.mGpuSpace;

	for (auto& gpuView : gpuViews)
		gpuSampleSpace.advance(gpuView, begin, outputSize);
}


void LogicSamples::create(NetworkTemplate& networkTemplate) {

	auto inputSize = networkTemplate.mInputSize
		, outputSize = networkTemplate.mLayerTemplates.back().mNodeCount;
	auto batchSize = networkTemplate.mBatchSize;
	auto sampleNum = 4 * 3; //XOR, AND, OR all have 4 samples
	mLogicSamples = TrainingManager::createGpuBatchedSamplesSpace(mFloatSpace
		, inputSize, outputSize, sampleNum, batchSize);

	auto begin = mFloatSpace.mGpuSpace.begin();

	CpuSamples andSamples = {
		{{ 0,0 }, {1,0}},
		{{ 0,1 }, {1,0}},
		{{ 1,0 }, {1,0}},
		{{ 1,1 }, {0,1}}
	};
	const CpuSamples xorSamples = {
		{{0,0}, {1,0}},
		{{0,1}, {0,1}},
		{{1,0}, {0,1}},
		{{1,1}, {1,0}}
	};
	const CpuSamples orSamples = {
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
void LogicSamples::destroy() {
	mFloatSpace.destroy();
}
	
LogicSamples::SamplesGroup LogicSamples::getSamples() {
	return { mXORSamples, mORSamples, mANDSamples, mLogicSamples };
}
	