#pragma once

#include "TrainingManager.h"

#include "Algorithms.h"

using namespace NetworkLib;

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
}

void TrainingManager::forEachNetwork(Cpu::NetworksMap& networks, ForEachFunctor&& functor) {

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


TrainingManager::GpuBatchedSamples TrainingManager::createGpuBatchedSamplesSpace(Gpu::LinkedFloatSpace& linkedSampleSpace
	, std::size_t inputSize, std::size_t outputSize, std::size_t sampleNum, std::size_t batchSize) {

	GpuBatchedSamples gpuBatchedSamples;
	auto batchNum = std::ceil(sampleNum / float(batchSize));
	gpuBatchedSamples.reserve(batchNum);

	auto inputOutputPairs = batchNum * batchSize * (inputSize + outputSize);
	linkedSampleSpace.create(inputOutputPairs);
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



Gpu::GpuView1 TrainingManager::advanceGpuViews(Gpu::LinkedFloatSpace& linkedSampleSpace
	, float*& begin, Gpu::GpuViews1& gpuViews, std::size_t outputSize) {

	auto& gpuSampleSpace = linkedSampleSpace.mGpuSpace;

	auto start = begin;

	for (auto& gpuView : gpuViews)
		gpuSampleSpace.advance(gpuView, begin, outputSize);

	auto end = begin;

	Gpu::GpuView1 gpuView1;
	gpuSampleSpace.advance(gpuView1, start, std::distance(start, end));
	return gpuView1;
}
