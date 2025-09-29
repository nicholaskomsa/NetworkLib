#pragma once

#include <random>

#include "Parallel.h"
#include "Environment.h"
#include "CpuNetwork.h"

namespace NetworkLib {
	namespace Gpu {

		class Network {
		public:

			void mirror(Cpu::Network& cpuNetwork) {

				mNetworkTemplate = cpuNetwork.mNetworkTemplate;

				const auto& nt = *mNetworkTemplate;
				auto& layerTemplates = nt.mLayerTemplates;
				auto batchSize = nt.mBatchSize;
				auto firstInputSize = nt.mInputSize;
				bool backwards = cpuNetwork.mPrimes.size() > 0;

 				mGpuFloats.create(cpuNetwork.mFloats);

				mLayers.resize(cpuNetwork.mLayers.size());

				auto begin = mGpuFloats.begin();
				
				auto groupComponent = [&](auto&& setupFunctor) ->GpuView1 {

					auto componentBegin = begin;

					auto inputSize = firstInputSize;
					for (const auto& [layer, layerTemplate] : std::views::zip(mLayers, layerTemplates)) {

						const auto n = layerTemplate.mNodeCount;

						setupFunctor( layer, layerTemplate, n, inputSize);
		
						inputSize = n;
					}

					std::size_t size = std::distance(componentBegin, begin);
					auto view = Cpu::Tensor::View1(componentBegin, std::array{ size });
					return { view, mGpuFloats.getGpu(view), componentBegin };
					};

				//mWeights, etc refer to all weights from all layers, they are grouped
				mWeights = groupComponent([&](auto& layer, auto& layerTemplate, std::size_t n, std::size_t inputSize) {
					layer.mActivationFunction = layerTemplate.mActivationFunction;
					mGpuFloats.advance(layer.mWeights, begin, n, inputSize);
					});
				mBias = groupComponent([&](auto& layer, auto& layerTemplate, std::size_t n, std::size_t inputSize) {
					mGpuFloats.advance(layer.mBias, begin, n);
					});
				mOutputs = groupComponent([&](auto& layer, auto& layerTemplate, std::size_t n, std::size_t inputSize) {
					mGpuFloats.advance(layer.mOutputs, begin, n, batchSize);
					});
				mActivations = groupComponent([&](auto& layer, auto& layerTemplate, std::size_t n, std::size_t inputSize) {
					mGpuFloats.advance(layer.mActivations, begin, n, batchSize);
					});

				if(backwards)
					mPrimes = groupComponent([&](auto& layer, auto& layerTemplate, std::size_t n, std::size_t inputSize) {
						mGpuFloats.advance(layer.mPrimes, begin, n, batchSize);
						});

				upload();
			}

			void destroy() {
				mGpuFloats.destroy();
			}

			void upload() {
				mGpuFloats.upload();
			}

			const GpuView1 forward(Environment& gpu, GpuView1 seen, std::size_t batch = 0) {
				
				for( auto& layer : mLayers ) 
					seen = layer.forward(gpu, seen, batch);
				
				return seen;
			}

			const GpuView2 forward(Environment& gpu, GpuView2 seenBatch) {

				for (auto& layer : mLayers)
					seenBatch = layer.forward(gpu, seenBatch);

				return seenBatch;
			}

			void backward(Environment& env, const GpuView1& seen, const GpuView1& desired, float learnRate, std::size_t batch = 0) {

				auto backLayer = [&]() {

					auto& back = mLayers.back();
					const auto sought = back.mActivations.viewColumn(batch);
					auto p1 = back.mPrimes.viewColumn(batch);

					//output layer makes a comparison between desired and sought
					env.errorFunction(back.mActivationFunction, desired, sought, p1);
					};

				auto hiddenLayers = [&]() {

					for (auto l : std::views::iota(0ULL, mLayers.size() - 1) | std::views::reverse) {

						auto& layer = mLayers[l];
						auto& nextLayer = mLayers[l + 1];

						auto p1 = layer.mPrimes.viewColumn(batch);
						auto o1 = layer.mOutputs.viewColumn(batch);
						auto np1 = nextLayer.mPrimes.viewColumn(batch);

						env.matTMulVec(nextLayer.mWeights, np1, p1);
						env.activationFunctionPrime(layer.mActivationFunction, o1, p1);
					}
					};

				auto updateWeights = [&]() {
					for (auto l : std::views::iota(0ULL, mLayers.size())) {

						auto& layer = mLayers[l];

						GpuView1 input;
						if (l == 0)
							input = seen;
						else
							input = mLayers[l-1].mActivations.viewColumn(batch);

						auto primes = layer.mPrimes.viewColumn(batch);

						env.updateWeights(input, layer.mWeights, primes, learnRate);
					}
					};

				backLayer();
				hiddenLayers();
				updateWeights();
			}

			void backward(Environment& env, const GpuView2& seenBatch, const GpuView2& desiredBatch, float learnRate) {

				auto backLayer = [&]() {

					auto& back = mLayers.back();
					const auto soughtBatch = back.mActivations;

					//output layer makes a comparison between desired and sought
					env.batchedErrorFunction(back.mActivationFunction, desiredBatch, soughtBatch, back.mPrimes);
					};

				auto hiddenLayers = [&]() {

					for (auto l : std::views::iota(0ULL, mLayers.size() - 1) | std::views::reverse) {

						auto& layer = mLayers[l];
						auto& nextLayer = mLayers[l + 1];

						env.batchedMatTMulVec(nextLayer.mWeights, nextLayer.mPrimes, layer.mPrimes);
						env.batchedActivationFunctionPrime(layer.mActivationFunction, layer.mOutputs, layer.mPrimes);
					}
					};

				auto updateWeights = [&]() {

					auto& front = mLayers.front();
					env.batchedUpdateWeights(seenBatch, front.mWeights, front.mPrimes, learnRate);

					for (auto l : std::views::iota(1ULL, mLayers.size())) {

						auto& layer = mLayers[l];
						auto& prevLayer = mLayers[l - 1];

						env.batchedUpdateWeights(prevLayer.mActivations, layer.mWeights, layer.mPrimes, learnRate);
					}
					};

				backLayer();
				hiddenLayers();
				updateWeights();
			}

			
			GpuView2 getSought() {
				return mLayers.back().mActivations;
			}
			GpuView2 getOutput() {
				return mLayers.back().mOutputs;
			}
			GpuView2 getPrimes() {
				return mLayers.back().mPrimes;
			}

			class Layer {
			public:

				const GpuView1 forward(Environment& env, const GpuView1& input, std::size_t batch = 0) {

					auto outputs1 = mOutputs.viewColumn(batch);
					auto activations1 = mActivations.viewColumn(batch);

					env.matMulVec(mWeights, input, outputs1);
					env.vecAddVec(mBias, outputs1);
					env.activationFunction(mActivationFunction, outputs1, activations1);

					return activations1;
				}
				const GpuView2& forward(Environment& env, const GpuView2& input) {
				
					env.batchedMatMulVec(mWeights, input, mOutputs);
					env.batchedBroadcastAdd(mBias, mOutputs);

					env.batchedActivationFunction(mActivationFunction, mOutputs, mActivations);
					return mActivations;
				}

				GpuView2 mWeights;
				GpuView1 mBias;
				GpuView2 mOutputs, mActivations, mPrimes;

				LayerTemplate::ActivationFunction mActivationFunction = LayerTemplate::ActivationFunction::None;
			};

			using Layers = std::vector<Layer>;

			Layer& getLayer(std::size_t i) {
				return mLayers[i];
			}

		public:
			NetworkTemplate* mNetworkTemplate = nullptr;

			FloatSpace1 mGpuFloats;

			GpuView1 mWeights, mBias, mOutputs, mActivations, mPrimes;

			Layers mLayers;
		};
	
		using NetworksView = std::span<Network>;
	}


}