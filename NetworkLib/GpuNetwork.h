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

 				mGpuFloats.resize(cpuNetwork.mFloats);
				mLayers.resize(cpuNetwork.mLayers.size());

				auto begin = mGpuFloats.begin();
				
				auto groupComponent = [&](auto&& setupFunctor) ->GpuView1 {

					auto componentBegin = begin;

					for (std::size_t idx = 0; const auto& [layer, layerTemplate] : std::views::zip(mLayers, layerTemplates))
						setupFunctor( layer, layerTemplate, idx++);
		 
					std::size_t size = std::distance(componentBegin, begin);
					auto view = Cpu::Tensor::View1(componentBegin, std::array{ size });
					return { view, mGpuFloats.getGpu(view), componentBegin };
					};

				//mWeights, etc refer to all weights from all layers, they are grouped
				mWeights = groupComponent([&](auto& layer, auto& layerTemplate, std::size_t idx) {

					switch (layerTemplate.mConvolutionType) {
					case LayerTemplate::ConvolutionType::Conv1: {

						mGpuFloats.advance(layer.mWeights, begin, layerTemplate.mKernelWidth, 1ULL, layerTemplate.mKernelNumber);
						break;
					}
					case LayerTemplate::ConvolutionType::None: {

						auto inputSize = (idx==0) ?
							nt.mInputSize : layerTemplates[idx-1].mNodeCount;

						mGpuFloats.advance(layer.mWeights, begin, layerTemplate.mNodeCount, inputSize, 1ULL);
						break;
					}
					}

					});

				mBias = groupComponent([&](auto& layer, auto& layerTemplate, std::size_t idx) {
					mGpuFloats.advance(layer.mBias, begin, layerTemplate.mNodeCount);
					});
				mOutputs = groupComponent([&](auto& layer, auto& layerTemplate, std::size_t idx) {
					mGpuFloats.advance(layer.mOutputs, begin, layerTemplate.mNodeCount, batchSize);
					});
				mActivations = groupComponent([&](auto& layer, auto& layerTemplate, std::size_t idx) {
					mGpuFloats.advance(layer.mActivations, begin, layerTemplate.mNodeCount, batchSize);
					});

				if(backwards)
					mPrimes = groupComponent([&](auto& layer, auto& layerTemplate, std::size_t idx) {
						mGpuFloats.advance(layer.mPrimes, begin, layerTemplate.mNodeCount, batchSize);
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
				
				auto& layersTemplates = mNetworkTemplate->mLayerTemplates;

				for(const auto& [layer, layerTemplate] : std::views::zip(mLayers, layersTemplates))
					seen = layer.forward(gpu, seen, layerTemplate, batch);
				
				return seen;
			}

			const GpuView2 forward(Environment& gpu, GpuView2 seenBatch) {

				auto& layerTemplates = mNetworkTemplate->mLayerTemplates;
				for (const auto& [layer, layerTemplate] : std::views::zip(mLayers, layerTemplates))
					seenBatch = layer.forward(gpu, seenBatch, layerTemplate);

				return seenBatch;
			}

			void backward(Environment& env, const GpuView1& seen, const GpuView1& desired, float learnRate, std::size_t batch = 0) {

				auto& layerTemplates = mNetworkTemplate->mLayerTemplates;

				auto backLayer = [&]() {

					auto& back = mLayers.back();
					const auto sought = back.mActivations.viewColumn(batch);
					auto p1 = back.mPrimes.viewColumn(batch);
					auto af = layerTemplates.back().mActivationFunction;
					//output layer makes a comparison between desired and sought
					env.errorFunction(af, desired, sought, p1);
					};

				auto hiddenLayers = [&]() {

					for (auto l : std::views::iota(0ULL, mLayers.size() - 1) | std::views::reverse) {

						auto& layer = mLayers[l];
						auto& nextLayer = mLayers[l + 1];

						auto af = layerTemplates[l].mActivationFunction;

						auto p1 = layer.mPrimes.viewColumn(batch);
						auto o1 = layer.mOutputs.viewColumn(batch);
						auto np1 = nextLayer.mPrimes.viewColumn(batch);

						env.matTMulVec(nextLayer.mWeights, np1, p1);
						env.activationFunctionPrime(af, o1, p1);
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

				auto& layerTemplates = mNetworkTemplate->mLayerTemplates;

				auto backLayer = [&]() {

					auto& back = mLayers.back();
					const auto soughtBatch = back.mActivations;
					auto af = layerTemplates.back().mActivationFunction;
					//output layer makes a comparison between desired and sought
					env.batchedErrorFunction(af, desiredBatch, soughtBatch, back.mPrimes);
					};

				auto hiddenLayers = [&]() {

					for (auto l : std::views::iota(0ULL, mLayers.size() - 1) | std::views::reverse) {

						auto& layer = mLayers[l];
						auto& nextLayer = mLayers[l + 1];
						auto af = layerTemplates[l].mActivationFunction;

						env.batchedMatTMulVec(nextLayer.mWeights, nextLayer.mPrimes, layer.mPrimes);
						env.batchedActivationFunctionPrime(af, layer.mOutputs, layer.mPrimes);
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

				const GpuView1 forward(Environment& env, const GpuView1& input, LayerTemplate& layerTemplate, std::size_t batch = 0) {

					auto outputs1 = mOutputs.viewColumn(batch);
					auto activations1 = mActivations.viewColumn(batch);
					auto af = layerTemplate.mActivationFunction;
					
					env.matMulVec(mWeights, input, outputs1);
					env.vecAddVec(mBias, outputs1);
					env.activationFunction(af, outputs1, activations1);

					return activations1;
				}
				const GpuView2& forward(Environment& env, const GpuView2& input, LayerTemplate& layerTemplate) {
					
					auto af = layerTemplate.mActivationFunction;
					
					env.batchedMatMulVec(mWeights, input, mOutputs);
					env.batchedBroadcastAdd(mBias, mOutputs);
					env.batchedActivationFunction(af, mOutputs, mActivations);

					return mActivations;
				}

				GpuView3 mWeights;
				GpuView1 mBias;
				GpuView2 mOutputs, mActivations, mPrimes;
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