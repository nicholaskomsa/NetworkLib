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

				for (const auto& [layer, layerTemplate] : std::views::zip(mLayers, layersTemplates))
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
						auto ct = layerTemplates[l].mConvolutionType;

						switch (ct) {
						case LayerTemplate::ConvolutionType::Conv1:
							env.conv1VecMulVec(nextLayer.mWeights, np1, p1);
							break;
						case LayerTemplate::ConvolutionType::None:
							env.matTMulVec(nextLayer.mWeights, np1, p1);
							break;
						}
						env.activationFunctionPrime(af, o1, p1);
					}
					};

				auto updateWeights = [&]() {
			                    
					auto update = [&](std::size_t l) {

						auto& layer = mLayers[l];
						auto input = (l == 0) ?
							seen : mLayers[l - 1].mActivations.viewColumn(batch);
						auto& layerTemplate = layerTemplates[l];

						auto primes = layer.mPrimes.viewColumn(batch);

						auto ct = layerTemplate.mConvolutionType;
						switch (ct) {
						case LayerTemplate::ConvolutionType::Conv1:
							env.conv1UpdateKernel(layer.mWeights, input, primes, learnRate);
							break;
						case LayerTemplate::ConvolutionType::None:
							env.updateWeights(input, layer.mWeights, primes, learnRate);
							break;
						}
						};

					for (auto l : std::views::iota(0ULL, mLayers.size()))
						update(l);

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
						
						auto& layerTemplate = layerTemplates[l];
						auto af = layerTemplate.mActivationFunction;
						auto ct = layerTemplate.mConvolutionType;

						switch (ct) {
						case LayerTemplate::ConvolutionType::Conv1:
							env.batchedConv1VecMulVec(nextLayer.mWeights, nextLayer.mPrimes, layer.mPrimes);
							break;
						case LayerTemplate::ConvolutionType::None:
							env.batchedMatTMulVec(nextLayer.mWeights, nextLayer.mPrimes, layer.mPrimes);
							break;
						}
						
						env.batchedActivationFunctionPrime(af, layer.mOutputs, layer.mPrimes);
					}
					};

				auto updateWeights = [&]() {

					auto update = [&](std::size_t l) {

						auto& layer = mLayers[l];
						auto& prevActivations = ( l == 0 ) ?
							seenBatch : mLayers[l - 1].mActivations;
						auto& layerTemplate = layerTemplates[l];

						auto ct = layerTemplate.mConvolutionType;
						switch (ct) {
						case LayerTemplate::ConvolutionType::Conv1:
							env.batchedConv1UpdateKernel(layer.mWeights, prevActivations, layer.mPrimes, learnRate);
							break;
						case LayerTemplate::ConvolutionType::None:
							env.batchedUpdateWeights(prevActivations, layer.mWeights, layer.mPrimes, learnRate);
							break;
						}
						};

					for (auto l : std::views::iota(0ULL, mLayers.size()))
						update(l);

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
					           
					switch (layerTemplate.mConvolutionType) {
					case LayerTemplate::ConvolutionType::Conv1: 
						env.conv1(mWeights, input, outputs1);
						break;
					case LayerTemplate::ConvolutionType::None:
						env.matMulVec(mWeights, input, outputs1);
						break;
					}
					
					env.vecAddVec(mBias, outputs1);
					env.activationFunction(af, outputs1, activations1);

					return activations1;
				}
				const GpuView2& forward(Environment& env, const GpuView2& input, LayerTemplate& layerTemplate) {
					
					auto af = layerTemplate.mActivationFunction;
	
					switch (layerTemplate.mConvolutionType) {
					case LayerTemplate::ConvolutionType::Conv1: 

						for (auto b : std::views::iota(0ULL, input.mView.extent(1))) {

							auto inputs1 = input.viewColumn(b);
							auto outputs1 = mOutputs.viewColumn(b);
							auto activations1 = mActivations.viewColumn(b);

							env.conv1(mWeights, inputs1, outputs1);
						}
						break;
					case LayerTemplate::ConvolutionType::None:

						env.batchedMatMulVec(mWeights, input, mOutputs);

						break;
					}

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