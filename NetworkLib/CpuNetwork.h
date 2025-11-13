#pragma once

#include <random>
#include <unordered_map>

#include "parallel.h"
#include "CpuTensor.h"
#include "NetworkTemplate.h"

namespace NetworkLib {


	namespace Cpu {

		class Network {
		public:

			void create(NetworkTemplate* networkTemplate, bool backwards = true) {

				mNetworkTemplate = networkTemplate;

				const auto& nt = *mNetworkTemplate;
				auto& layerTemplates = nt.mLayerTemplates;
				auto batchSize = nt.mBatchSize;

				std::size_t size = nt.getTotalSize(backwards);

				mFloats.create(size);

				mLayers.resize(layerTemplates.size());

				auto begin = mFloats.begin();

				auto groupComponent = [&](auto&& setupFunctor) ->View1 {

					auto componentBegin = begin;

					for (std::size_t idx =0; const auto& [layer, layerTemplate] : std::views::zip(mLayers, layerTemplates)) 
						setupFunctor(layer, layerTemplate, idx++);

					std::size_t size = std::distance(componentBegin, begin);
					auto view = Cpu::Tensor::View1(componentBegin, std::array{ size });
					return view;
					};

				//mWeights, etc refer to all weights from all layers, they are grouped
				mWeights = groupComponent([&](auto& layer, auto& layerTemplate, std::size_t idx) {

					switch( layerTemplate.mConvolutionType ) {
						case LayerTemplate::ConvolutionType::Conv1: {

							Tensor::advance(layer.mWeights, begin, layerTemplate.mKernelWidth, 1ULL, layerTemplate.mKernelNumber );
							break;
						}
						case LayerTemplate::ConvolutionType::None: {
						
							auto inputSize = ( idx == 0 ) ?
								nt.mInputSize : layerTemplates[idx-1].mNodeCount;

							Tensor::advance(layer.mWeights, begin, layerTemplate.mNodeCount, inputSize, 1ULL);
							break;
						}
					}
	
					});
				mBias = groupComponent([&](auto& layer, auto& layerTemplate, std::size_t idx) {
					Tensor::advance(layer.mBias, begin, layerTemplate.mNodeCount);
					});
				mOutputs = groupComponent([&](auto& layer, auto& layerTemplate, std::size_t idx) {
					Tensor::advance(layer.mOutputs, begin, layerTemplate.mNodeCount, batchSize);
					});
				mActivations = groupComponent([&](auto& layer, auto& layerTemplate, std::size_t idx) {
					Tensor::advance(layer.mActivations, begin, layerTemplate.mNodeCount, batchSize);
					});

				if (backwards)
					mPrimes = groupComponent([&](auto& layer, auto& layerTemplate, std::size_t idx) {
						Tensor::advance(layer.mPrimes, begin, layerTemplate.mNodeCount, batchSize);
						});
			}

			void destroy() {
				mFloats.destroy();
			}

			void mirror(const Network& other) {

				auto weights = Tensor::view(mWeights)
					, otherWeights = Tensor::view(other.mWeights);
				auto bias = Tensor::view(mBias)
					, otherBias = Tensor::view(other.mBias);

				std::copy(otherWeights.begin(), otherWeights.end(), weights.begin() );
				std::copy(otherBias.begin(), otherBias.end(), bias.begin());	
			}


			void initializeId(std::size_t id) {
				mId = id;
				std::mt19937_64 random(id);
				initialize(random);
			}
			void initialize(std::mt19937_64& random) {

				for (auto& layer : mLayers)
					layer.generate(random);

				applyKHScales();
			}


			void applyKHScales() {

				Parallel parallel(mLayers.size());
				parallel([&](auto& section) {

					for (auto l : section.mIotaView) {

						auto& layer = mLayers[l];

						std::size_t inputSize = (l == 0) ?
							mNetworkTemplate->mInputSize : mLayers[l - 1].mBias.extent(0);

						layer.applyKHScaleUniform(inputSize);
					}
					});
			}

			const View1 forward(View1 seen, std::size_t batch = 0) {

				for (auto& layer : mLayers)
					seen = layer.forward(seen, batch);

				return seen;
			}

			const View2 forward(View2 seenBatch) {

				for (auto& layer : mLayers)
					seenBatch = layer.forward(seenBatch);

				return seenBatch;
			}

			void resetScore() {
				mMisses = 0;
			}
			void resetMse() {
				mMse = 0.0f;
			}
			void score(View2 soughtBatch, View2 desiredBatch) {
				/*
				for (auto batch : std::views::iota(0ULL, soughtBatch.extent(1))) {
				
					auto sought = Tensor::viewColumn(soughtBatch, batch);
					auto desired = Tensor::viewColumn(desiredBatch, batch);
					
					auto size = sought.extent(0);

					auto soughtView = Cpu::Tensor::view(sought);
					auto desiredView = Cpu::Tensor::view(desired);

					auto maxSought = std::max_element(desiredView.begin(), desiredView.end());
					auto maxDesired = std::max_element(desiredView.begin(), desiredView.end());

					auto maxSoughtIndex = std::distance(soughtView.begin(), maxSought);
					auto maxDesiredIndex = std::distance(desiredView.begin(), maxDesired);

					if (maxSoughtIndex != maxDesiredIndex)
						++mMisses;
				}
				*/
			}
			static float mse( const View1& sought, const View1& desired) {
				float sum = 0.0f;
				for( auto i : std::views::iota(0ULL, sought.extent(0)) ) {
					
					auto diff = desired[i] - sought[i];
					sum += diff * diff;
				}
				sum /= sought.extent(0);
				return sum;
			}
			void mse(View2 soughtBatch, View2 desiredBatch) {

			}
			void backward(const View1& seen, const View1& desired, float learnRate, std::size_t batch = 0) {

			}

			void backward(const View2& seenBatch, const View2& desiredBatch, float learnRate) {

			}

			View2 getSought() {
				return mLayers.back().mActivations;
			}
			View2 getOutput() {
				return mLayers.back().mOutputs;
			}
			View2 getPrimes() {
				return mLayers.back().mPrimes;
			}

			class Layer {
			public:
				void generate(std::mt19937_64& random) {

					std::uniform_real_distribution<float> reals(-1.0f, 1.0f);

					auto weights = Tensor::view(mWeights);
					auto bias = Tensor::view(mBias);

					std::generate(weights.begin(), weights.end(), [&]() {
						return reals(random);
						});

					std::generate(bias.begin(), bias.end(), [&]() {
						return reals(random);
						});
				}

				void applyKHScaleUniform(std::size_t inputSize) {

					auto scale = std::sqrtf(6.0f / (inputSize + mBias.extent(0)));

					auto weights = Tensor::view(mWeights);

					for (auto& w : weights)
						w *= scale;
				}

				const View1 forward(const View1& input, std::size_t batch = 0) {

					//auto activations1 = Tensor::viewColumn(mOutputs, batch);

					//return activations1;
					return {};
				}
				const View2& forward(const View2& input) {
					return {};
				}

				View3 mWeights;
				View1 mBias;
				View2 mOutputs, mActivations, mPrimes;
			};

			using Layers = std::vector<Layer>;

			Layer& getLayer(std::size_t i) {
				return mLayers[i];
			}

		public:
			NetworkTemplate* mNetworkTemplate = nullptr;

			FloatSpace1 mFloats;

			View1 mWeights, mBias, mOutputs, mActivations, mPrimes;

			Layers mLayers;
			
			float mMse = 0.0f;
			std::size_t mMisses = 0;
			float mAccuracy = 0.0f;

			using Id = std::size_t;
			Id mId = 0;
		};

		using NetworksView = std::span<Network>;
		using Networks = std::vector<Network>;
		using NetworksMap = std::unordered_map<Network::Id, Network>;

		static void networkExample() {

			Network network;
			NetworkTemplate nt  ;
			nt.mInputSize = 4;
			nt.mBatchSize = 3;
			nt.mLayerTemplates = {
				{5, LayerTemplate::ActivationFunction::ReLU},
				{6, LayerTemplate::ActivationFunction::ReLU},
				{2, LayerTemplate::ActivationFunction::Softmax}
			};
			network.create(&nt);
			network.initializeId(1);
			//auto input = Tensor::View2((float*)nullptr, std::array{ 4, 3 });
			//auto output = network.forward(input, 0);
			network.destroy();
		}
	}
}