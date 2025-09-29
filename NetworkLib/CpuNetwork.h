#pragma once

#include <random>

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

				auto& firstInputSize = nt.mInputSize;
				std::size_t size = 0, inputSize = firstInputSize;
				for (auto& layerTemplate : layerTemplates) {
					const auto n = layerTemplate.mNodeCount;
					//weights + bias + batchSize * (outputs + activations)
					size += n * inputSize + n + batchSize * 2 * n;
					inputSize = n;
				}

				if (backwards)
					for (auto& layerTemplate : layerTemplates) {
						const auto n = layerTemplate.mNodeCount;
						size += batchSize * n; //batchSize * primes
					}

				mFloats.create(size);

				mLayers.resize(layerTemplates.size());

				auto begin = mFloats.begin();

				auto groupComponent = [&](auto&& setupFunctor) ->View1 {

					auto componentBegin = begin;

					inputSize = firstInputSize;
					for (const auto& [layer, layerTemplate] : std::views::zip(mLayers, layerTemplates)) {

						const auto n = layerTemplate.mNodeCount;

						setupFunctor(layer, layerTemplate, n, inputSize);

						inputSize = n;
					}

					std::size_t size = std::distance(componentBegin, begin);
					auto view = Cpu::Tensor::View1(componentBegin, std::array{ size });
					return view;
					};

				//mWeights, etc refer to all weights from all layers, they are grouped
				mWeights = groupComponent([&](auto& layer, auto& layerTemplate, std::size_t n, std::size_t inputSize) {
					layer.mActivationFunction = layerTemplate.mActivationFunction;
					Tensor::advance(layer.mWeights, begin, n, inputSize);
					});
				mBias = groupComponent([&](auto& layer, auto& layerTemplate, std::size_t n, std::size_t inputSize) {
					Tensor::advance(layer.mBias, begin, n);
					});
				mOutputs = groupComponent([&](auto& layer, auto& layerTemplate, std::size_t n, std::size_t inputSize) {
					Tensor::advance(layer.mOutputs, begin, n, batchSize);
					});
				mActivations = groupComponent([&](auto& layer, auto& layerTemplate, std::size_t n, std::size_t inputSize) {
					Tensor::advance(layer.mActivations, begin, n, batchSize);
					});

				if (backwards)
					mPrimes = groupComponent([&](auto& layer, auto& layerTemplate, std::size_t n, std::size_t inputSize) {
					Tensor::advance(layer.mPrimes, begin, n, batchSize);
						});
			}

			void destroy() {
				mFloats.destroy();
			}


			void intializeId(std::size_t id) {
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

					auto activations1 = Tensor::viewColumn(mOutputs, batch);

					return activations1;
				}
				const View2& forward(const View2& input) {

					return mActivations;
				}

				View2 mWeights;
				View1 mBias;
				View2 mOutputs, mActivations, mPrimes;

				LayerTemplate::ActivationFunction mActivationFunction = LayerTemplate::ActivationFunction::None;
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
		};

		using NetworksView = std::span<Network>;
		using Networks = std::vector<Network>;

		static void networkExample() {

			Network network;
			NetworkTemplate nt;
			nt.mInputSize = 4;
			nt.mBatchSize = 3;
			nt.mLayerTemplates = {
				{5, LayerTemplate::ActivationFunction::ReLU},
				{6, LayerTemplate::ActivationFunction::ReLU},
				{2, LayerTemplate::ActivationFunction::Softmax}
			};
			network.create(&nt);
			network.intializeId(1);
			auto input = Tensor::View2((float*)nullptr, std::array{ 4, 3 });
			//auto output = network.forward(input, 0);
			network.destroy();
		}
	}
}