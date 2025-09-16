#pragma once

#include <random>

#include "Parallel.h"
#include "GpuTensor.h"
#include "NetworkTemplate.h"

namespace NetworkLib {
	namespace Gpu {

		class Network {
		public:

			void create(NetworkTemplate* networkTemplate, bool backwards=true) {

				mNetworkTemplate = networkTemplate;

				const auto& nt = *mNetworkTemplate;
				auto& layerTemplates = nt.mLayerTemplates;
				auto batchSize = nt.mBatchSize;

				auto& firstInputSize = nt.mInputSize;
				std::size_t size = 0, inputSize = firstInputSize;
				for (auto& layerTemplate : layerTemplates) {
					const auto n = layerTemplate.mNodeCount;
					//weights + bias + batchSize * (outputs + activations)
					size += n * inputSize + n  + batchSize * 2 * n;
					inputSize = n;
				}

				if (backwards)
					for (auto& layerTemplate : layerTemplates) {
						const auto n = layerTemplate.mNodeCount;
						size += batchSize * n; //batchSize * primes
					}

				mGpuFloats.create(size);

				mLayers.resize(layerTemplates.size());

				auto begin = mGpuFloats.begin();
				inputSize = firstInputSize;
				for (const auto& [layer, layerTemplate] : std::views::zip(mLayers, layerTemplates)) {

					const auto n = layerTemplate.mNodeCount;
					const auto af = layerTemplate.mActivationFunction;
					
					layer.advance(mGpuFloats, begin, n, inputSize, af, batchSize, backwards);

					inputSize = n;
				}
			}

			void destroy() {
				mGpuFloats.destroy();
			}

			void intializeId( std::size_t id ){
				std::mt19937_64 random(id);
				initialize(random);
			}
			void initialize(std::mt19937_64& random) {

				for (auto& layer : mLayers) 
					layer.generate(random);
				
				applyKHScales();
			}

			void upload() {
				mGpuFloats.upload();
			}

			void applyKHScales() {

				Parallel parallel(mLayers.size());
				parallel([&](auto& section) {

					for (auto l : section.mIotaView) {

						auto& layer = mLayers[l];

						std::size_t inputSize = (l == 0)? 
							mNetworkTemplate->mInputSize : mLayers[l - 1].mBias.mSize;

						layer.applyKHScaleUniform(inputSize);
					}
					});
			}

			const GpuView1 forward(Environment& env, const GpuView1& seen) {
				
				GpuView1 i = seen;

				for( auto& layer : mLayers ) 
					i = layer.forward(env, i);
				
				return i;
			}

			const GpuView2& forward(Environment& env, const GpuView2& seenBatch) {


				const GpuView2* i2 = &seenBatch;

				for (auto& layer : mLayers)
					i2 = &layer.forward(env, *i2);

				return *i2;
			}

			void backward(Environment& env, const GpuView1& seen, const GpuView1& desired, std::size_t batch, float learnRate) {

				auto backLayer = [&]() {
					auto& back = mLayers.back();

					auto af = back.mActivationFunction;

					const auto sought = back.mActivations.viewColumn(batch);
					auto p1 = back.mPrimes.viewColumn(batch);

					//output layer makes a comparison between desired and sought
					env.errorFunction(af, desired, sought, p1);
					};

				auto hiddenLayers = [&]() {

					for (auto l : std::views::iota(0ULL, mLayers.size() - 1) | std::views::reverse) {

						auto& layer = mLayers[l];
						auto& nextLayer = mLayers[l + 1];

						auto p1 = layer.mPrimes.viewColumn(batch);
						auto o1 = layer.mOutputs.viewColumn(batch);
						auto np1 = nextLayer.mPrimes.viewColumn(batch);

						auto& nw2 = nextLayer.mWeights;

						auto af = layer.mActivationFunction;

						env.matTMulVec(nw2, np1, p1);
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

				auto backLayer = [&]() {
					auto& back = mLayers.back();

					auto af = back.mActivationFunction;

					const auto& soughtBatch = back.mActivations;
					auto& p2 = back.mPrimes;
					
					//output layer makes a comparison between desired and sought
					auto sought1 = soughtBatch.flatten();
					auto p1 = p2.flatten();
					auto desired1 = desiredBatch.flatten();

					env.errorFunction(af, desired1, sought1, p1);
					};

				auto hiddenLayers = [&]() {

					for (auto l : std::views::iota(0ULL, mLayers.size() - 1) | std::views::reverse) {

						auto& layer = mLayers[l];
						auto& nextLayer = mLayers[l + 1];

						auto& p2 = layer.mPrimes;
						auto& o2 = layer.mOutputs;
						auto& np2 = nextLayer.mPrimes;
						auto& nw2 = nextLayer.mWeights;

						auto af = layer.mActivationFunction;

						env.batchedMatTMulVec(nw2, np2, p2);

						auto o1 = o2.flatten();
						auto p1 = p2.flatten();
						env.activationFunctionPrime(af, o1, p1);
					}
					};

				auto updateWeights = [&]() {
					for (auto l : std::views::iota(0ULL, mLayers.size())) {

						auto& layer = mLayers[l];

						const GpuView2* input = nullptr;
						if (l == 0)
							input = &seenBatch;
						else
							input = &mLayers[l - 1].mActivations;

						env.batchedUpdateWeights(*input, layer.mWeights, layer.mPrimes, learnRate);
					}
					};

				backLayer();
				hiddenLayers();
				updateWeights();
			}

			
			const GpuView2 getSought() {
				return mLayers.back().mActivations;
			}
			class Layer {
			public:
				void generate(std::mt19937_64& random) {

					std::uniform_real_distribution<float> reals(-1.0f, 1.0f);

					std::generate(mWeights.begin(), mWeights.end(), [&]() {
						return reals(random);
						});

					std::generate(mBias.begin(), mBias.end(), [&]() {
						return reals(random);
						});
				}

				void applyKHScaleUniform(std::size_t inputSize) {

					auto scale = std::sqrtf(6.0f / (inputSize + mBias.mSize));

					for (auto& w : mWeights)
						w *= scale;
				}

				const GpuView1 forward(Environment& env, const GpuView1& input, std::size_t batch = 0) {

					auto outputs1 = mOutputs.viewColumn(batch);
					auto activations1 = mActivations.viewColumn(batch);

					env.matMulVec(mWeights, input, outputs1);
					env.vecAddVec(mBias, outputs1);
					env.activationFunction(mActivationFunction, outputs1, activations1);

					return activations1;
				}
				const GpuView2& forward(Environment& env, const GpuView2& input) {
				
					env.batchedMatMulVec1(mWeights, input, mOutputs);
					env.batchedBroadcastAdd(mBias, mOutputs);

					auto outputs1 = mOutputs.flatten();
					auto activations1 = mActivations.flatten();

					env.activationFunction(mActivationFunction, outputs1, activations1);
					return mActivations;
				}
				void advance(FloatSpace1& gpuFloats, float*& begin
					, std::size_t n, std::size_t inputSize
					, LayerTemplate::ActivationFunction af, std::size_t batchSize, bool backwards) {
					
					gpuFloats.advance(mWeights, begin, n, inputSize);
					gpuFloats.advance(mBias, begin, n);
					gpuFloats.advance(mOutputs, begin, n, batchSize);
					gpuFloats.advance(mActivations, begin, n, batchSize);

					mActivationFunction = af;

					if (backwards)
						gpuFloats.advance(mPrimes, begin, n, batchSize);
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

		private:
			NetworkTemplate* mNetworkTemplate = nullptr;

			FloatSpace1 mGpuFloats;

			Layers mLayers;
		};

	}
}