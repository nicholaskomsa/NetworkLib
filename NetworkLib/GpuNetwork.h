#pragma once

#include <random>

#include "Parallel.h"
#include "GpuTensor.h"
#include "NetworkTemplate.h"

namespace NetworkLib {
	namespace Gpu {

		class Network {
		public:
			static void applyKHScaleUniform(std::size_t inputSize, std::size_t size, GpuView2 weights) {

				auto scale = std::sqrtf(6.0f / (inputSize + size));

				for (auto& w : weights)
					w *= scale;
			}

			Network(NetworkTemplate* networkTemplate)
				: mNetworkTemplate(networkTemplate) {
			}

			void create() {

				auto& networkTemplate = *mNetworkTemplate;
				std::span<LayerTemplate> layerTemplates = networkTemplate.mLayerTemplates;

				auto& firstInputSize = networkTemplate.mInputSize;
				std::size_t size = 0, inputSize = firstInputSize;
				for (auto& layerTemplate : layerTemplates) {
					const auto n = layerTemplate.mNodeCount;
					size += inputSize * n + n * 3;
					inputSize = n;
				}

				mGpuFloats.create(size);

				mLayers.resize(layerTemplates.size());

				auto begin = mGpuFloats.begin();
				inputSize = firstInputSize;
				for (const auto& [layer, layerTemplate] : std::views::zip(mLayers, layerTemplates)) {

					const auto n = layerTemplate.mNodeCount;
					auto& [w, b, o, a, af] = layer;

					mGpuFloats.advance(w, begin, inputSize, n);
					mGpuFloats.advance(b, begin, n);
					mGpuFloats.advance(o, begin, n);
					mGpuFloats.advance(a, begin, n);

					af = layerTemplate.mActivationFunction;

					inputSize = n;
				}
			}
			void destroy() {
				mGpuFloats.destroy();
			}

			void initialize(std::mt19937& random) {

				std::uniform_real_distribution<float> reals(-1.0f, 1.0f);

				for (auto& layer : mLayers) {

					auto& w = layer.mWeights;
					auto& b = layer.mBias;

					std::generate(w.begin(), w.end(), [&]() {
						return reals(random);
						});

					std::generate(b.begin(), b.end(), [&]() {
						return reals(random);
						});
				}

				applyKHScales();
			}

			void upload() {
				mGpuFloats.mView.upload();
			}

			void applyKHScales() {

				Parallel parallel(mLayers.size());
				parallel([&](auto& section) {

					for (auto l : section.mIotaView) {

						auto& layer = mLayers[l];

						std::size_t inputSize
							, size = layer.mBias.mSize;

						if (l == 0)
							inputSize = mNetworkTemplate->mInputSize;
						else
							inputSize = mLayers[l - 1].mBias.mSize;

						auto& w = layer.mWeights;

						applyKHScaleUniform(inputSize, size, w);
					}
					});
			}

			void forward(Environment& env, const GpuView1& input) {
				
				const GpuView1* i = &input;

			
				for( auto& layer : mLayers ) {
					
					const auto& w = layer.mWeights;
					const auto& b = layer.mBias;
					auto& o = layer.mOutputs;
					auto& a = layer.mActivations;

					env.matMulVec(w, *i, o);
					env.vecAddVec(b, o);

					const auto& af = layer.mActivationFunction;

					if( env.activationFunction(af,o,a))
						i = &a;
					else
						i = &o;
				}
			}

		private:
			NetworkTemplate* mNetworkTemplate = nullptr;

			FloatSpace1 mGpuFloats;

			struct Layer {
				GpuView<Cpu::Tensor::View2> mWeights;
				GpuView<Cpu::Tensor::View1> mBias;
				GpuView<Cpu::Tensor::View1> mOutputs;
				GpuView<Cpu::Tensor::View1> mActivations;
				LayerTemplate::ActivationFunction mActivationFunction = LayerTemplate::ActivationFunction::None;
			};

			std::vector<Layer> mLayers;
		};

	}
}