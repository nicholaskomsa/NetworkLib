#pragma once

#include <vector>

namespace NetworkLib {

	struct LayerTemplate {
		std::size_t mNodeCount = 0;

		enum class ActivationFunction {
			None,
			ReLU,
			Sigmoid,
			Tanh,
			Softmax
		} mActivationFunction = ActivationFunction::None;

		enum class ConvolutionType {
			None,
			Conv1
		} mConvolutionType = ConvolutionType::None;

		std::size_t mKernelWidth = 0, mKernelHeight = 0;
		std::size_t mKernelNumber = 0;

		LayerTemplate(std::size_t nodeCount, ActivationFunction af) 
			: mNodeCount(nodeCount), mActivationFunction(af){}

		LayerTemplate(ConvolutionType convType, std::size_t kernelWidth, std::size_t kernelNumber, ActivationFunction af)
			: mNodeCount(0), mActivationFunction(af)
			, mConvolutionType(convType), mKernelWidth(kernelWidth), mKernelNumber(kernelNumber) {
		}
	};

	using LayerTemplates = std::vector<LayerTemplate>;

	struct NetworkTemplate {
		std::size_t mInputSize = 0, mBatchSize = 1;
		LayerTemplates mLayerTemplates;

		NetworkTemplate() = default;

		NetworkTemplate(std::size_t inputSize, std::size_t batchSize, const LayerTemplates& layerTemplates)
			: mInputSize(inputSize), mBatchSize(batchSize), mLayerTemplates(layerTemplates) {

			auto size = inputSize;
			for (auto& layer : mLayerTemplates) {

				switch (layer.mConvolutionType) {
					case LayerTemplate::ConvolutionType::Conv1: {
						//for Conv1, node count is determined by input size and kernel size
						size = size - layer.mKernelWidth + 1;
						layer.mNodeCount = size * layer.mKernelNumber;
						break;
					}
					case LayerTemplate::ConvolutionType::None:
						size = layer.mNodeCount;
						break;
				}
			}
		}
	
		std::size_t getTotalSize(bool backwards = true) const {

			std::size_t size = 0, inputSize = mInputSize;
			for (auto& layerTemplate : mLayerTemplates) {

				const auto n = layerTemplate.mNodeCount;
				//weights + bias + batchSize * (outputs + activations)
				std::size_t weightsSize;
				switch (layerTemplate.mConvolutionType) {
				case LayerTemplate::ConvolutionType::Conv1:
					//for Conv1, weights are determined by kernel size and number
					weightsSize = layerTemplate.mKernelWidth * layerTemplate.mKernelNumber;
					break;
				case LayerTemplate::ConvolutionType::None:
					weightsSize = n * inputSize;
					break;
				}
				size += weightsSize + n + mBatchSize * 2 * n;

				inputSize = n;
			}

			if (backwards)
				for (auto& layerTemplate : mLayerTemplates) {
					const auto n = layerTemplate.mNodeCount;
					size += mBatchSize * n; //batchSize * primes
				}

			return size;
		}
	};
}