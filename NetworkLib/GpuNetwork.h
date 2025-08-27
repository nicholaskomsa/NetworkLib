#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <source_location>

#include "CpuTensor.h"

namespace NetworkLib {
	namespace Gpu {

		struct Error : public std::system_error {

			Error(std::errc code, const std::string& message) 
				: std::system_error(int(code), std::generic_category(), message) {}

			static void cudaError(cudaError_t result, const std::source_location& location = std::source_location::current()) {

				auto message = std::format(
					"Cuda Error ={}:\n{}\n"
					"{}\n{}\n{}\n"
					, int(result), cudaGetErrorString(result)
					, location.file_name(), location.line(), location.function_name());
				
				throw Error(std::errc::operation_canceled, message);
			}
			static void blasError(cublasStatus_t result, const std::source_location& location = std::source_location::current()) {
				
				auto getBLASString = [&]() {
					switch (result) {
					case CUBLAS_STATUS_SUCCESS:          return "Success";
					case CUBLAS_STATUS_NOT_INITIALIZED:  return "cuBLAS not initialized";
					case CUBLAS_STATUS_ALLOC_FAILED:     return "Resource allocation failed";
					case CUBLAS_STATUS_INVALID_VALUE:    return "Invalid value";
					case CUBLAS_STATUS_ARCH_MISMATCH:    return "Architecture mismatch";
					case CUBLAS_STATUS_MAPPING_ERROR:    return "Memory mapping error";
					case CUBLAS_STATUS_EXECUTION_FAILED: return "Execution failed";
					case CUBLAS_STATUS_INTERNAL_ERROR:   return "Internal error";
					default:                             return "Unknown cuBLAS error";
					}

					};
				
				auto message = std::format(
					"BLAS Error ={}:\n{}\n"
					"{}\n{}\n{}\n"
					, int(result), getBLASString()
					, location.file_name(), location.line(), location.function_name());

				throw Error(std::errc::operation_canceled, message);
			}

			static void checkCuda(cudaError_t result, const std::source_location& location = std::source_location::current()) {
				if (result != cudaSuccess)
					cudaError(result, location);
			}
			static void checkBlas(cublasStatus_t result, const std::source_location& location = std::source_location::current()) {
				if (result != CUBLAS_STATUS_SUCCESS)
					blasError(result);
			}
		};

		std::size_t getMemSize(std::size_t floatCount) {
			return floatCount * sizeof(float);
		}

		template<Cpu::Tensor::ViewConcept ViewType>
		struct GpuView {
			ViewType mView;
			float* mGpu = nullptr, * mCpu = nullptr;

			GpuView() = default;
			GpuView(ViewType view, float* gpu, float* cpu)
				:mView(view), mGpu(gpu), mCpu(cpu) {}

			void upload() {
				auto memSize = getMemSize(Cpu::Tensor::area(mView));
				Error::checkCuda(cudaMemcpy(
					mGpu,
					data(),
					memSize,
					cudaMemcpyHostToDevice));
			}
			void downloadAsync(const cudaStream_t& stream) {
				auto memSize = getMemSize(Cpu::Tensor::area(mView));
				Error::checkCuda(cudaMemcpyAsync(
					data(),
					mGpu,
					memSize,
					cudaMemcpyDeviceToHost,
					stream));
			}
			float* begin() {
				return data();
			}
			float* end() {
				return data() + Cpu::Tensor::area(mView);
			}

			float* data() { return mView.data_handle(); }
		};

		class Environment {

			cublasHandle_t mHandle;
			cudaStream_t mStream;

		public:
			void create() {
				Error::checkBlas(cublasCreate(&mHandle));
				Error::checkCuda(cudaStreamCreate(&mStream));
				Error::checkBlas(cublasSetStream(mHandle, mStream));
			}
			void destroy() {
				Error::checkCuda(cudaStreamDestroy(mStream));
				Error::checkBlas(cublasDestroy(mHandle));
			}

			void vecAddVec(const float* a, float* b, std::size_t size) const {
				float alpha = 1.0f;
				auto result = cublasSaxpy(mHandle, size, &alpha, a, 1, b, 1);
				Error::checkBlas(result);
			}

			struct FloatSpace1 {

				GpuView<Cpu::Tensor::View1> mView;

				void allocate(std::size_t size) {
					auto memSize = getMemSize(size);
					Error::checkCuda(cudaMallocHost(&mView.mCpu, memSize));
					float* begin = mView.mCpu;
					Cpu::Tensor::advance(mView.mView, begin, size);
					
					Error::checkCuda(cudaMalloc(reinterpret_cast<void**>(&mView.mGpu), memSize));
				}

				void free() {
					freeHost();
					freeGpu();
				}
				void freeHost() {
					if (mView.mCpu)
						Error::checkCuda(cudaFreeHost(mView.mCpu));
					mView.mCpu = nullptr;
				}
				void freeGpu() {
					if (mView.mGpu)
						Error::checkCuda(cudaFree(mView.mGpu));
					mView.mGpu = nullptr;
				}

				template<Cpu::Tensor::ViewConcept ViewType>
				float* getGpu(ViewType& view) {
					float* phost = mView.mCpu;
					float* vhost = view.data_handle();
					float* vgpu = mView.mGpu + (vhost - phost);
					return vgpu;
				}

				float* begin() { return mView.begin(); }
				float* end() { return mView.end(); }

				template<Cpu::Tensor::ViewConcept ViewType, typename... Dimensions>
				void advance(GpuView<ViewType>& gpuView, float*& begin, Dimensions&&...dimensions) {
					ViewType view;
					auto source = begin;
					Cpu::Tensor::advance(view, begin, dimensions...);
					gpuView = { view, getGpu(view), source };
				}
			};

			void example() {
				
				FloatSpace1 fs1;
				fs1.allocate(200);

				auto begin = fs1.begin();

				GpuView<Cpu::Tensor::View1> v1;
				GpuView<Cpu::Tensor::View2> v2;

				fs1.advance(v1, begin, 100);
				fs1.advance(v2, begin, 10, 10);

				std::iota(fs1.begin(), fs1.end(), 0);
				v2.mView[5, 5] = 16.333f;

				//fs1.mView.upload();
				v2.upload();

				std::fill(fs1.begin(), fs1.end(), 0);

				v1.downloadAsync(mStream);
				v2.downloadAsync(mStream);

				std::for_each(v1.begin(), v1.end(), [&](auto& f) {

					std::cout << f << ",";
					});

				std::for_each(v2.begin(), v2.end(), [&](auto& f) {

					std::cout << f << ",";
					});

				fs1.free();
			}

		};

		class Network {

			NetworkTemplate* mNetworkTemplate = nullptr;

			Environment::FloatSpace1 mGpuFloats;

			struct Layer {
				GpuView<Cpu::Tensor::View2> mWeights;
				GpuView<Cpu::Tensor::View1> mBias;
			};

			std::vector<Layer> mLayers;

		public:

			Network(NetworkTemplate* networkTemplate)
				: mNetworkTemplate(networkTemplate) {
			}

			void create() {

				auto& networkTemplate = *mNetworkTemplate;
				std::span<LayerTemplate> layerTemplates = networkTemplate.mLayerTemplates;

				auto& firstInputSize = networkTemplate.mInputSize;
				std::size_t size = 0, inputSize = firstInputSize;
				for (auto [n] : layerTemplates) {
					size += inputSize * n + n;
					inputSize = n;
				}

				mGpuFloats.allocate(size);

				mLayers.resize(layerTemplates.size());

				auto begin = mGpuFloats.begin();
				inputSize = firstInputSize;
				for (const auto& [layer, layerTemplate] : std::views::zip(mLayers, layerTemplates)) {

					const auto& [n] = layerTemplate;
					auto& [w, b] = layer;

					mGpuFloats.advance(w, begin, inputSize, n);
					mGpuFloats.advance(b, begin, n);

					inputSize = n;
				}
			}
			void destroy() {
				mGpuFloats.free();
			}

		};

	}
}