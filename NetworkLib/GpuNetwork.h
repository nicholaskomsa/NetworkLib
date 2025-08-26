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

		class Environment {

			cublasHandle_t mHandle;
			cudaStream_t mStream;

		public:
			void setup() {
				Error::checkBlas(cublasCreate(&mHandle));
				Error::checkCuda(cudaStreamCreate(&mStream));
				Error::checkBlas(cublasSetStream(mHandle, mStream));
			}
			void shutdown() {
				Error::checkCuda(cudaStreamDestroy(mStream));
				Error::checkBlas(cublasDestroy(mHandle));
			}

			void vecAddVec(const float* a, float* b, std::size_t size) const {
				float alpha = 1.0f;
				auto result = cublasSaxpy(mHandle, size, &alpha, a, 1, b, 1);
				Error::checkBlas(result);
			}
			static std::size_t getMemSize(std::size_t floatCount) {
				return floatCount * sizeof(float);
			}
			template<Cpu::Tensor::ViewConcept ViewType>
			struct GpuView {
				ViewType mView;
				float* mGpu = nullptr;

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

			struct FloatSpace1 {

				GpuView<Cpu::Tensor::View1> mView;

				void allocate(std::size_t size) {
					auto memSize = getMemSize(size);
					float* phost = nullptr;
					Error::checkCuda(cudaMallocHost(&phost, memSize));
					Cpu::Tensor::advance(mView.mView, phost, size);
					
					Error::checkCuda(cudaMalloc(reinterpret_cast<void**>(&mView.mGpu), memSize));
				}

				void free() {
					float* phost = mView.mView.data_handle();
					if (phost)
						Error::checkCuda(cudaFreeHost(phost));
					if (mView.mGpu)
						Error::checkCuda(cudaFree(mView.mGpu));

					mView = {};
				}

				template<Cpu::Tensor::ViewConcept ViewType>
				float* getGpu(ViewType& view) {
					float* phost = mView.mView.data_handle();
					float* vhost = view.data_handle();
					float* vgpu = mView.mGpu + (vhost - phost);
					return vgpu;
				}

				float* begin() { return mView.begin(); }
				float* end() { return mView.end(); }

				template<Cpu::Tensor::ViewConcept ViewType, typename... Dimensions>
				GpuView<ViewType> advance(float*& begin, Dimensions&&...dimensions) {
					ViewType view;
					Cpu::Tensor::advance(view, begin, dimensions...);
					return { view, getGpu(view) };
				}
			};

			void example() {

				FloatSpace1 fs1;
				fs1.allocate(200);

				auto begin = fs1.begin();
				auto v1 = fs1.advance<Cpu::Tensor::View1>(begin, 100);
				auto v2 = fs1.advance<Cpu::Tensor::View2>(begin, 10, 10);

				v1.downloadAsync(mStream);
				v2.downloadAsync(mStream);

				//fs1.downloadAsync(v2, mStream);

				std::for_each(v1.begin(), v1.end(), [&](auto& f) {

					std::cout << f << ",";
					});

				std::for_each(v2.begin(), v2.end(), [&](auto& f) {

					std::cout << f << ",";
					});
				//std::for_each(fs1.begin(v2), fs1.end(v2), [&](auto& f) {

				//	std::cout << f << ",";
				//	});

				fs1.free();
			}

		};

		class Network {

		public:

		};

	}
}