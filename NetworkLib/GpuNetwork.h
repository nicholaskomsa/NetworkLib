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

			struct FloatSpace1 {

				Cpu::Tensor::View1 mHostView;
				float* mGpuFloats=nullptr;

				void allocate(std::size_t size) {
					auto memSize = size * sizeof(float);
					float* phost = nullptr;
					Error::checkCuda(cudaMallocHost(&phost, memSize));
					Cpu::Tensor::advance(mHostView, phost, size);
					
					Error::checkCuda(cudaMalloc(reinterpret_cast<void**>(&mGpuFloats), memSize));
				}

				void free() {
					float* phost = mHostView.data_handle();
					if (phost)
						Error::checkCuda(cudaFreeHost(phost));
					if (mGpuFloats)
						Error::checkCuda(cudaFree(mGpuFloats));

					mHostView = {};
					mGpuFloats = nullptr;
				}

				template<Cpu::Tensor::ViewConcept ViewType>
				float* getGpuOffset(const ViewType& view) {
					float* phost = mHostView.data_handle();
					float* vhost = view.data_handle();
					return mGpuFloats + ( vhost-phost );
				}

				template<Cpu::Tensor::ViewConcept ViewType>
				void upload(const ViewType& view) {	
					auto size = Cpu::Tensor::area(view);
					auto memSize = size * sizeof(float);
					const float* phost = view.data_handle();
					float* ghost = getGpuOffset(view);
					
					Error::checkCuda(cudaMemcpy(ghost, phost, memSize, cudaMemcpyHostToDevice));
				}
				template<Cpu::Tensor::ViewConcept ViewType>
				void downloadAsync(ViewType& view, const cudaStream_t& stream) {

					auto size = Cpu::Tensor::area(view);
					auto memSize = size * sizeof(float);
					float* phost = view.data_handle();
					float* ghost = getGpuOffset(view);

					Error::checkCuda(cudaMemcpyAsync(
						phost,
						ghost,
						memSize,
						cudaMemcpyDeviceToHost,
						stream));

				}

				template<Cpu::Tensor::ViewConcept ViewType>
				float* begin(ViewType& view) {
					return view.data_handle();
				}
				template<Cpu::Tensor::ViewConcept ViewType>
				float* end(ViewType& view) {
					return view.data_handle() + Cpu::Tensor::area(view);
				}
				float* begin() { return begin(mHostView); }
				float* end() { return end(mHostView); }

			};

			void example() {

				FloatSpace1 fs1;
				fs1.allocate(200);

				auto begin = fs1.begin();
				Cpu::Tensor::View1 v1, v2;
				Cpu::Tensor::advance(v1, begin, 100);
				Cpu::Tensor::advance(v2, begin, 100);

				std::iota(fs1.begin(), fs1.end(), 0);
				std::iota(fs1.begin(v2), fs1.end(v2), 0);

				fs1.upload(v1);
				fs1.upload(v2);

				std::fill(fs1.begin(v2), fs1.end(v2), 0);

				fs1.downloadAsync(v2, mStream);

				std::for_each(fs1.begin(v1), fs1.end(v1), [&](auto& f) {

					std::cout << f << ",";
					});
				
				std::for_each(fs1.begin(v2), fs1.end(v2), [&](auto& f) {

					std::cout << f << ",";
					});

				fs1.free();
			}

		};

		class Network {

		public:

		};

	}
}