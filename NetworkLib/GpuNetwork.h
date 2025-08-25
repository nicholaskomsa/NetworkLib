#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <source_location>

// Helper functions and utilities to work with CUDA
//#include <helper_functions.h>
//#include <helper_cuda.h>

namespace NetworkLib {
	namespace Gpu {

		struct Error : public std::system_error {

			Error(std::errc code, const std::string& message) 
				: std::system_error(int(code), std::generic_category(), message) {}

			static void cudaError(cudaError_t result, const std::source_location& location) {

				auto message = std::format(
					"Cuda Error ={}:\n{}\n"
					"{}\n{}\n{}\n"
					, int(result), cudaGetErrorString(result)
					, location.file_name(), location.line(), location.function_name());
				
				throw Error(std::errc::operation_canceled, message);
			}

			static void checkCuda(cudaError_t result, const std::source_location& location = std::source_location::current()) {
				if (result != cudaSuccess)
					cudaError(result, location);
			}
		};

		class GPUVector {

			float* mVector{ nullptr };
			int mLength{ 0 };

		public:
			~GPUVector() {
				free();
			}

			void allocate(int length) {

				if (length != mLength) {
					mLength = length;

					if (mVector != nullptr)
						free();

					Error::checkCuda(cudaMalloc(reinterpret_cast<void**>(&mVector), getMemSize()));
				}
			}
			void free() {
				if (mVector)
					Error::checkCuda(cudaFree(mVector));
				mVector = nullptr;
			}
			int getMemSize() const {
			}

			std::size_t getLength() const {

			}
			float* getData() const {

			}
		};

		class Environment {

			cublasHandle_t mHandle;
			cudaStream_t mStream;

		public:
			void setup() {
				cublasCreate(&mHandle);
				cudaStreamCreate(&mStream);

				cublasSetStream(mHandle, mStream);
			}
			void shutdown() {
				cudaStreamDestroy(mStream);
				cublasDestroy(mHandle);
			}
		};

		class Network {

		public:

		};

	}
}