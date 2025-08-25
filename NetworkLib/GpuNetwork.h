#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <source_location>


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

		class GPUVector {

			float* mVector = nullptr;
			int mLength = 0;

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
				return mLength * sizeof(float);
			}

			std::size_t getLength() const {
				return mLength;
			}
			float* getData() const {
				return mVector;
			}
		};

		class HostVector {

			float* mVector{ nullptr };

			GPUVector* mGPUVector{ nullptr };
		public:

			HostVector() = default;
			HostVector(GPUVector& gpuVector) {
				allocate(gpuVector);
			}

			~HostVector() {
				free();
			}

			void allocate(GPUVector& gpuVector) {
				bool allocate = true;

				if (mGPUVector) {

					auto oldSize = mGPUVector->getMemSize();
					auto newSize = gpuVector.getMemSize();

					if (oldSize == newSize)
						allocate = false;
					else
						free();
				}

				mGPUVector = &gpuVector;

				if (allocate)
					Error::checkCuda(cudaMallocHost(&mVector, getMemSize()));
			}
			void free() {
				if (mVector){
					Error::checkCuda(cudaFreeHost(mVector));
					mVector = nullptr;
				}
			}

			float* getData() { return mVector; }

			std::size_t getMemSize() const {
				return mGPUVector->getMemSize();
			}

			std::size_t getLength() const {
				return mGPUVector->getLength();
			}

			void download() const {
				Error::checkCuda(cudaMemcpy(mVector, mGPUVector->getData(), mGPUVector->getMemSize(), cudaMemcpyDeviceToHost));
			}
			void download(GPUVector& gpuVector) {
				allocate(gpuVector);
				download();
			}
			void upload() const {
				Error::checkCuda(cudaMemcpy(mGPUVector->getData(), mVector, mGPUVector->getMemSize(), cudaMemcpyHostToDevice));
			}

			float* begin() { return mVector; }
			float* end() { return mVector + getLength(); }

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

			void vecAddVec(const GPUVector& a, GPUVector& bOut) const {
				float alpha = 1.0f;
				int n = a.getLength();

				auto result = cublasSaxpy(mHandle, n, &alpha, a.getData(), 1, bOut.getData(), 1);
				Error::checkBlas(result);
			}

			void example() {

				GPUVector g1;
				g1.allocate(20);

				HostVector h1;
				h1.allocate(g1);

				std::iota(h1.begin(), h1.end(), 0);
				h1.upload();

				vecAddVec(g1, g1);

				h1.download();

				for (auto& v : h1)
					std::cout << v << ", ";

				g1.free();
			}

		};

		class Network {

		public:

		};

	}
}