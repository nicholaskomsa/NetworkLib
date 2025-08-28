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

		template<Cpu::Tensor::ViewConcept ViewType>
		struct GpuView {
			ViewType mView;
			float* mGpu = nullptr, * mCpu = nullptr;
			std::size_t mSize = 0;

			GpuView() = default;
			GpuView(ViewType view, float* gpu, float* cpu)
				:mView(view), mGpu(gpu), mCpu(cpu){

				setSize();
			}

			void setSize() {
				mSize = Cpu::Tensor::area(mView);
			}

			void upload() {
				Error::checkCuda(cudaMemcpy(
					mGpu,
					mCpu,
					mSize * sizeof(float),
					cudaMemcpyHostToDevice));
			}
			void downloadAsync(const cudaStream_t& stream) {
				Error::checkCuda(cudaMemcpyAsync(
					mCpu,
					mGpu,
					mSize * sizeof(float),
					cudaMemcpyDeviceToHost,
					stream));
			}
			float* begin() {
				return mCpu;
			}
			float* end() {
				return mCpu + mSize;
			}
		};

		using GpuView1 = GpuView<Cpu::Tensor::View1>;
		using GpuView2 = GpuView<Cpu::Tensor::View2>;

		struct FloatSpace1 {

			GpuView<Cpu::Tensor::View1> mView;

			void create(std::size_t size) {
				
				Error::checkCuda(cudaMallocHost(&mView.mCpu, size * sizeof(float)));
				float* begin = mView.mCpu;
				Cpu::Tensor::advance(mView.mView, begin, size);
				mView.mSize = size;
				Error::checkCuda(cudaMalloc(reinterpret_cast<void**>(&mView.mGpu), size * sizeof(float)));
			}

			void destroy() {
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

		class Environment {
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
			const cublasHandle_t& getBlas() {
				return mHandle;
			}
			const cudaStream_t& getStream() {
				return mStream;
			}
			operator cudaStream_t() {
				return mStream;
			}

			void vecAddVec(const GpuView1& a1, GpuView1& o1){
				std::size_t size = o1.mSize;
				float alpha = 1.0f;
				auto result = cublasSaxpy(mHandle, size, &alpha, a1.mGpu, 1, o1.mGpu, 1);
				Error::checkBlas(result);
			}
			void vecMulMat(const GpuView1& i1, const GpuView2& w2, GpuView1& o1){

				float alpha = 1.0f;
				float beta = 0.0f;

				int r = w2.mView.extent(0);
				int c = w2.mView.extent(1);

				int k = i1.mSize;

				if (r != k)
					throw std::logic_error("matrix * vec incorrect dimensions");

				auto result = cublasSgemv(mHandle,
					CUBLAS_OP_N,
					c, r,
					&alpha,
					w2.mGpu, c,
					i1.mGpu, 1,
					&beta,
					o1.mGpu, 1);
				Error::checkBlas(result);
			}

			void sync() {
				Error::checkCuda(cudaDeviceSynchronize());
			}
			
			void example() {

				create();

				Gpu::FloatSpace1 fs1;
				Gpu::GpuView<Cpu::Tensor::View1> i, b, o;
				Gpu::GpuView<Cpu::Tensor::View2> w;

				std::size_t inputSize = 784
					, biasSize = 10;

				fs1.create(inputSize + biasSize * inputSize + biasSize * 2);

				auto begin = fs1.begin();

				fs1.advance(i, begin, inputSize);
				fs1.advance(w, begin, inputSize, biasSize);
				fs1.advance(b, begin, biasSize);
				fs1.advance(o, begin, biasSize);

				std::fill(w.begin(), w.end(), 1);
				std::fill(i.begin(), i.end(), 1);
				std::fill(b.begin(), b.end(), 1);

				w.mView[100, 9] = 0;

				fs1.mView.upload();

				sync();

				auto forward = [&]() {
					vecMulMat(i, w, o);
					vecAddVec(b, o);
					};
				forward();

				o.downloadAsync(getStream());

				sync();

				//cpu relu
				for (auto& f : o) {
					f = std::max(f, 0.0f);
					std::print("{} ", f);
				}		

				fs1.destroy();

				destroy();
			}

		private:
			cublasHandle_t mHandle;
			cudaStream_t mStream;
		};
	}
}