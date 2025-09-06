#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <source_location>

#include "CpuTensor.h"
#include "NetworkTemplate.h"

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

		template<typename T>
		concept ViewConcept = Cpu::Tensor::ViewConcept<T>;

		template<ViewConcept ViewType>
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
			void downloadAsync(cudaStream_t stream) {
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

			GpuView1 mView;

			void create(std::size_t size) {
				
				Error::checkCuda(cudaMallocHost(&mView.mCpu, size * sizeof(float)));
				mView.mView = Cpu::Tensor::View1(mView.mCpu, size);
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

			template<ViewConcept ViewType>
			float* getGpu(ViewType& view) {
				float* phost = mView.mCpu;
				float* vhost = view.data_handle();
				float* vgpu = mView.mGpu + (vhost - phost);
				return vgpu;
			}

			float* begin() { return mView.begin(); }
			float* end() { return mView.end(); }

			template<ViewConcept ViewType, typename... Dimensions>
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
			cublasHandle_t getBlas() {
				return mHandle;
			}
			cudaStream_t getStream() {
				return mStream;
			}
			operator cudaStream_t() {
				return mStream;
			}

			void vecScale( GpuView1& a1, float scale) {
				cublasSscal(mHandle, a1.mSize, &scale, a1.mGpu, 1);
			}
			void vecAddVec(const GpuView1& a1, GpuView1& o1){
				std::size_t size = o1.mSize;
				float alpha = 1.0f;
				auto result = cublasSaxpy(mHandle, size, &alpha, a1.mGpu, 1, o1.mGpu, 1);
				Error::checkBlas(result);
			}
			void matMulVec(const GpuView2& w2, const GpuView1& i1, GpuView1& o1) {
				//cuda and mdspan are in C++ style of row-major
				//but cublas wants Fortran style, col-major,
				float alpha = 1.0f;
				float beta = 0.0f;

				int r = w2.mView.extent(0);
				int c = w2.mView.extent(1);

				int k = i1.mSize;

				//if (c != k)
			//		throw std::logic_error("matrix * vec incorrect dimensions");

				auto result = cublasSgemv(mHandle,
					CUBLAS_OP_N,
					r, c,
					&alpha,
					w2.mGpu, r,
					i1.mGpu, 1,
					&beta,
					o1.mGpu, 1);
				Error::checkBlas(result);
			}
			void matTMulVec(const GpuView2& w2, const GpuView1& i1, GpuView1& o1){
				
				float alpha = 1.0f;
				float beta = 0.0f;

				int r = w2.mView.extent(0);
				int c = w2.mView.extent(1);

				int k = i1.mSize;
				
				auto result = cublasSgemv(mHandle,
					CUBLAS_OP_T,
					r, c,
					&alpha,
					w2.mGpu, r,
					i1.mGpu, 1,
					&beta,
					o1.mGpu, 1);
				Error::checkBlas(result);
			}
	
			void relu(const GpuView1& o1, GpuView1& a1);
			void applyReluPrime(const GpuView1& a1, GpuView1& p1);
			void softmax(const GpuView1& o1, GpuView1& a1);
			void diff(const GpuView1& desired1, const GpuView1& sought1, GpuView1& primes1);
			void updateWeights(Environment& env, const GpuView1& seen, GpuView2& weights, const GpuView1& primes, float learnRate);

			bool activationFunction(LayerTemplate::ActivationFunction af, const GpuView1& o1, GpuView1& a1) {
				
				using ActivationFunction = LayerTemplate::ActivationFunction;
				switch( af) {
				case ActivationFunction::ReLU:
					relu(o1, a1);
					return true;
				case ActivationFunction::SoftmaxCrossEntropy:
					softmax(o1, a1);
					return true;
				case ActivationFunction::None:
					return false;
				}
				return false;
			}
			bool activationFunctionPrime(LayerTemplate::ActivationFunction af, const GpuView1& a1, GpuView1& p1) {

				using ActivationFunction = LayerTemplate::ActivationFunction;
				switch (af) {
				case ActivationFunction::ReLU:
					applyReluPrime(a1, p1);
					return true;
				case ActivationFunction::None:
					return false;
				}
				return false;
			}
			void errorFunction(LayerTemplate::ActivationFunction af, const GpuView1& desired, GpuView1& sought, GpuView1& p1) {
				switch (af) {
				case LayerTemplate::ActivationFunction::None:
				case LayerTemplate::ActivationFunction::SoftmaxCrossEntropy:
					//softmax-cross-entropy is a diff
					diff(desired, sought, p1);
					return;
				}
			}
			void sync() {
				Error::checkCuda(cudaDeviceSynchronize());
			}
			
			static void example() {

				Environment env;
				env.create();

				Gpu::FloatSpace1 fs1;
				Gpu::GpuView1 i, i2, b, o, a;
				Gpu::GpuView2 w;

				std::size_t inputSize = 3
					, biasSize = 2;

				fs1.create(inputSize*2 + inputSize * biasSize + biasSize * 3);

				auto begin = fs1.begin();

				fs1.advance(i, begin, inputSize);
				fs1.advance(i2, begin, inputSize);
				fs1.advance(w, begin, biasSize, inputSize);
				fs1.advance(b, begin, biasSize);
				fs1.advance(o, begin, biasSize);
				fs1.advance(a, begin, biasSize);

				std::fill(w.begin(), w.end(), 1);
				std::fill(i.begin(), i.end(), 1);
				std::fill(b.begin(), b.end(), 1);

				//for( auto i : std::views::iota(0, 5))
				//w.mView[1, i] = 0;
				for( auto i : std::views::iota(0ULL, o.mSize))
					o.mView[i] = 1;

				fs1.mView.upload();

				env.sync();

				auto forward = [&]() {


					//env.matMulVec(w, i, o);
					//env.vecAddVec(b, o);
					env.matTMulVec(w, o, i2);
					
					//env.softmax(o, a);
					};
				forward();

				i.downloadAsync(env.getStream());
				i2.downloadAsync(env.getStream());
				o.downloadAsync(env.getStream());
				a.downloadAsync(env.getStream());

				env.sync();

				for (const auto& of :	i2)
					std::print("{} ", of);

				//for (const auto& [of, af] : std::views::zip(o,a)) 
				//	std::print("{} {}; ", of, af);
				
				fs1.destroy();

				env.destroy();
			}

		private:
			cublasHandle_t mHandle;
			cudaStream_t mStream;
		};
	}
}