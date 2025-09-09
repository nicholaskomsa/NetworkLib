#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <source_location>

#include <sstream>

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

			static void missMatchError(auto& a, auto& b, const std::source_location& location = std::source_location::current()) {
				auto message = std::format("{}x{} mismatch", a, b);
				throw Error(std::errc::invalid_argument, message);
			}

			static void checkMissMatch(auto& a, auto& b, const std::source_location& location = std::source_location::current()) {
				if (a != b)
					missMatchError(a, b, location);
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
			void downloadAsync(cudaStream_t stream) const {
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
			const float* begin() const{
				return mCpu;
			}
			const float* end() const {
				return mCpu + mSize;
			}
		};

		using GpuView1 = GpuView<Cpu::Tensor::View1>;
		using GpuView2 = GpuView<Cpu::Tensor::View2>;

		struct FloatSpace1 {

			GpuView1 mView;

			void create(std::size_t size) {
				
				float* cpu, *gpu;
				Error::checkCuda(cudaMallocHost(&cpu, size * sizeof(float)));
				Error::checkCuda(cudaMalloc(reinterpret_cast<void**>(&gpu), size * sizeof(float)));
				mView = { Cpu::Tensor::View1(cpu, size), gpu, cpu };
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

			void upload(){
				mView.upload();
			}
			void downloadAsync(cudaStream_t stream) {
				mView.downloadAsync(stream);
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

			GpuView1 viewColumn(const GpuView2& v2, std::size_t col) {
				if (col >= v2.mView.extent(1))
					throw std::logic_error("column out of range");

				int rows = v2.mView.extent(0);

				return {
					 Cpu::Tensor::View1(v2.mCpu, rows)
					, v2.mGpu + col * rows
					, v2.mCpu + col * rows
				};
			}

			void vecScale( GpuView1& a1, float scale) {
				auto result = cublasSscal(mHandle, a1.mSize, &scale, a1.mGpu, 1);
				Error::checkBlas(result);
			}
			void vecAddVec(const GpuView1& a1, GpuView1& o1){
				float alpha = 1.0f;
				auto result = cublasSaxpy(mHandle, o1.mSize, &alpha, a1.mGpu, 1, o1.mGpu, 1);
				Error::checkBlas(result);
			}
			void matMulVec(const GpuView2& w2, const GpuView1& i1, GpuView1& o1) {
				//cuda is in C++ style of row-major
				//cublas wants Fortran style, col-major,
				//mdspan has been configured to be layout_left - cublas correct

				float alpha = 1.0f;
				float beta = 0.0f;

				std::size_t r = w2.mView.extent(0)
					, c = w2.mView.extent(1);

				Error::checkMissMatch(c, i1.mSize);

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
			void matMulVecBatch(const GpuView2& w2, const GpuView2& i2, GpuView2& o2) {

				int c = w2.mView.extent(1);
				int k = i2.mView.extent(0);

				Error::checkMissMatch(c, k);

				std::size_t batchSize = i2.mView.extent(1);

				for (auto b : std::views::iota(0ULL, batchSize)) {

					GpuView1 i1 = viewColumn(i2, b)
						, o1 = viewColumn(o2, b);

					matMulVec(w2, i1, o1);
				}
			}

			void matTMulVec(const GpuView2& w2, const GpuView1& i1, GpuView1& o1){
				
				float alpha = 1.0f;
				float beta = 0.0f;

				std::size_t r = w2.mView.extent(0)
					, c = w2.mView.extent(1);

				Error::checkMissMatch(r, i1.mSize);

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
			void updateWeights(const GpuView1& seen, GpuView2& weights, const GpuView1& primes, float learnRate);
			void copy(const GpuView1& source, GpuView1& dest) {
				auto result = cublasScopy(mHandle, source.mSize, source.mGpu, 1, dest.mGpu, 1);
				Error::checkBlas(result);
			}
			void activationFunction(LayerTemplate::ActivationFunction af, const GpuView1& o1, GpuView1& a1) {
				
				using ActivationFunction = LayerTemplate::ActivationFunction;
				switch( af) {
				case ActivationFunction::ReLU:
					relu(o1, a1);
					break;
				case ActivationFunction::SoftmaxCrossEntropy:
					softmax(o1, a1);
					break;
				case ActivationFunction::None:
					copy(o1, a1);
					break;
				}
			}
			void activationFunctionPrime(LayerTemplate::ActivationFunction af, const GpuView1& a1, GpuView1& p1) {

				using ActivationFunction = LayerTemplate::ActivationFunction;
				switch (af) {
				case ActivationFunction::ReLU:
					applyReluPrime(a1, p1);
					break;
				case ActivationFunction::None:
					break;
				}
			}
			void errorFunction(LayerTemplate::ActivationFunction af, const GpuView1& desired, const GpuView1& sought, GpuView1& p1) {
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

				Environment gpu;
				gpu.create();

				Gpu::FloatSpace1 fs1;
				Gpu::GpuView1 b1;
				Gpu::GpuView2 o2, a2, i2;
				Gpu::GpuView2 w2;

				std::size_t inputSize = 3
					, biasSize = 2
					, batchSize = 3;

				fs1.create((inputSize + biasSize*2)*batchSize 
					+ biasSize 
					+ biasSize * inputSize);

				auto begin = fs1.begin();

				fs1.advance(i2, begin, inputSize, batchSize);
				fs1.advance(w2, begin, biasSize, inputSize);
				fs1.advance(b1, begin, biasSize);
				fs1.advance(o2, begin, biasSize, batchSize);
				fs1.advance(a2, begin, biasSize, batchSize);

				std::fill(w2.begin(), w2.end(), 1);
				std::fill(i2.begin(), i2.end(), 1);
				std::fill(b1.begin(), b1.end(), 1);

				i2.mView[0, 1] = 0;

				fs1.upload();

				gpu.sync();

				auto forward = [&]() {

					gpu.matMulVecBatch(w2, i2, o2);
					//gpu.matMulVec(w, i, o);
					//gpu.vecAddVec(b, o);
				//	gpu.softmax(o, a);
					};
				forward();

				fs1.downloadAsync(gpu);

				gpu.sync();

				for (const auto& f : o2)
					std::print("{} ", f);

				fs1.destroy();

				gpu.destroy();
			}

		private:
			cublasHandle_t mHandle;
			cudaStream_t mStream;
		};
	}
}