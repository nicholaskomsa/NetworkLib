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

		using Dimension = Cpu::Tensor::Dimension;
		using Coordinate = Cpu::Tensor::Coordinate;

		struct Error : public std::system_error {

			Error(std::errc code, const std::string& message)
				: std::system_error(int(code), std::generic_category(), message) {}

			static void cudaError(cudaError_t result, const std::source_location& location = std::source_location::current()) {

				auto message = std::format(
					"Cuda Error ={}:\n{}"
					"\n{}\n{}\n{}\n"
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
					"BLAS Error ={}:\n{}"
					"\n{}\n{}\n{}\n"
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
					blasError(result, location);
			}

			static void missMatchError(Dimension a, Dimension b, const std::source_location& location = std::source_location::current()) {
				auto message = std::format("{}x{} mismatch\n{}\n{}\n{}\n", a, b, location.file_name(), location.line(), location.function_name());
				throw Error(std::errc::invalid_argument, message);
			}
			static void checkMissMatch(Dimension a, Dimension b, const std::source_location& location = std::source_location::current()) {
				if (a != b)
					missMatchError(a, b, location);
			}
			static void boundsError(Coordinate a, Dimension b, const std::source_location& location = std::source_location::current()) {
				auto message = std::format("{}, {} out of bounds\n{}\n{}\n{}\n", a, b, location.file_name(), location.line(), location.function_name());
				throw Error(std::errc::invalid_argument, message);
			}
			static void checkBounds(Coordinate a, Dimension b, const std::source_location& location = std::source_location::current()) {
				if (a >= b)
					boundsError(a, b, location);
			}

		
		};

		template<typename T>
		concept ViewConcept = Cpu::Tensor::ViewConcept<T>;

		template<ViewConcept ViewType>
		struct GpuView {
			ViewType mView;
			float* mGpu = nullptr, * mCpu = nullptr;
			Dimension mSize = 0;

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

			GpuView<Cpu::Tensor::View1> flatten() const {
				return {
					Cpu::Tensor::flatten(mView)
					, mGpu
					, mCpu
				};
			}

			GpuView<Cpu::Tensor::View1> viewColumn( Coordinate col) const{

				Error::checkBounds(col, mView.extent(1));

				auto cpuView = Cpu::Tensor::viewColumn(mView, col);

				int rows = mView.extent(0);
				std::size_t offset = col * rows;
				auto gpu = mGpu + offset;

				return {
					 cpuView
					, gpu
					, cpuView.data_handle()
				};

			}

		};

		using GpuView1 = GpuView<Cpu::Tensor::View1>;
		using GpuView2 = GpuView<Cpu::Tensor::View2>;

		struct Float {
			float* mGpu = nullptr, * mCpu = nullptr;

			void upload() {
				Error::checkCuda(cudaMemcpy(
					mGpu,
					mCpu,
					1 * sizeof(float),
					cudaMemcpyHostToDevice));
			}
			void downloadAsync(cudaStream_t stream) const {
				Error::checkCuda(cudaMemcpyAsync(
					mCpu,
					mGpu,
					1 * sizeof(float),
					cudaMemcpyDeviceToHost,
					stream));
			}

			operator float() const {
				return *mCpu;
			}
			Float& operator=(float v) {
				*mCpu = v;
				return *this;
			}
		};

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

			float* getGpu(float* cpu) {
				return mView.mGpu + (cpu - mView.mCpu);
			}

			template<ViewConcept ViewType>
			float* getGpu(ViewType& view) {
				return getGpu(view.data_handle());
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
			void advance(Float& f, float*& begin) {
				auto source = begin;
				f = { getGpu(source), source };
				++begin;
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

				mFloatSpace1.create(1);
				auto begin = mFloatSpace1.begin();
				mFloatSpace1.advance(mMseResult, begin);
			}
			void destroy() {

				mFloatSpace1.destroy();

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

				Dimension r = w2.mView.extent(0)
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
			void batchedMatMulVec1(const GpuView2& w2, const GpuView2& i2, GpuView2& o2) {

				int c = w2.mView.extent(1);
				int k = i2.mView.extent(0);

				Error::checkMissMatch(c, k);

				Dimension batchSize = i2.mView.extent(1);

				for (auto b : std::views::iota(0ULL, batchSize)) {

					const GpuView1 i1 = i2.viewColumn(b);
					GpuView1 o1 = o2.viewColumn(b);

					matMulVec(w2, i1, o1);
				}
			}
			void batchedMatMulVec(const GpuView2& w2, const GpuView2& i2, GpuView2& o2) {
				
				float alpha = 1.0f;
				float beta = 0.0f;

				int r = w2.mView.extent(0);       // rows of matrix
				int c = w2.mView.extent(1);       // cols of matrix
				int batchSize = i2.mView.extent(1);

				Error::checkMissMatch(c, i2.mView.extent(0));
				Error::checkMissMatch(batchSize, o2.mView.extent(1));

				// Leading dimensions
				int lda = r;  // w2: (r × c)
				int ldb = c;  // i2: (c × 1)
				int ldc = r;  // o2: (r × 1)

				// Strides
				long long strideA = 0;                  // w2 is shared across batches
				long long strideB = static_cast<long long>(c);  // each vector is c × 1
				long long strideC = static_cast<long long>(r);  // each output is r × 1

				auto result = cublasSgemmStridedBatched(
					mHandle,
					CUBLAS_OP_N, CUBLAS_OP_N,
					r, 1, c,
					&alpha,
					w2.mGpu, lda, strideA,
					i2.mGpu, ldb, strideB,
					&beta,
					o2.mGpu, ldc, strideC,
					batchSize
				);

				Error::checkBlas(result);
			}
			void batchedMatTMulVec(const GpuView2& w2, const GpuView2& i2, GpuView2& o2) {
				float alpha = 1.0f;
				float beta = 0.0f;

				int r = w2.mView.extent(0);       // rows of w2
				int c = w2.mView.extent(1);       // cols of w2
				int batchSize = i2.mView.extent(1);

				Error::checkMissMatch(r, i2.mView.extent(0));
				Error::checkMissMatch(batchSize, o2.mView.extent(1));

				// Leading dimensions
				int lda = r;  // w2: (r × c), column-major
				int ldb = r;  // i2: (r × 1), column-major
				int ldc = c;  // o2: (c × 1), column-major

				// Strides
				long long strideA = 0;                  // shared w2
				long long strideB = static_cast<long long>(r);  // input vector stride
				long long strideC = static_cast<long long>(c);  // output vector stride

				auto result = cublasSgemmStridedBatched(
					mHandle,
					CUBLAS_OP_T, CUBLAS_OP_N,  // transpose w2, no transpose i2
					c, 1, r,                   // output dim: (c × 1), inner dim: r
					&alpha,
					w2.mGpu, lda, strideA,
					i2.mGpu, ldb, strideB,
					&beta,
					o2.mGpu, ldc, strideC,
					batchSize
				);

				Error::checkBlas(result);
			}

			void matTMulVec(const GpuView2& w2, const GpuView1& i1, GpuView1& o1){
				
				float alpha = 1.0f;
				float beta = 0.0f;

				Dimension r = w2.mView.extent(0)
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
	
			void mse(const GpuView2& sought, const GpuView2& desired);
			float getMseResult() {
				mMseResult.downloadAsync(mStream);
				sync();
				return mMseResult;
			}
			void resetMseResult() {
				mMseResult = 0.0f;
				mMseResult.upload();
			}
			void relu(const GpuView1& o1, GpuView1& a1);
			void applyReluPrime(const GpuView1& a1, GpuView1& p1);
			void softmax(const GpuView1& o1, GpuView1& a1);
			void batchedSoftmax(const GpuView2& o2, GpuView2& a2);
			void diff(const GpuView1& desired1, const GpuView1& sought1, GpuView1& primes1);
			void updateWeights(const GpuView1& seen, GpuView2& weights, const GpuView1& primes, float learnRate);
			void copy(const GpuView1& source, GpuView1& dest) {
				auto result = cublasScopy(mHandle, source.mSize, source.mGpu, 1, dest.mGpu, 1);
				Error::checkBlas(result);
			}
			void batchedCopy(const GpuView2& source, GpuView2& dest);
			void batchedBroadcast(const GpuView1& source, GpuView2& dest);
			void batchedBroadcastAdd(const GpuView1& source, GpuView2& dest);
			void batchedDiff(const GpuView2& desired2, const GpuView2& sought2, GpuView2& primes2);
			void batchedUpdateWeights(const GpuView2& seen, GpuView2& weights, const GpuView2& primes, float learnRate);

			void activationFunction(LayerTemplate::ActivationFunction af, const GpuView1& o1, GpuView1& a1) {
				
				using ActivationFunction = LayerTemplate::ActivationFunction;
				switch( af) {
				case ActivationFunction::ReLU:
					relu(o1, a1);
					break;
				case ActivationFunction::Softmax:
					softmax(o1, a1);
					break;
				case ActivationFunction::None:
					copy(o1, a1);
					break;
				}
			}
			void batchedActivationFunction(LayerTemplate::ActivationFunction af, const GpuView2& o2, GpuView2& a2) {

				using ActivationFunction = LayerTemplate::ActivationFunction;
				switch (af) {
				case ActivationFunction::ReLU: {
					auto o1 = o2.flatten();
					auto a1 = a2.flatten();
					relu(o1, a1);
					break;
				}
				case ActivationFunction::Softmax:
					batchedSoftmax(o2, a2);
					break;
				case ActivationFunction::None:
					batchedCopy(o2, a2);
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
			void batchedActivationFunctionPrime(LayerTemplate::ActivationFunction af, const GpuView2& a2, GpuView2& p2) {

				using ActivationFunction = LayerTemplate::ActivationFunction;
				switch (af) {
				case ActivationFunction::ReLU: {
					auto p1 = p2.flatten();
					auto a1 = a2.flatten();
					applyReluPrime(a1, p1);
					break;
				}
				case ActivationFunction::None:
					break;
				}
			}
			
			void errorFunction(LayerTemplate::ActivationFunction af, const GpuView1& desired, const GpuView1& sought, GpuView1& p1) {
				switch (af) {
				case LayerTemplate::ActivationFunction::Softmax:
					//softmax-cross-entropy is a diff
					[[fallthrouh]];
				case LayerTemplate::ActivationFunction::None:
					[[fallthrouh]];
				default:
					diff(desired, sought, p1);
					return;
				}
			}
			void batchedErrorFunction(LayerTemplate::ActivationFunction af, const GpuView2& desired2, const GpuView2& sought2, GpuView2& p2) {
				switch (af) {
				case LayerTemplate::ActivationFunction::Softmax:
					//softmax-cross-entropy is a diff
					[[fallthrouh]];
				case LayerTemplate::ActivationFunction::None:
					[[fallthrouh]];
				default:
					batchedDiff(desired2, sought2, p2);
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
				Gpu::GpuView2 o2, a2, i2, d2, p2;
				Gpu::GpuView2 w2;
				Gpu::GpuView2 activations,softmax;

				std::size_t inputSize = 3
					, biasSize = 2
					, batchSize = 1;

				fs1.create((inputSize + biasSize*4)*batchSize 
					+ biasSize 
					+ biasSize * inputSize + biasSize*2*batchSize);

				auto begin = fs1.begin();

				fs1.advance(i2, begin, inputSize, batchSize);
				fs1.advance(w2, begin, biasSize, inputSize);
				fs1.advance(b1, begin, biasSize);
				fs1.advance(o2, begin, biasSize, batchSize);
				fs1.advance(a2, begin, biasSize, batchSize);
				fs1.advance(p2, begin, biasSize, batchSize);
				fs1.advance(d2, begin, biasSize, batchSize);
				fs1.advance(softmax, begin, biasSize, batchSize);
				fs1.advance(activations, begin, biasSize, batchSize);

				std::fill(w2.begin(), w2.end(), 1);
				std::fill(i2.begin(), i2.end(), 1);
				std::fill(b1.begin(), b1.end(), 1);
				std::fill(d2.begin(), d2.end(), 0.5);

				i2.mView[0, 1] = 0;

				d2.mView[0, 0] = .314;
				d2.mView[0, 1] = 1;
				d2.mView[0, 2] = 0;
			
				activations.mView[0, 0] = 0.3;
				activations.mView[1, 0] = -0.8;

				activations.mView[0, 1] = 0.9;
				activations.mView[1, 1] = 0.1;

				fs1.upload();

				gpu.sync();

				for (auto generation : std::views::iota(0, 5000)) {

					auto af = LayerTemplate::ActivationFunction::Softmax;

					auto forward = [&]() {

						gpu.batchedMatMulVec(w2, i2, o2);
						gpu.batchedBroadcastAdd(b1, o2);
						gpu.batchedActivationFunction(af, o2, a2);

						};
					forward();

					auto backward = [&]() {

						gpu.batchedErrorFunction(af, d2, a2, p2);

						gpu.batchedUpdateWeights(i2, w2, p2, 0.002);
						};
					backward();
				}
				gpu.batchedActivationFunction(LayerTemplate::ActivationFunction::Softmax, activations, softmax);

				fs1.downloadAsync(gpu);

				gpu.sync();

				for (const auto& f : softmax)
					std::print("{} ", f);
				std::println("");
				fs1.destroy();

				gpu.destroy();
			}

		private:
			cublasHandle_t mHandle;
			cudaStream_t mStream;

			FloatSpace1 mFloatSpace1;
			Float mMseResult;
		};
	}
}