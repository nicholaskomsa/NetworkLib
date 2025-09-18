#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <source_location>

#include "CpuTensor.h"
#include "NetworkTemplate.h"

namespace NetworkLib {

	namespace Gpu {

		using Dimension = Cpu::Tensor::Dimension;
		using Coordinate = Cpu::Tensor::Coordinate;

		struct Error : public std::system_error {

			Error(std::errc code, const std::string& message);
			static void checkCuda(cudaError_t result, const std::source_location& location = std::source_location::current());
			static void checkBlas(cublasStatus_t result, const std::source_location& location = std::source_location::current());
			static void checkMissMatch(Dimension a, Dimension b, const std::source_location& location = std::source_location::current());
			static void checkBounds(Coordinate a, Dimension b, const std::source_location& location = std::source_location::current());
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
				:mView(view), mGpu(gpu), mCpu(cpu) {

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

			void upload();
			void downloadAsync(cudaStream_t stream) const;

			operator float() const;
			Float& operator=(float v);
		};

		struct FloatSpace1 {

			GpuView1 mView;

			void create(std::size_t size);

			void destroy();

			void freeHost();
			void freeGpu();

			float* getGpu(float* cpu);

			template<ViewConcept ViewType>
			float* getGpu(ViewType& view) {
				return getGpu(view.data_handle());
			}

			float* begin();
			float* end();

			template<ViewConcept ViewType, typename... Dimensions>
			void advance(GpuView<ViewType>& gpuView, float*& begin, Dimensions&&...dimensions) {
				ViewType view;
				auto source = begin;
				Cpu::Tensor::advance(view, begin, dimensions...);
				gpuView = { view, getGpu(view), source };
			}
			void advance(Float& f, float*& begin);

			void upload();
			void downloadAsync(cudaStream_t stream);
		};

	}
}