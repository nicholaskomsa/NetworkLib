#pragma once

#include "CpuTensor.h"
#include "NetworkTemplate.h"

namespace NetworkLib {

	namespace Gpu {

		using Dimension = Cpu::Tensor::Dimension;
		using Coordinate = Cpu::Tensor::Coordinate;


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
			GpuView<Cpu::Tensor::View2> upDimension() {
				return {
					Cpu::Tensor::upDimension(mView)
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
			GpuView<Cpu::Tensor::View2> viewDepth(Coordinate depth) const {

				Error::checkBounds(depth, mView.extent(2));

				auto cpuView = Cpu::Tensor::viewDepth(mView, depth);

				auto rows = mView.extent(0);
				auto cols = mView.extent(1);

				std::size_t offset = cols * rows * depth;
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
		using GpuView3 = GpuView<Cpu::Tensor::View3>;

		struct Float {
			float* mGpu = nullptr, * mCpu = nullptr;

			void upload();
			void downloadAsync(cudaStream_t stream) const;

			operator float() const;
			Float& operator=(float v);
		};
		struct Int {
			int* mGpu = nullptr, *mCpu = nullptr;

			void upload();
			void downloadAsync(cudaStream_t stream) const;

			operator int() const;
			Int& operator=(int v);
		};
		struct FloatSpace1 {
			
			GpuView1 mView;

			void create(const Cpu::FloatSpace1& cpuSpace);
			void resize(const Cpu::FloatSpace1& cpuSpace);

			void destroy();

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
			void advance(Int& i, float*& begin);

			void upload();
			void downloadAsync(cudaStream_t stream);
		};

		class LinkedFloatSpace {
		public:
			Cpu::FloatSpace1 mCpuSpace;
			Gpu::FloatSpace1 mGpuSpace;
			
			void create(std::size_t size) {
				mCpuSpace.create(size);
				mGpuSpace.create(mCpuSpace);
			}
			void destroy() {
				mGpuSpace.destroy();
				mCpuSpace.destroy();
			}
		};
	}
}