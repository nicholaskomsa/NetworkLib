#pragma once

#include "CpuTensor.h"
#include "NetworkTemplate.h"

namespace NetworkLib {

	namespace Gpu {

		using Dimension = Cpu::Tensor::Dimension;
		using Coordinate = Cpu::Tensor::Coordinate;

		template<Cpu::Tensor::ViewConcept ViewType>
		struct GpuView {
			ViewType mView;
			using ViewDataType = ViewType::element_type;

			ViewDataType* mGpu = nullptr, * mCpu = nullptr;
			Dimension mSize = 0;

			using CpuView1 = Cpu::Tensor::View<ViewDataType, Cpu::Tensor::Dynamic>;
			using CpuView2 = Cpu::Tensor::View<ViewDataType, Cpu::Tensor::Dynamic, Cpu::Tensor::Dynamic>;

			GpuView() = default;
		
			GpuView(ViewType view, ViewDataType* gpu, ViewDataType* cpu)
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
			ViewDataType* begin() {
				return mCpu;
			}
			ViewDataType* end() {
				return mCpu + mSize;
			}
			const ViewDataType* begin() const{
				return mCpu;
			}
			const ViewDataType* end() const {
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
			GpuView<CpuView1> field(std::size_t offset, std::size_t size) const {
				
				auto cpuView = Cpu::Tensor::field(mView, offset, size);

				auto gpu = mGpu + offset;

				return {
					cpuView
					, gpu
					, cpuView.data_handle()
				};
			}

			GpuView<CpuView1> viewColumn( Coordinate col) const {

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

			GpuView<CpuView2> viewDepth(Coordinate depth) const {

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

		using GpuIntView1 = GpuView<Cpu::Tensor::IntView1>;
		using GpuIntView2 = GpuView<Cpu::Tensor::IntView2>;
		using GpuIntView3 = GpuView<Cpu::Tensor::IntView3>;

		using GpuViews1 = std::vector<Gpu::GpuView1>;
		using GpuViews1View = std::span<Gpu::GpuView1>;
		 
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

			float* getGpu(float* cpu) {
				return mView.mGpu + (cpu - mView.mCpu);
			}

			template<Cpu::Tensor::ViewConcept ViewType>
			float* getGpu(ViewType& view) {
				return getGpu(reinterpret_cast<float*>(view.data_handle()));
			}

			float* begin();
			float* end();

			template<Cpu::Tensor::ViewConcept ViewType, typename ViewDataType = ViewType::element_type
				, Cpu::Tensor::IntOrFloatConcept BeginType, Cpu::Tensor::DimensionsConcept... Dimensions>
			void advance(GpuView<ViewType>& gpuView, BeginType*& begin, Dimensions&&...dimensions) {
				ViewType view;

				ViewDataType* source = reinterpret_cast<ViewDataType*>(begin);
				Cpu::Tensor::advance(view, begin, dimensions...);
				ViewDataType* gpu = reinterpret_cast<ViewDataType*>(getGpu(view));

				gpuView = { view, gpu , source };
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