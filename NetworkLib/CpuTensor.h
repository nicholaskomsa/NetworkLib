#pragma once


#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#include <source_location>
#include <ranges>
#include <print>
#include <vector>
#include <mdspan>
#include <concepts>

namespace NetworkLib {
	
	namespace Cpu {

		namespace Tensor {

			using FloatType = float;
			using Floats = std::vector<FloatType>;
			using FloatsView = std::span<FloatType>;
			using Dimension = std::size_t;
			using Coordinate = std::size_t;
		}
	}

	struct Error : public std::system_error {

		Error(std::errc code, const std::string& message);
		static void checkCuda(cudaError_t result, const std::source_location& location = std::source_location::current());
		static void checkBlas(cublasStatus_t result, const std::source_location& location = std::source_location::current());
		static void checkMissMatch(Cpu::Tensor::Dimension a, Cpu::Tensor::Dimension b, const std::source_location& location = std::source_location::current());
		static void checkBounds(Cpu::Tensor::Coordinate a, Cpu::Tensor::Dimension b, const std::source_location& location = std::source_location::current());
	};

	namespace Cpu {

		namespace Tensor {

			template<typename... Dimensions>
			using View = std::mdspan<FloatType, std::extents<Dimension, Dimensions::value...>, std::layout_left>;

			using Dynamic = std::integral_constant<size_t, std::dynamic_extent>;
			template<std::size_t c>
			using DimensionConstant = std::integral_constant<std::size_t, c>;

			using View1 = View<Dynamic>;
			using View2 = View<Dynamic, Dynamic>;
			using View3 = View<Dynamic, Dynamic, Dynamic>;

			//match mdspans
			template<typename T>
			concept ViewConcept = requires {
				typename T::element_type;
				typename T::extents_type;
				typename T::layout_type;
				typename T::mapping_type;
				typename T::accessor_type;
			};

			template<typename T>
			concept DynamicViewConcept = ViewConcept<T> &&
				(std::remove_cvref_t<T>::extents_type::rank_dynamic() > 0);

			template<typename T>
			concept FixedViewConcept = ViewConcept<T> &&
				(std::remove_cvref_t<T>::extents_type::rank_dynamic() == 0);

			template<typename T>
			concept DimensionsConcept = std::convertible_to<std::remove_reference_t<T>, Dimension>;

			template<typename T>
			concept CoordinatesConcept = std::convertible_to<std::remove_reference_t<T>, Coordinate>;

			template<DimensionsConcept... Dimensions>
			std::size_t area(Dimensions&& ...dimensions) {
				return (... * dimensions);
			}

			template<ViewConcept ViewType>
			size_t area(const ViewType& view) {
				size_t result = 1;
				auto rank = ViewType::rank();
				for (auto i : std::views::iota(0ULL, rank))
					result *= view.extent(i);

				return result;
			}

			template<DynamicViewConcept ViewType, typename... Dimensions>
			void advance(ViewType& view, float*& begin, Dimensions ...dimensions) {

				view = ViewType(&*begin, std::array{ dimensions... });
				begin += area(view);
			}
			template<FixedViewConcept ViewType>
			void advance(ViewType& view, float*& begin) {

				view = ViewType(&*begin);
				begin += area(view);
			}

			static void advance(float*& f, float*& begin) {
				f = begin++;
			}

			template<ViewConcept ViewType>
			std::vector<Dimension> getShape(const ViewType& view) {

				auto rank = ViewType::rank();
				std::vector<Dimension> result(rank);
				for (auto i : std::views::iota(0ULL, rank))
					result[i] = view.extent(i);

				return result;
			}

			static View1 viewColumn(const View2& v2, Coordinate col) {

				auto rows = v2.extent(0);
				auto offset = rows * col;
				auto cpu = v2.data_handle() + offset;

				return View1(cpu, std::array{ rows });
			}
			static View2 viewDepth(const View3 v2, Coordinate depth) {

				auto rows = v2.extent(0);
				auto cols = v2.extent(1);

				auto offset = rows * cols * depth;
				auto cpu = v2.data_handle() + offset;

				return View2(cpu, std::array{ rows, cols });
			}

			template<ViewConcept ViewType>
			View1 flatten(ViewType& v) {
				return View1(v.data_handle(), std::array{ area(v) });
			}

			template<ViewConcept ViewType>
			FloatsView view(ViewType& v) {
				return FloatsView(v.data_handle(), area(v));
			}
		}

		using View1 = Tensor::View1;
		using View2 = Tensor::View2;
		using View3 = Tensor::View3;

		class FloatSpace1 {
		public:
			float* mCpu = nullptr;
			Tensor::View1 mView;

			void create(std::size_t size) {

				Error::checkCuda(cudaMallocHost(&mCpu, size * sizeof(float)));

				mView = Cpu::Tensor::View1(mCpu, size);
			}

			void destroy() {
				if (mCpu)
					Error::checkCuda(cudaFreeHost(mCpu));
				mCpu = nullptr;
			}

			float* begin() {
				return mCpu;
			}
			float* end() {
				return mCpu + mView.extent(0);
			}
		};
			
		static void example() {

			FloatSpace1 floatSpace;
			constexpr Tensor::Dimension a = 4, b = 5;
			Tensor::Dimension c = 6;
			floatSpace.create(a + b * c + a * b + a * b * c);

			auto begin = floatSpace.begin();
			Tensor::View1 v1;
			Tensor::View2 v2;

			using width = Tensor::DimensionConstant<a>;
			using height = Tensor::DimensionConstant<b>;
			Tensor::View<width, height> fv2(nullptr);
			Tensor::View<width, Tensor::Dynamic, Tensor::Dynamic> fv3;

			//Tensor::advance(v1, begin, a);
			//Tensor::advance(v2, begin, b, c);
			//Tensor::advance(fv2, begin);
			//Tensor::advance(fv3, begin, b, c);

			//v2[b - 1, c - 1] = 6.5f;
			//floatSpace.mView[2] = 7;
			//fv2[0, 2] = 9;
			//fv3[3, 3, 3] = 4.3;

			//std::println("{} {} {} {}", v2[b - 1, c - 1], v1[2], floatSpace.mView[30], fv2[0, 2]);

			auto v2Shape = Tensor::getShape(v2);
			for (auto d : v2Shape)
				std::print("{} ", d);
		}
	}
}