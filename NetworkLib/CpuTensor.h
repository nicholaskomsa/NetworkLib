#pragma once

#include <vector>
#include <mdspan>
#include <concepts>
#include <cstddef>
#include <type_traits>

namespace NetworkLib {
	
	namespace Cpu {

		namespace Tensor {

			using FloatType = float;
			using Floats = std::vector<FloatType>;
			using Dimension = std::size_t;
			using Coordinate = std::size_t;

			template<typename... Dimensions>
			using View = std::mdspan<FloatType, std::extents<Dimension, Dimensions::value...>>;

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
			void advance(ViewType& view, Floats::iterator& begin, Dimensions ...dimensions) {

				view = ViewType(&*begin, std::array{ dimensions... });
				std::advance(begin, area(view));
			}
			template<FixedViewConcept ViewType>
			void advance(ViewType& view, Floats::iterator& begin) {

				view = ViewType(&*begin);
				std::advance(begin, area(view));
			}
			template<Cpu::Tensor::DynamicViewConcept ViewType, typename... Dimensions>
			void advance(ViewType& view, float*& begin, Dimensions ...dimensions) {

				view = ViewType(&*begin, std::array{ dimensions... });
				begin+= area(view);
			}
			template<Cpu::Tensor::FixedViewConcept ViewType>
			void advance(ViewType& view, float*& begin) {

				view = ViewType(&*begin);
				begin+= area(view);
			}

			template<ViewConcept ViewType>
			std::vector<Dimension> getShape(const ViewType& view) {

				auto rank = ViewType::rank();
				std::vector<Dimension> result(rank);
				for (auto i : std::views::iota(0ULL, rank))
					result[i] = view.extent(i);

				return result;
			}

			template<DynamicViewConcept ViewType>
			class FloatSpace {
			public:
				Floats mFloats;
				ViewType mView;

				template<DimensionsConcept... Dimensions>
				void resize(Dimensions&& ...dimensions) {
					mFloats.resize(area(dimensions...));
					mView = ViewType(mFloats.data(), std::array{ dimensions... });
				}
			};

			using FloatSpace1 = FloatSpace<View1>;
			using FloatSpace2 = FloatSpace<View2>;
			using FloatSpace3 = FloatSpace<View3>;

			void example() {

				//namespace CpuTensor = NetworkLib::Cpu::Tensor;

				FloatSpace1 floatSpace1;
				constexpr Dimension a = 4, b = 5;
				Dimension c = 6;
				floatSpace1.resize(a + b * c + a * b + a * b * c);

				auto begin = floatSpace1.mFloats.begin();
				View1 v1;
				View2 v2;

				using width = DimensionConstant<a>;
				using height = DimensionConstant<b>;
				View<width, height> fv2(nullptr);
				View<width, Dynamic, Dynamic> fv3;

				advance(v1, begin, a);
				advance(v2, begin, b, c);
				advance(fv2, begin);
				advance(fv3, begin, b, c);

				v2[b - 1, c - 1] = 6.5f;
				floatSpace1.mView[2] = 7;
				fv2[0, 2] = 9;
				fv3[3, 3, 3] = 4.3;

				std::println("{} {} {} {}", v2[b - 1, c - 1], v1[2], floatSpace1.mView[30], fv2[0, 2]);

				auto v2Shape = getShape(v2);
				for (auto d : v2Shape)
					std::print("{} ", d);
			}
		}
	}
}