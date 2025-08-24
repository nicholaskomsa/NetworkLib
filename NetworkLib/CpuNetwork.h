#pragma once

#include <vector>
#include <mdspan>
#include <concepts>
#include <cstddef>
#include <type_traits>

namespace NetworkLib {

	using FloatType = float;
	using Floats = std::vector<FloatType>;
	using Dimension = std::size_t;
	using Coordinate = std::size_t;
	using Extents1 = std::extents<Dimension, std::dynamic_extent>;
	using Extents2 = std::extents<Dimension, std::dynamic_extent, std::dynamic_extent>;
	using Extents3 = std::extents<Dimension, std::dynamic_extent, std::dynamic_extent, std::dynamic_extent>;

	using View1 = std::mdspan<FloatType, Extents1>;
	using View2 = std::mdspan<FloatType, Extents2>;
	using View3 = std::mdspan<FloatType, Extents3>;
	
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
		for (size_t i = 0; i < view.rank(); ++i) 
			result *= view.extent(i);
		
		return result;
	}

	template<DynamicViewConcept ViewType, DimensionsConcept... Dimensions>
	void dynamicAdvance(ViewType& view, Floats::iterator& begin, Dimensions ...dimensions) {

		view = ViewType(&*begin, std::array{ dimensions... });
		std::advance(begin, area(dimensions...));
	}
	template<FixedViewConcept ViewType>
	void fixedAdvance(ViewType& view, Floats::iterator& begin) {

		view = ViewType(&*begin);
		std::advance(begin, area(view));
	}

	template<ViewConcept ViewType>
	std::vector<Dimension> getShape(const ViewType& view) {

		auto rank = ViewType::rank();
		std::vector<Dimension> result(rank);
		for (auto i : std::views::iota(0ULL, rank ) )
			result[i] = view.extent(i);
		
		return result;
	}

	template<ViewConcept ViewType>
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

	class CpuNetwork {
	

	public:

	};

}