#pragma once

#include <vector>
#include <mdspan>

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

	template<typename T>
	concept ViewConcept = std::is_same_v<T, View1>
		|| std::is_same_v<T, View2>
		|| std::is_same_v<T, View3>;

	template<typename... Dimensions>
	std::size_t area(Dimensions&& ...dimensions) {
		return (... * dimensions);
	}

	template<ViewConcept ViewT, typename... Dimensions>
	void advance(ViewT& view, Floats::iterator& begin, Dimensions&& ...dimensions) {

		view = ViewT(&*begin, std::array{ dimensions... });
		std::advance(begin, area(dimensions...));
	}

	template<ViewConcept ViewT>
	std::vector<Dimension> getShape(ViewT& view) {

		std::vector<Dimension> result;
		for (auto i : std::views::iota(0, ViewT::rank()) )
			result[i] = view.extents().extent(i);
		
		return result;
	}

	template<ViewConcept ViewType>
	class FloatSpace {
	public:
		Floats mFloats;
		ViewType mView;

		template<typename... Dimensions>
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