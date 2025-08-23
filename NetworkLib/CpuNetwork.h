#pragma once

#include <vector>
#include <mdspan>

namespace NetworkLib {

	using FloatType = float;
	using Floats = std::vector<FloatType>;
	using Extents1 = std::extents<size_t, std::dynamic_extent>;
	using Extents2 = std::extents<size_t, std::dynamic_extent, std::dynamic_extent>;
	using Extents3 = std::extents<size_t, std::dynamic_extent, std::dynamic_extent, std::dynamic_extent>;
	using View1D = std::mdspan<FloatType, Extents1>;
	using View2D = std::mdspan<FloatType, Extents2>;
	using View3D = std::mdspan<FloatType, Extents3>;

	using ColView = std::span<FloatType>;

	template<typename... Dimensions>
	std::size_t area(Dimensions&& ...dimensions) {
		return (... * dimensions);
	}

	template<typename T>
	concept ViewConcept = std::is_same_v<T, View1D>
		|| std::is_same_v<T, View2D>
		|| std::is_same_v<T, View3D>;

	template<ViewConcept ViewType, typename... Dimensions>
	float& at(ViewType& view, Dimensions&& ...dimensions) {
		return view[std::array{ dimensions... }];
	}
	template<ViewConcept ViewType, typename... Dimensions>
	void advance(ViewType& view, Floats::iterator& begin, Dimensions&& ...dimensions ) {
		
		view = ViewType( &*begin, dimensions... );
		
		std::advance(begin, area( dimensions... ) );
	}


	template<ViewConcept ViewType>
	class FloatSpace {
	public:
		Floats mFloats;
		ViewType mView;

		template<typename... Dimensions>
		void resize(Dimensions&& ...dimensions) {
			mFloats.resize(area(dimensions...));
			mView = ViewType(mFloats.data(), dimensions...);
		}
	};

	using FloatSpace1 = FloatSpace<View1D>;
	using FloatSpace2 = FloatSpace<View2D>;
	using FloatSpace3 = FloatSpace<View3D>;

	class CpuNetwork {
	

	public:

	};

}