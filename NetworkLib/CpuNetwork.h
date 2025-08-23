#pragma once

#include <vector>
#include <mdspan>

namespace NetworkLib {

	using FloatType = float;
	using Floats = std::vector<FloatType>;
	using Extents1 = std::extents<size_t, std::dynamic_extent>;
	using Extents2 = std::extents<size_t, std::dynamic_extent, std::dynamic_extent>;
	using Extents3 = std::extents<size_t, std::dynamic_extent, std::dynamic_extent, std::dynamic_extent>;
	
	template<typename... Dimensions>
	std::size_t area(Dimensions&& ...dimensions) {
		return (... * dimensions);
	}

	template<typename Extents>
	class View {
		using ViewType = std::mdspan<FloatType, Extents>;
		ViewType mView;

	public:

		View() = default;
		View(FloatType* data, Extents ext) : mView(data, ext) {}

		template<typename ...Coordinates>
		auto& operator[](Coordinates&& ...coordinates) const {
			return mView[coordinates...];
		}

		View& operator=(const View& other) = default;

		template<typename... Dimensions>
		void advance(Floats::iterator& begin, Dimensions&& ...dimensions) {

			mView = ViewType(&*begin, std::array{dimensions...});
			std::advance(begin, area(dimensions...));
		}

		void print_shape() const {
			std::cout << "Shape: " << ( mView.extents().extent(0)
				<< " x " << mView.extents().extent(1) << "\n";
		}
	};

	using View1 = View<Extents1>;
	using View2 = View<Extents2>;
	using View3 = View<Extents3>;

	template<typename T>
	concept ViewConcept = std::is_same_v<T, View1>
		|| std::is_same_v<T, View2>
		|| std::is_same_v<T, View3>;

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