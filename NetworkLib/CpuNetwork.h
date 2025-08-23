#pragma once

#include <vector>
#include <mdspan>

namespace NetworkLib {

	using FloatType = float;
	using Floats = std::vector<FloatType>;
	using Extents1 = std::extents<size_t, std::dynamic_extent>;
	using Extents2 = std::extents<size_t, std::dynamic_extent, std::dynamic_extent>;
	using Extents3 = std::extents<size_t, std::dynamic_extent, std::dynamic_extent, std::dynamic_extent>;
	using View1DType = std::mdspan<FloatType, Extents1>;
	using View2DType = std::mdspan<FloatType, Extents2>;
	using View3DType = std::mdspan<FloatType, Extents3>;

	using ColView = std::span<float>;

	template<typename T>
	concept ViewConcept = std::is_same_v<T, View1DType>
		|| std::is_same_v<T, View2DType>
		|| std::is_same_v<T, View3DType>;

	template<ViewConcept ViewType>
	class View {
	public:

		ViewType mView;

		template<typename... Dimensions>
		void create(FloatType* floats, Dimensions&& ...dimensions) {
			mView = ViewType(floats, dimensions...);
		}
		template<typename... Dimensions>
		void advance(Floats::iterator& floats, Dimensions&& ...dimensions) {

			create(&*floats, dimensions...);

			std::size_t size = (... * dimensions);
			std::advance(floats, size);
		}

		template<typename... Dimensions>
		float& at(Dimensions&& ...dimensions) {
			
			return mView[std::array{ dimensions... }];
		}
	};
	
	using View1D = View<View1DType>;
	using View2D = View<View2DType>;
	using View3D = View<View3DType>;

	template<typename T>
	concept AnyViewConcept = std::is_same_v<T, View1D>
		|| std::is_same_v<T, View2D>
		|| std::is_same_v<T, View3D>;

	template<AnyViewConcept AnyView>
	class FloatSpace : public AnyView {
	public:
		Floats mFloats;

		template<typename... Dimensions>
		void resize(Dimensions&& ...dimensions) {

			std::size_t size = ( ... * dimensions);
			mFloats.resize(size);
			AnyView::create(mFloats.data(), dimensions...);
		}
	};

	using Tensor1 = FloatSpace<View1D>;
	using Tensor2 = FloatSpace<View2D>;
	using Tensor3 = FloatSpace<View3D>;

	class CpuNetwork {
	

	public:

	};

}