#pragma once

#include <vector>
#include <mdspan>

namespace NetworkLib {

	using FloatType = float;
	using Floats = std::vector<FloatType>;
	using Extents1 = std::extents<size_t, std::dynamic_extent>;
	using Extents2 = std::extents<size_t, std::dynamic_extent, std::dynamic_extent>;
	using Extents3 = std::extents<size_t, std::dynamic_extent, std::dynamic_extent, std::dynamic_extent>;
	using View1 = std::mdspan<FloatType, Extents1>;
	using View2 = std::mdspan<FloatType, Extents2>;
	using View3 = std::mdspan<FloatType, Extents3>;

	using ColView = std::span<float>;

	template<typename T>
	concept View = std::is_same_v<T, View1> || std::is_same_v<T, View2> || std::is_same_v<T, View3>;

	template<View ViewT>
	class TensorView {
	public:

		ViewT mView;

		template<typename... Dimensions>
		void create(FloatType* floats, Dimensions&& ...dimensions) {
			mView = ViewT(floats, dimensions...);
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
	
	using TensorView1 = TensorView<View1>;
	using TensorView2 = TensorView<View2>;
	using TensorView3 = TensorView<View3>;

	template<typename ViewT>
	class TensorNew : public ViewT {
	public:
		Floats mFloats;

		template<typename... Dimensions>
		void create(Dimensions&& ...dimensions) {

			std::size_t size = ( ... * dimensions);
			mFloats.resize(size);
			ViewT::create(mFloats.data(), dimensions...);
		}
	};

	using Tensor1 = TensorNew<TensorView1>;
	using Tensor2 = TensorNew<TensorView2>;
	using Tensor3 = TensorNew<TensorView3>;

	class CpuNetwork {
	

	public:

	};

}