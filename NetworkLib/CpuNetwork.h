#pragma once

#include <vector>
#include <mdspan>

namespace NetworkLib {

	using Floats = std::vector<float>;
	using Extents1 = std::extents<size_t, std::dynamic_extent>;
	using Extents2 = std::extents<size_t, std::dynamic_extent, std::dynamic_extent>;
	using Extents3 = std::extents<size_t, std::dynamic_extent, std::dynamic_extent, std::dynamic_extent>;
	using View1 = std::mdspan<float, Extents1>;
	using View2 = std::mdspan<float, Extents2>;
	using View3 = std::mdspan<float, Extents3>;

	template<typename T>
	concept View = std::is_same_v<T, View1> || std::is_same_v<T, View2> || std::is_same_v<T, View3>;

	template<View ViewT>
	class TensorNew {
	public:

		Floats mFloats;
		ViewT mView;

		template<typename... Dimensions>
		void create(Dimensions&& ...dimensions) {

			std::size_t size = 1;
			(size *= ... *= dimensions);
			mFloats.resize(size);

			mView =  ViewT(mFloats.data(), dimensions...);
		}

		template<typename... Dimensions>
		float& at(Dimensions&& ...dimensions) {
			return mView[std::array{ dimensions... }];
		}
	};

	using Tensor1 = TensorNew<View1>;
	using Tensor2 = TensorNew<View2>;
	using Tensor3 = TensorNew<View3>;

	class CpuNetwork {
	

	public:

	};

}