#pragma once

#include <vector>
#include <span>

namespace NetworkLib {

	struct Tensor {

		//A tensor is a float object of one to four dimensions
		//where the first dimension is X, and can be spanned over width
		//the second dimension is Y, and can now be spanned over width or height, with the addition of spanT
		//the third dimension is Z
		//the fourth dimension is W
		//TensorView is the multidimensional float data and is broken up into spannable segments depending on dimensionality.
		using Floats = std::vector<float>;
		using View = std::span<float>;
		using ConstView = std::span<const float>;

		View mTensor;

		std::size_t mX{ 0 }, mY{ 0 }, mZ{ 0 }, mW{ 0 };

		Tensor() = default;
		Tensor(View floats, std::size_t x, std::size_t y = 0, std::size_t z = 0, std::size_t w = 0);
		Tensor(Floats::iterator& begin, std::size_t x, std::size_t y = 0, std::size_t z = 0, std::size_t w = 0);

		std::size_t size() const;
		std::size_t size2D() const;
		std::size_t size3D() const;
		std::size_t size4D() const;

		static View field(View view, std::size_t offset, std::size_t size);
		static ConstView constField(ConstView view, std::size_t offset, std::size_t size);

		View viewBlock();
		View viewBlock(std::size_t col);
		ConstView constViewBlock() const;
		ConstView constViewBlock(std::size_t col) const;

		float& at(std::size_t col);
		const float& cat(std::size_t col) const;

		float& at(std::size_t row, std::size_t col);
		const float& cat(std::size_t row, std::size_t col) const;

		float& at(std::size_t depth, std::size_t row, std::size_t col);
		const float& cat(std::size_t depth, std::size_t row, std::size_t col) const;
		float& at(std::size_t w, std::size_t z, std::size_t y, std::size_t x);

		View view();
		ConstView constView() const;

		View view(size_t col);
		ConstView constView(size_t col) const;

		View view(std::size_t depth, std::size_t col);
		ConstView constView(std::size_t depth, size_t col) const;
	};

}