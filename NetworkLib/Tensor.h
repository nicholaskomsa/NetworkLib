#pragma once

#include <span>

struct Tensor {

	//A tensor is a float object of one to four dimensions
	//where the first dimension is X, and can be spanned over width
	//the second dimension is Y, and can now be spanned over width or height, with the addition of spanT
	//the third dimension is Z
	//the fourth dimension is W
	//TensorView is the multidimensional float data and is broken up into spannable segments depending on dimensionality.

	using TensorView = std::span<float>;
	TensorView mTensor;

	std::size_t mX{ 0 }, mY{ 0 }, mZ{ 0 }, mW{ 0 };

	Tensor() = default;
	Tensor(TensorView floats, std::size_t x, std::size_t y = 0, std::size_t z = 0, std::size_t w = 0) : mTensor(floats), mX(x), mY(y), mZ(z), mW(w) {}


	std::size_t size() const {
		return mX;
	}
	std::size_t size2D() const {
		return mX * mY;
	}
	std::size_t size3D() const {
		return mX * mY * mZ;
	}
	std::size_t size4D() const {
		return mX * mY * mZ * mW;
	}

	TensorView spanTEnd(std::size_t col) const {

		std::size_t offset = col * mY + mY;
		auto begin = mTensor.begin();
		auto end = std::next(begin, offset);
		return { begin, end };
	}

	float& at(std::size_t col) const {
		return mTensor[col];
	}

	float& at(std::size_t row, std::size_t col) const {
		return  mTensor[row * mX + col];
	}
	float& atT(std::size_t row, std::size_t col) const {
		return  mTensor[col * mY + row];
	}

	float& at(std::size_t depth, std::size_t row, std::size_t col) const {
		return  mTensor[depth * (mY * mX) + row * mX + col];
	}
	float& atT(std::size_t depth, std::size_t row, std::size_t col) const {
		return  mTensor[depth * (mY * mX) + col * mY + row];
	}

	float& at(std::size_t w, std::size_t z, std::size_t y, std::size_t x) const {
		return  mTensor[w * (mZ * mY * mX) + z * (mY * mX) + y * mX + x];
	}

	TensorView span() const {
		return { &at(0), mX };
	}
	TensorView spanT() const {
		return { &at(0), mY };
	}

	TensorView span(std::size_t row) const {
		return { &at(row,0), mX };
	}
	TensorView spanT(size_t col) const {
		return { &atT(0, col), mY };
	}

	TensorView span(std::size_t depth, std::size_t row) const {
		return { &at(depth, row, 0), mX };
	}
	TensorView spanT(std::size_t depth, std::size_t col) const {
		return { &at(depth, 0, col), mY };
	}

	TensorView span(std::size_t w, std::size_t z, std::size_t y) const {
		return { &at(w, z, y, 0), mX };
	}


};