#pragma once

#include <span>

struct Tensor {

	//A tensor is a float object of one to four dimensions
	//where the first dimension is X, and can be spanned over width
	//the second dimension is Y, and can now be spanned over width or height, with the addition of spanT
	//the third dimension is Z
	//the fourth dimension is W
	//TensorView is the multidimensional float data and is broken up into spannable segments depending on dimensionality.

	using View = std::span<float>;
	using ConstView = std::span<const float>;

	std::vector<float> mFloats;

	View mTensor;

	std::size_t mX{ 0 }, mY{ 0 }, mZ{ 0 }, mW{ 0 };

	Tensor() = default;
	Tensor(View floats, std::size_t x, std::size_t y = 0, std::size_t z = 0, std::size_t w = 0) : mTensor(floats), mX(x), mY(y), mZ(z), mW(w) {}
	Tensor(std::size_t x, std::size_t y = 0, std::size_t z = 0, std::size_t w = 0) : mX(x), mY(y), mZ(z), mW(w) {
	
	
	}

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

	View viewTBlock(std::size_t col) {

		std::size_t offset = col * mY + mY;
		auto begin = mTensor.begin();
		auto end = std::next(begin, offset);
		return { begin, end };
	}

	float& at(std::size_t col) {
		return mTensor[col];
	}
	const float& cat(std::size_t col) const {
		return mTensor[col];
	}

	float& at(std::size_t row, std::size_t col) {
		return  mTensor[row * mX + col];
	}
	float& atT(std::size_t row, std::size_t col) {
		return  mTensor[col * mY + row];
	}

	const float& catT(std::size_t row, std::size_t col) const {
		return mTensor[col * mY + row];
	}

	float& at(std::size_t depth, std::size_t row, std::size_t col) {
		return  mTensor[depth * (mY * mX) + row * mX + col];
	}
	float& atT(std::size_t depth, std::size_t row, std::size_t col) {
		return  mTensor[depth * (mY * mX) + col * mY + row];
	}

	float& at(std::size_t w, std::size_t z, std::size_t y, std::size_t x) {
		return  mTensor[w * (mZ * mY * mX) + z * (mY * mX) + y * mX + x];
	}

	View view() {
		return { &at(0), mX };
	}
	
	ConstView constView() const {
		return { &cat(0), mX };
	}

	View view(std::size_t row) {
		return { &at(row,0), mX };
	}
	View viewT(size_t col)  {
		return { &atT(0, col), mY };
	}

	ConstView constViewT(size_t col) const {
		return { &catT(0, col), mY };
	}

	View view(std::size_t depth, std::size_t row) {
		return { &at(depth, row, 0), mX };
	}
	View viewT(std::size_t depth, std::size_t col) {
		return { &at(depth, 0, col), mY };
	}

	View view(std::size_t w, std::size_t z, std::size_t y) {
		return { &at(w, z, y, 0), mX };
	}


};