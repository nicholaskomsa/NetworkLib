#pragma once

#include <span>

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
	Tensor(View floats, std::size_t x, std::size_t y = 0, std::size_t z = 0, std::size_t w = 0) 
		: mTensor(floats), mX(x), mY(y), mZ(z), mW(w) {}

	Tensor(Floats::iterator& begin, std::size_t x, std::size_t y = 0, std::size_t z =0, std::size_t w = 0)
		: mX(x), mY(y), mZ(z), mW(w) {

		auto size = x;
		if( y != 0 ) 
			size *= y;
		if( z != 0 )
			size *= z;
		if( w != 0 )
			size *= w;

		mTensor = { begin, size };
		std::advance(begin, size);
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

	static View field(View view, std::size_t offset, std::size_t size) {
		return { view.begin() + offset, size };
	}
	static ConstView constField(ConstView view, std::size_t offset, std::size_t size) {
		return { view.cbegin() + offset, size };
	}

	View viewBlock(std::size_t col) {

		std::size_t offset = col * mY + mY;
		auto begin = mTensor.begin();
		auto end = std::next(begin, offset);
		return { begin, end };
	}
	ConstView constViewBlock(std::size_t col) const {

		std::size_t offset = col * mY + mY;
		auto begin = mTensor.begin();
		auto end = std::next(begin, offset);
		return { begin, end };
	}
	ConstView constViewBlock() const {

		std::size_t offset = mX * mY;
		auto begin = mTensor.begin();
		auto end = std::next(begin, offset);
		return { begin, end };
	}
	View viewBlock() {

		std::size_t offset = mX * mY;
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
		return  mTensor[col * mY + row];
	}
	const float& cat(std::size_t row, std::size_t col) const {
		return mTensor[col * mY + row];
	}


	float& at(std::size_t depth, std::size_t row, std::size_t col) {
		return  mTensor[depth * (mY * mX) + col * mY + row];
	}
	const float& cat(std::size_t depth, std::size_t row, std::size_t col) const  {
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

	View view(size_t col)  {
		return { &at(0, col), mY };
	}
	ConstView constView(size_t col) const {
		return { &cat(0, col), mY };
	}

	View view(std::size_t depth, std::size_t col) {
		return { &at(depth, 0, col), mY };
	}
	ConstView constView(std::size_t depth, size_t col) const {
		return { &cat(depth, 0, col), mY };
	}
};