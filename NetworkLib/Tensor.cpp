#include "Tensor.h"

using namespace NetworkLib;

Tensor::Tensor(View floats, std::size_t x, std::size_t y, std::size_t z, std::size_t w)
			: mTensor(floats), mX(x), mY(y), mZ(z), mW(w) {}

Tensor::Tensor(Floats::iterator& begin, std::size_t x, std::size_t y, std::size_t z, std::size_t w)
	: mX(x), mY(y), mZ(z), mW(w) {

	auto size = x;
	if (y != 0)
		size *= y;
	if (z != 0)
		size *= z;
	if (w != 0)
		size *= w;

	mTensor = { begin, size };
	std::advance(begin, size);
}

std::size_t Tensor::size() const {
	return mX;
}
std::size_t Tensor::size2D() const {
	return mX * mY;
}
std::size_t Tensor::size3D() const {
	return mX * mY * mZ;
}
std::size_t Tensor::size4D() const {
	return mX * mY * mZ * mW;
}

Tensor::View Tensor::field(View view, std::size_t offset, std::size_t size) {
	return { view.begin() + offset, size };
}
Tensor::ConstView Tensor::constField(ConstView view, std::size_t offset, std::size_t size) {
	return { view.cbegin() + offset, size };
}

Tensor::View Tensor::viewBlock(std::size_t col) {

	std::size_t offset = col * mY + mY;
	auto begin = mTensor.begin();
	auto end = std::next(begin, offset);
	return { begin, end };
}
Tensor::ConstView Tensor::constViewBlock(std::size_t col) const {

	std::size_t offset = col * mY + mY;
	auto begin = mTensor.begin();
	auto end = std::next(begin, offset);
	return { begin, end };
}
Tensor::ConstView Tensor::constViewBlock() const {

	std::size_t offset = mX * mY;
	auto begin = mTensor.begin();
	auto end = std::next(begin, offset);
	return { begin, end };
}
Tensor::View Tensor::viewBlock() {

	std::size_t offset = mX * mY;
	auto begin = mTensor.begin();
	auto end = std::next(begin, offset);
	return { begin, end };
}

float& Tensor::at(std::size_t col) {
	return mTensor[col];
}
const float& Tensor::cat(std::size_t col) const {
	return mTensor[col];
}

float& Tensor::at(std::size_t row, std::size_t col) {
	return  mTensor[col * mY + row];
}
const float& Tensor::cat(std::size_t row, std::size_t col) const {
	return mTensor[col * mY + row];
}


float& Tensor::at(std::size_t depth, std::size_t row, std::size_t col) {
	return  mTensor[depth * (mY * mX) + col * mY + row];
}
const float& Tensor::cat(std::size_t depth, std::size_t row, std::size_t col) const {
	return  mTensor[depth * (mY * mX) + col * mY + row];
}

float& Tensor::at(std::size_t w, std::size_t z, std::size_t y, std::size_t x) {
	return  mTensor[w * (mZ * mY * mX) + z * (mY * mX) + y * mX + x];
}

Tensor::View Tensor::view() {
	return { &at(0), mX };
}
Tensor::ConstView Tensor::constView() const {
	return { &cat(0), mX };
}

Tensor::View Tensor::view(size_t col) {
	return { &at(0, col), mY };
}
Tensor::ConstView Tensor::constView(size_t col) const {
	return { &cat(0, col), mY };
}

Tensor::View Tensor::view(std::size_t depth, std::size_t col) {
	return { &at(depth, 0, col), mY };
}
Tensor::ConstView Tensor::constView(std::size_t depth, size_t col) const {
	return { &cat(depth, 0, col), mY };
}
	