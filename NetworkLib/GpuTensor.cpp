#pragma once

#include <sstream>

#include "GpuTensor.h"

using namespace NetworkLib::Gpu;


void Float::upload() {
	Error::checkCuda(cudaMemcpy(
		mGpu,
		mCpu,
		1 * sizeof(float),
		cudaMemcpyHostToDevice));
}
void Float::downloadAsync(cudaStream_t stream) const {
	Error::checkCuda(cudaMemcpyAsync(
		mCpu,
		mGpu,
		1 * sizeof(float),
		cudaMemcpyDeviceToHost,
		stream));
}

Float::operator float() const {
	return *mCpu;
}
Float& Float::operator=(float v) {
	*mCpu = v;
	return *this;
}

void Int::upload() {
	Error::checkCuda(cudaMemcpy(
		mGpu,
		mCpu,
		1 * sizeof(float),
		cudaMemcpyHostToDevice));

}
void Int::downloadAsync(cudaStream_t stream) const {
	Error::checkCuda(cudaMemcpyAsync(
		mCpu,
		mGpu,
		1 * sizeof(float),
		cudaMemcpyDeviceToHost,
		stream));
}
Int::operator int() const {
	return *mCpu;
}

Int& Int::operator=(int v){
	*mCpu = v;
	return *this;
}

void FloatSpace1::create(const Cpu::FloatSpace1& cpuSpace) {

	float* cpu = cpuSpace.mCpu, * gpu=nullptr;
	std::size_t size = cpuSpace.mView.extent(0);
	Error::checkCuda(cudaMalloc(reinterpret_cast<void**>(&gpu), size * sizeof(float)));
	
	mView = { cpuSpace.mView, gpu, cpu };
}

void FloatSpace1::resize(const Cpu::FloatSpace1& cpuSpace) {

	float* cpu = cpuSpace.mCpu, * gpu = mView.mGpu;
	std::size_t size = cpuSpace.mView.extent(0);

	if (size != mView.mSize || mView.mGpu == nullptr) {
		destroy();
		Error::checkCuda(cudaMalloc(reinterpret_cast<void**>(&gpu), size * sizeof(float)));
	}

	mView = { cpuSpace.mView, gpu, cpu };
}
void FloatSpace1::destroy() {
	if (mView.mGpu)
		Error::checkCuda(cudaFree(mView.mGpu));
	mView.mGpu = nullptr;
}

float* FloatSpace1::getGpu(float* cpu) {
	return mView.mGpu + (cpu - mView.mCpu);
}

float* FloatSpace1::begin() { return mView.begin(); }
float* FloatSpace1::end() { return mView.end(); }

void FloatSpace1::advance(Float& f, float*& begin) {
	auto source = begin;
	f = { getGpu(source), source };
	++begin;
}
void FloatSpace1::advance(Int& i, float*& begin) {
	//float and int are same size so just reinterpret
	int* cpu = reinterpret_cast<int*>(begin);
	int* gpu = reinterpret_cast<int*>(getGpu(begin));
	i = { gpu, cpu };
	++begin;
}
void FloatSpace1::upload() {
	mView.upload();
}
void FloatSpace1::downloadAsync(cudaStream_t stream) {
	mView.downloadAsync(stream);
}
