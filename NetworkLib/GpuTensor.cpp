#pragma once

#include <sstream>

#include "GpuTensor.h"

using namespace NetworkLib::Gpu;

Error::Error(std::errc code, const std::string& message)
	: std::system_error(int(code), std::generic_category(), message) {}

void Error::checkCuda(cudaError_t result, const std::source_location& location) {
	if (result == cudaSuccess) return;

	auto message = std::format(
		"Cuda Error ={}:\n{}"
		"\n{}\n{}\n{}\n"
		, int(result), cudaGetErrorString(result)
		, location.file_name(), location.line(), location.function_name());

	throw Error(std::errc::operation_canceled, message);
}
void Error::checkBlas(cublasStatus_t result, const std::source_location& location) {
	if (result == CUBLAS_STATUS_SUCCESS) return;

	auto getBLASString = [&]() {
		switch (result) {
		case CUBLAS_STATUS_SUCCESS:          return "Success";
		case CUBLAS_STATUS_NOT_INITIALIZED:  return "cuBLAS not initialized";
		case CUBLAS_STATUS_ALLOC_FAILED:     return "Resource allocation failed";
		case CUBLAS_STATUS_INVALID_VALUE:    return "Invalid value";
		case CUBLAS_STATUS_ARCH_MISMATCH:    return "Architecture mismatch";
		case CUBLAS_STATUS_MAPPING_ERROR:    return "Memory mapping error";
		case CUBLAS_STATUS_EXECUTION_FAILED: return "Execution failed";
		case CUBLAS_STATUS_INTERNAL_ERROR:   return "Internal error";
		default:                             return "Unknown cuBLAS error";
		}
		};

	auto message = std::format(
		"BLAS Error ={}:\n{}"
		"\n{}\n{}\n{}\n"
		, int(result), getBLASString()
		, location.file_name(), location.line(), location.function_name());

	throw Error(std::errc::operation_canceled, message);
}
void Error::checkMissMatch(Dimension a, Dimension b, const std::source_location& location) {
	if (a == b) return;
	auto message = std::format("{}x{} mismatch\n{}\n{}\n{}\n", a, b, location.file_name(), location.line(), location.function_name());
	throw Error(std::errc::invalid_argument, message);
}
void Error::checkBounds(Coordinate a, Dimension b, const std::source_location& location) {
	if (a < b) return;
	auto message = std::format("{}, {} out of bounds\n{}\n{}\n{}\n", a, b, location.file_name(), location.line(), location.function_name());
	throw Error(std::errc::invalid_argument, message);
}

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

void FloatSpace1::create(std::size_t size) {

	float* cpu, * gpu;
	Error::checkCuda(cudaMallocHost(&cpu, size * sizeof(float)));
	Error::checkCuda(cudaMalloc(reinterpret_cast<void**>(&gpu), size * sizeof(float)));
	mView = { Cpu::Tensor::View1(cpu, size), gpu, cpu };
}

void FloatSpace1::destroy() {
	freeHost();
	freeGpu();
}

void FloatSpace1::freeHost() {
	if (mView.mCpu)
		Error::checkCuda(cudaFreeHost(mView.mCpu));
	mView.mCpu = nullptr;
}
void FloatSpace1::freeGpu() {
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

void FloatSpace1::upload() {
	mView.upload();
}
void FloatSpace1::downloadAsync(cudaStream_t stream) {
	mView.downloadAsync(stream);
}
