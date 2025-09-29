#pragma once

#include <format>

#include "cpuTensor.h"

using namespace NetworkLib;

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
void Error::checkMissMatch(Cpu::Tensor::Dimension a, Cpu::Tensor::Dimension b, const std::source_location& location) {
	if (a == b) return;
	auto message = std::format("{}x{} mismatch\n{}\n{}\n{}\n", a, b, location.file_name(), location.line(), location.function_name());
	throw Error(std::errc::invalid_argument, message);
}
void Error::checkBounds(Cpu::Tensor::Coordinate a, Cpu::Tensor::Dimension b, const std::source_location& location) {
	if (a < b) return;
	auto message = std::format("{}, {} out of bounds\n{}\n{}\n{}\n", a, b, location.file_name(), location.line(), location.function_name());
	throw Error(std::errc::invalid_argument, message);
}
