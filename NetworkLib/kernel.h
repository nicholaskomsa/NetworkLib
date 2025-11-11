#pragma once

#include <cuda_runtime.h>

namespace NetworkLib {
	namespace Gpu {
		namespace Kernel{
 
			void relu(cudaStream_t stream, const float* outputs, float* reluActivations, int size);
            void applyReluPrime(cudaStream_t stream, const float* reluActivations, float* primes, int size);
			void updateWeights(cudaStream_t stream, float* weights, const float* primes, const float* seen, int cols, int rows, float learnRate);

			void softmax(cudaStream_t stream, const float* outputs, float* softmaxActivations, int size);
			void batchedSoftmax(cudaStream_t stream, const float* outputs, float* softmaxActivations, int size, int batchSize);

			void sqe(cudaStream_t stream, const float* sought, const float* desired, float* result, int desiredSize, int batchSize);
			void sqe2(cudaStream_t stream, const float* sought, const float* desired, float* result, int desiredSize, int batchSize);

			void score(cudaStream_t stream, const float* soughtBatch, const float* desiredBatch, int* misses, int size, int batchSize);
			void score2(cudaStream_t stream, const float* soughtBatch, const float* desiredBatch, int* misses, int size, int batchSize);

			void diff(cudaStream_t stream, const float* desired, const float* sought, float* primes, int size);
			void diff2(cudaStream_t stream, const float* desired, const float* sought, float* primes, int sought2Size, int desired1Size);

			void batchedCopy(cudaStream_t stream, const float* src, float* dst, int size, int batchSize);
			void batchedBroadcast(cudaStream_t stream, const float* src, float* dst, int size, int batchSize);
			void batchedBroadcastAdd(cudaStream_t stream, const float* src, float* dst, int size, int batchSize);
			void batchedUpdateWeights(cudaStream_t stream, float* weights, const float* primes, const float* seen, int cols, int rows, int batchSize, float learnRate);
			
			void conv1(cudaStream_t stream, float* weights, float* primes, float* seen, int primesSize, int kernelRows, int kernelDepth, int kernel);
			void batchedConv1(cudaStream_t stream, float* weights, float* primes, float* seen, int primesSize, int kernelSize, int kernelDepth, int batchSize);
			void conv1UpdateKernel(cudaStream_t stream, float* weights, float* primes, float* seen
				, int primesSize, int kernelRows, int kernelDepth, int kernel, float learnRate);

			void batchedConv1UpdateKernel(cudaStream_t stream, float* weights, float* primes, float* seen, int primesSize, int kernelWidth, int kernelDepth, int batchSize, float learnRate);
			void conv1VecMulVec(cudaStream_t stream, float* weights, float* errors, float* primes, int kernelRows, int primesSize, int kernelDepth, int kernel);
			void batchedConv1VecMulVec(cudaStream_t stream, float* weights, float* errors, float* primes, int kernelRows, int primesSize, int kernelDepth, int batchSize);

		}
	}
}

