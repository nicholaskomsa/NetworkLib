#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "kernel.h"

namespace Kernel = NetworkLib::Gpu::Kernel;

__global__ void cuRelu(const float* outputs, float* reluActivations, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) 
        reluActivations[idx] = fmaxf(0.0f, outputs[idx]);
}
__device__ float kReluPrime(float value) {
    return value < 0.0f ? 0.0f : 1.0f;
}

__global__ void cuApplyReluPrime(const float* reluActivations, float* reluPrimes, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size)
        reluPrimes[idx] *= kReluPrime(reluActivations[idx]);
}

__global__ void cuSoftmax1024(const float* outputs, float* softmaxActivations, int size) {
    extern __shared__ float shared_exp[];

    int tid = threadIdx.x;

    if (tid < size) 
        shared_exp[tid] = expf(outputs[tid]);
    
    __syncthreads();

    // Block-local reduction
    float sum = 0.0f;
    if (tid == 0) {
        for (int i = 0; i < size; ++i) {
            sum += shared_exp[i];
        }
        shared_exp[0] = sum;
    }
    __syncthreads();

    if (tid < size) 
        softmaxActivations[tid] = shared_exp[tid] / shared_exp[0];
}
__global__ void cuDiff(const float* desired, const float* sought, float* primes, int size) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < size) 
        primes[i] = sought[i] - desired[i];
}
__global__ void cuBatchedCopy(const float* src, float* dst, int size, int batchSize) {
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    int col = blockIdx.y;

    if (row < size && col < batchSize) {
        int i = row + col * size; // column-major offset
        dst[i] = src[i];
    }
}

__global__ void cuUpdateWeights(float* weights, const float* primes, const float* seen, int r, int c, float learnRate) {
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Column index
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Row index

    if (col < c && row < r) {
        int index_col_major = row + col * r;

        weights[index_col_major] -= primes[row] * seen[col] * learnRate;
    }
}
__global__ void cuBroadcastVectorToColumns(const float* src, float* dst, int size, int batchSize) {
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    int col = blockIdx.y;

    if (row < size && col < batchSize) {
        dst[row + col * size] = src[row];  // column-major offset
    }
}
__global__ void cuBroadcastVectorToColumnsAdd(const float* src, float* dst, int size, int batchSize) {
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    int col = blockIdx.y;

    if (row < size && col < batchSize) {
        dst[row + col * size] += src[row];  // column-major offset
    }
}


void Kernel::relu(cudaStream_t stream, const float* outputs, float* reluActivations, int size) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    cuRelu <<<blocksPerGrid, threadsPerBlock, 0, stream >>>(outputs, reluActivations, size);
}
void Kernel::applyReluPrime(cudaStream_t stream, const float* reluActivations, float* primes, int size) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    cuApplyReluPrime <<<blocksPerGrid, threadsPerBlock, 0, stream >>> (reluActivations, primes, size);
}
void Kernel::softmax(cudaStream_t stream, const float* outputs, float* softmaxActivations, int size) {
    cuSoftmax1024 <<<1, size, size * sizeof(float), stream >>>(outputs, softmaxActivations, size);
}
void Kernel::diff(cudaStream_t stream, const float* desired, const float* sought, float* primes, int size) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    cuDiff <<<1, size, 0, stream >>> (desired, sought, primes, size);
}
void Kernel::updateWeights(cudaStream_t stream, float* weights, const float* primes, const float* seen, int rows, int cols, float learnRate) {

    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((cols + 32 - 1) / 32, (rows + 32 - 1) / 32);
    cuUpdateWeights<<<numBlocks, threadsPerBlock, 0, stream>>>(weights, primes, seen, rows, cols, learnRate);
}
void Kernel::batchedCopy(cudaStream_t stream, const float* src, float* dst, int size, int batchSize) {
    int threadsPerBlock = 256;
    int blocksPerRow = (size + threadsPerBlock - 1) / threadsPerBlock;

    dim3 grid(blocksPerRow, batchSize);
    dim3 block(threadsPerBlock);

    cuBatchedCopy<<<grid, block, 0, stream >>>(src, dst, size, batchSize);
}
void Kernel::batchedBroadcast(cudaStream_t stream, const float* src, float* dst, int size, int batchSize) {
    int threadsPerBlock = 256;
    int blocksPerRow = (size + threadsPerBlock - 1) / threadsPerBlock;

    dim3 grid(blocksPerRow, batchSize);
    dim3 block(threadsPerBlock);

    cuBroadcastVectorToColumns<<<grid, block, 0, stream>>>(src, dst, size, batchSize);
}
void Kernel::batchedBroadcastAdd(cudaStream_t stream, const float* src, float* dst, int size, int batchSize) {
    int threadsPerBlock = 256;
    int blocksPerRow = (size + threadsPerBlock - 1) / threadsPerBlock;

    dim3 grid(blocksPerRow, batchSize);
    dim3 block(threadsPerBlock);

    cuBroadcastVectorToColumnsAdd << <grid, block, 0, stream >> > (src, dst, size, batchSize);
}
