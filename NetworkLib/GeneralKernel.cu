#include <stdio.h>

#include <cfloat>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "kernel.h"

namespace Kernel = NetworkLib::Gpu::Kernel;

__global__ void cuRelu(const float* outputs, float* reluActivations, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size)
        reluActivations[idx] = fmaxf(0.0f, outputs[idx]);
}
void Kernel::relu(cudaStream_t stream, const float* outputs, float* reluActivations, int size) {
    int threadsPerBlock = std::min(size, 256);
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    cuRelu<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(outputs, reluActivations, size);
}
__device__ float kReluPrime(float value) {
    return value < 0.0f ? 0.0f : 1.0f;
}
__global__ void cuApplyReluPrime(const float* reluActivations, float* reluPrimes, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size)
        reluPrimes[idx] *= kReluPrime(reluActivations[idx]);
}
void Kernel::applyReluPrime(cudaStream_t stream, const float* reluActivations, float* primes, int size) {
    int threadsPerBlock = std::min(size, 256);
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    cuApplyReluPrime<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(reluActivations, primes, size);
}

__global__ void cuSoftmax1024(const float* outputs, float* softmaxActivations, int size) {
    extern __shared__ float shared_data[]; //[exp_vals[<1024], max_val]

    float* shared_exp = shared_data;
    float& shared_max = shared_data[size]; // single float for max

    int tid = threadIdx.x;

    // Step 1: Find max
    float local_max = -FLT_MAX;
    if (tid < size)
        local_max = outputs[tid];

    // Reduction to find max
    shared_exp[tid] = local_max;
    __syncthreads();

    for (int stride = size / 2; stride > 0; stride >>= 1) {

        if (tid < stride && tid + stride < size)
            shared_exp[tid] = fmaxf(shared_exp[tid], shared_exp[tid + stride]);

        __syncthreads();
    }

    if (tid == 0)
        shared_max = shared_exp[0];

    __syncthreads();

    // Step 2: Compute exp(x - max)
    if (tid < size)
        shared_exp[tid] = expf(outputs[tid] - shared_max);

    __syncthreads();

    // Step 3: Sum exp values
    float sum = 0.0f;
    if (tid == 0) {
        for (int i = 0; i < size; ++i)
            sum += shared_exp[i];

        shared_max = sum; // reuse shared_max as shared_sum
    }
    __syncthreads();

    // Step 4: Normalize
    if (tid < size)
        softmaxActivations[tid] = shared_exp[tid] / shared_max;
}
void Kernel::softmax(cudaStream_t stream, const float* outputs, float* softmaxActivations, int size) {
    int threadsPerBlock = std::min(256, size);
    cuSoftmax1024 << <1, threadsPerBlock, (size + 1) * sizeof(float), stream >> > (outputs, softmaxActivations, size);
}
__global__ void cuSoftmaxBatch1024(const float* outputs, float* softmaxActivations, int size) {
    extern __shared__ float shared_data[]; // [exp_vals[<1024], max_val]

    int tid = threadIdx.x;
    int batchIdx = blockIdx.x;

    // Offset for this batch vector
    const float* input = outputs + batchIdx * size;
    float* output = softmaxActivations + batchIdx * size;

    float* shared_exp = shared_data;
    float& shared_max = shared_data[size]; // single float for max

    // Step 1: Find max
    float local_max = -FLT_MAX;
    if (tid < size)
        local_max = input[tid];

    shared_exp[tid] = local_max;
    __syncthreads();

    for (int stride = size / 2; stride > 0; stride >>= 1) {
        if (tid < stride && tid + stride < size)
            shared_exp[tid] = fmaxf(shared_exp[tid], shared_exp[tid + stride]);
        __syncthreads();
    }

    if (tid == 0)
        shared_max = shared_exp[0];
    __syncthreads();

    // Step 2: Compute exp(x - max)
    if (tid < size)
        shared_exp[tid] = expf(input[tid] - shared_max);
    __syncthreads();

    // Step 3: Sum exp values
    float sum = 0.0f;
    if (tid == 0) {
        for (int i = 0; i < size; ++i)
            sum += shared_exp[i];
        shared_max = sum; // reuse shared_max as shared_sum
    }
    __syncthreads();

    // Step 4: Normalize
    if (tid < size)
        output[tid] = shared_exp[tid] / shared_max;
}
void Kernel::batchedSoftmax(cudaStream_t stream, const float* outputs, float* softmaxActivations, int size, int batchSize) {
    int sharedMemSize = sizeof(float) * (size + 1);

    cuSoftmaxBatch1024 << <batchSize, size, sharedMemSize, stream >> > (outputs, softmaxActivations, size);
}
__global__ void cuDiff(const float* desired, const float* sought, float* primes, int size) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < size)
        primes[i] = sought[i] - desired[i];
}
void Kernel::diff(cudaStream_t stream, const float* desired, const float* sought, float* primes, int size) {
    cuDiff << <1, size, 0, stream >> > (desired, sought, primes, size);
}
__global__ void cuDiff2(const float* desired, const float* sought, float* primes, int sought2Size, int desired1Size) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < sought2Size)
        primes[i] = sought[i] - desired[i % desired1Size];
}
void Kernel::diff2(cudaStream_t stream, const float* desired, const float* sought, float* primes, int sought2Size, int desired1Size) {
    cuDiff2 << <1, sought2Size, 0, stream >> > (desired, sought, primes, sought2Size, desired1Size);
}
__global__ void cuBatchedCopy(const float* src, float* dst, int size, int batchSize) {
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    int col = blockIdx.y;

    if (row < size && col < batchSize) {
        int i = row + col * size; // column-major offset
        dst[i] = src[i];
    }
}
void Kernel::batchedCopy(cudaStream_t stream, const float* src, float* dst, int size, int batchSize) {

    int threadsPerBlock = std::min(size, 256);
    int blocksPerRow = (size + threadsPerBlock - 1) / threadsPerBlock;

    dim3 grid(blocksPerRow, batchSize);
    dim3 block(threadsPerBlock);

    cuBatchedCopy << <grid, block, 0, stream >> > (src, dst, size, batchSize);
}
__global__ void cuMse(const float* sought, const float* desired, float* result, int desiredSize, int batchSize) {
   
    //sought2, desired1

    extern __shared__ float partialSum[]; // shared memory per block

    int row = threadIdx.x + blockIdx.x * blockDim.x;
    int col = blockIdx.y;
    int tid = threadIdx.x;

    float localSum = 0.0f;

    if (row < desiredSize && col < batchSize) {
        int i = row + col * desiredSize; // column-major offset
        float diff = sought[i] - desired[row];
        localSum = diff * diff;
    }

    partialSum[tid] = localSum;
    __syncthreads();

    // Block-level reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            partialSum[tid] += partialSum[tid + stride];

        __syncthreads();
    }

    // Accumulate per-batch sum into global result
    if (tid == 0) 
        atomicAdd(result, partialSum[0]);
    
}
__global__ void cuNormalizeMSE(float* result, int desiredSize, int batchSize) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int total = desiredSize * batchSize;
        *result /= total;
    }
}
void Kernel::mse(cudaStream_t stream, const float* sought, const float* desired, float* result, int desiredSize, int batchSize) {

    int threadsPerBlock = std::min(256, desiredSize);
    int blocksPerBatch = (desiredSize + threadsPerBlock - 1) / threadsPerBlock;
    dim3 grid(blocksPerBatch, batchSize);   // one block per batch row
    dim3 block(threadsPerBlock);
    size_t sharedMemSize = threadsPerBlock * sizeof(float);

    cuMse<<<grid, block, sharedMemSize, stream>>>(sought, desired, result, desiredSize, batchSize);
    //normalize later after mse reduced
}

__global__ void cuMse2(const float* sought, const float* desired, float* result, int desiredSize, int batchSize) {
    extern __shared__ float partialSum[]; // shared memory per block

    //sought2, desired2

    int row = threadIdx.x + blockIdx.x * blockDim.x;
    int col = blockIdx.y;
    int tid = threadIdx.x;

    float localSum = 0.0f;

    if (row < desiredSize && col < batchSize) {
        int i = row + col * desiredSize; // column-major offset
        float diff = sought[i] - desired[i];
        localSum = diff * diff;
    }

    partialSum[tid] = localSum;
    __syncthreads();

    // Block-level reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            partialSum[tid] += partialSum[tid + stride];

        __syncthreads();
    }

    // Accumulate per-batch sum into global result
    if (tid == 0) 
        atomicAdd(result, partialSum[0]);
    
}
void Kernel::mse2(cudaStream_t stream, const float* sought, const float* desired, float* result, int desiredSize, int batchSize) {
    int threadsPerBlock = std::min(256, desiredSize);
    int blocksPerBatch = (desiredSize + threadsPerBlock - 1) / threadsPerBlock;
    dim3 grid(blocksPerBatch, batchSize);   // one block per batch row
    dim3 block(threadsPerBlock);
    size_t sharedMemSize = threadsPerBlock * sizeof(float);

    cuMse2 << <grid, block, sharedMemSize, stream >> > (sought, desired, result, desiredSize, batchSize);
    //normalize later
}
__global__ void cuScore(const float* soughtBatch, const float* desiredBatch, int* misses, int size, int batchSize) {

    int batch = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch >= batchSize) return;

    const float* sought = soughtBatch + batch * size;
    const float* desired = desiredBatch + batch * size;

    int maxSoughtIdx = 0;
    int maxDesiredIdx = 0;
    float maxSoughtVal = sought[0];
    float maxDesiredVal = desired[0];

    for (int i = 1; i < size; ++i) {
        if (sought[i] > maxSoughtVal) {
            maxSoughtVal = sought[i];
            maxSoughtIdx = i;
        }
        if (desired[i] > maxDesiredVal) {
            maxDesiredVal = desired[i];
            maxDesiredIdx = i;
        }
    }

    if (maxSoughtIdx != maxDesiredIdx) {
        atomicAdd(misses, 1);
    }
}
void Kernel::score(cudaStream_t stream, const float* soughtBatch, const float* desiredBatch, int* misses, int size, int batchSize) {

    int threadsPerBlock = std::min(256, size);
    int blocks = (batchSize + threadsPerBlock - 1) / threadsPerBlock;

    cuScore<<<blocks, threadsPerBlock, 0, stream>>>(soughtBatch, desiredBatch, misses, size, batchSize);
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
void Kernel::batchedBroadcast(cudaStream_t stream, const float* src, float* dst, int size, int batchSize) {
    int threadsPerBlock = std::min(size, 256);
    int blocksPerRow = (size + threadsPerBlock - 1) / threadsPerBlock;

    dim3 grid(blocksPerRow, batchSize);
    dim3 block(threadsPerBlock);

    cuBroadcastVectorToColumns<<<grid, block, 0, stream>>>(src, dst, size, batchSize);
}
void Kernel::batchedBroadcastAdd(cudaStream_t stream, const float* src, float* dst, int size, int batchSize) {
    int threadsPerBlock = std::min(size, 256);;
    int blocksPerRow = (size + threadsPerBlock - 1) / threadsPerBlock;

    dim3 grid(blocksPerRow, batchSize);
    dim3 block(threadsPerBlock);

    cuBroadcastVectorToColumnsAdd<<<grid, block, 0, stream>>>(src, dst, size, batchSize);
}
