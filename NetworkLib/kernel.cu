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
__device__ float kReluPrime(float value) {
    return value < 0.0f ? 0.0f : 1.0f;
}

__global__ void cuApplyReluPrime(const float* reluActivations, float* reluPrimes, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size)
        reluPrimes[idx] *= kReluPrime(reluActivations[idx]);
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
__global__ void cuBatchedUpdateWeights(float* weights, const float* primes, const float* seen, int rows, int cols, int batchSize, float learnRate) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z;  // Each block handles one sample

    if (col < cols && row < rows && b < batchSize) {
        float prime_val = primes[b * rows + row];   // primes[row, b]
        float seen_val = seen[b * cols + col];     // seen[col, b]

        int index_col_major = row + col * rows;

        atomicAdd(&weights[index_col_major], -learnRate * prime_val * seen_val);
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
__global__ void cuMse(const float* sought, const float* desired, float* result, int size, int batchSize) {
    extern __shared__ float partialSum[]; // shared memory per block

    int row = threadIdx.x + blockIdx.x * blockDim.x;
    int col = blockIdx.y;
    int tid = threadIdx.x;

    float localSum = 0.0f;

    if (row < size && col < batchSize) {
        int i = row + col * size; // column-major offset
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
    if (tid == 0) {
        float batchMse = partialSum[0] / size / batchSize;
        atomicAdd(result, batchMse);
    }
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

    cuScore <<<blocks, threadsPerBlock, 0, stream >>> (soughtBatch, desiredBatch, misses, size, batchSize);
}
void Kernel::mse(cudaStream_t stream, const float* sought, const float* desired, float* result, int size, int batchSize) {

    int threadsPerBlock = std::min(256, size);;
    int blocksPerBatch = (size + threadsPerBlock - 1) / threadsPerBlock;
    dim3 grid(blocksPerBatch, batchSize);   // one block per batch row
    dim3 block(threadsPerBlock);
    size_t sharedMemSize = threadsPerBlock * sizeof(float);

    cuMse<<<grid, block, sharedMemSize, stream>>>(sought, desired, result, size, batchSize);
}

void Kernel::relu(cudaStream_t stream, const float* outputs, float* reluActivations, int size) {
    int threadsPerBlock = std::min(size, 256);
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    cuRelu <<<blocksPerGrid, threadsPerBlock, 0, stream >>>(outputs, reluActivations, size);
}
void Kernel::applyReluPrime(cudaStream_t stream, const float* reluActivations, float* primes, int size) {
    int threadsPerBlock = std::min(size, 256);
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    cuApplyReluPrime <<<blocksPerGrid, threadsPerBlock, 0, stream >>> (reluActivations, primes, size);
}
void Kernel::softmax(cudaStream_t stream, const float* outputs, float* softmaxActivations, int size) {
    int threadsPerBlock = std::min(256, size);
    cuSoftmax1024 <<<1, threadsPerBlock, (size +1)* sizeof(float), stream >>>(outputs, softmaxActivations, size);
}
void Kernel::batchedSoftmax(cudaStream_t stream, const float* outputs, float* softmaxActivations, int size, int batchSize) {
    int sharedMemSize = sizeof(float) * (size + 1);
    
    cuSoftmaxBatch1024 << <batchSize, size, sharedMemSize, stream >> > (outputs, softmaxActivations, size);
}
void Kernel::diff(cudaStream_t stream, const float* desired, const float* sought, float* primes, int size) {
    cuDiff <<<1, size, 0, stream >>> (desired, sought, primes, size);
}
void Kernel::updateWeights(cudaStream_t stream, float* weights, const float* primes, const float* seen, int rows, int cols, float learnRate) {

    int tx = std::min(32, cols);
    int ty = std::min(32, rows);
    dim3 threadsPerBlock(tx, ty);
    dim3 numBlocks((cols + tx - 1) / tx, (rows + ty - 1) / ty);

    cuUpdateWeights<<<numBlocks, threadsPerBlock, 0, stream>>>(weights, primes, seen, rows, cols, learnRate);
}
void Kernel::batchedCopy(cudaStream_t stream, const float* src, float* dst, int size, int batchSize) {
   
    int threadsPerBlock = std::min(size, 256);
    int blocksPerRow = (size + threadsPerBlock - 1) / threadsPerBlock;

    dim3 grid(blocksPerRow, batchSize);
    dim3 block(threadsPerBlock);

    cuBatchedCopy<<<grid, block, 0, stream >>>(src, dst, size, batchSize);
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

    cuBroadcastVectorToColumnsAdd << <grid, block, 0, stream >> > (src, dst, size, batchSize);
}
void Kernel::batchedUpdateWeights(cudaStream_t stream, float* weights, const float* primes, const float* seen, int rows, int cols, int batchSize, float learnRate) {
   
    int tx = std::min(32, cols);
    int ty = std::min(32, rows);
    dim3 threadsPerBlock(tx, ty);
    int tz = 0;

    //x* y* z <= 1024
    if (tx * ty * batchSize < 1024)
        tz = batchSize;
    else
        tz = std::min(batchSize, 1024 / (tx * ty));

    dim3 blockDim(tx, ty, 1);  // Threads per block
    dim3 gridDim((cols + tx-1) / tx, (rows + ty-1) / ty, tz);  // One block per batch
    
    cuBatchedUpdateWeights<<<gridDim, blockDim, 0, stream>>>(weights, primes, seen, rows, cols, batchSize, learnRate);
}

__global__ void cuConv1(float* seen, float* weights, float* primes, int primesSize, int kernelSize, int kernelDepth) {
  
    int p = blockIdx.x * blockDim.x + threadIdx.x; // output index
    int k = blockIdx.y * blockDim.y + threadIdx.y; // kernel index

    if (k < kernelDepth && p < primesSize) {

        float sum = 0.0f;
        for (int w = 0; w < kernelSize; ++w)
            sum += weights[k * kernelSize + w] * seen[p + w];
        
        primes[k * primesSize + p] = sum;
    }
}
__global__ void cuBatchedConv1(float* seen, float* weights, float* primes, int primesSize, int kernelWidth, int kernelDepth, int batchSize) {

    int p = blockIdx.x * blockDim.x + threadIdx.x; // output index
    int k = blockIdx.y * blockDim.y + threadIdx.y; // kernel index
    int b = blockIdx.z * blockDim.z + threadIdx.z;

    if (k < kernelDepth && p < primesSize && b < batchSize) {

        std::size_t seenBatchOffset = (primesSize + kernelWidth-1) * b + p;
        std::size_t primesBatchOffset = primesSize * kernelDepth * b + k * primesSize + p;

        float sum = 0.0f;
        for (int w = 0; w < kernelWidth; ++w)
            sum += weights[k * kernelWidth + w] * seen[seenBatchOffset + w];

        primes[primesBatchOffset] = sum / batchSize;
    }
}
__global__ void cuConv1VecMulVec(float* weights, float* errors, float* primes, int kernelSize, int kPrimesSize, int kernelDepth) {

    int p = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;

    if (p < kPrimesSize && k < kernelDepth) {
        int idx = kPrimesSize * k + p, wOffset = kernelSize * k;
        float e = errors[idx], sum = 0.0f;

        for (int w = 0; w < kernelSize; ++w)
            sum += weights[ wOffset + w ] * e;

        primes[idx] = sum;
    }
}
__global__ void cuBatchedConv1VecMulVec(float* weights, float* errors, float* primes, int kernelSize, int primesSize, int kernelDepth, int batchSize) {

    int p = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z * blockDim.z + threadIdx.z;

    if (p < primesSize && k < kernelDepth && b < batchSize) {

        int batchOffset = primesSize * kernelDepth * b;
        int idx = batchOffset + primesSize * k + p, wOffset = kernelSize * k;

        float e = errors[idx], sum = 0.0f;

        for (int w = 0; w < kernelSize; ++w)
            sum += weights[wOffset + w] * e;

        primes[idx] = sum;
    }
}
void Kernel::conv1VecMulVec(cudaStream_t stream, float* weights, float* errors, float* primes, int primesSize, int kernelWidth, int kernelDepth) {

    int kprimesSize = std::max(1, primesSize / kernelDepth);
    int tx = std::min(32, kprimesSize);       // threads per block in x → output positions
    int ty = std::min(32, kernelDepth);      // threads per block in y → kernel depth

    dim3 threadsPerBlock(tx, ty);
    dim3 numBlocks((kprimesSize + tx - 1) / tx, (kernelDepth + ty - 1) / ty);

    cuConv1VecMulVec<<<numBlocks, threadsPerBlock, 0, stream>>>(weights, errors, primes, kernelWidth, kprimesSize, kernelDepth);

}
void Kernel::batchedConv1VecMulVec(cudaStream_t stream, float* weights, float* errors, float* primes, int primesSize, int kernelWidth, int kernelDepth, int batchSize) {

    int kprimesSize = std::max(1, primesSize / kernelDepth);
    int tx = std::min(32, kprimesSize);       // output positions
    int ty = std::min(32, kernelDepth);       // kernel depth
    int tz = 0;

    //x* y* z <= 1024
    if (tx * ty * batchSize < 1024)
        tz = batchSize;         
    else
        tz = std::min(batchSize, 1024 / (tx * ty));

    dim3 threadsPerBlock(tx, ty, tz);
    dim3 numBlocks(
        (kprimesSize + tx - 1) / tx,
        (kernelDepth + ty - 1) / ty,
        (batchSize + tz - 1) / tz
    );

    cuBatchedConv1VecMulVec << <numBlocks, threadsPerBlock, 0, stream >> > (weights, errors, primes, kernelWidth, kprimesSize, kernelDepth, batchSize);

}
__global__ void cuConv1UpdateWeights(float* seen, float* weights, float* primes, int outputSize, int kernelSize, int kernelDepth, float learnRate) {
   
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // output index
    int kernel = blockIdx.y * blockDim.y + threadIdx.y; // kernel index

    if (idx < outputSize && kernel < kernelDepth ) {

        float prime_val = -learnRate * primes[kernel * outputSize + idx] / float(outputSize);
        int kernelOffset = kernel * kernelSize;

        for (int k = 0; k < kernelSize; ++k) 
            atomicAdd(&weights[kernelOffset + k], prime_val * seen[idx + k] );
    }
}

__global__ void cuBatchedConv1UpdateWeights(float* seen, float* weights, float* primes, int kPrimesSize, int kernelWidth, int kernelDepth, int batchSize, float learnRate) {

    int p = blockIdx.x * blockDim.x + threadIdx.x; // output index
    int k = blockIdx.y * blockDim.y + threadIdx.y; // kernel index
    int b = blockIdx.z * blockDim.z + threadIdx.z;

    if (k < kernelDepth && p < kPrimesSize && b < batchSize) {

        std::size_t seenBatchOffset = (kPrimesSize + kernelWidth - 1) * b + p;
        std::size_t primesBatchOffset = kPrimesSize * kernelDepth * b + k * kPrimesSize + p;

        float prime_val = -learnRate * primes[primesBatchOffset] / kPrimesSize / batchSize;
        int kernelOffset = k * kernelWidth;

        for (int w = 0; w < kernelWidth; ++w)
            atomicAdd(&weights[kernelOffset + w], prime_val * seen[seenBatchOffset+w]);
    }
}
void Kernel::conv1(cudaStream_t stream, float* weights, float* primes, float* seen, int primesSize, int kernelSize, int kernelDepth) {
    int kprimesSize = std::max(1, primesSize / kernelDepth);
    int tx = std::min(32, kprimesSize);       // threads per block in x → output positions
    int ty = std::min(32, kernelDepth);      // threads per block in y → kernel depth

    dim3 threadsPerBlock(tx, ty);
    dim3 numBlocks((kprimesSize + tx - 1) / tx, (kernelDepth + ty - 1) / ty);  

    cuConv1<<<numBlocks, threadsPerBlock, 0, stream>>>(seen, weights, primes, kprimesSize, kernelSize, kernelDepth);
   
}
void Kernel::batchedConv1(cudaStream_t stream, float* weights, float* output, float* seen, int primesSize, int kernelSize, int kernelDepth, int batchSize) {
    int kprimesSize = std::max(1, primesSize / kernelDepth);
    int tx = std::min(32, kprimesSize);       // threads per block in x → output positions
    int ty = std::min(32, kernelDepth);      // threads per block in y → kernel depth
    int tz = 0;

    //x* y* z <= 1024
    if (tx * ty * batchSize < 1024)
        tz = batchSize;
    else
        tz = std::min(batchSize, 1024 / (tx * ty));

    dim3 threadsPerBlock(tx, ty, tz);
    dim3 numBlocks(
        (kprimesSize + tx - 1) / tx,
        (kernelDepth + ty - 1) / ty,
        (batchSize + tz - 1) / tz
    );

    cuBatchedConv1<<<numBlocks, threadsPerBlock, 0, stream>>>(seen, weights, output, kprimesSize, kernelSize, kernelDepth, batchSize);
}
void Kernel::conv1UpdateKernel(cudaStream_t stream, float* weights, float* primes, float* seen, int primesSize, int kernelSize, int kernelDepth, float learnRate) {

    int kprimesSize = std::max(1, primesSize / kernelDepth);
    int tx = std::min(32, kprimesSize);       // threads per block in x → output positions
    int ty = std::min(32, kernelDepth);      // threads per block in y → kernel depth

    dim3 threadsPerBlock(tx, ty);
    dim3 numBlocks((kprimesSize + tx - 1) / tx, (kernelDepth + ty - 1) / ty);

    cuConv1UpdateWeights<<<numBlocks, threadsPerBlock, 0, stream>>>(seen, weights, primes, kprimesSize, kernelSize, kernelDepth, learnRate);
}
void Kernel::batchedConv1UpdateKernel(cudaStream_t stream, float* weights, float* primes, float* seen, int primesSize, int kernelSize, int kernelDepth, int batchSize, float learnRate) {

    int kprimesSize = std::max(1, primesSize / kernelDepth);
    int tx = std::min(32, kprimesSize);       // threads per block in x → output positions
    int ty = std::min(32, kernelDepth);      // threads per block in y → kernel depth
    int tz = 0;

    //x* y* z <= 1024
    if (tx * ty * batchSize < 1024)
        tz = batchSize;
    else
        tz = std::min(batchSize, 1024 / (tx * ty));

    dim3 threadsPerBlock(tx, ty, tz);
    dim3 numBlocks(
        (kprimesSize + tx - 1) / tx,
        (kernelDepth + ty - 1) / ty,
        (batchSize + tz - 1) / tz
    );
    cuBatchedConv1UpdateWeights<<<numBlocks, threadsPerBlock, 0, stream>>>(seen, weights, primes, primesSize, kernelSize, kernelDepth, batchSize, learnRate);
}