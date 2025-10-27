#include <stdio.h>

#include <cfloat>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "kernel.h"

namespace Kernel = NetworkLib::Gpu::Kernel;

__global__ void cuConv1(float* seen, float* weights, float* primes, int primesSize, int kernelWidth, int kernelDepth, int kernel) {

    int p = blockIdx.x * blockDim.x + threadIdx.x; // output index

    if (p < primesSize) {

        float sum = 0.0f;
        for (int w = 0; w < kernelWidth; ++w)
            sum += weights[kernel * kernelWidth + w] * seen[p + w];

        primes[kernel * primesSize + p] = sum;
    }
}
void Kernel::conv1(cudaStream_t stream, float* weights, float* primes, float* seen, int primesSize, int kernelWidth, int kernelDepth, int kernel) {
    int kprimesSize = std::max(1, primesSize / kernelDepth);
    int tx = std::min(32, kprimesSize);       // threads per block in x ? output positions

    dim3 threadsPerBlock(tx);
    dim3 numBlocks((kprimesSize + tx - 1) / tx);

    cuConv1 << <numBlocks, threadsPerBlock, 0, stream >> > (seen, weights, primes, kprimesSize, kernelWidth, kernelDepth, kernel);

}
__global__ void cuBatchedConv1(float* seen, float* weights, float* primes, int primesSize, int kernelWidth, int kernelDepth, int batchSize) {
    /*
    int p = blockIdx.x * blockDim.x + threadIdx.x; // output index
    int k = blockIdx.y * blockDim.y + threadIdx.y; // kernel index
    int b = blockIdx.z * blockDim.z + threadIdx.z;

    if (k < kernelDepth && p < primesSize && b < batchSize) {

        std::size_t seenBatchOffset = (primesSize + kernelWidth - 1) * b + p;
        std::size_t primesBatchOffset = primesSize * kernelDepth * b + k * primesSize + p;

        for (int w = 0; w < kernelWidth; ++w)
            atomicAdd(&primes[primesBatchOffset], weights[k * kernelWidth + w] * seen[seenBatchOffset + w]);
    }
    */
}
void Kernel::batchedConv1(cudaStream_t stream, float* weights, float* output, float* seen, int primesSize, int kernelSize, int kernelDepth, int batchSize) {
    int kprimesSize = std::max(1, primesSize / kernelDepth);
    int tx = std::min(32, kprimesSize);       // threads per block in x ? output positions
    int ty = std::min(32, kernelDepth);      // threads per block in y ? kernel depth
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

    cuBatchedConv1 << <numBlocks, threadsPerBlock, 0, stream >> > (seen, weights, output, kprimesSize, kernelSize, kernelDepth, batchSize);
}
__global__ void cuConv1VecMulVec(float* weights, float* errors, float* primes, int kernelWidth, int kPrimesSize, int kernelDepth, int kernel) {

    int p = blockIdx.x * blockDim.x + threadIdx.x;

    if (p < kPrimesSize) {

        int idx = kPrimesSize * kernel + p, wOffset = kernelWidth * kernel;

        float sum = 0.0f;
        for (int w = 0; w < kernelWidth; ++w)
            sum += weights[wOffset + w] * errors[idx];

        primes[idx] = sum;
    }
}

void Kernel::conv1VecMulVec(cudaStream_t stream, float* weights, float* errors, float* primes, int primesSize, int kernelWidth, int kernelDepth, int kernel) {

    int kprimesSize = std::max(1, primesSize / kernelDepth);
    int tx = std::min(32, kprimesSize);       // threads per block in x ? output positions

    dim3 threadsPerBlock(tx);
    dim3 numBlocks((kprimesSize + tx - 1) / tx);

    cuConv1VecMulVec << <numBlocks, threadsPerBlock, 0, stream >> > (weights, errors, primes, kernelWidth, kprimesSize, kernelDepth, kernel);

}

__global__ void cuBatchedConv1VecMulVec(float* weights, float* errors, float* primes, int kernelSize, int primesSize, int kernelDepth, int batchSize) {

    int p = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z * blockDim.z + threadIdx.z;

    if (p < primesSize && k < kernelDepth && b < batchSize) {

        int batchOffset = primesSize * kernelDepth * b;
        int idx = batchOffset + primesSize * k + p, wOffset = kernelSize * k;

        float e = errors[idx];

        for (int w = 0; w < kernelSize; ++w)
            atomicAdd(&primes[idx], weights[wOffset + w] * e);
    }
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
__global__ void cuConv1UpdateKernel(float* seen, float* weights, float* primes, int kPrimesSize, int kernelWidth, int kernel, float learnRate) {

    int p = blockIdx.x * blockDim.x + threadIdx.x; // kprimes index

    if (p < kPrimesSize) {

        int primesOffset = kPrimesSize * kernel;

        int kernelHeight = 1;

        float prime_val = primes[primesOffset + p] * -learnRate / kPrimesSize;

        for (int x = 0; x < kernelWidth; ++x) {
            for (int y = 0; y < kernelHeight; ++y) {

                int index_col_major = y + kernelHeight * (x + kernelWidth * kernel);
                atomicAdd(&weights[index_col_major], prime_val * seen[p + x]); //seen[p * kernelWidth + x] 2d seen
            }
        }
    }
}
void Kernel::conv1UpdateKernel(cudaStream_t stream, float* weights, float* primes, float* seen, int primesSize, int kernelSize, int kernelDepth, int kernel, float learnRate) {

    int kprimesSize = std::max(1, primesSize / kernelDepth);
    int tx = std::min(32, kprimesSize);       // threads per block in x ? output positions

    dim3 threadsPerBlock(tx);
    dim3 numBlocks((kprimesSize + tx - 1) / tx);

    cuConv1UpdateKernel<<<numBlocks, threadsPerBlock, 0, stream>>>(seen, weights, primes, kprimesSize, kernelSize, kernel, learnRate);
}

__global__ void cuBatchedConv1UpdateKernel(float* seen, float* weights, float* primes, int kPrimesSize, int kernelWidth, int kernelDepth, int batchSize, float learnRate) {

    /*
    *         int index_col_major = row + col * rows;
    */
    /*no work
    int p = blockIdx.x * blockDim.x + threadIdx.x; // output index
    int k = blockIdx.y * blockDim.y + threadIdx.y; // kernel index
    int b = blockIdx.z * blockDim.z + threadIdx.z;

    if (k < kernelDepth && p < kPrimesSize && b < batchSize) {

        std::size_t inputSize = kPrimesSize + kernelWidth - 1;
        std::size_t seenBatchOffset = inputSize * b + p;
        std::size_t primesBatchOffset = kPrimesSize * kernelDepth * b + k * kPrimesSize + p;

        float prime_val = -learnRate * primes[primesBatchOffset] / kPrimesSize;
        int kernelOffset = k * kernelWidth;

        for (int w = 0; w < kernelWidth; ++w)
            atomicAdd(&weights[kernelOffset + w], prime_val * seen[seenBatchOffset+w]);
    }
    */
}
void Kernel::batchedConv1UpdateKernel(cudaStream_t stream, float* weights, float* primes, float* seen, int primesSize, int kernelSize, int kernelDepth, int batchSize, float learnRate) {

    int kprimesSize = std::max(1, primesSize / kernelDepth);
    int tx = std::min(32, kprimesSize);       // threads per block in x ? output positions
    int ty = std::min(32, kernelDepth);      // threads per block in y ? kernel depth
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
    cuBatchedConv1UpdateKernel<<<numBlocks, threadsPerBlock, 0, stream>>>(seen, weights, primes, primesSize, kernelSize, kernelDepth, batchSize, learnRate);
}