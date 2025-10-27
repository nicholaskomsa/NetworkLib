#include <stdio.h>

#include <cfloat>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "kernel.h"

namespace Kernel = NetworkLib::Gpu::Kernel;

__global__ void cuUpdateWeights(float* weights, const float* primes, const float* seen, int r, int c, float learnRate) {
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Column index
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Row index

    if (col < c && row < r) {
        int index_col_major = row + col * r;

        weights[index_col_major] -= primes[row] * seen[col] * learnRate;
    }
}
void Kernel::updateWeights(cudaStream_t stream, float* weights, const float* primes, const float* seen, int rows, int cols, float learnRate) {

    int tx = std::min(32, cols);
    int ty = std::min(32, rows);
    dim3 threadsPerBlock(tx, ty);
    dim3 numBlocks((cols + tx - 1) / tx, (rows + ty - 1) / ty);

    cuUpdateWeights << <numBlocks, threadsPerBlock, 0, stream >> > (weights, primes, seen, rows, cols, learnRate);
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
    dim3 gridDim((cols + tx - 1) / tx, (rows + ty - 1) / ty, tz);  // One block per batch

    cuBatchedUpdateWeights << <gridDim, blockDim, 0, stream >> > (weights, primes, seen, rows, cols, batchSize, learnRate);
}
