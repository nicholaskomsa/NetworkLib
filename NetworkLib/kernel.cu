#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "kernel.h"

__global__ void krelu(float* outputs, float* activations, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        activations[idx] = fmaxf(0.0f, outputs[idx]);
    }
}

void NetworkLib::Gpu::Kernel::relu(cudaStream_t stream, float* outputs, float* activations, int size) {
    int threadsPerBlock = 64;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    krelu <<<blocksPerGrid, threadsPerBlock, 0, stream >>>(outputs, activations, size);
}
