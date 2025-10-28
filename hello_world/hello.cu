#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>

// CUDA Kernel Function
__global__ void helloWorldKernel() {
    printf("Hello, world -- block %d, thread %d\n", blockIdx.x, threadIdx.x);
}

int main() {
    std::cout << "Hello, world (CPU)\n" << std::endl;
    
    // Define blocks and threads per block
    const int N_BLOCKS = 8;
    const int THREADS_PER_BLOCK = 256;

    // Call CUDA Kernel function (runs on GPU)
    helloWorldKernel<<<N_BLOCKS, THREADS_PER_BLOCK>>>();

    // Wait until all GPU work is complete
    cudaDeviceSynchronize();

    // Check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cout << "CUDA Error: " << cudaGetErrorString(error) << std::endl;
        return -1;
    }
    
    std::cout << "\nProgram Complete (Success)!" << std::endl;

    return 0;
}