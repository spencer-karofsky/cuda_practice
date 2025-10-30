#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>


// CPU
void vectorAddCPU(
    const std::vector<float>& A,
    const std::vector<float>& B, 
    std::vector<float>& C,
    int n
) {
    for (int i = 0; i < n; ++i) {
        C[i] = A[i] + B[i];
    }
}

// CUDA (GPU)
__global__ void vectorAddKernel(float* A_d, float* B_d, float* C_d, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        C_d[i] = A_d[i] + B_d[i];
    }
}

int main() {
    // Run vector_add.py
    system("python3 vector_add.py");
    std::cout << "\n";

    srand(time(0));

    const int N = 1000000;
    size_t size = N * sizeof(float);

    const int THREADS_PER_BLOCK = 256;
    dim3 DimGrid((N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, 1, 1);
    dim3 DimBlock(THREADS_PER_BLOCK, 1, 1);

    std::vector<float> A(N), B(N), C_cpu(N), C_gpu(N);
    for (int i = 0; i < A.size(); ++i) {
        A[i] = rand() % 100;
        B[i] = rand() % 100;
    }

    // CPU vectorAdd vectorAdd execution and timing
    auto cpu_start = std::chrono::high_resolution_clock::now();
    vectorAddCPU(A, B, C_cpu, N);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    auto cpu_duration = std::chrono::duration_cast<std::chrono::milliseconds>(cpu_end - cpu_start);
    std::cout << "C++ vectorAddCPU() Time: " << cpu_duration.count() << " ms" << std::endl;


    // Allocate memory on the host (CPU) for pointers to the vectors
    float *A_d, *B_d, *C_d;
    cudaMalloc(&A_d, size);
    cudaMalloc(&B_d, size);
    cudaMalloc(&C_d, size);

    // Copy allocated memory from host (CPU) to device (GPU)
    cudaMemcpy(A_d, A.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B.data(), size, cudaMemcpyHostToDevice);
    
    // CUDA timing objects
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // GPU (CUDA) execution of vectorAdd execution and timing
    cudaEventRecord(start);
    vectorAddKernel<<<DimGrid, DimBlock>>>(A_d, B_d, C_d, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_ms = 0;
    cudaEventElapsedTime(&gpu_ms, start, stop);

    std::cout << "CUDA vectorAddKernel() Time: " << gpu_ms << " ms" << std::endl;

    // Copy vector from device back to host
    cudaMemcpy(C_gpu.data(), C_d, size, cudaMemcpyDeviceToHost);

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

    return 0;
}