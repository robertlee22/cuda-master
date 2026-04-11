#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iostream>

// A shape = (M,K), B shape = (K,N), C shape = (M,N)
constexpr int M = 4096 * 2;
constexpr int N = 4096 * 2;
constexpr int K = 1024 * 2;
constexpr int TILE = 32;

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            std::cerr << "CUDA error: " << cudaGetErrorString(err__)            \
                      << " (" << static_cast<int>(err__) << "), file "          \
                      << __FILE__ << ", line " << __LINE__ << std::endl;        \
            std::exit(EXIT_FAILURE);                                             \
        }                                                                       \
    } while (0)

__global__ void multiply(const float* A, const float* B, float* C, int m, int k, int n) {
    const int col = threadIdx.x + blockDim.x * blockIdx.x;
    const int row = threadIdx.y + blockDim.y * blockIdx.y;

    if (col < n && row < m) {
        float sum = 0.f;

        __shared__ float As[TILE][TILE];
        __shared__ float Bs[TILE][TILE];

        for (int t = 0; t * TILE < k; ++t) {
            const int ax = t * TILE + threadIdx.x;
            const int by = t * TILE + threadIdx.y;

            As[threadIdx.y][threadIdx.x] =
                (row < m && ax < k) ? A[static_cast<size_t>(row) * k + ax] : 0.f;
            Bs[threadIdx.y][threadIdx.x] =
                (by < k && col < n) ? B[static_cast<size_t>(by) * n + col] : 0.f;

            __syncthreads();

            for (int kk = 0; kk < TILE; ++kk) {
                sum += As[threadIdx.y][kk] * Bs[kk][threadIdx.x];
            }
            __syncthreads();
        }
        C[static_cast<size_t>(row) * n + col] = sum;
    }
}

int main() {
    float* A = nullptr;
    float* B = nullptr;
    float* C = nullptr;

    CUDA_CHECK(cudaMallocManaged(&A, sizeof(float) * static_cast<size_t>(M) * K));
    CUDA_CHECK(cudaMallocManaged(&B, sizeof(float) * static_cast<size_t>(K) * N));
    CUDA_CHECK(cudaMallocManaged(&C, sizeof(float) * static_cast<size_t>(M) * N));

    for (int i = 0; i < M * K; ++i) {
        A[i] = 1.f;
    }
    for (int i = 0; i < K * N; ++i) {
        B[i] = 1.f;
    }

    // Keep this version maximally compatible for Windows + MSVC + NVCC.
    // Explicitly migrate managed pages by touching data on host first, then synchronize.
    CUDA_CHECK(cudaDeviceSynchronize());

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    multiply<<<grid, block>>>(A, B, C, M, K, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    constexpr float expected = static_cast<float>(K);
    constexpr float tol = 1e-3f;
    bool pass = true;
    for (int i = 0; i < M * N; ++i) {
        if (std::fabs(C[i] - expected) > tol) {
            std::cout << "C[" << i << "] = " << C[i] << std::endl;
            pass = false;
            break;
        }
    }

    std::cout << (pass ? "pass!" : "error!") << std::endl;

    CUDA_CHECK(cudaFree(A));
    CUDA_CHECK(cudaFree(B));
    CUDA_CHECK(cudaFree(C));
    return 0;
}

/*
& "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\nvcc.exe" naive_multiply_kernel_optimize_2.cu -o naive_multiply_kernel_optimize_2.exe -ccbin "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.50.35717\bin\Hostx64\x64" -std=c++14 -Xcompiler "/utf-8 /EHsc /Od /Zi /MD" -Xptxas -O0 -G -allow-unsupported-compiler
*/