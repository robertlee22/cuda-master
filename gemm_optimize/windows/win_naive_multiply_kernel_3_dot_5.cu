#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>

// A (M,K) * B (K,N) -> C (M,N)
constexpr int M = 4096 * 2;
constexpr int N = 4096 * 2;
constexpr int K = 1024 * 2;

// 与 win_cute_gemm.cu 中 cta_tile / tC 一致：BLK_M=128, BLK_N=128, BLK_K=8；线程块 16×16，每线程 8×8 输出子块
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 8;
constexpr int TM = 8;
constexpr int TN = 8;
constexpr int THREADS = (BM / TM) * (BN / TN);  // 16 * 16 = 256

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            std::cerr << "CUDA error: " << cudaGetErrorString(err__)            \
                      << " (" << static_cast<int>(err__) << "), file "          \
                      << __FILE__ << ", line " << __LINE__ << std::endl;        \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

// +1 列填充，减轻 B 在 (N) 维上的 shared bank 冲突；A 在 K 维 +1 同理
__global__ void __launch_bounds__(THREADS, 2)
multiply(const int* __restrict__ A, const int* __restrict__ B, int* __restrict__ C,
         const int m, const int k, const int n) {
    __shared__ int As[BM][BK + 1];
    __shared__ int Bs[BK][BN + 1];

    const int tx = threadIdx.x % (BN / TN);
    const int ty = threadIdx.x / (BN / TN);

    // 与 win_cute_gemm.cu 的 dimGrid(ceil_div(M,bM), ceil_div(N,bN)) 一致：blockIdx.x -> M，blockIdx.y -> N
    const int row0 = blockIdx.x * BM + ty * TM;
    const int col0 = blockIdx.y * BN + tx * TN;

    int creg[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            creg[i][j] = 0;
        }
    }

    for (int tile = 0; tile * BK < k; ++tile) {
        const int k0 = tile * BK;

        for (int idx = threadIdx.x; idx < BM * BK; idx += THREADS) {
            const int ar = idx / BK;
            const int ac = idx % BK;
            const int rr = blockIdx.x * BM + ar;
            const int cc = k0 + ac;
            As[ar][ac] = (rr < m && cc < k) ? A[static_cast<size_t>(rr) * k + cc] : 0;
        }
        for (int idx = threadIdx.x; idx < BK * BN; idx += THREADS) {
            const int br = idx / BN;
            const int bc = idx % BN;
            const int rr = k0 + br;
            const int cc = blockIdx.y * BN + bc;
            Bs[br][bc] = (rr < k && cc < n) ? B[static_cast<size_t>(rr) * n + cc] : 0;
        }

        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            int a_vec[TM];
            int b_vec[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i) {
                a_vec[i] = As[ty * TM + i][kk];
            }
#pragma unroll
            for (int j = 0; j < TN; ++j) {
                b_vec[j] = Bs[kk][tx * TN + j];
            }
#pragma unroll
            for (int i = 0; i < TM; ++i) {
#pragma unroll
                for (int j = 0; j < TN; ++j) {
                    creg[i][j] += a_vec[i] * b_vec[j];
                }
            }
        }

        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int r = row0 + i;
            const int c = col0 + j;
            if (r < m && c < n) {
                C[static_cast<size_t>(r) * n + c] = creg[i][j];
            }
        }
    }
}

int main() {
    CUDA_CHECK(cudaSetDevice(0));

    int* A = nullptr;
    int* B = nullptr;
    int* C = nullptr;

    CUDA_CHECK(cudaMallocManaged(&A, sizeof(int) * static_cast<size_t>(M) * K));
    CUDA_CHECK(cudaMallocManaged(&B, sizeof(int) * static_cast<size_t>(K) * N));
    CUDA_CHECK(cudaMallocManaged(&C, sizeof(int) * static_cast<size_t>(M) * N));

    for (int i = 0; i < M * K; ++i) {
        A[i] = 1;
    }
    for (int i = 0; i < K * N; ++i) {
        B[i] = 1;
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    int concurrent_managed = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&concurrent_managed, cudaDevAttrConcurrentManagedAccess, device));
    if (concurrent_managed) {
        cudaMemLocation on_device{cudaMemLocationTypeDevice, device};
        constexpr unsigned int prefetch_flags = 0;
        CUDA_CHECK(cudaMemPrefetchAsync(A, sizeof(int) * static_cast<size_t>(M) * K, on_device,
                                        prefetch_flags, nullptr));
        CUDA_CHECK(cudaMemPrefetchAsync(B, sizeof(int) * static_cast<size_t>(K) * N, on_device,
                                        prefetch_flags, nullptr));
        CUDA_CHECK(cudaMemPrefetchAsync(C, sizeof(int) * static_cast<size_t>(M) * N, on_device,
                                        prefetch_flags, nullptr));
    }

    dim3 block(THREADS);
    dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);
    multiply<<<grid, block>>>(A, B, C, M, K, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    bool pass = true;
    for (int i = 0; i < M * N; ++i) {
        if (C[i] != K) {
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
& "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\nvcc.exe" win_naive_multiply_kernel_3_dot_5.cu -o naive_win_3_dot_5.exe -ccbin "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.50.35717\bin\Hostx64\x64" -std=c++14 -Xcompiler "/utf-8 /EHsc /O2 /MD" -allow-unsupported-compiler
*/
