#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iostream>

// A (M,K) * B (K,N) -> C (M,N)
constexpr int M = 4096 * 2;
constexpr int N = 4096 * 2;
constexpr int K = 1024 * 2;

// 与 3.5 相同 CTA / 线程子块；3.7 增加 shared 双缓冲 + cp.async 与计算重叠（需 sm_80+）
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

// cp.async 源须为 global；越界时从此字读取 0（与同步路径语义一致）
__device__ float g_cp_async_oob_zero = 0.f;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800

__device__ __forceinline__ void cp_async_ca_4B(void* dst_smem, const void* src_gmem) {
    const uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(dst_smem));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" : : "r"(smem), "l"(src_gmem) : "memory");
}

__device__ __forceinline__ void cp_async_commit_group() {
    asm volatile("cp.async.commit_group;\n" :: : "memory");
}

__device__ __forceinline__ void cp_async_wait_group0() {
    asm volatile("cp.async.wait_group 0;\n" :: : "memory");
}

template <typename BufAs, typename BufBs>
__device__ void load_tile_async(const float* __restrict__ A, const float* __restrict__ B, int m, int k,
                                int n, int tile, int buf, BufAs& As, BufBs& Bs) {
    const int k0 = tile * BK;

    for (int idx = threadIdx.x; idx < BM * BK; idx += THREADS) {
        const int ar = idx / BK;
        const int ac = idx % BK;
        const int rr = blockIdx.x * BM + ar;
        const int cc = k0 + ac;
        const float* src =
            (rr < m && cc < k) ? &A[static_cast<size_t>(rr) * k + cc] : &g_cp_async_oob_zero;
        cp_async_ca_4B(&As[buf][ar][ac], src);
    }
    for (int idx = threadIdx.x; idx < BK * BN; idx += THREADS) {
        const int br = idx / BN;
        const int bc = idx % BN;
        const int rr = k0 + br;
        const int cc = blockIdx.y * BN + bc;
        const float* src =
            (rr < k && cc < n) ? &B[static_cast<size_t>(rr) * n + cc] : &g_cp_async_oob_zero;
        cp_async_ca_4B(&Bs[buf][br][bc], src);
    }
}

template <typename BufAs, typename BufBs>
__device__ void compute_tile(int buf, int ty, int tx, float creg[TM][TN], const BufAs& As,
                             const BufBs& Bs) {
#pragma unroll
    for (int kk = 0; kk < BK; ++kk) {
        float a_vec[TM];
        float b_vec[TN];
#pragma unroll
        for (int i = 0; i < TM; ++i) {
            a_vec[i] = As[buf][ty * TM + i][kk];
        }
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            b_vec[j] = Bs[buf][kk][tx * TN + j];
        }
#pragma unroll
        for (int i = 0; i < TM; ++i) {
#pragma unroll
            for (int j = 0; j < TN; ++j) {
                creg[i][j] += a_vec[i] * b_vec[j];
            }
        }
    }
}

#endif  // __CUDA_ARCH__ >= 800

__global__ void __launch_bounds__(THREADS, 2)
multiply(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
         const int m, const int k, const int n) {
    const int tx = threadIdx.x % (BN / TN);
    const int ty = threadIdx.x / (BN / TN);

    const int row0 = blockIdx.x * BM + ty * TM;
    const int col0 = blockIdx.y * BN + tx * TN;

    float creg[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            creg[i][j] = 0.f;
        }
    }

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800

    __shared__ float As[2][BM][BK + 1];
    __shared__ float Bs[2][BK][BN + 1];

    const int num_tiles = (k + BK - 1) / BK;

    load_tile_async(A, B, m, k, n, 0, 0, As, Bs);
    cp_async_commit_group();
    cp_async_wait_group0();
    __syncthreads();

    for (int t = 0; t < num_tiles; ++t) {
        const int curr = t & 1;

        if (t + 1 < num_tiles) {
            load_tile_async(A, B, m, k, n, t + 1, 1 - curr, As, Bs);
            cp_async_commit_group();
        }

        compute_tile(curr, ty, tx, creg, As, Bs);

        __syncthreads();

        if (t + 1 < num_tiles) {
            cp_async_wait_group0();
        }
        __syncthreads();
    }

#else

    // 无 cp.async 的架构：退化为 3.5 单缓冲（仍正确）
    __shared__ float As[BM][BK + 1];
    __shared__ float Bs[BK][BN + 1];

    for (int tile = 0; tile * BK < k; ++tile) {
        const int k0 = tile * BK;

        for (int idx = threadIdx.x; idx < BM * BK; idx += THREADS) {
            const int ar = idx / BK;
            const int ac = idx % BK;
            const int rr = blockIdx.x * BM + ar;
            const int cc = k0 + ac;
            As[ar][ac] = (rr < m && cc < k) ? A[static_cast<size_t>(rr) * k + cc] : 0.f;
        }
        for (int idx = threadIdx.x; idx < BK * BN; idx += THREADS) {
            const int br = idx / BN;
            const int bc = idx % BN;
            const int rr = k0 + br;
            const int cc = blockIdx.y * BN + bc;
            Bs[br][bc] = (rr < k && cc < n) ? B[static_cast<size_t>(rr) * n + cc] : 0.f;
        }

        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            float a_vec[TM];
            float b_vec[TN];
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

#endif

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

    CUDA_CHECK(cudaDeviceSynchronize());

    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    int concurrent_managed = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&concurrent_managed, cudaDevAttrConcurrentManagedAccess, device));
    if (concurrent_managed) {
        cudaMemLocation on_device{cudaMemLocationTypeDevice, device};
        constexpr unsigned int prefetch_flags = 0;
        CUDA_CHECK(cudaMemPrefetchAsync(A, sizeof(float) * static_cast<size_t>(M) * K, on_device,
                                        prefetch_flags, nullptr));
        CUDA_CHECK(cudaMemPrefetchAsync(B, sizeof(float) * static_cast<size_t>(K) * N, on_device,
                                        prefetch_flags, nullptr));
        CUDA_CHECK(cudaMemPrefetchAsync(C, sizeof(float) * static_cast<size_t>(M) * N, on_device,
                                        prefetch_flags, nullptr));
    }

    dim3 block(THREADS);
    dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);
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
  双缓冲依赖 Ampere 及以上（cp.async）。建议指定架构，例如：
  & "...\nvcc.exe" win_naive_multiply_kernel_3_dot_7.cu -o naive_win_3_dot_7.exe `
    -gencode arch=compute_80,code=sm_80 -ccbin "...\Hostx64\x64" `
    -std=c++14 -Xcompiler "/utf-8 /EHsc /O2 /MD" -allow-unsupported-compiler
*/
