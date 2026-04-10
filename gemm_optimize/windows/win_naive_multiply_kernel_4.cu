#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>

// A (M,K) * B (K,N) -> C (M,N)
constexpr int M = 4096 * 2;
constexpr int N = 4096 * 2;
constexpr int K = 1024 * 2;

constexpr int BM = 32;
constexpr int BN = 32;
constexpr int BK = 32;
// 相对 v3 的 2×2：4×4 线程子块，单线程更多 FMA、更高算术强度；块内 8×8=64 线程。
constexpr int TM = 4;
constexpr int TN = 4;
constexpr int THREADS = (BM / TM) * (BN / TN);  // 8 * 8 = 64

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

// 全局内存用 int4 宽加载，再标量写入共享内存，避免 As/Bs 行步长非 16 字节对齐导致无法直接 int4 写 shared。
__device__ inline void store_a_tile(int (*As)[BK + 1], int ar, int ac, int rr, int cc, int m, int k,
                                    const int* __restrict__ A) {
    if (rr < m && cc + 3 < k) {
        const int4 v = *reinterpret_cast<const int4*>(&A[static_cast<size_t>(rr) * k + cc]);
        As[ar][ac + 0] = v.x;
        As[ar][ac + 1] = v.y;
        As[ar][ac + 2] = v.z;
        As[ar][ac + 3] = v.w;
    } else {
        for (int u = 0; u < 4; ++u) {
            As[ar][ac + u] = (rr < m && cc + u < k) ? A[static_cast<size_t>(rr) * k + cc + u] : 0;
        }
    }
}

__device__ inline void store_b_tile(int (*Bs)[BN + 1], int br, int bc, int rr, int cc, int k, int n,
                                    const int* __restrict__ B) {
    if (rr < k && cc + 3 < n) {
        const int4 v = *reinterpret_cast<const int4*>(&B[static_cast<size_t>(rr) * n + cc]);
        Bs[br][bc + 0] = v.x;
        Bs[br][bc + 1] = v.y;
        Bs[br][bc + 2] = v.z;
        Bs[br][bc + 3] = v.w;
    } else {
        for (int u = 0; u < 4; ++u) {
            Bs[br][bc + u] = (rr < k && cc + u < n) ? B[static_cast<size_t>(rr) * n + cc + u] : 0;
        }
    }
}

__global__ void __launch_bounds__(THREADS, 4)
multiply(const int* __restrict__ A, const int* __restrict__ B, int* __restrict__ C, const int m, const int k,
         const int n) {
    __shared__ int As[BM][BK + 1];
    __shared__ int Bs[BK][BN + 1];

    const int tx = threadIdx.x % (BN / TN);
    const int ty = threadIdx.x / (BN / TN);

    const int row0 = blockIdx.y * BM + ty * TM;
    const int col0 = blockIdx.x * BN + tx * TN;

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

        for (int idx = threadIdx.x; idx < (BM * BK) / 4; idx += THREADS) {
            const int ar = idx / (BK / 4);
            const int ac = (idx % (BK / 4)) * 4;
            const int rr = blockIdx.y * BM + ar;
            const int cc = k0 + ac;
            store_a_tile(As, ar, ac, rr, cc, m, k, A);
        }
        for (int idx = threadIdx.x; idx < (BK * BN) / 4; idx += THREADS) {
            const int br = idx / (BN / 4);
            const int bc = (idx % (BN / 4)) * 4;
            const int rr = k0 + br;
            const int cc = blockIdx.x * BN + bc;
            store_b_tile(Bs, br, bc, rr, cc, k, n, B);
        }

        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            const int a0 = As[ty * TM + 0][kk];
            const int a1 = As[ty * TM + 1][kk];
            const int a2 = As[ty * TM + 2][kk];
            const int a3 = As[ty * TM + 3][kk];
            const int b0 = Bs[kk][tx * TN + 0];
            const int b1 = Bs[kk][tx * TN + 1];
            const int b2 = Bs[kk][tx * TN + 2];
            const int b3 = Bs[kk][tx * TN + 3];
            creg[0][0] += a0 * b0;
            creg[0][1] += a0 * b1;
            creg[0][2] += a0 * b2;
            creg[0][3] += a0 * b3;
            creg[1][0] += a1 * b0;
            creg[1][1] += a1 * b1;
            creg[1][2] += a1 * b2;
            creg[1][3] += a1 * b3;
            creg[2][0] += a2 * b0;
            creg[2][1] += a2 * b1;
            creg[2][2] += a2 * b2;
            creg[2][3] += a2 * b3;
            creg[3][0] += a3 * b0;
            creg[3][1] += a3 * b1;
            creg[3][2] += a3 * b2;
            creg[3][3] += a3 * b3;
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
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
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
& "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\nvcc.exe" win_naive_multiply_kernel_4.cu -o naive_win_4_release.exe -ccbin "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.50.35717\bin\Hostx64\x64" -std=c++14 -Xcompiler "/utf-8 /EHsc /O2 /MD" -Xptxas -O3 -allow-unsupported-compiler
*/
