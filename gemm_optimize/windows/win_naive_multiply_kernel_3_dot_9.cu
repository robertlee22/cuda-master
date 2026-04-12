#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iostream>

// A (M,K) * B (K,N) -> C (M,N)
constexpr int M = 4096 * 2;
constexpr int N = 4096 * 2;
constexpr int K = 1024 * 2;

constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 16;       // ← 从 8 提升到 16: 主循环迭代数减半, sync 开销减半
constexpr int TM = 8;
constexpr int TN = 8;
constexpr int THREADS = (BM / TM) * (BN / TN);  // 256

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

/*
 * v3.9 — 基于 v3.8 诊断结果的修正:
 *
 *  v3.8 教训: 双缓冲在 SM 7.5（无 cp.async）上导致 STS 与 LDS 同时
 *  涌入 MIO 管线，MIO Throttle 从第一名变成压倒性瓶颈。
 *
 *  v3.9 策略:
 *    ✓ 保留 float4 向量化加载/存储（v3.8 验证 Long Scoreboard 几乎消失）
 *    ✗ 去掉双缓冲，回到单缓冲 + 两次 sync 的简洁结构
 *    ✓ BK 从 8 → 16:
 *       - 主循环迭代数减半 (256 → 128 for K=2048)
 *       - __syncthreads() 次数减半
 *       - 每 tile 的计算/访存比翻倍 (16×64=1024 FMA vs 8×64=512)
 *       - smem: As[128][17] + Bs[16][129] ≈ 17KB, 可跑 2 blocks/SM
 *
 *    float4 加载:
 *       A tile: 128×16 = 2048 floats = 512 float4, 每线程 2 个 float4
 *       B tile: 16×128 = 2048 floats = 512 float4, 每线程 2 个 float4
 */
__global__ void __launch_bounds__(THREADS, 2)
multiply(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
         const int m, const int k, const int n) {

    __shared__ float As[BM][BK + 1];    // [128][17], +1 防 bank conflict
    __shared__ float Bs[BK][BN + 1];    // [16][129]

    const int tx = threadIdx.x % (BN / TN);          // 0..15
    const int ty = threadIdx.x / (BN / TN);          // 0..15

    const int block_row = blockIdx.x * BM;
    const int block_col = blockIdx.y * BN;

    // 累加寄存器
    float creg[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
            creg[i][j] = 0.f;

    // =========== 主循环 ===========
    for (int tile = 0; tile * BK < k; ++tile) {
        const int k0 = tile * BK;

        // ---- float4 向量化加载 A tile ----
        // BM*BK/4 = 128*16/4 = 512 个 float4, 256 线程各做 2 个
#pragma unroll
        for (int idx = threadIdx.x; idx < (BM * BK / 4); idx += THREADS) {
            const int flat = idx << 2;               // idx * 4
            const int ar   = flat / BK;              // smem 行 0..127
            const int ac   = flat % BK;              // smem 列 0,4,8,12
            const int gr   = block_row + ar;         // global 行
            const int gc   = k0 + ac;                // global 列

            if (gr < m && gc + 3 < k) {
                float4 tmp = *reinterpret_cast<const float4*>(
                    &A[static_cast<size_t>(gr) * k + gc]);
                As[ar][ac    ] = tmp.x;
                As[ar][ac + 1] = tmp.y;
                As[ar][ac + 2] = tmp.z;
                As[ar][ac + 3] = tmp.w;
            } else {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int r = gr, c = gc + i;
                    As[ar][ac + i] = (r < m && c < k)
                        ? A[static_cast<size_t>(r) * k + c] : 0.f;
                }
            }
        }

        // ---- float4 向量化加载 B tile ----
        // BK*BN/4 = 16*128/4 = 512 个 float4, 256 线程各做 2 个
#pragma unroll
        for (int idx = threadIdx.x; idx < (BK * BN / 4); idx += THREADS) {
            const int flat = idx << 2;
            const int br   = flat / BN;              // smem 行 0..15
            const int bc   = flat % BN;              // smem 列 0,4,8,...,124
            const int gr   = k0 + br;
            const int gc   = block_col + bc;

            if (gr < k && gc + 3 < n) {
                float4 tmp = *reinterpret_cast<const float4*>(
                    &B[static_cast<size_t>(gr) * n + gc]);
                Bs[br][bc    ] = tmp.x;
                Bs[br][bc + 1] = tmp.y;
                Bs[br][bc + 2] = tmp.z;
                Bs[br][bc + 3] = tmp.w;
            } else {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int r = gr, c = gc + i;
                    Bs[br][bc + i] = (r < k && c < n)
                        ? B[static_cast<size_t>(r) * n + c] : 0.f;
                }
            }
        }

        __syncthreads();

        // ---- 计算: BK=16 次内循环 ----
#pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            float a_vec[TM];
            float b_vec[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i)
                a_vec[i] = As[ty * TM + i][kk];
#pragma unroll
            for (int j = 0; j < TN; ++j)
                b_vec[j] = Bs[kk][tx * TN + j];
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j)
                    creg[i][j] += a_vec[i] * b_vec[j];
        }

        __syncthreads();
    }

    // =========== float4 向量化写回 C ===========
#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int r = block_row + ty * TM + i;
        const int c = block_col + tx * TN;
        if (r < m && c + TN - 1 < n) {
            *reinterpret_cast<float4*>(
                &C[static_cast<size_t>(r) * n + c]) =
                make_float4(creg[i][0], creg[i][1], creg[i][2], creg[i][3]);
            *reinterpret_cast<float4*>(
                &C[static_cast<size_t>(r) * n + c + 4]) =
                make_float4(creg[i][4], creg[i][5], creg[i][6], creg[i][7]);
        } else if (r < m) {
#pragma unroll
            for (int j = 0; j < TN; ++j) {
                if (c + j < n)
                    C[static_cast<size_t>(r) * n + c + j] = creg[i][j];
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
    CUDA_CHECK(cudaDeviceGetAttribute(&concurrent_managed,
                                      cudaDevAttrConcurrentManagedAccess, device));
    if (concurrent_managed) {
        cudaMemLocation on_device{cudaMemLocationTypeDevice, device};
        constexpr unsigned int prefetch_flags = 0;
        CUDA_CHECK(cudaMemPrefetchAsync(A, sizeof(float) * static_cast<size_t>(M) * K,
                                        on_device, prefetch_flags, nullptr));
        CUDA_CHECK(cudaMemPrefetchAsync(B, sizeof(float) * static_cast<size_t>(K) * N,
                                        on_device, prefetch_flags, nullptr));
        CUDA_CHECK(cudaMemPrefetchAsync(C, sizeof(float) * static_cast<size_t>(M) * N,
                                        on_device, prefetch_flags, nullptr));
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
  编译（CUDA 12.6 + VS 2022）：

  & "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\nvcc.exe" ^
    win_naive_multiply_kernel_3_dot_9.cu -o naive_win_3_dot_9.exe ^
    -ccbin "...\MSVC\...\Hostx64\x64" ^
    -std=c++14 -Xcompiler "/utf-8 /EHsc /O2 /MD" -arch=sm_75
*/
