#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iostream>

// A (M,K) * B (K,N) -> C (M,N)
constexpr int M = 4096 * 2;
constexpr int N = 4096 * 2;
constexpr int K = 1024 * 2;

// 与 v3.5 保持相同的分块尺寸，方便对比
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

/*
 * v3.8 — 基于 v3.5 的三大改进:
 *
 *  1) float4 向量化全局加载/存储 — 减少 4 倍 load/store 指令数
 *     → 降低 Stall Long Scoreboard
 *
 *  2) 双缓冲 + 软件流水线 — 先发射下一 tile 的 LDG(落入寄存器)，
 *     再用当前 smem 做 FMA 计算，最后把寄存器 STS 到另一组 smem。
 *     由于 SM 7.5 的 LDG 在硬件层面是异步的（线程仅在"读取目标寄存器"
 *     时才 stall），FMA 计算期间 LDG 可以在后台完成。
 *     → 降低 Stall Wait（sync 从 2 次/迭代 → 1 次/迭代）
 *     → 隐藏 global memory latency
 *
 *  3) float4 写回 C — 减少 store 指令数
 *
 *  注意: Bs 的 4-way bank conflict 在 TM=TN=8 + 16×16 线程布局下
 *  是结构性的（stride=8, gcd(8,32)=8），无法仅靠 padding/swizzle
 *  消除，需要改线程映射才能彻底解决。本版保留这一限制，
 *  靠流水线和向量化弥补。
 */
__global__ void __launch_bounds__(THREADS, 2)
multiply(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
         const int m, const int k, const int n) {

    // =========== 双缓冲 shared memory ===========
    __shared__ float As[2][BM][BK + 1];   // +1 padding 保持与 v3.5 一致
    __shared__ float Bs[2][BK][BN + 1];

    // ---------- 线程→输出坐标映射（与 v3.5 相同）----------
    const int tx = threadIdx.x % (BN / TN);          // 0..15
    const int ty = threadIdx.x / (BN / TN);          // 0..15

    const int block_row = blockIdx.x * BM;
    const int block_col = blockIdx.y * BN;

    // ---------- 累加寄存器 ----------
    float creg[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
            creg[i][j] = 0.f;

    // =========== 预计算每线程的 global→smem 加载坐标 ===========
    //
    // A tile: BM×BK = 128×8 = 1024 个 float = 256 个 float4
    //   → 256 个线程，每线程恰好 1 个 float4
    //
    // B tile: BK×BN = 8×128 = 1024 个 float = 256 个 float4
    //   → 256 个线程，每线程恰好 1 个 float4
    //
    const int a_flat = threadIdx.x << 2;              // threadIdx.x * 4
    const int a_smem_row = a_flat / BK;               // 0..127
    const int a_smem_col = a_flat % BK;               // 0 或 4
    const int a_global_row = block_row + a_smem_row;  // 不随 tile 变化

    const int b_flat = threadIdx.x << 2;
    const int b_smem_row = b_flat / BN;               // 0..7
    const int b_smem_col = b_flat % BN;               // 0,4,8,...,124
    const int b_global_col = block_col + b_smem_col;  // 不随 tile 变化

    const int num_tiles = (k + BK - 1) / BK;

    // =========== 加载第 0 个 tile 到 buffer 0 ===========
    {
        const int gc = a_smem_col;                    // k_offset = 0
        if (a_global_row < m && gc + 3 < k) {
            float4 tmp = *reinterpret_cast<const float4*>(
                &A[static_cast<size_t>(a_global_row) * k + gc]);
            As[0][a_smem_row][a_smem_col    ] = tmp.x;
            As[0][a_smem_row][a_smem_col + 1] = tmp.y;
            As[0][a_smem_row][a_smem_col + 2] = tmp.z;
            As[0][a_smem_row][a_smem_col + 3] = tmp.w;
        } else {
            for (int i = 0; i < 4; ++i) {
                int r = a_global_row, c = gc + i;
                As[0][a_smem_row][a_smem_col + i] =
                    (r < m && c < k) ? A[static_cast<size_t>(r) * k + c] : 0.f;
            }
        }
    }
    {
        const int gr = b_smem_row;                    // k_offset = 0
        if (gr < k && b_global_col + 3 < n) {
            float4 tmp = *reinterpret_cast<const float4*>(
                &B[static_cast<size_t>(gr) * n + b_global_col]);
            Bs[0][b_smem_row][b_smem_col    ] = tmp.x;
            Bs[0][b_smem_row][b_smem_col + 1] = tmp.y;
            Bs[0][b_smem_row][b_smem_col + 2] = tmp.z;
            Bs[0][b_smem_row][b_smem_col + 3] = tmp.w;
        } else {
            for (int i = 0; i < 4; ++i) {
                int r = gr, c = b_global_col + i;
                Bs[0][b_smem_row][b_smem_col + i] =
                    (r < k && c < n) ? B[static_cast<size_t>(r) * n + c] : 0.f;
            }
        }
    }
    __syncthreads();

    // =========== 主循环：软件流水线 ===========
    //
    //  对于每一个 tile:
    //    ① 发射下一个 tile 的 LDG → 寄存器 (硬件异步)
    //    ② 用当前 smem buffer 做 FMA 计算
    //    ③ 把寄存器中的数据 STS → 另一组 smem buffer
    //    ④ __syncthreads()
    //
    //  这样 ① 的 LDG 延迟被 ② 的 FMA 计算所掩盖。
    //

    // 预取寄存器
    float a_prefetch[4];
    float b_prefetch[4];

    for (int tile = 0; tile < num_tiles; ++tile) {
        const int buf      = tile & 1;
        const int next_buf = buf ^ 1;
        const int next_k   = (tile + 1) * BK;
        const bool has_next = (next_k < k);

        // ---- ① 发射下一个 tile 的 LDG（异步，落入寄存器）----
        if (has_next) {
            const int gc = next_k + a_smem_col;
            if (a_global_row < m && gc + 3 < k) {
                float4 tmp = *reinterpret_cast<const float4*>(
                    &A[static_cast<size_t>(a_global_row) * k + gc]);
                a_prefetch[0] = tmp.x;
                a_prefetch[1] = tmp.y;
                a_prefetch[2] = tmp.z;
                a_prefetch[3] = tmp.w;
            } else {
                for (int i = 0; i < 4; ++i) {
                    int r = a_global_row, c = gc + i;
                    a_prefetch[i] = (r < m && c < k)
                        ? A[static_cast<size_t>(r) * k + c] : 0.f;
                }
            }

            const int gr = next_k + b_smem_row;
            if (gr < k && b_global_col + 3 < n) {
                float4 tmp = *reinterpret_cast<const float4*>(
                    &B[static_cast<size_t>(gr) * n + b_global_col]);
                b_prefetch[0] = tmp.x;
                b_prefetch[1] = tmp.y;
                b_prefetch[2] = tmp.z;
                b_prefetch[3] = tmp.w;
            } else {
                for (int i = 0; i < 4; ++i) {
                    int r = gr, c = b_global_col + i;
                    b_prefetch[i] = (r < k && c < n)
                        ? B[static_cast<size_t>(r) * n + c] : 0.f;
                }
            }
        }

        // ---- ② 用当前 buffer 做 FMA 计算 ----
#pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            float a_vec[TM];
            float b_vec[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i)
                a_vec[i] = As[buf][ty * TM + i][kk];
#pragma unroll
            for (int j = 0; j < TN; ++j)
                b_vec[j] = Bs[buf][kk][tx * TN + j];
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j)
                    creg[i][j] += a_vec[i] * b_vec[j];
        }

        // ---- ③ 把预取寄存器 STS → 下一组 smem buffer ----
        if (has_next) {
            As[next_buf][a_smem_row][a_smem_col    ] = a_prefetch[0];
            As[next_buf][a_smem_row][a_smem_col + 1] = a_prefetch[1];
            As[next_buf][a_smem_row][a_smem_col + 2] = a_prefetch[2];
            As[next_buf][a_smem_row][a_smem_col + 3] = a_prefetch[3];

            Bs[next_buf][b_smem_row][b_smem_col    ] = b_prefetch[0];
            Bs[next_buf][b_smem_row][b_smem_col + 1] = b_prefetch[1];
            Bs[next_buf][b_smem_row][b_smem_col + 2] = b_prefetch[2];
            Bs[next_buf][b_smem_row][b_smem_col + 3] = b_prefetch[3];
        }

        // ---- ④ 同步：确保当前计算完毕 + 下一 tile 数据写入完毕 ----
        __syncthreads();
    }

    // =========== 用 float4 向量化写回 C ===========
#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int r = block_row + ty * TM + i;
        const int c = block_col + tx * TN;
        if (r < m && c + TN - 1 < n) {
            *reinterpret_cast<float4*>(&C[static_cast<size_t>(r) * n + c]) =
                make_float4(creg[i][0], creg[i][1], creg[i][2], creg[i][3]);
            *reinterpret_cast<float4*>(&C[static_cast<size_t>(r) * n + c + 4]) =
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
  编译（CUDA 12.6 + VS 2022，按你的实际路径修改）：

  & "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\nvcc.exe" ^
    win_naive_multiply_kernel_3_dot_8.cu -o naive_win_3_dot_8.exe ^
    -ccbin "...\MSVC\...\Hostx64\x64" ^
    -std=c++14 -Xcompiler "/utf-8 /EHsc /O2 /MD" -arch=sm_75
*/
