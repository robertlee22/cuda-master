// CuTe GEMM，面向 NVIDIA RTX 2060 Max-Q（Turing，SM 7.5 / sm_75）
//
// 硬件要点（相对 Ampere+）：
// - 无 cp.async：CuTe 中 cp_async_fence / cp_async_wait 在 __CUDA_ARCH__ < 800 下为空操作；
//   copy() 默认走 AutoVectorizingCopy（全局→共享的向量化标量路径），与 Turing 匹配。
// - FP32 矩阵乘不走 Tensor Core；本文件仍为 float + CuTe 的 SIMT gemm，通过更大 K 分块与设备内存降低开销。
// - 2060 Max-Q 共享内存与寄存器资源有限：bK 过大易降占用，故在 bK=32 与共享约 32KiB/块 之间折中。

#include <cmath>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
#include <iostream>

#include <cute/tensor.hpp>

using namespace cute;

#define CHECK_CUDA(err)                                                                                \
    do {                                                                                               \
        cudaError_t e__ = (err);                                                                       \
        if (e__ != cudaSuccess) {                                                                     \
            std::cerr << "CUDA: " << cudaGetErrorString(e__) << " @ " << __FILE__ << ":" << __LINE__   \
                      << std::endl;                                                                    \
            std::exit(1);                                                                              \
        }                                                                                              \
    } while (0)

template <class ProblemShape, class CtaTiler, class TA, class AStride, class ASmemLayout, class AThreadLayout,
          class TB, class BStride, class BSmemLayout, class BThreadLayout, class TC, class CStride,
          class CSmemLayout, class CThreadLayout, class Alpha, class Beta>
__global__ static __launch_bounds__(decltype(size(CThreadLayout{}))::value) void gemm_device(
    ProblemShape shape_MNK, CtaTiler cta_tiler, TA const* A, AStride dA, ASmemLayout sA_layout,
    AThreadLayout tA, TB const* B, BStride dB, BSmemLayout sB_layout, BThreadLayout tB, TC* C, CStride dC,
    CSmemLayout, CThreadLayout tC, Alpha alpha, Beta beta) {
    Tensor mA = make_tensor(make_gmem_ptr(A), select<0, 2>(shape_MNK), dA);
    Tensor mB = make_tensor(make_gmem_ptr(B), select<1, 2>(shape_MNK), dB);
    Tensor mC = make_tensor(make_gmem_ptr(C), select<0, 1>(shape_MNK), dC);
    auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);

    Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{});
    Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step<X, _1, _1>{});
    Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1, X>{});

    __shared__ TA smemA[cosize_v<ASmemLayout>];
    __shared__ TB smemB[cosize_v<BSmemLayout>];

    Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout);
    Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout);

    Tensor tAgA = local_partition(gA, tA, threadIdx.x);
    Tensor tAsA = local_partition(sA, tA, threadIdx.x);
    Tensor tBgB = local_partition(gB, tB, threadIdx.x);
    Tensor tBsB = local_partition(sB, tB, threadIdx.x);

    Tensor tCsA = local_partition(sA, tC, threadIdx.x, Step<_1, X>{});
    Tensor tCsB = local_partition(sB, tC, threadIdx.x, Step<X, _1>{});
    Tensor tCgC = local_partition(gC, tC, threadIdx.x, Step<_1, _1>{});

    Tensor tCrC = make_tensor_like(tCgC);
    clear(tCrC);

    auto K_TILE_MAX = size<2>(tAgA);
    for (int k_tile = 0; k_tile < K_TILE_MAX; ++k_tile) {
        copy(tAgA(_, _, k_tile), tAsA);
        copy(tBgB(_, _, k_tile), tBsB);
        cp_async_fence();
        cp_async_wait<0>();
        __syncthreads();

        gemm(tCsA, tCsB, tCrC);
        __syncthreads();
    }
    axpby(alpha, tCrC, beta, tCgC);
}

int main() {
    // 与 win_cute_gemm.cu 相同规模，便于对比；K 须为 bK 整数倍（此处 bK=32，2048 整除）
    constexpr int M = 4096 * 2;
    constexpr int N = 4096 * 2;
    constexpr int K = 1024 * 2;

    CHECK_CUDA(cudaSetDevice(0));

    cudaDeviceProp prop{};
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    std::cout << "GPU: " << prop.name << "  SM " << prop.major << "." << prop.minor
              << "  [build with -arch=sm_75 for RTX 2060 series]" << std::endl;

    auto probShape = cute::make_shape(int(M), int(N), int(K));

    float* dA = nullptr;
    float* dB = nullptr;
    float* dC = nullptr;
    CHECK_CUDA(cudaMalloc(&dA, sizeof(float) * static_cast<size_t>(M) * K));
    CHECK_CUDA(cudaMalloc(&dB, sizeof(float) * static_cast<size_t>(N) * K));
    CHECK_CUDA(cudaMalloc(&dC, sizeof(float) * static_cast<size_t>(M) * N));

    std::vector<float> hA(static_cast<size_t>(M) * K, 1.0f);
    std::vector<float> hB(static_cast<size_t>(N) * K, 1.0f);
    std::vector<float> hC(static_cast<size_t>(M) * N, 0.0f);

    CHECK_CUDA(cudaMemcpy(dA, hA.data(), sizeof(float) * hA.size(), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB.data(), sizeof(float) * hB.size(), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dC, hC.data(), sizeof(float) * hC.size(), cudaMemcpyHostToDevice));

    // Turing 上较优经验配置：128³ 量级 CTA，K 维一次 32，减少 K 外层迭代（相对 bK=8 约 4×）
    auto bM = Int<128>{};
    auto bN = Int<128>{};
    auto bK = Int<32>{};
    auto cta_tile = cute::make_shape(bM, bN, bK);

    auto sA = make_layout(make_shape(bM, bK));
    auto sB = make_layout(make_shape(bN, bK));
    auto sC = make_layout(make_shape(bM, bN));

    auto tA = cute::make_layout(cute::make_shape(Int<32>{}, Int<8>{}));
    auto tB = cute::make_layout(cute::make_shape(Int<32>{}, Int<8>{}));
    auto tC = cute::make_layout(cute::make_shape(Int<16>{}, Int<16>{}));

    dim3 dimBlock(cute::size(tC));
    dim3 dimGrid(cute::size(cute::ceil_div(int(M), bM)), cute::size(cute::ceil_div(int(N), bN)));

    auto dA_stride = cute::make_stride(Int<1>{}, int(M));
    auto dB_stride = cute::make_stride(Int<1>{}, int(N));
    auto dC_stride = cute::make_stride(Int<1>{}, int(M));

    // 预热：稳定频率与缓存
    gemm_device<<<dimGrid, dimBlock>>>(probShape, cta_tile, dA, dA_stride, sA, tA, dB, dB_stride, sB, tB, dC,
                                       dC_stride, sC, tC, 1.0f, 0.0f);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t t0{}, t1{};
    CHECK_CUDA(cudaEventCreate(&t0));
    CHECK_CUDA(cudaEventCreate(&t1));
    const int bench_iters = 5;
    CHECK_CUDA(cudaEventRecord(t0));
    for (int it = 0; it < bench_iters; ++it) {
        gemm_device<<<dimGrid, dimBlock>>>(probShape, cta_tile, dA, dA_stride, sA, tA, dB, dB_stride, sB, tB, dC,
                                           dC_stride, sC, tC, 1.0f, 0.0f);
    }
    CHECK_CUDA(cudaEventRecord(t1));
    CHECK_CUDA(cudaEventSynchronize(t1));
    float ms_total = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms_total, t0, t1));
    CHECK_CUDA(cudaGetLastError());
    std::cout << "CuTe GEMM avg time (" << bench_iters << " runs): " << (ms_total / static_cast<float>(bench_iters))
              << " ms" << std::endl;
    CHECK_CUDA(cudaEventDestroy(t0));
    CHECK_CUDA(cudaEventDestroy(t1));

    CHECK_CUDA(cudaMemcpy(hC.data(), dC, sizeof(float) * hC.size(), cudaMemcpyDeviceToHost));

    const float expected = static_cast<float>(K);
    bool ok = true;
    for (size_t i = 0; i < hC.size(); ++i) {
        if (std::fabs(hC[i] - expected) > 1e-3f) {
            ok = false;
            if (i < 4) {
                std::cerr << "C[" << i << "] = " << hC[i] << "  expect " << expected << std::endl;
            }
            break;
        }
    }
    std::cout << (ok ? "pass!" : "not pass!") << std::endl;

    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));
    return ok ? 0 : 1;
}

/*
  RTX 2060 Max-Q（Turing / sm_75）示例编译：
  & "...\CUDA\v13.1\bin\nvcc.exe" win_cute_gemm_2.cu -o win_cute_gemm_2.exe `
    -I"C:\Users\primelee\Desktop\cutlass\include" -std=c++17 -arch=sm_75 `
    -ccbin "...\MSVC\...\Hostx64\x64" `
    -Xcompiler "/utf-8 /EHsc /O2 /MD /Zc:preprocessor" -allow-unsupported-compiler -Xptxas -O3
*/
