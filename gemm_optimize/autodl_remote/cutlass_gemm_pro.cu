#include <iostream>
#include <cuda_runtime.h>

#include <cutlass/gemm/device/gemm.h>
#include <cutlass/half.h>


#define CHECK_CUDA(call)                                     \
{                                                            \
    cudaError_t err = call;                                  \
    if (err != cudaSuccess) {                                \
        std::cerr << "CUDA Error: "                          \
                  << cudaGetErrorString(err) << std::endl;   \
        exit(1);                                             \
    }                                                        \
}

int main() {

    int M = 4096*4;
    int N = 4096*4;
    int K = 1024*4;

    using Element = cutlass::half_t;
    Element *A, *B, *C;

    

    CHECK_CUDA(cudaMalloc(&A, M * K * sizeof(Element)));
    CHECK_CUDA(cudaMalloc(&B, K * N * sizeof(Element)));
    CHECK_CUDA(cudaMalloc(&C, M * N * sizeof(Element)));

        // host memory
    Element *hA = new Element[M * K];
    Element *hB = new Element[K * N];

   for (int i = 0; i < M * K; i++) hA[i] = Element(1.0f);
    for (int i = 0; i < K * N; i++) hB[i] = Element(1.0f);

    CHECK_CUDA(cudaMemcpy(A, hA, M * K * sizeof(Element), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(B, hB, K * N * sizeof(Element), cudaMemcpyHostToDevice));

    using Gemm = cutlass::gemm::device::Gemm<
        Element,
        cutlass::layout::RowMajor,
        Element,
        cutlass::layout::RowMajor,
        Element,
        cutlass::layout::RowMajor,
        float
    >;

    Gemm gemm_op;

    cutlass::gemm::GemmCoord problem_size(M, N, K);

    typename Gemm::Arguments args{
        problem_size,
        {A, K},
        {B, N},
        {C, N},
        {C, N},
        {1.0f, 0.0f}
    };

    cutlass::Status status = gemm_op(args);

    if(status != cutlass::Status::kSuccess){
        std::cout<<"GEMM failed\n";
        return -1;
    }

    cudaDeviceSynchronize();

    Element result;
    CHECK_CUDA(cudaMemcpy(&result, C, sizeof(Element), cudaMemcpyDeviceToHost));
    std::cout << "C[0] = " << float(result) << std::endl;

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    return 0;
}

/*
nvcc  -O3 -arch=sm_90 -I ~/cuda/cutlass/include cutlass_demo.cu -o cutlass_gemm.out
nvcc  -O3 -I ~/cuda/cutlass/include cutlass_demo.cu -o cutlass_gemm.out

*/