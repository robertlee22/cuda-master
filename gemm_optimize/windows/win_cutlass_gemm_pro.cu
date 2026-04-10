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

    int M = 4096*2;
    int N = 4096*2;
    int K = 1024*2;

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
  RTX 2060 Max-Q（Turing / sm_75）示例编译：
  &  "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\nvcc.exe" win_cutlass_gemm_pro.cu -o win_cutlass_gemm_pro.exe `
    -I"C:\Users\primelee\Desktop\cutlass\include" -std=c++17 -arch=sm_75 `
    -ccbin "...\MSVC\...\Hostx64\x64" `
    -Xcompiler "/utf-8 /EHsc /O2 /MD /Zc:preprocessor" -allow-unsupported-compiler -Xptxas -O3

 & "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\nvcc.exe" win_cutlass_gemm_pro.cu -o win_cutlass_gemm_pro.exe -I"c:\Users\primelee\Desktop\cutlass\include" -std=c++17 -arch=sm_75 -ccbin "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.50.35717\bin\Hostx64\x64" -Xcompiler "/utf-8 /EHsc /O2 /MD /Zc:preprocessor" -allow-unsupported-compiler -Xptxas -O3

*/