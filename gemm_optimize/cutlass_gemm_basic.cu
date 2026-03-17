#include <iostream>
#include <cuda_runtime.h>

#include <cutlass/gemm/device/gemm.h>

int main() {

    int M = 256;
    int N = 256;
    int K = 256;

    float *A, *B, *C;

    cudaMallocManaged(&A, M*K*sizeof(float));
    cudaMallocManaged(&B, K*N*sizeof(float));
    cudaMallocManaged(&C, M*N*sizeof(float));

    for(int i=0;i<M*K;i++) A[i] = 1.0f;
    for(int i=0;i<K*N;i++) B[i] = 1.0f;

    using Gemm = cutlass::gemm::device::Gemm<
        float,
        cutlass::layout::RowMajor,
        float,
        cutlass::layout::RowMajor,
        float,
        cutlass::layout::RowMajor
    >;

    Gemm gemm_op;

    cutlass::Status status = gemm_op({
        {M,N,K},
        {A,K},
        {B,N},
        {C,N},
        {C,N},
        {1.0f,0.0f}
    });

    if(status != cutlass::Status::kSuccess){
        std::cout<<"GEMM failed\n";
        return -1;
    }

    cudaDeviceSynchronize();

    std::cout<<"C[0] = "<<C[0]<<std::endl;

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    return 0;
}