// cutlass-main/include
#include <iostream> 
#include <cute/tensor.hpp>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>


using namespace cute;

#define CHECK_CUDA_ERROR(err) \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA错误: " << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    }

// copy from official template 
template <class ProblemShape, class CtaTiler,
          class TA, class AStride, class ASmemLayout, class AThreadLayout,
          class TB, class BStride, class BSmemLayout, class BThreadLayout,
          class TC, class CStride, class CSmemLayout, class CThreadLayout,
          class Alpha, class Beta>
__global__ static
__launch_bounds__(decltype(size(CThreadLayout{}))::value)
void
gemm_device(ProblemShape shape_MNK, CtaTiler cta_tiler,
            TA const* A, AStride dA, ASmemLayout sA_layout, AThreadLayout tA,
            TB const* B, BStride dB, BSmemLayout sB_layout, BThreadLayout tB,
            TC      * C, CStride dC, CSmemLayout          , CThreadLayout tC,
            Alpha alpha, Beta beta)
{
    
    Tensor mA = make_tensor(make_gmem_ptr(A), select<0,2>(shape_MNK), dA); // (M,K)
    Tensor mB = make_tensor(make_gmem_ptr(B), select<1,2>(shape_MNK), dB); // (N,K)
    Tensor mC = make_tensor(make_gmem_ptr(C), select<0,1>(shape_MNK), dC); // (M,N)
    auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);  

    Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});  // (BLK_M,BLK_K,k)
    Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{});  // (BLK_N,BLK_K,k)
    Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{});  // (BLK_M,BLK_N)

    __shared__ TA smemA[cosize_v<ASmemLayout>]; 
    __shared__ TB smemB[cosize_v<BSmemLayout>];

    Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout); // (BLK_M, BLK_K)
    Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout) ;// (BLK_N, BLK_K)

    Tensor tAgA = local_partition(gA, tA , threadIdx.x); // (THR_M=4, THR_K=1, k)
    Tensor tAsA = local_partition(sA, tA, threadIdx.x); // (THR_M=4， THR_K=1 )

    Tensor tBgB = local_partition(gB, tB , threadIdx.x) ;
    Tensor tBsB = local_partition(sB, tB , threadIdx.x) ;

    Tensor tCsA = local_partition(sA, tC, threadIdx.x, Step<_1, X>{}); //(THR_M = 8,BLK_K = 8)
    Tensor tCsB = local_partition(sB, tC, threadIdx.x, Step<X, _1>{}); //(THR_M = 8,BLK_K =8 )
    Tensor tCgC = local_partition(gC, tC, threadIdx.x, Step<_1,_1>{});   // (THR_M=8,THR_N=8)

    Tensor tCrC = make_tensor_like(tCgC);

    clear(tCrC);

    auto K_TILE_MAX = size<2>(tAgA);

    for (int k_tile=0 ; k_tile < K_TILE_MAX ; ++k_tile){
        copy(tAgA(_,_,k_tile), tAsA); 
        copy(tBgB(_,_,k_tile), tBsB); 

        cp_async_fence();        // Label the end of (potential) cp.async instructions
        cp_async_wait<0>();      // Sync on all (potential) cp.async instructions
        __syncthreads(); 


        gemm(tCsA , tCsB , tCrC); 


        __syncthreads(); 
    }

    axpby(alpha, tCrC, beta, tCgC);


}


int main() {
    int M = 4096 * 2; 
    int N = 4096 * 2; 
    int K = 1024 * 2; 

    // 问题规模可为动态 int；CTA tile 必须用 Int<…> 静态形状，与官方 cute_officail_gemm_1 一致
    auto probShape = cute::make_shape(int(M), int(N), int(K));

    float *A; 
    float *B; 
    float *C; 

    CHECK_CUDA_ERROR(cudaMallocManaged(&A, sizeof(float)*M*K)); 
    CHECK_CUDA_ERROR(cudaMallocManaged(&B, sizeof(float)*N*K)); 
    CHECK_CUDA_ERROR(cudaMallocManaged(&C, sizeof(float)*N*M)); 

    for (int i = 0; i<M*K; i++){
        A[i] = 1.0f; 
    }
    for (int i= 0; i<N*K; i++){
        B[i] = 1.0f; 
    }

    std::cout<< "init done." << std::endl;

    // determine stride 
    // determine smem layout , cta tile
    auto bM = Int<128>{};
    auto bN = Int<128>{};
    auto bK = Int<8>{};
    auto cta_tile = cute::make_shape(bM, bN, bK);

    auto sA = make_layout(make_shape(bM, bK));
    auto sB = make_layout(make_shape(bN, bK));
    auto sC = make_layout(make_shape(bM, bN));

    // thread layout 

    auto tA = cute::make_layout(cute::make_shape(Int<32>{}, Int<8>{})); 
    auto tB = cute::make_layout(cute::make_shape(Int<32>{}, Int<8>{})); 
    auto tC = cute::make_layout(cute::make_shape(Int<16>{}, Int<16>{})); 
    
    dim3 dimBlock(cute::size(tC));
    dim3 dimGrid(cute::size(cute::ceil_div(int(M), bM)),
                 cute::size(cute::ceil_div(int(N), bN)));

    auto dA = cute::make_stride(Int<1>{}, int(M));
    auto dB = cute::make_stride(Int<1>{}, int(N));
    auto dC = cute::make_stride(Int<1>{}, int(M)); 
    
    gemm_device<<<dimGrid, dimBlock>>>(
        probShape, cta_tile, 
        A, dA, sA, tA, 
        B, dB, sB , tB, 
        C, dC, sC, tC,
        1.0f, 0.0f
    );

    cudaDeviceSynchronize(); 

    bool isPass = true; 
    for (int i = 0; i<M*N; i++){
        
            if ( abs(C[i] - 1024.0f *2) > 1e-5 ) {
                isPass = false;
            }
    
    }

    if (isPass){
        std::cout<< "pass!"<<std::endl; 
    }else{
        std::cout<< "not pass!"<<std::endl; 
    }
    
    return 0; 
}
/*
  Windows 示例（PowerShell，路径按本机修改）：
  & "...\CUDA\v13.1\bin\nvcc.exe" win_cute_gemm.cu -o win_cute_gemm.exe `
    -I"C:\Users\primelee\Desktop\cutlass\include" -std=c++17 `
    -ccbin "...\MSVC\...\Hostx64\x64" `
    -Xcompiler "/utf-8 /EHsc /O2 /MD /Zc:preprocessor" -allow-unsupported-compiler -arch=native
*/