#include <cuda_runtime.h>
#include <iostream>
using namespace std; 

// A shape = (M,K) , B shape = (K,N)
// C shape = (M,N)
int M = 1000
int K = 100
int N = 1000

__global__ void multiply(int *A, int *B, int *C, int M, int K, int N){
    // C implementation 
    // for(int i = 0; i<M, i++){
    //     for (int j = 0; j<N; j++){
    //         int sum = 0; 
    //         for(int k =0; k< K; k++){
    //             sum += A[ i*K+k ] * B[k*K + j]
    //         }
    //         C[ i*N +j ] = sum; 
    //     }
    // }

    // exec id  mi mj 的任务含有什么？
    // mi = blockIdx.x 
    // mj = threadIdx.x 
    
    //gi ,  gi * gridDim.x < M 
    // gj , gj * blockDim.x < N
    // row = mi + gi * gridDim.x
    // col = gj*blockDim.x *  + mj
    // C[row * N + col ] = sum 
    // k,  sum += A[ row*K + k] * B[k*K + col ]
    
    
    // cuda implementation 
    int mi = blockIdx.x ; 
    int mj = threadIdx.x ; 
    
    for(int gi=0; gi*gridDim.x < M; gi++){
        for(int gj=0; gj*blockDim.x < N; gj++){
            int sum = 0; 
            row = mi + gi*gridDim.x 
            col = gj*blockDim.x + mj
            for(int k = 0; k<K; k++){
                sum += A[row* K + k] * B[k*K + col]
            }
            C[row* N + col] = sum; 
        }
    }
    // compute object C[i][j] 
}

int main() {

    int *A, *B, *C 
    
    cudaMallocManaged(&A, sizeof(int) * M*K); 
    cudaMallocManaged(&B, sizeof(int) * K*N); 
    cudaMallocManaged(&C, sizeof(int)* N*M); 

    for(int i = 0; i< M*K; i++){
        A[i] = 1; 
    }
    for(int i =0; i<K*N; i++){
        B[i] = 1; 

    }

    //mem prefetch

    //kernel multiply
    multiply<<<16, 16>>>(A,B,C, M,K,N); 

    //check result 
    bool pass = true ;
    for(int i = 0; i< M*N; i++){
        if(C[i]!=100){
            pass = false ;
        }
    }

    if(pass){
        cout<< "pass!" <<endl; 
    }else{
        cout<< "error!" <<endl; 
    }

    cudaFree(A); 
    cudaFree(B); 
    cudaFree(C); 
    return 0 ; 
}