#include <cuda_runtime.h>
#include <iostream>
using namespace std; 

// A shape = (M,K) , B shape = (K,N)
// C shape = (M,N)
int M = 4096*2;
int N = 4096*2;
int K = 1024*2;

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
            int row = mi + gi*gridDim.x ;
            int col = gj*blockDim.x + mj;
            for(int k = 0; k<K; k++){
                sum += A[row* K + k] * B[k*N + col];
            }
            C[row* N + col] = sum; 
        }
    }
    // compute object C[i][j] 
}

int main() {

    int *A; 
    int *B; 
    int *C ;
    
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

    // cudaMemPrefetchAsync(A, sizeof(int)* M*K, 0, 0);
    // cudaMemPrefetchAsync(B, sizeof(int)* K*N, 0, 0);
    // cudaMemPrefetchAsync(C, sizeof(int)* M*N, 0, 0);

    //kernel multiply
    multiply<<<128, 256>>>(A,B,C, M,K,N); 
    cudaDeviceSynchronize(); 

    //check result 
    bool pass = true ;
    for(int i = 0; i< M*N; i++){
        if(C[i]!=K){
            cout << "C[" << i << "] = " << C[i] << endl;
            pass = false ;
            break;
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