#include <cuda_runtime.h>
#include <iostream>
using namespace std; 

// A shape = (M,K) , B shape = (K,N)
// C shape = (M,N)
int M = 4096;
int N = 4096;
int K = 1024;

#define TILE 16

__global__ void multiply(int *A, int *B, int *C, int M, int K, int N){
    // cuda implementation 
    int col = threadIdx.x + blockDim.x* blockIdx.x ; 
    int row =  threadIdx.y + blockDim.y * blockIdx.y; 
    
    if(col < N && row < M){
        int sum = 0; 

        __shared__ int As[TILE][TILE]; 
        __shared__ int Bs[TILE][TILE]; 
       
        for(int t = 0; t * TILE < K; t++){
            // fetch to tile , sync ; 
            int ax = t*TILE + threadIdx.x ; 
            int by = t*TILE + threadIdx.y ; 

            if(row < M && ax < K){
                As[threadIdx.y][threadIdx.x] = A[row*K + ax]; 
            }else{
                As[threadIdx.y][threadIdx.x] = 0;
            }

            if(by < K && col < N){
                Bs[threadIdx.y][threadIdx.x] = B[by*N + col]; 
            }else{
                Bs[threadIdx.y][threadIdx.x] = 0;
            }
            
            __syncthreads() ; 
            
            // compute 
            for(int k=0; k< TILE; k++){
                sum += As[threadIdx.y][k]*Bs[k][threadIdx.x]; 
            }
            __syncthreads() ;
        }
        C[row* N + col] = sum; 
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

    cudaMemPrefetchAsync(A, sizeof(int)* M*K, 0, 0);
    cudaMemPrefetchAsync(B, sizeof(int)* K*N, 0, 0);
    cudaMemPrefetchAsync(C, sizeof(int)* M*N, 0, 0);

    //kernel multiply
    dim3 block(16,16); 
    dim3 grid((N+15)/16, (M+16)/16);
    multiply<<<grid, block>>>(A,B,C, M,K,N); 
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