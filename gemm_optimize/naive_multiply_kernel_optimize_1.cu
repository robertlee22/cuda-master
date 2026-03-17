#include <cuda_runtime.h>
#include <iostream>
using namespace std; 

// A shape = (M,K) , B shape = (K,N)
// C shape = (M,N)
int M = 4096*4;
int N = 4096*4;
int K = 1024*4;

#define TILE 32

__global__ void multiply(int *A, int *B, int *C, int M, int K, int N){
    // cuda implementation 
    int col = threadIdx.x + blockDim.x* blockIdx.x ; 
    int row =  threadIdx.y + blockDim.y * blockIdx.y; 
    
    if(col < N && row < M){
        int sum = 0; 
       
        for(int k = 0; k<K; k++){
            sum += A[row* K + k] * B[k*N + col];
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
    dim3 block(TILE, TILE); 
    dim3 grid((N+TILE-1)/TILE, (M+TILE-1)/TILE);
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