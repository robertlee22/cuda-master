#include <cuda_runtime.h>

#include <iostream>
using namespace std;

int V1M = 1000000;

__global__ void add(int n, float *x, float *y, float *sum){
    int idx = blockIdx.x * blockDim.x + threadIdx.x ; 
    int STRIDE = blockDim.x * gridDim.x ;
    for (int i = idx; i < n; i+= STRIDE){
        sum[i] = x[i] + y[i]; 
    }
}

int main(){
    std::cout<< "hello cuda"<< std::endl;
    
    float *x , *y ,*sum ; 
    cudaMallocManaged(&x, V1M*sizeof(float));
    cudaMallocManaged(&y, V1M*sizeof(float));

    for(int i = 0; i<V1M; i++){
        x[i] = 1.0f; 
        y[i] = 2.0f; 
    }

    // kernel add 

    int block_dim = 256; 
    int grid_dim = (V1M + block_dim - 1) / block_dim ; 
    add<<<grid_dim, block_dim  >>>(V1M, x, y, sum); 

    // check result 
    for (int i = 0; i<V1M; i++){
        float errMax = 1e-5f; 
        float err = abs(x[i] + y[i] - 3.0f);
        if (err > errMax){
            cout << "error at index " << i << ": " << x[i] + y[i] << endl; 
            break; 
        }
    }

    cudaFree(x); 
    cudaFree(y); 
}