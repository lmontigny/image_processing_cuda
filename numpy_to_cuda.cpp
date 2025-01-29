#include <iostream>
#include <cmath>

__global__ void matrixMul(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

__global__ void sigmoidTransform(float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N * N) {
        C[idx] = 1.0f / (1.0f + expf(-C[idx]));
    }
}

int main() {
    int N = 2;
    float h_A[4] = {4, 2, 7, 5};
    float h_B[4] = {8, 6, 9, 1};
    float h_C[4];

    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, N * N * sizeof(float));
    cudaMalloc((void**)&d_B, N * N * sizeof(float));
    cudaMalloc((void**)&d_C, N * N * sizeof(float));

    cudaMemcpy(d_A, h_A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrixMul<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    sigmoidTransform<<<(N * N + 255) / 256, 256>>>(d_C, N);

    cudaMemcpy(h_C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N * N; ++i) {
        std::cout << h_C[i] << " ";
    }
    std::cout << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}

// Python code, numpy
/*
import numpy as np
def sigmoid(x):
return 1 / (1 + np.exp(-x))
def matrix_operations(A, B):
# Perform matrix multiplication
C = np.dot(A, B)
# Apply element-wise transformation: sigmoid function
C_transformed = sigmoid(C)
return C_transformed
A = [[4, 2], [7, 5]]
B = [[8, 6], [9, 1]]
*/
