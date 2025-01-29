__global__ void matrixMulShared(float* A, float* B, float* C, int N) {
    // Define shared memory for tiles of A and B
    __shared__ float sharedA[16][16];
    __shared__ float sharedB[16][16];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;

    // Loop over tiles of the input matrices
    for (int t = 0; t < (N + 15) / 16; ++t) {
        // Load elements into shared memory
        if (row < N && t * 16 + threadIdx.x < N) {
            sharedA[threadIdx.y][threadIdx.x] = A[row * N + t * 16 + threadIdx.x];
        } else {
            sharedA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (col < N && t * 16 + threadIdx.y < N) {
            sharedB[threadIdx.y][threadIdx.x] = B[(t * 16 + threadIdx.y) * N + col];
        } else {
            sharedB[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();  // Synchronize to ensure all threads have loaded their data

        // Compute the partial sum for this tile
        for (int k = 0; k < 16; ++k) {
            sum += sharedA[threadIdx.y][k] * sharedB[k][threadIdx.x];
        }

        __syncthreads();  // Synchronize before loading the next tile
    }

    // Write the result to the output matrix
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}
