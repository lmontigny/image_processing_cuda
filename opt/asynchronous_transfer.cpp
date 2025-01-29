int main() {
    int N = 2;
    float h_A[4] = {4, 2, 7, 5};
    float h_B[4] = {8, 6, 9, 1};
    float h_C[4];

    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, N * N * sizeof(float));
    cudaMalloc((void**)&d_B, N * N * sizeof(float));
    cudaMalloc((void**)&d_C, N * N * sizeof(float));

    // Create a CUDA stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Asynchronously copy data to the device
    cudaMemcpyAsync(d_A, h_A, N * N * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_B, h_B, N * N * sizeof(float), cudaMemcpyHostToDevice, stream);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch the kernel in the same stream
    matrixMulShared<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_A, d_B, d_C, N);
    sigmoidTransform<<<(N * N + 255) / 256, 256, 0, stream>>>(d_C, N);

    // Asynchronously copy the result back to the host
    cudaMemcpyAsync(h_C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost, stream);

    // Synchronize the stream to ensure all operations are complete
    cudaStreamSynchronize(stream);

    for (int i = 0; i < N * N; ++i) {
        std::cout << h_C[i] << " ";
    }
    std::cout << std::endl;

    // Clean up
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaStreamDestroy(stream);

    return 0;
}
