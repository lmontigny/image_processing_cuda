__global__ void squareImage1D(float* image, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        image[idx] = image[idx] * image[idx];
    }
}

int main() {
    int width = 1024;
    int height = 1024;
    int size = width * height;

    // Allocate and initialize the image array
    float* h_image = (float*)malloc(size * sizeof(float));
    // Initialize h_image with some values...

    // Allocate device memory
    float* d_image;
    cudaMalloc((void**)&d_image, size * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_image, h_image, size * sizeof(float), cudaMemcpyHostToDevice);

    // Kernel launch configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel
    squareImage1D<<<blocksPerGrid, threadsPerBlock>>>(d_image, size);

    // Copy result back to host
    cudaMemcpy(h_image, d_image, size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_image);

    // Free host memory
    free(h_image);

    return 0;
}
