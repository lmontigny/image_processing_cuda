__global__ void squareImage2D(float* image, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        image[idx] = image[idx] * image[idx];
    }
}

int main() {
    int width = 1024;
    int height = 1024;

    // Allocate and initialize the image array
    float* h_image = (float*)malloc(width * height * sizeof(float));
    // Initialize h_image with some values...

    // Allocate device memory
    float* d_image;
    cudaMalloc((void**)&d_image, width * height * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_image, h_image, width * height * sizeof(float), cudaMemcpyHostToDevice);

    // Kernel launch configuration
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch the kernel
    squareImage2D<<<blocksPerGrid, threadsPerBlock>>>(d_image, width, height);

    // Copy result back to host
    cudaMemcpy(h_image, d_image, width * height * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_image);

    // Free host memory
    free(h_image);

    return 0;
}
