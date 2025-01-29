/*
N=4 images
NCHW channel first -> RGB, RGB, RGB, RGB
NHWC channel last
RBG, 3 matrix 

img 3*1k*1k
4 img 4*3*1k*1k

b [0,3]
z [0,11]
c [0,2]
------ ------ ------
bacth index
*/


__global__ void squareImage2D(float* image, int width, int height, int dimz) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    int b = z/3;
    int c = z%3;

    if (x < width && y < height && z < dimz) {
        //int idx = y * width + x;
        int idx = b*width*height*3;
        int row = y * width * c;
        int col = x * c + c;
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
    dim3 threadsPerBlock(16, 16, 1);
    int dimz = N*C;
    dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (height + threadsPerBlock.y - 1) / threadsPerBlock.y,
                        dimz);

    // Launch the kernel
    squareImage2D<<<blocksPerGrid, threadsPerBlock>>>(d_image, width, height, dimz);

    // Copy result back to host
    cudaMemcpy(h_image, d_image, width * height * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_image);

    // Free host memory
    free(h_image);

    return 0;
}
