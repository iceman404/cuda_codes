#include <iostream>
#include <cuda.h>

#define N 10

// CUDA kernel for matrix multiplication
__global__ void matrixMultiplication(int* a, int* b, int* c)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        int sum = 0;
        for (int i = 0; i < N; ++i) {
            sum += a[row * N + i] * b[i * N + col];
        }
        c[row * N + col] = sum;
    }
}

int main()
{
    int a[N][N], b[N][N], c[N][N]; // Input and output matrices

    // Initialize input matrices
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            a[i][j] = i + j;
            b[i][j] = i - j;
        }
    }

    int* dev_a, * dev_b, * dev_c; // Device copies of input and output matrices

    // Allocate memory on the device
    cudaMalloc((void**)&dev_a, N * N * sizeof(int));
    cudaMalloc((void**)&dev_b, N * N * sizeof(int));
    cudaMalloc((void**)&dev_c, N * N * sizeof(int));

    // Copy input matrices from host to device
    cudaMemcpy(dev_a, a, N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, N * N * sizeof(int), cudaMemcpyHostToDevice);

    // Define grid and block dimensions for CUDA kernel
    dim3 threadsPerBlock(N, N);
    dim3 blocksPerGrid(1, 1);

    // Launch the CUDA kernel
    matrixMultiplication<<<blocksPerGrid, threadsPerBlock>>>(dev_a, dev_b, dev_c);

    // Copy the result back from the device to the host
    cudaMemcpy(c, dev_c, N * N * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the output matrix
    std::cout << "Matrix C (Result):" << std::endl;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << c[i][j] << " ";
        }
        std::cout << std::endl;
    }

    // Free device memory
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;
}
