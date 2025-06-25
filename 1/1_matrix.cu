#include <stdio.h>

__global__ void transpose_gpu(float *matFrom, float *matTo, int rows, int columns)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < columns && row < rows)
        matTo[col * rows + row] = matFrom[row * columns + col];
}

__global__ void transpose_gpu_coalesced(float *matFrom, float *matTo, int rows, int columns)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int col = idx % columns;
    int row = idx / columns;

    if (col < columns && row < rows)
        matTo[col * rows + row] = matFrom[row * columns + col];
}

void transpose_cpu(float *matFrom, float *matTo, int rows, int columns)
{
    // Iterate over input matrix and populate output matrix
    for (int row = 0; row < rows; ++row)
        for (int col = 0; col < columns; ++col)
        {
            matTo[col * rows + row] = matFrom[row * columns + col];
        }
}

inline cudaError_t checkCuda(cudaError_t result)
{
    if (result != cudaSuccess)
    {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    }
    return result;
}

void checkAsyncError()
{
    cudaError_t syncErr = cudaGetLastError();
    if (syncErr != cudaSuccess)
    {
        printf("Kernel launch error: %s\n", cudaGetErrorString(syncErr));
    }
}

void verify(float *a, float *b, int elCount)
{
    for (int i = 0; i < elCount; i++)
    {
        if (a[i] != b[i])
        {
            printf("Error! Matrices don't match!");
            return;
        }
    }
}

int main()
{
    clock_t start, end;

    const int ROWS = 2048 * 16;
    const int COLUMNS = 1024 * 16;
    const size_t TOTAL_SIZE = sizeof(float) * ROWS * COLUMNS;

    float *inputMatrixReference = (float *)malloc(TOTAL_SIZE);
    float *outputMatrixReference = (float *)malloc(TOTAL_SIZE);

    // Init host matrix with random data
    for (int row = 0; row < ROWS; ++row)
    {
        for (int col = 0; col < COLUMNS; ++col)
        {
            inputMatrixReference[row * COLUMNS + col] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        }
    }

    {
        // Transpose on CPU
        start = clock();
        transpose_cpu(inputMatrixReference, outputMatrixReference, ROWS, COLUMNS);
        end = clock();
        printf("CPU transpose - %f seconds\n", double(end - start) / CLOCKS_PER_SEC);
    }

    {
        // Transpose on the GPU (no unified memory)
        start = clock();
        float *inputMatrixDevice;
        float *outputMatrixDevice;
        float *outputMatrixHost = (float *)malloc(TOTAL_SIZE);
        cudaMalloc(&inputMatrixDevice, TOTAL_SIZE);
        cudaMalloc(&outputMatrixDevice, TOTAL_SIZE);

        cudaMemcpy(inputMatrixDevice, inputMatrixReference, TOTAL_SIZE, cudaMemcpyHostToDevice);

        dim3 threads_per_block(16, 16, 1);
        dim3 number_of_blocks((COLUMNS + threads_per_block.x - 1) / threads_per_block.x, (ROWS + threads_per_block.y - 1) / threads_per_block.y, 1);
        transpose_gpu<<<number_of_blocks, threads_per_block>>>(inputMatrixDevice, outputMatrixDevice, ROWS, COLUMNS);
        cudaMemcpy(outputMatrixHost, outputMatrixDevice, TOTAL_SIZE, cudaMemcpyDeviceToHost);
        checkCuda(cudaDeviceSynchronize());
        end = clock();

        // Verify correctness & performance
        checkAsyncError();
        verify(outputMatrixHost, outputMatrixReference, ROWS * COLUMNS);
        printf("GPU transpose (no UM) - %f seconds\n", double(end - start) / CLOCKS_PER_SEC);

        // Free resources
        cudaFree(inputMatrixDevice);
        cudaFree(outputMatrixDevice);
        free(outputMatrixHost);
        checkCuda(cudaDeviceSynchronize());
    }

    {
        // Transpose on the GPU (unified memory)
        start = clock();
        float *inputMatrixDevice;
        float *outputMatrixDevice;
        cudaMallocManaged(&inputMatrixDevice, TOTAL_SIZE);
        cudaMallocManaged(&outputMatrixDevice, TOTAL_SIZE);

        cudaMemcpy(inputMatrixDevice, inputMatrixReference, TOTAL_SIZE, cudaMemcpyHostToDevice);

        dim3 threads_per_block(16, 16, 1);
        dim3 number_of_blocks((COLUMNS + threads_per_block.x - 1) / threads_per_block.x, (ROWS + threads_per_block.y - 1) / threads_per_block.y, 1);
        transpose_gpu<<<number_of_blocks, threads_per_block>>>(inputMatrixDevice, outputMatrixDevice, ROWS, COLUMNS);
        checkCuda(cudaDeviceSynchronize());
        end = clock();

        // Verify correctness & performance
        checkAsyncError();
        verify(outputMatrixDevice, outputMatrixReference, ROWS * COLUMNS);
        printf("GPU transpose (UM) - %f seconds\n", double(end - start) / CLOCKS_PER_SEC);

        // Free resources
        cudaFree(inputMatrixDevice);
        cudaFree(outputMatrixDevice);
        checkCuda(cudaDeviceSynchronize());
    }

    {
        // Transpose on the GPU (unified memory + coalesced memory access)
        start = clock();
        float *inputMatrixDevice;
        float *outputMatrixDevice;
        cudaMallocManaged(&inputMatrixDevice, TOTAL_SIZE);
        cudaMallocManaged(&outputMatrixDevice, TOTAL_SIZE);

        cudaMemcpy(inputMatrixDevice, inputMatrixReference, TOTAL_SIZE, cudaMemcpyHostToDevice);

        int threads_per_block = 128;
        int number_of_blocks = (ROWS * COLUMNS + threads_per_block - 1) / threads_per_block;
        transpose_gpu_coalesced<<<number_of_blocks, threads_per_block>>>(inputMatrixDevice, outputMatrixDevice, ROWS, COLUMNS);
        checkCuda(cudaDeviceSynchronize());
        end = clock();

        // Verify correctness & performance
        checkAsyncError();
        verify(outputMatrixDevice, outputMatrixReference, ROWS * COLUMNS);
        printf("GPU transpose (UM + coalesced) - %f seconds\n", double(end - start) / CLOCKS_PER_SEC);

        // Free resources
        cudaFree(inputMatrixDevice);
        cudaFree(outputMatrixDevice);
        checkCuda(cudaDeviceSynchronize());
    }

    {
        // Transpose on the GPU (no unified memory+ coalesced)
        start = clock();
        float *inputMatrixDevice;
        float *outputMatrixDevice;
        float *outputMatrixHost = (float *)malloc(TOTAL_SIZE);
        cudaMalloc(&inputMatrixDevice, TOTAL_SIZE);
        cudaMalloc(&outputMatrixDevice, TOTAL_SIZE);

        cudaMemcpy(inputMatrixDevice, inputMatrixReference, TOTAL_SIZE, cudaMemcpyHostToDevice);

        int threads_per_block = 128;
        int number_of_blocks = (ROWS * COLUMNS + threads_per_block - 1) / threads_per_block;
        transpose_gpu_coalesced<<<number_of_blocks, threads_per_block>>>(inputMatrixDevice, outputMatrixDevice, ROWS, COLUMNS);
        cudaMemcpy(outputMatrixHost, outputMatrixDevice, TOTAL_SIZE, cudaMemcpyDeviceToHost);
        checkCuda(cudaDeviceSynchronize());
        end = clock();

        // Verify correctness & performance
        checkAsyncError();
        verify(outputMatrixHost, outputMatrixReference, ROWS * COLUMNS);
        printf("GPU transpose (no UM + coalesced) - %f seconds\n", double(end - start) / CLOCKS_PER_SEC);

        // Free resources
        cudaFree(inputMatrixDevice);
        cudaFree(outputMatrixDevice);
        free(outputMatrixHost);
        checkCuda(cudaDeviceSynchronize());
    }

    free(inputMatrixReference);
    free(outputMatrixReference);
}
