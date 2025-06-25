#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#define FILTER_WIDTH 5
#define TILE_SIZE 16
#define TILE_SIZE_SEPARABLE 256
#define CHANNELS 3
// Constants for filter definitions
__constant__ float d_filter[81]; // Max 9x9 filter

// Example filter kernels students can implement:

// Sum (3x3)
float sum9x9[81] = {
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

// Box blur (3x3)
float boxBlur3x3[9] = {
    1 / 9.0f, 1 / 9.0f, 1 / 9.0f,
    1 / 9.0f, 1 / 9.0f, 1 / 9.0f,
    1 / 9.0f, 1 / 9.0f, 1 / 9.0f};

// Box blur (9x9)
float boxBlur9x9[81] = {
    1 / 81.0f, 1 / 81.0f, 1 / 81.0f, 1 / 81.0f, 1 / 81.0f, 1 / 81.0f, 1 / 81.0f, 1 / 81.0f, 1 / 81.0f,
    1 / 81.0f, 1 / 81.0f, 1 / 81.0f, 1 / 81.0f, 1 / 81.0f, 1 / 81.0f, 1 / 81.0f, 1 / 81.0f, 1 / 81.0f,
    1 / 81.0f, 1 / 81.0f, 1 / 81.0f, 1 / 81.0f, 1 / 81.0f, 1 / 81.0f, 1 / 81.0f, 1 / 81.0f, 1 / 81.0f,
    1 / 81.0f, 1 / 81.0f, 1 / 81.0f, 1 / 81.0f, 1 / 81.0f, 1 / 81.0f, 1 / 81.0f, 1 / 81.0f, 1 / 81.0f,
    1 / 81.0f, 1 / 81.0f, 1 / 81.0f, 1 / 81.0f, 1 / 81.0f, 1 / 81.0f, 1 / 81.0f, 1 / 81.0f, 1 / 81.0f,
    1 / 81.0f, 1 / 81.0f, 1 / 81.0f, 1 / 81.0f, 1 / 81.0f, 1 / 81.0f, 1 / 81.0f, 1 / 81.0f, 1 / 81.0f,
    1 / 81.0f, 1 / 81.0f, 1 / 81.0f, 1 / 81.0f, 1 / 81.0f, 1 / 81.0f, 1 / 81.0f, 1 / 81.0f, 1 / 81.0f,
    1 / 81.0f, 1 / 81.0f, 1 / 81.0f, 1 / 81.0f, 1 / 81.0f, 1 / 81.0f, 1 / 81.0f, 1 / 81.0f, 1 / 81.0f,
    1 / 81.0f, 1 / 81.0f, 1 / 81.0f, 1 / 81.0f, 1 / 81.0f, 1 / 81.0f, 1 / 81.0f, 1 / 81.0f, 1 / 81.0f};

// Gaussian blur (5x5)
float gaussianBlur5x5[25] = {
    1 / 273.0f, 4 / 273.0f, 7 / 273.0f, 4 / 273.0f, 1 / 273.0f,
    4 / 273.0f, 16 / 273.0f, 26 / 273.0f, 16 / 273.0f, 4 / 273.0f,
    7 / 273.0f, 26 / 273.0f, 41 / 273.0f, 26 / 273.0f, 7 / 273.0f,
    4 / 273.0f, 16 / 273.0f, 26 / 273.0f, 16 / 273.0f, 4 / 273.0f,
    1 / 273.0f, 4 / 273.0f, 7 / 273.0f, 4 / 273.0f, 1 / 273.0f};

// Sobel edge detection (horizontal)
float sobelX[9] = {
    -1, 0, 1,
    -2, 0, 2,
    -1, 0, 1};

// Sobel edge detection (vertical)
float sobelY[9] = {
    -1, -2, -1,
    0, 0, 0,
    1, 2, 1};

// Sharpen filter
float sharpen[9] = {
    0, -1, 0,
    -1, 5, -1,
    0, -1, 0};

// Utility to check for CUDA errors
#define CHECK_CUDA_ERROR(call)                                        \
    {                                                                 \
        cudaError_t err = call;                                       \
        if (err != cudaSuccess)                                       \
        {                                                             \
            fprintf(stderr, "CUDA Error: %s at line %d in file %s\n", \
                    cudaGetErrorString(err), __LINE__, __FILE__);     \
            exit(EXIT_FAILURE);                                       \
        }                                                             \
    }

void checkAsyncError()
{
    cudaError_t syncErr = cudaGetLastError();
    if (syncErr != cudaSuccess)
    {
        printf("Kernel launch error: %s\n", cudaGetErrorString(syncErr));
    }
}

// Structure to hold image data
typedef struct
{
    unsigned char *data;
    int width;
    int height;
    int channels; // 1 for grayscale, 3 for RGB, 4 for RGBA
} Image;

// CPU implementation of 2D convolution
void convolutionCPU(const Image *input, Image *output, const float *filter, int filterWidth)
{
    for (int ch = 0; ch < input->channels; ch++)
    {
        for (int y = 0; y < input->height; y++)
        {
            for (int x = 0; x < input->width; x++)
            {
                float val = 0;

                for (int dx = -(filterWidth - 1) / 2; dx <= (filterWidth - 1) / 2; dx++)
                {
                    int filterX = dx + (filterWidth - 1) / 2;
                    for (int dy = -(filterWidth - 1) / 2; dy <= (filterWidth - 1) / 2; dy++)
                    {
                        int filterY = dy + (filterWidth - 1) / 2;

                        int offsetX = max(0, min(input->width - 1, x + dx));
                        int offsetY = max(0, min(input->height - 1, y + dy));

                        float pixelVal = ((float)(input->data[(offsetY * (input->width) + offsetX) * input->channels + ch])) / 255.0f;
                        float filterVal = filter[filterY * filterWidth + filterX];

                        val += pixelVal * filterVal;
                    }
                }

                output->data[(y * (input->width) + x) * input->channels + ch] = max(0, min(255, (int)roundf(val * 255.0f)));
            }
        }
    }
}

// Naive GPU implementation - each thread computes one output pixel
__global__ void convolutionKernelNaive(unsigned char *input, unsigned char *output,
                                       float *filter, int filterWidth,
                                       int width, int height, int channels)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    for (int ch = 0; ch < channels; ch++)
    {
        float val = 0;

        for (int dx = -(filterWidth - 1) / 2; dx <= (filterWidth - 1) / 2; dx++)
        {
            int filterX = dx + (filterWidth - 1) / 2;
            for (int dy = -(filterWidth - 1) / 2; dy <= (filterWidth - 1) / 2; dy++)
            {
                int filterY = dy + (filterWidth - 1) / 2;

                int offsetX = max(0, min(width - 1, x + dx));
                int offsetY = max(0, min(height - 1, y + dy));

                float pixelVal = ((float)(input[(offsetY * width + offsetX) * channels + ch])) / 255.0f;
                float filterVal = filter[filterY * filterWidth + filterX];

                val += pixelVal * filterVal;
            }
        }

        output[(y * width + x) * channels + ch] = max(0, min(255, (int)roundf(val * 255.0f)));
    }
}

// Shared memory implementation
__global__ void convolutionKernelShared(unsigned char *input, unsigned char *output,
                                        float *filter, int filterWidth,
                                        int width, int height, int channels)
{
    const int SHARED_DATA_SIZE = (TILE_SIZE + FILTER_WIDTH - 1) * (TILE_SIZE + FILTER_WIDTH - 1) * CHANNELS;
    __shared__ unsigned char imageTile[SHARED_DATA_SIZE];

    // Pre-load image data
    const int imTileOffsetX = blockIdx.x * blockDim.x;
    const int imTileOffsetY = blockIdx.y * blockDim.y;
    for (int i = threadIdx.x + threadIdx.y * blockDim.x; i < SHARED_DATA_SIZE; i += blockDim.x * blockDim.y)
    {
        int imCoordX = max(0, min(width - 1, imTileOffsetX + (i / CHANNELS) % (TILE_SIZE + FILTER_WIDTH - 1) - (FILTER_WIDTH - 1) / 2));
        int imCoordY = max(0, min(height - 1, imTileOffsetY + (i / CHANNELS) / (TILE_SIZE + FILTER_WIDTH - 1) - (FILTER_WIDTH - 1) / 2));

        imageTile[i] = input[(imCoordX + imCoordY * width) * CHANNELS + i % CHANNELS];
    }
    __syncthreads();

    // Perform convolution
    int x = imTileOffsetX + threadIdx.x;
    int y = imTileOffsetY + threadIdx.y;

    if (x >= width || y >= height)
        return;

    for (int ch = 0; ch < channels; ch++)
    {
        float val = 0;

        for (int filterX = 0; filterX < filterWidth; filterX++)
        {
            for (int filterY = 0; filterY < filterWidth; filterY++)
            {
                float pixelVal = ((float)(imageTile[((TILE_SIZE + FILTER_WIDTH - 1) * (threadIdx.y + filterY) + (threadIdx.x + filterX)) * channels + ch])) / 255.0f;
                float filterVal = filter[filterY * filterWidth + filterX];

                val += pixelVal * filterVal;
            }
        }

        output[(y * width + x) * channels + ch] = max(0, min(255, (int)roundf(val * 255.0f)));
    }
}

// Shared memory + constant filter implementation
__global__ void convolutionKernelConstant(unsigned char *input, unsigned char *output,
                                          int filterWidth, int width, int height, int channels)
{
    const int SHARED_DATA_SIZE = (TILE_SIZE + FILTER_WIDTH - 1) * (TILE_SIZE + FILTER_WIDTH - 1) * CHANNELS;
    __shared__ unsigned char imageTile[SHARED_DATA_SIZE];

    // Pre-load image data
    const int imTileOffsetX = blockIdx.x * blockDim.x;
    const int imTileOffsetY = blockIdx.y * blockDim.y;
    for (int i = threadIdx.x + threadIdx.y * blockDim.x; i < SHARED_DATA_SIZE; i += blockDim.x * blockDim.y)
    {
        int imCoordX = max(0, min(width - 1, imTileOffsetX + (i / CHANNELS) % (TILE_SIZE + FILTER_WIDTH - 1) - (FILTER_WIDTH - 1) / 2));
        int imCoordY = max(0, min(height - 1, imTileOffsetY + (i / CHANNELS) / (TILE_SIZE + FILTER_WIDTH - 1) - (FILTER_WIDTH - 1) / 2));

        imageTile[i] = input[(imCoordX + imCoordY * width) * CHANNELS + i % CHANNELS];
    }
    __syncthreads();

    // Perform convolution
    int x = imTileOffsetX + threadIdx.x;
    int y = imTileOffsetY + threadIdx.y;

    if (x >= width || y >= height)
        return;

    for (int ch = 0; ch < channels; ch++)
    {
        float val = 0;

        for (int filterX = 0; filterX < filterWidth; filterX++)
        {
            for (int filterY = 0; filterY < filterWidth; filterY++)
            {
                float pixelVal = ((float)(imageTile[((TILE_SIZE + FILTER_WIDTH - 1) * (threadIdx.y + filterY) + (threadIdx.x + filterX)) * channels + ch])) / 255.0f;
                float filterVal = d_filter[filterY * filterWidth + filterX];

                val += pixelVal * filterVal;
            }
        }

        output[(y * width + x) * channels + ch] = max(0, min(255, (int)roundf(val * 255.0f)));
    }
}

// Shared memory + constant filter separable implementation
__global__ void convolutionKernelHorizontal(unsigned char *input, unsigned char *output,
                                            int filterWidth, int width, int height, int channels)
{
    const int SHARED_DATA_SIZE = (TILE_SIZE_SEPARABLE + FILTER_WIDTH - 1) * CHANNELS;
    __shared__ __align__(16) uchar4 imageTile[SHARED_DATA_SIZE];

    // Pre-load image data
    const int imTileOffsetX = blockIdx.x * blockDim.x;
    int x = imTileOffsetX + threadIdx.x;
    int y = blockIdx.y * blockDim.y;
    for (int i = threadIdx.x; i < SHARED_DATA_SIZE; i += blockDim.x)
    {
        int imCoordX = max(0, min(width - 1, imTileOffsetX + (i / CHANNELS) - (FILTER_WIDTH - 1) / 2));

        imageTile[i].x = input[(imCoordX + y * width) * CHANNELS + i % CHANNELS];
    }
    __syncthreads();

    // Perform convolution
    if (x >= width || y >= height)
        return;

    for (int ch = 0; ch < channels; ch++)
    {
        float val = 0;

        for (int filterX = 0; filterX < filterWidth; filterX++)
        {
            float pixelVal = ((float)(imageTile[(threadIdx.x + filterX) * channels + ch].x)) / 255.0f;
            float filterVal = d_filter[filterX];

            val += pixelVal * filterVal;
        }

        output[(y * width + x) * channels + ch] = max(0, min(255, (int)roundf(val * 255.0f)));
    }
}
__global__ void convolutionKernelVertical(unsigned char *input, unsigned char *output,
                                          int filterWidth, int width, int height, int channels)
{
    const int SHARED_DATA_SIZE = (TILE_SIZE_SEPARABLE + FILTER_WIDTH - 1) * CHANNELS;
    __shared__ __align__(16) uchar4 imageTile[SHARED_DATA_SIZE];

    // Pre-load image data
    const int imTileOffsetY = blockIdx.y * blockDim.y;
    int x = blockIdx.x * blockDim.x;
    int y = imTileOffsetY + threadIdx.y;
    for (int i = threadIdx.y; i < SHARED_DATA_SIZE; i += blockDim.y)
    {
        int imCoordY = max(0, min(height - 1, imTileOffsetY + (i / CHANNELS) - (FILTER_WIDTH - 1) / 2));

        imageTile[i].x = input[(x + imCoordY * width) * CHANNELS + i % CHANNELS];
    }
    __syncthreads();

    // Perform convolution

    if (x >= width || y >= height)
        return;

    for (int ch = 0; ch < channels; ch++)
    {
        float val = 0;

        for (int filterY = 0; filterY < filterWidth; filterY++)
        {
            float pixelVal = ((float)(imageTile[(threadIdx.y + filterY) * channels + ch].x)) / 255.0f;
            float filterVal = d_filter[filterY];

            val += pixelVal * filterVal;
        }

        output[(y * width + x) * channels + ch] = max(0, min(255, (int)roundf(val * 255.0f)));
    }
}

void verify(unsigned char *a, unsigned char *b, int elCount)
{
    for (int i = 0; i < elCount; i++)
    {
        if (a[i] != b[i])
        {
            printf("Error! Images don't match!");
            return;
        }
    }
}

// Main function to compare implementations
int main(int argc, char **argv)
{
    const int WIDTH = 1920;
    const int HEIGHT = 1080;
    const size_t TOTAL_SIZE = sizeof(unsigned char) * WIDTH * HEIGHT * CHANNELS;

    Image *imSrc = (Image *)malloc(sizeof(Image));
    imSrc->channels = CHANNELS;
    imSrc->width = WIDTH;
    imSrc->height = HEIGHT;
    imSrc->data = (unsigned char *)malloc(TOTAL_SIZE);

    Image *imDst = (Image *)malloc(sizeof(Image));
    imDst->channels = CHANNELS;
    imDst->width = WIDTH;
    imDst->height = HEIGHT;
    imDst->data = (unsigned char *)malloc(TOTAL_SIZE);

    // Generate random source image
    for (int i = 0; i < WIDTH * HEIGHT * CHANNELS; i++)
    {
        imSrc->data[i] = rand() % 256;
    }

    // Choose filter
    float *filterCPU = sum9x9; // Using sum here as an example because its filter weights work both for separable & unufied kernels without modifications
    cudaMemcpyToSymbol(d_filter, filterCPU, FILTER_WIDTH * FILTER_WIDTH * sizeof(float));

    clock_t start, end;

    {
        // CPU implementation
        start = clock();
        convolutionCPU(imSrc, imDst, filterCPU, FILTER_WIDTH);
        end = clock();
        printf("CPU convolution - %f seconds\n", double(end - start) / CLOCKS_PER_SEC);
    }

    {
        // GPU implementation
        unsigned char *cpuImg = (unsigned char *)malloc(TOTAL_SIZE);

        start = clock();
        unsigned char *imSrcGPU;
        cudaMalloc(&imSrcGPU, TOTAL_SIZE);
        unsigned char *imDstGPU;
        cudaMalloc(&imDstGPU, TOTAL_SIZE);
        float *filterGPU;
        cudaMalloc(&filterGPU, FILTER_WIDTH * FILTER_WIDTH * sizeof(float));

        cudaMemcpy(imSrcGPU, imSrc->data, TOTAL_SIZE, cudaMemcpyHostToDevice);
        cudaMemcpy(filterGPU, filterCPU, FILTER_WIDTH * FILTER_WIDTH * sizeof(float), cudaMemcpyHostToDevice);

        dim3 threads_per_block(TILE_SIZE, TILE_SIZE, 1);
        dim3 number_of_blocks((WIDTH + threads_per_block.x - 1) / threads_per_block.x, (HEIGHT + threads_per_block.y - 1) / threads_per_block.y, 1);
        convolutionKernelNaive<<<number_of_blocks, threads_per_block>>>(imSrcGPU, imDstGPU, filterGPU, FILTER_WIDTH, WIDTH, HEIGHT, CHANNELS);
        cudaMemcpy(cpuImg, imDstGPU, TOTAL_SIZE, cudaMemcpyDeviceToHost);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        end = clock();

        // Verify correctness & performance
        checkAsyncError();
        verify(imDst->data, cpuImg, WIDTH * HEIGHT * CHANNELS);
        printf("GPU convolution (naive) - %f seconds\n", double(end - start) / CLOCKS_PER_SEC);

        // Free resources
        cudaFree(imSrcGPU);
        cudaFree(imDstGPU);
        cudaFree(filterGPU);
        free(cpuImg);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    }

    {
        // GPU shared memory implementation
        unsigned char *cpuImg = (unsigned char *)malloc(TOTAL_SIZE);

        start = clock();
        unsigned char *imSrcGPU;
        cudaMalloc(&imSrcGPU, TOTAL_SIZE);
        unsigned char *imDstGPU;
        cudaMalloc(&imDstGPU, TOTAL_SIZE);
        float *filterGPU;
        cudaMalloc(&filterGPU, FILTER_WIDTH * FILTER_WIDTH * sizeof(float));

        cudaMemcpy(imSrcGPU, imSrc->data, TOTAL_SIZE, cudaMemcpyHostToDevice);
        cudaMemcpy(filterGPU, filterCPU, FILTER_WIDTH * FILTER_WIDTH * sizeof(float), cudaMemcpyHostToDevice);

        dim3 threads_per_block(TILE_SIZE, TILE_SIZE, 1);
        dim3 number_of_blocks((WIDTH + threads_per_block.x - 1) / threads_per_block.x, (HEIGHT + threads_per_block.y - 1) / threads_per_block.y, 1);
        convolutionKernelShared<<<number_of_blocks, threads_per_block>>>(imSrcGPU, imDstGPU, filterGPU, FILTER_WIDTH, WIDTH, HEIGHT, CHANNELS);
        cudaMemcpy(cpuImg, imDstGPU, TOTAL_SIZE, cudaMemcpyDeviceToHost);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        end = clock();

        // Verify correctness & performance
        checkAsyncError();
        verify(imDst->data, cpuImg, WIDTH * HEIGHT * CHANNELS);
        printf("GPU convolution (shared memory) - %f seconds\n", double(end - start) / CLOCKS_PER_SEC);

        // Free resources
        cudaFree(imSrcGPU);
        cudaFree(imDstGPU);
        cudaFree(filterGPU);
        free(cpuImg);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    }

    {
        // GPU shared memory + constant filter implementation
        unsigned char *cpuImg = (unsigned char *)malloc(TOTAL_SIZE);

        start = clock();
        unsigned char *imSrcGPU;
        cudaMalloc(&imSrcGPU, TOTAL_SIZE);
        unsigned char *imDstGPU;
        cudaMalloc(&imDstGPU, TOTAL_SIZE);

        cudaMemcpy(imSrcGPU, imSrc->data, TOTAL_SIZE, cudaMemcpyHostToDevice);

        dim3 threads_per_block(TILE_SIZE, TILE_SIZE, 1);
        dim3 number_of_blocks((WIDTH + threads_per_block.x - 1) / threads_per_block.x, (HEIGHT + threads_per_block.y - 1) / threads_per_block.y, 1);
        convolutionKernelConstant<<<number_of_blocks, threads_per_block>>>(imSrcGPU, imDstGPU, FILTER_WIDTH, WIDTH, HEIGHT, CHANNELS);
        cudaMemcpy(cpuImg, imDstGPU, TOTAL_SIZE, cudaMemcpyDeviceToHost);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        end = clock();

        // Verify correctness & performance
        checkAsyncError();
        verify(imDst->data, cpuImg, WIDTH * HEIGHT * CHANNELS);
        printf("GPU convolution (shared memory + constant filter) - %f seconds\n", double(end - start) / CLOCKS_PER_SEC);

        // Free resources
        cudaFree(imSrcGPU);
        cudaFree(imDstGPU);
        free(cpuImg);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    }

    {
        // GPU shared memory + constant filter separable implementation
        unsigned char *cpuImg = (unsigned char *)malloc(TOTAL_SIZE);

        start = clock();
        unsigned char *imSrcGPU;
        cudaMalloc(&imSrcGPU, TOTAL_SIZE);
        unsigned char *imDstGPU;
        cudaMalloc(&imDstGPU, TOTAL_SIZE);

        cudaMemcpy(imSrcGPU, imSrc->data, TOTAL_SIZE, cudaMemcpyHostToDevice);

        dim3 threads_per_block(TILE_SIZE_SEPARABLE, 1, 1);
        dim3 number_of_blocks((WIDTH + threads_per_block.x - 1) / threads_per_block.x, (HEIGHT + threads_per_block.y - 1) / threads_per_block.y, 1);
        convolutionKernelHorizontal<<<number_of_blocks, threads_per_block>>>(imSrcGPU, imDstGPU, FILTER_WIDTH, WIDTH, HEIGHT, CHANNELS);
        threads_per_block = dim3(1, TILE_SIZE_SEPARABLE, 1);
        number_of_blocks = dim3((WIDTH + threads_per_block.x - 1) / threads_per_block.x, (HEIGHT + threads_per_block.y - 1) / threads_per_block.y, 1);
        convolutionKernelVertical<<<number_of_blocks, threads_per_block>>>(imDstGPU, imSrcGPU, FILTER_WIDTH, WIDTH, HEIGHT, CHANNELS);
        cudaMemcpy(cpuImg, imSrcGPU, TOTAL_SIZE, cudaMemcpyDeviceToHost);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        end = clock();

        // Verify correctness & performance
        checkAsyncError();
        verify(imDst->data, cpuImg, WIDTH * HEIGHT * CHANNELS);
        printf("GPU convolution (shared memory + constant filter + separable) - %f seconds\n", double(end - start) / CLOCKS_PER_SEC);

        // Free resources
        cudaFree(imSrcGPU);
        cudaFree(imDstGPU);
        free(cpuImg);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    }

    return 0;
}