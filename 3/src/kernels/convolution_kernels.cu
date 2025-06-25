#include "kernels.h"
#include <cuda_runtime.h>
#include <plog/Log.h>
#include "../input_args_parser/input_args_parser.h"
#include "../utils/filter_utils.h"

namespace cuda_filter
{

// CUDA error checking
#define CHECK_CUDA_ERROR(call)                                                          \
    {                                                                                   \
        cudaError_t err = call;                                                         \
        if (err != cudaSuccess)                                                         \
        {                                                                               \
            PLOG_ERROR << "CUDA error in " << #call << ": " << cudaGetErrorString(err); \
            return;                                                                     \
        }                                                                               \
    }

    __device__ float luminance(float color[3])
    {
        return 0.2126f * color[0] + 0.7152f * color[1] + 0.0722f * color[2];
    }

    __device__ void srgbToLinear(float color[3], float output[3])
    {
        for (int i = 0; i < 3; i++)
        {
            output[i] = pow(color[i], 2.2);
        }
    }

    __device__ void lineartoSrgb(float color[3], float output[3])
    {
        for (int i = 0; i < 3; i++)
        {
            output[i] = pow(color[i], 1.0f / 2.2f);
        }
    }

    __device__ void tonemapReinhard(float color[3], float output[3])
    {
        float exposure = 1.5;

        for (int i = 0; i < 3; i++)
        {
            output[i] *= exposure / (1. + color[i] / exposure);
            output[i] = pow(output[i], 1.0f / 2.2f);
        }
    }

    __device__ void tonemapFilmic(float color[3], float output[3])
    {
        for (int i = 0; i < 3; i++)
        {
            output[i] = max(0.0f, color[i] - 0.004f);
            output[i] = (output[i] * (6.2f * output[i] + .5f)) / (output[i] * (6.2f * output[i] + 1.7f) + 0.06f);
        }
    }

    __device__ void applySaturation(float color[3], float saturation, float output[3])
    {
        float lum = luminance(color);
        for (int i = 0; i < 3; i++)
        {
            output[i] = saturation * color[i] + (1.0f - saturation) * lum;
        }
    }

    __device__ void applyExposure(float color[3], float exposure, float output[3])
    {
        for (int i = 0; i < 3; i++)
        {
            output[i] = exposure * color[i];
        }
    }

    __device__ void applyGamma(float color[3], float gamma, float output[3])
    {
        for (int i = 0; i < 3; i++)
        {
            output[i] = pow(color[i], gamma);
        }
    }

    __global__ void hdrKernel(const unsigned char *input, unsigned char *output,
                              int width, int height, int channels,
                              float exposure, float gamma, float saturation, ToneMapper tonemap)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= width || y >= height)
            return;

        float color[3] = {((float)input[(y * width + x) * channels + 0]) / 255.0f,
                          (float)(input[(y * width + x) * channels + 1]) / 255.0f,
                          (float)(input[(y * width + x) * channels + 2]) / 255.0f};

        // srgb to linear
        srgbToLinear(color, color);

        // saturation
        applySaturation(color, saturation, color);

        // exposure
        applyExposure(color, exposure, color);

        // tonemap
        switch (tonemap)
        {
        case ToneMapper::FILMIC:
            tonemapFilmic(color, color);
            break;
        case ToneMapper::REINHARD:
            tonemapReinhard(color, color);
            break;
        default:
            break;
        }

        // gamma
        applyGamma(color, gamma, color);

        // linear rgb -> srgb
        lineartoSrgb(color, color);

        // Record result
        for (int c = 0; c < 3; c++)
        {
            output[(y * width + x) * channels + c] = static_cast<unsigned char>(min(max(color[c] * 255.0f, 0.0f), 255.0f));
        }
    }

    // CUDA kernel for 2D convolution
    __global__ void convolutionKernel(const unsigned char *input, unsigned char *output,
                                      const float *kernel, int width, int height,
                                      int channels, int kernelSize)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= width || y >= height)
            return;

        int radius = kernelSize / 2;

        for (int c = 0; c < channels; c++)
        {
            float sum = 0.0f;

            for (int ky = -radius; ky <= radius; ky++)
            {
                for (int kx = -radius; kx <= radius; kx++)
                {
                    int ix = min(max(x + kx, 0), width - 1);
                    int iy = min(max(y + ky, 0), height - 1);

                    float kernelValue = kernel[(ky + radius) * kernelSize + (kx + radius)];
                    float pixelValue = input[(iy * width + ix) * channels + c];

                    sum += pixelValue * kernelValue;
                }
            }

            // Clamp the result to [0, 255]
            output[(y * width + x) * channels + c] = static_cast<unsigned char>(min(max(sum, 0.0f), 255.0f));
        }
    }

    void applyFilterGPU(const cv::Mat &input, cv::Mat &output, const cv::Mat &kernel, FilterOptions options)
    {
        if (input.empty() || kernel.empty())
        {
            PLOG_ERROR << "Input image or kernel is empty";
            return;
        }

        // Ensure output has the same size and type as input
        output.create(input.size(), input.type());

        // Get image dimensions
        int width = input.cols;
        int height = input.rows;
        int channels = input.channels();
        int kernelSize = kernel.rows;

        // Allocate device memory
        unsigned char *d_input = nullptr;
        unsigned char *d_output = nullptr;
        float *d_kernel = nullptr;

        size_t imageSize = width * height * channels * sizeof(unsigned char);
        size_t kernelSize_bytes = kernelSize * kernelSize * sizeof(float);

        // Copy kernel to CPU float array
        float *h_kernel = new float[kernelSize * kernelSize];
        for (int i = 0; i < kernelSize; i++)
        {
            for (int j = 0; j < kernelSize; j++)
            {
                h_kernel[i * kernelSize + j] = kernel.at<float>(i, j);
            }
        }

        // Allocate device memory
        CHECK_CUDA_ERROR(cudaMalloc(&d_input, imageSize));
        CHECK_CUDA_ERROR(cudaMalloc(&d_output, imageSize));
        CHECK_CUDA_ERROR(cudaMalloc(&d_kernel, kernelSize_bytes));

        // Copy data to device
        CHECK_CUDA_ERROR(cudaMemcpy(d_input, input.data, imageSize, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_kernel, h_kernel, kernelSize_bytes, cudaMemcpyHostToDevice));

        // Define block and grid dimensions
        dim3 blockDim(16, 16);
        dim3 gridDim(cuda::divUp(width, blockDim.x), cuda::divUp(height, blockDim.y));

        // Launch kernel
        switch (cuda_filter::FilterUtils::stringToFilterType(options.filterType))
        {
        case FilterType::HDR_TONEMAPPING:
            hdrKernel<<<gridDim, blockDim>>>(d_input, d_output, width, height, channels, options.exposure, options.gamma, options.saturation, options.toneMapper);
            break;
        default:
            convolutionKernel<<<gridDim, blockDim>>>(d_input, d_output, d_kernel, width, height, channels, kernelSize);
        }

        // Check for kernel launch errors
        CHECK_CUDA_ERROR(cudaGetLastError());

        // Synchronize to ensure kernel execution is complete
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        // Copy result back to host
        CHECK_CUDA_ERROR(cudaMemcpy(output.data, d_output, imageSize, cudaMemcpyDeviceToHost));

        // Free device memory
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_kernel);

        // Free host memory
        delete[] h_kernel;
    }

    void applyFilterCPU(const cv::Mat &input, cv::Mat &output, const cv::Mat &kernel)
    {
        if (input.empty() || kernel.empty())
        {
            PLOG_ERROR << "Input image or kernel is empty";
            return;
        }

        // Ensure output has the same size and type as input
        output.create(input.size(), input.type());

        // Get image dimensions
        int width = input.cols;
        int height = input.rows;
        int channels = input.channels();
        int kernelSize = kernel.rows;
        int radius = kernelSize / 2;

        // Convert kernel to float array for faster access
        float *h_kernel = new float[kernelSize * kernelSize];
        for (int i = 0; i < kernelSize; i++)
        {
            for (int j = 0; j < kernelSize; j++)
            {
                h_kernel[i * kernelSize + j] = kernel.at<float>(i, j);
            }
        }

        // Process each pixel
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                for (int c = 0; c < channels; c++)
                {
                    float sum = 0.0f;

                    // Apply kernel
                    for (int ky = -radius; ky <= radius; ky++)
                    {
                        for (int kx = -radius; kx <= radius; kx++)
                        {
                            int ix = std::min(std::max(x + kx, 0), width - 1);
                            int iy = std::min(std::max(y + ky, 0), height - 1);

                            float kernelValue = h_kernel[(ky + radius) * kernelSize + (kx + radius)];
                            float pixelValue = input.at<cv::Vec3b>(iy, ix)[c];

                            sum += pixelValue * kernelValue;
                        }
                    }

                    // Clamp the result to [0, 255]
                    output.at<cv::Vec3b>(y, x)[c] = static_cast<unsigned char>(std::min(std::max(sum, 0.0f), 255.0f));
                }
            }
        }

        delete[] h_kernel;
    }

} // namespace cuda_filter
