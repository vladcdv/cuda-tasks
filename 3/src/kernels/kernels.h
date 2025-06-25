#pragma once

#include <opencv2/opencv.hpp>
#include "../input_args_parser/input_args_parser.h"

namespace cuda_filter
{

    void applyFilterGPU(const cv::Mat &input, cv::Mat &output, const cv::Mat &kernel, FilterOptions options);
    void applyFilterCPU(const cv::Mat &input, cv::Mat &output, const cv::Mat &kernel);

    namespace cuda
    {
// CUDA-specific type declarations and helper functions
#ifdef __CUDACC__
        // These will only be visible to CUDA compiler
        __host__ __device__ inline int divUp(int a, int b)
        {
            return (a + b - 1) / b;
        }
#endif
    }

} // namespace cuda_filter