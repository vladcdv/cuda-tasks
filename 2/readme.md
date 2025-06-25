# 2 Convolution

## Conclusions

Naive GPU implementation seems to be better than CPU implementation for images larger than 512x512.

For shared memory implementation, tile sizes don't seem to have a measurable impact on performance between 8x8 and 16x16 tiles. However, 32x32 does appear to be slightly worse.

Upgrading filter from local memory to constant memory makes a difference, although quite small (from 0.147 to 0.144 seconds).

Effect of different filter sizes on performance (operation duration shown in seconds; 15360x8640 image used)

| Filter Size | CPU | GPU (Naive) | GPU (Shared Memory) | GPU (Shared Memory + Constant Filter) | GPU (Shared Memory + Constant Filter + Separable) | GPU (Shared Memory + Constant Filter + Separable + Aligned) |
| ----------- | --- | ----------- | ------------------- | ------------------------------------- | ------------------------------------------------- | ----------------------------------------------------------- |
| 9x9         | 265 | 0.23        | 0.2                 | 0.2                                   | 0.15                                              | 0.148                                                       |
| 7x7         | 166 | 0.18        | 0.17                | 0.17                                  | 0.147                                             | 0.148                                                       |
| 5x5         | 93  | 0.15        | 0.15                | 0.15                                  | 0.145                                             | 0.145                                                       |
| 3x3         | 41  | 0.13        | 0.13                | 0.15                                  | 0.145                                             | 0.144                                                       |
| 1x1         | 15  | 0.12        | 0.12                | 0.12                                  | 0.143                                             | 0.160                                                       |

Based on the above results, it can be seen that filter size does indeed negatively impact performance, although CPU based solution scales much more poorly than GPU accelerated ones.

Separable kernel has a substantial performance boost for larger kernels, which is to be expected as it decreases algorithmic complexity of iterating over filter weights from O(n^2) to O(n) where n is the dimension of the filter (e.g. 9x9 filter would have n=9).

Alignment & avoiding bank conflicts didn't seem to make much of a noticeable difference on performance of convolution.

## Bottlenecks & Possible Optimizations

Looking at Nvidia Nsight a good chunk of time is taken up by memory operations - copying data to and from GPU. In this particular example, it would be more optimal to create data right on the GPU. Also, comparison to reference CPU algorithm should be performed on the GPU as well, using reduction and atomic operations. After CPU computes reference result, it can be copied to the GPU and stored there for subsequent kernel dispatches, which would allow us to avoid copying the data back to CPU and performing a sequential comparison of every element.
