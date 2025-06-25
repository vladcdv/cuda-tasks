# 1 CUDA Matrix Transposition

## Conclusions

GPU accelerated matrix transposition becomes more performant only after matrix size has grown substantially (around 16k by 8k matrix dimensions is the turning point on my hardware), since otherwise overhead of communication and memory exchanges between CPU and GPU is too large for the parallelism and high VRAM bandwidth of the GPU to hide this latency.

16x16 block size appears to be optimal block size for non-coalesced kernels. Coalesced kernels don't seem to have as much of an effect from varying block dimensions (the effect is too small to detect via perf clocks), although 128 threads per block seems to work reasonably well.

Performance outcomes (for 32k x 16k matrix, measured on RTX 3070):

| Method                  | Duration (seconds) |
| ----------------------- | ------------------ |
| CPU                     | 3.13               |
| GPU (UM)                | 0.91               |
| GPU (UM + coalesced)    | 0.92               |
| GPU (no UM)             | 0.72               |
| GPU (no UM + coalesced) | 0.73               |

From the above it can be concluded that direct memory transfers are better than Unified Memory, in the case of GPU accelerated transpose, likely because they don't have the overhead of a complex memory managing mechanism, and there is no way to hide the additional UM latency during a simple matrix transpose. Small differences with and without coalesced memory access don't seem to make any substantial difference, from which we can conclude that either other parts are contributing way more to the overhead, or memory accesses are "coalesced enough" across all implementations. Also note that I'm working on a Windows machine and cannot use or test pre-fetching https://forums.developer.nvidia.com/t/invalid-device-ordinal/303172

## Further Optimizations

Currently, CUDA kernel only performs a single element transposition per thread, which can be quite wasteful if GPU becomes fully saturated and with more blocks waiting to be dispatched, as there is an extra performance penalty associated with dispatching new blocks. Instead, only as many blocks should be dispatched as are required to saturate GPU, and if there are more matrix elements, already dispatched threads should just perform transposition for those elements by accessing memory with stride.

Additionally, only memory reads are currently coalesced, there might be a more optimal memory access pattern that would localize both reads and writes, and yield overall better performance.
