# binary-matcher

Adapted from [CUDAHammingMean](https://github.com/komrad36/CUDAHammingMean.git) for modern CUDA toolchains and NVIDIA Jetson.

Provides CUDA primitives for pairwise Hamming-distance matching between 256-bit binary descriptors using shared-memory caching and warp-level reductions.

All functionality is contained in `binary_matcher.cuh` and `binary_matcher.cu`. `main.cpp` provides a simple benchmark and example demonstrating library usage.