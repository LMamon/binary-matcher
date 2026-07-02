#include <cuda_runtime_api.h>
#include <npp.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(expr) do {                           \
    cudaError_t cuda_result = (expr);                   \
    if(cuda_result != cudaSuccess) {                    \
        fprintf(stderr,                                 \
                "CUDA Error: %s:%i:%d = %s\n",  \
                __FILE__,                               \
                __LINE__,                               \
                cuda_result,                            \
                cudaGetErrorString(cuda_result));       \
        exit(EXIT_FAILURE);                             \
    }                                                   \
} while(0)

#define CUDA_KERNEL_CHECK() do {                        \
    CUDA_CHECK(cudaGetLastError());                     \
    CUDA_CHECK(cudaDeviceSynchronize());                \
} while(0)

#define NPP_CHECK(expr) do {                            \
    NppStatus npp_result = (expr);                      \
    if (npp_result != NPP_SUCCESS) {                    \
        fprintf(stderr,                                 \
                "NPP Error: %s:%i:%d\n",                \
                __FILE__,                               \
                __LINE__,                               \
                npp_result);                            \
        exit(EXIT_FAILURE);                             \
    }                                                   \
} while(0)

#define TRT_CHECK(expr) do {                            \
    if (!(expr)) {                                      \
        fprintf(stderr,                                 \
                "TensorRT Error: %s:%i\n",              \
                __FILE__,                               \
                __LINE__);                              \
        exit(EXIT_FAILURE);                             \
    }                                                   \
} while(0)
