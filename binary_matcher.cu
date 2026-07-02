#include "cuda_check.h"
#include "binary_matcher.cuh"
#include "binary_matcher_kernels.cuh"

#include <vector>
#include <cstring>
#include <cuda_runtime.h>
#include <cstdint>

__device__ __forceinline__ int hammingDistance256(const uint8_t* a, const uint8_t* b) {
    const auto* a64 = reinterpret_cast<const uint64_t*>(a);
    const auto* b64 = reinterpret_cast<const uint64_t*>(b);

    int dist = 0;

    dist += __popcll(a64[0] ^ b64[0]);
    dist += __popcll(a64[1] ^ b64[1]);
    dist += __popcll(a64[2] ^ b64[2]);
    dist += __popcll(a64[3] ^ b64[3]);

    return dist;
}

__global__ void binary_cost_matrix_kernel(const uint8_t* queryDescriptors,
                                                const int* queryCounts,
                                                const uint8_t* referenceDescriptors,
                                                const int* referenceCounts,
                                                int numQueries,
                                                int numReferences,
                                                float* costMatrix) {

    const int referenceIdx = blockIdx.x;
    const int queryIdx = blockIdx.y;

    if (queryIdx >= numQueries || referenceIdx >= numReferences) return;

    const int queryCount = queryCounts[queryIdx];
    const int referenceCount = referenceCounts[referenceIdx];

    // no descriptors available.
    if (queryCount <= 0 || referenceCount <= 0) {
        if (threadIdx.x == 0) {
            costMatrix[queryIdx * numReferences + referenceIdx] = 1.0f;
        }

        return;
    }

    const uint8_t* queryBase = queryDescriptors + queryIdx * MAX_DESCRIPTOR_COUNT * ORB_DESCRIPTOR_SIZE;
    const uint8_t* referenceBase = referenceDescriptors + referenceIdx * MAX_DESCRIPTOR_COUNT * ORB_DESCRIPTOR_SIZE;

    // cache reference descriptors in shared memory.
    __shared__ uint8_t sharedReference[MAX_DESCRIPTOR_COUNT * ORB_DESCRIPTOR_SIZE];

    for (int i = threadIdx.x; i < referenceCount * ORB_DESCRIPTOR_SIZE; i += blockDim.x) {
        sharedReference[i] = referenceBase[i];
    }

    __syncthreads();

    const int lane = threadIdx.x & 31;
    const int warpId = threadIdx.x >> 5;

    constexpr int WARPS_PER_BLOCK = 256 / 32;

    // best distance for the descriptor currently
    // assigned to this warp.
    int bestDist = 256;

    // store one best match per query descriptor.
    __shared__ int descriptorBest[MAX_DESCRIPTOR_COUNT];

    // a warp may process multiple query descriptors.
    //
    // with:
    //     32 descriptors
    //     8 warps
    //
    // warp 0 handles:
    //     0,8,16,24
    //
    // warp 1 handles:
    //     1,9,17,25
    // etc.

    for (int queryDescIdx = warpId; queryDescIdx < queryCount; queryDescIdx += WARPS_PER_BLOCK) {
        bestDist = 256;
        const uint8_t* queryDesc = queryBase + queryDescIdx * ORB_DESCRIPTOR_SIZE;

        for (int d = lane; d < referenceCount; d += 32) {
            const uint8_t* referenceDesc = sharedReference + d * ORB_DESCRIPTOR_SIZE;
            const int dist = hammingDistance256(queryDesc, referenceDesc);

            bestDist = min(bestDist, dist);
        }

        // warp reduction:
        // find best detection descriptor match
        // for this query descriptor.
        for (int offset = 16; offset > 0; offset >>= 1) {
            bestDist = min(bestDist, __shfl_down_sync(0xffffffff, bestDist, offset));
        }

        if (lane == 0) {
            descriptorBest[queryDescIdx] = bestDist;
        }
    }

    __syncthreads();

    // reduce descriptor matches into a single
    // normalized similarity score.

    if (warpId == 0) {
        int sum = 0;

        if (lane < queryCount) {
            sum = descriptorBest[lane];
        }

        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }

        if (lane == 0) {
            const float meanDist = static_cast<float>(sum) / static_cast<float>(queryCount);
            const float similarity = 1.0f - (meanDist / 256.0f);

            // cost convention:
            //
            // 0.0 = strong match
            // 1.0 = weak match
            costMatrix[queryIdx * numReferences + referenceIdx] = 1.0f - similarity;
        }
    }
}

void initializeBinaryMatcherScratch(BinaryMatcherScratch& scratch, int maxQueries, int maxReferences) {
    scratch.maxQueries = maxQueries;
    scratch.maxReferences = maxReferences;

    const size_t descriptorSetBytes = MAX_DESCRIPTOR_COUNT * ORB_DESCRIPTOR_SIZE;

    CUDA_CHECK(cudaMalloc(&scratch.d_queryDescriptors, maxQueries * descriptorSetBytes));
    CUDA_CHECK(cudaMalloc(&scratch.d_referenceDescriptors, maxReferences * descriptorSetBytes));
    CUDA_CHECK(cudaMalloc(&scratch.d_queryCounts, maxQueries * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&scratch.d_referenceCounts, maxReferences * sizeof(int)));

    CUDA_CHECK(cudaMalloc(&scratch.d_costMatrix, maxQueries * maxReferences * sizeof(float)));
}

void destroyBinaryMatcherScratch(BinaryMatcherScratch& scratch) {
    if (scratch.d_queryDescriptors) cudaFree(scratch.d_queryDescriptors);
    if (scratch.d_referenceDescriptors) cudaFree(scratch.d_referenceDescriptors);
    if (scratch.d_queryCounts) cudaFree(scratch.d_queryCounts);
    if (scratch.d_referenceCounts) cudaFree(scratch.d_referenceCounts);
    if (scratch.d_costMatrix) cudaFree(scratch.d_costMatrix);

    scratch = BinaryMatcherScratch{};
}

void computeBinaryCostMatrixCUDA(BinaryMatcherScratch& scratch,
                                        const BinaryDescriptorSet* queries,
                                        int numQueries,

                                        const BinaryDescriptorSet* references,
                                        int numReferences,

                                        float* h_costMatrix) {

    const size_t descriptorSetBytes = MAX_DESCRIPTOR_COUNT * ORB_DESCRIPTOR_SIZE;

    std::vector<uint8_t> queryDescriptors(numQueries * descriptorSetBytes);

    std::vector<uint8_t> referenceDescriptors(numReferences * descriptorSetBytes);

    std::vector<int> queryCounts(numQueries);
    std::vector<int> referenceCounts(numReferences);

    for (int i = 0; i < numQueries; ++i) {
        std::memcpy(queryDescriptors.data() + i * descriptorSetBytes,
                    queries[i].descriptors,
                    descriptorSetBytes);

        queryCounts[i] = queries[i].descriptorCount;
    }    

        for (int i = 0; i < numReferences; ++i) {
        std::memcpy(referenceDescriptors.data() + i * descriptorSetBytes,
                    references[i].descriptors,
                    descriptorSetBytes);

        referenceCounts[i] = references[i].descriptorCount;
    }

    CUDA_CHECK(cudaMemcpy(scratch.d_queryDescriptors,
                            queryDescriptors.data(),
                            numQueries * descriptorSetBytes,
                            cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(scratch.d_referenceDescriptors,
                            referenceDescriptors.data(),
                            numReferences * descriptorSetBytes,
                            cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(scratch.d_queryCounts,
                            queryCounts.data(),
                            numQueries * sizeof(int),
                            cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(scratch.d_referenceCounts,
                            referenceCounts.data(),
                            numReferences * sizeof(int),
                            cudaMemcpyHostToDevice));

    dim3 block(256);
    dim3 grid(numReferences, numQueries);

    binary_cost_matrix_kernel<<<grid, block>>>(scratch.d_queryDescriptors,
                                                scratch.d_queryCounts,
                                                scratch.d_referenceDescriptors,
                                                scratch.d_referenceCounts,
                                                numQueries,
                                                numReferences,
                                                scratch.d_costMatrix);

    CUDA_KERNEL_CHECK();

    CUDA_CHECK(cudaMemcpy(h_costMatrix,
                            scratch.d_costMatrix,
                            numQueries * numReferences * sizeof(float),
                            cudaMemcpyDeviceToHost));
}

float computeBinarySimilarityCUDA(BinaryMatcherScratch& scratch, const BinaryDescriptorSet& a,
                                                                    const BinaryDescriptorSet& b) {
    
    float cost = 1.0f;
    computeBinaryCostMatrixCUDA(scratch, &a, 1, &b, 1, &cost);

    return 1.0f - cost;
}