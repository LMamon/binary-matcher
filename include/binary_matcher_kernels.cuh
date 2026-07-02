#pragma once

#include <cstdint>

__global__ void binary_cost_matrix_kernel(
    const uint8_t* queryDescriptors,
    const int* queryCounts,

    const uint8_t* referenceDescriptors,
    const int* referenceCounts,

    int numQueries,
    int numReferences,

    float* costMatrix);
    