#pragma once

#include <cstdint>

constexpr int ORB_DESCRIPTOR_SIZE = 32;
constexpr int MAX_DESCRIPTOR_COUNT = 32;

struct BinaryDescriptorSet {
    const uint8_t* descriptors = nullptr;
    int descriptorCount;
};

struct BinaryMatcherScratch {
    uint8_t* d_queryDescriptors = nullptr;
    uint8_t* d_referenceDescriptors = nullptr;

    int* d_queryCounts = nullptr;
    int* d_referenceCounts = nullptr;

    float* d_costMatrix = nullptr;

    int maxQueries = 0;
    int maxReferences = 0;
};

void initializeBinaryMatcherScratch(BinaryMatcherScratch& scratch, int maxQueries, int maxReferences);

void destroyBinaryMatcherScratch(BinaryMatcherScratch& scratch);

void computeBinaryCostMatrixCUDA(BinaryMatcherScratch& scratch,

                                    const BinaryDescriptorSet* queries,
                                    int numQueries,

                                    const BinaryDescriptorSet* references,
                                    int numReferences,

                                    float* costMatrix);

float computeBinarySimilarityCUDA(BinaryMatcherScratch& scratch,
                                    const BinaryDescriptorSet& query,
                                    const BinaryDescriptorSet& reference);
