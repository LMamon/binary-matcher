#include <iostream>
#include <vector>
#include <random>
#include <iomanip>

#include "binary_matcher.cuh"

#define CUDA_EXPRESSION_CHECKER
#include "cuda_check.h"

constexpr int NUM_QUERIES = 32;
constexpr int NUM_REFERENCES = 32;

int main() {
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<int> dist(0, 255);

    constexpr size_t descriptorSetBytes =MAX_DESCRIPTOR_COUNT * ORB_DESCRIPTOR_SIZE;

    std::vector<uint8_t> queryStorage(NUM_QUERIES * descriptorSetBytes);
    std::vector<uint8_t> referenceStorage(NUM_REFERENCES * descriptorSetBytes);

    for (auto& b : queryStorage)
        b = static_cast<uint8_t>(dist(rng));

    for (auto& b : referenceStorage)
        b = static_cast<uint8_t>(dist(rng));

    std::vector<BinaryDescriptorSet> queries(NUM_QUERIES);
    std::vector<BinaryDescriptorSet> references(NUM_REFERENCES);

    for (int i = 0; i < NUM_QUERIES; ++i){
        queries[i].descriptors = queryStorage.data() + i * descriptorSetBytes;
        queries[i].descriptorCount = MAX_DESCRIPTOR_COUNT;
    }

    for (int i = 0; i < NUM_REFERENCES; ++i) {
        references[i].descriptors = referenceStorage.data() + i * descriptorSetBytes;
        references[i].descriptorCount = MAX_DESCRIPTOR_COUNT;
    }

    BinaryMatcherScratch scratch;

    initializeBinaryMatcherScratch(scratch, NUM_QUERIES, NUM_REFERENCES);
    constexpr int ITERATIONS = 100;

    cudaEvent_t start;
    cudaEvent_t stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::vector<float> costMatrix(NUM_QUERIES * NUM_REFERENCES);

    // Warmup
    computeBinaryCostMatrixCUDA(scratch,
                                queries.data(),
                                NUM_QUERIES,
                                references.data(),
                                NUM_REFERENCES,
                                costMatrix.data());

    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < ITERATIONS; ++i) {
        computeBinaryCostMatrixCUDA(scratch,
                                    queries.data(),
                                    NUM_QUERIES,
                                    references.data(),
                                    NUM_REFERENCES,
                                    costMatrix.data());
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float totalRuntimeMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&totalRuntimeMs, start, stop));

    const float meanRuntimeMs = totalRuntimeMs / static_cast<float>(ITERATIONS);

    float similarity = computeBinarySimilarityCUDA(scratch,
                                                    queries[0],
                                                    references[0]);

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "========================================\n";
    std::cout << "Binary Matcher\n";
    std::cout << "----------------------------------------\n";
    std::cout << "Queries        : " << NUM_QUERIES << '\n';
    std::cout << "References     : " << NUM_REFERENCES << '\n';
    std::cout << "Descriptors    : "
            << MAX_DESCRIPTOR_COUNT
            << " x "
            << ORB_DESCRIPTOR_SIZE * 8
            << "-bit\n";
    std::cout << "Iterations     : " << ITERATIONS << '\n';
    std::cout << "Warmup         : complete\n";
    std::cout << "Total Runtime  : " << totalRuntimeMs << " ms\n";
    std::cout << "Mean Runtime   : " << meanRuntimeMs << " ms\n";
    std::cout << "----------------------------------------\n";
    std::cout << "Similarity     : " << similarity << '\n';
    std::cout << "First Cost     : " << costMatrix.front() << '\n';
    std::cout << "========================================\n";

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    destroyBinaryMatcherScratch(scratch);

    return 0;
}