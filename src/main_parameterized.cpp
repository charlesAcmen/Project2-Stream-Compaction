/**
 * @file      main_parameterized.cpp
 * @brief     Parameterized radix sort performance test
 * @authors   Based on Kai Ninomiya's original main.cpp
 * @date      2025
 *
 * Tests GPU radix sort against CPU std::sort with configurable
 * array size and value range. All numbers are non-negative.
 *
 * Usage:
 *   radixsort_test -n <size> -m <max_value>
 *   radixsort_test -h            (show help)
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <stream_compaction/radixsort.h>

// ---------------------------------------------------------------------------
// Default parameters
// ---------------------------------------------------------------------------
int SIZE      = 1 << 20;   // 1M elements
int MAX_VALUE = 10000;     // Non-negative values only

// ---------------------------------------------------------------------------
// Print usage information
// ---------------------------------------------------------------------------
void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("\nOptions:\n");
    printf("  -n <size>    Number of elements (default: 1048576)\n");
    printf("  -m <max>     Maximum value (default: 10000, must be >= 0)\n");
    printf("  -h           Show this help\n");
    printf("\nExamples:\n");
    printf("  %s -n 1000000 -m 50000\n", progName);
    printf("  %s -n 5000000\n", progName);
}

// ---------------------------------------------------------------------------
// Generate random non-negative integers (fixed seed for reproducibility)
// ---------------------------------------------------------------------------
void genRandomArray(int n, int *a, int maxValue) {
    srand(2025);
    for (int i = 0; i < n; i++) {
        if (maxValue > RAND_MAX) {
            long long val = ((long long)rand() * (RAND_MAX + 1) + rand()) % (maxValue + 1);
            a[i] = (int)val;
        } else {
            a[i] = rand() % (maxValue + 1);
        }
    }
}

// ---------------------------------------------------------------------------
// Verify an array is sorted in ascending order
// ---------------------------------------------------------------------------
bool verifySorted(int n, const int *arr) {
    for (int i = 1; i < n; i++) {
        if (arr[i] < arr[i - 1]) {
            printf("  ERROR: arr[%d]=%d > arr[%d]=%d\n",
                   i - 1, arr[i - 1], i, arr[i]);
            return false;
        }
    }
    return true;
}

// =======================================================================
int main(int argc, char* argv[]) {
    // ---- Parse command line ----
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            SIZE = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            MAX_VALUE = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-h") == 0) {
            printUsage(argv[0]);
            return 0;
        } else {
            printf("Unknown option: %s\n\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }

    if (SIZE <= 0) {
        printf("Error: size must be positive\n");
        return 1;
    }
    if (MAX_VALUE < 0) {
        printf("Error: max value must be non-negative\n");
        return 1;
    }

    // ---- Header ----
    printf("\n");
    printf("========================================\n");
    printf("      RADIX SORT PERFORMANCE TEST\n");
    printf("========================================\n");
    printf("  Array size:     %d\n", SIZE);
    printf("  Max value:      %d\n", MAX_VALUE);
    printf("  Memory:         %.2f MB (per array)\n",
           SIZE * sizeof(int) / (1024.0 * 1024.0));
    printf("========================================\n\n");

    // ---- Allocate and generate data ----
    int *data_gpu = new int[SIZE];
    int *data_cpu = new int[SIZE];

    printf("Generating random data...\n");
    genRandomArray(SIZE, data_gpu, MAX_VALUE);
    memcpy(data_cpu, data_gpu, SIZE * sizeof(int));

    int minVal = *std::min_element(data_gpu, data_gpu + SIZE);
    int maxVal = *std::max_element(data_gpu, data_gpu + SIZE);
    printf("  Data range: [%d, %d]\n", minVal, maxVal);

    // Sample input
    int nSample = (SIZE < 20) ? SIZE : 20;
    printf("  Sample input (first %d):  [ ", nSample);
    for (int i = 0; i < nSample; i++) printf("%d ", data_gpu[i]);
    printf("]\n\n");

    // ====================================================================
    // GPU Radix Sort
    // ====================================================================
    printf("---- GPU Radix Sort ----\n");
    StreamCompaction::RadixSort::radixsort(SIZE, data_gpu);
    float gpuTime =
        StreamCompaction::RadixSort::timer().getGpuElapsedTimeForPreviousOperation();
    printf("  elapsed: %.6f ms (CUDA Measured)\n", gpuTime);

    bool gpuOk = verifySorted(SIZE, data_gpu);
    printf("  %s\n", gpuOk ? "PASSED" : "FAILED");

    printf("  Sample output (first %d): [ ", nSample);
    for (int i = 0; i < nSample; i++) printf("%d ", data_gpu[i]);
    printf("]\n");
    printf("  Sample output (last %d):  [ ",
           (SIZE < 20) ? SIZE : 20);
    for (int i = (SIZE < 20 ? 0 : SIZE - 20); i < SIZE; i++)
        printf("%d ", data_gpu[i]);
    printf("]\n\n");

    // ====================================================================
    // CPU std::sort (baseline)
    // ====================================================================
    printf("---- CPU std::sort ----\n");
    StreamCompaction::RadixSort::timer().startCpuTimer();
    std::sort(data_cpu, data_cpu + SIZE);
    StreamCompaction::RadixSort::timer().endCpuTimer();
    float cpuTime =
        StreamCompaction::RadixSort::timer().getCpuElapsedTimeForPreviousOperation();
    printf("  elapsed: %.6f ms (std::chrono Measured)\n", cpuTime);
    printf("  %s\n", verifySorted(SIZE, data_cpu) ? "PASSED" : "FAILED");

    // ====================================================================
    // Comparison
    // ====================================================================
    printf("\n---- Comparison ----\n");

    bool match = true;
    for (int i = 0; i < SIZE; i++) {
        if (data_gpu[i] != data_cpu[i]) {
            printf("  Mismatch at [%d]: GPU=%d  CPU=%d\n",
                   i, data_gpu[i], data_cpu[i]);
            match = false;
            break;
        }
    }
    printf("  Results: %s\n", match ? "MATCH" : "DO NOT MATCH");

    printf("\n  Performance:\n");
    printf("    CPU:  %.6f ms\n", cpuTime);
    printf("    GPU:  %.6f ms\n", gpuTime);
    if (gpuTime < cpuTime) {
        printf("    Speedup:  %.2fx (GPU faster)\n", cpuTime / gpuTime);
    } else {
        printf("    Ratio:    %.2fx (CPU faster)\n", gpuTime / cpuTime);
    }

    printf("\n");

    delete[] data_gpu;
    delete[] data_cpu;
    return 0;
}
