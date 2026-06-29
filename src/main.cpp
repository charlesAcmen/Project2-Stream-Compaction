/**
 * @file      main.cpp
 * @brief     Stream compaction test program
 * @authors   Kai Ninomiya
 * @date      2015
 * @copyright University of Pennsylvania
 */

#include <cstdio>
#include <algorithm>
#include <stream_compaction/cpu.h>
#include <stream_compaction/naive.h>
#include <stream_compaction/efficient.h>
#include <stream_compaction/thrust.h>
#include <stream_compaction/radixsort.h>
#include "testing_helpers.hpp"

const int SIZE = 1 << 22;     // 4M elements for scan/compact
const int NPOT = SIZE - 3;    // Non-power-of-two
const int RADIX_SIZE = 1 << 8; // Small size for radix sort correctness

int *a = new int[SIZE];
int *b = new int[SIZE];
int *c = new int[SIZE];

/**
 * Helper: verify that an array is sorted in ascending order.
 */
bool verifySorted(int n, const int *arr) {
    for (int i = 1; i < n; i++) {
        if (arr[i] < arr[i - 1]) {
            printf("    NOT SORTED at index %d: arr[%d]=%d, arr[%d]=%d\n",
                   i, i - 1, arr[i - 1], i, arr[i]);
            return false;
        }
    }
    return true;
}

int main(int argc, char* argv[]) {
    // ========================================================================
    // SCAN TESTS
    // ========================================================================
    printf("\n");
    printf("****************\n");
    printf("** SCAN TESTS **\n");
    printf("****************\n");

    genArray(SIZE - 1, a, 50);
    a[SIZE - 1] = 0;
    printArray(SIZE, a, true);

    // CPU scan (reference)
    zeroArray(SIZE, b);
    printDesc("cpu scan, power-of-two");
    StreamCompaction::CPU::scan(SIZE, b, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(),
                     "(std::chrono Measured)");
    printArray(SIZE, b, true);

    zeroArray(SIZE, c);
    printDesc("cpu scan, non-power-of-two");
    StreamCompaction::CPU::scan(NPOT, c, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(),
                     "(std::chrono Measured)");
    printArray(NPOT, c, true);
    printCmpResult(NPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("naive scan, power-of-two");
    StreamCompaction::Naive::scan(SIZE, c, a);
    printElapsedTime(StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation(),
                     "(CUDA Measured)");
    printCmpResult(SIZE, b, c);

    zeroArray(SIZE, c);
    printDesc("naive scan, non-power-of-two");
    StreamCompaction::Naive::scan(NPOT, c, a);
    printElapsedTime(StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation(),
                     "(CUDA Measured)");
    printCmpResult(NPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient scan, power-of-two");
    StreamCompaction::Efficient::scan(SIZE, c, a);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(),
                     "(CUDA Measured)");
    printCmpResult(SIZE, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient scan, non-power-of-two");
    StreamCompaction::Efficient::scan(NPOT, c, a);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(),
                     "(CUDA Measured)");
    printCmpResult(NPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("thrust scan, power-of-two");
    StreamCompaction::Thrust::scan(SIZE, c, a);
    printElapsedTime(StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation(),
                     "(CUDA Measured)");
    printCmpResult(SIZE, b, c);

    zeroArray(SIZE, c);
    printDesc("thrust scan, non-power-of-two");
    StreamCompaction::Thrust::scan(NPOT, c, a);
    printElapsedTime(StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation(),
                     "(CUDA Measured)");
    printCmpResult(NPOT, b, c);

    // ========================================================================
    // STREAM COMPACTION TESTS
    // ========================================================================
    printf("\n");
    printf("*****************************\n");
    printf("** STREAM COMPACTION TESTS **\n");
    printf("*****************************\n");

    genArray(SIZE - 1, a, 4);
    a[SIZE - 1] = 0;
    printArray(SIZE, a, true);

    int count, expectedCount, expectedNPOT;

    // CPU compact without scan (reference)
    zeroArray(SIZE, b);
    printDesc("cpu compact without scan, power-of-two");
    count = StreamCompaction::CPU::compactWithoutScan(SIZE, b, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(),
                     "(std::chrono Measured)");
    expectedCount = count;
    printArray(count, b, true);
    printCmpLenResult(count, expectedCount, b, b);

    zeroArray(SIZE, c);
    printDesc("cpu compact without scan, non-power-of-two");
    count = StreamCompaction::CPU::compactWithoutScan(NPOT, c, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(),
                     "(std::chrono Measured)");
    expectedNPOT = count;
    printArray(count, c, true);
    printCmpLenResult(count, expectedNPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("cpu compact with scan");
    count = StreamCompaction::CPU::compactWithScan(SIZE, c, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(),
                     "(std::chrono Measured)");
    printArray(count, c, true);
    printCmpLenResult(count, expectedCount, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient compact, power-of-two");
    count = StreamCompaction::Efficient::compact(SIZE, c, a);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(),
                     "(CUDA Measured)");
    printCmpLenResult(count, expectedCount, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient compact, non-power-of-two");
    count = StreamCompaction::Efficient::compact(NPOT, c, a);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(),
                     "(CUDA Measured)");
    printCmpLenResult(count, expectedNPOT, b, c);

    // ========================================================================
    // RADIX SORT TESTS
    // ========================================================================
    printf("\n");
    printf("************************\n");
    printf("** RADIX SORT TESTS   **\n");
    printf("************************\n");

    // -- Test 1: small fixed array, non-power-of-two --
    {
        int small[] = { 7, 0, 9, 3, 1, 5, 0, 2, 8, 4 };
        int nSmall = sizeof(small) / sizeof(small[0]);
        int smallSorted[10];
        memcpy(smallSorted, small, nSmall * sizeof(int));
        std::sort(smallSorted, smallSorted + nSmall);

        printf("\n--- Test 1: small fixed array (n=%d) ---\n", nSmall);
        printf("    input:      [ ");
        for (int i = 0; i < nSmall; i++) printf("%d ", small[i]);
        printf("]\n");
        printf("    expected:   [ ");
        for (int i = 0; i < nSmall; i++) printf("%d ", smallSorted[i]);
        printf("]\n");

        StreamCompaction::RadixSort::radixsort(nSmall, small);
        printElapsedTime(
            StreamCompaction::RadixSort::timer().getGpuElapsedTimeForPreviousOperation(),
            "(CUDA Measured)");

        printf("    actual:     [ ");
        for (int i = 0; i < nSmall; i++) printf("%d ", small[i]);
        printf("]\n");

        bool ok = verifySorted(nSmall, small) && (cmpArrays(nSmall, small, smallSorted) == 0);
        printf("    %s\n", ok ? "passed" : "FAIL VALUE");
    }

    // -- Test 2: all zeros --
    {
        int allZeros[] = { 0, 0, 0, 0, 0, 0, 0, 0 };
        int nZ = 8;
        printf("\n--- Test 2: all zeros (n=%d) ---\n", nZ);
        StreamCompaction::RadixSort::radixsort(nZ, allZeros);
        printElapsedTime(
            StreamCompaction::RadixSort::timer().getGpuElapsedTimeForPreviousOperation(),
            "(CUDA Measured)");
        bool ok = verifySorted(nZ, allZeros);
        printf("    %s\n", ok ? "passed" : "FAIL");
    }

    // -- Test 3: sorted already --
    {
        int sorted[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
        int nS = 10;
        printf("\n--- Test 3: already sorted (n=%d) ---\n", nS);
        StreamCompaction::RadixSort::radixsort(nS, sorted);
        printElapsedTime(
            StreamCompaction::RadixSort::timer().getGpuElapsedTimeForPreviousOperation(),
            "(CUDA Measured)");
        bool ok = verifySorted(nS, sorted);
        printf("    %s\n", ok ? "passed" : "FAIL");
    }

    // -- Test 4: reverse order --
    {
        int rev[] = { 9, 8, 7, 6, 5, 4, 3, 2, 1, 0 };
        int nR = 10;
        printf("\n--- Test 4: reverse order (n=%d) ---\n", nR);
        StreamCompaction::RadixSort::radixsort(nR, rev);
        printElapsedTime(
            StreamCompaction::RadixSort::timer().getGpuElapsedTimeForPreviousOperation(),
            "(CUDA Measured)");
        bool ok = verifySorted(nR, rev);
        printf("    %s\n", ok ? "passed" : "FAIL");
    }

    // -- Test 5: all same value --
    {
        int same[] = { 5, 5, 5, 5, 5, 5, 5, 5 };
        int nSame = 8;
        printf("\n--- Test 5: all same value (n=%d) ---\n", nSame);
        StreamCompaction::RadixSort::radixsort(nSame, same);
        printElapsedTime(
            StreamCompaction::RadixSort::timer().getGpuElapsedTimeForPreviousOperation(),
            "(CUDA Measured)");
        bool ok = verifySorted(nSame, same);
        printf("    %s\n", ok ? "passed" : "FAIL");
    }

    // -- Test 6: single element --
    {
        int one[] = { 42 };
        printf("\n--- Test 6: single element (n=1) ---\n");
        StreamCompaction::RadixSort::radixsort(1, one);
        printElapsedTime(
            StreamCompaction::RadixSort::timer().getGpuElapsedTimeForPreviousOperation(),
            "(CUDA Measured)");
        printf("    %s\n", (one[0] == 42) ? "passed" : "FAIL");
    }

    // -- Test 7: power-of-two size random array --
    {
        printf("\n--- Test 7: random array, power-of-two (n=%d) ---\n", RADIX_SIZE);
        int *randomData = new int[RADIX_SIZE];
        srand(42);
        for (int i = 0; i < RADIX_SIZE; i++) {
            randomData[i] = rand() % 1000;
        }
        int *cpuSorted = new int[RADIX_SIZE];
        memcpy(cpuSorted, randomData, RADIX_SIZE * sizeof(int));
        std::sort(cpuSorted, cpuSorted + RADIX_SIZE);

        StreamCompaction::RadixSort::radixsort(RADIX_SIZE, randomData);
        printElapsedTime(
            StreamCompaction::RadixSort::timer().getGpuElapsedTimeForPreviousOperation(),
            "(CUDA Measured)");
        bool ok = verifySorted(RADIX_SIZE, randomData) &&
                  (cmpArrays(RADIX_SIZE, randomData, cpuSorted) == 0);
        printf("    %s\n", ok ? "passed" : "FAIL");
        delete[] randomData;
        delete[] cpuSorted;
    }

    // -- Test 8: non-power-of-two random array --
    {
        int nNonPot = RADIX_SIZE - 3;
        printf("\n--- Test 8: random array, non-power-of-two (n=%d) ---\n", nNonPot);
        int *randomData = new int[nNonPot];
        srand(99);
        for (int i = 0; i < nNonPot; i++) {
            randomData[i] = rand() % 5000;
        }
        int *cpuSorted = new int[nNonPot];
        memcpy(cpuSorted, randomData, nNonPot * sizeof(int));
        std::sort(cpuSorted, cpuSorted + nNonPot);

        StreamCompaction::RadixSort::radixsort(nNonPot, randomData);
        printElapsedTime(
            StreamCompaction::RadixSort::timer().getGpuElapsedTimeForPreviousOperation(),
            "(CUDA Measured)");
        bool ok = verifySorted(nNonPot, randomData) &&
                  (cmpArrays(nNonPot, randomData, cpuSorted) == 0);
        printf("    %s\n", ok ? "passed" : "FAIL");
        delete[] randomData;
        delete[] cpuSorted;
    }

    // -- Test 9: large values (up to 2^30) --
    {
        int nLargeVal = 1000;
        printf("\n--- Test 9: large values (n=%d, max~2^30) ---\n", nLargeVal);
        int *largeData = new int[nLargeVal];
        srand(2025);
        for (int i = 0; i < nLargeVal; i++) {
            // Generate non-negative values up to ~2^30
            largeData[i] = (rand() * 12345 + rand()) & 0x3FFFFFFF;
        }
        int *cpuSorted = new int[nLargeVal];
        memcpy(cpuSorted, largeData, nLargeVal * sizeof(int));
        std::sort(cpuSorted, cpuSorted + nLargeVal);

        StreamCompaction::RadixSort::radixsort(nLargeVal, largeData);
        printElapsedTime(
            StreamCompaction::RadixSort::timer().getGpuElapsedTimeForPreviousOperation(),
            "(CUDA Measured)");
        bool ok = verifySorted(nLargeVal, largeData) &&
                  (cmpArrays(nLargeVal, largeData, cpuSorted) == 0);
        printf("    %s\n", ok ? "passed" : "FAIL");
        delete[] largeData;
        delete[] cpuSorted;
    }

    printf("\n");

    system("pause"); // stop Win32 console from closing on exit
    delete[] a;
    delete[] b;
    delete[] c;
}
