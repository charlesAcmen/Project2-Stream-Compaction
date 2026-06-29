#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace RadixSort {
        StreamCompaction::Common::PerformanceTimer& timer();

        /**
         * Maps an array of ints to an array of 0s and 1s.
         * For each element in idata, if bit b of idata[idx] == 0,
         * then bools[idx] = 1; otherwise bools[idx] = 0.
         * This marks elements with bit=0 for priority placement
         * (ascending order radix sort).
         *
         * @param n       Number of elements
         * @param bools   Output boolean array (1 if bit=0, 0 if bit=1)
         * @param idata   Input data array
         * @param bitPos  Bit position to extract (0 = LSB)
         */
        __global__ void kernMapToBoolean(int n, int *bools, const int *idata, int bitPos);

        /**
         * Performs an exclusive scan on the bools array.
         * (Placeholder -- the actual implementation reuses
         * Efficient::kernUpSweep / Efficient::kernDownSweep.)
         */
        __global__ void kernExclusiveScan(int n, int *scanned, const int *bools);

        /**
         * Scatters the elements in idata into odata for one radix-sort pass.
         * Elements with bit=0 (bools=1) go to the front section at scanned[idx].
         * Elements with bit=1 (bools=0) go to the back section after all zeros.
         *
         * @param n          Number of elements
         * @param odata      Output array
         * @param idata      Input array
         * @param bools      1 if bit=0, 0 if bit=1
         * @param scanned    Exclusive scan indices for bit=0 elements
         * @param totalZeros Total count of elements with bit=0
         */
        __global__ void kernScatter(int n, int *odata,
                                     const int *idata, const int *bools,
                                     const int *scanned, int totalZeros);

        /**
         * Sorts an array of non-negative integers in ascending order.
         * Uses LSB radix sort with work-efficient parallel scan.
         *
         * @param n     Number of elements in data.
         * @param data  The array to sort (sorted in-place).
         */
        void radixsort(int n, int *data);
    }
}
