#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace RadixSort {
        StreamCompaction::Common::PerformanceTimer& timer();

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
