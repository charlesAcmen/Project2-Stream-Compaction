#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace RadixSort {
        StreamCompaction::Common::PerformanceTimer& timer();
        //sort array with non-negative elements in an ascending order
        void radixsort(int n, int *data);
    }
}
