#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Naive {
        StreamCompaction::Common::PerformanceTimer& timer();

        void scan(int n, int *odata, const int *idata);

        // Kernel exported for direct use by benchmarks
        __global__ void kernNaiveScan(int n, int d, int *odata, const int *idata);
    }
}
