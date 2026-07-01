#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace SharedMem {

        StreamCompaction::Common::PerformanceTimer& timer();

        // ---- Kernels (exported for direct use by benchmarks) ----

        /// Naive shared-memory scan kernel (GPU Gems 3, Example 39-1).
        /// Launched with 1 block x n threads.
        /// Dynamic shared memory: 2 * n * sizeof(int).
        __global__ void kernNaiveSharedScan(int n, int *g_odata,
                                            const int *g_idata);

        /// Work-efficient shared-memory scan WITH bank-conflict avoidance
        /// (GPU Gems 3, Example 39-2 + Listing 39-3).
        /// Launched with 1 block x n_padded threads.
        /// Dynamic shared memory: (n_padded + #padding_slots) * sizeof(int).
        __global__ void kernWorkEfficientSharedScan(int n_original,
                                                    int n_padded,
                                                    int *g_odata,
                                                    const int *g_idata);

        /// Same as kernWorkEfficientSharedScan WITHOUT bank-conflict
        /// padding (for benchmarking comparison).
        __global__ void kernWorkEfficientSharedScanNoBC(int n_original,
                                                        int n_padded,
                                                        int *g_odata,
                                                        const int *g_idata);

        // ---- Host wrappers ----

        /// Naive shared-memory scan (GPU Gems 3, Sec.39.2.1, Example 39-1).
        /// Single-block: n <= blockSize (1024).  Includes memory operations
        /// and GPU timing internally.
        void scanNaive(int n, int *odata, const int *idata);

        /// Work-efficient shared-memory scan WITH bank-conflict avoidance.
        /// Single-block: n <= blockSize (1024).
        void scanWorkEfficient(int n, int *odata, const int *idata);

        /// Same as scanWorkEfficient but WITHOUT bank-conflict avoidance
        /// (for benchmarking comparison).
        void scanWorkEfficientNoBC(int n, int *odata, const int *idata);

    }
}
