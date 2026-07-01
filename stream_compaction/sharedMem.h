#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace SharedMem {

        StreamCompaction::Common::PerformanceTimer& timer();

        /**
         * Naive shared-memory scan (GPU Gems 3, Section 39.2.1, Example 39-1).
         *
         * Implements the Hillis & Steele (1986) parallel scan using a single
         * thread block and shared memory with double-buffering to avoid race
         * conditions.  O(n log n) work -- not work-efficient, but included for
         * comparison with the global-memory naive scan and the work-efficient
         * shared-memory scan.
         *
         * This is a single-block kernel: n must be <= blockSize (128).
         * Cross-block support is not implemented yet.
         */
        void scanNaive(int n, int *odata, const int *idata);

        /**
         * Work-efficient shared-memory scan with bank-conflict avoidance
         * (GPU Gems 3, Sections 39.2.2-39.2.3, Example 39-2 + Listing 39-3).
         *
         * Implements the Blelloch (1990) tree-based two-phase algorithm
         * (up-sweep reduce + down-sweep distribute) entirely in shared memory
         * within a single thread block.  O(n) work; each thread handles one
         * element.
         *
         * Bank conflicts are eliminated by padding the shared-memory array
         * with CONFLICT_FREE_OFFSET so that strided accesses in the tree
         * traversal never alias the same bank (see Section 39.2.3).
         *
         * This is a single-block kernel: n must be <= blockSize (128).
         * Cross-block support is not implemented yet.
         */
        void scanWorkEfficient(int n, int *odata, const int *idata);

        /**
         * Same work-efficient algorithm as scanWorkEfficient but WITHOUT
         * bank-conflict avoidance.  Kept for benchmarking comparison.
         */
        void scanWorkEfficientNoBC(int n, int *odata, const int *idata);

    }
}
