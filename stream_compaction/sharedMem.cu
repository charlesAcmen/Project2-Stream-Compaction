#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include "common.h"
#include "sharedMem.h"

namespace StreamCompaction {
    namespace SharedMem {

        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        // ---------------------------------------------------------------
        // Constants
        // ---------------------------------------------------------------

        /** Number of threads per block -- also the max elements
         *  a single-block shared-memory scan can handle.
         *  Modern GPUs support up to 1024 threads/block.
         *  Shared memory footprint at n=1024:
         *    naive:    2 x 1024 x 4 B = 8 KB
         *    efficient: (1024 + 1024/32) x 4 B ≈ 4.2 KB
         *  Both fit comfortably within the 48 KB / 100 KB per-SM limit. */
        const int blockSize = 1024;

        // ---------------------------------------------------------------
        // Bank-conflict avoidance (GPU Gems 3, Section 39.2.3)
        // ---------------------------------------------------------------

        /**
         * Modern NVIDIA GPUs (Compute Capability >= 3.0) have 32 shared-
         * memory banks, each 4 bytes wide.  The Blelloch tree scan
         * accesses memory with stride = 2*offset, which causes multiple
         * threads in the same warp to hit the same bank.
         *
         * CONFLICT_FREE_OFFSET(i) adds one padding slot every NUM_BANKS
         * elements, spreading strided accesses across different banks.
         *
         * For NUM_BANKS = 32:  every 32 logical elements we skip 1 slot.
         */
        #define NUM_BANKS       32
        #define LOG_NUM_BANKS   5

        /** Compute the number of padding slots to insert before logical
         *  index (n).  The formula is (n / NUM_BANKS), i.e. skip one
         *  slot every 32 elements.
         *
         *  @param  n   Logical (unpadded) array index.
         *  @return     Number of extra slots to skip.                    */
        __host__ __device__ inline int conflictFreeOffset(int n) {
            return n >> LOG_NUM_BANKS;          // n / 32
        }

        // ---------------------------------------------------------------
        // Kernel: naive shared-memory scan  (Example 39-1 / Sec.39.2.1)
        // ---------------------------------------------------------------

        /**
         * Naive Hillis & Steele parallel scan using shared memory.
         *
         * Double-buffers inside the dynamic shared-memory array `temp[]`
         * to avoid race conditions between reads and writes.  Each
         * iteration doubles the stride distance until the full prefix
         * sum is computed.
         *
         * @param  n          Number of elements to scan.
         * @param  g_odata    Output array (device pointer).
         * @param  g_idata    Input array  (device pointer).
         *
         * @note   Launched with 1 block x n threads.
         * @note   Dynamic shared memory = 2 * n * sizeof(int).           */
        __global__ void kernNaiveSharedScan(int n, int *g_odata,
                                            const int *g_idata)
        {
            // -- Double-buffered shared memory:
            //    temp[0..n-1]   = buffer 0 (pin / pout)
            //    temp[n..2n-1]  = buffer 1
            extern __shared__ int temp[];

            int thid = threadIdx.x;
            if (thid >= n) return;

            int pout = 0, pin = 1;

            // ---- Initial load (right-shift by 1 for exclusive scan) ----
            // Example 39-1 loads g_idata[thid-1] so that the inclusive
            // scan naturally becomes exclusive.
            temp[pout * n + thid] = (thid > 0) ? g_idata[thid - 1] : 0;
            __syncthreads();

            // ---- Iterative doubling (ping-pong between buffers) ----
            for (int offset = 1; offset < n; offset <<= 1) {
                pout = 1 - pout;           // swap write / read
                pin  = 1 - pout;

                if (thid >= offset) {
                    temp[pout * n + thid] =
                        temp[pin * n + thid] +
                        temp[pin * n + thid - offset];
                } else {
                    temp[pout * n + thid] = temp[pin * n + thid];
                }
                __syncthreads();
            }

            // ---- Write result ----
            g_odata[thid] = temp[pout * n + thid];
        }

        // ---------------------------------------------------------------
        // Kernel: work-efficient scan WITH bank-conflict avoidance
        //         (Example 39-2 + Listing 39-3 / Sec.39.2.2-39.2.3)
        // ---------------------------------------------------------------

        /**
         * Blelloch (1990) work-efficient exclusive scan, single block,
         * with bank-conflict-free padding.
         *
         * Algorithm overview:
         *   1. Load  n_original  elements into shared memory (with
         *      padding); zero-fill padded threads.
         *   2. Up-sweep  (reduce):  build partial sums bottom-up.
         *   3. Zero the root  (exclusive scan).
         *   4. Down-sweep (distribute): propagate partial sums top-down.
         *   5. Write back results for the original elements only.
         *
         * @param  n_original  Real element count (<= blockSize).
         * @param  n_padded    Power-of-2 padded count for the tree.
         * @param  g_odata     Output array (device pointer).
         * @param  g_idata     Input array  (device pointer).
         *
         * @note   Launched with 1 block x n_padded threads.
         * @note   Dynamic shared memory =
         *         (n_padded + #padding_slots) * sizeof(int).            */
        __global__ void kernWorkEfficientSharedScan(int n_original,
                                                    int n_padded,
                                                    int *g_odata,
                                                    const int *g_idata)
        {
            extern __shared__ int temp[];   // padded shared memory

            int thid = threadIdx.x;

            // ---- Load data into padded shared memory ----
            // Extra threads (thid >= n_original) contribute zero.
            int padded_i = thid + conflictFreeOffset(thid);
            temp[padded_i] = (thid < n_original) ? g_idata[thid] : 0;
            __syncthreads();

            // ---- Up-sweep (reduce) phase ----
            int offset = 1;
            for (int d = n_padded >> 1; d > 0; d >>= 1) {
                __syncthreads();   // wait for previous iteration

                if (thid < d) {
                    int ai = offset * (2 * thid + 1) - 1;
                    int bi = offset * (2 * thid + 2) - 1;
                    ai += conflictFreeOffset(ai);
                    bi += conflictFreeOffset(bi);
                    temp[bi] += temp[ai];
                }
                offset <<= 1;
            }

            // ---- Clear the last element (root -> exclusive scan) ----
            if (thid == 0) {
                int last = (n_padded - 1) +
                           conflictFreeOffset(n_padded - 1);
                temp[last] = 0;
            }

            // ---- Down-sweep (distribute) phase ----
            for (int d = 1; d < n_padded; d <<= 1) {
                offset >>= 1;
                __syncthreads();

                if (thid < d) {
                    int ai = offset * (2 * thid + 1) - 1;
                    int bi = offset * (2 * thid + 2) - 1;
                    ai += conflictFreeOffset(ai);
                    bi += conflictFreeOffset(bi);

                    int t = temp[ai];
                    temp[ai] = temp[bi];
                    temp[bi] += t;
                }
            }
            __syncthreads();

            // ---- Write results (original elements only) ----
            if (thid < n_original) {
                g_odata[thid] = temp[padded_i];
            }
        }

        // ---------------------------------------------------------------
        // Kernel: work-efficient scan WITHOUT bank-conflict avoidance
        //         (for benchmarking comparison)
        // ---------------------------------------------------------------

        /**
         * Identical to kernWorkEfficientSharedScan, but skips the
         * conflictFreeOffset() padding so bank-conflicts are preserved.
         * Useful for measuring the benefit of the padding technique.     */
        __global__ void kernWorkEfficientSharedScanNoBC(int n_original,
                                                        int n_padded,
                                                        int *g_odata,
                                                        const int *g_idata)
        {
            extern __shared__ int temp[];

            int thid = threadIdx.x;

            // ---- Load ----
            temp[thid] = (thid < n_original) ? g_idata[thid] : 0;
            __syncthreads();

            // ---- Up-sweep ----
            int offset = 1;
            for (int d = n_padded >> 1; d > 0; d >>= 1) {
                __syncthreads();

                if (thid < d) {
                    int ai = offset * (2 * thid + 1) - 1;
                    int bi = offset * (2 * thid + 2) - 1;
                    temp[bi] += temp[ai];
                }
                offset <<= 1;
            }

            // ---- Clear root ----
            if (thid == 0) {
                temp[n_padded - 1] = 0;
            }

            // ---- Down-sweep ----
            for (int d = 1; d < n_padded; d <<= 1) {
                offset >>= 1;
                __syncthreads();

                if (thid < d) {
                    int ai = offset * (2 * thid + 1) - 1;
                    int bi = offset * (2 * thid + 2) - 1;

                    int t = temp[ai];
                    temp[ai] = temp[bi];
                    temp[bi] += t;
                }
            }
            __syncthreads();

            // ---- Write ----
            if (thid < n_original) {
                g_odata[thid] = temp[thid];
            }
        }

        // ===============================================================
        // Host-side wrappers
        // ===============================================================

        /**
         * Naive shared-memory scan (GPU Gems 3 Sec.39.2.1, Example 39-1).
         *
         * Single-block kernel:  only n <= blockSize elements are scanned.
         * If n > blockSize, only the first blockSize elements are
         * processed (cross-block support is left for future work).
         *
         * The naive algorithm handles non-power-of-2 n gracefully
         * without padding.                                                 */
        void scanNaive(int n, int *odata, const int *idata) {
            if (n <= 0) return;

            // Single-block only
            int effectiveN = std::min(n, blockSize);

            // ---- Device allocations ----
            int *dev_idata, *dev_odata;
            cudaMalloc((void**)&dev_idata, effectiveN * sizeof(int));
            cudaMalloc((void**)&dev_odata, effectiveN * sizeof(int));
            checkCUDAError("scanNaive: cudaMalloc failed");

            cudaMemcpy(dev_idata, idata,
                       effectiveN * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("scanNaive: cudaMemcpy H->D failed");

            // ---- Launch (dynamic shared memory = 2 buffers) ----
            size_t sharedMemBytes = 2 * effectiveN * sizeof(int);

            timer().startGpuTimer();
            kernNaiveSharedScan<<<1, effectiveN, sharedMemBytes>>>(
                effectiveN, dev_odata, dev_idata);
            checkCUDAError("scanNaive: kernNaiveSharedScan failed");
            timer().endGpuTimer();

            // ---- Copy result back ----
            cudaMemcpy(odata, dev_odata,
                       effectiveN * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("scanNaive: cudaMemcpy D->H failed");

            // ---- Cleanup ----
            cudaFree(dev_idata);
            cudaFree(dev_odata);
        }

        /**
         * Helper: compute the padded shared-memory size for the
         * work-efficient kernel.
         *
         * @param  nPadded   Logical element count (must be power of 2).
         * @return           Number of int slots needed (nPadded +
         *                   padding).                                       */
        static int paddedSharedMemSlots(int nPadded) {
            int maxLogical = nPadded - 1;
            return maxLogical + conflictFreeOffset(maxLogical) + 1;
        }

        /**
         * Work-efficient shared-memory scan WITH bank-conflict avoidance
         * (GPU Gems 3, Sec.39.2.2-39.2.3, Example 39-2 + Listing 39-3).
         *
         * Single-block: only n <= blockSize is processed.  Non-power-of-2
         * arrays are handled by padding with zeros to the next power of
         * two.
         *
         * Cross-block support is NOT implemented yet; n > blockSize
         * silently clamps to blockSize.                                     */
        void scanWorkEfficient(int n, int *odata, const int *idata) {
            if (n <= 0) return;

            // Single-block only
            int effectiveN = std::min(n, blockSize);

            // Round up to next power of 2 for the binary tree
            int paddedN = 1 << ilog2ceil(effectiveN);

            // Determine shared-memory size (with bank-conflict padding)
            int smemSlots = paddedSharedMemSlots(paddedN);
            size_t sharedMemBytes = smemSlots * sizeof(int);

            // ---- Device allocations ----
            int *dev_idata, *dev_odata;
            cudaMalloc((void**)&dev_idata, effectiveN * sizeof(int));
            cudaMalloc((void**)&dev_odata, effectiveN * sizeof(int));
            checkCUDAError("scanWorkEfficient: cudaMalloc failed");

            cudaMemcpy(dev_idata, idata,
                       effectiveN * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("scanWorkEfficient: cudaMemcpy H->D failed");

            // ---- Launch ----
            timer().startGpuTimer();
            kernWorkEfficientSharedScan<<<1, paddedN, sharedMemBytes>>>(
                effectiveN, paddedN, dev_odata, dev_idata);
            checkCUDAError(
                "scanWorkEfficient: kernWorkEfficientSharedScan failed");
            timer().endGpuTimer();

            // ---- Copy result back ----
            cudaMemcpy(odata, dev_odata,
                       effectiveN * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("scanWorkEfficient: cudaMemcpy D->H failed");

            // ---- Cleanup ----
            cudaFree(dev_idata);
            cudaFree(dev_odata);
        }

        /**
         * Same as scanWorkEfficient but WITHOUT bank-conflict avoidance.
         * Useful for measuring the performance impact of the padding
         * technique.                                                        */
        void scanWorkEfficientNoBC(int n, int *odata, const int *idata) {
            if (n <= 0) return;

            int effectiveN = std::min(n, blockSize);
            int paddedN = 1 << ilog2ceil(effectiveN);
            size_t sharedMemBytes = paddedN * sizeof(int);  // no padding

            int *dev_idata, *dev_odata;
            cudaMalloc((void**)&dev_idata, effectiveN * sizeof(int));
            cudaMalloc((void**)&dev_odata, effectiveN * sizeof(int));
            checkCUDAError("scanWorkEfficientNoBC: cudaMalloc failed");

            cudaMemcpy(dev_idata, idata,
                       effectiveN * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("scanWorkEfficientNoBC: cudaMemcpy H->D failed");

            timer().startGpuTimer();
            kernWorkEfficientSharedScanNoBC<<<1, paddedN, sharedMemBytes>>>(
                effectiveN, paddedN, dev_odata, dev_idata);
            checkCUDAError(
                "scanWorkEfficientNoBC: kernel launch failed");
            timer().endGpuTimer();

            cudaMemcpy(odata, dev_odata,
                       effectiveN * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("scanWorkEfficientNoBC: cudaMemcpy D->H failed");

            cudaFree(dev_idata);
            cudaFree(dev_odata);
        }

    } // namespace SharedMem
} // namespace StreamCompaction
