#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "radixsort.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace RadixSort {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        const int blockSize = 128;

        /**
         * Extracts a specific bit from each element.
         * Maps bit=0 to 1 and bit=1 to 0, so that the subsequent scan
         * assigns positions to bit=0 elements first (ascending order).
         *
         * @param n       Number of elements
         * @param bools   Output: 1 if bit is 0, 0 if bit is 1
         * @param data    Input data array
         * @param bitPos  Bit position to extract (0 = LSB)
         */
        __global__ void kernExtractBit(int n, int *bools, const int *data, int bitPos) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) {
                return;
            }
            int bit = (data[index] >> bitPos) & 1;
            bools[index] = (bit == 0) ? 1 : 0;
        }

        /**
         * Scatters elements to correct positions for one radix-sort pass.
         * Elements with bit=0 go to the front section (via scanned indices).
         * Elements with bit=1 go to the back section (after all bit=0 elements).
         *
         * @param n          Number of elements
         * @param odata      Output array
         * @param idata      Input array
         * @param bools      1 if element bit=0, 0 if bit=1
         * @param scanned    Exclusive scan of bools (write positions for bit=0)
         * @param totalZeros Total count of elements with bit=0
         */
        __global__ void kernScatterRadix(int n, int *odata, const int *idata,
                                          const int *bools, const int *scanned,
                                          int totalZeros) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) {
                return;
            }

            if (bools[index] == 1) {
                // bit=0: place in the front section
                odata[scanned[index]] = idata[index];
            } else {
                // bit=1: place in the back section
                // onesBeforeMe = index - (number of zeros before me)
                int onesBeforeMe = index - scanned[index];
                odata[totalZeros + onesBeforeMe] = idata[index];
            }
        }

        /**
         * Performs a work-efficient exclusive scan on the indices array.
         * Uses Efficient::kernUpSweep and Efficient::kernDownSweep.
         *
         * On entry:  dev_indices contains the data to scan (size paddedN, power of 2)
         * On exit:   dev_indices[0..n-1] contains the exclusive prefix sum
         *
         * @param n         Number of real elements
         * @param paddedN   Power-of-2 padded size
         * @param indices   Device array (size paddedN) to scan in-place
         * @return          The total sum of all n elements (before zeroing root)
         */
        int scanIndices(int n, int paddedN, int *indices) {
            int log2ceil = ilog2ceil(paddedN);

            // Up-sweep phase
            for (int d = 0; d < log2ceil; d++) {
                int numThreads = paddedN / (1 << (d + 1));
                dim3 blocks((numThreads + blockSize - 1) / blockSize);
                Efficient::kernUpSweep<<<blocks, blockSize>>>(paddedN, d, indices);
                checkCUDAError("kernUpSweep in radix sort failed");
            }

            // After up-sweep, the last element contains the total sum
            int total;
            cudaMemcpy(&total, indices + paddedN - 1, sizeof(int), cudaMemcpyDeviceToHost);

            // Set root to zero for exclusive scan
            cudaMemset(indices + paddedN - 1, 0, sizeof(int));

            // Down-sweep phase
            for (int d = log2ceil - 1; d >= 0; d--) {
                int numThreads = paddedN / (1 << (d + 1));
                dim3 blocks((numThreads + blockSize - 1) / blockSize);
                Efficient::kernDownSweep<<<blocks, blockSize>>>(paddedN, d, indices);
                checkCUDAError("kernDownSweep in radix sort failed");
            }

            return total;
        }

        /**
         * Sorts an array of non-negative integers in ascending order
         * using LSB radix sort with work-efficient parallel scan.
         *
         * Processes all 32 bits. For non-negative integers,
         * bit 31 (sign bit) is always 0, so the sort remains correct.
         *
         * @param n     Number of elements
         * @param data  Array to sort (sorted in-place on return)
         */
        void radixsort(int n, int *data) {
            const int numBits = 32;

            // Round up to next power of 2 for the scan
            int paddedN = 1 << ilog2ceil(n);

            // Allocate device memory
            int *dev_data, *dev_temp, *dev_bools, *dev_indices;
            cudaMalloc((void**)&dev_data,    n * sizeof(int));
            cudaMalloc((void**)&dev_temp,    n * sizeof(int));
            cudaMalloc((void**)&dev_bools,   n * sizeof(int));
            cudaMalloc((void**)&dev_indices, paddedN * sizeof(int));
            checkCUDAError("radix sort cudaMalloc failed");

            // Copy input data to device
            cudaMemcpy(dev_data, data, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("radix sort cudaMemcpy to device failed");

            timer().startGpuTimer();

            dim3 fullBlocks((n + blockSize - 1) / blockSize);

            for (int bit = 0; bit < numBits; bit++) {
                // Step 1: Extract current bit
                //   bools[i] = 1 if data[i] has bit=0, 0 if bit=1
                kernExtractBit<<<fullBlocks, blockSize>>>(n, dev_bools, dev_data, bit);
                checkCUDAError("kernExtractBit failed");

                // Step 2: Exclusive scan on bools to get write positions
                cudaMemcpy(dev_indices, dev_bools, n * sizeof(int), cudaMemcpyDeviceToDevice);
                if (paddedN > n) {
                    cudaMemset(dev_indices + n, 0, (paddedN - n) * sizeof(int));
                }
                int totalZeros = scanIndices(n, paddedN, dev_indices);

                // Step 3: Scatter — bit=0 elements to front, bit=1 to back
                kernScatterRadix<<<fullBlocks, blockSize>>>(
                    n, dev_temp, dev_data, dev_bools, dev_indices, totalZeros);
                checkCUDAError("kernScatterRadix failed");

                // Swap buffers for next bit iteration
                int *tmp = dev_data;
                dev_data = dev_temp;
                dev_temp = tmp;
            }

            timer().endGpuTimer();

            // Copy sorted result back to host
            cudaMemcpy(data, dev_data, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("radix sort cudaMemcpy to host failed");

            // Free device memory
            cudaFree(dev_data);
            cudaFree(dev_temp);
            cudaFree(dev_bools);
            cudaFree(dev_indices);
        }
    }
}
