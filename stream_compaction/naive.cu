#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        const int blockSize = 128;

        /**
         * Naive parallel scan kernel
         * Implements the algorithm from GPU Gems 3, Chapter 39.2.1
         * 
         * @param n      Number of elements
         * @param d      Offset distance for this iteration (2^d)
         * @param odata  Output array
         * @param idata  Input array
         */
        __global__ void kernNaiveScan(int n, int d, int *odata, const int *idata) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            
            if (index >= n) {
                return;
            }

            // For exclusive scan: shift everything right by 1
            // odata[k] = (k >= 2^d) ? idata[k-2^d] + idata[k] : idata[k]
            if (index >= d) {
                odata[index] = idata[index - d] + idata[index];
            } else {
                odata[index] = idata[index];
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            // Allocate device memory
            int *dev_buf1, *dev_buf2;
            cudaMalloc((void**)&dev_buf1, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_buf1 failed");
            cudaMalloc((void**)&dev_buf2, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_buf2 failed");

            // Copy input data to device, shift right by 1 for exclusive scan
            cudaMemset(dev_buf1, 0, sizeof(int));
            cudaMemcpy(dev_buf1 + 1, idata, (n - 1) * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy to device failed");

            timer().startGpuTimer();

            // Calculate grid dimensions
            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

            // Perform log2(n) iterations
            int iterations = ilog2ceil(n);
            for (int d = 1; d <= iterations; d++) {
                int offset = 1 << (d - 1); // 2^(d-1)
                
                // Ping-pong between two buffers to avoid race conditions
                if (d % 2 == 1) {
                    kernNaiveScan<<<fullBlocksPerGrid, blockSize>>>(n, offset, dev_buf2, dev_buf1);
                } else {
                    kernNaiveScan<<<fullBlocksPerGrid, blockSize>>>(n, offset, dev_buf1, dev_buf2);
                }
                checkCUDAError("kernNaiveScan failed");
            }

            timer().endGpuTimer();

            // Copy result back to host from the correct buffer
            if (iterations % 2 == 1) {
                cudaMemcpy(odata, dev_buf2, n * sizeof(int), cudaMemcpyDeviceToHost);
            } else {
                cudaMemcpy(odata, dev_buf1, n * sizeof(int), cudaMemcpyDeviceToHost);
            }
            checkCUDAError("cudaMemcpy to host failed");

            // Free device memory
            cudaFree(dev_buf1);
            cudaFree(dev_buf2);
        }
    }
}
