#include <cstdio>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            odata[0]=0;
            for(int i = 1;i<n;i++){
                odata[i] = odata[i-1] + idata[i-1];
            }
            timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            int index= 0;
            int count = 0;
            for(int i = 0;i<n;i++){
                if(idata[i]!=0){
                    odata[index++] = idata[i];
                    count ++;
                }
            }
            timer().endCpuTimer();
            return count;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            //map
            int* map = new int[n];
            for(int i = 0 ;i < n;i++){
                map[i] = !(idata[i]==0);
            }
            //scan
            int* scanOut = new int[n];
            //do not call scan func here,cause timer can be fired only once 
            scanOut[0]=0;
            for(int i = 1;i<n;i++){
                scanOut[i] = scanOut[i-1] + map[i-1];
            }
            int count = 0;
            for(int i = 0;i<n;i++){
                if(map[i]!=0){
                    count ++;
                    odata[scanOut[i]] = idata[i];
                }
            }
            //scatter
            timer().endCpuTimer();
            delete[] map;
            delete[] scanOut;
            return count;
        }
    }
}
