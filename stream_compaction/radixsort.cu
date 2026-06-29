#include "radixsort.h"

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
         *
         * @param n      The number of elements in data.
         * @param data  The array of elements to sort.
         */
         void radixsort(int n, int *data){

         }
    }
}