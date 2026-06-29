#pragma once

#include "common.h"
#include "device_types.h"
#include <__clang_cuda_runtime_wrapper.h>

namespace StreamCompaction {
    namespace RadixSort {
        StreamCompaction::Common::PerformanceTimer& timer();
        //map the input array to an array of 0s and 1s,That is, for each element in idata,
        //if bit b of idata[idx] == 0,then bools[idx] = 1; otherwise false;
        __global__ void kernMapToBoolean(int n, int *bools, const int *idata);
        //perform an exclusive scan on bools
        __global__ void kernExclusiveScan(int n,int* scanned,const int* bools);
        //scatter the elements in idata into odata, using bools to determine 
        //whether an element should be scattered, and using scanned to determine 
        //the location to scatter to.
        __global__ void kernScatter(int n,int *odata, const int *idata, const int *bools, const int *scanned);
        //sort array with non-negative elements in an ascending order
        void radixsort(int n, int *data);
    }
}
