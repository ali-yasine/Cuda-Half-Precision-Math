#ifndef KERNELS_H
#define KERNELS_H

#include <cuda_fp16.h>

constexpr int _UNROLL_FACTOR = 128; // unroll factor for the kernels


template<typename Op>
__global__ void unary_op_kernel(const half*  in,
                                half*        out,
                                unsigned long long* timing_data,
                                size_t       n,
                                int          iterations)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {

        volatile half x = in[idx];
        volatile half res;
        unsigned long long t0 = clock64();
        #pragma unroll _UNROLL_FACTOR
        for (int i = 0; i < iterations; ++i) {
            res = Op{}(x);
        }
        unsigned long long t1 = clock64();

        timing_data[idx] = t1 - t0;
        out[idx]      = res;
    }
}

template<typename Op>
__global__ void binary_op_kernel(const half*  a,
                                 const half*  b,
                                 half*        out,
                                 unsigned long long* timing_data,
                                 size_t       n,
                                 int          iterations)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {

        half x = a[idx], y = b[idx];
        volatile half res;
        unsigned long long t0 = clock64();
        #pragma unroll _UNROLL_FACTOR
        for (int i = 0; i < iterations; ++i) {
            res = Op{}(x, y);
        }
        unsigned long long t1 = clock64();

        timing_data[idx] = t1 - t0;
        out[idx]      = res;
    }
}



#endif 