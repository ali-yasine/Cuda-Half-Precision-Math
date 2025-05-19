#ifndef KERNELS_H
#define KERNELS_H

#include <cuda_fp16.h>

constexpr int _UNROLL_FACTOR = 8; // unroll factor for the kernels


template<typename Op>
__global__ void unary_op_kernel(const half*  in,
                                half*        out,
                                uint64_t* timing_data,
                                size_t       n,
                                int          iterations) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {

        volatile half x = in[idx];
        volatile half res;
        uint64_t t0 = clock64();
        #pragma unroll _UNROLL_FACTOR
        for (int i = 0; i < iterations; ++i) {
            res = Op{}(x);
        }
        uint64_t t1 = clock64();

        timing_data[idx] = t1 - t0;
        out[idx]      = res;
    }
}

template<typename Op>
__global__ void binary_op_kernel(const half*  a,
                                 const half*  b,
                                 half*        out,
                                 uint64_t* timing_data,
                                 size_t       n,
                                 int          iterations) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {

        volatile half x = a[idx];
        volatile half  y = b[idx];
        volatile half res;
        uint64_t t0 = clock64();
        #pragma unroll _UNROLL_FACTOR
        for (int i = 0; i < iterations; ++i) {
            res = Op{}(x, y);
        }
        uint64_t t1 = clock64();

        timing_data[idx] = t1 - t0;
        out[idx]      = res;
    }
}



#endif 