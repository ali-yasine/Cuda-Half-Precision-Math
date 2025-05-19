#ifndef UTIL_H
#define UTIL_H

#include <string>
#include <bit>
#include <cstdint>
#include <cstdio>

#define cudaErrCheck(val) check((val), #val, __FILE__, __LINE__)
inline void check(cudaError_t err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        // std::cerr << "CUDA Runtime Error at: " << file << ":" << line
        //     << std::endl;
        // std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        printf("CUDA Runtime Error at: %s:%d\n", file, line);
        printf("%s %s\n", cudaGetErrorString(err), func);
        abort();
    }
}

__host__ __device__ consteval uint16_t float_to_fp16_bits(float value) noexcept {
    uint32_t bits = std::bit_cast<uint32_t>(value);
    uint32_t sign = (bits >> 16) & 0x8000u;
    uint32_t expo = (bits >> 23) & 0xFFu;
    uint32_t mant = bits & 0x007FFFFFu;

    if (expo == 0xFFu) {
        // NaN or ±Inf
        uint16_t m = static_cast<uint16_t>(mant >> 13);
        if (mant && (m == 0)) m = 1;
        return sign | 0x7C00u | m;
    }
    int32_t e = int32_t(expo) - 127 + 15;
    if (e >= 31) {
        // overflow → ±Inf
        return sign | 0x7C00u;
    } else if (e <= 0) {
        // subnormal or zero
        if (e < -10) return sign;
        mant |= 0x00800000u;
        uint32_t shift = uint32_t(1 - e);
        uint16_t m = static_cast<uint16_t>((mant + (1u << (shift-1))) >> shift);
        return sign | m;
    } else {
        // normal case
        uint16_t E = static_cast<uint16_t>(e) << 10;
        uint16_t M = static_cast<uint16_t>((mant + 0x00001000u) >> 13);
        return sign | E | M;
    }
}
#define COMPTIME_FLOAT2HALF(x) __ushort_as_half( float_to_fp16_bits(x) )


#endif