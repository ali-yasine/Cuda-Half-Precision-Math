#ifndef HALF_MATH_CUH
#define HALF_MATH_CUH

#include <cuda_fp16.h>
#include <limits.h>
#include "util.h"


#define N 10 // 10 bit mantissa
#define B 15 // bias is 15 for half precision
#define HALF_ONE            __ushort_as_half((unsigned short) 0x3C00U)
#define HALF_INV_LN_2       __ushort_as_half((unsigned short) 0x3dc5) //1/ln(2)
#define HALF_LN2            __ushort_as_half((unsigned short) 0x398C)  /*FP16(ln2) */
#define HALF_LN2_ERR        __ushort_as_half((unsigned short) 0x8AF4)  /* ln2 − FP16(ln2) in FP16 */
#define HALF_COEFF_ONE      __ushort_as_half((unsigned short) 0x3160) //used for fast exp ~ 0.167969
#define HALF_COEFF_TWO      __ushort_as_half((unsigned short) 0x3809) //used for fast exp ~ 0.504395
#define HALF_TWO_POW_NEG24  __ushort_as_half((unsigned short) 0x0001)  /*  2^(1 - N - B) ≈5.96e−08. i.e smallest positive number */

//TODO: add error handling for x = 0
template<bool precise = true>
static inline __device__ half fast_hrcp(half x) {
    // magic constant representing n = USHORT_MAX - (1 << N) + K where K = 2^N * root of m_k^2 +8m_k -8 = 0
    constexpr short MAGIC1 = SHRT_MAX - (3<<N) + ((1<<N)*0.899f);
    half y0 = __ushort_as_half(MAGIC1 - __half_as_ushort(x));
    
    half r0 = __hfma(__hneg(x), y0, HALF_ONE);
    half y;
    if constexpr (precise) {
        y = __hfma(y0, __hfma(r0, r0, r0), y0); //y = y0 + y0* (r0 + r0*r0)
    } else {
        y = __hfma(y0, r0, y0); //y = y0 + y0*r0
    }
    return y;
}

//TODO: add error handling for y = 0
static inline __device__ half fast_div(half x, half y) {
    half w = fast_hrcp<true>(y); // w ~ 1/y
    half z0 = w * y;
    half z = __hfma(w, __hfma(x, z0, __hneg(y)), z0);
    return z; 
}



/* Compute the magic constant for the reciprocal square root kernel, 
 * n is the number of mantissa bits, b is the bias, T is the return type
 * for context google "fast inverse square root"
 */
template<typename T>
consteval T inv_sqrt_magic_constant(int n, int b) {

    constexpr float t = 3.7309796f;

    float value = (1 << n) * (((b - 1) / 2.0f) + b + ((t - 2.0f) / 4.0f));
    return static_cast<T>(value);
}

static inline __device__ half fast_inv_sqrt(half x) {

    constexpr short magic_const = inv_sqrt_magic_constant<short>(N, B);
    half y0 = __ushort_as_half(magic_const - (__half_as_ushort(x) >> 1)); 
    half r0 = __hfma(__hneg(x), y0 * y0, HALF_ONE); //r0 = 1 -(x * y0^2)

    half temp = __hfma(r0, COMPTIME_FLOAT2HALF(0.375f), COMPTIME_FLOAT2HALF(0.5)); //1/2 + 3/8 r0
    temp = r0 * temp; //r0 * (1/2 + 3/8 r0)
    return __hfma(y0, temp, y0); //y = y0 + y0 * r0 * (1/2 + 3/8 r0)
}



template <bool precise = true>
static inline __device__ half fast_sqrt(half x) {
    
    constexpr short magic_const = inv_sqrt_magic_constant<short>(N, B);
    half w = __ushort_as_half(magic_const - (__half_as_ushort(x) >> 1));  //rough estimate for 1/sqrt(x)
    half w_over_two = w * COMPTIME_FLOAT2HALF(0.5f);
    half y = x * w; //y0 = x * w
    half temp = __hfma(y, __hneg(y), x); //x - y0^2
    y = __hfma(w_over_two, temp, y); //y1 = y0 + 1/2 w * (x - y0^2)
    if constexpr (precise) { //do a second iteration if flag is true
        temp = __hfma(y, __hneg(y), x); //x - y1^2
        y = __hfma(w_over_two, temp, y); //y = y1 + 1/2 w * (x - y1^2)
    } 
    return y;
}


static inline __device__ short get_exponent(half x) {
    constexpr unsigned short mask = 0x7C00; //0111 1100 0000 0000
    unsigned short exp = __half_as_ushort(x) & mask;
    return exp >> N;
}


//returns the fractional part of the half number as a half number
static inline __device__ half get_mantissa(half x) {
    constexpr unsigned short mask = (1u << N) - 1; 
    constexpr unsigned short scaling_factor = (B-1 + B) << N; //2B - 1 - B = B - 1, the required scaling factor
    unsigned short mantissa = mask & __half_as_ushort(x);
    return __ushort_as_half(scaling_factor) * __ushort_as_half(mantissa);
}

static inline __device__ half fast_exp(half x) {
    

    const half z = HALF_TWO_POW_NEG24 * (HALF_INV_LN_2 * x); //z = 2^(1 - B - N) * (x/ln(2))
    const half neg_n = COMPTIME_FLOAT2HALF(-32768.0f) * (COMPTIME_FLOAT2HALF(512) * z); //we actually only need -n and not n
    const half two_pow_n = __ushort_as_half((__half_as_ushort(z) + B) << N);
    const half r = __hfma(neg_n, HALF_LN2_ERR, __hfma(HALF_LN2, neg_n, x));//r = (x - n c_hi) - n c_lo
    
    const half p_r = __hfma(r, __hfma(r, __hfma(HALF_COEFF_ONE, r, HALF_COEFF_TWO), HALF_ONE), HALF_ONE);
    
    return two_pow_n * p_r;
}


static inline __device__ half fast_log(half x) {
    
    const unsigned short e = __half_as_ushort(x * COMPTIME_FLOAT2HALF(0.75f)) >> N;  
    
    const half n = COMPTIME_FLOAT2HALF(32768.0f) * (COMPTIME_FLOAT2HALF(512.0F) * __ushort_as_half(e)) - COMPTIME_FLOAT2HALF(float(B - 1));
    
    half r = __ushort_as_half(__half_as_ushort(x) - ((e - B + 1) << N)) - HALF_ONE;
    
    const half p_r = r * __hfma(r, __hfma(r, __hfma(r, COMPTIME_FLOAT2HALF(-0.275390625), COMPTIME_FLOAT2HALF(0.353271484375f)), COMPTIME_FLOAT2HALF(-0.499267578125f)), HALF_ONE); 
    
    const half y = __hfma(n, HALF_LN2, __hfma(n, HALF_LN2_ERR, p_r));
    return y;
}




#undef N
#undef B
#undef HALF_ONE
#undef HALF_INV_LN_2
#undef HALF_LN2
#undef HALF_LN2_ERR

#endif
