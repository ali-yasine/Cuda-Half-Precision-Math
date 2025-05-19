#ifndef ERROR_METRICS_CUH
#define ERROR_METRICS_CUH

#include <half_math.cuh>

struct FastReciprocal {
    __device__ __forceinline__ half operator()(half x) const {
        return fast_rcp(x);
    }
    static float err(float xf, float af) {
        // 1.0f − x * result
        return fabsf(1.0f - xf * af);
    }
};
struct BuiltinReciprocal {
    __device__ __forceinline__ half operator()(half x) const {
        return hrcp(x);
    }
    static float err(float xf, float af) {
        return fabsf(1.0f - xf * af);
    }
};

struct FastInvSqrt {
    __device__ __forceinline__ half operator()(half x) const {
        return fast_inv_sqrt(x);
    }
    static float err(float xf, float af) {
        // 1.0f − (result² * x)
        return fabsf(1.0f - af * af * xf);
    }
};
struct BuiltinInvSqrt {
    __device__ __forceinline__ half operator()(half x) const {
        return hrsqrt(x);
    }
    static float err(float xf, float af) {
        return fabsf(1.0f - af * af * xf);
    }
};

struct FastSqrt {
    __device__ __forceinline__ half operator()(half x) const {
        return fast_sqrt<true>(x);
    }
    static float err(float xf, float af) {
        if (fabsf(xf) < 1e-20f) {
            // avoid divide-by-zero
            return fabsf(af*af - xf);
        } else {
            // 1.0f − (result² / x)
            return fabsf(1.0f - (af*af)/xf);
        }
    }
};

struct BuiltinSqrt {
    __device__ __forceinline__ half operator()(half x) const {
        return hsqrt(x);
    }
    static float err(float xf, float af) {
        if (fabsf(xf) < 1e-20f) {
            return fabsf(af*af - xf);
        } else {
            return fabsf(1.0f - (af*af)/xf);
        }
    }
};

struct FastExp {
    __device__ __forceinline__ half operator()(half x) const {
        return fast_exp(x);
    }
    static float err(float xf, float af) {
        float tv = expf(xf);
        if (fabsf(tv) < 1e-20f) {
            return fabsf(af - tv);
        } else {
            return fabsf(1.0f - af / tv);
        }
    }
};

struct BuiltinExp {
    __device__ __forceinline__ half operator()(half x) const {
        return hexp(x);
    }
    static float err(float xf, float af) {
        float tv = expf(xf);
        if (fabsf(tv) < 1e-20f) {
            return fabsf(af - tv);
        } else {
            return fabsf(1.0f - af / tv);
        }
    }
};

struct FastLog {
    __device__ __forceinline__ half operator()(half x) const {
        return fast_log(x);
    }
    static float err(float xf, float af) {
        float tv = logf(xf);
        if (fabsf(tv) < 1e-9f) {
            return fabsf(af - tv);
        } else {
            return fabsf(1.0f - af / tv);
        }
    }
};
struct BuiltinLog {
    __device__ __forceinline__ half operator()(half x) const {
        return hlog(x);
    }
    static float err(float xf, float af) {
        float tv = logf(xf);
        if (fabsf(tv) < 1e-9f) {
            return fabsf(af - tv);
        } else {
            return fabsf(1.0f - af / tv);
        }
    }
};

struct FastDivision {
    __device__ __forceinline__ half operator()(half x, half y) const {
        return fast_div(x, y);
    }
    static float err(float num_f, float den_f, float af) {
        if (fabsf(num_f) < 1e-20f && fabsf(den_f) < 1e-20f) {
            return fabsf(af - 0.0f);
        } else if (fabsf(num_f) < 1e-20f) {
            return fabsf(af - 0.0f);
        } else if (fabsf(den_f) < 1e-20f) {
            // fallback relative error
            return fabsf((af * den_f - num_f) / num_f);
        } else {
            return fabsf(1.0f - (af * den_f) / num_f);
        }
    }
};
struct BuiltinDivision {
    __device__ __forceinline__ half operator()(half x, half y) const {
        return x / y;
    }
    static float err(float num_f, float den_f, float af) {
        if (fabsf(num_f) < 1e-20f && fabsf(den_f) < 1e-20f) {
            return fabsf(af - 0.0f);
        } else if (fabsf(num_f) < 1e-20f) {
            return fabsf(af - 0.0f);
        } else if (fabsf(den_f) < 1e-20f) {
            return fabsf((af * den_f - num_f) / num_f);
        } else {
            return fabsf(1.0f - (af * den_f) / num_f);
        }
    }
};


template<typename Op>
void compute_errors_unary(const half* in,
                          const half* approx,
                          size_t       n,
                          float&       max_e,
                          float&       mean_e,
                          float&       rmse_e)
{
    double sum_abs = 0.0, sum_sq = 0.0;
    max_e = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float xf = __half2float(in[i]);
        float af = __half2float(approx[i]);
        float r  = Op::err(xf, af);
        sum_abs += r;
        sum_sq  += double(r)*double(r);
        if (r > max_e) max_e = r;
    }
    mean_e = float(sum_abs / n);
    rmse_e = float(std::sqrt(sum_sq / n));
}

template<typename Op>
void compute_errors_binary(const half* num_in,
                           const half* den_in,
                           const half* approx,
                           size_t       n,
                           float&       max_e,
                           float&       mean_e,
                           float&       rmse_e)
{
    double sum_abs = 0.0, sum_sq = 0.0;
    max_e = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float nf = __half2float(num_in[i]);
        float df = __half2float(den_in[i]);
        float af = __half2float(approx[i]);
        float r  = Op::err(nf, df, af);
        sum_abs += r;
        sum_sq  += double(r)*double(r);
        if (r > max_e) max_e = r;
    }
    mean_e = float(sum_abs / n);
    rmse_e = float(std::sqrt(sum_sq / n));
}

#endif