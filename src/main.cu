
#include <iostream>
#include <vector>
#include <random>
#include <numeric>
#include <cuda_fp16.h> 

#include "kernels.cuh" 
#include "util.h"
#include "error_metrics.cuh"
#include "half_math.cuh"



// Generic test runners
template<typename Tag>
void run_unary_test(
    const char* label,
    half*        d_in,
    half*        d_out,
    unsigned long long* d_timing,
    const std::vector<half>&    h_in,
    std::vector<half>&          h_out,
    std::vector<unsigned long long>& h_timing,
    int numBlocks,
    int threadsPerBlock,
    int n_samples,
    int iterations)
{

    cudaEvent_t start, stop; //should probably be initialized once outside the tests and reused but I digress
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 1) launch
    cudaEventRecord(start);
    unary_op_kernel<Tag><<<numBlocks,threadsPerBlock>>>(d_in, d_out, d_timing, n_samples, iterations);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    // 2) copy back
    cudaMemcpy(h_out .data(), d_out,    n_samples*sizeof(half),              cudaMemcpyDeviceToHost);
    cudaMemcpy(h_timing.data(), d_timing, n_samples*sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    
    // 3) accumulate clocks
    auto total_cycles    = std::accumulate(h_timing.begin(), h_timing.end(), 0ull);
    double avg_cycles    = total_cycles / (n_samples * static_cast<double>(iterations));
    double ops_per_cycle = 1.0 / avg_cycles;

    // 4) compute errors
    float max_err, mean_err, rmse_err;
    compute_errors_unary<Tag>(h_in.data(), h_out.data(), n_samples, max_err, mean_err, rmse_err);
    
    // 5) print
    printf("%-13s | %10.6f | %13.6f | %11.5f | %e | %e | %e\n",
       label,
       elapsedTime,
       ops_per_cycle,
       avg_cycles,
       max_err, mean_err, rmse_err);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

template<typename Tag>
void run_binary_test(
    const char* label,
    half*        d_x,
    half*        d_y,
    half*        d_out,
    unsigned long long* d_timing,
    const std::vector<half>&    h_x,
    const std::vector<half>&    h_y,
    std::vector<half>&          h_out,
    std::vector<unsigned long long>& h_timing,
    int numBlocks,
    int threadsPerBlock,
    int n_samples,
    int iterations)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    // 1) launch
    cudaEventRecord(start);
    binary_op_kernel<Tag><<<numBlocks,threadsPerBlock>>>(d_x, d_y, d_out, d_timing, n_samples, iterations);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    // 2) copy back
    cudaMemcpy(h_out .data(), d_out,    n_samples*sizeof(half),              cudaMemcpyDeviceToHost);
    cudaMemcpy(h_timing.data(), d_timing, n_samples*sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    
    // 3) accumulate clocks
    auto total_cycles = std::accumulate(h_timing.begin(), h_timing.end(), 0ull);
    double avg_cycles = total_cycles / (n_samples * static_cast<double>(iterations));
    double ops_per_cycle = 1.0 / avg_cycles;
    // 4) compute errors
    float max_err, mean_err, rmse_err;
    compute_errors_binary<Tag>(h_x.data(), h_y.data(), h_out.data(), n_samples, max_err, mean_err, rmse_err);
    
    // 5) print
    printf("%-13s | %10.6f | %13.6f | %11.5f | %e | %e | %e\n",
        label,
        elapsedTime,
        ops_per_cycle,
        avg_cycles,
        max_err, mean_err, rmse_err);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}



int main() {

    constexpr uint64_t n_samples = 1<<26; 
    constexpr uint32_t threadsPerBlock = 1024;
    constexpr uint32_t numBlocks = (n_samples + threadsPerBlock - 1) / threadsPerBlock;
    int iterations = 100000; 

    //  Data Generation 
    std::mt19937_64 rng(123456789ull);
    std::uniform_real_distribution<float> dist_rec(0.1f, 10.0f);
    std::uniform_real_distribution<float> dist_div_num(0.1f, 20.0f);
    std::uniform_real_distribution<float> dist_div_den(0.1f, 20.0f);

    std::vector<half> h_rec_in(n_samples);
    std::vector<half> h_div_x(n_samples), h_div_y(n_samples);
    for (size_t i = 0; i < n_samples; ++i) { 
        h_rec_in[i] = __float2half(dist_rec(rng));
        h_div_x[i] = __float2half(dist_div_num(rng));
        h_div_y[i] = __float2half(dist_div_den(rng));
    }


    std::vector<half> h_out_custom(n_samples), h_out_builtin(n_samples);
    std::vector<unsigned long long> h_timing_data(n_samples);


    //  Device Memory Allocation 
    half *d_in, *d_custom_out, *d_builtin_out;
    half *d_div_x, *d_div_y;
    unsigned long long *d_timing_data;

    cudaMalloc(&d_in,           n_samples * sizeof(half));
    cudaMalloc(&d_custom_out,   n_samples * sizeof(half));
    cudaMalloc(&d_builtin_out,  n_samples * sizeof(half));
    cudaMalloc(&d_div_x,        n_samples * sizeof(half));
    cudaMalloc(&d_div_y,        n_samples * sizeof(half));
    cudaMalloc(&d_timing_data,  n_samples * sizeof(unsigned long long));

    //  Copy inputs to device 
    cudaMemcpy(d_in,    h_rec_in.data(), n_samples*sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_div_x, h_div_x.data(),  n_samples*sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_div_y, h_div_y.data(),  n_samples*sizeof(half), cudaMemcpyHostToDevice);
    
    // print error header
    printf("Method        | Time (ms)  |   Ops/Cycle   | Cycles/Op   |   MaxErr     |  MeanErr     |   RMSErr\n");
    printf("-----------------------------------------------------------------------------------------------\n");


    // Reciprocal 
    run_unary_test<FastReciprocal>(
        "Rec.Custom",
        d_in, d_custom_out, d_timing_data,
        h_rec_in, h_out_custom, h_timing_data,
        numBlocks, threadsPerBlock, n_samples, iterations
    );
    run_unary_test<BuiltinReciprocal>(
        "Rec.BuiltIn",
        d_in, d_builtin_out, d_timing_data,
        h_rec_in, h_out_builtin, h_timing_data,
        numBlocks, threadsPerBlock, n_samples, iterations
    );

    //  Division 
    run_binary_test<FastDivision>(
        "Div.Custom",
        d_div_x, d_div_y, d_custom_out, d_timing_data,
        h_div_x, h_div_y, h_out_custom, h_timing_data,
        numBlocks, threadsPerBlock, n_samples, iterations
    );
    run_binary_test<BuiltinDivision>(
        "Div.BuiltIn",
        d_div_x, d_div_y, d_builtin_out, d_timing_data,
        h_div_x, h_div_y, h_out_builtin, h_timing_data,
        numBlocks, threadsPerBlock, n_samples, iterations
    );

    //  Inv Sqrt 
    run_unary_test<FastInvSqrt>(
        "InvSq.Custom",
        d_in, d_custom_out, d_timing_data,
        h_rec_in, h_out_custom, h_timing_data,
        numBlocks, threadsPerBlock, n_samples, iterations
    );
    run_unary_test<BuiltinInvSqrt>(
        "InvSq.BuiltIn",
        d_in, d_builtin_out, d_timing_data,
        h_rec_in, h_out_builtin, h_timing_data,
        numBlocks, threadsPerBlock, n_samples, iterations
    );

    //  Sqrt 
    run_unary_test<FastSqrt>(
        "Sqrt.Custom",
        d_in, d_custom_out, d_timing_data,
        h_rec_in, h_out_custom, h_timing_data,
        numBlocks, threadsPerBlock, n_samples, iterations
    );
    run_unary_test<BuiltinSqrt>(
        "Sqrt.BuiltIn",
        d_in, d_builtin_out, d_timing_data,
        h_rec_in, h_out_builtin, h_timing_data,
        numBlocks, threadsPerBlock, n_samples, iterations
    );

    //  Exp 
    run_unary_test<FastExp>(
        "Exp.Custom",
        d_in, d_custom_out, d_timing_data,
        h_rec_in, h_out_custom, h_timing_data,
        numBlocks, threadsPerBlock, n_samples, iterations
    );
    run_unary_test<BuiltinExp>(
        "Exp.BuiltIn",
        d_in, d_builtin_out, d_timing_data,
        h_rec_in, h_out_builtin, h_timing_data,
        numBlocks, threadsPerBlock, n_samples, iterations
    );

    //  Log 
    run_unary_test<FastLog>(
        "Log.Custom",
        d_in, d_custom_out, d_timing_data,
        h_rec_in, h_out_custom, h_timing_data,
        numBlocks, threadsPerBlock, n_samples, iterations
    );
    run_unary_test<BuiltinLog>(
        "Log.BuiltIn",
        d_in, d_builtin_out, d_timing_data,
        h_rec_in, h_out_builtin, h_timing_data,
        numBlocks, threadsPerBlock, n_samples, iterations
    );

    //  Cleanup 
    cudaFree(d_in); cudaFree(d_custom_out); cudaFree(d_builtin_out);
    cudaFree(d_div_x); cudaFree(d_div_y);
    cudaFree(d_timing_data);

    return 0;
}