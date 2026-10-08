#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "../kernels/1_naive.cu"
#include <cstdio>
#include <vector>
#include <random>
#include <cmath>

constexpr int N = 4096;
constexpr int block_size = 16;
constexpr int iters = 10;

template<typename F>
float time_ms(F&& fn, int iters) {
    fn();

    cudaEvent_t start;
    cudaEventCreate(&start);
    cudaEvent_t stop;
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    for (int i = 0; i < iters; ++i) {
        fn();
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms / iters;
}

double calc_gflops(float ms) {
    return 2.0 * N * N * N / (ms / 1000.0) / 1e9;
}

int main() {
    std::vector<float> h_A(N * N);
    std::vector<float> h_B(N * N);
    std::vector<float> h_C(N * N);
    std::vector<float> h_C_ref(N * N);

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (float& x : h_A) {
        x = dist(gen);
    }

    for (float& x : h_B) {
        x = dist(gen);
    }

    float *d_A;
    float *d_B;
    float *d_C;

    cudaMalloc(&d_A, N * N * sizeof(float));
    cudaMalloc(&d_B, N * N * sizeof(float));
    cudaMalloc(&d_C, N * N * sizeof(float));

    cudaMemcpy(d_A, h_A.data(), N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), N * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(block_size, block_size);
    dim3 grid((N + block_size - 1) / block_size, (N + block_size - 1) / block_size);

    auto run_naive = [&]() {
        matmul_naive<<<grid, block>>>(d_A, d_B, d_C, N);
    };

    float naive_ms = time_ms(run_naive, iters);
    double naive_gflops = calc_gflops(naive_ms);
    cudaMemcpy(h_C.data(), d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    float *d_C_ref;
    cudaMalloc(&d_C_ref, N * N * sizeof(float));

    cublasHandle_t handle;
    cublasCreate(&handle);
    const float alpha = 1.0f;
    const float beta = 0.0f;

    auto run_cublas = [&]() {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                    &alpha, d_B, N, d_A, N, &beta, d_C_ref, N);
    };

    float cublas_ms = time_ms(run_cublas, iters);
    double cublas_gflops = calc_gflops(cublas_ms);
    cudaMemcpy(h_C_ref.data(), d_C_ref, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    int incorrect = 0;
    for (int i = 0; i < N * N; ++i) {
        if (std::fabs(h_C[i] - h_C_ref[i]) > 1e-3f) incorrect++;
    }

    printf("N = %d\n", N);
    printf("naive:  %8.3f ms  %8.3f GFLOP/s\n", naive_ms, naive_gflops);
    printf("cuBLAS: %8.3f ms  %8.3f GFLOP/s\n", cublas_ms, cublas_gflops);
    printf("naive is %.2f%% of cuBLAS\n", 100.0 * naive_gflops / cublas_gflops);
    printf("mismatches vs cuBLAS: %d\n", incorrect);

    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_C_ref);
}
