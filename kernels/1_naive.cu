#include <cuda_runtime.h>

__global__ void matmul_naive(const float* A, const float* B, float* C, int N) {
    int r = blockDim.y * blockIdx.y + threadIdx.y;
    int c = blockDim.x * blockIdx.x + threadIdx.x;
    float sum = 0.0f;
    if (r < N && c < N) {
        for (int i = 0; i < N; ++i) {
            sum += A[N * r + i] * B[N * i + c];
        }
        C[N * r + c] = sum;
    }
}