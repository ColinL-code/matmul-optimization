# Matrix Multiplication with CUDA

Learning about writing and optimizing kernels by implementing matrix multiplication.

The kernels I've written so far:
- kernels/1_naive.cu: Straightforward implementation where each thread computes one element of the output

Each kernel is tested against cuBLAS for speed and correctness (bench/bench.cu). The kernels are run on an NVIDIA L4 GPU via Modal (modal_app.py). Run with Modal by using the following command:

```bash
modal run modal_app.py --file bench/bench.cu
```

## Results

N = 4096, FP32, NVIDIA L4

| Kernel | Block size | Time (ms) | GFLOP/s | % of cuBLAS (same run) |
|---|---|---|---|---|
| 1_naive (`x` indexes columns, current version) | 32x32 | 112.429 | 1222.453 | 8.95% |
| 1_naive (`x` indexes rows) | 32x32 | 605.859 | 226.850 | 1.69% |

