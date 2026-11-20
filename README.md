# Experiment 6: CUDA Memory Optimization

## 1. Overview

This experiment investigates GPU memory access optimization using CUDA. It compares uncoalesced and coalesced memory access patterns during matrix transpose and evaluates GPU performance using matrix multiplication.

The experiment was executed using an NVIDIA Tesla T4 GPU in Google Colab. Three independent runs were performed to observe performance consistency and calculate average measurements.

## 2. Objectives

- Understand GPU global memory access patterns.
- Compare uncoalesced and coalesced memory access.
- Implement shared-memory tiled matrix transpose.
- Measure execution time and effective memory bandwidth.
- Calculate transpose performance speedup.
- Implement CUDA matrix multiplication.
- Evaluate matrix multiplication performance in GFLOPS.
- Verify computational correctness.

## 3. Hardware and Software Environment

| Component | Specification |
|---|---|
| GPU Model | NVIDIA Tesla T4 |
| Compute Capability | 7.5 |
| Platform | Google Colab |
| Programming Language | CUDA C++ |
| Compiler | NVIDIA CUDA Compiler (NVCC) |
| Transpose Matrix | 4096 × 4096 |
| Matrix Multiplication | 1024 × 1024 |

## 4. Experiments Performed

### 4.1 Uncoalesced Matrix Transpose

The uncoalesced transpose implementation accesses global memory in a pattern that is less efficient for GPU memory transactions.

This implementation is used as the baseline for comparison.

### 4.2 Coalesced Matrix Transpose

The optimized transpose implementation uses shared-memory tiling to improve memory access patterns.

Coalesced memory access allows neighboring threads to access neighboring memory locations more efficiently, improving memory bandwidth utilization.

### 4.3 CUDA Matrix Multiplication

Matrix multiplication is performed on the GPU using CUDA. Execution time and computational performance are measured for a 1024 × 1024 matrix.

## 5. Experimental Results

Three independent runs were performed using the same GPU and matrix dimensions.

### 5.1 Transpose Results

| Metric | Run 01 | Run 02 | Run 03 | Average |
|---|---:|---:|---:|---:|
| Uncoalesced Time (ms) | 2.1549 | 2.1117 | 2.9228 | 2.3965 |
| Uncoalesced Bandwidth (GB/s) | 62.29 | 63.56 | 45.92 | 57.26 |
| Coalesced Time (ms) | 0.7333 | 0.7137 | 0.8137 | 0.7536 |
| Coalesced Bandwidth (GB/s) | 183.04 | 188.05 | 164.95 | 178.68 |
| Transpose Speedup | 2.94× | 2.96× | 3.59× | 3.16× |

### 5.2 Matrix Multiplication Results

| Metric | Run 01 | Run 02 | Run 03 | Average |
|---|---:|---:|---:|---:|
| Matrix Size | 1024 × 1024 | 1024 × 1024 | 1024 × 1024 | 1024 × 1024 |
| Execution Time (ms) | 7.9907 | 7.3442 | 8.9805 | 8.1051 |
| Performance (GFLOPS) | 268.75 | 292.41 | 239.13 | 266.76 |
| Correctness | PASS | PASS | PASS | PASS |

## 6. Observation Table

| Kernel | Matrix Size N | Time (ms) | Speedup | Bandwidth (GB/s) / GFLOPS | Result (PASS/FAIL) |
|---|---:|---:|---|---:|---|
| Transpose: uncoalesced | 4096 | 2.3965 | 1.0× (baseline) | 57.26 GB/s | PASS |
| Transpose: coalesced | 4096 | 0.7536 | 3.16× | 178.68 GB/s | PASS |
| Naive matrix multiplication | 1024 | 8.1051 | – | 266.76 GFLOPS | PASS |

**Speedup of coalesced over uncoalesced transpose = 3.16×**

**GPU model used:** NVIDIA Tesla T4

**Compute capability:** 7.5

## 7. Analysis and Discussion

### Why the coalesced transpose wins

In the uncoalesced transpose kernel, threads access global memory in a less efficient pattern. Memory accesses by neighboring threads are not fully contiguous, resulting in additional memory transactions and reduced memory bandwidth utilization.

In the coalesced transpose kernel, shared-memory tiling is used to reorganize data. Threads access global memory in a more contiguous pattern, improving memory transaction efficiency and reducing execution time.

The experimental results demonstrate this improvement:

- Average uncoalesced time: **2.3965 ms**
- Average coalesced time: **0.7536 ms**
- Average speedup: **3.16×**

### Padding

The optimized transpose uses a padded shared-memory tile (`TILE + 1`), which helps reduce shared-memory bank conflicts when threads access the transposed tile.

Padding improves shared-memory access patterns and helps avoid bank conflicts.

The current experiment compares uncoalesced and coalesced transpose implementations. A separate timing comparison with padding removed was not performed.

### Matrix Multiplication Analysis

The naive matrix multiplication kernel performed multiplication and accumulation on the GPU for a 1024 × 1024 matrix.

The average measured performance across three runs was **266.76 GFLOPS**, with an average execution time of **8.1051 ms**.

All three executions passed their correctness checks.

### Performance Variation

Execution time and bandwidth varied between runs because GPU performance can be affected by system load, GPU scheduling and runtime conditions.

Despite the variation, coalesced transpose was faster in all three runs, demonstrating consistent improvement.

## 8. Screenshots of Experimental Results

### Run 01

![Run 01 - CUDA Memory Optimization Results](evidence/01.png)

### Run 02

![Run 02 - CUDA Memory Optimization Results](evidence/02.png)

### Run 03

![Run 03 - CUDA Memory Optimization Results](evidence/03.png)



## 09. How to Execute

The experiment can be executed in Google Colab with a GPU runtime.

### Compile the CUDA program

```bash
nvcc -O2 -o exp6 exp6.cu
```

### Execute the program

```bash
./exp6
```

### Optional: Execute using smaller matrix dimensions

```bash
./exp6 2048 512
```

The first argument specifies the transpose matrix dimension and the second argument specifies the matrix multiplication dimension.

## 10. Conclusion

This experiment demonstrates the importance of efficient GPU memory access patterns.

The coalesced matrix transpose implementation achieved an average speedup of approximately **3.16×** over the uncoalesced implementation. It increased average effective memory bandwidth from **57.26 GB/s to 178.68 GB/s**.

The matrix multiplication implementation achieved an average performance of **266.76 GFLOPS** across three runs.

All three executions passed their correctness checks. The experiment highlights the benefits of memory coalescing and shared-memory optimization in CUDA programming.

---

**Experiment Status: Completed Successfully**

**Correctness: PASS in all three runs.**
