#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define TILE 32
#define BLK 16

// ============================================================
// CUDA ERROR CHECKING
// ============================================================

void cudaCheck(cudaError_t status, const char *file, int line)
{
    if (status != cudaSuccess)
    {
        fprintf(stderr, "\nCUDA ERROR: %s\n", cudaGetErrorString(status));
        fprintf(stderr, "File: %s\nLine: %d\n", file, line);
        exit(EXIT_FAILURE);
    }
}

// ============================================================
// 1. UNCOALESCED TRANSPOSE
// ============================================================

__global__ void transposeNaive(const float *input,
                              float *output,
                              int N)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < N && y < N)
    {
        output[x * N + y] = input[y * N + x];
    }
}

// ============================================================
// 2. COALESCED TRANSPOSE USING SHARED MEMORY
// ============================================================

__global__ void transposeCoalesced(const float *input,
                                   float *output,
                                   int N)
{
    __shared__ float tile[TILE][TILE + 1];

    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;

    // Coalesced read
    if (x < N && y < N)
    {
        tile[threadIdx.y][threadIdx.x] =
            input[y * N + x];
    }

    __syncthreads();

    // Transposed block coordinates
    int transposedX = blockIdx.y * TILE + threadIdx.x;
    int transposedY = blockIdx.x * TILE + threadIdx.y;

    // Coalesced write
    if (transposedX < N && transposedY < N)
    {
        output[transposedY * N + transposedX] =
            tile[threadIdx.x][threadIdx.y];
    }
}

// ============================================================
// 3. NAIVE MATRIX MULTIPLICATION
// ============================================================

__global__ void matMulNaive(const float *A,
                            const float *B,
                            float *C,
                            int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N)
    {
        float sum = 0.0f;

        for (int k = 0; k < N; k++)
        {
            sum += A[row * N + k] * B[k * N + col];
        }

        C[row * N + col] = sum;
    }
}

// ============================================================
// 4. TIMING: UNCOALESCED TRANSPOSE
// ============================================================

float timeNaiveTranspose(float *output,
                        const float *input,
                        int N,
                        dim3 grid,
                        dim3 block)
{
    cudaEvent_t startEvent, stopEvent;

    cudaCheck(cudaEventCreate(&startEvent), __FILE__, __LINE__);
    cudaCheck(cudaEventCreate(&stopEvent), __FILE__, __LINE__);

    // Warm-up
    transposeNaive<<<grid, block>>>(input, output, N);

    cudaCheck(cudaGetLastError(), __FILE__, __LINE__);
    cudaCheck(cudaDeviceSynchronize(), __FILE__, __LINE__);

    // Start timer
    cudaCheck(cudaEventRecord(startEvent), __FILE__, __LINE__);

    for (int i = 0; i < 20; i++)
    {
        transposeNaive<<<grid, block>>>(input, output, N);
    }

    cudaCheck(cudaGetLastError(), __FILE__, __LINE__);

    // Stop timer
    cudaCheck(cudaEventRecord(stopEvent), __FILE__, __LINE__);
    cudaCheck(cudaEventSynchronize(stopEvent), __FILE__, __LINE__);

    float milliseconds = 0.0f;

    cudaCheck(cudaEventElapsedTime(
        &milliseconds, startEvent, stopEvent), __FILE__, __LINE__);

    milliseconds /= 20.0f;

    cudaCheck(cudaEventDestroy(startEvent), __FILE__, __LINE__);
    cudaCheck(cudaEventDestroy(stopEvent), __FILE__, __LINE__);

    return milliseconds;
}

// ============================================================
// 5. TIMING: COALESCED TRANSPOSE
// ============================================================

float timeCoalescedTranspose(float *output,
                            const float *input,
                            int N,
                            dim3 grid,
                            dim3 block)
{
    cudaEvent_t startEvent, stopEvent;

    cudaCheck(cudaEventCreate(&startEvent), __FILE__, __LINE__);
    cudaCheck(cudaEventCreate(&stopEvent), __FILE__, __LINE__);

    // Warm-up
    transposeCoalesced<<<grid, block>>>(input, output, N);

    cudaCheck(cudaGetLastError(), __FILE__, __LINE__);
    cudaCheck(cudaDeviceSynchronize(), __FILE__, __LINE__);

    // Start timer
    cudaCheck(cudaEventRecord(startEvent), __FILE__, __LINE__);

    for (int i = 0; i < 20; i++)
    {
        transposeCoalesced<<<grid, block>>>(input, output, N);
    }

    cudaCheck(cudaGetLastError(), __FILE__, __LINE__);

    // Stop timer
    cudaCheck(cudaEventRecord(stopEvent), __FILE__, __LINE__);
    cudaCheck(cudaEventSynchronize(stopEvent), __FILE__, __LINE__);

    float milliseconds = 0.0f;

    cudaCheck(cudaEventElapsedTime(
        &milliseconds, startEvent, stopEvent), __FILE__, __LINE__);

    milliseconds /= 20.0f;

    cudaCheck(cudaEventDestroy(startEvent), __FILE__, __LINE__);
    cudaCheck(cudaEventDestroy(stopEvent), __FILE__, __LINE__);

    return milliseconds;
}

// ============================================================
// 6. TRANSPOSE CORRECTNESS CHECK
// ============================================================

bool checkTranspose(const float *output,
                   const float *input,
                   int N)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            float expected = input[i * N + j];
            float actual = output[j * N + i];

            if (fabsf(actual - expected) > 1e-5f)
            {
                return false;
            }
        }
    }

    return true;
}

// ============================================================
// 7. MATRIX MULTIPLICATION CORRECTNESS CHECK
// ============================================================

bool checkMatMul(const float *C, int N)
{
    // A and B contain all ones.
    // Therefore, C[i][j] must equal N.

    size_t elements = (size_t)N * N;

    for (size_t i = 0; i < elements; i++)
    {
        if (fabsf(C[i] - (float)N) > 1e-3f)
        {
            return false;
        }
    }

    return true;
}

// ============================================================
// 8. MAIN
// ============================================================

int main(int argc, char **argv)
{
    int NT = 4096;
    int NM = 1024;

    // Optional sizes from command line
    if (argc >= 3)
    {
        NT = atoi(argv[1]);
        NM = atoi(argv[2]);
    }

    if (NT <= 0 || NM <= 0)
    {
        printf("Invalid matrix size!\n");
        return EXIT_FAILURE;
    }

    printf("\n");
    printf("========================================\n");
    printf(" CUDA MEMORY OPTIMIZATION EXPERIMENT\n");
    printf("========================================\n");

    // --------------------------------------------------------
    // GPU INFORMATION
    // --------------------------------------------------------

    cudaDeviceProp prop;

    cudaCheck(cudaGetDeviceProperties(&prop, 0),
              __FILE__, __LINE__);

    printf("GPU              : %s\n", prop.name);
    printf("Transpose matrix : %d x %d\n", NT, NT);
    printf("MatMul matrix    : %d x %d\n", NM, NM);

    // --------------------------------------------------------
    // MEMORY SIZES
    // --------------------------------------------------------

    size_t transposeElements = (size_t)NT * NT;
    size_t matmulElements = (size_t)NM * NM;

    size_t transposeBytes =
        transposeElements * sizeof(float);

    size_t matmulBytes =
        matmulElements * sizeof(float);

    // --------------------------------------------------------
    // HOST MEMORY ALLOCATION
    // --------------------------------------------------------

    float *h_in = (float *)malloc(transposeBytes);
    float *h_out = (float *)malloc(transposeBytes);

    float *h_A = (float *)malloc(matmulBytes);
    float *h_B = (float *)malloc(matmulBytes);
    float *h_C = (float *)malloc(matmulBytes);

    if (h_in == NULL || h_out == NULL ||
        h_A == NULL || h_B == NULL || h_C == NULL)
    {
        printf("Host memory allocation failed!\n");

        free(h_in);
        free(h_out);
        free(h_A);
        free(h_B);
        free(h_C);

        return EXIT_FAILURE;
    }

    // --------------------------------------------------------
    // INITIALIZE MATRICES
    // --------------------------------------------------------

    for (size_t i = 0; i < transposeElements; i++)
    {
        h_in[i] = (float)(i % 100);
    }

    for (size_t i = 0; i < matmulElements; i++)
    {
        h_A[i] = 1.0f;
        h_B[i] = 1.0f;
        h_C[i] = 0.0f;
    }

    // --------------------------------------------------------
    // GPU MEMORY ALLOCATION
    // --------------------------------------------------------

    float *d_in = nullptr;
    float *d_out = nullptr;

    float *d_A = nullptr;
    float *d_B = nullptr;
    float *d_C = nullptr;

    cudaCheck(cudaMalloc((void **)&d_in, transposeBytes),
              __FILE__, __LINE__);

    cudaCheck(cudaMalloc((void **)&d_out, transposeBytes),
              __FILE__, __LINE__);

    cudaCheck(cudaMalloc((void **)&d_A, matmulBytes),
              __FILE__, __LINE__);

    cudaCheck(cudaMalloc((void **)&d_B, matmulBytes),
              __FILE__, __LINE__);

    cudaCheck(cudaMalloc((void **)&d_C, matmulBytes),
              __FILE__, __LINE__);

    // --------------------------------------------------------
    // COPY INPUTS TO GPU
    // --------------------------------------------------------

    cudaCheck(cudaMemcpy(d_in, h_in, transposeBytes,
                        cudaMemcpyHostToDevice),
              __FILE__, __LINE__);

    cudaCheck(cudaMemcpy(d_A, h_A, matmulBytes,
                        cudaMemcpyHostToDevice),
              __FILE__, __LINE__);

    cudaCheck(cudaMemcpy(d_B, h_B, matmulBytes,
                        cudaMemcpyHostToDevice),
              __FILE__, __LINE__);

    // ========================================================
    // TRANSPOSE EXPERIMENT
    // ========================================================

    dim3 blockTranspose(TILE, TILE);

    dim3 gridTranspose(
        (NT + TILE - 1) / TILE,
        (NT + TILE - 1) / TILE
    );

    // --------------------------------------------------------
    // UNCOALESCED TRANSPOSE
    // --------------------------------------------------------

    float timeNaive = timeNaiveTranspose(
        d_out, d_in, NT,
        gridTranspose, blockTranspose
    );

    cudaCheck(cudaMemcpy(h_out, d_out, transposeBytes,
                        cudaMemcpyDeviceToHost),
              __FILE__, __LINE__);

    bool passNaive = checkTranspose(h_out, h_in, NT);

    double bandwidthNaive =
        (2.0 * transposeBytes) /
        (timeNaive * 1e-3) / 1e9;

    // --------------------------------------------------------
    // COALESCED TRANSPOSE
    // --------------------------------------------------------

    float timeCoalesced = timeCoalescedTranspose(
        d_out, d_in, NT,
        gridTranspose, blockTranspose
    );

    cudaCheck(cudaMemcpy(h_out, d_out, transposeBytes,
                        cudaMemcpyDeviceToHost),
              __FILE__, __LINE__);

    bool passCoalesced = checkTranspose(h_out, h_in, NT);

    double bandwidthCoalesced =
        (2.0 * transposeBytes) /
        (timeCoalesced * 1e-3) / 1e9;

    double speedup = timeNaive / timeCoalesced;

    // ========================================================
    // MATRIX MULTIPLICATION EXPERIMENT
    // ========================================================

    dim3 blockMatMul(BLK, BLK);

    dim3 gridMatMul(
        (NM + BLK - 1) / BLK,
        (NM + BLK - 1) / BLK
    );

    // Warm-up
    matMulNaive<<<gridMatMul, blockMatMul>>>(
        d_A, d_B, d_C, NM
    );

    cudaCheck(cudaGetLastError(), __FILE__, __LINE__);
    cudaCheck(cudaDeviceSynchronize(), __FILE__, __LINE__);

    // Create CUDA events
    cudaEvent_t matStart, matStop;

    cudaCheck(cudaEventCreate(&matStart), __FILE__, __LINE__);
    cudaCheck(cudaEventCreate(&matStop), __FILE__, __LINE__);

    // Start timing
    cudaCheck(cudaEventRecord(matStart), __FILE__, __LINE__);

    // Execute matrix multiplication
    matMulNaive<<<gridMatMul, blockMatMul>>>(
        d_A, d_B, d_C, NM
    );

    cudaCheck(cudaGetLastError(), __FILE__, __LINE__);

    // Stop timing
    cudaCheck(cudaEventRecord(matStop), __FILE__, __LINE__);
    cudaCheck(cudaEventSynchronize(matStop), __FILE__, __LINE__);

    float matmulTime = 0.0f;

    cudaCheck(cudaEventElapsedTime(
        &matmulTime, matStart, matStop),
        __FILE__, __LINE__);

    // Copy output back to CPU
    cudaCheck(cudaMemcpy(h_C, d_C, matmulBytes,
                        cudaMemcpyDeviceToHost),
              __FILE__, __LINE__);

    bool passMatMul = checkMatMul(h_C, NM);

    double gflops =
        (2.0 * NM * NM * NM) /
        (matmulTime * 1e-3) / 1e9;

    cudaCheck(cudaEventDestroy(matStart), __FILE__, __LINE__);
    cudaCheck(cudaEventDestroy(matStop), __FILE__, __LINE__);

    // ========================================================
    // DISPLAY RESULTS
    // ========================================================

    printf("\n");
    printf("========================================\n");
    printf("         TRANSPOSE RESULTS\n");
    printf("========================================\n");

    printf("\nUncoalesced Transpose:\n");
    printf("Time           : %.4f ms\n", timeNaive);
    printf("Bandwidth      : %.2f GB/s\n", bandwidthNaive);
    printf("Correctness    : %s\n",
           passNaive ? "PASS" : "FAIL");

    printf("\nCoalesced Transpose:\n");
    printf("Time           : %.4f ms\n", timeCoalesced);
    printf("Bandwidth      : %.2f GB/s\n", bandwidthCoalesced);
    printf("Correctness    : %s\n",
           passCoalesced ? "PASS" : "FAIL");

    printf("\nTranspose Speedup:\n");
    printf("%.2fx\n", speedup);

    printf("\n");
    printf("========================================\n");
    printf("     MATRIX MULTIPLICATION RESULTS\n");
    printf("========================================\n");

    printf("Matrix Size    : %d x %d\n", NM, NM);
    printf("Execution Time : %.4f ms\n", matmulTime);
    printf("Performance    : %.2f GFLOPS\n", gflops);
    printf("Correctness    : %s\n",
           passMatMul ? "PASS" : "FAIL");

    printf("\n");
    printf("========================================\n");
    printf("       EXPERIMENT COMPLETED\n");
    printf("========================================\n");

    // ========================================================
    // FREE GPU MEMORY
    // ========================================================

    cudaCheck(cudaFree(d_in), __FILE__, __LINE__);
    cudaCheck(cudaFree(d_out), __FILE__, __LINE__);

    cudaCheck(cudaFree(d_A), __FILE__, __LINE__);
    cudaCheck(cudaFree(d_B), __FILE__, __LINE__);
    cudaCheck(cudaFree(d_C), __FILE__, __LINE__);

    // ========================================================
    // FREE HOST MEMORY
    // ========================================================

    free(h_in);
    free(h_out);

    free(h_A);
    free(h_B);
    free(h_C);

    return EXIT_SUCCESS;
}
