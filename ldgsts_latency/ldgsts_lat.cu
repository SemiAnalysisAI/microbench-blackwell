#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_pipeline.h>

// Required compile-time configs
#ifndef CTAS_PER_SM
#error "CTAS_PER_SM must be defined"
#endif
#ifndef THREADS_PER_BLOCK
#error "THREADS_PER_BLOCK must be defined"
#endif
#ifndef LOAD_T
#error "LOAD_T must be defined"
#endif

constexpr int NUM_ITERS = 10;
constexpr int32_t NUM_SMS = 20;         // B200 has 148 SMs
constexpr int32_t L2_SIZE = 33554432;   // B200 L2 cache is 126.5 MiB
constexpr int32_t LOAD_SIZE = sizeof(LOAD_T);

// ============================================================================
// Kernel: exactly 1 LDGSTS per thread — no pipeline stages, no loop.
//
// Each thread loads its own unique slot from DRAM into shared memory.
// clock64() around the LDGSTS+wait measures the true round-trip latency.
// ============================================================================
__global__ void ldgstsLatKernel(LOAD_T *arr, int32_t iter_offset, uint32_t *cycles_out) {
    __shared__ LOAD_T buff[THREADS_PER_BLOCK];

    // Each thread owns a unique LOAD_T slot in the global array
    int32_t slot = blockIdx.x * blockDim.x + threadIdx.x;

    // Measure just the LDGSTS + wait latency.
    // Use asm volatile to prevent compiler reordering of clock reads
    // relative to the memory operations.
    uint64_t start, end;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(start) :: "memory");

    __pipeline_memcpy_async(&buff[threadIdx.x], arr + iter_offset + slot, LOAD_SIZE);
    __pipeline_commit();       // LDGDEPBAR
    __pipeline_wait_prior(0);  // DEPBAR.LE SB, 0 — wait for the single in-flight load

    // Volatile sink: prevent compiler DCE of the entire load sequence
    (void)(*reinterpret_cast<volatile uint32_t*>(&buff[threadIdx.x]));

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(end) :: "memory");

    // Write per-thread cycle count
    cycles_out[blockIdx.x * blockDim.x + threadIdx.x] = (uint32_t)(end - start);
}


int main() {
    int32_t num_blks = NUM_SMS * CTAS_PER_SM;
    int32_t threads_total = num_blks * THREADS_PER_BLOCK;
    size_t arr_size = (size_t)NUM_ITERS * threads_total * LOAD_SIZE;

    fprintf(stderr,
        "Config: CTAS_PER_SM=%d THREADS_PER_BLOCK=%d LOAD_SIZE=%d\n"
        "  num_blks=%d threads_total=%d arr_size=%.2f MiB\n",
        CTAS_PER_SM, THREADS_PER_BLOCK, LOAD_SIZE,
        num_blks, threads_total, arr_size / (1024.0 * 1024.0)
    );

    // Alloc + random-init host data
    char *arr = (char*) malloc(arr_size);
    srand((uint32_t)(CTAS_PER_SM * THREADS_PER_BLOCK + LOAD_SIZE));
    for (size_t i = 0; i < arr_size; i++) {
        arr[i] = (char)rand();
    }

    LOAD_T *d_arr;
    cudaMalloc(&d_arr, arr_size);
    cudaMemcpy(d_arr, arr, arr_size, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    // Device + host buffers for per-thread cycle counts
    uint32_t *d_cycles;
    cudaMalloc(&d_cycles, threads_total * sizeof(uint32_t));
    uint32_t *h_cycles = (uint32_t*) malloc(threads_total * sizeof(uint32_t));

    // Accumulators across iterations for overall stats
    uint32_t overall_min = UINT32_MAX;
    uint32_t overall_max = 0;
    // Collect per-iter medians to compute overall median
    uint32_t iter_medians[NUM_ITERS];

    // Launch NUM_ITERS times — each launch reads a distinct DRAM region.
    for (int i = 0; i < NUM_ITERS; i++) {
        // Flush L2: alloc, touch, free — forces eviction of prior data
        void *flush_buf;
        cudaMalloc(&flush_buf, L2_SIZE);
        cudaMemset(flush_buf, 0xA5, L2_SIZE);
        cudaDeviceSynchronize();
        cudaFree(flush_buf);

        int32_t iter_offset = i * threads_total;
        ldgstsLatKernel<<<num_blks, THREADS_PER_BLOCK>>>(d_arr, iter_offset, d_cycles);
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            fprintf(stderr, "Kernel failed on iter %d: %s\n", i, cudaGetErrorString(err));
            return 1;
        }

        cudaMemcpy(h_cycles, d_cycles, threads_total * sizeof(uint32_t), cudaMemcpyDeviceToHost);

        // Compute per-iteration stats
        std::sort(h_cycles, h_cycles + threads_total);
        uint32_t iter_min = h_cycles[0];
        uint32_t iter_max = h_cycles[threads_total - 1];
        uint32_t iter_med = h_cycles[threads_total / 2];

        overall_min = std::min(overall_min, iter_min);
        overall_max = std::max(overall_max, iter_max);
        iter_medians[i] = iter_med;

        fprintf(stderr, "  iter %d: min=%u med=%u max=%u cycles\n", i, iter_min, iter_med, iter_max);
    }

    // Overall median = median of per-iter medians
    std::sort(iter_medians, iter_medians + NUM_ITERS);
    uint32_t overall_med = iter_medians[NUM_ITERS / 2];

    // Print parseable summary to stdout
    printf("SUMMARY,%u,%u,%u\n", overall_med, overall_min, overall_max);

    cudaFree(d_cycles);
    cudaFree(d_arr);
    free(h_cycles);
    free(arr);
    return 0;
}
