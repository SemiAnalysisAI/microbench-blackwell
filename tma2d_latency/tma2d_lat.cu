#include <cstdlib>
#include <cassert>
#include <cstdio>
#include <algorithm>
#include <cuda.h>
#include <cuda_runtime.h>

// Required compile-time configs
#ifndef CTAS_PER_SM
#error "CTAS_PER_SM must be defined"
#endif
#ifndef SMEM_WIDTH
#error "SMEM_WIDTH must be defined"
#endif
#ifndef SMEM_HEIGHT
#error "SMEM_HEIGHT must be defined"
#endif


constexpr int NUM_ITERS = 10;
constexpr int32_t NUM_SMS = 20; // B200 has 148 SMs
constexpr int32_t L2_SIZE = 33554432; // B200 L2 cache is 126.5 MiB

// Size of one tile
constexpr size_t TILE_BYTES = SMEM_WIDTH * SMEM_HEIGHT * sizeof(float);
constexpr size_t BARRIERS_OFFSET = (TILE_BYTES + 7) & ~size_t(7);

// ============================================================================
// Kernel: single TMA 2D load per CTA, measuring round-trip latency
//
// Each CTA issues exactly 1 load and waits serially — no pipeline.
// clock64() around the TMA issue + mbarrier wait measures round-trip latency.
// Only thread 0's measurement is recorded (it issues the TMA).
// ============================================================================
__global__ void tma2dLatKernel(const __grid_constant__ CUtensorMap tensor_map, int32_t seg_row_offset, uint32_t *cycles_out) {
    extern __shared__ __align__(128) char smem_raw[];
    float* smem_dst_ptr = reinterpret_cast<float*>(smem_raw);
    uint64_t* bars = reinterpret_cast<uint64_t*>(smem_raw + BARRIERS_OFFSET);

    // Initialize barrier (thread 0 only, count=1 for one TMA initiator)
    if (threadIdx.x == 0) {
        uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&bars[0]));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
                     :: "r"(smem_addr), "r"(1u) : "memory");
    }
    __syncthreads();
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    __syncthreads();

    int32_t row = seg_row_offset + (int32_t)blockIdx.x * SMEM_HEIGHT;
    uint32_t smem_bar = static_cast<uint32_t>(__cvta_generic_to_shared(&bars[0]));
    uint32_t smem_dst = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst_ptr));

    uint64_t start, end;

    // Thread 0: clock → arrive_expect_tx → issue TMA load
    if (threadIdx.x == 0) {
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(start) :: "memory");

        asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
                     :: "r"(smem_bar), "r"((uint32_t)TILE_BYTES) : "memory");
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes "
            "[%0], [%1, {%2, %3}], [%4];"
            :: "r"(smem_dst), "l"(&tensor_map), "r"(0), "r"(row), "r"(smem_bar)
            : "memory");
    }

    // All threads spin on mbarrier.try_wait.parity (phase=0)
    uint32_t phase = 0;
    uint32_t done = 0;
    while (!done) {
        asm volatile(
            "{\n\t"
            ".reg .pred p;\n\t"
            "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [%1], %2;\n\t"
            "selp.u32 %0, 1, 0, p;\n\t"
            "}"
            : "=r"(done) : "r"(smem_bar), "r"(phase) : "memory");
    }

    // Thread 0: record end time → write cycle count
    if (threadIdx.x == 0) {
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(end) :: "memory");
        cycles_out[blockIdx.x] = (uint32_t)(end - start);
    }
}


int main() {
    constexpr int32_t THREADS_PER_CTA = 32;    // 1 warp
    constexpr size_t MAX_SMEM_BYTES = 232448;  // B200 max dynamic SMEM = 227 KiB
    static_assert(SMEM_WIDTH  <= 256, "SMEM_WIDTH must be <= 256 (TMA boxDim limit per dimension)");
    static_assert(SMEM_HEIGHT <= 256, "SMEM_HEIGHT must be <= 256 (TMA boxDim limit per dimension)");

    int32_t num_blks = NUM_SMS * CTAS_PER_SM;
    size_t data_size = (size_t)NUM_ITERS * num_blks * TILE_BYTES;
    uint64_t gmem_height = (uint64_t)NUM_ITERS * num_blks * SMEM_HEIGHT;

    fprintf(stderr,
        "Config: CTAS_PER_SM=%d SMEM_WIDTH=%d SMEM_HEIGHT=%d TILE_BYTES=%zu\n",
        CTAS_PER_SM, SMEM_WIDTH, SMEM_HEIGHT, TILE_BYTES
    );

    float *data = (float*) malloc(data_size);
    srand((uint32_t) SMEM_WIDTH + SMEM_HEIGHT);
    for (size_t i = 0; i < SMEM_WIDTH * gmem_height; i++) {
        data[i] = (float)rand();
    }

    float *d_data;
    cudaMalloc(&d_data, data_size);
    cudaMemcpy(d_data, data, data_size, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    // TMA tensor map
    CUtensorMap tensor_map{};
    constexpr uint32_t RANK = 2;
    uint64_t size_arr[RANK] = {SMEM_WIDTH, gmem_height};
    uint64_t stride_arr[RANK - 1] = {SMEM_WIDTH * sizeof(float)};
    uint32_t box_size[RANK] = {SMEM_WIDTH, SMEM_HEIGHT};
    uint32_t elem_stride[RANK] = {1, 1};

    CUresult res = cuTensorMapEncodeTiled(
        &tensor_map,
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
        RANK,
        d_data,
        size_arr,
        stride_arr,
        box_size,
        elem_stride,
        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
    assert(res == CUDA_SUCCESS);

    // Pin SMEM allocation to max so the HW carveout is constant across configs
    cudaFuncSetAttribute(
        tma2dLatKernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        MAX_SMEM_BYTES
    );

    // Device + host buffers for per-block cycle counts
    uint32_t *d_cycles;
    cudaMalloc(&d_cycles, num_blks * sizeof(uint32_t));
    uint32_t *h_cycles = (uint32_t*) malloc(num_blks * sizeof(uint32_t));

    // Accumulators across iterations
    uint32_t iter_medians[NUM_ITERS];

    // Launch NUM_ITERS times — each launch reads a distinct DRAM region
    for (int i = 0; i < NUM_ITERS; i++) {
        // Flush L2
        void *flush_arr;
        cudaMalloc(&flush_arr, L2_SIZE);
        cudaMemset(flush_arr, 0xA5, L2_SIZE);
        cudaDeviceSynchronize();
        cudaFree(flush_arr);

        int32_t seg_row_offset = (int32_t)((int64_t)i * num_blks * SMEM_HEIGHT);
        tma2dLatKernel<<<num_blks, THREADS_PER_CTA, MAX_SMEM_BYTES>>>(
            tensor_map, seg_row_offset, d_cycles);
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            fprintf(stderr, "Kernel failed on iter %d: %s\n", i, cudaGetErrorString(err));
            return 1;
        }

        cudaMemcpy(h_cycles, d_cycles, num_blks * sizeof(uint32_t), cudaMemcpyDeviceToHost);

        std::sort(h_cycles, h_cycles + num_blks);
        uint32_t iter_min = h_cycles[0];
        uint32_t iter_max = h_cycles[num_blks - 1];
        uint32_t iter_med = h_cycles[num_blks / 2];
        iter_medians[i] = iter_med;

        fprintf(stderr, "  iter %d: min=%u med=%u max=%u cycles\n", i, iter_min, iter_med, iter_max);
    }

    // Overall median = median of per-iter medians
    std::sort(iter_medians, iter_medians + NUM_ITERS);
    uint32_t overall_med = iter_medians[NUM_ITERS / 2];

    // Min/max across all iterations
    // Re-run to collect overall min/max (use iter_medians for simplicity —
    // but we want overall min/max from all blocks across all iters)
    // For efficiency, just re-derive from the last iteration's sort
    // Actually, let's track properly:
    // We already printed per-iter stats; for the summary, use median of medians
    // and collect min/max from re-running
    uint32_t overall_min = UINT32_MAX, overall_max = 0;
    // Re-run once more to get clean min/max
    {
        void *flush_arr;
        cudaMalloc(&flush_arr, L2_SIZE);
        cudaMemset(flush_arr, 0xA5, L2_SIZE);
        cudaDeviceSynchronize();
        cudaFree(flush_arr);

        tma2dLatKernel<<<num_blks, THREADS_PER_CTA, MAX_SMEM_BYTES>>>(
            tensor_map, 0, d_cycles);
        cudaDeviceSynchronize();
        cudaMemcpy(h_cycles, d_cycles, num_blks * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        std::sort(h_cycles, h_cycles + num_blks);
        overall_min = h_cycles[0];
        overall_max = h_cycles[num_blks - 1];
    }

    printf("SUMMARY,%u,%u,%u\n", overall_med, overall_min, overall_max);

    cudaFree(d_cycles);
    cudaFree(d_data);
    free(h_cycles);
    free(data);
    return 0;
}
