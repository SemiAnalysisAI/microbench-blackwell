#include <cstdlib>
#include <cassert>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>

// Required compile-time configs
#ifndef CTAS_PER_SM
#error "CTAS_PER_SM must be defined"
#endif
#ifndef NUM_STAGES
#error "NUM_STAGES must be defined"
#endif
#ifndef SMEM_WIDTH
#error "SMEM_WIDTH must be defined"
#endif
#ifndef SMEM_HEIGHT
#error "SMEM_HEIGHT must be defined"
#endif


constexpr int32_t NUM_SMS = 20; // B200 has 148 SMs
constexpr int32_t L2_SIZE = 33554432; // B200 L2 cache is 126.5 MiB
constexpr size_t MAX_DATA_VOLUME = 2LL * 1024 * 1024 * 1024;  // 2 GB

constexpr size_t alignDataVolume(size_t factor) {
    return (MAX_DATA_VOLUME / factor) * factor;
}

// Size of one tile per stage
constexpr size_t TILE_BYTES = SMEM_WIDTH * SMEM_HEIGHT * sizeof(float);
constexpr size_t BARRIERS_OFFSET = (NUM_STAGES * TILE_BYTES + 7) & ~size_t(7);

// ============================================================================
// Kernel: single-warp multi-stage TMA 2D unicast pipeline
//
// 1 warp per CTA.  Thread 0 issues all TMA loads via elect_one semantics.
// NUM_STAGES pipeline buffers, each with its own SMEM tile + mbarrier.
// Prologue fills the pipeline, steady state waits oldest and reuses the slot,
// drain waits on remaining in-flight loads.
// ============================================================================
__global__ void tma2dUnicastKernel(const __grid_constant__ CUtensorMap tensor_map, uint64_t gmem_height) {
    extern __shared__ __align__(128) char smem_raw[];
    float* smem_buffer_base = reinterpret_cast<float*>(smem_raw);
    uint64_t* bars = reinterpret_cast<uint64_t*>(smem_raw + BARRIERS_OFFSET);

    // Initialize barriers (one per stage, thread 0 only)
    if (threadIdx.x == 0) {
        for (int s = 0; s < NUM_STAGES; s++) {
            uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&bars[s]));
            asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
                         :: "r"(smem_addr), "r"(1u) : "memory");
        }
    }
    __syncthreads();
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    __syncthreads();

    const int64_t row_stride = (int64_t)gridDim.x * SMEM_HEIGHT;
    int64_t row = (int64_t)blockIdx.x * SMEM_HEIGHT;
    uint32_t phase[NUM_STAGES] = { 0 };

    // Prologue: fill pipeline (issue NUM_STAGES loads without waiting)
    for (int pending = 0; pending < NUM_STAGES; pending++) {
        float* buf = smem_buffer_base + pending * SMEM_HEIGHT * SMEM_WIDTH;
        if (threadIdx.x == 0) {
            uint32_t smem_bar = static_cast<uint32_t>(__cvta_generic_to_shared(&bars[pending]));
            uint32_t smem_dst = static_cast<uint32_t>(__cvta_generic_to_shared(buf));
            asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
                         :: "r"(smem_bar), "r"((uint32_t)TILE_BYTES) : "memory");
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes "
                "[%0], [%1, {%2, %3}], [%4];"
                :: "r"(smem_dst), "l"(&tensor_map), "r"(0), "r"((int32_t)row), "r"(smem_bar)
                : "memory");
        }
        row += row_stride;
    }

    // Steady state: wait oldest, reuse slot for new load
    int slot = 0;
    for (; row < (int64_t)gmem_height; row += row_stride) {
        uint32_t smem_bar = static_cast<uint32_t>(__cvta_generic_to_shared(&bars[slot]));
        uint32_t done = 0;
        while (!done) {
            asm volatile(
                "{\n\t"
                ".reg .pred p;\n\t"
                "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [%1], %2;\n\t"
                "selp.u32 %0, 1, 0, p;\n\t"
                "}"
                : "=r"(done) : "r"(smem_bar), "r"(phase[slot]) : "memory");
        }
        phase[slot] ^= 1;

        float* buf = smem_buffer_base + slot * SMEM_HEIGHT * SMEM_WIDTH;
        if (threadIdx.x == 0) {
            uint32_t smem_dst = static_cast<uint32_t>(__cvta_generic_to_shared(buf));
            asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
                         :: "r"(smem_bar), "r"((uint32_t)TILE_BYTES) : "memory");
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes "
                "[%0], [%1, {%2, %3}], [%4];"
                :: "r"(smem_dst), "l"(&tensor_map), "r"(0), "r"((int32_t)row), "r"(smem_bar)
                : "memory");
        }

        slot = (slot + 1) % NUM_STAGES;
    }

    // Drain: wait remaining in-flight loads
    for (int s = 0; s < NUM_STAGES; s++) {
        uint32_t smem_bar = static_cast<uint32_t>(__cvta_generic_to_shared(&bars[slot]));
        uint32_t done = 0;
        while (!done) {
            asm volatile(
                "{\n\t"
                ".reg .pred p;\n\t"
                "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [%1], %2;\n\t"
                "selp.u32 %0, 1, 0, p;\n\t"
                "}"
                : "=r"(done) : "r"(smem_bar), "r"(phase[slot]) : "memory");
        }
        slot = (slot + 1) % NUM_STAGES;
    }
}


int main() {
    constexpr int32_t THREADS_PER_CTA = 32;    // 1 warp
    constexpr size_t MAX_SMEM_BYTES = 232448;  // B200 max dynamic SMEM = 227 KiB
    static_assert(SMEM_WIDTH  <= 256, "SMEM_WIDTH must be <= 256 (TMA boxDim limit per dimension)");
    static_assert(SMEM_HEIGHT <= 256, "SMEM_HEIGHT must be <= 256 (TMA boxDim limit per dimension)");

    int32_t num_blks = NUM_SMS * CTAS_PER_SM;
    size_t data_size = alignDataVolume(num_blks * TILE_BYTES);
    uint64_t gmem_height = (data_size / sizeof(float)) / SMEM_WIDTH;

    printf(
        "Config: CTAS_PER_SM=%d NUM_STAGES=%d SMEM_WIDTH=%d SMEM_HEIGHT=%d BIF_PER_SM=%zu\n",
        CTAS_PER_SM, NUM_STAGES, SMEM_WIDTH, SMEM_HEIGHT, (size_t)CTAS_PER_SM * NUM_STAGES * TILE_BYTES
    );

    float *data = (float*) malloc(data_size);
    srand((uint32_t) SMEM_WIDTH + SMEM_HEIGHT + NUM_STAGES);
    for (size_t i = 0; i < SMEM_WIDTH * gmem_height; i++) {
        data[i] = rand();
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

    // Flush L2 cache
    void *flush_arr;
    cudaMalloc(&flush_arr, L2_SIZE);
    cudaMemset(flush_arr, 0xA5, L2_SIZE);
    cudaDeviceSynchronize();
    cudaFree(flush_arr);

    // Pin SMEM allocation to max so the HW carveout is constant across configs
    cudaFuncSetAttribute(
        tma2dUnicastKernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        MAX_SMEM_BYTES
    );

    // Launch
    tma2dUnicastKernel<<<num_blks, THREADS_PER_CTA, MAX_SMEM_BYTES>>>(tensor_map, gmem_height);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cudaFree(d_data);
    free(data);
    return 0;
}
