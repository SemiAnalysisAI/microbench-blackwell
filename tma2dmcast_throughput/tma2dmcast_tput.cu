#include <cstdlib>
#include <cassert>
#include <cstdio>
#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

constexpr int32_t NUM_SMS = 20;                              // B200 has 148 SMs
constexpr int32_t L2_SIZE = 33554432;                        // B200 L2 cache is 126.5 MiB
constexpr size_t MAX_DATA_VOLUME = 2LL * 1024 * 1024 * 1024;  // 2 GB

constexpr size_t alignDataVolume(size_t factor) {
    return (MAX_DATA_VOLUME / factor) * factor;
}

// Required compile-time configs
#ifndef CLUSTER_SIZE
#error "CLUSTER_SIZE must be defined"
#endif
#ifndef SMEM_WIDTH
#error "SMEM_WIDTH must be defined"
#endif
#ifndef SMEM_HEIGHT
#error "SMEM_HEIGHT must be defined"
#endif
#ifndef USE_MULTICAST
#define USE_MULTICAST 1
#endif
#ifndef NUM_WARPS
#define NUM_WARPS 4
#endif
// For mode 3: how many CTAs share the same data (logical group, not a HW cluster)
// Defaults to CLUSTER_SIZE for convenience but can be overridden independently.
#ifndef SHARING_GROUP_SIZE
#define SHARING_GROUP_SIZE CLUSTER_SIZE
#endif
// Compile-time configuration
constexpr int32_t kClusterSize = CLUSTER_SIZE;
constexpr int32_t kSmemWidth = SMEM_WIDTH;
constexpr int32_t kSmemHeight = SMEM_HEIGHT;
constexpr int32_t kNumWarps = NUM_WARPS;
constexpr int32_t kThreadsPerCta = kNumWarps * 32;
constexpr int32_t kSharingGroupSize = SHARING_GROUP_SIZE;

// ============================================================================
// Multicast modes:
//   0 = implicit L2 sharing, intra-cluster
//       All CTAs in a HW cluster read the SAME address.
//       CTAs are co-located on the same GPC => LRC can coalesce.
//       L2 sees reuse from cluster-mates on the same GPC.
//
//   1 = explicit TMA multicast
//       Leader CTA issues cp.async.bulk.tensor...multicast to all cluster peers.
//       Data fetched once from L2, fanned out via SM-to-SM network.
//       L2 sees ~1 request per tile regardless of cluster size.
//
//   2 = no sharing baseline
//       Each CTA in the cluster reads DIFFERENT unique data.
//       No L2 reuse possible between cluster members.
//       True worst-case L2 traffic baseline.
//
//   3 = implicit L2 sharing, cross-cluster
//       CLUSTER_SIZE is forced to 1 (no HW cluster, no LRC dedup).
//       SHARING_GROUP_SIZE consecutive blocks read the SAME address.
//       CTAs are freely scheduled across GPCs by the HW — they're NOT
//       co-located on the same GPC, so each goes through a different LRC.
//       L2 is the only cache that can provide reuse.
//       Compare with mode 0 to isolate LRC vs L2-only dedup.
//       Sweep SHARING_GROUP_SIZE to higher values (8, 16, ...) to stress L2.
// ============================================================================
constexpr int kMulticastMode = USE_MULTICAST;

// For mode 3, enforce cluster_size=1 at compile time
static_assert(kMulticastMode != 3 || kClusterSize == 1,
    "Mode 3 (cross-cluster) requires CLUSTER_SIZE=1 (no HW cluster)");

// Size of one tile that each warp loads
constexpr size_t kTileBytes = kSmemWidth * kSmemHeight * sizeof(float);
constexpr size_t kTotalBufferBytes = kNumWarps * kTileBytes;
constexpr size_t kBarriersBytes = kNumWarps * sizeof(uint64_t);
constexpr size_t kBarriersOffset = (kTotalBufferBytes + 7) & ~size_t(7);
constexpr size_t kDynamicSmemBytes = kBarriersOffset + kBarriersBytes;

// B200 max dynamic smem = 227 KiB = 232448 bytes
constexpr size_t kMaxDynamicSmem = 227 * 1024;
static_assert(kDynamicSmemBytes <= kMaxDynamicSmem,
    "Tile configuration exceeds B200 max dynamic shared memory (227 KiB)");

// TMA box_size constraints (SWIZZLE_NONE, float32)
static_assert(kSmemWidth <= 128,
    "SMEM_WIDTH must be <= 128 for float32 with CU_TENSOR_MAP_SWIZZLE_NONE");
static_assert(kSmemHeight <= 256,
    "SMEM_HEIGHT must be <= 256 (TMA boxDim limit)");

//
// Inline PTX helper functions
//

__device__ __forceinline__ void mbarrier_init(uint64_t* bar, uint32_t count) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
                 :: "r"(smem_addr), "r"(count) : "memory");
}

__device__ __forceinline__ void fence_mbarrier_init_release_cluster() {
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}

__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint64_t* bar, uint32_t tx_count) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
                 :: "r"(smem_addr), "r"(tx_count) : "memory");
}

__device__ __forceinline__ bool mbarrier_try_wait_parity(uint64_t* bar, uint32_t phase_parity) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    uint32_t wait_complete;
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [%1], %2;\n\t"
        "selp.u32 %0, 1, 0, p;\n\t"
        "}"
        : "=r"(wait_complete)
        : "r"(smem_addr), "r"(phase_parity)
        : "memory");
    return wait_complete != 0;
}

__device__ __forceinline__ void tma_load_2d_multicast(
    void* dst, const void* tensor_map, int32_t x, int32_t y, uint64_t* bar, uint16_t cta_mask) {
    uint32_t smem_dst = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    uint32_t smem_bar = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster "
        "[%0], [%1, {%2, %3}], [%4], %5;"
        :: "r"(smem_dst), "l"(tensor_map), "r"(x), "r"(y), "r"(smem_bar), "h"(cta_mask)
        : "memory");
}

__device__ __forceinline__ void tma_load_2d(
    void* dst, const void* tensor_map, int32_t x, int32_t y, uint64_t* bar) {
    uint32_t smem_dst = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    uint32_t smem_bar = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes "
        "[%0], [%1, {%2, %3}], [%4];"
        :: "r"(smem_dst), "l"(tensor_map), "r"(x), "r"(y), "r"(smem_bar)
        : "memory");
}

// ============================================================================
// Kernel
// ============================================================================
template<int MulticastMode>
__global__ __cluster_dims__(kClusterSize, 1, 1)
void bulkAsyncCopyTensor2DKernel(const __grid_constant__ CUtensorMap tensor_map, uint64_t gmem_height) {
    extern __shared__ __align__(128) char smem_raw[];

    float* smem_buffer_base = reinterpret_cast<float*>(smem_raw);
    uint64_t* bars = reinterpret_cast<uint64_t*>(smem_raw + kBarriersOffset);

    cg::cluster_group cluster = cg::this_cluster();
    int block_rank = cluster.block_rank();

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    // Initialize barriers (one per warp)
    if (lane_id == 0) {
        mbarrier_init(&bars[warp_id], 1);
    }
    __syncthreads();
    fence_mbarrier_init_release_cluster();
    cluster.sync();

    constexpr uint16_t ctaMask = (1 << kClusterSize) - 1;

    uint32_t phase = 0;

    // Rows one CTA processes per iteration (all warps)
    constexpr int32_t rows_per_cta_iter = kNumWarps * kSmemHeight;

    // Rows consumed per "group" per iteration:
    //   Mode 0/1/3: all CTAs in the group read the same data => rows_per_cta_iter
    //   Mode 2:     each CTA reads unique data => kClusterSize * rows_per_cta_iter
    constexpr int32_t rows_per_group_iter = (MulticastMode == 2)
        ? kClusterSize * rows_per_cta_iter
        : rows_per_cta_iter;

    // How many independent groups tile the grid
    const int32_t num_groups = [&]() {
        if constexpr (MulticastMode == 3) {
            return (int32_t)(gridDim.x / kSharingGroupSize);
        } else {
            return (int32_t)(gridDim.x / kClusterSize);
        }
    }();

    const int32_t my_group = [&]() {
        if constexpr (MulticastMode == 3) {
            return (int32_t)(blockIdx.x / kSharingGroupSize);
        } else {
            return (int32_t)(blockIdx.x / kClusterSize);
        }
    }();

    const int32_t rank_in_group = [&]() {
        if constexpr (MulticastMode == 3) {
            return (int32_t)(blockIdx.x % kSharingGroupSize);
        } else {
            return block_rank;
        }
    }();

    // Iteration stride (in groups): for modes 0/1, use gridDim.x so the
    // leading CTA's memory access pattern is identical across cluster sizes.
    // Mode 2 already CL-independent (rows_per_group_iter scales with CL).
    const int32_t stride_groups = [&]() {
        if constexpr (MulticastMode <= 1) return (int32_t)gridDim.x;
        else return num_groups;
    }();

    for (int64_t iter = 0; ; iter++) {
        int64_t group_base = rows_per_group_iter * my_group
                           + iter * (int64_t)rows_per_group_iter * stride_groups;

        if (group_base >= (int64_t)gmem_height) break;

        // Compute this CTA's base row
        int64_t cta_base;
        if constexpr (MulticastMode == 2) {
            cta_base = group_base + block_rank * rows_per_cta_iter;
        } else {
            cta_base = group_base;
        }

        int64_t my_row = cta_base + warp_id * kSmemHeight;
        float* my_buffer = smem_buffer_base + warp_id * kSmemHeight * kSmemWidth;

        if (my_row < (int64_t)gmem_height) {
            if (lane_id == 0) {
                mbarrier_arrive_expect_tx(&bars[warp_id], kTileBytes);
            }

            if constexpr (MulticastMode == 1) {
                if (block_rank == 0 && lane_id == 0) {
                    tma_load_2d_multicast(my_buffer, &tensor_map, 0, (int32_t)my_row, &bars[warp_id], ctaMask);
                }
            } else {
                if (lane_id == 0) {
                    tma_load_2d(my_buffer, &tensor_map, 0, (int32_t)my_row, &bars[warp_id]);
                }
            }

            while (!mbarrier_try_wait_parity(&bars[warp_id], phase));
        }

        phase ^= 1;
        cluster.sync();
    }
}


int main() {
    constexpr int32_t kCtasPerSm = 1;
    int32_t num_blks = (NUM_SMS * kCtasPerSm / kClusterSize) * kClusterSize;

    // For mode 3, round down to multiple of kSharingGroupSize
    if constexpr (kMulticastMode == 3) {
        num_blks = (num_blks / kSharingGroupSize) * kSharingGroupSize;
    }

    int32_t num_clusters = num_blks / kClusterSize;
    int32_t num_groups = (kMulticastMode == 3) ? num_blks / kSharingGroupSize : num_clusters;

    // Data allocation: target ~2 GB, aligned to the iteration stride so every
    // iteration is complete.
    //
    // For modes 0/1 the iteration stride uses num_blks (not num_groups) so
    // the leading CTA iterates identically regardless of cluster size.
    // For mode 2 the stride is already CL-independent (CL factors cancel).
    // For mode 3 the stride uses num_groups as before.
    constexpr uint64_t kGmemWidth = kSmemWidth;

    size_t rows_per_group_iter = (size_t)kNumWarps * kSmemHeight;
    if (kMulticastMode == 2) rows_per_group_iter *= kClusterSize;

    size_t stride_groups = (kMulticastMode <= 1) ? (size_t)num_blks : (size_t)num_groups;
    size_t iter_stride_bytes = rows_per_group_iter * stride_groups * sizeof(float) * kGmemWidth;

    size_t data_size = alignDataVolume(iter_stride_bytes);
    // Cap at MAX_DATA_VOLUME
    if (data_size > MAX_DATA_VOLUME) data_size = alignDataVolume(iter_stride_bytes);
    uint64_t gmem_height = (data_size / sizeof(float)) / kGmemWidth;

    const char* mode_names[] = {
        "implicit_l2_intra_cluster",
        "explicit_tma_multicast",
        "no_sharing",
        "implicit_l2_cross_cluster"
    };

    printf("Config: CLUSTER=%d SHARING_GROUP=%d WIDTH=%d HEIGHT=%d MODE=%d (%s)\n",
           kClusterSize, kSharingGroupSize, kSmemWidth, kSmemHeight,
           kMulticastMode, mode_names[kMulticastMode]);
    printf("  Tile: %zu bytes/load, bif: %zu KiB, smem: %zu bytes, data: %.2f MiB, gmem: %lux%lu\n",
           kTileBytes, (kNumWarps * kTileBytes) / 1024,
           kDynamicSmemBytes,
           (double)data_size / (1024*1024),
           (unsigned long)kGmemWidth, (unsigned long)gmem_height);
    printf("  Blocks: %d, Groups: %d, Threads/block: %d\n",
           num_blks, num_groups, kThreadsPerCta);

    float *data = (float*) malloc(data_size);
    if (!data) { fprintf(stderr, "Host malloc failed\n"); return 1; }
    srand((uint32_t) kSmemWidth + kSmemHeight);
    for (size_t i = 0; i < kGmemWidth * gmem_height; i++) {
        data[i] = rand();
    }

    float *d_data;
    cudaMalloc(&d_data, data_size);
    cudaMemcpy(d_data, data, data_size, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    CUtensorMap tensor_map{};
    constexpr uint32_t rank = 2;
    uint64_t size[rank] = {kGmemWidth, gmem_height};
    uint64_t stride[rank - 1] = {kGmemWidth * sizeof(float)};
    uint32_t box_size[rank] = {kSmemWidth, kSmemHeight};
    uint32_t elem_stride[rank] = {1, 1};

    CUresult res = cuTensorMapEncodeTiled(
        &tensor_map,
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
        rank,
        d_data,
        size,
        stride,
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

    // Opt in to dynamic shared memory > 48 KiB
    auto kernel_fn = bulkAsyncCopyTensor2DKernel<kMulticastMode>;

    // Pin smem allocation to max so the HW carveout is constant (228 KiB)
    // across all tile sizes.  Even though the kernel has no explicit L1
    // loads, the TMA engine lives inside L1TEX and its internal request
    // tracking may be affected by the SRAM partition.  Pinning eliminates
    // carveout-transition artifacts between the 100 and 132 KiB steps.
    cudaFuncSetAttribute(
        kernel_fn,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        kMaxDynamicSmem
    );

    // Launch
    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeClusterDimension;
    attribute[0].val.clusterDim.x = kClusterSize;
    attribute[0].val.clusterDim.y = 1;
    attribute[0].val.clusterDim.z = 1;

    cudaLaunchConfig_t config = {0};
    config.gridDim = num_blks;
    config.blockDim = kThreadsPerCta;
    config.dynamicSmemBytes = kMaxDynamicSmem;
    config.attrs = attribute;
    config.numAttrs = 1;

    cudaError_t err = cudaLaunchKernelEx(&config, kernel_fn, tensor_map, gmem_height);
    if (err != cudaSuccess) {
        fprintf(stderr, "Launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cudaFree(d_data);
    free(data);
    return 0;
}
