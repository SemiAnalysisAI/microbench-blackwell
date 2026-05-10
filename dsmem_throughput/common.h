#ifndef DSMEM_COMMON_H
#define DSMEM_COMMON_H

#include <cstdint>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

constexpr int32_t NUM_SMS = 20;                              // B200 has 20 SMs
constexpr int32_t L2_SIZE = 33554432;                        // B200 L2 cache is 32 MiB
constexpr size_t MAX_DATA_VOLUME = 2LL * 1024 * 1024 * 1024;  // 2 GB

#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t err = (call);                                                \
        if (err != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                         \
                    __FILE__, __LINE__, cudaGetErrorString(err));                \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    } while (0)

// For checking kernel launches (asynchronous errors)
#define CUDA_CHECK_LAST()                                                        \
    do {                                                                         \
        cudaError_t err = cudaGetLastError();                                    \
        if (err != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA kernel error at %s:%d: %s\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(err));                \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    } while (0)

constexpr size_t alignDataVolume(size_t factor) {
    return (MAX_DATA_VOLUME / factor) * factor;
}

enum class DsmemAccessMode : int {
    LOCAL = 0,  // Each CTA reads from its own smem
    BCAST = 1,  // CTA 0 pushes to all other CTAs
    RING  = 2   // Every CTA n pushes to CTA (n+STRIDE)%CL
};

template<DsmemAccessMode Mode>
__device__ __forceinline__ int32_t get_cluster_rank(cg::cluster_group &cluster);

template<>
__device__ __forceinline__ int32_t get_cluster_rank<DsmemAccessMode::LOCAL>(cg::cluster_group &cluster) {
    return cluster.block_rank();  // Each CTA reads its own smem
    // return 0; // 直接返回0，表示每个CTA都从自己的smem读取数据
}

template<>
__device__ __forceinline__ int32_t get_cluster_rank<DsmemAccessMode::BCAST>(cg::cluster_group &cluster) {
    if (cluster.block_rank() == 0)
      return 1;
    return 0; 
}

// 如果你需要增加一个 RING 模式
template<>
__device__ __forceinline__ int32_t get_cluster_rank<DsmemAccessMode::RING>(cg::cluster_group &cluster) {
    // 读取下一个相邻 Block 的数据，形成一个环 (0读1, 1读2, 2读3, 3读0)
    uint32_t rank = cluster.block_rank();
    uint32_t num_blocks = cluster.num_blocks();
    return (rank + 1) % num_blocks; 
}

template<typename T, DsmemAccessMode dsmemMode>
__device__ __forceinline__ float dsmem_load(const T *remote_buffer);

template<>
__device__ __forceinline__ float dsmem_load<float4, DsmemAccessMode::LOCAL>(const float4 *remote_buffer_addr) {
    float dummy_val[4];
    asm volatile(
        "ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];"
        : "=f"(dummy_val[0]), "=f"(dummy_val[1]), "=f"(dummy_val[2]), "=f"(dummy_val[3])
        : "l"(remote_buffer_addr)
        : "memory"
    );
    return dummy_val[0] + dummy_val[1] + dummy_val[2] + dummy_val[3];
}

template<>
__device__ __forceinline__ float dsmem_load<float4, DsmemAccessMode::BCAST>(const float4 *remote_buffer_addr) {
    float dummy_val[4];
    asm volatile(
        "ld.shared::cluster.v4.f32 {%0, %1, %2, %3}, [%4];"
        : "=f"(dummy_val[0]), "=f"(dummy_val[1]), "=f"(dummy_val[2]), "=f"(dummy_val[3])
        : "l"(remote_buffer_addr)
        : "memory"
    );
    return dummy_val[0] + dummy_val[1] + dummy_val[2] + dummy_val[3];
}

template<>
__device__ __forceinline__ float dsmem_load<float, DsmemAccessMode::LOCAL>(const float *remote_buffer_addr) {
    float val;
    asm volatile(
        "ld.shared.f32 %0, [%1];"
        : "=f"(val)
        : "l"(remote_buffer_addr)
        : "memory"
    );
    return val;
}

// 2. BCAST 模式下的 float 读取 (跨 SM 读取)
template<>
__device__ __forceinline__ float dsmem_load<float, DsmemAccessMode::BCAST>(const float *remote_buffer_addr) {
    float val;
    asm volatile(
        "ld.shared::cluster.f32 %0, [%1];"
        : "=f"(val)
        : "l"(remote_buffer_addr)
        : "memory"
    );
    return val;
}

// 3. RING 模式下的 float 读取 (底层指令和 BCAST 一样，都是跨 SM)
template<>
__device__ __forceinline__ float dsmem_load<float, DsmemAccessMode::RING>(const float *remote_buffer_addr) {
    float val;
    asm volatile(
        "ld.shared::cluster.f32 %0, [%1];"
        : "=f"(val)
        : "l"(remote_buffer_addr)
        : "memory"
    );
    return val;
}

#endif // DSMEM_COMMON_H
