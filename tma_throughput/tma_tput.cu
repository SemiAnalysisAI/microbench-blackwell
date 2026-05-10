#include <cstdio>
#include <cstdlib>
#include <cooperative_groups.h>
#include <cuda/ptx>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;
namespace ptx = cuda::ptx;

constexpr int32_t NUM_SMS = 20;                              // B200 has 148 SMs
constexpr int32_t L2_SIZE = 33554432;                        // B200 L2 cache is 126.5 MiB
constexpr size_t MAX_DATA_VOLUME = 2LL * 1024 * 1024 * 1024;  // 2 GB

constexpr size_t alignDataVolume(size_t factor) {
    return (MAX_DATA_VOLUME / factor) * factor;
}


// Required compile-time configs (pass via: make CTAS_PER_SM=X NUM_STAGES=Y LOAD_SIZE=32768)
#ifndef CTAS_PER_SM
#error "CTAS_PER_SM must be defined"
#endif
#ifndef NUM_STAGES
#error "NUM_STAGES must be defined"
#endif
#ifndef LOAD_SIZE
#error "LOAD_SIZE must be defined"
#endif

constexpr size_t ELEMS_PER_LOAD = LOAD_SIZE / sizeof(float);


__global__ void bulkAsyncCopyKernel(float *arr, size_t N) {
    __shared__ alignas(16) float buff[NUM_STAGES][ELEMS_PER_LOAD];
    __shared__ alignas(8) uint64_t bar[NUM_STAGES];

    if (threadIdx.x == 0) {
        for (int32_t s = 0; s < NUM_STAGES; s++) {
            ptx::mbarrier_init(&bar[s], 1);
        }
    }
    __syncthreads();

    int32_t base = ELEMS_PER_LOAD * blockIdx.x;
    int32_t stride = ELEMS_PER_LOAD * gridDim.x;
    int32_t num_iters = (N - base) / stride;

    for (int32_t i = 0; i < num_iters; i++) {
        int32_t slot = i % NUM_STAGES;
        uint32_t parity = (i % (NUM_STAGES * 2)) < NUM_STAGES ? 1 : 0;

        while (i >= NUM_STAGES && !ptx::mbarrier_try_wait_parity(&bar[slot], parity));

        int32_t offset = base + i * stride;
        if (threadIdx.x == 0) {
            cg::invoke_one(cg::coalesced_threads(), [&] () {
                ptx::cp_async_bulk(ptx::space_shared, ptx::space_global, buff[slot], arr + offset, static_cast<uint32_t>(LOAD_SIZE), &bar[slot]);
                ptx::mbarrier_arrive_expect_tx(ptx::sem_release, ptx::scope_cta, ptx::space_shared, &bar[slot], static_cast<uint32_t>(LOAD_SIZE));
            });
        }
    }

    for (int32_t i = num_iters; i < num_iters + NUM_STAGES; i++) {
        int32_t slot = i % NUM_STAGES;
        uint32_t parity = (i % (NUM_STAGES * 2)) < NUM_STAGES ? 1 : 0;
        while (!ptx::mbarrier_try_wait_parity(&bar[slot], parity));
    }
}


int main() {
    int32_t num_blks = NUM_SMS * CTAS_PER_SM;
    size_t arr_size = alignDataVolume(num_blks * LOAD_SIZE * NUM_STAGES);
    size_t N = arr_size / sizeof(float);

    float *arr = (float*) malloc(arr_size);
    srand((uint32_t) NUM_STAGES * LOAD_SIZE);
    for (size_t i = 0; i < N; i++) {
        arr[i] = rand();
    }

    float *d_arr;
    cudaMalloc(&d_arr, arr_size);
    cudaMemcpy(d_arr, arr, arr_size, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    void *flush_arr;
    cudaMalloc(&flush_arr, L2_SIZE);
    cudaMemset(flush_arr, 0xA5, L2_SIZE);
    cudaDeviceSynchronize();
    cudaFree(flush_arr);

    bulkAsyncCopyKernel<<<num_blks, 1>>>(d_arr, N);
    cudaDeviceSynchronize();

    cudaFree(d_arr);
    free(arr);
    return 0;
}
