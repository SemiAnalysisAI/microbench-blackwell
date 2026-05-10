/*
 * CUTLASS GEMM Mainloop Throughput Benchmark
 *
 * Measures steady-state mainloop throughput of CUTLASS Blackwell (SM100)
 * GEMM kernels, varying pipeline stages and CTA tile sizes.
 *
 * Compile-time parameters (via -D flags):
 *   TILE_M, TILE_N, TILE_K  - CTA tile dimensions
 *   STAGES                  - smem pipeline depth (0 = auto)
 *   DTYPE_ID                - 0=BF16, 1=FP8_E4M3
 */

#include <iostream>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <utility>

#include "cute/tensor.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

// ---- Compile-time defaults ----
#ifndef TILE_M
#define TILE_M 128
#endif
#ifndef TILE_N
#define TILE_N 128
#endif
#ifndef TILE_K
#define TILE_K 64
#endif
#ifndef STAGES
#define STAGES 0
#endif
#ifndef DTYPE_ID
#define DTYPE_ID 0
#endif

// ---- Constants ----
static constexpr int NUM_SMS = 20;  // B200
static constexpr int K_DIM = 131072; // 128K — vocab-sized K to isolate mainloop steady state
static constexpr int WARMUP_ITERS = 1;
static constexpr int BENCH_ITERS = 20;

// ---- dtype selection ----
#if DTYPE_ID == 0
using ElementAB = cutlass::bfloat16_t;
static const char* dtype_str = "BF16";
#elif DTYPE_ID == 1
using ElementAB = cutlass::float_e4m3_t;
static const char* dtype_str = "FP8_E4M3";
#else
#error "Unknown DTYPE_ID (0=BF16, 1=FP8_E4M3)"
#endif

using ElementAccumulator = float;
using ElementD = cutlass::bfloat16_t;  // Output always BF16

// ---- Helper: zero init (throughput benchmark, correctness not needed) ----
template <class Element>
void initialize_block(cutlass::DeviceAllocation<Element>& block) {
  cudaMemset(block.get(), 0, block.size() * sizeof(Element));
}

// ---- Pick best factorization of NUM_SMS into (m_tiles, n_tiles) ----
// Picks the pair that makes M and N most square.
static std::pair<int,int> pick_factorization(int tile_m, int tile_n) {
  std::vector<std::pair<int,int>> factors;
  for (int i = 1; i <= NUM_SMS; ++i) {
    if (NUM_SMS % i == 0) {
      factors.push_back({i, NUM_SMS / i});
    }
  }
  double best_ratio = 1e18;
  std::pair<int,int> best = {1, NUM_SMS};
  for (auto [mt, nt] : factors) {
    int M = mt * tile_m;
    int N = nt * tile_n;
    double ratio = (M > N) ? double(M) / N : double(N) / M;
    if (ratio < best_ratio) {
      best_ratio = ratio;
      best = {mt, nt};
    }
  }
  return best;
}

// ---- GEMM type construction ----
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutD = cutlass::layout::ColumnMajor;

using ClusterShape = Shape<_1, _1, _1>;
using MmaTileShape = Shape<Int<TILE_M>, Int<TILE_N>, Int<TILE_K>>;

static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementAB>::value;
static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementAB>::value;
static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

using MainloopSchedule = cutlass::gemm::KernelTmaWarpSpecialized1SmSm100;
using EpilogueSchedule = cutlass::epilogue::NoSmemWarpSpecialized1Sm;

using FusionOp = cutlass::epilogue::fusion::LinearCombination<
    ElementD, float, void, float>;  // No C source (beta=0)

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    MmaTileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, float,
    void, LayoutD, AlignmentD,   // No C input
    ElementD, LayoutD, AlignmentD,
    EpilogueSchedule,
    FusionOp
  >::CollectiveOp;

#if STAGES == 0
using StageCountType = cutlass::gemm::collective::StageCountAutoCarveout<
    static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>;
#else
using StageCountType = cutlass::gemm::collective::StageCount<STAGES>;
#endif

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    ElementAB, LayoutA, AlignmentA,
    ElementAB, LayoutB, AlignmentB,
    ElementAccumulator,
    MmaTileShape, ClusterShape,
    StageCountType,
    MainloopSchedule
  >::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue
>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

#endif // CUTLASS_ARCH_MMA_SM100_SUPPORTED

int main() {
#if !defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
  std::cerr << "SM100 not supported by this build." << std::endl;
  return 1;
#else

  // ---- Compute GEMM dimensions ----
  // Single tile: M=TILE_M, N=TILE_N — isolates one CTA's mainloop throughput
  int M = TILE_M;
  int N = TILE_N;
  int K = K_DIM;

  // ---- Strides ----
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
  auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

  // ---- Allocate and init ----
  cutlass::DeviceAllocation<ElementAB> block_A(M * K);
  cutlass::DeviceAllocation<ElementAB> block_B(K * N);
  cutlass::DeviceAllocation<ElementD>  block_D(M * N);

  initialize_block(block_A);
  initialize_block(block_B);

  // ---- HW info ----
  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = 0;
  hw_info.sm_count = 1;  // Single tile = single CTA

  // ---- Arguments ----
  using ProblemShape = typename Gemm::GemmKernel::ProblemShape;
  ProblemShape problem_size{M, N, K, 1};

  typename Gemm::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    problem_size,
    {block_A.get(), stride_A, block_B.get(), stride_B},
    {{1.0f, 0.0f},       // alpha, beta
     nullptr, stride_D,   // C (unused)
     block_D.get(), stride_D},
    hw_info
  };

  Gemm gemm_op;

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  auto status = gemm_op.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    std::cerr << "Cannot implement: TILE_M=" << TILE_M
              << " TILE_N=" << TILE_N << " TILE_K=" << TILE_K
              << " STAGES=" << STAGES << " dtype=" << dtype_str
              << " error=" << cudaGetErrorString(cudaGetLastError()) << std::endl;
    return 1;
  }

  status = gemm_op.initialize(arguments, workspace.get());
  if (status != cutlass::Status::kSuccess) {
    std::cerr << "Init failed: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
    return 1;
  }

  // ---- Warm-up ----
  for (int i = 0; i < WARMUP_ITERS; ++i) {
    gemm_op.run();
  }
  cudaDeviceSynchronize();

  // ---- Benchmark ----
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  for (int i = 0; i < BENCH_ITERS; ++i) {
    gemm_op.run();
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float total_ms = 0;
  cudaEventElapsedTime(&total_ms, start, stop);
  float avg_ms = total_ms / BENCH_ITERS;

  // ---- Compute TFLOPS (single CTA) ----
  double flops = 2.0 * M * N * K;  // multiply-add = 2 ops
  double tflops = (flops / (avg_ms * 1e-3)) / 1e12;

  // ---- Report ----
  // CSV: M,N,K,TILE_M,TILE_N,TILE_K,STAGES,dtype,time_ms,TFLOPS
  std::cout << M << "," << N << "," << K << ","
            << TILE_M << "," << TILE_N << "," << TILE_K << ","
            << STAGES << "," << dtype_str << ","
            << avg_ms << "," << tflops << std::endl;

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return 0;
#endif
}
