#!/bin/bash
# Sweep over CUTLASS GEMM mainloop configurations
# Outputs results to results.csv
#
# Phase 1: Build all binaries in parallel (nvcc is the bottleneck)
# Phase 2: Run sequentially on GPU (need exclusive access for accurate timing)
#
# MMA atom shapes (from sm100_shapes.py):
#   BF16 (DTYPE_ID=0): M={64,128}, N={8..256 step 8/16}, K_atom=16
#   FP8  (DTYPE_ID=1): M={64,128}, N={8..256 step 8/16}, K_atom=32
# CTA TILE_K must be a multiple of K_atom.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
OUTFILE="$SCRIPT_DIR/results.csv"
MAX_JOBS="${MAX_JOBS:-$(nproc)}"

mkdir -p "$BUILD_DIR"
echo "M,N,K,TILE_M,TILE_N,TILE_K,STAGES,dtype,time_ms,TFLOPS" > "$OUTFILE"

STAGE_LIST=(0 2 3 4 5 6 7 8)

# BF16 tile configs: TILE_M,TILE_N,TILE_K
# K_atom=16, so TILE_K must be a multiple of 16.
# Larger TILE_K may fail for large (M,N) due to SMEM limits.
BF16_TILES=(
  # Full TILE_K sweep for all (M,N) combos
  # M=64
  "64,16,16"  "64,16,32"  "64,16,64"  "64,16,128"  "64,16,256"
  "64,32,16"  "64,32,32"  "64,32,64"  "64,32,128"  "64,32,256"
  "64,64,16"  "64,64,32"  "64,64,64"  "64,64,128"  "64,64,256"
  "64,128,16" "64,128,32" "64,128,64" "64,128,128" "64,128,256"
  "64,192,16" "64,192,32" "64,192,64" "64,192,128" "64,192,256"
  "64,256,16" "64,256,32" "64,256,64" "64,256,128" "64,256,256"
  # M=128
  "128,16,16"  "128,16,32"  "128,16,64"  "128,16,128"  "128,16,256"
  "128,32,16"  "128,32,32"  "128,32,64"  "128,32,128"  "128,32,256"
  "128,64,16"  "128,64,32"  "128,64,64"  "128,64,128"  "128,64,256"
  "128,128,16" "128,128,32" "128,128,64" "128,128,128" "128,128,256"
  "128,192,16" "128,192,32" "128,192,64" "128,192,128" "128,192,256"
  "128,256,16" "128,256,32" "128,256,64" "128,256,128" "128,256,256"
)

# FP8 tile configs: TILE_M,TILE_N,TILE_K
# K_atom=32, so TILE_K must be a multiple of 32.
# Larger TILE_K may fail for large (M,N) due to SMEM limits.
FP8_TILES=(
  # Full TILE_K sweep for all (M,N) combos
  # M=64
  "64,16,32"  "64,16,64"  "64,16,128"  "64,16,256"  "64,16,512"
  "64,32,32"  "64,32,64"  "64,32,128"  "64,32,256"  "64,32,512"
  "64,64,32"  "64,64,64"  "64,64,128"  "64,64,256"  "64,64,512"
  "64,128,32" "64,128,64" "64,128,128" "64,128,256" "64,128,512"
  "64,192,32" "64,192,64" "64,192,128" "64,192,256" "64,192,512"
  "64,256,32" "64,256,64" "64,256,128" "64,256,256" "64,256,512"
  # M=128
  "128,16,32"  "128,16,64"  "128,16,128"  "128,16,256"  "128,16,512"
  "128,32,32"  "128,32,64"  "128,32,128"  "128,32,256"  "128,32,512"
  "128,64,32"  "128,64,64"  "128,64,128"  "128,64,256"  "128,64,512"
  "128,128,32" "128,128,64" "128,128,128" "128,128,256" "128,128,512"
  "128,192,32" "128,192,64" "128,192,128" "128,192,256" "128,192,512"
  "128,256,32" "128,256,64" "128,256,128" "128,256,256" "128,256,512"
)

# Collect all (TILE_M,TILE_N,TILE_K,STAGES,DTYPE_ID) configs
ALL_CONFIGS=()
for tile in "${BF16_TILES[@]}"; do
  IFS=',' read -r TM TN TK <<< "$tile"
  for S in "${STAGE_LIST[@]}"; do
    ALL_CONFIGS+=("${TM},${TN},${TK},${S},0")
  done
done
for tile in "${FP8_TILES[@]}"; do
  IFS=',' read -r TM TN TK <<< "$tile"
  for S in "${STAGE_LIST[@]}"; do
    ALL_CONFIGS+=("${TM},${TN},${TK},${S},1")
  done
done

TOTAL=${#ALL_CONFIGS[@]}
echo "=== Phase 1: Building $TOTAL binaries (max $MAX_JOBS parallel jobs) ==="

# ---- Phase 1: Parallel builds ----
CUTLASS_PATH="../../cutlass"
NVCCFLAGS="-std=c++17 --expt-relaxed-constexpr -gencode=arch=compute_110a,code=\"sm_110a,compute_110a\""
INCLUDES="-I${CUTLASS_PATH}/include -I${CUTLASS_PATH}/tools/util/include"
SRC="$SCRIPT_DIR/cutlass_gemm_bench.cu"

job_count=0
fail_count=0

for cfg in "${ALL_CONFIGS[@]}"; do
  IFS=',' read -r TM TN TK S D <<< "$cfg"
  BIN_NAME="bench_${TM}_${TN}_${TK}_s${S}_d${D}"
  BIN_PATH="$BUILD_DIR/$BIN_NAME"
  LOG_PATH="$BUILD_DIR/${BIN_NAME}.log"

  # Skip if already built
  if [[ -x "$BIN_PATH" ]]; then
    continue
  fi

  DEFINES="-DTILE_M=$TM -DTILE_N=$TN -DTILE_K=$TK -DSTAGES=$S -DDTYPE_ID=$D"
  NVCC_PATH="/usr/local/cuda-13.2/bin/nvcc"
  (
    if $NVCC_PATH $NVCCFLAGS $INCLUDES $DEFINES -lcuda -o "$BIN_PATH" "$SRC" > "$LOG_PATH" 2>&1; then
      echo "  [OK] $BIN_NAME"
    else
      echo "  [FAIL] $BIN_NAME (see $LOG_PATH)"
    fi
  ) &

  job_count=$((job_count + 1))
  if (( job_count >= MAX_JOBS )); then
    wait -n 2>/dev/null || true
    job_count=$((job_count - 1))
  fi
done

# Wait for all remaining builds
wait
echo "=== Phase 1 complete ==="

# ---- Phase 2: Sequential runs ----
echo "=== Phase 2: Running benchmarks sequentially ==="

done_count=0
for cfg in "${ALL_CONFIGS[@]}"; do
  IFS=',' read -r TM TN TK S D <<< "$cfg"
  BIN_NAME="bench_${TM}_${TN}_${TK}_s${S}_d${D}"
  BIN_PATH="$BUILD_DIR/$BIN_NAME"
  done_count=$((done_count + 1))

  if [[ -x "$BIN_PATH" ]]; then
    echo -ne "  [$done_count/$TOTAL] $BIN_NAME ... "
    if result=$("$BIN_PATH" 2>/dev/null); then
      echo "$result" >> "$OUTFILE"
      echo "$result"
    else
      echo "# FAILED: $cfg" >> "$OUTFILE"
      echo "FAILED"
    fi
  else
    echo "  [$done_count/$TOTAL] $BIN_NAME ... BUILD_FAILED"
    echo "# BUILD_FAILED: $cfg" >> "$OUTFILE"
  fi
done

echo "=== Done. Results in $OUTFILE ==="
wc -l "$OUTFILE"
