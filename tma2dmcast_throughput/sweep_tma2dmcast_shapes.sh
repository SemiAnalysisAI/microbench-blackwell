#!/bin/bash
set -euo pipefail

NCU_BIN="/usr/local/cuda-13.2/bin/ncu"
OUTPUT_FILE="tma2dmcast_sweep_full.csv"
BUILD_DIR="build_sweep"
SRC="tma2dmcast_tput.cu"

NUM_WARPS=4
MAX_DYNAMIC_SMEM=232448
COMPILE_JOBS=${COMPILE_JOBS:-$(nproc)}

# ============================================================================
# Modes / sizes / shapes
# ============================================================================
HW_CLUSTER_SIZES=("2" "4")
CROSS_CLUSTER_GROUP_SIZES=("2" "4" "8" "16" "32")

SHAPE_COMBINATIONS=(
    "32 8"    "64 4"    "128 4"
    "32 16"   "64 8"
    "32 32"   "64 16"   "128 8"
    "32 48"   "64 24"   "128 12"
    "32 64"   "64 32"   "128 16"
    "32 80"   "64 40"   "128 20"
    "32 96"   "64 48"   "128 24"
    "32 112"  "64 56"   "128 28"
    "32 128"  "64 64"   "128 32"
    "32 160"  "64 80"   "128 40"
    "32 176"  "64 88"   "128 44"
    "64 96"   "128 48"
    "128 52"  "128 56"  "128 60"
    "64 128"  "128 64"
    "128 72"  "128 80"
    "64 192"  "128 96"
    "64 224"  "128 112"
)

# ============================================================================
# NCU metrics
# ============================================================================
METRICS=$(cat <<'METRICS_EOF'
gpu__time_duration.sum,
sm__cycles_elapsed.avg,
l1tex__tmain_requests.sum,
smsp__l1tex_tmain_requests.sum,
l1tex__m_l1tex2xbar_req_cycles_active_op_tma.sum,
l1tex__m_l1tex2xbar_read_requests_mem_global_op_tma_ld_dest_multicast.sum,
l1tex__m_l1tex2xbar_read_requests_mem_global_op_tma_ld_dest_multicast_size_32B.sum,
l1tex__m_l1tex2xbar_read_requests_mem_global_op_tma_ld_dest_multicast_size_64B.sum,
l1tex__m_l1tex2xbar_read_requests_mem_global_op_tma_ld_dest_multicast_size_96B.sum,
l1tex__m_l1tex2xbar_read_requests_mem_global_op_tma_ld_dest_multicast_size_128B.sum,
l1tex__m_l1tex2xbar_read_requests_replayed_mem_global_op_tma_ld_dest_multicast.sum,
l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld.sum,
l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld_dest_self.sum,
l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld_dest_multicast.sum,
l1tex__m_xbar2l1tex_read_sectors_mem_global_op_tma_ld.sum,
l1tex__m_xbar2l1tex_read_sectors_mem_global_op_tma_ld_dest_self.sum,
l1tex__m_xbar2l1tex_read_sectors_mem_global_op_tma_ld_dest_multicast.sum,
l1tex__m_xbar2l1tex_read_bytes_pipe_tma.sum,
l1tex__m_xbar2l1tex_read_sectors_pipe_tma.sum,
lts__t_sectors.sum,
lts__t_sectors_op_read.sum,
lts__t_sectors_srcunit_tex.sum,
lts__t_sectors_srcunit_tex_op_read.sum,
lts__t_requests_srcunit_tex.sum,
lts__t_sectors_srcunit_tex_lookup_hit.sum,
lts__t_sectors_srcunit_tex_lookup_miss.sum,
lts__t_sectors_lookup_hit.sum,
lts__t_sectors_lookup_miss.sum,
lts__t_sectors_hit_rate.pct,
dram__bytes_read.sum,
dram__bytes.sum,
dram__sectors_read.sum,
dram__throughput.avg.pct_of_peak_sustained_elapsed,
gcc__xbar2gcc_sectors.sum,
smsp__warps_issue_stalled_barrier.avg,
smsp__warps_issue_stalled_long_scoreboard.avg,
smsp__warps_issue_stalled_mio_throttle.avg,
lrc__ilc_input_sectors.sum
METRICS_EOF
)
METRICS=$(echo "$METRICS" | tr -d '\n' | tr -d ' ')

# ============================================================================
# CSV header
# ============================================================================
HEADER="mode,mode_name,cluster_size,sharing_group,smem_width,smem_height,"
HEADER+="tile_bytes,total_smem_bytes,"
HEADER+="gpu_time_ns,"
HEADER+="sm_cycles,"
HEADER+="tma_instructions,"
HEADER+="smsp_tma_instructions,"
HEADER+="tma_xbar_active_cycles,"
HEADER+="tma_mcast_requests,"
HEADER+="tma_mcast_req_32B,"
HEADER+="tma_mcast_req_64B,"
HEADER+="tma_mcast_req_96B,"
HEADER+="tma_mcast_req_128B,"
HEADER+="tma_mcast_req_replayed,"
HEADER+="tma_ld_bytes,"
HEADER+="tma_ld_bytes_self,"
HEADER+="tma_ld_bytes_mcast,"
HEADER+="tma_ld_sectors,"
HEADER+="tma_ld_sectors_self,"
HEADER+="tma_ld_sectors_mcast,"
HEADER+="tma_pipe_bytes,"
HEADER+="tma_pipe_sectors,"
HEADER+="lts_sectors_total,"
HEADER+="lts_sectors_read,"
HEADER+="lts_sectors_tex,"
HEADER+="lts_sectors_tex_read,"
HEADER+="lts_requests_tex,"
HEADER+="lts_tex_hit,"
HEADER+="lts_tex_miss,"
HEADER+="lts_hit,"
HEADER+="lts_miss,"
HEADER+="lts_hit_rate_pct,"
HEADER+="dram_bytes_read,"
HEADER+="dram_bytes_total,"
HEADER+="dram_sectors_read,"
HEADER+="dram_throughput_pct_peak,"
HEADER+="gcc_xbar2gcc_sectors,"
HEADER+="stall_barrier,"
HEADER+="stall_long_scoreboard,"
HEADER+="stall_mio_throttle,"
HEADER+="lrc_ilc_input_sectors"
echo "$HEADER" > "$OUTPUT_FILE"

# ============================================================================
# Helpers
# ============================================================================
# extract_metric() {
#     local ncu_output="$1"
#     local metric_name="$2"
#     local val
#     val=$(echo "$ncu_output" | grep "\"${metric_name}\"" | tail -1 \
#         | awk -F',' '{print $NF}' | tr -d '"' | tr -d ' ')
#     if [[ -z "$val" ]]; then echo "N/A"; else echo "$val"; fi
# }

extract_metric() {
    local ncu_output="$1"
    local metric_name="$2"
    local val
    # 1. 将分隔符改为 '","' 避免切断带逗号的数字
    # 2. 增加 tr -d ',' 将提取出数字中的千分位逗号删掉
    val=$(echo "$ncu_output" | grep "\"${metric_name}\"" | tail -1 \
        | awk -F'","' '{print $NF}' | tr -d '"' | tr -d ',' | tr -d ' ')
    if [[ -z "$val" ]]; then echo "N/A"; else echo "$val"; fi
}



mode_name() {
    case "$1" in
        0) echo "implicit_l2_intra" ;;
        1) echo "explicit_mcast" ;;
        2) echo "no_sharing" ;;
        3) echo "implicit_l2_cross" ;;
    esac
}

validate_smem() {
    local w=$1 h=$2
    if (( w > 128 || h > 256 )); then return 1; fi
    local buf=$((NUM_WARPS * w * h * 4))
    local off=$(( (buf + 7) & ~7 ))
    TOTAL_SMEM=$((off + NUM_WARPS * 8))
    TILE_BYTES=$((w * h * 4))
    if (( TOTAL_SMEM > MAX_DYNAMIC_SMEM )); then return 1; fi
    return 0
}

# Unique binary name for a config
bin_name() {
    local mode=$1 cluster=$2 sharing=$3 w=$4 h=$5
    echo "${BUILD_DIR}/tma_m${mode}_c${cluster}_g${sharing}_${w}x${h}"
}

# Arch flags — adjust for your GPU.
#   B200/B100: compute_100a / sm_100a
#   H100:      compute_90a  / sm_90a
# The 'a' suffix = arch-accelerated features (TMA multicast needs this).
NVCC_GENCODE="${NVCC_GENCODE:--gencode=arch=compute_110a,code=sm_110a}"

# ============================================================================
# Phase 1: enumerate all configs, compile in parallel
# ============================================================================
echo ""
echo "================================================================"
echo "=== Phase 1: Compiling all configs (parallel, -j${COMPILE_JOBS}) ==="
echo "================================================================"

mkdir -p "$BUILD_DIR"

JOBFILE=$(mktemp)
CONFIG_LIST=$(mktemp)
trap "rm -f $JOBFILE $CONFIG_LIST" EXIT

add_config() {
    local mode=$1 cluster=$2 sharing=$3 w=$4 h=$5
    if ! validate_smem "$w" "$h"; then return; fi
    local bin
    bin=$(bin_name "$mode" "$cluster" "$sharing" "$w" "$h")
    if [[ -f "${bin}" ]]; then return; fi
    echo "$mode $cluster $sharing $w $h $bin" >> "$CONFIG_LIST"
    printf '/usr/local/cuda-13.2/bin/nvcc %s -O2 -std=c++17 -lcuda -DCLUSTER_SIZE=%d -DSMEM_WIDTH=%d -DSMEM_HEIGHT=%d -DUSE_MULTICAST=%d -DNUM_WARPS=%d -DSHARING_GROUP_SIZE=%d -o %s %s 2>&1 && echo "OK: %s" || echo "FAIL: %s"\n' \
        "$NVCC_GENCODE" "$cluster" "$w" "$h" "$mode" "$NUM_WARPS" "$sharing" "$bin" "$SRC" "$bin" "$bin" \
        >> "$JOBFILE"
}

# Shape sweep: modes 0 (implicit L2) and 1 (explicit multicast), CL=1/2/4
for combo in "${SHAPE_COMBINATIONS[@]}"; do
    read -r W H <<< "$combo"
    add_config "0" "1" "1" "$W" "$H"
    add_config "1" "1" "1" "$W" "$H"
done

for mode in "0" "1"; do
    for cs in "${HW_CLUSTER_SIZES[@]}"; do
        for combo in "${SHAPE_COMBINATIONS[@]}"; do
            read -r W H <<< "$combo"
            add_config "$mode" "$cs" "$cs" "$W" "$H"
        done
    done
done

TOTAL_COMPILE=$(wc -l < "$CONFIG_LIST")
echo "Compiling ${TOTAL_COMPILE} unique binaries with -j${COMPILE_JOBS} ..."
t0=$(date +%s)

# Each "command" in JOBFILE is one multi-line nvcc invocation ending at the CMD
# heredoc boundary. We use bash -c to run each.  The jobfile has one nvcc per line.
cat "$JOBFILE" | xargs -P "$COMPILE_JOBS" -I{} bash -c '{}'

t1=$(date +%s)
echo "Compilation done in $((t1 - t0))s."

COMPILED_OK=$(ls "${BUILD_DIR}"/tma_m* 2>/dev/null | wc -l)
echo "Binaries built: ${COMPILED_OK} / ${TOTAL_COMPILE}"
echo ""

# ============================================================================
# Phase 2: profile sequentially (NCU needs exclusive GPU access)
# ============================================================================
echo "================================================================"
echo "=== Phase 2: Profiling with NCU (sequential) ==="
echo "================================================================"

profiled=0
skipped=0

run_ncu() {
    local mode=$1 cluster=$2 sharing=$3 w=$4 h=$5
    local mname
    mname=$(mode_name "$mode")

    validate_smem "$w" "$h" || return  # sets TILE_BYTES, TOTAL_SMEM

    local bin
    bin=$(bin_name "$mode" "$cluster" "$sharing" "$w" "$h")

    if [[ ! -x "$bin" ]]; then
        echo "SKIP (no binary): ${bin}"
        skipped=$((skipped + 1))
        return
    fi

    echo "--- MODE=${mode}(${mname}) C=${cluster} G=${sharing} ${w}x${h} ---"

    local NCU_OUT
    NCU_OUT=$(
        sudo "$NCU_BIN" \
            --clock-control none \
            --csv \
            --metrics "$METRICS" \
            "./${bin}" 2>/dev/null
    ) || {
        echo "  NCU FAILED"
        skipped=$((skipped + 1))
        return
    }

    # --- Extract all metrics ---
    local gpu_time=$(extract_metric "$NCU_OUT" "gpu__time_duration.sum")
    local sm_cycles=$(extract_metric "$NCU_OUT" "sm__cycles_elapsed.avg")
    local tma_inst=$(extract_metric "$NCU_OUT" "l1tex__tmain_requests.sum")
    local smsp_tma=$(extract_metric "$NCU_OUT" "smsp__l1tex_tmain_requests.sum")
    local tma_cyc=$(extract_metric "$NCU_OUT" "l1tex__m_l1tex2xbar_req_cycles_active_op_tma.sum")
    local mc_req=$(extract_metric "$NCU_OUT" "l1tex__m_l1tex2xbar_read_requests_mem_global_op_tma_ld_dest_multicast.sum")
    local mc_32=$(extract_metric "$NCU_OUT" "l1tex__m_l1tex2xbar_read_requests_mem_global_op_tma_ld_dest_multicast_size_32B.sum")
    local mc_64=$(extract_metric "$NCU_OUT" "l1tex__m_l1tex2xbar_read_requests_mem_global_op_tma_ld_dest_multicast_size_64B.sum")
    local mc_96=$(extract_metric "$NCU_OUT" "l1tex__m_l1tex2xbar_read_requests_mem_global_op_tma_ld_dest_multicast_size_96B.sum")
    local mc_128=$(extract_metric "$NCU_OUT" "l1tex__m_l1tex2xbar_read_requests_mem_global_op_tma_ld_dest_multicast_size_128B.sum")
    local mc_replay=$(extract_metric "$NCU_OUT" "l1tex__m_l1tex2xbar_read_requests_replayed_mem_global_op_tma_ld_dest_multicast.sum")
    local ld_bytes=$(extract_metric "$NCU_OUT" "l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld.sum")
    local ld_bytes_self=$(extract_metric "$NCU_OUT" "l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld_dest_self.sum")
    local ld_bytes_mc=$(extract_metric "$NCU_OUT" "l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld_dest_multicast.sum")
    local ld_sec=$(extract_metric "$NCU_OUT" "l1tex__m_xbar2l1tex_read_sectors_mem_global_op_tma_ld.sum")
    local ld_sec_self=$(extract_metric "$NCU_OUT" "l1tex__m_xbar2l1tex_read_sectors_mem_global_op_tma_ld_dest_self.sum")
    local ld_sec_mc=$(extract_metric "$NCU_OUT" "l1tex__m_xbar2l1tex_read_sectors_mem_global_op_tma_ld_dest_multicast.sum")
    local pipe_bytes=$(extract_metric "$NCU_OUT" "l1tex__m_xbar2l1tex_read_bytes_pipe_tma.sum")
    local pipe_sec=$(extract_metric "$NCU_OUT" "l1tex__m_xbar2l1tex_read_sectors_pipe_tma.sum")
    local lts_total=$(extract_metric "$NCU_OUT" "lts__t_sectors.sum")
    local lts_read=$(extract_metric "$NCU_OUT" "lts__t_sectors_op_read.sum")
    local lts_tex=$(extract_metric "$NCU_OUT" "lts__t_sectors_srcunit_tex.sum")
    local lts_tex_read=$(extract_metric "$NCU_OUT" "lts__t_sectors_srcunit_tex_op_read.sum")
    local lts_req_tex=$(extract_metric "$NCU_OUT" "lts__t_requests_srcunit_tex.sum")
    local lts_tex_hit=$(extract_metric "$NCU_OUT" "lts__t_sectors_srcunit_tex_lookup_hit.sum")
    local lts_tex_miss=$(extract_metric "$NCU_OUT" "lts__t_sectors_srcunit_tex_lookup_miss.sum")
    local lts_hit=$(extract_metric "$NCU_OUT" "lts__t_sectors_lookup_hit.sum")
    local lts_miss=$(extract_metric "$NCU_OUT" "lts__t_sectors_lookup_miss.sum")
    local lts_hit_pct=$(extract_metric "$NCU_OUT" "lts__t_sectors_hit_rate.pct")
    local dram_read=$(extract_metric "$NCU_OUT" "dram__bytes_read.sum")
    local dram_total=$(extract_metric "$NCU_OUT" "dram__bytes.sum")
    local dram_sectors=$(extract_metric "$NCU_OUT" "dram__sectors_read.sum")
    local dram_pct=$(extract_metric "$NCU_OUT" "dram__throughput.avg.pct_of_peak_sustained_elapsed")
    local gcc_xbar=$(extract_metric "$NCU_OUT" "gcc__xbar2gcc_sectors.sum")
    local stall_bar=$(extract_metric "$NCU_OUT" "smsp__warps_issue_stalled_barrier.avg")
    local stall_lsb=$(extract_metric "$NCU_OUT" "smsp__warps_issue_stalled_long_scoreboard.avg")
    local stall_mio=$(extract_metric "$NCU_OUT" "smsp__warps_issue_stalled_mio_throttle.avg")
    local lrc_ilc=$(extract_metric "$NCU_OUT" "lrc__ilc_input_sectors.sum")

    printf "  t=%-8s  tma_ld=%s sec  L2_tex_rd=%s  L2_tex_h/m=%s/%s  DRAM=%s\n" \
        "$gpu_time" "$ld_sec" "$lts_tex_read" "$lts_tex_hit" "$lts_tex_miss" "$dram_read"

    local ROW="${mode},${mname},${cluster},${sharing},${w},${h},"
    ROW+="${TILE_BYTES},${TOTAL_SMEM},"
    ROW+="${gpu_time},${sm_cycles},"
    ROW+="${tma_inst},${smsp_tma},${tma_cyc},"
    ROW+="${mc_req},${mc_32},${mc_64},${mc_96},${mc_128},${mc_replay},"
    ROW+="${ld_bytes},${ld_bytes_self},${ld_bytes_mc},"
    ROW+="${ld_sec},${ld_sec_self},${ld_sec_mc},"
    ROW+="${pipe_bytes},${pipe_sec},"
    ROW+="${lts_total},${lts_read},"
    ROW+="${lts_tex},${lts_tex_read},${lts_req_tex},${lts_tex_hit},${lts_tex_miss},"
    ROW+="${lts_hit},${lts_miss},${lts_hit_pct},"
    ROW+="${dram_read},${dram_total},${dram_sectors},${dram_pct},"
    ROW+="${gcc_xbar},${stall_bar},${stall_lsb},${stall_mio},"
    ROW+="${lrc_ilc}"
    echo "$ROW" >> "$OUTPUT_FILE"
    profiled=$((profiled + 1))
}

# Run NCU in the same order configs were compiled
while IFS=' ' read -r mode cluster sharing w h bin; do
    run_ncu "$mode" "$cluster" "$sharing" "$w" "$h"
done < "$CONFIG_LIST"

echo ""
echo "========================================================================"
echo "Sweep complete.  Profiled: ${profiled}  Skipped: ${skipped}"
echo "Results: ${OUTPUT_FILE}"
echo "========================================================================"
echo ""
echo "=== TMA metric guide ==="
echo ""
echo "TMA engine:"
echo "  tma_instructions        — cp.async.bulk.tensor ops issued"
echo "  tma_xbar_active_cycles  — cycles L1TEX→XBAR active for TMA"
echo ""
echo "TMA multicast outgoing (L1TEX → L2):"
echo "  tma_mcast_requests      — read requests for multicast TMA loads"
echo "  tma_mcast_req_{32..128}B — by request size"
echo "  tma_mcast_req_replayed  — replays from XBAR backpressure"
echo "  (nonzero only in mode 1)"
echo ""
echo "TMA incoming data (L2 → L1TEX):"
echo "  tma_ld_bytes/sectors       — total data received per SM"
echo "  tma_ld_*_self              — from non-multicast loads"
echo "  tma_ld_*_mcast             — from multicast loads"
echo ""
echo "L2 TMA-path:"
echo "  lts_tex_hit/miss  — L2 hit/miss for TEX-sourced traffic"
echo "  Compare lts_sectors_tex_read[mode0] vs [mode3] at same N"
echo "  to measure LRC dedup (mode0 has LRC, mode3 doesn't)"
