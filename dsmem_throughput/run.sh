#!/bin/bash
set -euo pipefail
NVCC=/usr/local/cuda-13.2/bin/nvcc
SRC="dsmem_tput.cu"
BIN="dsmem_tput.out"
NVCC_GENCODE='-gencode=arch=compute_110a,code="sm_110a,compute_110a"'
THREADS_PER_CTA=512
LOAD_T=float
N_ITERS=10000
OUTPUT_FILE="dsmem_cpasync_sweep.csv"

NCU_METRICS="\
sm__cycles_elapsed.avg,\
gpu__time_duration.avg,\
sm__sass_data_bytes_mem_shared_op_ld.sum.per_second,\
sm__sass_data_bytes_mem_shared_op_ld.max.per_second,\
sm__sass_data_bytes_mem_shared_op_ld.max.pct_of_peak_sustained_active,\
sm__sass_data_bytes_mem_shared_op_st.sum.per_second,\
sm__sass_data_bytes_mem_shared_op_st.max.per_second,\
sm__sass_data_bytes_mem_shared_op_st.max.pct_of_peak_sustained_active"

# Dump SASS instruction counts from the compiled binary
print_sass_mem_summary() {
    local bin=$1
    local all_instrs
    all_instrs=$(cuobjdump --dump-sass "$bin" 2>/dev/null \
        | grep -oP '^\s+/\*\w+\*/\s+\K[A-Z][A-Z0-9_.]+' \
        | sort | uniq -c | sort -rn)

    echo "  SASS memory/sync instructions:"
    echo "$all_instrs" \
        | grep -iE 'LD|ST|ATOM|RED|BAR|CP|MAPA|MBARRIER|QSPC|BLK|BULK|ASYNC|UCPY' \
        | while read count instr; do
            printf "    %-30s %d\n" "$instr" "$count"
          done

    echo "  All SASS instructions:"
    echo "$all_instrs" \
        | while read count instr; do
            printf "    %-30s %d\n" "$instr" "$count"
          done
}

# Pretty-print a bytes/s value with auto units
fmt_bps() {
    awk -v v="$1" 'BEGIN {
        if (v >= 1e12)      printf "%.2f TB/s", v/1e12
        else if (v >= 1e9)  printf "%.2f GB/s", v/1e9
        else if (v >= 1e6)  printf "%.2f MB/s", v/1e6
        else if (v >= 1e3)  printf "%.2f KB/s", v/1e3
        else                printf "%.0f B/s",  v
    }'
}

# Header
echo "mode,cluster_size,stride,total_bytes,gpu_time_ns,sm_cycles,smem_ld_sum_Bps,smem_ld_max_Bps,smem_ld_pct_peak,smem_st_sum_Bps,smem_st_max_Bps,smem_st_pct_peak,push_throughput_GBs" > "$OUTPUT_FILE"

declare -a CONFIGS=()

# mode, cluster_size, stride
# LOCAL baseline
#for cl in 2 4 8; do
#    CONFIGS+=("local $cl 1")
#done

# BCAST
for cl in 2 4; do
    CONFIGS+=("bcast $cl 1")
done

# RING stride=1
for cl in 2 4 8; do
    CONFIGS+=("ring $cl 1")
done

# RING stride=2
for cl in 4 8; do
    CONFIGS+=("ring $cl 2")
done

# Compute total_bytes from the same formula as the kernel
barrier_section=128
buffer_bytes=$(( ((232448 - barrier_section) / 9) & ~127 ))
load_size=4  # sizeof(float)
n_buffer=$((buffer_bytes / load_size))
total_bytes=$(( (n_buffer * load_size / 16) * 16 ))

total=${#CONFIGS[@]}
idx=0

for cfg in "${CONFIGS[@]}"; do
    read -r mode cl stride <<< "$cfg"
    idx=$((idx + 1))
    echo "[$idx/$total] mode=$mode cl=$cl stride=$stride"

    $NVCC $NVCC_GENCODE -std=c++17 -Xptxas=-v \
        -DCLUSTER_SIZE=$cl -DTHREADS_PER_CTA=$THREADS_PER_CTA \
        -DLOAD_T=$LOAD_T -DSTRIDE=$stride \
        -DACCESS_MODE=$([ "$mode" = "local" ] && echo 0 || ([ "$mode" = "bcast" ] && echo 1 || echo 2)) \
        -o "$BIN" "$SRC" 2>&1 | grep -v "^$"

    print_sass_mem_summary "$BIN"

    NCU_OUTPUT=$(
        sudo $(which ncu) --clock-control none --csv --metrics "$NCU_METRICS" \
            ./"$BIN" 2>/dev/null
    )

    gpu_time=$(echo "$NCU_OUTPUT" | grep "gpu__time_duration" | awk -F'","' '{print $NF}' | tr -d '"'|tr -d ',')
    sm_cycles=$(echo "$NCU_OUTPUT" | grep "sm__cycles_elapsed" | awk -F'","' '{print $NF}' | tr -d '"'|tr -d ',')
    smem_ld_sum=$(echo "$NCU_OUTPUT" | grep "op_ld.*sum.per_second" | awk -F'","' '{print $NF}' | tr -d '"'|tr -d ',')
    smem_ld_max=$(echo "$NCU_OUTPUT" | grep "op_ld.*max.per_second" | tail -1 | awk -F'","' '{print $NF}' | tr -d '"'|tr -d ',')
    smem_ld_pct=$(echo "$NCU_OUTPUT" | grep "op_ld.*pct_of_peak" | awk -F'","' '{print $NF}' | tr -d '"'|tr -d ',')
    smem_st_sum=$(echo "$NCU_OUTPUT" | grep "op_st.*sum.per_second" | awk -F'","' '{print $NF}' | tr -d '"'|tr -d ',')
    smem_st_max=$(echo "$NCU_OUTPUT" | grep "op_st.*max.per_second" | tail -1 | awk -F'","' '{print $NF}' | tr -d '"'|tr -d ',')
    smem_st_pct=$(echo "$NCU_OUTPUT" | grep "op_st.*pct_of_peak" | awk -F'","' '{print $NF}' | tr -d '"'|tr -d ',')

    # Push throughput (GB/s) = data_moved / gpu_time_ns
    #   local: all CL CTAs read their own smem -> CL * N_ITERS * total_bytes
    #   bcast: CTA 0 pushes to (CL-1) consumers -> (CL-1) * N_ITERS * total_bytes
    #   ring:  all CL CTAs each push once       -> CL * N_ITERS * total_bytes
    case "$mode" in
        local) n_transfers=$cl ;;
        bcast) n_transfers=$((cl - 1)) ;;
        ring)  n_transfers=$cl ;;
    esac
    data_moved=$(echo "$N_ITERS $total_bytes $n_transfers" | awk '{printf "%.0f", $1 * $2 * $3}')
    push_tput=$(echo "$data_moved $gpu_time" | awk '{if ($2 > 0) printf "%.4f", $1 / $2; else print "0"}')

    smem_ld_fmt=$(fmt_bps "$smem_ld_max")
    smem_st_fmt=$(fmt_bps "$smem_st_max")

    echo "$mode,$cl,$stride,$total_bytes,$gpu_time,$sm_cycles,$smem_ld_sum,$smem_ld_max,$smem_ld_pct,$smem_st_sum,$smem_st_max,$smem_st_pct,$push_tput" >> "$OUTPUT_FILE"
    echo "  -> push=${push_tput} GB/s  smem_ld=${smem_ld_fmt}  smem_st=${smem_st_fmt}  ld_pct=${smem_ld_pct}%  st_pct=${smem_st_pct}%"
done

echo ""
echo "Results saved to $OUTPUT_FILE"
echo ""
column -t -s',' "$OUTPUT_FILE"
