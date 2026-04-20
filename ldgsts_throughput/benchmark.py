#!/usr/bin/env python3
"""
Benchmark LDGSTS throughput across configurations using ncu.
Sweeps CTAS_PER_SM, NUM_STAGES, THREADS_PER_BLOCK, and LOAD_T.
Collects hardware counter metrics via Nsight Compute and outputs to CSV.
"""

import subprocess
import csv
import sys
import os
import argparse

# Load types: name -> size in bytes
LOAD_TYPES = {
    'float': 4,
    'float2': 8,
    'float4': 16,
}

# Default sweep ranges
DEFAULT_CTAS_PER_SM = [1, 2, 3, 4]
DEFAULT_NUM_STAGES = [1, 2, 4]
DEFAULT_THREADS_PER_BLOCK = [64, 128, 256]
DEFAULT_LOAD_TYPES = ['float', 'float2', 'float4']

# ncu metrics to collect
NCU_METRICS = [
    'dram__bytes_read.sum.per_second',
    'sm__cycles_elapsed.avg',
    'sm__sass_inst_executed_op_ldgsts.sum',
    'sm__sass_inst_executed_op_ldgsts.sum.per_cycle_elapsed',
]

CSV_FIELDS = [
    'CTAsPerSM', 'NumStages', 'ThreadsPerBlock', 'LoadType', 'LoadBytes',
    'DRAMBandwidthBps', 'SMCyclesAvg', 'LDGSTSTotal', 'LDGSTSWarpPerCycle',
]


def parse_int_list(s):
    """Parse comma-separated int list: '1,2,4,8'"""
    return [int(x.strip()) for x in s.split(',')]


def find_ncu():
    """Find ncu binary path."""
    result = subprocess.run(["/usr/local/cuda-13.2/bin/ncu"], capture_output=True, text=True)
    if result.returncode == 0:
        return result.stdout.strip()
    # Try common locations
    for path in ["/usr/local/cuda-13.2/bin/ncu", "/opt/nvidia/nsight-compute/ncu"]:
        if os.path.exists(path):
            return path
    return None


def parse_ncu_csv(output):
    """Parse ncu --csv output and return dict of metric name -> value."""
    metrics = {}
    for line in output.split('\n'):
        # ncu CSV format: "ID","Process ID","Process Name",...,"Metric Name","Metric Unit","Metric Value"
        if not line.startswith('"'):
            continue
        parts = line.split('","')
        if len(parts) < 3:
            continue
        # Last three fields are metric name, unit, value
        metric_name = parts[-3].strip('"')
        metric_value = parts[-1].strip('"').replace(',', '')
        if metric_name in NCU_METRICS:
            try:
                metrics[metric_name] = float(metric_value)
            except ValueError:
                pass
    return metrics


def run_benchmark(ctas_per_sm, num_stages, threads_per_block, load_t, ncu_path, verbose=False):
    """Compile and profile benchmark for given configuration."""
    label = f"CTAs={ctas_per_sm}, stages={num_stages}, threads={threads_per_block}, load={load_t}"

    clean_cmd = ["make", "clean"]
    build_cmd = [
        "make", "ldgsts_tput.out",
        f"CTAS_PER_SM={ctas_per_sm}",
        f"NUM_STAGES={num_stages}",
        f"THREADS_PER_BLOCK={threads_per_block}",
        f"LOAD_T={load_t}",
    ]
    ncu_cmd = [
        "sudo", ncu_path,
        "--clock-control", "none",
        "--target-processes", "all",
        "--csv",
        "--metrics", ",".join(NCU_METRICS),
        "./ldgsts_tput.out",
    ]

    try:
        subprocess.run(clean_cmd, capture_output=True, check=True)

        if verbose:
            print(f"Building {label}...", file=sys.stderr)
        result = subprocess.run(build_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Build failed for {label}:", file=sys.stderr)
            print(result.stderr, file=sys.stderr)
            return None

        if verbose:
            print(f"Profiling {label}...", file=sys.stderr)
        result = subprocess.run(ncu_cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            print(f"ncu failed for {label}:", file=sys.stderr)
            print(result.stderr, file=sys.stderr)
            return None
        metrics = parse_ncu_csv(result.stdout)
        if not metrics:
            print(f"Could not parse ncu output for {label}", file=sys.stderr)
            if verbose:
                print(f"stdout: {result.stdout[:500]}", file=sys.stderr)
            return None

        return {
            'CTAsPerSM': ctas_per_sm,
            'NumStages': num_stages,
            'ThreadsPerBlock': threads_per_block,
            'LoadType': load_t,
            'LoadBytes': LOAD_TYPES[load_t],
            'DRAMBandwidthBps': metrics.get('dram__bytes_read.sum.per_second', 0),
            'SMCyclesAvg': metrics.get('sm__cycles_elapsed.avg', 0),
            'LDGSTSTotal': metrics.get('sm__sass_inst_executed_op_ldgsts.sum', 0),
            'LDGSTSWarpPerCycle': metrics.get('sm__sass_inst_executed_op_ldgsts.sum.per_cycle_elapsed', 0),
        }

    except subprocess.TimeoutExpired:
        print(f"Timeout for {label}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error for {label}: {e}", file=sys.stderr)
        return None


def main():
    parser = argparse.ArgumentParser(description='Benchmark LDGSTS throughput via ncu')
    parser.add_argument('-o', '--output', default='benchmark_results.csv',
                        help='Output CSV file (default: benchmark_results.csv)')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite CSV instead of appending')
    parser.add_argument('--ctas', type=str, default=None,
                        help=f'CTAs per SM, comma-separated (default: {DEFAULT_CTAS_PER_SM})')
    parser.add_argument('--stages', type=str, default=None,
                        help=f'Pipeline stages, comma-separated (default: {DEFAULT_NUM_STAGES})')
    parser.add_argument('--threads', type=str, default=None,
                        help=f'Threads per block, comma-separated (default: {DEFAULT_THREADS_PER_BLOCK})')
    parser.add_argument('--load-types', type=str, default=None,
                        help=f'Load types, comma-separated (default: {",".join(DEFAULT_LOAD_TYPES)})')
    args = parser.parse_args()

    ncu_path = "/usr/local/cuda-13.2/bin/ncu" #find_ncu()
    if not ncu_path:
        print("Error: ncu not found. Install NVIDIA Nsight Compute.", file=sys.stderr)
        return 1
    print(f"Using ncu: {ncu_path}", file=sys.stderr)

    ctas_list = parse_int_list(args.ctas) if args.ctas else DEFAULT_CTAS_PER_SM
    stages_list = parse_int_list(args.stages) if args.stages else DEFAULT_NUM_STAGES
    threads_list = parse_int_list(args.threads) if args.threads else DEFAULT_THREADS_PER_BLOCK
    load_types = [x.strip() for x in args.load_types.split(',')] if args.load_types else DEFAULT_LOAD_TYPES

    for lt in load_types:
        if lt not in LOAD_TYPES:
            print(f"Error: Invalid load type '{lt}'. Valid: {list(LOAD_TYPES.keys())}", file=sys.stderr)
            return 1

    total_runs = len(ctas_list) * len(stages_list) * len(threads_list) * len(load_types)
    print(f"Running {total_runs} configurations...", file=sys.stderr)

    # Setup CSV
    file_exists = os.path.exists(args.output) and not args.overwrite
    csv_file = open(args.output, 'a' if file_exists else 'w', newline='')
    writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
    if not file_exists:
        writer.writeheader()
        csv_file.flush()

    result_count = 0
    run_idx = 0
    for load_t in load_types:
        for ctas in ctas_list:
            for stages in stages_list:
                for threads in threads_list:
                    run_idx += 1
                    result = run_benchmark(ctas, stages, threads, load_t,
                                           ncu_path, verbose=args.verbose)
                    if result:
                        writer.writerow(result)
                        csv_file.flush()
                        result_count += 1
                        bw_gbps = result['DRAMBandwidthBps'] / 1e9
                        ipc = result['LDGSTSWarpPerCycle']
                        print(f"[{run_idx}/{total_runs}] CTAs={ctas:2d}, stages={stages}, "
                              f"threads={threads:4d}, load={load_t:6s}: "
                              f"{bw_gbps:.2f} GB/s, {ipc:.4f} LDGSTS/cyc", file=sys.stderr)
                    else:
                        print(f"[{run_idx}/{total_runs}] FAILED: CTAs={ctas}, stages={stages}, "
                              f"threads={threads}, load={load_t}", file=sys.stderr)

    csv_file.close()

    if result_count > 0:
        print(f"\nSaved {result_count} results to {args.output}", file=sys.stderr)
    else:
        print("No successful results", file=sys.stderr)
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
