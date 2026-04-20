#!/usr/bin/env python3
"""
Benchmark TMA 2D unicast throughput across configurations using ncu.
Sweeps SMEM_HEIGHT and NUM_STAGES with fixed SMEM_WIDTH=128 and CTAS_PER_SM=1.
Collects hardware counter metrics via Nsight Compute and outputs to CSV.
"""

import subprocess
import csv
import sys
import os
import argparse

# Default sweep ranges
DEFAULT_CTAS_PER_SM = [1]
DEFAULT_NUM_STAGES = [1, 2, 4]
DEFAULT_SMEM_WIDTH = 128
DEFAULT_SMEM_HEIGHTS = [1, 2, 4, 8, 16, 32, 64, 128, 256]

# ncu metrics to collect
NCU_METRICS = [
    'dram__bytes_read.sum.per_second',
    'sm__cycles_elapsed.avg',
    'sm__sass_inst_executed_op_tma.avg',
]

CSV_FIELDS = [
    'CTAsPerSM', 'NumStages', 'SmemWidth', 'SmemHeight', 'BytesInFlightPerSM',
    'DRAMBandwidthBps', 'SMCyclesAvg', 'TMAAvgOps', 'CyclesPerTMA',
]


def find_ncu():
    """Find ncu binary path."""
    result = subprocess.run(["which", "ncu"], capture_output=True, text=True)
    # result = "/usr/local/cuda-13.2/bin/ncu"
    if result.returncode == 0:
        return result.stdout.strip()
    result = "/usr/local/cuda-13.2/bin/ncu"
    for path in ["/usr/local/cuda-13.2/bin/ncu", "/opt/nvidia/nsight-compute/ncu"]:
        if os.path.exists(path):
            return path
    return None


def parse_ncu_csv(output):
    """Parse ncu --csv output and return dict of metric name -> value."""
    metrics = {}
    for line in output.split('\n'):
        if not line.startswith('"'):
            continue
        parts = line.split('","')
        if len(parts) < 3:
            continue
        metric_name = parts[-3].strip('"')
        metric_value = parts[-1].strip('"').replace(',', '')
        if metric_name in NCU_METRICS:
            try:
                metrics[metric_name] = float(metric_value)
            except ValueError:
                pass
    return metrics


def run_benchmark(ctas, stages, smem_w, smem_h, ncu_path, verbose=False):
    """Compile and profile benchmark for given configuration."""
    label = f"CTAs={ctas}, stages={stages}, w={smem_w}, h={smem_h}"

    clean_cmd = ["make", "clean"]
    build_cmd = [
        "make", "tma2d_tput.out",
        f"CTAS_PER_SM={ctas}",
        f"NUM_STAGES={stages}",
        f"SMEM_WIDTH={smem_w}",
        f"SMEM_HEIGHT={smem_h}",
    ]
    ncu_cmd = [
        "sudo", ncu_path,
        "--clock-control", "none",
        "--csv",
        "--metrics", ",".join(NCU_METRICS),
        "./tma2d_tput.out",
    ]

    try:
        subprocess.run(clean_cmd, capture_output=True, check=True)

        if verbose:
            print(f"Building {label}...", file=sys.stderr)
        result = subprocess.run(build_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Build failed for {label}:", file=sys.stderr)
            if verbose:
                print(result.stderr, file=sys.stderr)
            return None

        if verbose:
            print(f"Profiling {label}...", file=sys.stderr)
        result = subprocess.run(ncu_cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            print(f"ncu failed for {label}:", file=sys.stderr)
            if verbose:
                print(result.stderr, file=sys.stderr)
            return None

        metrics = parse_ncu_csv(result.stdout)
        if not metrics:
            print(f"Could not parse ncu output for {label}", file=sys.stderr)
            if verbose:
                print(f"stdout: {result.stdout[:500]}", file=sys.stderr)
            return None

        bif_per_sm = ctas * stages * smem_w * smem_h * 4  # float32
        return {
            'CTAsPerSM': ctas,
            'NumStages': stages,
            'SmemWidth': smem_w,
            'SmemHeight': smem_h,
            'BytesInFlightPerSM': bif_per_sm,
            'DRAMBandwidthBps': metrics.get('dram__bytes_read.sum.per_second', 0),
            'SMCyclesAvg': metrics.get('sm__cycles_elapsed.avg', 0),
            'TMAAvgOps': metrics.get('sm__sass_inst_executed_op_tma.avg', 0),
            'CyclesPerTMA': (metrics['sm__cycles_elapsed.avg']
                             / metrics.get('sm__sass_inst_executed_op_tma.avg', 0)
                             if metrics.get('sm__sass_inst_executed_op_tma.avg') else 0),
        }

    except subprocess.TimeoutExpired:
        print(f"Timeout for {label}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error for {label}: {e}", file=sys.stderr)
        return None


def main():
    parser = argparse.ArgumentParser(description='Benchmark TMA 2D unicast throughput via ncu')
    parser.add_argument('-o', '--output', default='tma2d_tput_results.csv',
                        help='Output CSV file (default: tma2d_tput_results.csv)')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite CSV instead of appending')
    parser.add_argument('--ctas', nargs='+', type=int, default=None,
                        help=f'CTAs per SM (default: {DEFAULT_CTAS_PER_SM})')
    parser.add_argument('--stages', nargs='+', type=int, default=None,
                        help=f'Pipeline stages (default: {DEFAULT_NUM_STAGES})')
    parser.add_argument('--heights', nargs='+', type=int, default=None,
                        help=f'SMEM heights (default: {DEFAULT_SMEM_HEIGHTS})')
    args = parser.parse_args()

    ncu_path = "/usr/local/cuda-13.2/bin/ncu"
    if not ncu_path:
        print("Error: ncu not found. Install NVIDIA Nsight Compute.", file=sys.stderr)
        return 1
    print(f"Using ncu: {ncu_path}", file=sys.stderr)

    ctas_list = args.ctas or DEFAULT_CTAS_PER_SM
    stages_list = args.stages or DEFAULT_NUM_STAGES
    heights_list = args.heights or DEFAULT_SMEM_HEIGHTS
    smem_w = DEFAULT_SMEM_WIDTH

    configs = [
        (ctas, stages, smem_w, h)
        for ctas in ctas_list
        for h in heights_list
        for stages in stages_list
    ]

    total_runs = len(configs)
    print(f"Running {total_runs} configurations...", file=sys.stderr)

    # Setup CSV
    file_exists = os.path.exists(args.output) and not args.overwrite
    csv_file = open(args.output, 'a' if file_exists else 'w', newline='')
    writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
    if not file_exists:
        writer.writeheader()
        csv_file.flush()

    result_count = 0
    for run_idx, (ctas, stages, w, h) in enumerate(configs, 1):
        result = run_benchmark(ctas, stages, w, h, ncu_path, verbose=args.verbose)
        if result:
            writer.writerow(result)
            csv_file.flush()
            result_count += 1
            bw_gbps = result['DRAMBandwidthBps'] / 1e9
            cycles_per_tma = result['CyclesPerTMA']
            bif_kib = result['BytesInFlightPerSM'] / 1024
            print(f"[{run_idx}/{total_runs}] CTAs={ctas}, stages={stages}, "
                  f"h={h:3d} ({bif_kib:6.1f} KiB BIF): "
                  f"{bw_gbps:.2f} GB/s, {cycles_per_tma:.1f} cyc/TMA",
                  file=sys.stderr)
        else:
            print(f"[{run_idx}/{total_runs}] FAILED: CTAs={ctas}, stages={stages}, h={h}",
                  file=sys.stderr)

    csv_file.close()

    if result_count > 0:
        print(f"\nSaved {result_count} results to {args.output}", file=sys.stderr)
    else:
        print("No successful results", file=sys.stderr)
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
