#!/usr/bin/env python3
"""
Benchmark LDGSTS load latency across configurations.
Each thread issues exactly 1 LDGSTS and waits serially — no pipeline stages.
The kernel uses clock64() to measure per-thread round-trip DRAM latency,
then reports min/median/max cycles across all threads via stdout.
Each launch reads a distinct DRAM region, and an explicit L2 flush is
performed between iterations to ensure cold-cache (DRAM) latency.
"""

import subprocess
import csv
import sys
import os
import argparse

# Load type -> byte size
LOAD_TYPES = {'float': 4, 'float2': 8, 'float4': 16}

# Default sweep: union of MLA + MHA-range configs
DEFAULT_CTAS    = [1, 2, 3, 4]
DEFAULT_THREADS = [32, 64, 128, 256]
DEFAULT_LOADS   = ['float', 'float2', 'float4']

# MLA preset: 1 CTA/SM, 2 load warps, float4
MLA_CTAS    = [1]
MLA_THREADS = [32, 64, 128]
MLA_LOADS   = ['float4']

# MHA-range preset: full sweep
MHA_CTAS    = [1, 2, 3, 4]
MHA_THREADS = [64, 128, 256]
MHA_LOADS   = ['float', 'float2', 'float4']

CSV_FIELDS = [
    'CTAsPerSM', 'ThreadsPerBlock', 'LoadType', 'LoadBytes',
    'LatencyCyclesMedian', 'LatencyCyclesMin', 'LatencyCyclesMax',
    'LatencyNsMedian',
]

B200_CLOCK_GHZ = 1.575  # B200 SM clock ~1.575 GHz
NUM_SMS = 20


def run_benchmark(ctas, threads, load_t, verbose=False):
    """Compile and run benchmark for given configuration."""
    load_bytes = LOAD_TYPES[load_t]
    label = f"CTAs={ctas}, threads={threads:3d}, load={load_t}"

    clean_cmd = ["make", "clean"]
    build_cmd = [
        "make", "ldgsts_lat.out",
        f"CTAS_PER_SM={ctas}",
        f"THREADS_PER_BLOCK={threads}",
        f"LOAD_T={load_t}",
    ]
    run_cmd = ["./ldgsts_lat.out"]

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
            print(f"Running {label}...", file=sys.stderr)
        result = subprocess.run(run_cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            print(f"Run failed for {label}:", file=sys.stderr)
            if verbose:
                print(result.stderr, file=sys.stderr)
            return None

        # Parse SUMMARY line from stdout: SUMMARY,<median>,<min>,<max>
        for line in result.stdout.strip().split('\n'):
            if line.startswith('SUMMARY,'):
                parts = line.split(',')
                lat_cyc_med = float(parts[1])
                lat_cyc_min = float(parts[2])
                lat_cyc_max = float(parts[3])
                break
        else:
            print(f"No SUMMARY line in output for {label}", file=sys.stderr)
            if verbose:
                print(f"stdout: {result.stdout[:500]}", file=sys.stderr)
                print(f"stderr: {result.stderr[:500]}", file=sys.stderr)
            return None

        lat_ns_med = lat_cyc_med / B200_CLOCK_GHZ

        return {
            'CTAsPerSM':           ctas,
            'ThreadsPerBlock':     threads,
            'LoadType':            load_t,
            'LoadBytes':           load_bytes,
            'LatencyCyclesMedian': lat_cyc_med,
            'LatencyCyclesMin':    lat_cyc_min,
            'LatencyCyclesMax':    lat_cyc_max,
            'LatencyNsMedian':     lat_ns_med,
        }

    except subprocess.TimeoutExpired:
        print(f"Timeout for {label}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error for {label}: {e}", file=sys.stderr)
        return None


def main():
    parser = argparse.ArgumentParser(description='Benchmark LDGSTS load latency')
    parser.add_argument('-o', '--output', default='ldgsts_lat_results.csv',
                        help='Output CSV file (default: ldgsts_lat_results.csv)')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite CSV instead of appending')
    parser.add_argument('--preset', choices=['mha', 'mla'], default=None,
                        help='Use preset config: mha (4 CTA variants) or mla (1 CTA/SM, float4)')
    parser.add_argument('--ctas', nargs='+', type=int, default=None,
                        help=f'CTAs per SM to sweep (default: {DEFAULT_CTAS})')
    parser.add_argument('--threads', nargs='+', type=int, default=None,
                        help=f'Threads per block to sweep (default: {DEFAULT_THREADS})')
    parser.add_argument('--load-types', nargs='+', default=None,
                        choices=list(LOAD_TYPES.keys()),
                        help=f'Load types to sweep (default: {DEFAULT_LOADS})')
    args = parser.parse_args()

    # Resolve sweep params: CLI > preset > defaults
    if args.preset == 'mla':
        base_ctas, base_threads, base_loads = MLA_CTAS, MLA_THREADS, MLA_LOADS
    elif args.preset == 'mha':
        base_ctas, base_threads, base_loads = MHA_CTAS, MHA_THREADS, MHA_LOADS
    else:
        base_ctas, base_threads, base_loads = DEFAULT_CTAS, DEFAULT_THREADS, DEFAULT_LOADS

    ctas_list    = args.ctas or base_ctas
    threads_list = args.threads or base_threads
    loads_list   = args.load_types or base_loads

    configs = [
        (ctas, threads, load_t)
        for ctas    in ctas_list
        for threads in threads_list
        for load_t  in loads_list
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
    for run_idx, (ctas, threads, load_t) in enumerate(configs, 1):
        result = run_benchmark(ctas, threads, load_t, verbose=args.verbose)
        if result:
            writer.writerow(result)
            csv_file.flush()
            result_count += 1
            lat_cyc  = result['LatencyCyclesMedian']
            lat_ns   = result['LatencyNsMedian']
            print(
                f"[{run_idx}/{total_runs}] CTAs={ctas}, threads={threads:3d}, load={load_t}: "
                f"{lat_cyc:.1f} cyc  {lat_ns:.1f} ns",
                file=sys.stderr,
            )
        else:
            print(f"[{run_idx}/{total_runs}] FAILED: CTAs={ctas}, threads={threads}, load={load_t}",
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
