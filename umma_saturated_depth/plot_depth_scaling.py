#!/usr/bin/env python3
"""Plot throughput vs pipeline depth for different data formats.

Usage:
    python plot_depth_scaling.py                        # defaults: M=64, N=64, CTA=1
    python plot_depth_scaling.py --m 128 --n 256 --cta 2
"""

import argparse
import csv
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

parser = argparse.ArgumentParser(description="Plot throughput vs pipeline depth")
parser.add_argument("--m", type=int, default=64, help="M dimension (default: 64)")
parser.add_argument("--n", type=int, default=64, help="N dimension (default: 64)")
parser.add_argument("--cta", type=int, default=1, help="CTA group: 1 or 2 (default: 1)")
args = parser.parse_args()

# Hardware theoretical peak: 512*K (1SM), 1024*K (2SM)
FORMAT_K = {
    "BF16": 16, "E4M3": 32, "S8": 32, "F4": 64, "MXF8": 32, "MXF4": 64,
}

def hw_peak(fmt, cta_group):
    k = FORMAT_K[fmt]
    return 512 * k if cta_group == 1 else 1024 * k

# Group formats by bit-width for theoretical max lines
BIT_GROUPS = {
    "16-bit": ["BF16"],
    "8-bit": ["E4M3", "S8", "MXF8"],
    "4-bit": ["F4", "MXF4"],
}

# Read data
with open("tput_ts_small_depth.csv") as f:
    rows = list(csv.DictReader(f))

# For each (Format, PipelineDepth), take best K value
best = {}
for row in rows:
    fmt = row["Format"]
    cta = int(row["CTAGroup"])
    m, n = int(row["M"]), int(row["N"])
    depth = int(row["PipelineDepth"])
    flops = float(row["FLOPsPerCycle"])
    if cta != args.cta or m != args.m or n != args.n:
        continue
    key = (fmt, depth)
    if key not in best or flops > best[key]:
        best[key] = flops

# Build series per Format -> list of (depth, flops)
formats = ["BF16", "E4M3", "S8", "F4", "MXF8", "MXF4"]
series = defaultdict(list)
for (fmt, depth), flops in best.items():
    series[fmt].append((depth, flops))

for key in series:
    series[key].sort()

fmt_colors = {
    "BF16": "#2D6A8F", "E4M3": "#CC5555", "S8": "#44AA66",
    "F4": "#DD8833", "MXF8": "#8855BB", "MXF4": "#CC6699",
}
fmt_markers = {
    "BF16": "o", "E4M3": "s", "S8": "D", "F4": "^", "MXF8": "v", "MXF4": "P",
}

# Jitter offsets to separate overlapping lines
jitter = {
    "BF16": -0.12, "E4M3": -0.08, "S8": -0.04,
    "F4": 0.04, "MXF8": 0.08, "MXF4": 0.12,
}

fig, ax = plt.subplots(figsize=(10, 6))

# Theoretical max horizontal lines
hline_colors = {"16-bit": "#2D6A8F", "8-bit": "#CC5555", "4-bit": "#DD8833"}
for bit_label, fmts in BIT_GROUPS.items():
    # All formats in a bit group have the same peak; pick the first present
    peak = hw_peak(fmts[0], args.cta)
    # Only draw if at least one format in this group has data
    if any(f in series for f in fmts):
        ax.axhline(y=peak, color=hline_colors[bit_label], linewidth=1.2,
                   linestyle=":", alpha=0.6)
        ax.text(10.15, peak, f"{bit_label} peak: {peak} FLOP/cyc",
                va="center", fontsize=8.5, color=hline_colors[bit_label])

for fmt in formats:
    if fmt not in series:
        continue
    data = series[fmt]
    xs = [d + jitter[fmt] for d, _ in data]
    ys = [f for _, f in data]
    ax.plot(xs, ys, marker=fmt_markers[fmt], color=fmt_colors[fmt],
            markersize=8, linewidth=2, alpha=0.85, label=fmt)

ax.set_xlabel("Num In-Flight MMAs", fontsize=12)
ax.set_ylabel("Throughput (FLOPs/cycle)", fontsize=12)
ax.set_xticks(range(1, 11))
ax.set_ylim(bottom=0)
ax.grid(alpha=0.3)
ax.legend(fontsize=10)

cta_label = "1SM" if args.cta == 1 else "2SM"
ax.set_title(f"Throughput vs Pipeline Depth — {cta_label}, M={args.m}, N={args.n}", fontsize=13)

fig.tight_layout()
outfile = f"depth_scaling_cta{args.cta}_m{args.m}_n{args.n}.png"
fig.savefig(outfile, dpi=150, bbox_inches="tight")
print(f"Saved {outfile}")
plt.show()
