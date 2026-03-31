#!/usr/bin/env python3
"""Plot throughput vs pipeline depth, grouped by bit-width.

Each line = one N value, showing the max throughput across all formats
in the bit-width group (and all M values) at each pipeline depth.
1SM and 2SM are plotted side by side.

Usage:
    python plot_depth_by_bitwidth.py --bits 4
    python plot_depth_by_bitwidth.py --bits 8
    python plot_depth_by_bitwidth.py --bits 16
"""

import argparse
import csv
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

parser = argparse.ArgumentParser(description="Plot throughput vs pipeline depth by bit-width")
parser.add_argument("--bits", type=int, required=True, choices=[4, 8, 16], help="Bit width: 4, 8, or 16")
args = parser.parse_args()

BIT_GROUPS = {
    16: ["BF16"],
    8: ["E4M3", "S8", "MXF8"],
    4: ["F4", "MXF4"],
}

FORMAT_K = {
    "BF16": 16, "E4M3": 32, "S8": 32, "F4": 64, "MXF8": 32, "MXF4": 64,
}

def hw_peak(fmt, cta_group):
    k = FORMAT_K[fmt]
    return 512 * k if cta_group == 1 else 1024 * k

target_fmts = set(BIT_GROUPS[args.bits])

with open("tput_ts_small_depth.csv") as f:
    rows = list(csv.DictReader(f))

# Build best throughput per (cta, N, depth)
best = defaultdict(float)
for row in rows:
    fmt = row["Format"]
    if fmt not in target_fmts:
        continue
    cta = int(row["CTAGroup"])
    n = int(row["N"])
    depth = int(row["PipelineDepth"])
    flops = float(row["FLOPsPerCycle"])
    key = (cta, n, depth)
    if flops > best[key]:
        best[key] = flops

# Build series per (cta, N) -> list of (depth, flops)
series = defaultdict(list)
for (cta, n, depth), flops in best.items():
    series[(cta, n)].append((depth, flops))
for key in series:
    series[key].sort()

# Get all N values across both CTA groups
all_n = sorted(set(n for (_, n) in series.keys()))

cmap = cm.viridis
n_colors = {n: cmap(i / max(len(all_n) - 1, 1)) for i, n in enumerate(all_n)}
if len(all_n) > 1:
    n_colors[all_n[-1]] = "#B8A000"

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

for ax, cta in zip(axes, [1, 2]):
    cta_label = "1SM" if cta == 1 else "2SM"

    peak = hw_peak(BIT_GROUPS[args.bits][0], cta)
    ax.axhline(y=100, color="#888888", linewidth=1.5, linestyle=":", alpha=0.7)
    ax.text(1, 100, f"{args.bits}-bit peak: {peak} FLOP/cyc",
            va="bottom", ha="left", fontsize=9, color="#555555")

    for n in all_n:
        if (cta, n) not in series:
            continue
        data = series[(cta, n)]
        xs = [d for d, _ in data]
        ys = [f / peak * 100 for _, f in data]
        ax.plot(xs, ys, marker="o", color=n_colors[n],
                markersize=6, linewidth=1.8, alpha=0.85, label=f"N={n}")
        for (d, f), pct in zip(data, ys):
            if d == 4:
                ax.annotate(f"{pct:.1f}%", (d, pct), textcoords="offset points",
                            xytext=(-4, 4), fontsize=7.5, color=n_colors[n], ha="right")

    ax.set_xlabel("Num In-Flight MMAs", fontsize=12)
    ax.set_ylabel("SoL Throughput (%)", fontsize=12)
    ax.set_xticks(range(1, 11))
    ax.set_ylim(0, 105)
    ax.set_yticks(range(0, 110, 10))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10)
    ax.set_title(f"{cta_label} MMA", fontsize=13)

fig.suptitle(f"Throughput vs. In-Flight Instructions: {args.bits}-Bit Formats", fontsize=14, y=1.01)
fig.tight_layout()
outfile = f"depth_scaling_{args.bits}bit.png"
fig.savefig(outfile, dpi=150, bbox_inches="tight")
print(f"Saved {outfile}")
