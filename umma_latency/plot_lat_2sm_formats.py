#!/usr/bin/env python3
"""Plot 2SM latency comparison across data types.

Bar chart grouped by N, one bar per format.
Two vertically stacked panels: M=128 (top), M=256 (bottom).
"""

import csv
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Read latency results
with open("lat_results_full.csv") as f:
    rows = list(csv.DictReader(f))

# For each (Format, CTAGroup, M, N), take best ABLayout (lowest latency)
data = {}
for row in rows:
    fmt = row["Format"]
    cta = int(row["CTAGroup"])
    m, n = int(row["M"]), int(row["N"])
    cycles = float(row["MedianCycles"])
    key = (fmt, cta, m, n)
    if key not in data or cycles < data[key]:
        data[key] = cycles

formats = ["BF16", "E4M3", "S8", "F4", "MXF8", "MXF4"]
fmt_colors = {
    "BF16": "#2D6A8F", "E4M3": "#CC5555", "S8": "#44AA66",
    "F4": "#DD8833", "MXF8": "#8855BB", "MXF4": "#CC6699",
}

n_formats = len(formats)
bar_width = 0.12

fig, axes = plt.subplots(2, 1, figsize=(14, 9))

panels = [
    ("M=128", 128),
    ("M=256", 256),
]

for ax, (title, m_val) in zip(axes, panels):
    all_ns = sorted(set(
        n for (f, c, m, n) in data
        if c == 2 and m == m_val
    ))

    group_positions = np.arange(len(all_ns))
    offsets = np.linspace(-(n_formats - 1) / 2 * bar_width,
                           (n_formats - 1) / 2 * bar_width, n_formats)

    for i, fmt in enumerate(formats):
        xs, ys = [], []
        for j, n in enumerate(all_ns):
            key = (fmt, 2, m_val, n)
            if key in data:
                xs.append(group_positions[j] + offsets[i])
                ys.append(data[key])

        ax.bar(xs, ys, width=bar_width,
               color=fmt_colors[fmt], edgecolor="white", linewidth=0.3)

        # Value labels on top of bars
        for x, y in zip(xs, ys):
            ax.text(x, y + 2, f"{y:.0f}", ha="center", va="bottom",
                    fontsize=6.5, fontweight="bold", color=fmt_colors[fmt])

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("N", fontsize=11)
    ax.set_ylabel("Latency (cycles)", fontsize=11)
    ax.set_xticks(group_positions)
    ax.set_xticklabels([str(n) for n in all_ns], fontsize=10)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.3)

# Legend
handles = [mpatches.Patch(facecolor=fmt_colors[fmt], label=fmt) for fmt in formats]
for ax in axes:
    ax.legend(handles=handles, loc="upper left", fontsize=9)

fig.suptitle("2SM MMA Latency by Format and N", fontsize=13, y=0.98)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.subplots_adjust(hspace=0.3)
fig.savefig("lat_2sm_formats.png", dpi=150, bbox_inches="tight")
print("Saved lat_2sm_formats.png")
plt.show()
