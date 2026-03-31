#!/usr/bin/env python3
"""Plot latency (cycles) vs N, faceted by CTAGroup.

Each line = (Format, M). X-axis = N.
Two panels: 1SM and 2SM.
Color = format, linestyle = M.
"""

import csv
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

# Read latency results
with open("lat_results_full.csv") as f:
    rows = list(csv.DictReader(f))

# For each (Format, CTAGroup, M, N), take best ABLayout (lowest latency)
best = {}
for row in rows:
    fmt = row["Format"]
    cta = int(row["CTAGroup"])
    m, n = int(row["M"]), int(row["N"])
    cycles = float(row["MedianCycles"])
    key = (fmt, cta, m, n)
    if key not in best or cycles < best[key]:
        best[key] = cycles

# Build series per (Format, CTAGroup, M) -> list of (N, cycles)
formats = ["BF16", "E4M3", "S8", "F4", "MXF8", "MXF4"]
series = defaultdict(list)
for (fmt, cta, m, n), cycles in best.items():
    series[(fmt, cta, m)].append((n, cycles))

for key in series:
    series[key].sort()

# Format colors
fmt_colors = {
    "BF16": "#2D6A8F", "E4M3": "#CC5555", "S8": "#44AA66",
    "F4": "#DD8833", "MXF8": "#8855BB", "MXF4": "#CC6699",
}
fmt_markers = {
    "BF16": "o", "E4M3": "s", "S8": "D", "F4": "^", "MXF8": "v", "MXF4": "P",
}
# M line styles: assigned per-panel by rank (smaller M = dashed, larger M = solid)
m_rank_linestyles = ["--", "-"]

fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)

for ax, cta in zip(axes, [1, 2]):
    cta_label = "1SM" if cta == 1 else "2SM"
    ax.set_title(f"{cta_label} (CTA_GROUP={cta})", fontsize=12)

    # Collect all N values for this panel
    all_n = sorted(set(n for (fmt, c, m) in series if c == cta
                       for n, _ in series[(fmt, c, m)]))
    n_to_x = {n: i for i, n in enumerate(all_n)}
    x_pos = np.arange(len(all_n))

    # Get sorted M values for this panel to assign linestyles by rank
    panel_m_values = sorted(set(m for (f, c, m) in series if c == cta))
    m_to_ls = {m_val: m_rank_linestyles[i] for i, m_val in enumerate(panel_m_values)}

    for fmt in formats:
        for m_val in sorted(set(m for (f, c, m) in series if f == fmt and c == cta)):
            if (fmt, cta, m_val) not in series:
                continue
            data = series[(fmt, cta, m_val)]
            xs = [n_to_x[n] for n, _ in data]
            ys = [cycles for _, cycles in data]
            ax.plot(xs, ys, marker=fmt_markers[fmt], linestyle=m_to_ls[m_val],
                    color=fmt_colors[fmt], markersize=7, linewidth=1.8, alpha=0.85)

    ax.set_xlabel("N", fontsize=11)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(n) for n in all_n], fontsize=10)
    ax.grid(alpha=0.3)

axes[0].set_ylabel("Latency (cycles)", fontsize=11)
axes[0].set_ylim(0, 300)

# Per-panel legend
for ax, cta in zip(axes, [1, 2]):
    panel_handles = []
    for fmt in formats:
        h = mlines.Line2D([], [], color=fmt_colors[fmt], marker=fmt_markers[fmt],
                           linestyle="-", markersize=5, linewidth=1.5, label=fmt)
        panel_handles.append(h)

    # Spacer
    panel_handles.append(mlines.Line2D([], [], color="none", label=""))

    # Only M values that appear in this panel
    legend_m_values = sorted(set(m for (f, c, m) in series if c == cta))
    legend_m_ls = {m_val: m_rank_linestyles[i] for i, m_val in enumerate(legend_m_values)}
    for m_val in legend_m_values:
        h = mlines.Line2D([], [], color="#666666", linestyle=legend_m_ls[m_val],
                           linewidth=1.8, label=f"M={m_val}")
        panel_handles.append(h)

    ax.legend(handles=panel_handles, loc="upper left", fontsize=10)

fig.suptitle("MMA Latency vs N (by Format and M)", fontsize=13)

fig.tight_layout()
fig.savefig("lat_comparison.png", dpi=150, bbox_inches="tight")
print("Saved lat_comparison.png")
plt.show()
