#!/usr/bin/env python3

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
from scipy.cluster.hierarchy import linkage, leaves_list, optimal_leaf_ordering, fcluster
from scipy.spatial.distance import squareform

CMAP20 = plt.colormaps["tab20"]


def cluster_within_gpc(member_idxs, dist_mat):
    if len(member_idxs) <= 2:
        return list(member_idxs)
    sub = dist_mat[np.ix_(member_idxs, member_idxs)]
    np.fill_diagonal(sub, 0)
    dvec = squareform(sub, checks=False)
    Z = linkage(dvec, method="average")
    Z_opt = optimal_leaf_ordering(Z, dvec)
    return [member_idxs[i] for i in leaves_list(Z_opt)]


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "results/distance.csv"

    df = pd.read_csv(csv_path)
    sms = sorted(df["sm_a"].unique())
    n = len(sms)
    sm_to_idx = {s: i for i, s in enumerate(sms)}

    dist = np.zeros((n, n))
    sm_gpc = {}
    for _, row in df.iterrows():
        i = sm_to_idx[row["sm_a"]]
        j = sm_to_idx[row["sm_b"]]
        dist[i, j] = row["mean_abs_diff"]
        sm_gpc[row["sm_a"]] = row["gpc_a"]
        sm_gpc[row["sm_b"]] = row["gpc_b"]

    gpcs = defaultdict(list)
    for sm in sms:
        gpcs[sm_gpc[sm]].append(sms.index(sm))

    gpc_ids = sorted(gpcs.keys())
    gpc_dist = np.zeros((len(gpc_ids), len(gpc_ids)))
    for i, gi in enumerate(gpc_ids):
        for j, gj in enumerate(gpc_ids):
            vals = [dist[a, b] for a in gpcs[gi] for b in gpcs[gj] if a != b]
            gpc_dist[i, j] = np.mean(vals) if vals else 0

    gpc_dvec = squareform(gpc_dist + gpc_dist.T, checks=False) / 2
    Z_gpc = linkage(gpc_dvec, method="ward")
    gpc_clusters = fcluster(Z_gpc, t=2, criterion="maxclust")
    die_half = {gpc_ids[i]: "A" if gpc_clusters[i] == 1 else "B"
                for i in range(len(gpc_ids))}

    singletons = {g: m for g, m in gpcs.items() if len(m) <= 2}
    full_gpcs = {g: m for g, m in gpcs.items() if len(m) > 2}

    die_a_full = sorted(g for g in full_gpcs if die_half[g] == "A")
    die_a_sing = sorted(g for g in singletons if die_half[g] == "A")
    die_b_full = sorted(g for g in full_gpcs if die_half[g] == "B")
    die_b_sing = sorted(g for g in singletons if die_half[g] == "B")

    sm_order_a, sm_order_b = [], []
    gpc_bounds = []
    pos = 0
    for g in die_a_full + die_a_sing:
        ordered = cluster_within_gpc(gpcs[g], dist)
        sm_order_a.extend(ordered)
        gpc_bounds.append((pos, pos + len(ordered), g, "A"))
        pos += len(ordered)
    n_a = len(sm_order_a)
    for g in die_b_full + die_b_sing:
        ordered = cluster_within_gpc(gpcs[g], dist)
        sm_order_b.extend(ordered)
        gpc_bounds.append((pos, pos + len(ordered), g, "B"))
        pos += len(ordered)
    sm_order = sm_order_a + sm_order_b

    dist_reord = dist[np.ix_(sm_order, sm_order)]

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    im = ax.imshow(dist_reord, cmap="viridis", interpolation="nearest")
    ax.set_title("SM-SM L2 Latency Difference", fontsize=13)

    ax.axhline(y=n_a - 0.5, color="white", linewidth=2.5)
    ax.axvline(x=n_a - 0.5, color="white", linewidth=2.5)
    ax.text(n_a / 2, 3, "Die A", ha="center", va="top",
            fontsize=12, fontweight="bold", color="white")
    ax.text(n_a + (n - n_a) / 2, n_a + 3, "Die B", ha="center", va="top",
            fontsize=12, fontweight="bold", color="white")

    label_color = "#aaaaaa"
    for start, end, gpc_id, die in gpc_bounds:
        sz = end - start
        rect = mpatches.Rectangle((start-0.5, start-0.5), sz, sz,
                                   linewidth=2, edgecolor=label_color, facecolor="none")
        ax.add_patch(rect)
        mid = (start + end) / 2
        is_sing = len(gpcs[gpc_id]) <= 2
        label = f"GPC{int(gpc_id)}*" if is_sing else f"GPC{int(gpc_id)}"
        ax.text(-1.5, mid, label, fontsize=7, ha="right", va="center",
                color=label_color, fontweight="bold")

    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, shrink=0.8, label="mean |diff| per address (cycles)")

    fig.tight_layout()
    out = csv_path.replace(".csv", "") + "_sm_distance.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.close()

    mask = ~np.eye(n, dtype=bool)
    print(f"SM-SM distance: mean={dist[mask].mean():.1f}  "
          f"min={dist[mask].min():.1f}  max={dist[mask].max():.1f}")

    intra = [dist[a,b] for g in gpcs.values() for a in g for b in g if a != b]
    inter = [dist[a,b] for gi in gpcs.values() for gj in gpcs.values()
             if gi is not gj for a in gi for b in gj]
    print(f"Intra-GPC: {np.mean(intra):.1f}  Inter-GPC: {np.mean(inter):.1f}")

    same_die = [dist[a,b] for gi_id, gi in gpcs.items() for gj_id, gj in gpcs.items()
                if die_half[gi_id] == die_half[gj_id] and gi is not gj
                for a in gi for b in gj]
    cross_die = [dist[a,b] for gi_id, gi in gpcs.items() for gj_id, gj in gpcs.items()
                 if die_half[gi_id] != die_half[gj_id]
                 for a in gi for b in gj]
    if same_die and cross_die:
        print(f"Same-die inter-GPC: {np.mean(same_die):.1f}  "
              f"Cross-die: {np.mean(cross_die):.1f}")

    print()
    for die_label in ["A", "B"]:
        die_gpcs = sorted(g for g in gpcs if die_half[g] == die_label)
        full = [g for g in die_gpcs if len(gpcs[g]) > 2]
        sing = [g for g in die_gpcs if len(gpcs[g]) <= 2]
        total_sms = sum(len(gpcs[g]) for g in die_gpcs)
        tpc_counts = [f"{len(gpcs[g])//2}" for g in die_gpcs]
        gpc_strs = [f"GPC{int(g)}" for g in full] + [f"GPC{int(g)}*" for g in sing]
        print(f"Die {die_label}: [{', '.join(tpc_counts)}] TPCs  ({', '.join(gpc_strs)}, {total_sms} SMs)")


if __name__ == "__main__":
    main()
