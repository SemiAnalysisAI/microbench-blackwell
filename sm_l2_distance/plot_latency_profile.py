#!/usr/bin/env python3

import sys
import pandas as pd
import matplotlib.pyplot as plt

LINE_SIZE = 128


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "results/latency_profile.csv"
    df = pd.read_csv(csv_path)

    sm = int(df["sm"].iloc[0])
    gpc = int(df["gpc"].iloc[0])

    df["cumul_mb"] = (df["rank"] + 1) * LINE_SIZE / (1024 * 1024)
    df = df[df["latency"] > 0]

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.plot(df["cumul_mb"], df["latency"], linewidth=1.0, color="#1f77b4")

    ax.set_xlim(0, 120)
    ax.set_ylim(0, df[df["cumul_mb"] <= 120]["latency"].max() * 1.08)
    ax.set_xlabel("Cumulative Data Size (MB)", fontsize=11)
    ax.set_ylabel("Latency (cycles)", fontsize=11)
    ax.set_title("L2 Latency Profile — Pointer Chase", fontsize=13)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = csv_path.replace(".csv", "") + ".png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.close()


if __name__ == "__main__":
    main()
