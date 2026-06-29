#!/usr/bin/env python3
"""
plot.py -- Read benchmark CSVs and generate publication-quality charts.

Usage:
  python scripts/plot.py results/20260629_190000/
  python scripts/plot.py results/20260629_190000/ --no-show

Outputs PNG files alongside the CSV files in the results folder.
"""

import csv
import sys
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------
# Colorblind-friendly palette (Wong 2011, Nature Methods)
CPU_COLOR       = "#333333"
NAIVE_COLOR     = "#E69F00"
EFFICIENT_COLOR = "#0072B2"
THRUST_COLOR    = "#009E73"
GPU_RADIX_COLOR = "#CC79A7"

LINE_WIDTH    = 1.8
MARKER_SIZE   = 6
FONT_SIZE     = 11
TITLE_SIZE    = 13
DPI           = 200

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": FONT_SIZE,
    "axes.titlesize": TITLE_SIZE,
    "axes.labelsize": FONT_SIZE,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": FONT_SIZE - 1,
    "ytick.labelsize": FONT_SIZE - 1,
    "legend.fontsize": FONT_SIZE - 1,
    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "#cccccc",
    "grid.alpha": 0.3,
    "grid.color": "#aaaaaa",
})


def read_csv(path):
    """Read a CSV file and return (headers, list_of_rows)."""
    with open(path, newline="") as f:
        reader = csv.reader(f)
        headers = next(reader)
        rows = []
        for row in reader:
            rows.append([float(v) for v in row])
    return headers, rows


def plot_scan(csv_path, out_dir):
    """Chart 1: Prefix-sum (Scan) -- CPU / Naive / Efficient / Thrust."""
    _, rows = read_csv(csv_path)
    sizes   = np.array([r[0] for r in rows], dtype=int)
    cpu     = np.array([r[1] for r in rows])
    naive   = np.array([r[2] for r in rows])
    eff     = np.array([r[3] for r in rows])
    thrust  = np.array([r[4] for r in rows])

    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.loglog(sizes, cpu,    "s-", color=CPU_COLOR,       linewidth=LINE_WIDTH,
              markersize=MARKER_SIZE, label="CPU (serial)")
    ax.loglog(sizes, naive,  "o-", color=NAIVE_COLOR,     linewidth=LINE_WIDTH,
              markersize=MARKER_SIZE, label="Naive GPU")
    ax.loglog(sizes, eff,    "D-", color=EFFICIENT_COLOR, linewidth=LINE_WIDTH,
              markersize=MARKER_SIZE, label="Work-Efficient GPU")
    ax.loglog(sizes, thrust, "^-", color=THRUST_COLOR,    linewidth=LINE_WIDTH,
              markersize=MARKER_SIZE, label="Thrust (library)")

    ax.set_xlabel("Array Size")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Prefix Sum (Scan) Performance")
    ax.legend(loc="upper left")

    # Power-of-2 tick labels
    ax.set_xticks(sizes)
    ax.set_xticklabels([f"$2^{{{int(np.log2(s))}}}$" for s in sizes],
                       rotation=35, ha="right", fontsize=FONT_SIZE - 3)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())

    ax.grid(True, which="major", linestyle="-", linewidth=0.4)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.2)

    fig.tight_layout()
    out_path = os.path.join(out_dir, "chart_scan_comparison.png")
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    print(f"  -> {out_path}")


def plot_radix(csv_path, out_dir):
    """Chart 2: Radix Sort -- CPU std::sort vs GPU Radix Sort."""
    _, rows = read_csv(csv_path)
    sizes   = np.array([r[0] for r in rows], dtype=int)
    cpu     = np.array([r[1] for r in rows])
    gpu     = np.array([r[2] for r in rows])

    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.loglog(sizes, cpu, "s-",  color=CPU_COLOR,       linewidth=LINE_WIDTH,
              markersize=MARKER_SIZE, label="CPU std::sort")
    ax.loglog(sizes, gpu, "D--", color=GPU_RADIX_COLOR, linewidth=LINE_WIDTH,
              markersize=MARKER_SIZE, label="GPU Radix Sort")

    ax.set_xlabel("Array Size")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Radix Sort Performance")
    ax.legend(loc="upper left")

    ax.set_xticks(sizes)
    ax.set_xticklabels([f"$2^{{{int(np.log2(s))}}}$" for s in sizes],
                       rotation=35, ha="right", fontsize=FONT_SIZE - 3)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())

    # Annotate crossover region
    for i, (c, g) in enumerate(zip(cpu, gpu)):
        if g < c:
            ax.annotate(
                f"GPU faster\nfrom $2^{{{int(np.log2(sizes[i]))}}}$",
                xy=(sizes[i], gpu[i]),
                xytext=(sizes[i] * 4, gpu[i] * 2),
                arrowprops=dict(arrowstyle="->", color="#555555", lw=0.8),
                fontsize=FONT_SIZE - 2, color="#555555",
            )
            break

    ax.grid(True, which="major", linestyle="-", linewidth=0.4)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.2)

    fig.tight_layout()
    out_path = os.path.join(out_dir, "chart_radix_comparison.png")
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    print(f"  -> {out_path}")


def plot_radix_speedup(csv_path, out_dir):
    """Chart 3: Radix Sort speedup ratio (cpu_time / gpu_time)."""
    _, rows = read_csv(csv_path)
    sizes   = np.array([r[0] for r in rows], dtype=int)
    cpu     = np.array([r[1] for r in rows])
    gpu     = np.array([r[2] for r in rows])
    speedup = cpu / gpu

    fig, ax = plt.subplots(figsize=(9, 4.5))

    ax.semilogx(sizes, speedup, "D-", color=GPU_RADIX_COLOR,
                linewidth=LINE_WIDTH, markersize=MARKER_SIZE + 1)

    # Reference line at 1.0 (break-even)
    ax.axhline(y=1.0, color="#999999", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.text(sizes[-1] * 0.5, 1.15, "break-even", color="#999999",
            fontsize=FONT_SIZE - 2, ha="right")

    ax.set_xlabel("Array Size")
    ax.set_ylabel("Speedup  (CPU / GPU)")
    ax.set_title("Radix Sort GPU Speedup")

    ax.set_xticks(sizes)
    ax.set_xticklabels([f"$2^{{{int(np.log2(s))}}}$" for s in sizes],
                       rotation=35, ha="right", fontsize=FONT_SIZE - 3)

    # Label the max speedup
    idx_max = np.argmax(speedup)
    ax.annotate(
        f"{speedup[idx_max]:.1f}x",
        xy=(sizes[idx_max], speedup[idx_max]),
        xytext=(sizes[idx_max] * 2.5, speedup[idx_max] + 0.3),
        arrowprops=dict(arrowstyle="->", color="#555555", lw=0.8),
        fontsize=FONT_SIZE - 1, fontweight="bold", color=GPU_RADIX_COLOR,
    )

    ax.grid(True, which="major", linestyle="-", linewidth=0.4)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.2)

    fig.tight_layout()
    out_path = os.path.join(out_dir, "chart_radix_speedup.png")
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    print(f"  -> {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/plot.py <results_folder>")
        print("Example: python scripts/plot.py results/20260629_190000/")
        sys.exit(1)

    data_dir = Path(sys.argv[1])
    if not data_dir.is_dir():
        print(f"ERROR: not a directory: {data_dir}")
        sys.exit(1)

    scan_csv  = data_dir / "scan_comparison.csv"
    radix_csv = data_dir / "radix_comparison.csv"

    print(f"Plotting from: {data_dir}")

    if scan_csv.exists():
        plot_scan(scan_csv, data_dir)
    else:
        print(f"  (no scan CSV found)")

    if radix_csv.exists():
        plot_radix(radix_csv, data_dir)
        plot_radix_speedup(radix_csv, data_dir)
    else:
        print(f"  (no radix CSV found)")

    print("Done.")


if __name__ == "__main__":
    main()
