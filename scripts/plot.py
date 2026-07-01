#!/usr/bin/env python3
"""
plot.py -- Read benchmark CSVs and generate intuitive performance charts.

Usage:
  python scripts/plot.py results/20260629_190000/

Outputs PNG files alongside the CSV files in the results folder.

Charts generated:
  chart_scan_comparison.png       -- Log-log: all scan methods
  chart_compact_comparison.png    -- Log-log: stream compaction methods
  chart_radix_comparison.png      -- Log-log: CPU vs GPU radix sort
  chart_scan_speedup.png          -- Scan: GPU speedup vs CPU serial
  chart_compact_speedup.png       -- Compaction: GPU speedup vs CPU
  chart_radix_speedup.png         -- Radix: GPU speedup vs CPU
  chart_overview.png              -- Combined: all GPU methods speedup summary
"""

import csv
import sys
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ---------------------------------------------------------------------------
# Style constants — colorblind-friendly (Wong 2011, Nature Methods)
# ---------------------------------------------------------------------------
CPU_COLOR           = "#333333"
NAIVE_COLOR         = "#E69F00"
EFFICIENT_COLOR     = "#0072B2"
THRUST_COLOR        = "#009E73"
COMPACT_CPU_NOSCAN  = "#8B4513"
COMPACT_CPU_SCAN    = "#D55E00"
COMPACT_GPU_COLOR   = "#56B4E9"
GPU_RADIX_COLOR     = "#CC79A7"

LINE_WIDTH   = 1.8
MARKER_SIZE  = 5.5
FONT_SIZE    = 11
TITLE_SIZE   = 13
DPI          = 200

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
    "legend.fontsize": FONT_SIZE - 2,
    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "#cccccc",
    "grid.alpha": 0.3,
    "grid.color": "#aaaaaa",
})


def read_csv(path):
    """Read a CSV file, return (headers, list_of_rows)."""
    with open(path, newline="") as f:
        reader = csv.reader(f)
        headers = next(reader)
        rows = []
        for row in reader:
            rows.append([float(v) for v in row])
    return headers, rows


# ======================================================================
# Chart 1: Scan comparison (log-log)
# ======================================================================
def plot_scan(csv_path, out_dir):
    _, rows = read_csv(csv_path)
    sizes  = np.array([r[0] for r in rows], dtype=int)
    cpu    = np.array([r[1] for r in rows])
    naive  = np.array([r[2] for r in rows])
    eff    = np.array([r[3] for r in rows])
    thrust = np.array([r[4] for r in rows])

    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.loglog(sizes, cpu,    "s-",  color=CPU_COLOR,       linewidth=LINE_WIDTH,
              markersize=MARKER_SIZE, label="CPU serial scan", zorder=4)
    ax.loglog(sizes, naive,  "o--", color=NAIVE_COLOR,     linewidth=LINE_WIDTH,
              markersize=MARKER_SIZE, label="Naive GPU scan", zorder=3)
    ax.loglog(sizes, eff,    "D-",  color=EFFICIENT_COLOR, linewidth=LINE_WIDTH,
              markersize=MARKER_SIZE, label="Work-Efficient GPU scan", zorder=5)
    ax.loglog(sizes, thrust, "^-",  color=THRUST_COLOR,    linewidth=LINE_WIDTH,
              markersize=MARKER_SIZE, label="Thrust scan (library)", zorder=2)

    # Annotate slopes for reference
    mid_x = sizes[len(sizes) // 3]
    mid_y_cpu = cpu[len(cpu) // 3]
    ax.annotate("O(n)", xy=(mid_x, mid_y_cpu), xytext=(mid_x * 3, mid_y_cpu * 0.6),
                fontsize=8, color=CPU_COLOR,
                arrowprops=dict(arrowstyle="->", color=CPU_COLOR, lw=0.6))

    ax.set_xlabel("Array Size N")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Prefix Sum (Scan) — Performance vs N")
    ax.legend(loc="upper left")

    ax.set_xticks(sizes)
    ax.set_xticklabels([f"$2^{{{int(np.log2(s))}}}$" for s in sizes],
                       rotation=35, ha="right", fontsize=FONT_SIZE - 3)
    ax.grid(True, which="major", linestyle="-", linewidth=0.4)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.2)

    fig.tight_layout()
    out_path = os.path.join(out_dir, "chart_scan_comparison.png")
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    print(f"  -> {out_path}")


# ======================================================================
# Chart 2: Stream Compaction comparison (log-log)
# ======================================================================
def plot_compact(csv_path, out_dir):
    _, rows = read_csv(csv_path)
    sizes  = np.array([r[0] for r in rows], dtype=int)
    cpu_no = np.array([r[1] for r in rows])
    cpu_sc = np.array([r[2] for r in rows])
    gpu    = np.array([r[3] for r in rows])

    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.loglog(sizes, cpu_no, "s-",  color=COMPACT_CPU_NOSCAN, linewidth=LINE_WIDTH,
              markersize=MARKER_SIZE, label="CPU compact (no scan)", zorder=4)
    ax.loglog(sizes, cpu_sc, "o--", color=COMPACT_CPU_SCAN, linewidth=LINE_WIDTH,
              markersize=MARKER_SIZE, label="CPU compact (with scan)", zorder=3)
    ax.loglog(sizes, gpu,    "D-",  color=COMPACT_GPU_COLOR, linewidth=LINE_WIDTH,
              markersize=MARKER_SIZE, label="GPU work-efficient compact", zorder=5)

    ax.set_xlabel("Array Size N")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Stream Compaction — Performance vs N (~50% zeros)")
    ax.legend(loc="upper left")

    ax.set_xticks(sizes)
    ax.set_xticklabels([f"$2^{{{int(np.log2(s))}}}$" for s in sizes],
                       rotation=35, ha="right", fontsize=FONT_SIZE - 3)
    ax.grid(True, which="major", linestyle="-", linewidth=0.4)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.2)

    fig.tight_layout()
    out_path = os.path.join(out_dir, "chart_compact_comparison.png")
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    print(f"  -> {out_path}")


# ======================================================================
# Chart 3: Radix Sort comparison (log-log)
# ======================================================================
def plot_radix(csv_path, out_dir):
    _, rows = read_csv(csv_path)
    sizes = np.array([r[0] for r in rows], dtype=int)
    cpu   = np.array([r[1] for r in rows])
    gpu   = np.array([r[2] for r in rows])

    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.loglog(sizes, cpu, "s-",  color=CPU_COLOR,       linewidth=LINE_WIDTH,
              markersize=MARKER_SIZE, label="CPU std::sort", zorder=3)
    ax.loglog(sizes, gpu, "D--", color=GPU_RADIX_COLOR, linewidth=LINE_WIDTH,
              markersize=MARKER_SIZE, label="GPU Radix Sort (LSB)", zorder=4)

    # Find and annotate crossover
    for i, (c, g) in enumerate(zip(cpu, gpu)):
        if g < c:
            ax.annotate(
                f"GPU overtakes CPU\nat N=$2^{{{int(np.log2(sizes[i]))}}}$",
                xy=(sizes[i], gpu[i]),
                xytext=(sizes[i] * 0.4, gpu[i] * 2.5),
                arrowprops=dict(arrowstyle="->", color="#555555", lw=0.8),
                fontsize=FONT_SIZE - 2, color="#555555",
                ha="center",
            )
            break

    ax.set_xlabel("Array Size N")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Radix Sort — Performance vs N")
    ax.legend(loc="upper left")

    ax.set_xticks(sizes)
    ax.set_xticklabels([f"$2^{{{int(np.log2(s))}}}$" for s in sizes],
                       rotation=35, ha="right", fontsize=FONT_SIZE - 3)
    ax.grid(True, which="major", linestyle="-", linewidth=0.4)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.2)

    fig.tight_layout()
    out_path = os.path.join(out_dir, "chart_radix_comparison.png")
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    print(f"  -> {out_path}")


# ======================================================================
# Chart 4: Scan speedup (GPU / CPU) — linear X, log Y optional
# ======================================================================
def plot_scan_speedup(csv_path, out_dir):
    _, rows = read_csv(csv_path)
    sizes  = np.array([r[0] for r in rows], dtype=int)
    cpu    = np.array([r[1] for r in rows])
    naive  = np.array([r[2] for r in rows])
    eff    = np.array([r[3] for r in rows])
    thrust = np.array([r[4] for r in rows])

    naive_sp  = cpu / naive
    eff_sp    = cpu / eff
    thrust_sp = cpu / thrust

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.semilogx(sizes, naive_sp,  "o--", color=NAIVE_COLOR,     linewidth=LINE_WIDTH,
                markersize=MARKER_SIZE, label="Naive GPU")
    ax.semilogx(sizes, eff_sp,    "D-",  color=EFFICIENT_COLOR, linewidth=LINE_WIDTH,
                markersize=MARKER_SIZE, label="Work-Efficient GPU")
    ax.semilogx(sizes, thrust_sp, "^-",  color=THRUST_COLOR,    linewidth=LINE_WIDTH,
                markersize=MARKER_SIZE, label="Thrust (library)")

    ax.axhline(y=1.0, color="#999999", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.text(sizes[-1] * 0.65, 1.08, "CPU baseline = 1.0x", color="#999999",
            fontsize=9, ha="right")

    # Annotate peak speedups
    for sp, lbl, clr in [(thrust_sp, "Thrust", THRUST_COLOR),
                           (eff_sp, "Work-Eff.", EFFICIENT_COLOR)]:
        i = np.argmax(sp)
        ax.annotate(f"{lbl}\n{sp[i]:.1f}x",
                    xy=(sizes[i], sp[i]),
                    xytext=(sizes[i] * 1.5, sp[i] - 0.3),
                    arrowprops=dict(arrowstyle="->", color="#555555", lw=0.7),
                    fontsize=8, fontweight="bold", color=clr)

    ax.set_xlabel("Array Size N")
    ax.set_ylabel("Speedup vs CPU serial scan")
    ax.set_title("Scan — GPU Speedup Ratio")
    ax.legend(loc="upper left")

    ax.set_xticks(sizes)
    ax.set_xticklabels([f"$2^{{{int(np.log2(s))}}}$" for s in sizes],
                       rotation=35, ha="right", fontsize=FONT_SIZE - 3)
    ax.grid(True, which="major", linestyle="-", linewidth=0.4)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.2)

    fig.tight_layout()
    out_path = os.path.join(out_dir, "chart_scan_speedup.png")
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    print(f"  -> {out_path}")


# ======================================================================
# Chart 5: Compaction speedup
# ======================================================================
def plot_compact_speedup(csv_path, out_dir):
    _, rows = read_csv(csv_path)
    sizes  = np.array([r[0] for r in rows], dtype=int)
    cpu_no = np.array([r[1] for r in rows])
    cpu_sc = np.array([r[2] for r in rows])
    gpu    = np.array([r[3] for r in rows])

    gpu_sp_no = cpu_no / gpu
    gpu_sp_sc = cpu_sc / gpu

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.semilogx(sizes, gpu_sp_no, "D-", color=COMPACT_GPU_COLOR, linewidth=LINE_WIDTH,
                markersize=MARKER_SIZE, label="GPU vs CPU (no scan)")
    ax.semilogx(sizes, gpu_sp_sc, "o--", color=COMPACT_CPU_SCAN, linewidth=LINE_WIDTH,
                markersize=MARKER_SIZE, label="GPU vs CPU (with scan)")

    ax.axhline(y=1.0, color="#999999", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.text(sizes[-1] * 0.65, 1.08, "CPU baseline = 1.0x", color="#999999",
            fontsize=9, ha="right")

    i = np.argmax(gpu_sp_no)
    ax.annotate(f"{gpu_sp_no[i]:.1f}x",
                xy=(sizes[i], gpu_sp_no[i]),
                xytext=(sizes[i] * 1.5, gpu_sp_no[i] + 0.3),
                arrowprops=dict(arrowstyle="->", color="#555555", lw=0.7),
                fontsize=9, fontweight="bold", color=COMPACT_GPU_COLOR)

    ax.set_xlabel("Array Size N")
    ax.set_ylabel("Speedup vs CPU")
    ax.set_title("Stream Compaction — GPU Speedup Ratio (~50% zeros)")
    ax.legend(loc="upper left")

    ax.set_xticks(sizes)
    ax.set_xticklabels([f"$2^{{{int(np.log2(s))}}}$" for s in sizes],
                       rotation=35, ha="right", fontsize=FONT_SIZE - 3)
    ax.grid(True, which="major", linestyle="-", linewidth=0.4)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.2)

    fig.tight_layout()
    out_path = os.path.join(out_dir, "chart_compact_speedup.png")
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    print(f"  -> {out_path}")


# ======================================================================
# Chart 6: Radix sort speedup
# ======================================================================
def plot_radix_speedup(csv_path, out_dir):
    _, rows = read_csv(csv_path)
    sizes  = np.array([r[0] for r in rows], dtype=int)
    cpu    = np.array([r[1] for r in rows])
    gpu    = np.array([r[2] for r in rows])
    speedup = cpu / gpu

    fig, ax = plt.subplots(figsize=(9, 5))

    # Color bars above/below break-even
    colors = [GPU_RADIX_COLOR if s >= 1.0 else "#D55E00" for s in speedup]
    bars = ax.bar(np.arange(len(sizes)), speedup, color=colors, width=0.6, edgecolor="white")

    ax.axhline(y=1.0, color="#333333", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.text(len(sizes) - 1.3, 1.08, "break-even", color="#333333", fontsize=9, ha="right")

    ax.set_xticks(np.arange(len(sizes)))
    ax.set_xticklabels([f"$2^{{{int(np.log2(s))}}}$" for s in sizes],
                       rotation=35, ha="right", fontsize=FONT_SIZE - 3)
    ax.set_xlabel("Array Size N")
    ax.set_ylabel("Speedup (CPU / GPU)")
    ax.set_title("Radix Sort — GPU Speedup Ratio")

    # Value labels on bars
    for i, (s, sp) in enumerate(zip(sizes, speedup)):
        va = "bottom" if sp >= 1.0 else "top"
        y_off = 0.05 if sp >= 1 else -0.05
        ax.text(i, sp + y_off, f"{sp:.1f}x", ha="center", fontsize=7,
                fontweight="bold", va=va, color=colors[i])

    ax.grid(True, which="major", axis="y", linestyle="-", linewidth=0.4)
    ax.grid(True, which="minor", axis="y", linestyle=":", linewidth=0.2)

    fig.tight_layout()
    out_path = os.path.join(out_dir, "chart_radix_speedup.png")
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    print(f"  -> {out_path}")


# ======================================================================
# Chart 7: Combined Overview — GPU speedup for all algorithms
# ======================================================================
def plot_overview(scan_csv, compact_csv, radix_csv, out_dir):
    """Unified view: best GPU method from each category vs its CPU baseline."""
    fig, ax = plt.subplots(figsize=(9, 5.5))

    # Scan: use Thrust as best GPU
    if scan_csv and os.path.exists(scan_csv):
        _, rows = read_csv(scan_csv)
        sizes_s = np.array([r[0] for r in rows], dtype=int)
        cpu_s   = np.array([r[1] for r in rows])
        thrust  = np.array([r[4] for r in rows])
        sp_s    = cpu_s / thrust
        ax.semilogx(sizes_s, sp_s, "^-", color=THRUST_COLOR, linewidth=LINE_WIDTH,
                    markersize=MARKER_SIZE, label="Scan (Thrust)")

    # Compact: GPU efficient
    if compact_csv and os.path.exists(compact_csv):
        _, rows = read_csv(compact_csv)
        sizes_c = np.array([r[0] for r in rows], dtype=int)
        cpu_c   = np.array([r[1] for r in rows])
        gpu_c   = np.array([r[3] for r in rows])
        sp_c    = cpu_c / gpu_c
        ax.semilogx(sizes_c, sp_c, "D-", color=COMPACT_GPU_COLOR, linewidth=LINE_WIDTH,
                    markersize=MARKER_SIZE, label="Stream Compaction")

    # Radix
    if radix_csv and os.path.exists(radix_csv):
        _, rows = read_csv(radix_csv)
        sizes_r = np.array([r[0] for r in rows], dtype=int)
        cpu_r   = np.array([r[1] for r in rows])
        gpu_r   = np.array([r[2] for r in rows])
        sp_r    = cpu_r / gpu_r
        ax.semilogx(sizes_r, sp_r, "s-", color=GPU_RADIX_COLOR, linewidth=LINE_WIDTH,
                    markersize=MARKER_SIZE, label="Radix Sort")

    ax.axhline(y=1.0, color="#333333", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.text(ax.get_xlim()[1] * 0.5, 1.08, "CPU baseline", color="#333333",
            fontsize=9, ha="center")

    ax.set_xlabel("Array Size N")
    ax.set_ylabel("GPU Speedup vs CPU")
    ax.set_title("GPU Acceleration Summary — Speedup vs N")
    ax.legend(loc="upper left")

    ax.grid(True, which="major", linestyle="-", linewidth=0.4)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.2)

    fig.tight_layout()
    out_path = os.path.join(out_dir, "chart_overview.png")
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    print(f"  -> {out_path}")


# ======================================================================
# Chart 8: Throughput (elements/ms) — better for comparing raw speed
# ======================================================================
def plot_throughput(scan_csv, compact_csv, radix_csv, out_dir):
    """Show throughput (M elements / ms) for GPU implementations only."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # --- Scan throughput ---
    if scan_csv and os.path.exists(scan_csv):
        _, rows = read_csv(scan_csv)
        sizes  = np.array([r[0] for r in rows], dtype=int)
        eff    = np.array([r[3] for r in rows])  # work-efficient
        thrust = np.array([r[4] for r in rows])
        ax1.loglog(sizes, sizes / eff / 1e3,    "D-", color=EFFICIENT_COLOR,
                   linewidth=LINE_WIDTH, markersize=MARKER_SIZE,
                   label="Work-Efficient GPU")
        ax1.loglog(sizes, sizes / thrust / 1e3, "^-", color=THRUST_COLOR,
                   linewidth=LINE_WIDTH, markersize=MARKER_SIZE,
                   label="Thrust")

    ax1.set_xlabel("Array Size N")
    ax1.set_ylabel("Throughput (Melements/ms)")
    ax1.set_title("Scan — GPU Throughput")
    ax1.legend(loc="lower right")
    ax1.grid(True, which="major", linestyle="-", linewidth=0.4)
    ax1.grid(True, which="minor", linestyle=":", linewidth=0.2)

    # --- Radix throughput ---
    if radix_csv and os.path.exists(radix_csv):
        _, rows = read_csv(radix_csv)
        sizes = np.array([r[0] for r in rows], dtype=int)
        cpu   = np.array([r[1] for r in rows])
        gpu   = np.array([r[2] for r in rows])
        ax2.loglog(sizes, sizes / cpu / 1e3, "s-",  color=CPU_COLOR,
                   linewidth=LINE_WIDTH, markersize=MARKER_SIZE,
                   label="CPU std::sort")
        ax2.loglog(sizes, sizes / gpu / 1e3, "D--", color=GPU_RADIX_COLOR,
                   linewidth=LINE_WIDTH, markersize=MARKER_SIZE,
                   label="GPU Radix Sort")

    ax2.set_xlabel("Array Size N")
    ax2.set_ylabel("Throughput (Melements/ms)")
    ax2.set_title("Radix Sort — Throughput Comparison")
    ax2.legend(loc="upper left")
    ax2.grid(True, which="major", linestyle="-", linewidth=0.4)
    ax2.grid(True, which="minor", linestyle=":", linewidth=0.2)

    fig.tight_layout()
    out_path = os.path.join(out_dir, "chart_throughput.png")
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    print(f"  -> {out_path}")


# Shared memory scan colours
SHARED_COLOR        = "#D55E00"   # shared-memory lines (warm orange)
GLOBAL_COLOR        = "#0072B2"   # global-memory lines (blue)
BC_COLOR            = "#009E73"   # bank-conflict-free (green)
NOBC_COLOR          = "#CC79A7"   # bank-conflicted (magenta/pink)
SHARED_NAIVE_COLOR  = "#E69F00"   # shared naive


# ======================================================================
# Chart 9: Shared-Memory vs Global-Memory Scan (log-log)
# ======================================================================
def plot_sharedmem_comparison(csv_path, out_dir):
    """Compare all scan methods at small N (where shared-mem is valid)."""
    _, rows = read_csv(csv_path)
    sizes  = np.array([r[0] for r in rows], dtype=int)
    nv_gl  = np.array([r[1] for r in rows])   # naive global
    nv_sh  = np.array([r[2] for r in rows])   # naive shared
    ef_gl  = np.array([r[3] for r in rows])   # efficient global
    ef_bc  = np.array([r[4] for r in rows])   # efficient shared (BC)
    ef_nbc = np.array([r[5] for r in rows])   # efficient shared (no BC)

    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.loglog(sizes, nv_gl,  "o--", color=NAIVE_COLOR,          linewidth=LINE_WIDTH,
              markersize=MARKER_SIZE, label="Naive Global Mem", zorder=3)
    ax.loglog(sizes, nv_sh,  "o-",  color=SHARED_NAIVE_COLOR,   linewidth=LINE_WIDTH,
              markersize=MARKER_SIZE, label="Naive Shared Mem", zorder=4)
    ax.loglog(sizes, ef_gl,  "D--", color=EFFICIENT_COLOR,      linewidth=LINE_WIDTH,
              markersize=MARKER_SIZE, label="Efficient Global Mem", zorder=2)
    ax.loglog(sizes, ef_bc,  "s-",  color=BC_COLOR,             linewidth=LINE_WIDTH,
              markersize=MARKER_SIZE, label="Efficient Shared Mem (bank-conflict free)", zorder=5)
    ax.loglog(sizes, ef_nbc, "s--", color=NOBC_COLOR,           linewidth=LINE_WIDTH,
              markersize=MARKER_SIZE, label="Efficient Shared Mem (bank-conflicted)", zorder=1)

    ax.set_xlabel("Array Size N")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Shared Memory vs Global Memory Scan — Small N")
    ax.legend(loc="upper left", fontsize=FONT_SIZE - 3)

    ax.set_xticks(sizes)
    ax.set_xticklabels([str(s) for s in sizes], rotation=35, ha="right")
    ax.grid(True, which="major", linestyle="-", linewidth=0.4)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.2)

    fig.tight_layout()
    out_path = os.path.join(out_dir, "chart_sharedmem_comparison.png")
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    print(f"  -> {out_path}")


# ======================================================================
# Chart 10: Shared-Memory Speedup Ratio (shared / global)
# ======================================================================
def plot_sharedmem_speedup(csv_path, out_dir):
    """GPU shared-memory speedup vs global-memory baseline (both GPU)."""
    _, rows = read_csv(csv_path)
    sizes  = np.array([r[0] for r in rows], dtype=int)
    nv_gl  = np.array([r[1] for r in rows])
    nv_sh  = np.array([r[2] for r in rows])
    ef_gl  = np.array([r[3] for r in rows])
    ef_bc  = np.array([r[4] for r in rows])

    # Speedup = how many times faster shared-mem is vs global-mem
    naive_sp   = nv_gl / nv_sh       # > 1 means shared is faster
    eff_sp     = ef_gl / ef_bc       # > 1 means shared is faster

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.semilogx(sizes, naive_sp, "o-",  color=SHARED_NAIVE_COLOR, linewidth=LINE_WIDTH,
                markersize=MARKER_SIZE, label="Naive: Shared / Global")
    ax.semilogx(sizes, eff_sp,   "s-",  color=BC_COLOR,           linewidth=LINE_WIDTH,
                markersize=MARKER_SIZE, label="Efficient: Shared (BC-free) / Global")

    ax.axhline(y=1.0, color="#999999", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.text(sizes[0] * 1.1, 1.03, "global-mem baseline = 1.0x",
            color="#999999", fontsize=9, ha="left")

    ax.set_xlabel("Array Size N")
    ax.set_ylabel("Speedup (Shared / Global)")
    ax.set_title("Shared Memory Advantage — Speedup vs Global Memory Scan")
    ax.legend(loc="upper left")

    ax.set_xticks(sizes)
    ax.set_xticklabels([str(s) for s in sizes], rotation=35, ha="right")
    ax.grid(True, which="major", linestyle="-", linewidth=0.4)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.2)

    fig.tight_layout()
    out_path = os.path.join(out_dir, "chart_sharedmem_speedup.png")
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    print(f"  -> {out_path}")


# ======================================================================
# Chart 11: Bank-Conflict Avoidance Impact
# ======================================================================
def plot_bank_conflict_impact(csv_path, out_dir):
    """Show the direct impact of bank-conflict-free padding.

    Left:  absolute time (BC vs no-BC) — line chart.
    Right: speedup ratio (no-BC / BC)  — bar chart.
    A ratio > 1 means padding helps; ~1 means padding is neutral.     """
    _, rows = read_csv(csv_path)
    sizes  = np.array([r[0] for r in rows], dtype=int)
    ef_bc  = np.array([r[4] for r in rows])
    ef_nbc = np.array([r[5] for r in rows])

    speedup = ef_nbc / ef_bc            # > 1 => BC-free is faster

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # --- Left: absolute time comparison ---
    ax1.semilogy(sizes, ef_bc,  "s-",  color=BC_COLOR,   linewidth=LINE_WIDTH,
                 markersize=MARKER_SIZE, label="Bank-conflict free (padded)")
    ax1.semilogy(sizes, ef_nbc, "s--", color=NOBC_COLOR,  linewidth=LINE_WIDTH,
                 markersize=MARKER_SIZE, label="Bank-conflicted (unpadded)")
    ax1.set_xlabel("Array Size N")
    ax1.set_ylabel("Time (ms) — log scale")
    ax1.set_title("Absolute Time: BC-free vs BC-conflicted")
    ax1.legend(loc="upper left")
    ax1.set_xticks(sizes)
    ax1.set_xticklabels([str(s) for s in sizes], rotation=35, ha="right")
    ax1.grid(True, which="major", linestyle="-", linewidth=0.4)

    # --- Right: speedup bars ---
    bar_colors = [BC_COLOR if s >= 1.0 else NOBC_COLOR for s in speedup]
    ax2.bar(np.arange(len(sizes)), speedup, color=bar_colors,
            width=0.55, edgecolor="white")
    ax2.axhline(y=1.0, color="#333333", linestyle="--", linewidth=1.0, alpha=0.8)
    ax2.set_xticks(np.arange(len(sizes)))
    ax2.set_xticklabels([str(s) for s in sizes], rotation=35, ha="right")
    ax2.set_xlabel("Array Size N")
    ax2.set_ylabel("Speedup ratio (unpadded / padded)")
    ax2.set_title("BC-Avoidance Speedup: > 1 means padding helps")

    # Value labels
    for i, (s, sp) in enumerate(zip(sizes, speedup)):
        va = "bottom" if sp >= 1 else "top"
        y_off = 0.005 if sp >= 1 else -0.005
        ax2.text(i, sp + y_off * max(speedup), f"{sp:.2f}x",
                 ha="center", fontsize=8, fontweight="bold",
                 va=va, color=bar_colors[i])

    ax2.grid(True, which="major", axis="y", linestyle="-", linewidth=0.4)

    fig.tight_layout()
    out_path = os.path.join(out_dir, "chart_bank_conflict_impact.png")
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    print(f"  -> {out_path}")


# ======================================================================
# Chart 12: Shared-Memory Throughput (M elements / ms)
# ======================================================================
def plot_sharedmem_throughput(csv_path, out_dir):
    """Throughput for shared-memory scan implementations."""
    _, rows = read_csv(csv_path)
    sizes  = np.array([r[0] for r in rows], dtype=int)
    nv_sh  = np.array([r[2] for r in rows])
    ef_gl  = np.array([r[3] for r in rows])
    ef_bc  = np.array([r[4] for r in rows])

    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.semilogy(sizes, sizes / ef_gl / 1e3, "D--", color=EFFICIENT_COLOR,
                linewidth=LINE_WIDTH, markersize=MARKER_SIZE,
                label="Efficient Global Mem")
    ax.semilogy(sizes, sizes / nv_sh / 1e3, "o-", color=SHARED_NAIVE_COLOR,
                linewidth=LINE_WIDTH, markersize=MARKER_SIZE,
                label="Naive Shared Mem")
    ax.semilogy(sizes, sizes / ef_bc / 1e3, "s-", color=BC_COLOR,
                linewidth=LINE_WIDTH, markersize=MARKER_SIZE,
                label="Efficient Shared Mem (BC-free)")

    ax.set_xlabel("Array Size N")
    ax.set_ylabel("Throughput (Melements / ms)")
    ax.set_title("Shared Memory Scan — Throughput")
    ax.legend(loc="upper left")

    ax.set_xticks(sizes)
    ax.set_xticklabels([str(s) for s in sizes], rotation=35, ha="right")
    ax.grid(True, which="major", linestyle="-", linewidth=0.4)

    fig.tight_layout()
    out_path = os.path.join(out_dir, "chart_sharedmem_throughput.png")
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

    scan_csv    = data_dir / "scan_comparison.csv"
    compact_csv = data_dir / "compact_comparison.csv"
    radix_csv   = data_dir / "radix_comparison.csv"

    print(f"Plotting from: {data_dir}")
    print()

    if scan_csv.exists():
        plot_scan(scan_csv, data_dir)
        plot_scan_speedup(scan_csv, data_dir)
    else:
        print("  (no scan CSV found)")

    if compact_csv.exists():
        plot_compact(compact_csv, data_dir)
        plot_compact_speedup(compact_csv, data_dir)
    else:
        print("  (no compact CSV found)")

    if radix_csv.exists():
        plot_radix(radix_csv, data_dir)
        plot_radix_speedup(radix_csv, data_dir)
    else:
        print("  (no radix CSV found)")

    sharedmem_csv = data_dir / "sharedmem_comparison.csv"
    if sharedmem_csv.exists():
        print()
        plot_sharedmem_comparison(sharedmem_csv, data_dir)
        plot_sharedmem_speedup(sharedmem_csv, data_dir)
        plot_bank_conflict_impact(sharedmem_csv, data_dir)
        plot_sharedmem_throughput(sharedmem_csv, data_dir)
    else:
        print("  (no sharedmem CSV found)")

    # Combined overview and throughput — only when at least one large-scale
    # CSV (scan / compact / radix) exists.  Shared-memory-only runs skip these.
    large_scale_csvs = [scan_csv, compact_csv, radix_csv]
    n_large = sum(int(c.exists()) for c in large_scale_csvs)
    if n_large >= 1:
        print()
        plot_overview(scan_csv, compact_csv, radix_csv, data_dir)
        plot_throughput(scan_csv, compact_csv, radix_csv, data_dir)
    else:
        print("\n  (skipping overview/throughput — no large-scale CSVs found)")

    print("\nDone.")


if __name__ == "__main__":
    main()
