#!/usr/bin/env python3
"""
run_benchmark.py -- Build and run the parameterized benchmark,
collecting CSV results into a single timestamped folder under results/,
then optionally generate charts.

Usage:
  python scripts/run_benchmark.py                    # all benchmarks + charts
  python scripts/run_benchmark.py --mode scan        # scan only
  python scripts/run_benchmark.py --mode radix       # radix sort only
  python scripts/run_benchmark.py --no-plot          # skip chart generation
"""

import subprocess
import sys
import os
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BUILD_DIR    = PROJECT_ROOT / "build"
BENCHMARK_EXE = BUILD_DIR / "bin" / "Release" / "benchmark.exe"

# ---------------------------------------------------------------------------
# Size ranges -- powers of 2 from 2^8 up through GPU-saturating sizes.
# ---------------------------------------------------------------------------
SCAN_SIZES = [
    256, 512, 1024, 2048, 4096, 8192,
    16384, 32768, 65536, 131072, 262144,
    524288, 1048576, 2097152, 4194304,
]

COMPACT_SIZES = [
    256, 512, 1024, 2048, 4096, 8192,
    16384, 32768, 65536, 131072, 262144,
    524288, 1048576, 2097152, 4194304,
]

RADIX_SIZES = [
    256, 512, 1024, 2048, 4096, 8192,
    16384, 32768, 65536, 131072, 262144,
    524288, 1048576, 2097152,
]

# Shared-memory scan: single-block; only sizes <= blockSize (128) are valid.
# Includes power-of-2 and non-power-of-2 to probe padding + tree shape effects.
SHAREDMEM_SIZES = [
    32, 48, 64, 96, 100, 127, 128,
]


def sizes_to_arg(sizes):
    return ",".join(str(s) for s in sizes)


def run_benchmark(mode, sizes, out_dir):
    """Run one benchmark mode, writing CSV into *out_dir*."""
    sizes_str = sizes_to_arg(sizes)
    cmd = [
        str(BENCHMARK_EXE),
        "--mode", mode,
        "--sizes", sizes_str,
        "--output", str(out_dir),
    ]

    print(f"\n{'='*60}")
    print(f"  MODE:  {mode}")
    print(f"  SIZES: {len(sizes)} points  ({sizes[0]} .. {sizes[-1]})")
    print(f"  OUT:   {out_dir}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT),
                            capture_output=False, text=True)
    if result.returncode != 0:
        print(f"  ERROR: benchmark returned {result.returncode}")
        return False
    return True


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run GPU benchmarks")
    parser.add_argument("--mode",
                        choices=["scan", "compact", "radix", "sharedmem", "all"],
                        default="all")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip chart generation")
    args = parser.parse_args()

    # ---- Build ----
    print("Building Release ...")
    subprocess.run(["cmake", "--build", str(BUILD_DIR), "--config", "Release"],
                   check=True, cwd=str(PROJECT_ROOT))
    if not BENCHMARK_EXE.exists():
        print(f"ERROR: benchmark not found at {BENCHMARK_EXE}")
        sys.exit(1)

    # ---- Create single output folder for this experiment ----
    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    out_dir = PROJECT_ROOT / "results" / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nExperiment folder: {out_dir}")

    # ---- Run ----
    ok = True
    if args.mode in ("scan", "all"):
        ok &= run_benchmark("scan", SCAN_SIZES, out_dir)
    if args.mode in ("compact", "all"):
        ok &= run_benchmark("compact", COMPACT_SIZES, out_dir)
    if args.mode in ("radix", "all"):
        ok &= run_benchmark("radix", RADIX_SIZES, out_dir)
    if args.mode in ("sharedmem", "all"):
        ok &= run_benchmark("sharedmem", SHAREDMEM_SIZES, out_dir)

    # ---- Plot ----
    if ok and not args.no_plot:
        print(f"\n{'='*60}")
        print("  Generating charts ...")
        print(f"{'='*60}")
        plot_script = PROJECT_ROOT / "scripts" / "plot.py"
        subprocess.run([sys.executable, str(plot_script), str(out_dir)],
                       check=False, cwd=str(PROJECT_ROOT))
    elif not ok:
        print("\nSome benchmarks failed -- skipping chart generation.")

    print(f"\nDone. Results: {out_dir}")
    if out_dir.exists():
        for f in sorted(out_dir.iterdir()):
            print(f"  {f.name}")


if __name__ == "__main__":
    main()
