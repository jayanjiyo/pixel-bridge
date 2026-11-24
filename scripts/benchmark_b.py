"""Module B quantitative evaluation (plan report section 6).  Owner: M3.

    python scripts/benchmark_b.py [--folder data/illustrations] [--factor 4] [--mode palette]

Writes results/benchmark_b_<mode>.csv and results/benchmark_b_<mode>.png.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pixelbridge.evaluation.benchmark import (benchmark_module_b, list_images,  # noqa: E402
                                              save_csv, summarize)
from pixelbridge.module_b.pipeline import METHODS  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default="data/illustrations")
    ap.add_argument("--factor", type=int, default=4)
    ap.add_argument("--mode", default="palette", choices=["palette", "smooth"])
    ap.add_argument("--methods", nargs="*", default=list(METHODS))
    args = ap.parse_args()

    paths = list_images(args.folder)
    if not paths:
        print(f"No images in {args.folder}")
        return 1
    rows = benchmark_module_b(paths, args.factor, args.methods, args.mode)
    save_csv(rows, f"results/benchmark_b_{args.mode}.csv")
    summary = summarize(rows, ["psnr", "ssim", "seconds"])

    print(f"{'method':10s} {'PSNR':>8s} {'SSIM':>8s} {'time(s)':>8s}")
    for m, s in summary.items():
        print(f"{m:10s} {s['psnr']:8.2f} {s['ssim']:8.4f} {s['seconds']:8.3f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(summary)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(names, [summary[m]["psnr"] for m in names])
    axes[0].set_title(f"PSNR (dB), x{args.factor}, {len(paths)} images")
    axes[1].bar(names, [summary[m]["ssim"] for m in names])
    axes[1].set_title("SSIM")
    for ax in axes:
        ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(f"results/benchmark_b_{args.mode}.png", dpi=150)
    print(f"saved results/benchmark_b_{args.mode}.csv and .png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
