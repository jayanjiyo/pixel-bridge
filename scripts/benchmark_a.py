"""Module A evaluation: ours vs. the simple baseline (plan report section 6).  Owner: M3.

    python scripts/benchmark_a.py [--folder data/photos] [--width 64] [--colors 16]

Writes results/benchmark_a.csv, results/benchmark_a.png, and the output
images of both methods to results/module_a/ (for the survey and the report).
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pixelbridge.core.io import read_image, write_image  # noqa: E402
from pixelbridge.evaluation.benchmark import (baseline_pixel_art, benchmark_module_a,  # noqa: E402
                                              list_images, save_csv, summarize)
from pixelbridge.module_a.pipeline import PixelArtParams, photo_to_pixel_art  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default="data/photos")
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--colors", type=int, default=16)
    args = ap.parse_args()

    paths = list_images(args.folder)
    if not paths:
        print(f"No images in {args.folder}")
        return 1
    params = PixelArtParams(out_width=args.width, n_colors=args.colors)
    rows = benchmark_module_a(paths, params)
    save_csv(rows, "results/benchmark_a.csv")
    fields = ["delta_e", "isolated", "edge_ratio", "seconds"]
    summary = summarize(rows, fields)
    print(f"{'method':10s} " + " ".join(f"{f:>10s}" for f in fields))
    for m, s in summary.items():
        print(f"{m:10s} " + " ".join(f"{s[f]:10.3f}" for f in fields))

    for path in paths:
        stem = os.path.splitext(os.path.basename(path))[0]
        photo = read_image(path)
        write_image(f"results/module_a/{stem}_ours.png", photo_to_pixel_art(photo, params).image)
        write_image(f"results/module_a/{stem}_baseline.png",
                    baseline_pixel_art(photo, args.width, args.colors))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(summary)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    for ax, f, title in zip(axes, ["delta_e", "isolated", "edge_ratio"],
                            ["Mean Delta E (lower is better)", "Isolated pixels (lower)",
                             "Preserved edges (higher)"]):
        ax.bar(names, [summary[m][f] for m in names])
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig("results/benchmark_a.png", dpi=150)
    print("saved results/benchmark_a.csv, results/benchmark_a.png, results/module_a/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
