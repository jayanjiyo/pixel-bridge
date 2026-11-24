"""Benchmark runners for the evaluation plan (plan report section 6).

Glue code, already implemented. OWNER of the evaluation design: M3 --
change the defaults here if the team decides on a different protocol.
Scripts that call these: scripts/benchmark_a.py, scripts/benchmark_b.py
"""
from __future__ import annotations

import csv
import os
import time
from dataclasses import replace

import numpy as np

from ..core import resize
from ..core.io import read_image
from ..module_a import palette as palette_mod
from ..module_a.downsample import target_size
from ..module_a.pipeline import PixelArtParams, photo_to_pixel_art
from ..module_b.pipeline import METHODS, SUPPORTED_SCALES, upscale
from . import metrics


# ---------------------------------------------------------------- Module B
def make_lowres_pair(img: np.ndarray, factor: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Create a pixel-art-like input from a flat-color illustration.

    The image is cropped to a multiple of `factor` (reference) and every
    factor-th pixel (block center) is sampled, which keeps hard colors
    like real pixel art. Returns (small, reference).
    """
    h = (img.shape[0] // factor) * factor
    w = (img.shape[1] // factor) * factor
    reference = img[:h, :w]
    c = factor // 2
    small = np.ascontiguousarray(reference[c::factor, c::factor])
    return small, reference


def benchmark_module_b(paths, factor: int = 4, methods=METHODS, mode: str = "palette") -> list[dict]:
    """Shrink each illustration, enlarge it again with every method, and
    compare with the original (PSNR, SSIM, time)."""
    rows = []
    for path in paths:
        small, reference = make_lowres_pair(read_image(path), factor)
        for m in methods:
            if factor not in SUPPORTED_SCALES[m]:
                continue
            t0 = time.perf_counter()
            out = upscale(small, factor, m, mode)
            seconds = time.perf_counter() - t0
            rows.append({
                "file": os.path.basename(path), "method": m, "mode": mode,
                "psnr": metrics.psnr(reference, out),
                "ssim": metrics.ssim(reference, out),
                "seconds": seconds,
            })
    return rows


# ---------------------------------------------------------------- Module A
def baseline_pixel_art(photo: np.ndarray, out_width: int, n_colors: int) -> np.ndarray:
    """The simple method we compare against: area downscaling + median cut
    palette + nearest palette color (no smoothing, outline, or cleanup)."""
    h, w = target_size(photo.shape[0], photo.shape[1], out_width)
    small = resize.resize_area(photo, h, w)
    pal = palette_mod.median_cut(small, n_colors)
    return pal[palette_mod.map_to_palette(small, pal)]


def benchmark_module_a(paths, params: PixelArtParams | None = None) -> list[dict]:
    """Compare our Module A with the baseline on each photo.

    Reference = the photo area-averaged to the pixel art size (the "true"
    colors). Metrics: mean Delta E, number of isolated pixels, ratio of
    preserved strong edges, processing time.
    """
    p = params or PixelArtParams()
    rows = []
    for path in paths:
        photo = read_image(path)
        h, w = target_size(photo.shape[0], photo.shape[1], p.out_width)
        reference = resize.resize_area(photo, h, w)

        t0 = time.perf_counter()
        ours = photo_to_pixel_art(photo, p).image
        t_ours = time.perf_counter() - t0

        t0 = time.perf_counter()
        base = baseline_pixel_art(photo, p.out_width, p.n_colors)
        t_base = time.perf_counter() - t0

        for name, out, seconds in (("ours", ours, t_ours), ("baseline", base, t_base)):
            rows.append({
                "file": os.path.basename(path), "method": name,
                "delta_e": metrics.mean_delta_e(reference, out),
                "isolated": metrics.count_isolated_pixels(out),
                "edge_ratio": metrics.edge_preservation_ratio(reference, out),
                "seconds": seconds,
            })
    return rows


def params_variant(params: PixelArtParams, **changes) -> PixelArtParams:
    """Copy of params with some fields changed (for ablation studies)."""
    return replace(params, **changes)


# ---------------------------------------------------------------- helpers
def save_csv(rows: list[dict], path: str) -> None:
    if not rows:
        return
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict], fields: list[str], key: str = "method") -> dict[str, dict[str, float]]:
    """Mean of each field per method (inf values, e.g. PSNR of identical images, are skipped)."""
    groups: dict[str, dict[str, list[float]]] = {}
    for r in rows:
        g = groups.setdefault(r[key], {f: [] for f in fields})
        for f in fields:
            v = float(r[f])
            if np.isfinite(v):
                g[f].append(v)
    return {k: {f: (float(np.mean(v)) if v else float("nan")) for f, v in g.items()}
            for k, g in groups.items()}


def list_images(folder: str) -> list[str]:
    exts = (".png", ".jpg", ".jpeg", ".bmp")
    if not os.path.isdir(folder):
        return []
    return sorted(os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(exts))
