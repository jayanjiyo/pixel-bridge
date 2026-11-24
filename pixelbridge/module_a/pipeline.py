"""Module A pipeline: connects the steps.

OWNER: M1 (Leader) -- already implemented in the skeleton.
Step owners implement the functions it calls; nobody needs to edit this
file to make their step work. Order of steps:

    preprocess -> downsample -> palette -> quantize (nearest or dither)
               -> outline -> cleanup

(Dithering happens while mapping colors to the palette, so it comes before
the outline in code even though the plan report lists it after.)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..core.io import check_rgb
from . import cleanup, dither, downsample, outline, palette, preprocess


@dataclass
class PixelArtParams:
    out_width: int = 64             # 32..128 pixels
    working_size: int = 512         # longer side before downsampling
    smooth: bool = True             # bilateral filter
    enhance: float = 0.0            # 0 = off, 1 = strong contrast/saturation boost
    downsample: str = "edge"        # "block" (Stage 1) | "edge" (Stage 2)
    n_colors: int = 16              # 4..32
    palette: str = "auto"           # "auto" or a key of palette.FIXED_PALETTES
    palette_method: str = "kmeans"  # "median_cut" (Stage 1) | "kmeans" (Stage 2)
    dither: str = "none"            # "none" | "fs" | "bayer"
    dither_strength: float = 1.0    # 0..1
    dither_flat_only: bool = True   # dither only in smooth regions (gradient mask)
    outline: bool = True
    cleanup: bool = True


@dataclass
class PixelArtResult:
    image: np.ndarray                                       # (h, w, 3) uint8 final pixel art
    indices: np.ndarray                                     # (h, w) int32
    palette: np.ndarray                                     # (K, 3) uint8
    steps: dict[str, np.ndarray] = field(default_factory=dict)   # intermediate RGB images
    timings: dict[str, float] = field(default_factory=dict)      # seconds per step


def photo_to_pixel_art(photo: np.ndarray, params: PixelArtParams | None = None) -> PixelArtResult:
    p = params or PixelArtParams()
    check_rgb(photo, "photo")
    steps: dict[str, np.ndarray] = {}
    timings: dict[str, float] = {}

    def timed(name, fn, *args, **kwargs):
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        timings[name] = time.perf_counter() - t0
        return out

    # 1. preprocess
    work = timed("preprocess", preprocess.preprocess, photo,
                 working_size=p.working_size, smooth=p.smooth, enhance_amount=p.enhance)
    steps["1_preprocess"] = work

    # 2. downsample
    out_h, out_w = downsample.target_size(work.shape[0], work.shape[1], p.out_width)
    if p.downsample == "block":
        small = timed("downsample", downsample.downsample_block_mode, work, out_h, out_w)
    elif p.downsample == "edge":
        small = timed("downsample", downsample.downsample_edge_aware, work, out_h, out_w)
    else:
        raise ValueError(f"unknown downsample method: {p.downsample}")
    steps["2_downsample"] = small

    # 3. palette
    if p.palette == "auto":
        pal = timed("palette", palette.median_cut, small, p.n_colors)
        if p.palette_method == "kmeans":
            pal = timed("palette_kmeans", palette.kmeans_lab, small, p.n_colors, init=pal)
    else:
        pal = palette.FIXED_PALETTES[p.palette]

    # 4. quantize (nearest color or dithering)
    if p.dither == "none":
        idx = timed("quantize", palette.map_to_palette, small, pal)
    else:
        mask = timed("dither_mask", dither.gradient_mask, small) if p.dither_flat_only else None
        if p.dither == "fs":
            idx = timed("quantize", dither.floyd_steinberg, small, pal,
                        strength=p.dither_strength, mask=mask)
        elif p.dither == "bayer":
            idx = timed("quantize", dither.bayer, small, pal,
                        strength=p.dither_strength, mask=mask)
        else:
            raise ValueError(f"unknown dither method: {p.dither}")
    steps["3_quantize"] = pal[idx]

    # 5. outline
    if p.outline:
        idx = timed("outline", outline.add_outline, idx, pal)
        steps["4_outline"] = pal[idx]

    # 6. cleanup
    if p.cleanup:
        idx = timed("cleanup_isolated", cleanup.remove_isolated, idx)
        idx = timed("cleanup_lines", cleanup.fix_one_pixel_lines, idx)
        steps["5_cleanup"] = pal[idx]

    idx = np.asarray(idx, dtype=np.int32)
    return PixelArtResult(image=pal[idx], indices=idx, palette=pal, steps=steps, timings=timings)
