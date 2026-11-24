"""Module A: photo -> pixel art.

    preprocess.py  M2   resize to working size, bilateral filter, enhance
    downsample.py  M2   block-mode and edge-aware downsampling
    palette.py     M2   median cut, k-means (CIELAB), fixed palettes, mapping
    dither.py      M3   Floyd-Steinberg, Bayer, gradient mask
    outline.py     M3   one-pixel dark outlines
    cleanup.py     M3   isolated pixel removal, one-pixel line correction
    pipeline.py    M1   connects the steps (already implemented)
"""
from .pipeline import PixelArtParams, PixelArtResult, photo_to_pixel_art  # noqa: F401
