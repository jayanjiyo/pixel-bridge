"""Image input/output and dtype helpers.

OWNER: M1 (Leader) -- already implemented in the skeleton.

OpenCV is used ONLY in this file (imdecode / imencode). The course allows
OpenCV for input and output; every other module must work on NumPy arrays
with our own code. `scripts/check_rules.py` checks this automatically.

Conventions (see README.md):
    RGB image   : np.ndarray (H, W, 3) uint8, channel order R, G, B
    float image : np.ndarray (H, W, 3) float32, values in [0, 1]
"""
from __future__ import annotations

import os

import cv2
import numpy as np


def read_image(path: str, background=(255, 255, 255)) -> np.ndarray:
    """Read an image file as an RGB uint8 array of shape (H, W, 3).

    - Grayscale files are expanded to three channels.
    - Images with an alpha channel (RGBA sprites) are composited over
      `background` (an RGB tuple).
    - 16-bit images are reduced to 8 bits.
    - Phone photos are turned upright (the EXIF orientation is applied).
    - Works with non-ASCII (e.g. Korean) paths on Windows.
    """
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Cannot decode image: {path}")
    if img.dtype == np.uint8 and img.ndim == 3 and img.shape[2] == 3:
        # A phone stores a portrait photo sideways plus an EXIF note "rotate me".
        # IMREAD_UNCHANGED ignores that note; IMREAD_COLOR applies it. Only plain
        # 8-bit color images are decoded again, so alpha and 16-bit files keep
        # the handling below.
        upright = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if upright is not None:
            img = upright
    if img.dtype == np.uint16:
        img = (img >> 8).astype(np.uint8)
    elif img.dtype != np.uint8:
        raise ValueError(f"Unsupported image dtype {img.dtype}: {path}")

    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img.shape[2] == 4:
        bgr = img[..., :3].astype(np.float32)
        alpha = img[..., 3:4].astype(np.float32) / 255.0
        bg_bgr = np.array(background[::-1], dtype=np.float32)
        img = np.round(bgr * alpha + bg_bgr * (1.0 - alpha)).astype(np.uint8)
    return np.ascontiguousarray(img[..., ::-1])  # BGR -> RGB


def write_image(path: str, img: np.ndarray) -> None:
    """Write an RGB uint8 (H, W, 3) or gray (H, W) image. Format from extension."""
    img = to_uint8(img)
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    ext = os.path.splitext(path)[1] or ".png"
    out = img[..., ::-1] if img.ndim == 3 else img  # RGB -> BGR
    ok, buf = cv2.imencode(ext, np.ascontiguousarray(out))
    if not ok:
        raise ValueError(f"Cannot encode image as {ext}: {path}")
    buf.tofile(path)


def encode_png(img: np.ndarray) -> bytes:
    """Encode an RGB image as PNG bytes (used by the GUI to display images)."""
    img = to_uint8(img)
    out = img[..., ::-1] if img.ndim == 3 else img
    ok, buf = cv2.imencode(".png", np.ascontiguousarray(out))
    if not ok:
        raise ValueError("PNG encoding failed")
    return buf.tobytes()


def to_float(img: np.ndarray) -> np.ndarray:
    """uint8 [0, 255] -> float32 [0, 1]. Float input is returned as float32."""
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    return img.astype(np.float32, copy=False)


def to_uint8(img: np.ndarray) -> np.ndarray:
    """float [0, 1] -> uint8 [0, 255] with rounding and clipping. uint8 is returned as is."""
    if img.dtype == np.uint8:
        return img
    return np.clip(np.round(img * 255.0), 0, 255).astype(np.uint8)


def check_rgb(img: np.ndarray, name: str = "image") -> None:
    """Raise ValueError unless `img` is an RGB uint8 array of shape (H, W, 3)."""
    if not isinstance(img, np.ndarray):
        raise ValueError(f"{name} must be a NumPy array, got {type(img).__name__}")
    if img.ndim != 3 or img.shape[2] != 3 or img.dtype != np.uint8:
        raise ValueError(f"{name} must be (H, W, 3) uint8, got {img.shape} {img.dtype}")
