"""Pixel backgrounds and compositing.

- pixel_background: the user's own landmark photo through Module A
  (no outline, fewer colors) -> a pixel-art background.
- default_background: a built-in Namsan / N Seoul Tower scene drawn in code.
- compose_scene: put the character on the background; optionally re-map the
  character's colors to the background palette so the two look like one picture.
"""
from __future__ import annotations

import numpy as np

from ..core.color import rgb_to_lab
from ..module_a.pipeline import PixelArtParams, photo_to_pixel_art
from .sprite import Character


def pixel_background(photo: np.ndarray, width: int = 128, n_colors: int = 24) -> tuple[np.ndarray, np.ndarray]:
    """Landmark photo -> (pixel background (h, width, 3) uint8, its palette (K, 3) uint8)."""
    r = photo_to_pixel_art(photo, PixelArtParams(out_width=width, n_colors=n_colors,
                                                 outline=False, cleanup=True))
    return r.image, r.palette


def default_background(width: int = 128, height: int = 96) -> np.ndarray:
    """Built-in scene: dusk sky, Namsan hill with N Seoul Tower, stone plaza."""
    t = np.arange(height)[:, None] / height
    sky = np.array([255, 170, 120]) * (1 - t) + np.array([120, 110, 190]) * t
    sky = np.round(sky / 24) * 24                          # color bands like pixel art
    bg = np.repeat(sky[:, None, :], width, axis=1)
    x = np.arange(width)
    tx = int(width * 0.69)
    hill = (height * 0.73 - height * 0.23 * np.exp(-((x - tx) / (width * 0.2)) ** 2)).astype(int)
    for i in range(width):
        bg[hill[i]:, i] = (70, 120, 80)
        bg[hill[i]:hill[i] + 2, i] = (95, 150, 95)
    top = hill[tx]
    bg[top - 30:top, tx - 1:tx + 1] = (230, 230, 235)              # tower
    bg[top - 26:top - 22, tx - 3:tx + 3] = (240, 240, 245)
    bg[top - 25:top - 23, tx - 3:tx + 3] = (200, 90, 90)
    bg[top - 38:top - 30, tx] = (220, 220, 225)
    bg[top - 39, tx] = (255, 80, 80)
    for sx, sy in ((20, 12), (40, 6), (60, 18), (110, 10), (8, 30)):
        if sx < width and sy < height:
            bg[sy, sx] = (255, 240, 200)
    ground = int(height * 0.9)
    bg[ground:, :] = (150, 140, 130)
    bg[ground, :] = (120, 110, 100)
    bg[ground:, ::8] = (135, 125, 115)
    return np.clip(bg, 0, 255).astype(np.uint8)


def _to_palette(img: np.ndarray, palette: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Nearest palette color in CIELAB for every pixel except `keep` (eyes, highlights)."""
    lab = rgb_to_lab(img.reshape(-1, 3))
    pal = rgb_to_lab(palette)
    idx = ((lab[:, None, :] - pal[None]) ** 2).sum(-1).argmin(1)
    out = palette[idx].reshape(img.shape)
    out[keep] = img[keep]
    return out


def compose_scene(background: np.ndarray, character: Character, x: float = 0.25,
                  ground: float = 0.92, palette: np.ndarray | None = None) -> np.ndarray:
    """Place the character with its feet at (x * width, ground * height).

    palette: if given, the character is re-colored with it (shared palette),
    except near-black and white pixels so the eyes keep their contrast.
    """
    scene = background.copy()
    img, mask = character.image, character.mask
    if palette is not None:
        L = rgb_to_lab(img)[..., 0]
        img = _to_palette(img, palette, keep=(L < 25) | (L > 95))
    h, w = img.shape[:2]
    H, W = scene.shape[:2]
    top = int(round(ground * H)) - h
    left = int(round(x * W)) - w // 2
    ys, xs = slice(max(0, top), min(H, top + h)), slice(max(0, left), min(W, left + w))
    sub = (slice(ys.start - top, ys.stop - top), slice(xs.start - left, xs.stop - left))
    region = scene[ys, xs]
    m = mask[sub]
    region[m] = img[sub][m]
    return scene
