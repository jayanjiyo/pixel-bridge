"""xBR scaler (levels 1 and 2).

OWNER: M4.   Tests: tests/test_module_b.py
Plan report section 4.6. STAGE 2.
"""
from __future__ import annotations

import numpy as np

from .similarity import yuv_distance

# Two colors are "equal" when their weighted YUV distance is below this value
# (the threshold of the original xBR implementation, which uses the same
# 48 / 7 / 6 weights as similarity.YUV_WEIGHTS).
EQ_THRESHOLD = 155.0

# Edge types found by the corner test
_DIA, _SHALLOW, _STEEP = 0, 1, 2   # 1:1, ~1:2 (hugs the bottom row), ~2:1 (hugs the right column)


def _kernels(k: int) -> np.ndarray:
    """Blend weights (3, k, k) of the bottom-right corner for each edge type.

    Output sub-pixel (i, j) of a k x k block is at (x, y) = ((j+.5)/k, (i+.5)/k);
    its distance to the corner is t = sx*(1-x) + sy*(1-y) and the weight of the
    neighbor color is clip(1 - t). sx / sy < 1 stretches the filled area along
    that axis, which turns the 1:1 diagonal into a ~1:2 or ~2:1 slope.
    """
    pos = (np.arange(k) + 0.5) / k
    dx = (1.0 - pos)[None, :]
    dy = (1.0 - pos)[:, None]
    out = np.empty((3, k, k), np.float32)
    for t, (sx, sy) in enumerate(((1.0, 1.0), (0.5, 1.0), (1.0, 0.5))):
        out[t] = np.clip(1.0 - (sx * dx + sy * dy), 0.0, 1.0)
    return out


def _corner(img: np.ndarray, level2: bool):
    """xBR edge test for the bottom-right corner of every pixel.

    Returns (mask (H, W) bool, kind (H, W) int, color (H, W, 3) uint8):
    where mask is True an edge ends at that corner, `kind` is the slope type
    and `color` is the neighbor color (right or below) to spread along the edge.
    """
    h, w = img.shape[:2]
    p = np.pad(img, ((2, 2), (2, 2), (0, 0)), mode="edge")

    def at(dy, dx):
        return p[2 + dy:2 + dy + h, 2 + dx:2 + dx + w]

    # 5x5 neighborhood (rows y-2..y+2, columns x-2..x+2); E = center
    b, c, d, e, f, g, hh, i = (at(-1, 0), at(-1, 1), at(0, -1), at(0, 0),
                               at(0, 1), at(1, -1), at(1, 0), at(1, 1))
    f4, i4, h5, i5 = at(0, 2), at(1, 2), at(2, 0), at(2, 1)

    def dist(x, y):
        return yuv_distance(x, y)

    def eq(x, y):
        return dist(x, y) < EQ_THRESHOLD

    not_flat = ~eq(e, hh) & ~eq(e, f)
    e_sum = dist(e, c) + dist(e, g) + dist(i, h5) + dist(i, f4) + 4.0 * dist(hh, f)
    i_sum = dist(hh, d) + dist(hh, i5) + dist(f, i4) + dist(f, b) + 4.0 * dist(e, i)
    extra = ((~eq(f, b) & ~eq(hh, d))
             | (eq(e, i) & ~eq(f, i4) & ~eq(hh, i5))
             | eq(e, g) | eq(e, c))
    mask = not_flat & (e_sum < i_sum) & extra

    kind = np.full((h, w), _DIA, np.uint8)
    if level2:
        ke, ki = dist(f, g), dist(hh, c)
        ex2 = ~eq(e, c) & ~eq(b, c)
        ex3 = ~eq(e, g) & ~eq(d, g)
        shallow = (ke * 2.0 <= ki) & ex3
        steep = (ke >= ki * 2.0) & ex2 & ~shallow
        kind[shallow] = _SHALLOW
        kind[steep] = _STEEP

    use_f = dist(e, f) <= dist(e, hh)
    color = np.where(use_f[..., None], f, hh)
    return mask, kind, color


def xbr(img: np.ndarray, scale: int = 2, mode: str = "palette", level2: bool = True) -> np.ndarray:
    """xBR: for each corner of a pixel, compare the YUV color differences
    (similarity.yuv_distance) along the two diagonal directions in the 5x5
    neighborhood; if an edge is detected, fill the corner of the output block
    along that edge. Level 2 also detects edges with a slope of about 1:2.

    scale : 2, 3 or 4
    mode  : "palette" -> output uses only colors of the input (crisp)
            "smooth"  -> blend colors along the edge (anti-aliased)
    (H, W, 3) uint8 -> (scale*H, scale*W, 3) uint8
    A uniform image must stay uniform.
    """
    if scale not in (2, 3, 4):
        raise ValueError(f"xbr supports scales 2, 3, 4, not {scale}")
    if mode not in ("palette", "smooth"):
        raise ValueError("mode must be 'palette' or 'smooth'")
    h, w = img.shape[:2]
    kern = _kernels(scale)
    out = np.repeat(np.repeat(img, scale, axis=0), scale, axis=1)

    # The test is written for the bottom-right corner only; rotate the image
    # (and the output built so far) by 90 degrees to handle the other three.
    for r in range(4):
        src = np.rot90(img, r)
        cur = np.ascontiguousarray(np.rot90(out, r))
        mask, kind, color = _corner(np.ascontiguousarray(src), level2)
        hr, wr = src.shape[:2]
        # per-pixel kernel (Hr, k, Wr, k); zero where there is no edge
        wgt = kern[kind].transpose(0, 2, 1, 3) * mask[:, None, :, None]
        blocks = cur.reshape(hr, scale, wr, scale, 3)
        col = color[:, None, :, None, :]
        if mode == "palette":
            blocks = np.where((wgt >= 0.5)[..., None], col, blocks)
        else:
            mixed = blocks.astype(np.float32) * (1.0 - wgt[..., None]) + col * wgt[..., None]
            blocks = np.rint(mixed).astype(np.uint8)
        out = np.ascontiguousarray(np.rot90(blocks.reshape(hr * scale, wr * scale, 3), -r))
    return out
