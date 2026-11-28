"""OUR METHOD: extended slope handling for gentle edges (1:3, 1:4, ...).

OWNER: M4.   Tests: tests/test_module_b.py
Plan report section 4.7. STAGE 2 -- this is the main contribution of the
project, so keep debug outputs (slope maps) for the report figures.

Idea. A gentle edge in pixel art is a staircase whose steps are n pixels wide
(slope 1:n). xBR only looks at a 5x5 window, so it can follow 1:1 and ~1:2
edges but not longer steps. Here we measure the full length n of every step
(up to max_run pixels) and redraw the edge as one straight line through the
middle of each step, so a 1:4 staircase becomes a smooth 1:4 line.

All four edge orientations are handled by one routine that only knows
"color P above color Q, step runs horizontally": the image is flipped and/or
transposed to bring the other orientations into that frame.
"""
from __future__ import annotations

import numpy as np

from .similarity import yuv_distance
from .xbr import EQ_THRESHOLD, xbr

# Frames (forward, inverse). Both work on arrays of any rank >= 2.
_FRAMES = (
    (lambda a: a, lambda a: a),                                              # P above Q, horizontal run
    (lambda a: a[::-1], lambda a: a[::-1]),                                  # P below Q
    (lambda a: np.swapaxes(a, 0, 1), lambda a: np.swapaxes(a, 0, 1)),        # P left of Q, vertical run
    (lambda a: np.swapaxes(a[::-1], 0, 1), lambda a: np.swapaxes(a, 0, 1)[::-1]),  # P right of Q
)


def _eq(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return yuv_distance(a, b) < EQ_THRESHOLD


def _pack(img: np.ndarray) -> np.ndarray:
    v = img.astype(np.int64)
    return (v[..., 0] << 16) | (v[..., 1] << 8) | v[..., 2]


def _stairs(img: np.ndarray, max_run: int):
    """Horizontal steps with color P above color Q (one frame).

    A step is a maximal horizontal run of pixels whose color is P and whose
    lower neighbor is Q (P != Q). It is a staircase (not a straight line or a
    bump) when exactly one of its two ends turns the boundary.

    Returns (n, cl, rises): n (H, W) int run length (0 = not a valid step of
    length <= max_run), cl = pixels of the run to the left of this pixel,
    rises = the boundary goes up to the right.
    """
    h, w = img.shape[:2]
    below = np.concatenate([img[1:], img[-1:]], axis=0)
    edge = ~_eq(img, below)
    key = np.where(edge, (_pack(img) << 24) | _pack(below), -2)

    m = max_run + 1
    padded = np.pad(key, ((0, 0), (m, m)), constant_values=-1)
    cl = np.zeros((h, w), np.int32)
    cr = np.zeros((h, w), np.int32)
    left = np.ones((h, w), bool)
    right = np.ones((h, w), bool)
    for s in range(1, m + 1):
        left &= padded[:, m - s:m - s + w] == key
        right &= padded[:, m + s:m + s + w] == key
        cl += left
        cr += right
    n = 1 + cl + cr

    ys = np.arange(h)[:, None]
    xr = np.clip(np.arange(w)[None, :] + cr + 1, 0, w - 1)
    xl = np.clip(np.arange(w)[None, :] - cl - 1, 0, w - 1)
    ends_right = ~_eq(img[ys, xr], img)   # P stops on the right -> boundary goes up to the right
    ends_left = ~_eq(img[ys, xl], img)    # P stops on the left  -> boundary goes up to the left
    valid = edge & (n <= max_run) & (ends_right != ends_left)
    return np.where(valid, n, 0).astype(np.int32), cl, ends_right


def estimate_slopes(img: np.ndarray, max_run: int = 8) -> np.ndarray:
    """Measure the run lengths of same-colored pixels along edges over a range
    wider than xBR's 5x5 window and estimate the local edge slope.

    Returns an (H, W) float32 map: 0 = no edge, otherwise the run length n of
    a 1:n edge at that pixel (sign or a second map for direction is up to you).
    Used by upscale_slope and for visualization in the report.

    Here: +n for steps running horizontally (edge flatter than 45 degrees),
    -n for steps running vertically (steeper than 45 degrees). A pixel lies on
    the "upper" side of the edge it touches; if it touches several edges the
    longest run wins. Straight edges (run longer than max_run) are 0.
    """
    best = np.zeros(img.shape[:2], np.float32)
    for idx, (fwd, inv) in enumerate(_FRAMES):
        n = inv(_stairs(np.ascontiguousarray(fwd(img)), max_run)[0]).astype(np.float32)
        n = n if idx < 2 else -n
        better = np.abs(n) > np.abs(best)
        best = np.where(better, n, best)
    return best


def _draw_steps(out: np.ndarray, img: np.ndarray, k: int, mode: str, max_run: int) -> np.ndarray:
    """Redraw every horizontal step (P above Q) of one frame as a straight line."""
    h, w = img.shape[:2]
    n, cl, rises = _stairs(img, max_run)
    mask = n >= 2                      # n == 1 is a plain diagonal: xBR already does it
    if not mask.any():
        return out
    below = np.concatenate([img[1:], img[-1:]], axis=0)

    nf = np.maximum(n, 1).astype(np.float32)
    sub = (np.arange(k, dtype=np.float32) + 0.5) / k
    # horizontal offset of every sub-pixel from the middle of its run: (H, 1, W, k)
    dx = cl[:, None, :, None] + sub[None, None, None, :] - nf[:, None, :, None] / 2.0
    sgn = np.where(rises, 1.0, -1.0)[:, None, :, None]
    line = 1.0 - sgn * dx / nf[:, None, :, None]          # boundary height, in rows from the top of the P row
    ys = sub[None, :, None, None]

    blocks = out.reshape(h, k, w, k, 3)
    m4 = mask[:, None, :, None]
    p = img[:, None, :, None, :]
    q = below[:, None, :, None, :]

    def paint(target_rows, y_off):
        cover = np.clip((line - (ys + y_off)) * k + 0.5, 0.0, 1.0) * m4     # share of P
        if mode == "palette":
            new = np.where((cover >= 0.5)[..., None], p, q)
        else:
            new = np.rint(p.astype(np.float32) * cover[..., None]
                          + q.astype(np.float32) * (1.0 - cover[..., None])).astype(np.uint8)
        return new, m4

    # P row: the pixel itself
    new, m_ = paint(None, 0.0)
    blocks[:] = np.where(m_[..., None], new, blocks)
    # Q row: the pixel below, whose block holds the lower half of the boundary
    new, m_ = paint(None, 1.0)
    tgt = blocks[1:]
    blocks[1:] = np.where(m_[:-1][..., None], new[:-1], tgt)
    return blocks.reshape(h * k, w * k, 3)


def upscale_slope(img: np.ndarray, scale: int = 2, mode: str = "palette",
                  max_run: int = 8) -> np.ndarray:
    """Enlarge pixel art; near edges, fill the output pixels according to a
    straight line with the estimated slope (estimate_slopes) instead of a
    staircase. Elsewhere behave like xBR or nearest.

    mode : "palette" (only input colors) or "smooth" (blend along the edge)
    (H, W, 3) uint8 -> (scale*H, scale*W, 3) uint8
    A uniform image must stay uniform.
    """
    if scale not in (2, 3, 4):
        raise ValueError(f"upscale_slope supports scales 2, 3, 4, not {scale}")
    if mode not in ("palette", "smooth"):
        raise ValueError("mode must be 'palette' or 'smooth'")
    out = xbr(img, scale=scale, mode=mode)       # base: handles diagonals and ~1:2 edges
    for fwd, inv in _FRAMES:
        src = np.ascontiguousarray(fwd(img))
        cur = np.ascontiguousarray(fwd(out))
        done = _draw_steps(cur, src, scale, mode, max_run)
        out = np.ascontiguousarray(inv(done))
    return out
