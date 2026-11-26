"""Module A step 6: cleanup rules used by pixel artists.

OWNER: M3.   Tests: tests/test_module_a_back.py
Plan report section 4.5.

Both functions work on index images (H, W) int32 and return a new index
image of the same shape; they never add new palette entries.
"""
from __future__ import annotations

import numpy as np

from ..core.filters import neighbor

_EIGHT = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)]


def remove_isolated(indices: np.ndarray) -> np.ndarray:
    """A pixel whose index differs from ALL eight neighbors is replaced by
    the most common neighbor index (mode filter). Image borders count the
    replicated edge as neighbors (core.filters.neighbor).

    Ties between equally common neighbors go to the first one in the order
    up-left, up, up-right, left, right, down-left, down, down-right.
    Decisions use the input image only (one pass, no chain reaction).
    """
    idx = np.asarray(indices, dtype=np.int32)
    nbs = np.stack([neighbor(idx, dy, dx) for dy, dx in _EIGHT])     # (8, H, W)
    isolated = (nbs != idx).all(axis=0)
    if not isolated.any():
        return idx.copy()
    counts = (nbs[:, None] == nbs[None, :]).sum(axis=1)              # (8, H, W)
    mode = np.take_along_axis(nbs, counts.argmax(axis=0)[None], axis=0)[0]
    return np.where(isolated, mode, idx).astype(np.int32)


def fix_one_pixel_lines(indices: np.ndarray) -> np.ndarray:
    """Make diagonal lines exactly one pixel wide.

    In a diagonal line, a pixel that makes an L-shaped corner is redundant:
    it has a same-color neighbor horizontally AND vertically, the fourth
    pixel of that 2x2 square has another color, and the line does not go
    on straight through it. Replace it with the surrounding (background)
    color; the line stays 8-connected.

        X . . .        X . . .
        . X X .   ->   . X . .        (45-degree line with an extra pixel)
        . . X .        . . X .

    Must NOT change: filled areas, straight lines, and the right-angle
    corners of rectangles (there the line continues straight on both sides).

    Exact rule for a pixel P of color c, for each corner direction (dy, dx):
      1. arms:      P + (0, dx) == c and P + (dy, 0) == c
      2. open 2x2:  P + (dy, dx) != c
      3. no straight run through P and nothing hanging on its outer side:
                    P - (0, dx), P - (dy, 0), P - (dy, dx) all != c
         (so removing P cannot disconnect anything: the two arms touch
         diagonally, and every other neighbor of P touches an arm)
      4. not a rectangle corner: NOT (both arms continue straight,
                    P + (0, 2dx) == c and P + (2dy, 0) == c)
    P gets the most common color of its three outer pixels from rule 3.

    Two-pixel-wide diagonals: there EVERY middle pixel passes the rule, and
    removing them all at once would cut the line. So each round removes only
    candidates that are more than 2 pixels apart (first in raster order wins),
    and rounds repeat until nothing changes. The result is a 1-px diagonal.
    """
    out = np.asarray(indices, dtype=np.int32).copy()
    h, w = out.shape
    for _ in range(_MAX_ROUNDS):
        redundant, background = _l_corners(out)
        if not redundant.any():
            break
        blocked = np.zeros((h, w), bool)
        for y, x in zip(*np.nonzero(redundant)):          # few pixels: loop is fine
            if blocked[y, x]:
                continue
            out[y, x] = background[y, x]
            blocked[max(y - 2, 0): y + 3, max(x - 2, 0): x + 3] = True
    return out


_MAX_ROUNDS = 16


def _l_corners(idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mask of redundant L-corner pixels (rules 1-4 of fix_one_pixel_lines)
    and the background color each of them should get."""
    def at(dy, dx):
        return neighbor(idx, dy, dx)

    redundant = np.zeros(idx.shape, bool)
    background = idx.copy()
    for dy in (-1, 1):
        for dx in (-1, 1):
            h_out, v_out, d_out = at(0, -dx), at(-dy, 0), at(-dy, -dx)
            here = ((at(0, dx) == idx) & (at(dy, 0) == idx)
                    & (at(dy, dx) != idx)
                    & (h_out != idx) & (v_out != idx) & (d_out != idx)
                    & ~((at(0, 2 * dx) == idx) & (at(2 * dy, 0) == idx))
                    & ~redundant)
            bg = np.where((h_out == v_out) | (h_out == d_out), h_out,
                          np.where(v_out == d_out, v_out, h_out))
            background = np.where(here, bg, background)
            redundant |= here
    return redundant, background
