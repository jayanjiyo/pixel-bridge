"""Classic rule-based pixel art scalers: Scale2x, Scale3x, Eagle.

OWNER: M4.   Tests: tests/test_module_b.py
Plan report section 4.6. STAGE 1 (scale2x, scale3x), STAGE 2 (eagle).

Neighbor names used below (core.filters.neighbor gives each one as a
whole shifted image, so the rules can be written without pixel loops):

        A            (A = up,   B = right, C = left, D = down)
      C P B
        D
"""
from __future__ import annotations

import numpy as np

from ..core.filters import neighbor


def _eq(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Exact color equality, (H, W, 3) x (H, W, 3) -> (H, W) bool."""
    return (a == b).all(axis=-1)


def _interleave(blocks, k: int, shape) -> np.ndarray:
    """Place k*k per-pixel images (row-major, each (H, W, 3)) into one image
    of size (k*H, k*W, 3): output pixel (k*y + i, k*x + j) = blocks[i*k + j][y, x]."""
    h, w = shape[:2]
    out = np.empty((k * h, k * w, 3), dtype=blocks[0].dtype)
    for idx, blk in enumerate(blocks):
        out[idx // k::k, idx % k::k] = blk
    return out


def scale2x(img: np.ndarray) -> np.ndarray:
    """Scale2x / EPX. Output 2x2 block of pixel P:

        E0 = A if (C == A and C != D and A != B) else P     (top-left)
        E1 = B if (A == B and A != C and B != D) else P     (top-right)
        E2 = C if (D == C and D != B and C != A) else P     (bottom-left)
        E3 = D if (B == D and B != A and D != C) else P     (bottom-right)

    Equality is exact color equality. (H, W, 3) uint8 -> (2H, 2W, 3) uint8
    """
    p = img
    a, b = neighbor(img, -1, 0), neighbor(img, 0, 1)
    c, d = neighbor(img, 0, -1), neighbor(img, 1, 0)
    ca, cd, ab, bd = _eq(c, a), _eq(c, d), _eq(a, b), _eq(b, d)
    e0 = np.where((ca & ~cd & ~ab)[..., None], a, p)
    e1 = np.where((ab & ~ca & ~bd)[..., None], b, p)
    e2 = np.where((cd & ~bd & ~ca)[..., None], c, p)
    e3 = np.where((bd & ~ab & ~cd)[..., None], d, p)
    return _interleave([e0, e1, e2, e3], 2, img.shape)


def scale3x(img: np.ndarray) -> np.ndarray:
    """Scale3x (3x3 version of the rules, see scale2x.it).
    (H, W, 3) uint8 -> (3H, 3W, 3) uint8"""
    # 3x3 neighborhood  A B C / D E F / G H I  (E = center pixel)
    e = img
    a, b, c = neighbor(img, -1, -1), neighbor(img, -1, 0), neighbor(img, -1, 1)
    d, f = neighbor(img, 0, -1), neighbor(img, 0, 1)
    g, h, i = neighbor(img, 1, -1), neighbor(img, 1, 0), neighbor(img, 1, 1)

    db, dh, bf, fh = _eq(d, b), _eq(d, h), _eq(b, f), _eq(f, h)
    ne = lambda x: ~_eq(e, x)          # E != X
    # the four "diagonal edge" conditions (one per corner of the center pixel)
    tl = db & ~dh & ~bf                # edge around the top-left corner
    tr = bf & ~db & ~fh                # top-right
    bl = dh & ~fh & ~db                # bottom-left
    br = fh & ~bf & ~dh                # bottom-right

    def pick(cond, color):
        return np.where(cond[..., None], color, e)

    e0 = pick(tl, d)
    e1 = pick((tl & ne(c)) | (tr & ne(a)), b)
    e2 = pick(tr, f)
    e3 = pick((bl & ne(a)) | (tl & ne(g)), d)
    e4 = e
    e5 = pick((tr & ne(i)) | (br & ne(c)), f)
    e6 = pick(bl, d)
    e7 = pick((br & ne(g)) | (bl & ne(i)), h)
    e8 = pick(br, f)
    return _interleave([e0, e1, e2, e3, e4, e5, e6, e7, e8], 3, img.shape)


def eagle(img: np.ndarray) -> np.ndarray:
    """Eagle: a corner of the 2x2 output block takes the color of the three
    neighbors around that corner (e.g. up, up-left, left) when all three are
    equal; otherwise it stays P. (H, W, 3) uint8 -> (2H, 2W, 3) uint8"""
    p = img
    s_, t, u = neighbor(img, -1, -1), neighbor(img, -1, 0), neighbor(img, -1, 1)
    v, w = neighbor(img, 0, -1), neighbor(img, 0, 1)
    x, y, z = neighbor(img, 1, -1), neighbor(img, 1, 0), neighbor(img, 1, 1)
    e0 = np.where((_eq(v, s_) & _eq(s_, t))[..., None], s_, p)   # up-left corner
    e1 = np.where((_eq(t, u) & _eq(u, w))[..., None], u, p)      # up-right corner
    e2 = np.where((_eq(v, x) & _eq(x, y))[..., None], x, p)      # down-left corner
    e3 = np.where((_eq(w, z) & _eq(z, y))[..., None], z, p)      # down-right corner
    return _interleave([e0, e1, e2, e3], 2, img.shape)
