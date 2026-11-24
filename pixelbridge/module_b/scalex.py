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


def scale2x(img: np.ndarray) -> np.ndarray:
    """Scale2x / EPX. Output 2x2 block of pixel P:

        E0 = A if (C == A and C != D and A != B) else P     (top-left)
        E1 = B if (A == B and A != C and B != D) else P     (top-right)
        E2 = C if (D == C and D != B and C != A) else P     (bottom-left)
        E3 = D if (B == D and B != A and D != C) else P     (bottom-right)

    Equality is exact color equality. (H, W, 3) uint8 -> (2H, 2W, 3) uint8
    """
    raise NotImplementedError("TODO(M4): scale2x")


def scale3x(img: np.ndarray) -> np.ndarray:
    """Scale3x (3x3 version of the rules, see scale2x.it).
    (H, W, 3) uint8 -> (3H, 3W, 3) uint8"""
    raise NotImplementedError("TODO(M4): scale3x")


def eagle(img: np.ndarray) -> np.ndarray:
    """Eagle: a corner of the 2x2 output block takes the color of the three
    neighbors around that corner (e.g. up, up-left, left) when all three are
    equal; otherwise it stays P. (H, W, 3) uint8 -> (2H, 2W, 3) uint8"""
    raise NotImplementedError("TODO(M4): eagle")
