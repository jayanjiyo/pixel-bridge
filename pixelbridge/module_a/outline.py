"""Module A step 4: one-pixel outlines.

OWNER: M3.   Tests: tests/test_module_a_back.py
Plan report section 4.5.
"""
from __future__ import annotations

import numpy as np

from ..core.color import rgb_to_gray, rgb_to_lab
from ..core.filters import neighbor

# A darker color should be at least this much darker (CIELAB L*) to be seen
# as an outline. If no palette color is that dark, the darkest one is used.
MIN_L_DROP = 15.0
# An outline color must have a similar hue (a*b* distance at most HUE_TOL) or
# be nearly gray (chroma at most NEUTRAL_CHROMA): a dark gray outline suits
# any color, but e.g. a red outline around green grass looks wrong.
HUE_TOL = 40.0
NEUTRAL_CHROMA = 15.0


def darker_index(palette: np.ndarray) -> np.ndarray:
    """For each palette color, the index of a darker palette color to use
    for outlines (e.g. the darkest color with a similar hue, or simply the
    nearest color with lower CIELAB L*). The darkest color maps to itself.

    Our rule: candidates are the colors that are darker in BOTH CIELAB L* and
    BT.601 luminance, and that have a similar hue (a*b* distance <= HUE_TOL)
    or are nearly gray (chroma <= NEUTRAL_CHROMA). Among the candidates that
    are clearly darker (L* lower by at least MIN_L_DROP) we take the one
    nearest in CIELAB, so the outline keeps the hue of the object ("a darker
    shade of the same color"). If no candidate is clearly darker we take the
    darkest candidate; if there is no candidate at all the color maps to
    itself (that color gets no outline).

    palette (K, 3) uint8 -> (K,) int32
    """
    lab = rgb_to_lab(palette).astype(np.float64)
    L = lab[:, 0]
    luma = rgb_to_gray(palette).astype(np.float64)
    dist = np.sqrt(np.sum((lab[:, None, :] - lab[None, :, :]) ** 2, axis=-1))
    ab_dist = np.sqrt(np.sum((lab[:, None, 1:] - lab[None, :, 1:]) ** 2, axis=-1))
    neutral = np.hypot(lab[:, 1], lab[:, 2]) <= NEUTRAL_CHROMA

    out = np.arange(len(palette), dtype=np.int32)
    for i in range(len(palette)):
        darker = (L < L[i]) & (luma <= luma[i]) & ((ab_dist[i] <= HUE_TOL) | neutral)
        if not darker.any():
            continue
        clear = darker & (L <= L[i] - MIN_L_DROP)
        if clear.any():
            cand = np.flatnonzero(clear)
            out[i] = cand[np.argmin(dist[i, cand])]
        else:
            cand = np.flatnonzero(darker)
            out[i] = cand[np.argmin(L[cand])]
    return out


def add_outline(indices: np.ndarray, palette: np.ndarray, threshold: float = 25.0) -> np.ndarray:
    """Draw a one-pixel dark outline where neighboring colors differ a lot.

    For each pair of 4-neighbors whose CIELAB distance is above `threshold`,
    replace ONE side (e.g. the brighter pixel, or the pixel of the object
    rather than the background -- decide and document) with
    darker_index(palette)[...]. Only palette colors may be used, so the
    result is again a valid index image. A uniform image must not change.

    Our decision: the DARKER side of the pair (lower L*) gets the outline.
    - Changing the darker pixel to an even darker color is the smaller color
      change (lower Delta E to the photo) than turning a bright pixel dark.
    - The bright region keeps its shape and size; the outline sits on the
      inner rim of the darker shape.
    If both sides have the same L*, the pixel with the smaller palette index
    is chosen, so exactly one side changes. All decisions use the INPUT
    image (no chain reaction).

    indices (H, W) int32, palette (K, 3) uint8 -> indices (H, W) int32
    """
    idx = np.asarray(indices, dtype=np.int32)
    lab = rgb_to_lab(palette).astype(np.float64)
    L = lab[:, 0]
    dist = np.sqrt(np.sum((lab[:, None, :] - lab[None, :, :]) ** 2, axis=-1))
    darker = darker_index(palette)

    mark = np.zeros(idx.shape, bool)
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nb = neighbor(idx, dy, dx)              # replicate border: same color
        strong = dist[idx, nb] > threshold
        this_side = (L[idx] < L[nb]) | ((L[idx] == L[nb]) & (idx < nb))
        mark |= strong & this_side
    return np.where(mark, darker[idx], idx).astype(np.int32)
