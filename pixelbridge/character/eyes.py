"""Find the two eyes in a front-facing portrait, without using skin color.

Skin color alone is not reliable: light hair, beige walls, the neck and the
chest all pass a skin threshold, so a head box built from it is often wrong.
The eyes are a better anchor. Everything else (face width, hairline, hair,
eyebrows, mouth) is then measured at known positions relative to them
(see features.measure_from_eyes).

Method
1. Dark blobs at several sizes: local maxima of a difference of Gaussians
   (DoG) of the gray image. Eyes, eyebrows, nostrils and mouth corners all
   give such blobs, and so do gaps in the hair.
2. Pairs of blobs at about the same height. A pair is kept if the two patches
   are mirror images of each other (normalized cross-correlation of one patch
   with the flipped other patch) and the bridge between them is brighter.
3. Face structure decides which pair is the eyes:
   - the cheeks below the pair are smooth (low gradient) and brighter than the
     blobs (hair and clothes are textured);
   - a similar pair just above it (the eyebrows) or a narrower pair below it
     (nostrils or mouth corners) must exist, and both add to the score;
   - if a skin-colored region exists, the pair lies in its upper part and the
     cheeks are skin-colored.
4. Two attempts. The first uses only strong blobs anywhere in the image. If it
   finds nothing (a small face: the eyes are weak blobs and hair or clothes fill
   the list), the second looks for weaker blobs near skin only, and its result
   is accepted only with a very high score.
"""
from __future__ import annotations

import numpy as np

from ..core import resize
from ..core.color import rgb_to_gray
from ..core.filters import SOBEL_MAX, filter2d, gaussian_kernel, gradient_magnitude, neighbor
from .detect import largest_component, skin_mask

WORK = 320                              # longer side of the working image
SCALES = (1.2, 1.7, 2.4, 3.4, 4.8)      # blob sizes (sigma, in working pixels)
# (weakest DoG response, largest eye-line slope, blobs near skin only, lowest score accepted)
ATTEMPTS = ((0.020, 0.3, False, 0.9),
            (0.012, 0.4, True, 2.4))


def _blur(x: np.ndarray, sigma: float) -> np.ndarray:
    size = int(2 * round(3 * sigma) + 1)
    return filter2d(x, gaussian_kernel(size, sigma)).astype(np.float64)


def _dark_blobs(gray: np.ndarray, min_response: float, where: np.ndarray | None = None,
                limit: int = 250) -> list[tuple[float, int, int, float]]:
    """(response, y, x, sigma) of the strongest dark blobs, only inside `where` if given."""
    out = []
    for sigma in SCALES:
        dog = _blur(gray, sigma * 1.6) - _blur(gray, sigma)       # > 0 on dark blobs
        peak = dog > min_response
        if where is not None:
            peak &= where
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy or dx:
                    peak &= dog >= neighbor(dog, dy, dx)
        ys, xs = np.nonzero(peak)
        out += [(float(dog[y, x]), int(y), int(x), sigma) for y, x in zip(ys, xs)]
    out.sort(reverse=True)
    return out[:limit]


def _symmetric_pairs(gray: np.ndarray, blobs, max_tilt: float) -> list[dict]:
    h, w = gray.shape
    pairs = []
    for i in range(len(blobs)):
        _, y1, x1, s1 = blobs[i]
        for j in range(i + 1, len(blobs)):
            _, y2, x2, s2 = blobs[j]
            if x1 == x2 or max(s1, s2) / min(s1, s2) > 2.1:
                continue
            (ly, lx), (ry, rx) = ((y1, x1), (y2, x2)) if x1 < x2 else ((y2, x2), (y1, x1))
            dx = rx - lx
            sigma = (s1 + s2) / 2
            if abs(ry - ly) > max_tilt * dx or not (4 * sigma <= dx <= 30 * sigma) or dx < 8:
                continue
            r = int(max(3, 0.3 * dx))
            if min(ly, ry) - r < 0 or max(ly, ry) + r >= h or lx - r < 0 or rx + r >= w:
                continue
            left = gray[ly - r:ly + r + 1, lx - r:lx + r + 1]
            right = gray[ry - r:ry + r + 1, rx - r:rx + r + 1][:, ::-1]      # mirrored
            a, b = left - left.mean(), right - right.mean()
            ncc = float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-9))
            bridge = float(gray[(ly + ry) // 2, (lx + rx) // 2] - (gray[ly, lx] + gray[ry, rx]) / 2)
            if ncc < 0.45 or bridge < 0.03:
                continue
            pairs.append(dict(ly=ly, lx=lx, ry=ry, rx=rx, dx=dx, ncc=ncc, bridge=bridge,
                              my=(ly + ry) / 2, mx=(lx + rx) / 2))
    return pairs


def find_eyes(photo: np.ndarray) -> dict | None:
    """Eyes of a front-facing portrait.

    Returns {"left": (x, y), "right": (x, y), "score": float} in photo pixels
    ("left" = left in the image), or None when no trustworthy pair is found.
    """
    h, w = photo.shape[:2]
    s = min(1.0, WORK / max(h, w))                              # small photos are used as they are
    small = photo if s == 1.0 else resize.resize_area(photo, max(1, round(h * s)), max(1, round(w * s)))
    gray = rgb_to_gray(small).astype(np.float64)
    hh, ww = gray.shape
    skin = skin_mask(small)
    skin_frac = _blur(skin.astype(np.float64), 2.5)
    busy_map = _blur(gradient_magnitude(gray.astype(np.float32)) / SOBEL_MAX, 1.5)
    region = None                       # upper part of the largest skin region, if there is one
    comp = largest_component(skin)
    if comp.sum() > 0.01 * comp.size:
        cy, cx = np.nonzero(comp)
        region = (cy.min(), cy.min() + 0.75 * (cy.max() - cy.min() + 1), cx.min(), cx.max())

    for min_response, max_tilt, near_skin_only, min_score in ATTEMPTS:
        if near_skin_only and region is None:
            continue
        where = (_blur(skin.astype(np.float64), 4.0) > 0.15) if near_skin_only else None
        pairs = _symmetric_pairs(gray, _dark_blobs(gray, min_response, where), max_tilt)
        best, best_score = None, -1e9
        for p in pairs:
            dx = p["dx"]
            if region is not None and not (region[0] <= p["my"] <= region[1] and region[2] <= p["mx"] <= region[3]):
                continue
            y0, y1 = int(p["my"] + 0.25 * dx), int(min(hh, p["my"] + 0.75 * dx))
            if y1 <= y0:
                continue
            q = int(max(1, 0.25 * dx))
            cols = [slice(max(0, int(x) - q), int(x) + q + 1) for x in (p["lx"], p["rx"])]
            busy = float(np.mean([busy_map[y0:y1, c].mean() for c in cols]))
            cheek_lum = float(np.mean([gray[y0:y1, c].mean() for c in cols]))
            eye_lum = (gray[p["ly"], p["lx"]] + gray[p["ry"], p["rx"]]) / 2
            yb = int(min(hh - 1, p["my"] + 0.5 * dx))
            cheek_skin = float((skin_frac[yb, p["lx"]] + skin_frac[yb, p["rx"]]) / 2)
            if busy > 0.10 or cheek_lum - eye_lum < 0.05 or (region is not None and cheek_skin < 0.3):
                continue
            brow = lower = 0.0
            for o in pairs:
                if o is p or abs(o["mx"] - p["mx"]) > 0.25 * dx:
                    continue
                gap = o["my"] - p["my"]
                if -0.65 * dx < gap < -0.15 * dx and 0.75 < o["dx"] / dx < 1.6:      # eyebrows above
                    brow = max(brow, o["ncc"])
                elif 0.4 * dx < gap < 1.5 * dx and o["dx"] < 0.9 * dx:              # nostrils / mouth below
                    lower = max(lower, o["ncc"])
            if brow == 0.0 and lower == 0.0:
                continue                    # two symmetric dots alone are not a face
            score = p["ncc"] + brow + 0.6 * lower + min(p["bridge"], 0.3) + 0.5 * cheek_skin - 6 * busy
            if score > best_score:
                best, best_score = p, score
        if best is not None and best_score >= min_score:
            return {"left": (best["lx"] / s, best["ly"] / s), "right": (best["rx"] / s, best["ry"] / s),
                    "score": float(best_score)}
    return None


# ---------------------------------------------------------------- face layout search
# Where things are in a human face, in eye distances from the middle between the
# eyes: (dy, dx, height, width) of a box. Eyes and mouth are darker than the skin
# next to them; the bridge of the nose, the cheeks and the chin are skin.
LAYOUT = {
    "eye_l": (0.0, -0.5, 0.22, 0.40), "eye_r": (0.0, 0.5, 0.22, 0.40),
    "bridge": (0.05, 0.0, 0.25, 0.22),
    "cheek_l": (0.6, -0.55, 0.30, 0.40), "cheek_r": (0.6, 0.55, 0.30, 0.40),
    "mouth": (1.15, 0.0, 0.20, 0.60),
    "chin": (1.6, 0.0, 0.20, 0.50),
}
LAYOUT_SIZES = (14, 17, 21, 26, 32, 39, 48, 59, 72, 88, 108)      # eye distances tried (working pixels)
LAYOUT_MIN_SCORE = 0.30


def _box_mean(img: np.ndarray, bh: int, bw: int) -> np.ndarray:
    """Mean of the bh x bw box centered on every pixel (integral image, replicate border)."""
    bh, bw = max(1, bh) | 1, max(1, bw) | 1                         # odd sizes
    ry, rx = bh // 2, bw // 2
    padded = np.pad(img, ((ry + 1, ry), (rx + 1, rx)), mode="edge")
    ii = padded.cumsum(0).cumsum(1)
    h, w = img.shape
    total = ii[bh:bh + h, bw:bw + w] - ii[:h, bw:bw + w] - ii[bh:bh + h, :w] + ii[:h, :w]
    return total / (bh * bw)


def _fits_face_width(skin_row: np.ndarray, x: int, d: int) -> bool:
    """True if the skin run around column x (gaps up to d/4 bridged) is 1/0.62 .. 1/0.28 eye distances wide."""
    n, gap = len(skin_row), max(2, d // 4)
    left = right = x
    miss = 0
    for c in range(x, -1, -1):
        if skin_row[c]:
            left, miss = c, 0
        else:
            miss += 1
            if miss > gap:
                break
    miss = 0
    for c in range(x, n):
        if skin_row[c]:
            right, miss = c, 0
        else:
            miss += 1
            if miss > gap:
                break
    width = right - left + 1
    return 0.28 * width <= d <= 0.62 * width


def find_eyes_by_layout(photo: np.ndarray) -> dict | None:
    """Eyes from the layout of a face, for photos where the eyes are not clear dark blobs
    (light eyes, flash, a small or slightly turned face).

    For every position and several sizes, the mean brightness of the LAYOUT boxes is
    compared: each eye must be darker than the cheek below it, the bridge of the nose
    brighter than the eyes, the mouth darker than the cheeks and the chin. Left and
    right must look alike. The best position gives the eye boxes; inside each box the
    darkest spot is the eye. Same return value as find_eyes.
    """
    h, w = photo.shape[:2]
    s = min(1.0, WORK / max(h, w))
    small = photo if s == 1.0 else resize.resize_area(photo, max(1, round(h * s)), max(1, round(w * s)))
    gray = rgb_to_gray(small).astype(np.float64)
    hh, ww = gray.shape
    skin = skin_mask(small).astype(np.float64)
    use_skin = skin.mean() > 0.01
    skin_bool = skin > 0.5
    busy_map = gradient_magnitude(gray.astype(np.float32)).astype(np.float64) / SOBEL_MAX
    best = None
    for d in LAYOUT_SIZES:
        if d * 2.4 > ww or d * 3.0 > hh:
            break

        def box(name: str, src: np.ndarray = gray) -> np.ndarray:
            dy, dx, bh, bw = LAYOUT[name]
            return neighbor(_box_mean(src, round(bh * d), round(bw * d)), round(dy * d), round(dx * d))
        el, er, br = box("eye_l"), box("eye_r"), box("bridge")
        cl, cr, mo, ch = box("cheek_l"), box("cheek_r"), box("mouth"), box("chin")
        eye_c = np.minimum(cl - el, cr - er)
        bridge_c = br - (el + er) / 2
        mouth_c = (cl + cr) / 2 - mo
        score = (eye_c + 0.7 * bridge_c + 0.5 * mouth_c + 0.3 * (ch - mo)
                 - np.abs(el - er) - np.abs(cl - cr))
        ok = (eye_c > 0.04) & (bridge_c > 0.02) & (mouth_c > 0.0)
        ok &= (box("cheek_l", busy_map) + box("cheek_r", busy_map)) / 2 < 0.08      # cheeks are smooth
        if use_skin:
            ok &= np.minimum(box("cheek_l", skin), box("cheek_r", skin)) > 0.5
        m = int(1.3 * d)                                               # the whole layout must be inside the image
        ok[:int(0.4 * d)] = False
        ok[hh - int(1.8 * d):] = False
        ok[:, :m] = False
        ok[:, ww - m:] = False
        score = np.where(ok, score, -np.inf)
        for _ in range(8):                                             # best positions of this size, one by one
            y, x = np.unravel_index(np.argmax(score), score.shape)
            if not np.isfinite(score[y, x]) or (best is not None and score[y, x] <= best[0]):
                break
            # size check: the eye distance is 0.28 .. 0.62 of the face width at cheek height,
            # otherwise the "face" is one eye with its eyebrow, or two people
            row = skin_bool[min(hh - 1, y + round(0.6 * d))] if use_skin else None
            if row is None or _fits_face_width(row, x, d):
                best = (float(score[y, x]), int(y), int(x), d)
                break
            score[max(0, y - d // 2):y + d // 2 + 1, max(0, x - d // 2):x + d // 2 + 1] = -np.inf
    if best is None or best[0] < LAYOUT_MIN_SCORE:
        return None
    sc, y, x, d = best
    smooth = _blur(gray, max(1.0, d / 12))
    eyes = []
    for side in (-1, 1):                                               # center of the dark part of each eye box
        y0, y1 = max(0, y - round(0.2 * d)), min(hh, y + round(0.2 * d) + 1)
        x0, x1 = max(0, x + round((side * 0.5 - 0.35) * d)), min(ww, x + round((side * 0.5 + 0.35) * d) + 1)
        patch = smooth[y0:y1, x0:x1]
        dark = patch <= patch.min() + 0.25 * (patch.max() - patch.min())   # one darkest pixel would be an eye corner
        py, px = np.nonzero(dark)
        eyes.append(((x0 + px.mean()) / s, (y0 + py.mean()) / s))
    return {"left": eyes[0], "right": eyes[1], "score": sc}


# ---------------------------------------------------------------- tilted heads
TILTS = (18.0, -18.0)                   # the search is repeated on the photo rotated by these angles
TILT_MARGIN = 0.8                       # a rotated copy must beat the upright score by this much


def rotate(img: np.ndarray, degrees: float, center: tuple[float, float] | None = None) -> np.ndarray:
    """Rotate an image about `center` (default: its middle), same size, bilinear, replicate border.

    out(p) = img(R(degrees) (p - c) + c), so a point q of `img` lands at
    rotate_point(q, degrees, c) in the output.
    """
    h, w = img.shape[:2]
    cx, cy = ((w - 1) / 2, (h - 1) / 2) if center is None else center
    t = np.radians(degrees)
    ys, xs = np.mgrid[:h, :w].astype(np.float64)
    sx = np.cos(t) * (xs - cx) - np.sin(t) * (ys - cy) + cx
    sy = np.sin(t) * (xs - cx) + np.cos(t) * (ys - cy) + cy
    sx, sy = np.clip(sx, 0, w - 1), np.clip(sy, 0, h - 1)
    x0, y0 = np.floor(sx).astype(int), np.floor(sy).astype(int)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    fx, fy = (sx - x0), (sy - y0)
    if img.ndim == 3:
        fx, fy = fx[..., None], fy[..., None]
    src = img.astype(np.float64)
    out = ((src[y0, x0] * (1 - fx) + src[y0, x1] * fx) * (1 - fy)
           + (src[y1, x0] * (1 - fx) + src[y1, x1] * fx) * fy)
    return np.clip(np.round(out), 0, 255).astype(img.dtype) if img.dtype == np.uint8 else out.astype(img.dtype)


def rotate_point(q: tuple[float, float], degrees: float, center: tuple[float, float]) -> tuple[float, float]:
    """Where the point q = (x, y) of the input of rotate() is in its output."""
    t = np.radians(-degrees)
    x, y = q[0] - center[0], q[1] - center[1]
    return (float(np.cos(t) * x - np.sin(t) * y + center[0]), float(np.sin(t) * x + np.cos(t) * y + center[1]))


def find_eyes_tilted(photo: np.ndarray, finder=find_eyes) -> dict | None:
    """`finder` on the photo, and on copies rotated by +-18 degrees (a tilted head).

    The upright result is kept unless a rotated copy scores clearly higher
    (TILT_MARGIN): on an upright face the three scores are close, and a rotated
    copy must not win by chance. The eyes are returned in the coordinates of the
    ORIGINAL photo, with the angle that was used ("tilt").
    """
    best = finder(photo)
    if best is not None:
        best = dict(best, tilt=0.0)
    h, w = photo.shape[:2]
    s = min(1.0, 2 * WORK / max(h, w))                 # rotate a smaller copy: faster
    small = photo if s == 1.0 else resize.resize_area(photo, max(1, round(h * s)), max(1, round(w * s)))
    c = ((small.shape[1] - 1) / 2, (small.shape[0] - 1) / 2)
    need = -np.inf if best is None else best["score"] + TILT_MARGIN
    for deg in TILTS:
        eyes = finder(rotate(small, deg))
        if eyes is None or eyes["score"] <= need:
            continue
        need = eyes["score"]
        back = [rotate_point(eyes[k], -deg, c) for k in ("left", "right")]      # undo the rotation
        best = {"left": (back[0][0] / s, back[0][1] / s), "right": (back[1][0] / s, back[1][1] / s),
                "score": eyes["score"], "tilt": deg}
    return best


def upright(photo: np.ndarray, eyes: dict) -> tuple[np.ndarray, dict]:
    """Rotate the photo about the middle of the eyes so that the eye line is level.
    Returns the rotated photo and the eyes in it. Nearly level eyes: returned unchanged."""
    (lx, ly), (rx, ry) = eyes["left"], eyes["right"]
    angle = float(np.degrees(np.arctan2(ry - ly, rx - lx)))
    if abs(angle) < 3.0:
        return photo, eyes
    c = ((lx + rx) / 2, (ly + ry) / 2)
    turned = rotate(photo, angle, c)
    return turned, dict(eyes, left=rotate_point((lx, ly), angle, c), right=rotate_point((rx, ry), angle, c))
