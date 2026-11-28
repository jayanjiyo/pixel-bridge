"""Measure a face at positions fixed by the two eyes.

Once the eyes are known (eyes.find_eyes), every other feature sits at a known
place relative to them, in units of the eye distance d:

        hair band          about 1.2 d above the eyes and higher
        forehead / bangs   0.5 d .. 0.95 d above
        eyebrows           0.22 d .. 0.6 d above
        cheeks             0.35 d .. 0.8 d below      -> the skin model used to segment the photo
        nose               0.15 d .. 0.85 d below     -> the face color of the sprite (average of the middle)
        mouth              0.95 d .. 1.5 d below
        neck / shoulders   1.3 d .. 3.6 d below       -> hair length
        chest              3.8 d .. 5 d below         -> shirt color (low, below a low neckline)

So we do not need one skin threshold for everybody: the skin color is read
from the cheeks of THIS photo, the hair color from just above THIS hairline,
and both are then used to segment the rest (distance in CIELAB).

Measured -> drawn (sprite.py):
    face width, jaw width        face shape of the sprite
    hair color, length, volume   hair template (crop / short / medium / long) and color
    forehead coverage            bangs
    hair outline and strands     hair drawn from the photo (hair_grid), not from a template
    eyebrow darkness and color   eyebrows
    eye height                   eye size
    small box at the pupil       eye color
    bright teeth in the mouth    open smile
    lip color                    mouth color
    dark, colorless lower face   mustache / beard
"""
from __future__ import annotations

import numpy as np

from ..core.color import rgb_to_gray, rgb_to_lab
from .eyes import _box_mean
from .hair import EYE_ROW, FACES, HEAD_H, HEAD_W, SHAPE_ROWS
from .measure import Attributes, _median, _resize

U = 40                       # the crop is resized so that the eye distance is U pixels

# Decision thresholds (in units of the eye distance, or CIELAB units).
SKIN_TOL = 13.0
EYE_BOX = 0.05              # eye color box: half size around the pupil (eye distances)
SHIRT = (3.8, 5.0)          # shirt color: from / to below the eyes (eye distances)
NOSE = (0.15, 0.85, 0.3)    # face color box: from / to below the eyes, half width (eye distances)
FOREHEAD_TOL = 18.0         # looser skin test for the forehead (real hair is at least ~21 away)
HAIR_AB_MIN, HAIR_L_MARGIN = 9.0, 12.0     # hair model: smallest hue tolerance, margin on the lightness range
BG_RATIO = 1.0              # hair must be this much closer (CIELAB) to the hair model than to the background
HAIR_SIDE_LOW = 1.3         # hair envelope beside the shoulders (long hair spreads out)
HAIR_SIDE, HAIR_DOME = 0.6, 1.3      # hair envelope: beside the face / above the hairline (eye distances)
SHELL = 0.6                 # hair outside the face is drawn at this fraction of its real thickness
TWO_TONE = 18.0             # L* gap between the dark and the light half that means two-tone hair
HAIR_PASSES = 2             # the model follows connected hair into highlights and long ends
FACE_SLIM, FACE_ROUND = 2.05, 2.40
JAW_POINTED = 0.74
VOLUME_FLAT, VOLUME_BIG = 0.45, 0.95
BROW_THIN, BROW_BOLD = 14.0, 26.0
EYE_SMALL, EYE_SMILING = 0.30, 0.26
TEETH_MIN = 0.035
TONE_STEP = 9.0             # L* difference from the typical hair that counts as a darker / lighter strand
LIP_REDDER = 6.0            # lips: a* this much above the skin
BEARD_DARKER, BEARD_GRAYER = 14.0, 0.6    # facial hair: darker than the cheeks and much less colorful
CROP_MAX = 0.25             # hair beside the cheeks below this: cut above the ears


def hair_grid(hair: np.ndarray, lightness: np.ndarray, ex: float, ey: float, face: str,
              face_w: float) -> tuple[np.ndarray, np.ndarray]:
    """Hair mask of the photo -> hair silhouette and tone on the sprite grid
    (two SHAPE_ROWS x HEAD_W arrays: bool, and int8 with -1 / 0 / +1).

    The sprite is a chibi: its head is squashed compared with a real head, so the
    mapping is not one scale. Across, the measured face width maps onto the width
    of the sprite face. Up, 2.15 eye distances (about the top of the head) map onto
    the rows above the sprite eyes; down, the chin (1.75 eye distances) maps onto
    the bottom of the sprite face. A cell is hair when at least 45% of the photo
    pixels it covers are hair (area sampling). Its tone says whether those hair
    pixels are clearly darker or lighter than the typical hair of this photo
    (dark roots, bleached tips, highlights). Single stray cells are removed and
    single holes are filled (3x3 majority).
    """
    eye_row = HEAD_H * EYE_ROW + 0.5                       # center of the sprite eyes
    eye_col = HEAD_W * 0.5                                 # middle between the sprite eyes
    sx = float(np.clip(HEAD_W * FACES[face] / max(face_w / 2, 0.8), 5.0, 7.5))
    sy_up = eye_row / 2.15
    sy_down = (HEAD_H * 0.94 - eye_row) / 1.75
    h, w = hair.shape
    grid = np.zeros((SHAPE_ROWS, HEAD_W), bool)
    level = np.zeros((SHAPE_ROWS, HEAD_W))
    base = float(np.median(lightness[hair])) if hair.any() else 0.0

    face_half = HEAD_W * FACES[face]                       # half width of the sprite face

    def src_y(v: float) -> int:
        return int(np.clip(ey + (v / sy_up if v < 0 else v / sy_down) * U, 0, h))

    def src_x(g: float) -> int:
        # beside the face one sprite pixel covers more of the photo (1 / SHELL): on a chibi head
        # the real hair thickness would look far too big next to the small face
        a = abs(g)
        u = a / sx if a <= face_half else face_half / sx + (a - face_half) / (sx * SHELL)
        return int(np.clip(ex + np.sign(g) * u * U, 0, w))
    for gy in range(SHAPE_ROWS):
        ya, yb = src_y(gy - eye_row), src_y(gy + 1 - eye_row)
        for gx in range(HEAD_W):
            xa, xb = src_x(gx - eye_col), src_x(gx + 1 - eye_col)
            block = hair[ya:yb, xa:xb]
            grid[gy, gx] = block.size > 0 and block.mean() >= 0.45
            if grid[gy, gx]:
                level[gy, gx] = float(np.median(lightness[ya:yb, xa:xb][block])) - base
    padded = np.pad(grid, 1).astype(int)
    around = sum(padded[1 + dy:1 + dy + SHAPE_ROWS, 1 + dx:1 + dx + HEAD_W]
                 for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx)
    grid = (grid & (around >= 2)) | (around >= 6)
    tone = np.where(level > TONE_STEP, 1, np.where(level < -TONE_STEP, -1, 0)).astype(np.int8)
    return grid, np.where(grid, tone, 0).astype(np.int8)


def _dist(lab: np.ndarray, ref: np.ndarray, wl: float) -> np.ndarray:
    """Color distance in CIELAB with less weight on lightness (shading changes L most)."""
    d = lab - ref
    return np.sqrt(wl * d[..., 0] ** 2 + d[..., 1] ** 2 + d[..., 2] ** 2)


def measure_from_eyes(photo: np.ndarray, eyes: dict) -> Attributes:
    """Attributes of a front-facing portrait, measured relative to the eyes."""
    (lx, ly), (rx, ry) = eyes["left"], eyes["right"]
    d = float(np.hypot(rx - lx, ry - ly))
    cx, cy = (lx + rx) / 2, (ly + ry) / 2
    ph, pw = photo.shape[:2]
    x0, x1 = int(max(0, cx - 2.4 * d)), int(min(pw, cx + 2.4 * d))
    y0, y1 = int(max(0, cy - 2.6 * d)), int(min(ph, cy + (SHIRT[1] + 0.2) * d))
    crop = photo[y0:y1, x0:x1]
    f = U / d
    work = _resize(crop, max(8, round(crop.shape[0] * f)), max(8, round(crop.shape[1] * f)))
    h, w = work.shape[:2]
    lab = rgb_to_lab(work).astype(np.float64)
    L = lab[..., 0]
    ex, ey = (cx - x0) * f, (cy - y0) * f                 # eye midpoint in the working image
    elx, erx = (lx - x0) * f, (rx - x0) * f

    def rows(a: float, b: float) -> slice:                # a, b in eye distances below the eyes
        return slice(int(np.clip(ey + a * U, 0, h)), int(np.clip(ey + b * U, 0, h)))

    def cols(a: float, b: float, center: float | None = None) -> slice:
        c = ex if center is None else center
        return slice(int(np.clip(c + a * U, 0, w)), int(np.clip(c + b * U, 0, w)))

    # ---- skin: color of the cheeks below the eyes
    cheeks = [(rows(0.35, 0.8), cols(-0.3, 0.3, elx)), (rows(0.35, 0.8), cols(-0.3, 0.3, erx))]
    cheek_lab = np.concatenate([lab[r, c].reshape(-1, 3) for r, c in cheeks])
    cheek_rgb = np.concatenate([work[r, c].reshape(-1, 3) for r, c in cheeks])
    skin_lab = np.median(cheek_lab, axis=0)
    skin_rgb = _median(cheek_rgb, (230, 190, 160))          # replaced below by the average around the nose
    skin = _dist(lab, skin_lab, 0.3) < SKIN_TOL
    left_l = float(lab[cheeks[0]][..., 0].mean())
    right_l = float(lab[cheeks[1]][..., 0].mean())
    shadow = "right" if right_l < left_l - 2 else "left" if left_l < right_l - 2 else None

    # ---- face shape: skin width at cheek level and at mouth level
    def width(a: float, b: float) -> float:
        # extent of the skin around the face center (lips, teeth and nostrils make holes,
        # so the outermost skin pixels are used, not one unbroken run)
        near = skin[rows(a, b), cols(-1.5, 1.5)]
        ws = [np.ptp(np.nonzero(r)[0]) + 1 for r in near if r.any()]
        return float(np.median(ws)) / U if ws else 0.0
    face_w = width(0.3, 0.6)
    jaw_w = width(1.15, 1.4)
    jaw_ratio = jaw_w / face_w if face_w > 0 else 1.0
    face = "slim" if 0 < face_w < FACE_SLIM else "round" if face_w > FACE_ROUND else "oval"
    jaw = "pointed" if 0.4 <= jaw_ratio < JAW_POINTED else "round"      # below 0.4 the measurement failed

    # ---- hairline: going up from the eyebrows, the first long run of non-skin rows
    # The forehead is often lit differently from the cheeks, so a looser skin test is used
    # here: otherwise a bright forehead is taken for (blond) hair.
    forehead_skin = _dist(lab, skin_lab, 0.3) < FOREHEAD_TOL
    band = cols(-0.5, 0.5)
    hairline, run = None, 0
    for r in range(int(ey - 0.3 * U), -1, -1):
        if forehead_skin[r, band].mean() < 0.4:
            run += 1
            if run >= int(0.2 * U):                        # longer than an eyebrow is thick
                hairline = r + run - 1
                break
        else:
            run = 0
    top_seed = hairline if hairline is not None else int(max(0, ey - 1.5 * U))
    forehead = (ey - top_seed) / U

    # ---- hair: a color MODEL from just above the hairline, grown from that seed
    # Hair is not one color: it has shadows, shine, dark roots and bleached tips.
    # So the model is a hue (a*, b*) with a tolerance plus a RANGE of lightness, both
    # taken from the seed pixels. A second model, from the borders of the crop, is the
    # background: a pixel that fits the background better is not hair. Finally only
    # the region connected to the seed is kept (region growing).
    seed = (slice(max(0, top_seed - int(0.4 * U)), max(1, top_seed - int(0.08 * U))), cols(-0.9, 0.9))
    seed_mask = np.zeros((h, w), bool)
    seed_mask[seed] = True
    seed_mask &= ~forehead_skin
    if seed_mask.sum() < 20:
        seed_mask[seed] = True
    hair_rgb = _median(work[seed_mask], (50, 35, 30))

    def model(px: np.ndarray) -> tuple[np.ndarray, float, float, float]:
        ab = np.median(px[:, 1:], axis=0)
        spread = float(np.percentile(np.hypot(px[:, 1] - ab[0], px[:, 2] - ab[1]), 90))
        lo, hi = np.percentile(px[:, 0], [5, 95])
        return ab, max(HAIR_AB_MIN, spread + 4.0), float(lo) - HAIR_L_MARGIN, float(hi) + HAIR_L_MARGIN

    def fits(m) -> np.ndarray:
        ab, tol, lo, hi = m
        return (np.hypot(lab[..., 1] - ab[0], lab[..., 2] - ab[1]) < tol) & (L > lo) & (L < hi)

    def grow(allowed: np.ndarray, seeds: np.ndarray) -> np.ndarray:
        region = seeds & allowed
        for _ in range(h + w):
            wider = region.copy()
            wider[1:] |= region[:-1]
            wider[:-1] |= region[1:]
            wider[:, 1:] |= region[:, :-1]
            wider[:, :-1] |= region[:, 1:]
            wider &= allowed
            if wider.sum() == region.sum():
                break
            region = wider
        return region

    # background samples: the top corners and the side borders of the upper half
    q = int(0.45 * U)
    border = np.zeros((h, w), bool)
    border[:q, :q] = border[:q, -q:] = True
    border[:int(ey), :max(2, q // 3)] = border[:int(ey), -max(2, q // 3):] = True
    left_half = np.arange(w)[None, :] < w // 2
    top_part = np.arange(h)[:, None] < q
    groups = [g for g in (border & left_half & top_part, border & ~left_half & top_part,
                          border & left_half & ~top_part, border & ~left_half & ~top_part) if g.sum() >= 20]
    bg_lab = [np.median(lab[g], axis=0) for g in groups]

    def halves(px: np.ndarray) -> list[np.ndarray]:
        """Dark half and light half of the pixels, as two groups when they clearly differ
        (two-tone hair: dark roots and highlights), else one group."""
        mid = np.median(px[:, 0])
        dark, light = px[px[:, 0] <= mid], px[px[:, 0] > mid]
        if len(dark) >= 20 and len(light) >= 20 and np.median(light[:, 0]) - np.median(dark[:, 0]) > TWO_TONE:
            return [dark, light]
        return [px]

    # Where hair can be, from the layout of a head: a dome above the hairline, a band
    # beside the face, and (long hair) the same band further down. A couch, a wall or
    # a hand of the same color outside this envelope is not hair.
    half = ((face_w / 2 if face_w > 0 else 1.1) + HAIR_SIDE) * U
    yy, xx = np.mgrid[:h, :w]
    # beside the face the band is narrow (a couch or a wall must not get in); below the jaw it
    # widens, because long hair spreads over the shoulders
    spread = np.clip((yy - (ey + 1.0 * U)) / (1.5 * U), 0.0, 1.0) * (HAIR_SIDE_LOW - HAIR_SIDE) * U
    envelope = np.abs(xx - ex) <= half + spread
    dome = ((xx - ex) / half) ** 2 + ((yy - top_seed) / (HAIR_DOME * U)) ** 2 <= 1
    envelope &= (yy >= top_seed) | dome

    hair = np.zeros((h, w), bool)
    region_px = lab[seed_mask]
    for _ in range(HAIR_PASSES):                           # later passes: model from the grown region
        tones = halves(region_px)
        cand = np.zeros((h, w), bool)
        for t in tones:
            cand |= fits(model(t))
        cand &= ~skin & envelope
        if bg_lab:                                         # drop what looks more like the background
            d_hair = np.min([_dist(lab, np.median(t, axis=0), 0.5) for t in tones], axis=0)
            d_bg = np.min([_dist(lab, b, 0.5) for b in bg_lab], axis=0)
            cand &= d_hair <= BG_RATIO * d_bg              # clearly closer to the hair than to the background
        # long hair: the hair beside the face is often cut off from the top by a highlight,
        # so a side that is mostly hair-colored becomes a seed too
        seeds = seed_mask.copy()
        for lo, hi in ((-1.6, -1.15), (1.15, 1.6)):
            side_box = (rows(0.0, 1.0), cols(lo, hi))
            if cand[side_box].size and cand[side_box].mean() >= 0.35:
                seeds[side_box] |= cand[side_box]
        hair = grow(cand, seeds)
        if hair.sum() < 50:
            break
        region_px = lab[hair]
    if hair.sum() >= 50:
        hair_rgb = np.median(work[hair], axis=0)           # color of all the hair, not only of the seed band

    # ---- face color: the average of the middle of the face, around the nose
    # The box between the eyes, from just below the eye line down to the tip of the nose.
    # It is the part of the face least touched by hair, a beard, the shadow of the jaw or
    # the background. Pixels that are not skin (the nostrils) and hair are left out.
    nose = (rows(NOSE[0], NOSE[1]), cols(-NOSE[2], NOSE[2]))
    nose_skin = forehead_skin[nose] & ~hair[nose]
    if nose_skin.sum() >= 30:
        skin_rgb = work[nose][nose_skin].mean(axis=0)
    # the background has the hair color too (dark hair in a dark room): the outline cannot be trusted
    leak = bool(border.any() and hair[border].mean() > 0.35)

    top = top_seed
    miss = 0
    for r in range(top_seed - 1, -1, -1):
        if hair[r, band].mean() >= 0.4:
            top, miss = r, 0
        else:
            miss += 1
            if miss > 3:
                break
    volume_u = (top_seed - top) / U
    volume = "normal" if leak else "flat" if volume_u < VOLUME_FLAT else "big" if volume_u > VOLUME_BIG else "normal"

    # ---- hair length: hair beside the neck, then beside the shoulders
    def side(a: float, b: float, near: float, far: float) -> float:
        r = rows(a, b)
        parts = [hair[r, cols(-far, -near)], hair[r, cols(near, far)]]
        vals = [p.mean() for p in parts if p.size]
        return float(max(vals)) if vals else 0.0
    beside_face = side(0.0, 0.9, 1.25, 1.7)
    beside_neck = side(1.3, 2.2, 1.0, 1.6)
    beside_shoulder = side(2.6, 3.6, 1.0, 2.1)

    # ---- shirt: below the neck, not skin and not hair
    # Low on the chest: with a low neckline the area just below the neck is skin. If the photo
    # ends before that (a close-up), the lowest part of the photo is used.
    shirt_rows = rows(SHIRT[0], SHIRT[1])
    if shirt_rows.stop - shirt_rows.start < int(0.4 * U):
        shirt_rows = slice(max(int(ey + 2.8 * U), h - int(1.0 * U)), h)
    region = (shirt_rows, cols(-1.0, 1.0))
    reg_rgb = work[region].reshape(-1, 3)
    shirt = np.array([200.0, 60.0, 70.0])
    shirt_is_hair_colored = False
    if len(reg_rgb) > 50:
        ok = ~(forehead_skin[region] | hair[region]).reshape(-1)       # the loose skin test: a shaded chest too
        shirt_is_hair_colored = bool(hair[region].mean() > 0.5)
        shirt = np.median(reg_rgb[ok] if ok.sum() > 50 else reg_rgb, axis=0)
    if leak:
        length = "short"
    elif beside_neck < 0.4:
        length = "crop" if beside_face < CROP_MAX else "short"
    elif beside_shoulder >= 0.3 and not shirt_is_hair_colored:
        length = "long"
    else:
        length = "medium"

    # ---- bangs: hair on the forehead
    fb = rows(-0.95, -0.5)
    cov_c = float(hair[fb, cols(-0.35, 0.35)].mean()) if hair[fb, cols(-0.35, 0.35)].size else 0.0
    cov_l = float(hair[fb, cols(-0.95, -0.35)].mean()) if hair[fb, cols(-0.95, -0.35)].size else 0.0
    cov_r = float(hair[fb, cols(0.35, 0.95)].mean()) if hair[fb, cols(0.35, 0.95)].size else 0.0
    part = 0.5
    if cov_c > 0.5 and min(cov_l, cov_r) > 0.3:
        bangs = "full"
    elif abs(cov_l - cov_r) > 0.45 and max(cov_l, cov_r) > 0.5:
        bangs = "side"
        part = 0.38 if cov_l > cov_r else 0.62
    elif min(cov_l, cov_r) > 0.45:
        bangs = "curtain"
    else:
        bangs = "none"

    # ---- eyebrows: how much darker than the skin
    brow_dark = float(np.mean([skin_lab[0] - np.percentile(L[rows(-0.6, -0.22), cols(-0.35, 0.35, c)], 15)
                               for c in (elx, erx)]))
    brow_px, brow_rgb, brow_hair = [], [], []
    for c in (elx, erx):
        reg = (rows(-0.6, -0.22), cols(-0.35, 0.35, c))
        brow_px.append(L[reg].reshape(-1))
        brow_rgb.append(work[reg].reshape(-1, 3))
        brow_hair.append(hair[reg].mean() if hair[reg].size else 0.0)
    covered = float(np.mean(brow_hair)) > 0.5                 # bangs hang over the eyebrows
    brows = "none" if covered or brow_dark < BROW_THIN else "thin" if brow_dark < BROW_BOLD else "bold"
    brow_color = None
    if brows != "none":
        lum, rgb = np.concatenate(brow_px), np.concatenate(brow_rgb)
        brow_color = np.median(rgb[lum <= np.percentile(lum, 20)], axis=0)

    # ---- eyes: height of the dark part
    def eye_height(c: float, y: float) -> float:
        r = slice(int(np.clip(y - 0.22 * U + (0), 0, h)), int(np.clip(y + 0.22 * U, 0, h)))
        dark = L[r, cols(-0.3, 0.3, c)] < skin_lab[0] - 22
        return float((dark.mean(axis=1) >= 0.15).sum()) / U if dark.size else 0.0
    eye_h = (eye_height(elx, (ly - y0) * f) + eye_height(erx, (ry - y0) * f)) / 2
    eye_size = "small" if eye_h < EYE_SMALL else "big"

    # ---- eye color: a small box around the pupil, read at the full resolution of the photo
    def pupil_color(x: float, y: float) -> np.ndarray | None:
        hw, hh = max(2, int(0.2 * d)), max(2, int(0.15 * d))
        px0, py0 = int(max(0, x - hw)), int(max(0, y - hh))
        patch = photo[py0:int(y + hh) + 1, px0:int(x + hw) + 1]
        if patch.size == 0:
            return None
        size = max(1, int(round(0.06 * d)))
        dark = _box_mean(rgb_to_gray(patch).astype(np.float64), size, size)
        cy_, cx_ = np.unravel_index(np.argmin(dark), dark.shape)       # the pupil: the darkest small spot
        half = max(1, int(round(EYE_BOX * d)))
        box = patch[max(0, cy_ - half):cy_ + half + 1, max(0, cx_ - half):cx_ + half + 1]
        return box.reshape(-1, 3).mean(axis=0)
    pupils = [c for c in (pupil_color(lx, ly), pupil_color(rx, ry)) if c is not None]
    eye_color = np.mean(pupils, axis=0) if pupils else None

    # ---- mouth: bright, colorless pixels are teeth
    mouth = lab[rows(0.95, 1.5), cols(-0.5, 0.5)]
    teeth_frac = float(((mouth[..., 0] > 68) & (np.hypot(mouth[..., 1], mouth[..., 2]) < 13)).mean()) if mouth.size else 0.0
    teeth = teeth_frac > TEETH_MIN
    mouth_px = mouth.reshape(-1, 3)
    mouth_rgb = work[rows(0.95, 1.5), cols(-0.5, 0.5)].reshape(-1, 3)
    is_teeth = (mouth_px[:, 0] > 68) & (np.hypot(mouth_px[:, 1], mouth_px[:, 2]) < 13)
    is_lip = (mouth_px[:, 1] > skin_lab[1] + LIP_REDDER) & ~is_teeth
    lip = np.median(mouth_rgb[is_lip], axis=0) if is_lip.sum() >= 30 else None

    # ---- facial hair: clearly darker than the cheeks AND much less colorful (a shadow keeps the skin hue)
    skin_chroma = float(np.hypot(skin_lab[1], skin_lab[2]))

    def hairy(a: float, b: float, half: float) -> tuple[bool, float, float]:
        reg = lab[rows(a, b), cols(-half, half)].reshape(-1, 3)
        if not len(reg):
            return False, 0.0, 0.0
        med = np.median(reg, axis=0)
        darker, chroma = float(skin_lab[0] - med[0]), float(np.hypot(med[1], med[2]))
        return darker > BEARD_DARKER and chroma < BEARD_GRAYER * skin_chroma, darker, chroma
    mustache, must_dark, _ = hairy(0.72, 0.92, 0.4)
    chin, chin_dark, _ = hairy(1.5, 1.8, 0.45)
    beard = "full" if mustache and chin else "mustache" if mustache else "chin" if chin else "none"

    # hair silhouette for the sprite: only when the hair could be told from the background
    shape, tone = hair_grid(hair, L, ex, ey, face, face_w) if hairline is not None and not leak else (None, None)
    # the real colors of the darker and the lighter strands (dark roots, highlights)
    hair_colors = None
    if shape is not None and hair.any():
        base = float(np.median(L[hair]))
        dark_px, light_px = hair & (L < base - TONE_STEP), hair & (L > base + TONE_STEP)
        hair_colors = (np.median(work[dark_px], axis=0) if dark_px.sum() >= 30 else None,
                       np.median(work[light_px], axis=0) if light_px.sum() >= 30 else None)

    box = (int(max(0, cx - 1.6 * d)), int(max(0, cy - 2.0 * d)), int(min(pw, cx + 1.6 * d)), int(min(ph, cy + 1.8 * d)))
    return Attributes(
        box=box, smile=teeth, happy_eyes=bool(teeth and eye_h < EYE_SMILING), bangs=bangs, part=part, length=length,
        skin=skin_rgb, hair=hair_rgb, shirt=shirt, shadow_side=shadow,
        scores=dict(eye_distance=d, face_w=face_w, jaw_ratio=jaw_ratio, forehead=float(forehead),
                    hairline_found=hairline is not None, volume=float(volume_u), background_like_hair=leak,
                    beside_face=beside_face, beside_neck=beside_neck, beside_shoulder=beside_shoulder,
                    bangs_center=cov_c, bangs_left=cov_l, bangs_right=cov_r,
                    brow_dark=brow_dark, eye_h=float(eye_h), teeth=teeth_frac, mustache_dark=must_dark, chin_dark=chin_dark,
                    eyes_score=float(eyes.get("score", 0.0))),
        face=face, jaw=jaw, brows=brows, eyes=eye_size, teeth=teeth, volume=volume,
        hair_shape=shape, hair_tone=tone, hair_colors=hair_colors, brow_color=brow_color, lip=lip, beard=beard,
        eye_color=eye_color)
