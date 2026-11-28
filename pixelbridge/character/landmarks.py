"""Built-in Seoul landmark backgrounds, drawn in code (128x96, no image files).

    namsan       N Seoul Tower on Namsan at dusk
    gwanghwamun  Gwanghwamun gate with Bukaksan behind
    hangang      Banpo bridge rainbow fountain at night
    lotte        Lotte World Tower over Seokchon lake with cherry blossoms
    bukchon      Bukchon hanok alley

Every scene has a stone plaza from row GROUND down, so compose_scene(..., ground=0.92)
puts the character's feet on it. Drawn from scratch, so the repo carries no photo copyrights.
"""
from __future__ import annotations

import numpy as np

W, H = 128, 96
GROUND = int(H*0.9)

def sky(top, bottom, step=20):
    t = np.arange(H)[:, None]/H
    s = np.array(top)*(1-t) + np.array(bottom)*t
    s = np.round(s/step)*step
    return np.repeat(s[:, None, :], W, 1).astype(float)

def poly(img, pts, color):
    yy, xx = np.mgrid[:H, :W] + 0.5
    inside = np.zeros((H, W), bool)
    n = len(pts)
    for i in range(n):
        (x1, y1), (x2, y2) = pts[i], pts[(i+1) % n]
        cond = ((y1 > yy) != (y2 > yy)) & (xx < (x2-x1)*(yy-y1)/(y2-y1+1e-9) + x1)
        inside ^= cond
    img[inside] = color

def rect(img, x0, y0, x1, y1, color):
    img[max(0, y0):y1, max(0, x0):x1] = color

def roof(img, x0, x1, ridge, eave, color, edge, lift=3):
    xc, hw = (x0+x1)/2, (x1-x0)/2
    for x in range(max(0, x0), min(W, x1)):
        u = abs(x-xc)/hw
        e = int(round(eave - lift*u**4))
        r = int(round(ridge + (eave-ridge)*0.35*u))
        img[r:e, x] = color
        if (x - x0) % 2 == 0: img[r+1:e, x] = np.array(color)*0.85
        img[e-1, x] = edge
        img[r, x] = np.array(color)*0.7

def plaza(img, color=(170, 160, 150), line=(140, 130, 120)):
    rect(img, 0, GROUND, W, H, color); img[GROUND, :] = line; img[GROUND:, ::8] = np.array(color)*0.9

def stars(img, pts, c=(255, 240, 200)):
    for x, y in pts: img[y, x] = c

def namsan():
    bg = sky((255, 170, 120), (120, 110, 190), 24)
    x = np.arange(W); tx = 88
    hill = (H*0.73 - H*0.23*np.exp(-((x-tx)/(W*0.2))**2)).astype(int)
    for i in range(W): bg[hill[i]:, i] = (70, 120, 80); bg[hill[i]:hill[i]+2, i] = (95, 150, 95)
    top = hill[tx]
    rect(bg, tx-1, top-30, tx+1, top, (230, 230, 235)); rect(bg, tx-3, top-26, tx+3, top-22, (240, 240, 245))
    rect(bg, tx-3, top-25, tx+3, top-23, (200, 90, 90)); rect(bg, tx, top-38, tx+1, top-30, (220, 220, 225)); bg[top-39, tx] = (255, 80, 80)
    stars(bg, [(20, 12), (40, 6), (60, 18), (110, 10), (8, 30)]); plaza(bg)
    return bg

def gwanghwamun():
    bg = sky((120, 180, 240), (200, 230, 250), 20)
    poly(bg, [(0, 52), (18, 34), (34, 26), (52, 30), (70, 20), (92, 28), (112, 36), (128, 44), (128, 70), (0, 70)], (105, 135, 115))   # Bukaksan
    rect(bg, 0, 60, W, GROUND, (150, 80, 60)); rect(bg, 0, 58, W, 60, (70, 70, 75))                 # palace wall
    rect(bg, 30, 56, 98, GROUND, (200, 195, 180))                                                  # stone base
    for cx in (44, 64, 84):                                                                        # 3 arches
        for y in range(70, GROUND):
            hw = 5 if y > 74 else int(np.sqrt(max(0, 25-(74-y)**2*1.3)))
            bg[y, cx-hw:cx+hw] = (40, 35, 35)
    rect(bg, 38, 47, 90, 56, (170, 60, 50)); bg[50, 38:90] = (60, 140, 110); bg[52, 38:90] = (60, 140, 110)   # lower pavilion with dancheong
    roof(bg, 30, 98, 41, 49, (70, 72, 80), (150, 150, 160), lift=4)
    rect(bg, 44, 33, 84, 41, (170, 60, 50)); bg[36, 44:84] = (60, 140, 110)
    roof(bg, 38, 90, 26, 35, (70, 72, 80), (150, 150, 160), lift=4)
    plaza(bg, (205, 200, 190), (175, 170, 160))
    return bg

def hangang():
    bg = sky((15, 20, 55), (60, 50, 110), 15)
    stars(bg, [(10, 8), (30, 4), (52, 12), (75, 6), (98, 14), (120, 5), (110, 22)])
    rng = np.random.default_rng(3); x = 0
    while x < W:                                                                                    # skyline
        w_ = int(rng.integers(5, 11)); h_ = int(rng.integers(10, 30))
        rect(bg, x, 52-h_, x+w_, 52, (35, 40, 70))
        for yy in range(52-h_+2, 51, 3):
            for xx in range(x+1, min(W, x+w_-1), 2):
                if rng.random() < 0.4: bg[yy, xx] = (250, 220, 120)
        x += w_ + int(rng.integers(0, 3))
    rect(bg, 0, 52, W, 78, (25, 40, 90))                                                           # river
    for yy in range(54, 78, 3):
        for xx in range(int(rng.integers(0, 6)), W, int(rng.integers(6, 12))): bg[yy, xx:xx+3] = (70, 90, 150)
    rect(bg, 0, 58, W, 61, (120, 120, 130)); bg[58, :] = (170, 170, 180)                           # bridge deck
    for px in range(8, W, 20): rect(bg, px, 61, px+2, 66, (90, 90, 100))
    cols = [(255, 80, 80), (255, 170, 60), (250, 240, 90), (90, 220, 120), (80, 160, 255), (180, 110, 255)]
    for i, xx in enumerate(range(4, W, 4)):                                                       # moonlight rainbow fountain
        c = cols[(i//3) % len(cols)]
        for k in range(1, 8): bg[61+k, xx + k//3] = np.array(c)*(1-k/12)
    rect(bg, 0, 78, W, GROUND, (60, 110, 70))                                                       # riverside park
    plaza(bg, (120, 120, 125), (90, 90, 95))
    return bg

def lotte():
    bg = sky((150, 200, 245), (235, 240, 250), 20)
    rect(bg, 0, 60, W, 72, (150, 165, 185))                                                         # low buildings
    for x in range(0, W, 9): rect(bg, x, 54 + (x*7) % 9, x+7, 72, (165, 175, 195))
    poly(bg, [(78, 72), (98, 72), (92, 8), (89, 2), (86, 2), (84, 8)], (205, 220, 235))            # tower
    poly(bg, [(88, 72), (98, 72), (92, 8), (89, 2)], (170, 190, 215))
    for yy in range(12, 72, 4): bg[yy, 82:96] = np.where(bg[yy, 82:96].sum(-1)[:, None] > 500, np.array([185, 200, 220]), bg[yy, 82:96])
    rect(bg, 0, 72, W, 82, (70, 120, 170))                                                          # Seokchon lake
    for xx in range(0, W, 7): bg[76, xx:xx+3] = (130, 170, 210)
    for tx_ in (6, 20, 34, 50, 112, 124):                                                           # cherry trees
        for dy in range(-6, 1):
            for dx in range(-4, 5):
                if dx*dx + dy*dy*1.6 < 18: bg[78+dy, min(W-1, max(0, tx_+dx))] = (245, 180, 200) if (dx+dy) % 3 else (230, 150, 180)
        rect(bg, tx_, 78, tx_+1, 84, (110, 80, 60))
    rect(bg, 0, 82, W, GROUND, (95, 150, 90)); plaza(bg)
    return bg

def bukchon():
    bg = sky((170, 210, 245), (240, 235, 225), 20)
    poly(bg, [(0, 40), (40, 34), (80, 38), (128, 30), (128, 60), (0, 60)], (150, 170, 175))       # distant city
    rect(bg, 98, 22, 99, 34, (220, 220, 225)); rect(bg, 96, 24, 101, 26, (230, 230, 235))          # tiny N Seoul Tower
    poly(bg, [(44, GROUND), (84, GROUND), (70, 52), (58, 52)], (190, 180, 165))                    # alley going uphill
    for side, xs in ((-1, (0, 46)), (1, (82, 128))):                                               # hanok rows
        for k, y in enumerate((50, 62, 74)):
            x0, x1 = xs
            if side < 0: x1 = 58 - (2-k)*6 if k < 2 else 46
            else: x0 = 70 + (2-k)*6 if k < 2 else 82
            rect(bg, x0, y+6, x1, y+16, (215, 200, 170)); rect(bg, x0, y+12, x1, y+16, (150, 140, 130))
            for wx in range(x0+3, x1-3, 8): rect(bg, wx, y+7, wx+3, y+11, (140, 90, 60))
            roof(bg, x0-3, x1+3, y-2, y+7, (75, 78, 88), (140, 140, 150), lift=3)
    plaza(bg, (175, 165, 150), (145, 135, 120))
    return bg


LANDMARKS = {
    "namsan": ("N Seoul Tower", namsan),
    "gwanghwamun": ("Gwanghwamun", gwanghwamun),
    "hangang": ("Hangang bridge", hangang),
    "lotte": ("Lotte World Tower", lotte),
    "bukchon": ("Bukchon Hanok", bukchon),
}


def landmark_background(name: str = "namsan", rng: np.random.Generator | None = None) -> tuple[str, np.ndarray]:
    """Return (key, background (96, 128, 3) uint8). name may be a key from LANDMARKS or "random"."""
    if name == "random":
        rng = rng if rng is not None else np.random.default_rng()
        name = list(LANDMARKS)[int(rng.integers(len(LANDMARKS)))]
    if name not in LANDMARKS:
        raise ValueError(f"unknown landmark {name!r}; choose from {', '.join(LANDMARKS)} or random")
    bg = LANDMARKS[name][1]()
    return name, np.clip(bg, 0, 255).astype(np.uint8)
