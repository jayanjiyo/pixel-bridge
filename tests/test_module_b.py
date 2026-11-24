"""M4: module_b/similarity.py, scalex.py, xbr.py, slope.py

    python -m pytest tests/test_module_b.py
"""
import numpy as np
import pytest

from pixelbridge.module_b import scalex, similarity, slope, xbr

R, G, B = (200, 0, 0), (0, 160, 0), (0, 0, 220)
rng = np.random.default_rng(5)


def distinct_colors(h, w):
    """Image where no two pixels share a color (all rules stay inactive)."""
    n = h * w
    vals = rng.choice(256 ** 3, size=n, replace=False)
    return np.stack([vals // 65536, (vals // 256) % 256, vals % 256], -1).reshape(h, w, 3).astype(np.uint8)


def sprite():
    """Small pixel art: a disc with a diagonal stripe, 4 colors."""
    yy, xx = np.mgrid[:16, :16]
    img = np.zeros((16, 16, 3), np.uint8)
    img[:] = (250, 250, 250)
    img[(yy - 8) ** 2 + (xx - 8) ** 2 < 36] = R
    img[np.abs(yy - xx // 3 - 4) < 1] = B           # gentle 1:3 slope
    return img


def colors_of(img):
    return {tuple(c) for c in img.reshape(-1, 3)}


# ------------------------------------------------------------------ similarity
def test_yuv_distance():
    a = rng.integers(0, 256, (5, 5, 3), dtype=np.uint8)
    b = rng.integers(0, 256, (5, 5, 3), dtype=np.uint8)
    assert np.allclose(similarity.yuv_distance(a, a), 0)
    assert np.allclose(similarity.yuv_distance(a, b), similarity.yuv_distance(b, a))
    assert similarity.yuv_distance(a, b).shape == (5, 5)
    assert similarity.similar(a, a).all()


# ------------------------------------------------------------------ Scale2x / Scale3x / Eagle
def test_scale2x_rules():
    img = np.zeros((3, 3, 3), np.uint8)
    img[1, 1] = G                    # P
    img[0, 1] = img[1, 0] = R        # A (up) == C (left)
    img[1, 2] = img[2, 1] = B        # B (right) == D (down)
    img[0, 0] = img[0, 2] = img[2, 0] = img[2, 2] = (90, 90, 90)
    out = scalex.scale2x(img)
    assert out.shape == (6, 6, 3) and out.dtype == np.uint8
    block = out[2:4, 2:4]            # output block of the center pixel
    assert tuple(block[0, 0]) == R   # E0 = A
    assert tuple(block[0, 1]) == G   # E1 = P
    assert tuple(block[1, 0]) == G   # E2 = P
    assert tuple(block[1, 1]) == B   # E3 = D


@pytest.mark.parametrize("fn,k", [(scalex.scale2x, 2), (scalex.scale3x, 3), (scalex.eagle, 2)])
def test_scalers_basic_properties(fn, k):
    flat = np.full((5, 7, 3), 77, np.uint8)
    assert np.array_equal(fn(flat), np.full((5 * k, 7 * k, 3), 77, np.uint8))
    d = distinct_colors(6, 6)                        # no equal neighbors -> nearest
    assert np.array_equal(fn(d), np.repeat(np.repeat(d, k, 0), k, 1))
    s = sprite()
    assert colors_of(fn(s)) <= colors_of(s)          # never invents colors


# ------------------------------------------------------------------ xBR and ours
@pytest.mark.parametrize("fn", [xbr.xbr, slope.upscale_slope])
@pytest.mark.parametrize("scale", [2, 3, 4])
def test_rule_upscalers(fn, scale):
    s = sprite()
    out = fn(s, scale=scale, mode="palette")
    assert out.shape == (16 * scale, 16 * scale, 3) and out.dtype == np.uint8
    assert colors_of(out) <= colors_of(s)            # palette mode keeps colors
    flat = np.full((6, 6, 3), 40, np.uint8)
    assert (fn(flat, scale=scale, mode="smooth") == 40).all()
    smooth = fn(s, scale=scale, mode="smooth")
    assert smooth.shape == out.shape


def test_ours_differs_from_nearest_on_gentle_slope():
    s = sprite()
    out = slope.upscale_slope(s, scale=4, mode="palette")
    assert not np.array_equal(out, np.repeat(np.repeat(s, 4, 0), 4, 1))


def test_estimate_slopes_shape():
    m = slope.estimate_slopes(sprite())
    assert m.shape == (16, 16)
