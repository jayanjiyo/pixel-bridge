"""M3: module_a/dither.py, outline.py, cleanup.py

    python -m pytest tests/test_module_a_back.py
"""
import numpy as np
import pytest

from pixelbridge.module_a import cleanup, dither, outline

BW = np.array([[0, 0, 0], [255, 255, 255]], np.uint8)


# ------------------------------------------------------------------ dither
def test_gradient_mask():
    flat = np.full((20, 20, 3), 128, np.uint8)
    assert dither.gradient_mask(flat).all()
    edge = flat.copy()
    edge[:, 10:] = 255
    m = dither.gradient_mask(edge)
    assert m.shape == (20, 20) and m.dtype == bool
    assert not m[:, 9:11].any() and m[:, :5].all()


@pytest.mark.parametrize("fn", [dither.floyd_steinberg, dither.bayer])
def test_dither_exact_palette_color_stays(fn):
    img = np.full((16, 16, 3), 255, np.uint8)
    idx = fn(img, BW)
    assert idx.shape == (16, 16) and idx.dtype == np.int32
    assert (idx == 1).mean() >= 0.9


@pytest.mark.parametrize("fn", [dither.floyd_steinberg, dither.bayer])
def test_dither_mid_gray_is_half_white(fn):
    img = np.full((32, 32, 3), 128, np.uint8)
    white = (fn(img, BW) == 1).mean()
    assert 0.35 < white < 0.65


@pytest.mark.parametrize("fn", [dither.floyd_steinberg, dither.bayer])
def test_dither_strength_zero_and_mask(fn):
    img = np.full((16, 16, 3), 100, np.uint8)       # nearest is black
    assert (fn(img, BW, strength=0.0) == 0).all()
    assert (fn(img, BW, mask=np.zeros((16, 16), bool)) == 0).all()


# ------------------------------------------------------------------ outline
PAL = np.array([[20, 20, 20], [120, 90, 60], [240, 220, 200]], np.uint8)


def test_darker_index():
    d = outline.darker_index(PAL)
    assert d.shape == (3,) and d[0] == 0
    luminance = PAL.astype(float) @ [0.299, 0.587, 0.114]
    assert all(luminance[d[i]] <= luminance[i] for i in range(3))


def test_outline_uniform_unchanged():
    idx = np.full((10, 10), 2, np.int32)
    assert np.array_equal(outline.add_outline(idx, PAL), idx)


def test_outline_draws_dark_pixels_on_edges():
    idx = np.full((12, 12), 2, np.int32)     # light background
    idx[4:8, 4:8] = 1                        # mid-tone object
    out = outline.add_outline(idx, PAL)
    assert out.shape == idx.shape and out.min() >= 0 and out.max() < len(PAL)
    changed = out != idx
    assert changed.any()
    luminance = PAL.astype(float) @ [0.299, 0.587, 0.114]
    assert (luminance[out[changed]] <= luminance[idx[changed]]).all()
    assert not changed[0, :].any()           # far from the edge: untouched


# ------------------------------------------------------------------ cleanup
def test_remove_isolated_pixel():
    idx = np.zeros((7, 7), np.int32)
    idx[3, 3] = 1
    out = cleanup.remove_isolated(idx)
    assert out.dtype == np.int32 and (out == 0).all()


def test_remove_isolated_keeps_small_shapes():
    idx = np.zeros((7, 7), np.int32)
    idx[2:4, 2:4] = 1                        # 2x2 block is not isolated
    idx[5, 1:4] = 2                          # 1-px horizontal line is not isolated
    assert np.array_equal(cleanup.remove_isolated(idx), idx)


def test_fix_one_pixel_lines_removes_l_corner():
    idx = np.zeros((8, 8), np.int32)
    for i in range(1, 7):
        idx[i, i] = 1                        # clean diagonal line
    idx[2, 3] = 1                            # extra pixel: L corner between (2,2) and (3,3)
    out = cleanup.fix_one_pixel_lines(idx)
    assert out[2, 3] == 0
    assert all(out[i, i] == 1 for i in range(1, 7))


def test_fix_one_pixel_lines_keeps_straight_lines_and_blocks():
    idx = np.zeros((8, 8), np.int32)
    idx[1, 1:7] = 1                          # horizontal line
    idx[4:7, 4:7] = 2                        # filled block
    assert np.array_equal(cleanup.fix_one_pixel_lines(idx), idx)


def test_fix_one_pixel_lines_keeps_rectangle_corners():
    idx = np.zeros((9, 9), np.int32)
    idx[1, 1:8] = idx[7, 1:8] = 1            # 1-px rectangle outline
    idx[1:8, 1] = idx[1:8, 7] = 1
    assert np.array_equal(cleanup.fix_one_pixel_lines(idx), idx)
