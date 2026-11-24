"""M1 (Leader): core/color.py, core/filters.py, core/resize.py

    python -m pytest tests/test_core.py

OpenCV appears here ONLY as a reference to check our own implementations.
"""
import cv2
import numpy as np
import pytest

from pixelbridge.core import color, filters, resize

rng = np.random.default_rng(42)
IMG = rng.integers(0, 256, (24, 30, 3), dtype=np.uint8)


# ------------------------------------------------------------------ color
def test_gray_weights():
    g = color.rgb_to_gray(np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255]], np.uint8))
    assert g.dtype == np.float32
    assert np.allclose(g, [0.299, 0.587, 0.114, 1.0], atol=1e-3)


def test_yuv_known_values_and_roundtrip():
    yuv = color.rgb_to_yuv(np.array([[255, 255, 255], [128, 128, 128]], np.uint8))
    assert np.allclose(yuv[:, 0], [1.0, 128 / 255], atol=1e-3)   # Y
    assert np.allclose(yuv[:, 1:], 0, atol=1e-3)                  # gray -> U = V = 0
    back = color.yuv_to_rgb(color.rgb_to_yuv(IMG))
    assert back.shape == IMG.shape
    assert np.abs(back * 255 - IMG).max() < 1.0


def test_lab_matches_opencv():
    ours = color.rgb_to_lab(IMG)
    ref = cv2.cvtColor(IMG.astype(np.float32) / 255, cv2.COLOR_RGB2Lab)
    assert ours.shape == IMG.shape and ours.dtype == np.float32
    assert np.abs(ours - ref).max() < 0.5


def test_lab_roundtrip_and_palette_shape():
    back = color.lab_to_rgb(color.rgb_to_lab(IMG))
    assert np.abs(back * 255 - IMG).max() < 1.0
    pal = np.array([[0, 0, 0], [255, 255, 255]], np.uint8)
    lab = color.rgb_to_lab(pal)
    assert lab.shape == (2, 3)
    assert np.allclose(lab[:, 0], [0, 100], atol=0.5)


# ------------------------------------------------------------------ filters
def test_filter2d_matches_opencv():
    gray = rng.random((20, 25)).astype(np.float32)
    k = rng.random((3, 5)).astype(np.float32)
    ref = cv2.filter2D(gray, -1, k, borderType=cv2.BORDER_REPLICATE)
    out = filters.filter2d(gray, k)
    assert out.shape == gray.shape and out.dtype == np.float32
    assert np.allclose(out, ref, atol=1e-4)


def test_filter2d_color_image():
    out = filters.filter2d(IMG, np.ones((3, 3), np.float32) / 9)
    assert out.shape == IMG.shape


def test_gaussian_kernel():
    k = filters.gaussian_kernel(11, 1.5)
    assert k.shape == (11, 11)
    assert abs(k.sum() - 1) < 1e-5
    assert k[5, 5] == k.max() and np.allclose(k, k.T)


def test_sobel_matches_opencv_and_sign():
    gray = rng.random((20, 25)).astype(np.float32)
    gx, gy = filters.sobel(gray)
    assert np.allclose(gx, cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_REPLICATE), atol=1e-4)
    assert np.allclose(gy, cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3, borderType=cv2.BORDER_REPLICATE), atol=1e-4)
    step = np.zeros((5, 6), np.float32)
    step[:, 3:] = 1.0                      # brighter to the right
    sx, sy = filters.sobel(step)
    assert sx[2, 2] > 0 and np.allclose(sy, 0)
    mag = filters.gradient_magnitude(step)
    assert np.allclose(mag, np.hypot(sx, sy))


# ------------------------------------------------------------------ resize
def test_area_integer_factor_is_block_mean():
    out = resize.resize_area(IMG, 12, 10)
    ref = IMG.reshape(12, 2, 10, 3, 3).astype(np.float64).mean(axis=(1, 3))
    assert out.shape == (12, 10, 3) and out.dtype == np.uint8
    assert np.abs(out.astype(int) - np.round(ref)).max() <= 1


def test_area_non_integer_factor_shape():
    out = resize.resize_area(IMG, 7, 11)
    assert out.shape == (7, 11, 3)
    flat = np.full((24, 30, 3), 77, np.uint8)
    assert np.all(resize.resize_area(flat, 7, 11) == 77)


def test_nearest_integer_upscale_is_repeat():
    out = resize.resize_nearest(IMG, 72, 90)
    assert np.array_equal(out, np.repeat(np.repeat(IMG, 3, 0), 3, 1))


@pytest.mark.parametrize("size", [(48, 60), (72, 90), (96, 120), (31, 47)])
def test_bilinear_matches_opencv(size):
    out = resize.resize_bilinear(IMG, *size)
    ref = cv2.resize(IMG, (size[1], size[0]), interpolation=cv2.INTER_LINEAR)
    assert out.shape == ref.shape and out.dtype == np.uint8
    assert np.abs(out.astype(int) - ref).max() <= 1


@pytest.mark.parametrize("size", [(48, 60), (96, 120)])
def test_bicubic_matches_opencv_with_a075(size):
    out = resize.resize_bicubic(IMG, *size, a=-0.75)
    ref = cv2.resize(IMG, (size[1], size[0]), interpolation=cv2.INTER_CUBIC)
    assert out.shape == ref.shape and out.dtype == np.uint8
    assert np.abs(out.astype(int) - ref).max() <= 2


def test_bicubic_keys_default():
    flat = np.full((8, 8, 3), 200, np.uint8)
    assert np.all(resize.resize_bicubic(flat, 32, 32) == 200)
    out = resize.resize_bicubic(IMG, 48, 60)
    assert out.shape == (48, 60, 3) and out.dtype == np.uint8


def test_resize_keeps_float_dtype_and_gray():
    f = IMG.astype(np.float32) / 255
    assert resize.resize_bilinear(f, 48, 60).dtype == np.float32
    assert resize.resize_nearest(IMG[..., 0], 48, 60).shape == (48, 60)
