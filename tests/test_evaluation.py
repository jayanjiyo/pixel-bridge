"""M3: evaluation/metrics.py

    python -m pytest tests/test_evaluation.py
"""
import math

import numpy as np

from pixelbridge.evaluation import metrics

rng = np.random.default_rng(11)
IMG = rng.integers(0, 256, (40, 50, 3), dtype=np.uint8)


def test_psnr():
    assert metrics.psnr(IMG, IMG) == float("inf")
    a = np.zeros((10, 10, 3), np.uint8)
    b = np.full((10, 10, 3), 10, np.uint8)      # MSE = 100
    assert abs(metrics.psnr(a, b) - 10 * math.log10(255 ** 2 / 100)) < 1e-6


def test_ssim():
    assert abs(metrics.ssim(IMG, IMG) - 1.0) < 1e-6
    assert metrics.ssim(IMG, 255 - IMG) < 0.5
    noisy = np.clip(IMG + rng.normal(0, 20, IMG.shape), 0, 255).astype(np.uint8)
    assert 0 < metrics.ssim(IMG, noisy) < 1


def test_mean_delta_e():
    assert metrics.mean_delta_e(IMG, IMG) == 0
    black = np.zeros((4, 4, 3), np.uint8)
    white = np.full((4, 4, 3), 255, np.uint8)
    assert abs(metrics.mean_delta_e(black, white) - 100) < 1.0   # L: 0 -> 100


def test_count_isolated_pixels():
    idx = np.zeros((9, 9), np.int32)
    idx[4, 4] = 1
    idx[1:3, 1:3] = 2                            # not isolated
    assert metrics.count_isolated_pixels(idx) == 1
    rgb = np.zeros((9, 9, 3), np.uint8)
    rgb[4, 4] = 255
    assert metrics.count_isolated_pixels(rgb) == 1


def test_edge_preservation_ratio():
    img = np.zeros((20, 20, 3), np.uint8)
    img[:, 10:] = 255
    assert metrics.edge_preservation_ratio(img, img) == 1.0
    assert metrics.edge_preservation_ratio(img, np.zeros_like(img)) == 0.0
    flat = np.full((20, 20, 3), 90, np.uint8)
    assert metrics.edge_preservation_ratio(flat, img) == 1.0
