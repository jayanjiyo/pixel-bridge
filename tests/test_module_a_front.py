"""M2: module_a/preprocess.py, downsample.py, palette.py

    python -m pytest tests/test_module_a_front.py
"""
import numpy as np
import pytest

from pixelbridge.module_a import downsample, palette, preprocess

rng = np.random.default_rng(7)


def blocky(out_h=6, out_w=8, k=5):
    """Image made of k x k single-color blocks, and its small version."""
    small = rng.integers(0, 256, (out_h, out_w, 3), dtype=np.uint8)
    return np.repeat(np.repeat(small, k, 0), k, 1), small


# ------------------------------------------------------------------ preprocess
def test_bilateral_keeps_constant_image():
    flat = np.full((40, 50, 3), (120, 60, 200), np.uint8)
    out = preprocess.bilateral_filter(flat)
    assert out.shape == flat.shape and out.dtype == np.uint8
    assert np.abs(out.astype(int) - flat).max() <= 1


def test_bilateral_keeps_edges_and_smooths_noise():
    img = np.zeros((40, 40, 3), np.float32)
    img[:, 20:] = 200
    noisy = np.clip(img + rng.normal(0, 8, img.shape), 0, 255).astype(np.uint8)
    out = preprocess.bilateral_filter(noisy).astype(float)
    assert out[:, 22:, :].mean() - out[:, :18, :].mean() > 150   # edge kept
    assert out[:, 25:35].std() < noisy[:, 25:35].astype(float).std()  # noise reduced


def test_enhance():
    img = rng.integers(60, 200, (20, 20, 3), dtype=np.uint8)
    assert np.abs(preprocess.enhance(img, 0.0).astype(int) - img).max() <= 1
    out = preprocess.enhance(img, 1.0)
    assert out.shape == img.shape and out.dtype == np.uint8
    assert out.astype(float).std() > img.astype(float).std()


# ------------------------------------------------------------------ downsample
def test_target_size():
    assert downsample.target_size(300, 600, 64) == (32, 64)


@pytest.mark.parametrize("fn", [downsample.downsample_block_mode, downsample.downsample_edge_aware])
def test_downsample_single_color_blocks(fn):
    big, small = blocky()
    out = fn(big, 6, 8)
    assert out.shape == (6, 8, 3) and out.dtype == np.uint8
    assert np.abs(out.astype(int) - small).max() <= 1


def test_block_mode_takes_majority_not_average():
    block = np.zeros((4, 4, 3), np.uint8)
    block[:] = (250, 250, 250)
    block[:, :1] = (0, 0, 255)            # 25% blue, 75% white
    out = downsample.downsample_block_mode(block, 1, 1)
    assert np.abs(out[0, 0].astype(int) - 250).max() <= 2


def test_edge_aware_keeps_thin_dark_line():
    img = np.full((40, 40, 3), 230, np.uint8)
    img[18:21, :] = 10                      # 3-px dark line, blocks are 8 px
    out = downsample.downsample_edge_aware(img, 5, 5)
    assert (out[..., 0] < 100).any()        # the line survives somewhere


# ------------------------------------------------------------------ palette
def test_fixed_palettes():
    for name, pal in palette.FIXED_PALETTES.items():
        assert pal.ndim == 2 and pal.shape[1] == 3 and pal.dtype == np.uint8, name


def four_color_image():
    """30 x 30 image with exactly 225 pixels of each of 4 colors."""
    colors = np.array([[200, 30, 30], [30, 200, 30], [30, 30, 200], [240, 240, 240]], np.uint8)
    idx = rng.permutation(np.repeat(np.arange(4), 225)).reshape(30, 30)
    return colors[idx], colors


def test_median_cut_recovers_distinct_colors():
    img, colors = four_color_image()
    pal = palette.median_cut(img, 4)
    assert pal.dtype == np.uint8 and pal.shape[1] == 3 and len(pal) <= 4
    for c in colors:
        assert np.abs(pal.astype(int) - c).sum(axis=1).min() <= 3


def test_median_cut_limits_colors():
    img = rng.integers(0, 256, (40, 40, 3), dtype=np.uint8)
    assert 1 <= len(palette.median_cut(img, 8)) <= 8


def test_kmeans_two_clusters():
    a = np.clip(rng.normal(50, 5, (200, 3)), 0, 255)
    b = np.clip(rng.normal(200, 5, (200, 3)), 0, 255)
    img = np.concatenate([a, b]).reshape(20, 20, 3).astype(np.uint8)
    init = np.array([[100, 100, 100], [150, 150, 150]], np.uint8)
    pal = palette.kmeans_lab(img, 2, init=init)
    assert pal.shape == (2, 3) and pal.dtype == np.uint8
    centers = sorted(pal.mean(axis=1))
    assert abs(centers[0] - 50) < 8 and abs(centers[1] - 200) < 8


def test_kmeans_without_init():
    img, _ = four_color_image()
    assert palette.kmeans_lab(img, 4).shape[1] == 3


def test_map_to_palette():
    pal = np.array([[0, 0, 0], [255, 255, 255], [255, 0, 0]], np.uint8)
    img = np.array([[[20, 20, 20], [230, 230, 230], [220, 30, 20]]], np.uint8)
    idx = palette.map_to_palette(img, pal)
    assert idx.shape == (1, 3) and idx.dtype == np.int32
    assert idx.tolist() == [[0, 1, 2]]
