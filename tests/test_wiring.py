"""Skeleton tests -- must ALWAYS pass (CI blocks the merge otherwise).

Every algorithm is replaced by a trivial fake, so these tests check only
that the pipelines, I/O, benchmark glue and rule checker are wired correctly.
"""
import os
import sys

import numpy as np
import pytest

from pixelbridge.core import filters, io, resize
from pixelbridge.evaluation import benchmark, metrics
from pixelbridge.module_a import cleanup, dither, downsample, outline, palette, preprocess
from pixelbridge.module_a.pipeline import PixelArtParams, photo_to_pixel_art
from pixelbridge.module_b import pipeline as pipe_b
from pixelbridge.module_b import scalex, slope, xbr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def fake_resize(img, out_h, out_w, *args, **kwargs):
    ys = np.linspace(0, img.shape[0] - 1, out_h).round().astype(int)
    xs = np.linspace(0, img.shape[1] - 1, out_w).round().astype(int)
    return img[ys][:, xs]


def fake_repeat(k):
    return lambda img, *a, **kw: np.repeat(np.repeat(img, k, 0), k, 1)


def gray_palette(img, n, *args, **kwargs):
    return np.repeat(np.linspace(0, 255, n).astype(np.uint8)[:, None], 3, axis=1)


@pytest.fixture
def fake_a(monkeypatch):
    monkeypatch.setattr(resize, "resize_area", fake_resize)
    monkeypatch.setattr(preprocess, "bilateral_filter", lambda img, **kw: img)
    monkeypatch.setattr(preprocess, "enhance", lambda img, amount: img)
    monkeypatch.setattr(downsample, "downsample_block_mode", fake_resize)
    monkeypatch.setattr(downsample, "downsample_edge_aware", fake_resize)
    monkeypatch.setattr(palette, "median_cut", gray_palette)
    monkeypatch.setattr(palette, "kmeans_lab", lambda img, n, init=None, **kw: init)
    monkeypatch.setattr(palette, "map_to_palette",
                        lambda img, pal: (img[..., 0].astype(int) * len(pal) // 256).astype(np.int32))
    monkeypatch.setattr(dither, "gradient_mask", lambda img, **kw: np.ones(img.shape[:2], bool))
    zeros = lambda img, pal, **kw: np.zeros(img.shape[:2], np.int32)  # noqa: E731
    monkeypatch.setattr(dither, "floyd_steinberg", zeros)
    monkeypatch.setattr(dither, "bayer", zeros)
    monkeypatch.setattr(outline, "add_outline", lambda idx, pal, **kw: idx)
    monkeypatch.setattr(cleanup, "remove_isolated", lambda idx: idx)
    monkeypatch.setattr(cleanup, "fix_one_pixel_lines", lambda idx: idx)


@pytest.fixture
def fake_b(monkeypatch):
    for name in ("resize_nearest", "resize_bilinear", "resize_bicubic"):
        monkeypatch.setattr(resize, name, lambda img, h, w, *a, **kw: fake_resize(img, h, w))
    monkeypatch.setattr(scalex, "scale2x", fake_repeat(2))
    monkeypatch.setattr(scalex, "eagle", fake_repeat(2))
    monkeypatch.setattr(scalex, "scale3x", fake_repeat(3))
    monkeypatch.setattr(xbr, "xbr", lambda img, scale=2, mode="palette", **kw: fake_repeat(scale)(img))
    monkeypatch.setattr(slope, "upscale_slope",
                        lambda img, scale=2, mode="palette", **kw: fake_repeat(scale)(img))


# ------------------------------------------------------------------ core helpers
def test_neighbor_replicates_border():
    a = np.arange(12).reshape(3, 4)
    right = filters.neighbor(a, 0, 1)
    assert right[0, 0] == a[0, 1] and right[0, 3] == a[0, 3]
    up = filters.neighbor(a, -1, 0)
    assert up[0, 2] == a[0, 2] and up[2, 2] == a[1, 2]
    rgb = np.random.default_rng(0).integers(0, 256, (5, 6, 3), dtype=np.uint8)
    assert filters.neighbor(rgb, 2, -1).shape == rgb.shape


def test_io_roundtrip(tmp_path):
    img = np.random.default_rng(1).integers(0, 256, (7, 9, 3), dtype=np.uint8)
    path = str(tmp_path / "한글 경로" / "x.png")
    io.write_image(path, img)
    assert np.array_equal(io.read_image(path), img)


def test_io_alpha_is_composited(tmp_path):
    import cv2
    rgba = np.zeros((2, 2, 4), np.uint8)
    rgba[0, 0] = (0, 0, 255, 255)   # opaque red (BGRA)
    path = str(tmp_path / "a.png")
    cv2.imwrite(path, rgba)          # fully transparent elsewhere
    img = io.read_image(path, background=(255, 255, 255))
    assert tuple(img[0, 0]) == (255, 0, 0)
    assert tuple(img[1, 1]) == (255, 255, 255)


# ------------------------------------------------------------------ pipelines
@pytest.mark.parametrize("changes", [
    {},
    {"downsample": "block", "palette_method": "median_cut", "outline": False, "cleanup": False},
    {"dither": "fs"},
    {"dither": "bayer", "dither_flat_only": False},
    {"palette": "handheld_green"},
])
def test_module_a_pipeline(fake_a, changes):
    photo = np.random.default_rng(2).integers(0, 256, (300, 600, 3), dtype=np.uint8)
    params = PixelArtParams(out_width=64, n_colors=8, **changes)
    r = photo_to_pixel_art(photo, params)
    assert r.image.shape == (32, 64, 3) and r.image.dtype == np.uint8
    assert r.indices.shape == (32, 64) and r.indices.dtype == np.int32
    assert np.array_equal(r.image, r.palette[r.indices])
    assert "preprocess" in r.timings and "1_preprocess" in r.steps


def test_module_b_pipeline(fake_b):
    img = np.random.default_rng(3).integers(0, 256, (10, 12, 3), dtype=np.uint8)
    for method, scales in pipe_b.SUPPORTED_SCALES.items():
        for s in scales:
            out = pipe_b.upscale(img, s, method)
            assert out.shape == (10 * s, 12 * s, 3), (method, s)
    with pytest.raises(ValueError):
        pipe_b.upscale(img, 3, "scale2x")
    results = pipe_b.upscale_all(img, 4)
    assert "scale3x" not in results and results["ours"]["image"].shape == (40, 48, 3)


def test_upscale_all_reports_missing_methods():
    img = np.zeros((4, 4, 3), np.uint8)
    results = pipe_b.upscale_all(img, 2, methods=("ours",))
    r = results["ours"]
    assert (r["image"] is None and "TODO" in r["error"]) or r["image"] is not None


# ------------------------------------------------------------------ benchmark glue
def test_benchmark_glue(fake_a, fake_b, monkeypatch, tmp_path):
    monkeypatch.setattr(metrics, "psnr", lambda a, b: 30.0)
    monkeypatch.setattr(metrics, "ssim", lambda a, b: 0.9)
    monkeypatch.setattr(metrics, "mean_delta_e", lambda a, b: 5.0)
    monkeypatch.setattr(metrics, "count_isolated_pixels", lambda a: 3)
    monkeypatch.setattr(metrics, "edge_preservation_ratio", lambda a, b, **kw: 0.8)
    img = np.random.default_rng(4).integers(0, 256, (130, 90, 3), dtype=np.uint8)
    path = str(tmp_path / "img.png")
    io.write_image(path, img)

    small, ref = benchmark.make_lowres_pair(img, 4)
    assert ref.shape == (128, 88, 3) and small.shape == (32, 22, 3)

    rows_b = benchmark.benchmark_module_b([path], factor=4)
    assert {r["method"] for r in rows_b} == {m for m in pipe_b.METHODS if 4 in pipe_b.SUPPORTED_SCALES[m]}
    rows_a = benchmark.benchmark_module_a([path], PixelArtParams(out_width=32))
    assert [r["method"] for r in rows_a] == ["ours", "baseline"]

    benchmark.save_csv(rows_b, str(tmp_path / "b.csv"))
    summary = benchmark.summarize(rows_b, ["psnr", "ssim"])
    assert summary["ours"]["psnr"] == 30.0


# ------------------------------------------------------------------ rules / CLI
def test_skeleton_follows_course_rules():
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import check_rules
    assert check_rules.check() == []


def test_cli_module_a(fake_a, tmp_path):
    import main
    photo = np.random.default_rng(5).integers(0, 256, (100, 200, 3), dtype=np.uint8)
    src, dst = str(tmp_path / "p.png"), str(tmp_path / "out.png")
    io.write_image(src, photo)
    assert main.main(["a", src, dst, "--width", "40", "--save-steps", str(tmp_path / "steps")]) == 0
    assert io.read_image(dst).shape == (20, 40, 3)
    assert os.path.exists(tmp_path / "steps" / "1_preprocess.png")
