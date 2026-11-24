# PixelBridge

Two-way conversion between photos and pixel art — Digital Image Processing Team Project 02,
**Team13 Tseerei** (Global Team).

- **Module A** — photo → pixel art: smoothing, edge-aware downsampling, palette (median cut + k-means in CIELAB), dithering, outlines, cleanup
- **Module B** — pixel art → high resolution: nearest / bilinear / bicubic, Scale2x / Scale3x / Eagle, xBR, and **our extended slope method**

Python 3 + NumPy. OpenCV is used **only for image input/output**; every image
processing step is our own code (checked by `scripts/check_rules.py`).

Roles and deadlines: **[docs/PLAN.md](docs/PLAN.md)** · Git workflow: **[CONTRIBUTING.md](CONTRIBUTING.md)** · 한국어: **[docs/ko/README.md](docs/ko/README.md)**

## Quick start

```bash
git clone https://github.com/<account>/pixel-bridge.git && cd pixel-bridge
python -m pip install -r requirements.txt
python -m pytest tests/test_wiring.py      # skeleton check: must pass
python main.py                             # GUI
```

Command line:

```bash
python main.py a photo.jpg out.png --width 64 --colors 16 --dither fs --save-steps results/steps
python main.py b sprite.png big.png --scale 4 --method ours --mode palette
python main.py roundtrip photo.jpg results/roundtrip --width 64 --scale 4 --method ours
```

Until a step is implemented, the program prints `Not implemented yet: TODO(Mx): <function>`.

### Visual Studio 2022

Open `PixelBridge.sln` (needs the *Python development* workload). Select a Python
environment with the packages from `requirements.txt`; the startup file is `main.py`
(no arguments → GUI). After adding files, run `python scripts/gen_vsproj.py` so they
appear in Solution Explorer.

## Project structure (owner)

```
main.py                     CLI + GUI launcher                         M1
pixelbridge/
  core/      io.py          image read/write (OpenCV only here)        done
             color.py       gray, YUV, CIELAB                          M1
             filters.py     neighbor (done), filter2d, Gaussian, Sobel M1
             resize.py      area, nearest, bilinear, bicubic           M1
  module_a/  pipeline.py    connects the steps                         done
             preprocess.py  bilateral filter, enhance                  M2
             downsample.py  block-mode / edge-aware downsampling       M2
             palette.py     median cut, k-means (CIELAB), mapping      M2
             dither.py      Floyd-Steinberg, Bayer, gradient mask      M3
             outline.py     one-pixel dark outline                     M3
             cleanup.py     isolated pixels, one-pixel lines           M3
  module_b/  pipeline.py    method dispatch                            done
             similarity.py  YUV color distance                         M4
             scalex.py      Scale2x, Scale3x, Eagle                    M4
             xbr.py         xBR levels 1-2                             M4
             slope.py       our extended slope method                  M4
  evaluation/metrics.py     PSNR, SSIM, Delta E, isolated px, edges    M3
             benchmark.py   benchmark runners                          done
  gui/app.py                Tkinter GUI                                M1
scripts/     benchmark_a.py, benchmark_b.py (M3), check_rules.py, gen_vsproj.py
tests/       one test file per owner (see below)
data/        photos/, pixelart/, illustrations/  (see data/README.md)
```

## Conventions (everyone)

| Thing | Format |
|---|---|
| RGB image | `np.ndarray (H, W, 3) uint8`, channel order **R, G, B** (`core.io.read_image` converts from OpenCV's BGR) |
| Float image | `float32 (H, W, 3)` in `[0, 1]` (`core.io.to_float` / `to_uint8`) |
| Palette | `(K, 3) uint8` RGB |
| Index image | `(H, W) int32`; pixel art = `palette[indices]` |
| Sizes | numpy order: `(out_h, out_w)` — height first |
| Gradients | normalized magnitude = `gradient_magnitude(gray) / filters.SOBEL_MAX` ∈ [0, 1] |

Speed: no Python loops over pixels where a vectorized version is easy. Loop over the
few kernel offsets and use `core.filters.neighbor(img, dy, dx)` (see its docstring).
Pixel loops are acceptable for Floyd–Steinberg (pixel art is small).

## Tests = definition of done

| Owner | Command |
|---|---|
| M1 | `python -m pytest tests/test_core.py` |
| M2 | `python -m pytest tests/test_module_a_front.py` |
| M3 | `python -m pytest tests/test_module_a_back.py tests/test_evaluation.py` |
| M4 | `python -m pytest tests/test_module_b.py` |
| all | `python -m pytest` (red until every TODO is done) |

Tests may use OpenCV as a *reference* to check our implementations; the program in
`pixelbridge/` may not. A task is done when its tests pass and the PR is merged.

## Evaluation

```bash
python scripts/benchmark_b.py --folder data/illustrations --factor 4   # PSNR / SSIM per method
python scripts/benchmark_a.py --folder data/photos                     # ours vs. baseline
```

Results go to `results/` (not committed).
