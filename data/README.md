# Test data

| Folder | Content | Collected by | Used for |
|---|---|---|---|
| `photos/` | ~20 photos of people, animals, objects (taken by us; team member photos for the round-trip demo) | M2 | Module A development, `scripts/benchmark_a.py`, survey |
| `pixelart/` | ~30 public domain pixel art images (sprites, icons) | M4 | Module B development, survey |
| `illustrations/` | ~20 high-resolution flat-color illustrations (public domain) | M4 | `scripts/benchmark_b.py` (shrink ×4, enlarge, compare) |

Rules: only our own photos or public domain images (record the source of each
downloaded file in `SOURCES.md`); resize photos to at most 1024 px on the longer
side before committing; PNG for pixel art (never JPEG — it adds noise to flat colors).
