# PixelBridge – Roles and Deadlines (Nov 24 – Dec 3)

All times are KST. "PR due" means: the PR is open on GitHub **and** the member's own
test file passes. The leader (M1) reviews and merges by **12:00 the next day**.

| Member | Name | GitHub ID |
|---|---|---|
| M1 (Leader) | | |
| M2 | | |
| M3 | | |
| M4 | | |

## 1. Roles

| | Area | Files | Test file | Report / slides |
|---|---|---|---|---|
| **M1 (Leader)** | Core + integration | `core/color.py`, `core/filters.py`, `core/resize.py` (also the nearest / bilinear / bicubic baselines of Module B), `gui/app.py`, pipelines, `main.py`, VS project, submission | `tests/test_core.py` | ① Outline, ② overall design, ③ timeline, ④ discussion process; assembles the report and slides |
| **M2** | Module A, steps 1–3 | `module_a/preprocess.py`, `downsample.py`, `palette.py` | `tests/test_module_a_front.py` | ② Module A steps 1–3; **demo video**; 20 test photos |
| **M3** | Module A, steps 4–6 + evaluation | `module_a/dither.py`, `outline.py`, `cleanup.py`, `evaluation/metrics.py`, `scripts/benchmark_*.py` | `tests/test_module_a_back.py`, `tests/test_evaluation.py` | ② Module A steps 4–6; ⑥ evaluation results; **blind survey** |
| **M4** | Module B | `module_b/similarity.py`, `scalex.py`, `xbr.py`, `slope.py` (our method) | `tests/test_module_b.py` | ② Module B; our contribution vs. xBR; pixel art + illustration test sets |

Everyone: ⑤ one paragraph of "Team's thoughts", and **rehearse the whole talk in English**
(the presenter is chosen at random).

## 2. Day-by-day deadlines

| Day | M1 (Leader) | M2 | M3 | M4 | Team checkpoint |
|---|---|---|---|---|---|
| **Tue 11/24** | Skeleton on `main` (tonight). PR: `rgb_to_gray/yuv/lab`, `filter2d`, `gaussian_kernel`, `sobel`, `resize_area`, `resize_nearest` (23:59) | Clone, run `tests/test_wiring.py`, read own stubs + tests, create branch | same | same | Everyone can run the skeleton |
| **Wed 11/25** — Stage 1 | Merge core by 12:00. PR: `resize_bilinear`, `resize_bicubic`. Ask the TA (e-Class Q&A) that a VS **Python** project is OK | PR: `downsample_block_mode`, `median_cut`, `map_to_palette`. Put 20 photos in `data/photos` | PR: `floyd_steinberg`, `bayer`, `psnr`, `ssim` | PR: `similarity`, `scale2x`, `scale3x`, `eagle`. Collect 30 pixel art + 20 illustrations | 23:59: Stage 1 runs end to end (see §4) |
| **Thu 11/26** — Stage 2a | GUI parameters, round trip, test on Windows + VS 2022 | PR: `bilateral_filter`, `enhance`, `downsample_edge_aware`, `kmeans_lab` | PR: `gradient_mask`, `darker_index`, `add_outline`, `remove_isolated`, `fix_one_pixel_lines` | PR: `xbr` (level 1 + 2, both modes) | 23:59: full Module A with default settings; Module B with xBR |
| **Fri 11/27** — Stage 2b | Integration bug fixes, `scripts/gen_vsproj.py`, README usage | Tune default parameters on the 20 photos; choose the round-trip demo photos | PR: `mean_delta_e`, `count_isolated_pixels`, `edge_preservation_ratio`; first benchmark run; survey draft | PR: `estimate_slopes`, `upscale_slope` (**18:00 checkpoint**, see §5) | **23:59 feature freeze** |
| **Sat 11/28** — code due 22:00 | 18:00 final merge + tag `v1.0`; clean clone on Windows → open `.sln` → run → zip → **submit by 21:00** | Bug-fix PRs only (until 18:00) | Final benchmark on `v1.0` (CSV + graphs); publish the survey | Bug-fix PRs only (until 18:00) | `Team13_Tseerei_Src.zip` submitted |
| **Sun 11/29** | Report + slide templates by 12:00; write ①②③④ | Own report section + 2–3 slides; shoot the demo video | Own section + slides | Own section + slides | 23:59: all drafts in. Everyone recruits 5 survey respondents |
| **Mon 11/30** | Merge report v1 + slides v1 (23:59) | Edit the demo video (2–3 min, mp4) | 18:00 close the survey → results into report and slides | Review | Everyone reviews v1 |
| **Tue 12/1** — report/PPT due 22:00 | 18:00 final; zip and **submit both by 21:00** | Final video | Final numbers check | Final check | Both zips submitted |
| **Wed 12/2** | 21:00 online rehearsal: each member presents the whole talk once | | | | |
| **Thu 12/3** | Final presentation + live demo (keep the demo video as a backup) | | | | If our slot is 12/10, rehearse again on 12/9 |

## 3. Daily routine

- **22:00 check-in in KakaoTalk**: what was merged / PR link, and any blocker.
  Take screenshots — they go into report section ④ (design process of group discussion).
- Blocked for more than 1 hour? Post it right away; do not wait for the check-in.
- Only edit your own files. If you need a change in someone else's file or in a
  pipeline, ask the owner (or M1).

## 4. Stage 1 check (Wed 11/25, 23:59)

```
python main.py a data/photos/<photo>.jpg results/a.png --downsample block --palette-method median_cut --no-smooth --no-outline --no-cleanup
python main.py a data/photos/<photo>.jpg results/a_fs.png --downsample block --palette-method median_cut --no-smooth --no-outline --no-cleanup --dither fs
python main.py b results/a.png results/b.png --method scale2x --scale 4
```

## 5. Cut lines (decided in advance)

- Stage 3 (animated sprite sheets) is dropped.
- `xbr` not passing its tests by Thu 23:59 → M1 joins M4; our method is built on Scale2x/Eagle instead of xBR.
- `upscale_slope` not working by **Fri 18:00** → submit palette mode only; smooth mode becomes future work in the report.
- Module A too slow → lower `working_size` from 512 to 256.

## 6. Submission checklist

**Source code — Sat 11/28, 22:00** (we submit by 21:00)
- [ ] `python scripts/gen_vsproj.py` run on the final `main`
- [ ] Opens and runs in Visual Studio 2022 from a clean clone (Windows)
- [ ] `python scripts/check_rules.py` prints OK
- [ ] Zip name `Team13_Tseerei_Src.zip` → e-Class / Image Processing / Assignments / Team project 02 - SRC

**Report + presentation — Tue 12/1, 22:00** (we submit by 21:00)
- [ ] Final report (.docx): ① outline ② design ③ timeline ④ discussion process (KakaoTalk captures) ⑤ team's thoughts ⑥ etc.
- [ ] Slides (.pptx) with every member's **English name**, "Global Team" label kept
- [ ] Demo video (.mp4)
- [ ] `Team13_Tseerei_PPT.zip` → e-Class / Image Processing / Assignments / Team project 02 - PPT
- [ ] `Team13_Tseerei_Final.zip` → e-Class / Team Project / Final
