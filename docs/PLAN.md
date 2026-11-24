# PixelBridge – 역할 분배 및 데드라인 (11/24 – 12/3)

모든 시간은 한국 시간(KST) 기준입니다. "PR 마감"은 **GitHub에 PR이 올라가 있고 본인 테스트 파일이 통과한 상태**를 뜻합니다.
팀장(M1)은 **다음 날 12:00까지** 리뷰하고 merge합니다.

| 멤버 | 이름 | GitHub ID |
|---|---|---|
| M1 (팀장) | | |
| M2 | | |
| M3 | | |
| M4 | | |

## 1. 역할

| | 담당 영역 | 파일 | 테스트 파일 | 리포트 / 발표 |
|---|---|---|---|---|
| **M1 (팀장)** | core + 통합 | `core/color.py`, `core/filters.py`, `core/resize.py` (Module B 비교용 nearest / bilinear / bicubic 포함), `gui/app.py`, 파이프라인, `main.py`, VS 프로젝트, 제출 | `tests/test_core.py` | ① 개요, ② 전체 설계, ③ 타임라인, ④ 토의 과정; 리포트·슬라이드 취합 |
| **M2** | Module A 1–3단계 | `module_a/preprocess.py`, `downsample.py`, `palette.py` | `tests/test_module_a_front.py` | ② Module A 1–3단계; **데모 영상**; 테스트 사진 20장 |
| **M3** | Module A 4–6단계 + 평가 | `module_a/dither.py`, `outline.py`, `cleanup.py`, `evaluation/metrics.py`, `scripts/benchmark_*.py` | `tests/test_module_a_back.py`, `tests/test_evaluation.py` | ② Module A 4–6단계; ⑥ 평가 결과; **블라인드 설문** |
| **M4** | Module B | `module_b/similarity.py`, `scalex.py`, `xbr.py`, `slope.py` (우리 방법) | `tests/test_module_b.py` | ② Module B; xBR 대비 우리 기여점; 픽셀아트·일러스트 테스트셋 |

전원: ⑤ "Team's thoughts" 한 문단 작성, **발표 전체를 영어로 연습** (발표자는 랜덤 선정).

## 2. 날짜별 데드라인

| 날짜 | M1 (팀장) | M2 | M3 | M4 | 팀 체크포인트 |
|---|---|---|---|---|---|
| **11/24 (화)** | 오늘 밤 뼈대를 `main`에 push. PR: `rgb_to_gray/yuv/lab`, `filter2d`, `gaussian_kernel`, `sobel`, `resize_area`, `resize_nearest` (23:59) | clone, `tests/test_wiring.py` 실행, 본인 스텁·테스트 읽기, 브랜치 생성 | 동일 | 동일 | 전원 뼈대 실행 가능 |
| **11/25 (수)** — Stage 1 | 12:00까지 core merge. PR: `resize_bilinear`, `resize_bicubic`. e-Class Q&A로 조교에게 VS **Python** 프로젝트 제출 가능 여부 확인 | PR: `downsample_block_mode`, `median_cut`, `map_to_palette`. `data/photos`에 사진 20장 | PR: `floyd_steinberg`, `bayer`, `psnr`, `ssim` | PR: `similarity`, `scale2x`, `scale3x`, `eagle`. 픽셀아트 30장 + 일러스트 20장 수집 | 23:59: Stage 1 전체 실행 (§4) |
| **11/26 (목)** — Stage 2a | GUI 파라미터, 라운드트립, Windows + VS 2022 테스트 | PR: `bilateral_filter`, `enhance`, `downsample_edge_aware`, `kmeans_lab` | PR: `gradient_mask`, `darker_index`, `add_outline`, `remove_isolated`, `fix_one_pixel_lines` | PR: `xbr` (레벨 1 + 2, 두 모드) | 23:59: 기본 설정으로 Module A 전체; Module B는 xBR까지 |
| **11/27 (금)** — Stage 2b | 통합 버그 수정, `scripts/gen_vsproj.py`, README 사용법 | 사진 20장으로 기본 파라미터 튜닝; 라운드트립 데모 사진 선정 | PR: `mean_delta_e`, `count_isolated_pixels`, `edge_preservation_ratio`; 벤치마크 1차 실행; 설문 초안 | PR: `estimate_slopes`, `upscale_slope` (**18:00 중간점검**, §5 참고) | **23:59 기능 동결** |
| **11/28 (토)** — 코드 마감 22:00 | 18:00 최종 merge + `v1.0` 태그; Windows에서 새로 clone → `.sln` 열기 → 실행 → zip → **21:00까지 제출** | 18:00까지 버그 수정 PR만 | `v1.0`으로 최종 벤치마크 (CSV + 그래프); 설문 배포 | 18:00까지 버그 수정 PR만 | `Team13_Tseerei_Src.zip` 제출 |
| **11/29 (일)** | 12:00까지 리포트·슬라이드 템플릿 공유; ①②③④ 작성 | 본인 섹션 + 슬라이드 2–3장; 데모 영상 촬영 | 본인 섹션 + 슬라이드 | 본인 섹션 + 슬라이드 | 23:59: 전원 초안 제출. 각자 설문 응답자 5명 모집 |
| **11/30 (월)** | 리포트 v1 + 슬라이드 v1 취합 (23:59) | 데모 영상 편집 (2–3분, mp4) | 18:00 설문 마감 → 결과를 리포트·슬라이드에 반영 | 검토 | 전원 v1 검토 |
| **12/1 (화)** — 리포트/PPT 마감 22:00 | 18:00 최종본; 두 zip 모두 **21:00까지 제출** | 영상 최종본 | 수치 최종 확인 | 최종 확인 | 두 zip 제출 |
| **12/2 (수)** | 21:00 온라인 리허설: 전원이 발표 전체를 한 번씩 | | | | |
| **12/3 (목)** | 최종 발표 + 라이브 데모 (데모 영상은 백업으로 준비) | | | | 우리 순서가 12/10이면 12/9에 다시 리허설 |

## 3. 매일 루틴

- **22:00 카카오톡 체크인**: merge된 것 / PR 링크, 막힌 점.
  스크린샷을 남겨 두세요 — 리포트 ④(그룹 토의 과정)에 들어갑니다.
- 1시간 넘게 막히면 체크인을 기다리지 말고 바로 공유하기.
- 본인 파일만 수정하기. 다른 사람 파일이나 파이프라인 수정이 필요하면 담당자(또는 M1)에게 요청.

## 4. Stage 1 확인 (11/25 수, 23:59)

```
python main.py a data/photos/<photo>.jpg results/a.png --downsample block --palette-method median_cut --no-smooth --no-outline --no-cleanup
python main.py a data/photos/<photo>.jpg results/a_fs.png --downsample block --palette-method median_cut --no-smooth --no-outline --no-cleanup --dither fs
python main.py b results/a.png results/b.png --method scale2x --scale 4
```

## 5. 미리 정해 둔 축소 기준

- Stage 3 (애니메이션 스프라이트 시트)는 하지 않음.
- 목요일 23:59까지 `xbr` 테스트 미통과 → M1이 M4 지원; 우리 방법은 xBR 대신 Scale2x/Eagle 위에 구현.
- **금요일 18:00**까지 `upscale_slope` 미완성 → palette 모드만 제출; smooth 모드는 리포트의 향후 과제로.
- Module A가 너무 느리면 → `working_size`를 512에서 256으로 낮춤.

## 6. 제출 체크리스트

**소스 코드 — 11/28 (토) 22:00** (우리는 21:00까지 제출)
- [ ] 최종 `main`에서 `python scripts/gen_vsproj.py` 실행
- [ ] Windows에서 새로 clone해서 Visual Studio 2022로 열고 실행되는지 확인
- [ ] `python scripts/check_rules.py` 결과 OK
- [ ] zip 이름 `Team13_Tseerei_Src.zip` → e-Class / Image Processing / Assignments / Team project 02 - SRC

**리포트 + 발표자료 — 12/1 (화) 22:00** (우리는 21:00까지 제출)
- [ ] 최종 리포트 (.docx): ① 개요 ② 설계 ③ 타임라인 ④ 토의 과정 (카톡 캡처) ⑤ 팀 소감 ⑥ 기타
- [ ] 슬라이드 (.pptx): 전원 **영문 이름** 포함, "Global Team" 표기 유지
- [ ] 데모 영상 (.mp4)
- [ ] `Team13_Tseerei_PPT.zip` → e-Class / Image Processing / Assignments / Team project 02 - PPT
- [ ] `Team13_Tseerei_Final.zip` → e-Class / Team Project / Final

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
