# PixelBridge (한국어 안내)

사진과 픽셀아트를 양방향으로 변환하는 프로그램 — 디지털영상처리 팀 프로젝트 02,
**Team13 Tseerei** (Global Team). 영문 원본: [README.md](../../README.md)

- **Module A** — 사진 → 픽셀아트: 스무딩, 엣지 인식 다운샘플링, 팔레트(median cut + CIELAB k-means), 디더링, 외곽선, 정리
- **Module B** — 픽셀아트 → 고해상도: nearest / bilinear / bicubic, Scale2x / Scale3x / Eagle, xBR, **우리의 확장 기울기 방법**

Python 3 + NumPy. OpenCV는 **이미지 입출력에만** 사용하고, 모든 영상처리 단계는 직접 구현합니다
(`scripts/check_rules.py`가 자동으로 검사).

역할·데드라인: **[PLAN.md](PLAN.md)** · Git 작업 방식: **[CONTRIBUTING.md](CONTRIBUTING.md)** · 테스트 데이터: **[DATA.md](DATA.md)**

## 빠른 시작

```bash
git clone https://github.com/<계정>/pixel-bridge.git && cd pixel-bridge
python -m pip install -r requirements.txt
python -m pytest tests/test_wiring.py      # 뼈대 확인: 반드시 통과 (13 passed)
python main.py                             # GUI 실행
```

명령줄:

```bash
python main.py a photo.jpg out.png --width 64 --colors 16 --dither fs --save-steps results/steps
python main.py b sprite.png big.png --scale 4 --method ours --mode palette
python main.py roundtrip photo.jpg results/roundtrip --width 64 --scale 4 --method ours
```

아직 구현되지 않은 단계는 `Not implemented yet: TODO(Mx): <함수명>`을 출력합니다.

### Visual Studio 2022

`PixelBridge.sln`을 엽니다 (*Python 개발* 워크로드 필요). `requirements.txt` 패키지가 설치된
Python 환경을 선택하세요. 시작 파일은 `main.py`(인자 없이 실행 → GUI)입니다. 파일을 추가한 뒤에는
`python scripts/gen_vsproj.py`를 실행해야 솔루션 탐색기에 보입니다.

## 프로젝트 구조 (담당자)

```
main.py                     CLI + GUI 실행                          M1
pixelbridge/
  core/      io.py          이미지 읽기/쓰기 (OpenCV는 여기서만)     완료
             color.py       gray, YUV, CIELAB                       M1
             filters.py     neighbor (완료), filter2d, 가우시안, 소벨 M1
             resize.py      area, nearest, bilinear, bicubic        M1
  module_a/  pipeline.py    단계 연결                               완료
             preprocess.py  bilateral 필터, enhance                 M2
             downsample.py  블록 최빈색 / 엣지 인식 다운샘플링       M2
             palette.py     median cut, k-means(CIELAB), 매핑       M2
             dither.py      Floyd-Steinberg, Bayer, 그래디언트 마스크 M3
             outline.py     1픽셀 어두운 외곽선                     M3
             cleanup.py     고립 픽셀, 1픽셀 선 정리                M3
  module_b/  pipeline.py    방법 선택                               완료
             similarity.py  YUV 색 거리                             M4
             scalex.py      Scale2x, Scale3x, Eagle                 M4
             xbr.py         xBR 레벨 1-2                            M4
             slope.py       우리의 확장 기울기 방법                  M4
  evaluation/metrics.py     PSNR, SSIM, Delta E, 고립 픽셀, 엣지     M3
             benchmark.py   벤치마크 실행                           완료
  gui/app.py                Tkinter GUI                             M1
scripts/     benchmark_a.py, benchmark_b.py (M3), check_rules.py, gen_vsproj.py
tests/       담당자별 테스트 파일 (아래 참고)
data/        photos/, pixelart/, illustrations/
```

## 공통 규약 (전원)

| 항목 | 형식 |
|---|---|
| RGB 이미지 | `np.ndarray (H, W, 3) uint8`, 채널 순서 **R, G, B** (`core.io.read_image`가 OpenCV의 BGR을 변환) |
| float 이미지 | `float32 (H, W, 3)`, 값 범위 `[0, 1]` (`core.io.to_float` / `to_uint8`) |
| 팔레트 | `(K, 3) uint8` RGB |
| 인덱스 이미지 | `(H, W) int32`; 픽셀아트 = `palette[indices]` |
| 크기 | numpy 순서 `(out_h, out_w)` — **높이 먼저** |
| 그래디언트 | 정규화 크기 = `gradient_magnitude(gray) / filters.SOBEL_MAX` ∈ [0, 1] |

속도: 벡터화가 쉬운 곳에서 픽셀 단위 Python 반복문을 쓰지 마세요. 몇 개 안 되는 커널 오프셋에 대해서만
반복하고 `core.filters.neighbor(img, dy, dx)`를 쓰면 됩니다 (docstring 참고).
Floyd–Steinberg는 픽셀 반복문을 써도 괜찮습니다 (픽셀아트가 작으므로).

## 테스트 = 완료 기준

| 담당 | 명령 |
|---|---|
| M1 | `python -m pytest tests/test_core.py` |
| M2 | `python -m pytest tests/test_module_a_front.py` |
| M3 | `python -m pytest tests/test_module_a_back.py tests/test_evaluation.py` |
| M4 | `python -m pytest tests/test_module_b.py` |
| 전체 | `python -m pytest` (모든 TODO가 끝나기 전까지는 빨간색) |

테스트에서는 우리 구현을 검증하기 위한 *기준값*으로 OpenCV를 써도 되지만, `pixelbridge/` 안의 프로그램
코드에서는 안 됩니다. 테스트가 통과하고 PR이 merge되면 그 작업은 완료입니다.

## 평가

```bash
python scripts/benchmark_b.py --folder data/illustrations --factor 4   # 방법별 PSNR / SSIM
python scripts/benchmark_a.py --folder data/photos                     # 우리 방법 vs 기준 방법
```

결과는 `results/`에 저장됩니다 (커밋하지 않음).
