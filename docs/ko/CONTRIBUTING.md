# Git 작업 방식 (한국어)

영문 원본: [CONTRIBUTING.md](../../CONTRIBUTING.md)

## 브랜치

- `main`에 직접 push하지 않습니다. 모든 변경은 PR로 올립니다 (팀장이 Settings → Branches에서
  브랜치 보호를 켭니다. 계정 플랜상 안 되면 팀 규칙으로 지킵니다).
- 작업마다 브랜치 하나: `<멤버>/<영역>-<기능>`
  - `m1/core-color`, `m2/a-palette`, `m3/a-dither`, `m4/b-scale2x`
- PR은 작게: 함수 한두 개씩이 가장 좋습니다 (리뷰가 쉽고 충돌이 적음).

## 매일 작업 순서

```bash
git switch main && git pull                       # 최신 main에서 시작
git switch -c m2/a-palette                        # 작업용 새 브랜치
# ... 구현 후:
python -m pytest tests/test_module_a_front.py     # 본인 테스트 파일
python scripts/check_rules.py                     # OpenCV는 입출력에만
git add -A && git commit -m "feat(a/palette): median cut and map_to_palette"
git push -u origin m2/a-palette                   # GitHub에서 PR 생성
```

작업 중에 `main`이 바뀌었다면: `git pull --rebase origin main` → 충돌 해결 → 다시 push
(`git push --force-with-lease`).

## Pull Request

- PR 템플릿을 채웁니다 (테스트 결과 줄, 처리 시간, 시각적인 결과면 전/후 이미지).
- CI가 규칙 검사와 뼈대 테스트를 돌립니다. 둘 다 초록색이어야 합니다.
- 리뷰어: M1 (M1이 바쁘면 다른 멤버 누구나). **Squash and merge**로 합칩니다.
- 마감 = `docs/PLAN.md`에 적힌 날 23:59까지 본인 테스트가 통과한 상태로 PR 생성.

## 커밋 메시지

`feat(a/palette): ...`, `fix(b/xbr): ...`, `test(core): ...`, `docs: ...`

## 규칙

- 본인 파일만 수정합니다 (README "프로젝트 구조" 참고). 다른 파일 수정이 필요하면 담당자에게 요청.
- `pixelbridge/` 안에서는 OpenCV(`core/io.py` 제외), SciPy, scikit-image, Pillow 등을 호출하지 않습니다.
  라이브러리 내장 함수는 `tests/`에서 우리 결과를 검증할 때만 사용합니다.
- 테스트 이미지: 직접 찍은 사진이나 퍼블릭 도메인 이미지만, 파일당 약 1 MB 이하.
- `results/`, zip 파일, IDE 폴더는 커밋하지 않습니다 (`.gitignore`에 포함).
