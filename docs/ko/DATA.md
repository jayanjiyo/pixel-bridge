# 테스트 데이터 (한국어)

영문 원본: [data/README.md](../../data/README.md)

| 폴더 | 내용 | 수집 담당 | 용도 |
|---|---|---|---|
| `photos/` | 사람·동물·사물 사진 약 20장 (직접 촬영; 라운드트립 데모용 팀원 사진 포함) | M2 | Module A 개발, `scripts/benchmark_a.py`, 설문 |
| `pixelart/` | 퍼블릭 도메인 픽셀아트 약 30장 (스프라이트, 아이콘) | M4 | Module B 개발, 설문 |
| `illustrations/` | 단색 위주의 고해상도 일러스트 약 20장 (퍼블릭 도메인) | M4 | `scripts/benchmark_b.py` (4배 축소 → 확대 → 원본과 비교) |

규칙: 직접 찍은 사진이나 퍼블릭 도메인 이미지만 사용 (다운로드한 파일은 출처를 `data/SOURCES.md`에 기록);
사진은 커밋 전에 긴 변을 1024px 이하로 줄이기; 픽셀아트는 PNG만 (JPEG는 단색 영역에 노이즈가 생김).
