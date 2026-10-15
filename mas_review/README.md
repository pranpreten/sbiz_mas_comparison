# 협력 후검토형

구조별 실행 흐름은 `graph.py`, 역할 지침은 `prompts.py` 또는 그래프의 역할별 지침에 있습니다. 공유 모델·도구·예산·근거 전달은 `common/orchestration`을 사용합니다.

`main_project`에서 실행합니다.

```bat
python -m mas_review.evaluate
python -m mas_review.main "서울 청운효자동에 카페가 얼마나 있어?"
python -m mas_review.evaluate --local-run --limit 2
python -m mas_review.evaluate --run
```

기본 evaluate는 검사만 합니다. `main`, `--local-run`, `--run`은 실제 모델을 호출합니다. `--run`은 LangSmith 설정과 데이터셋·평가기 연결이 필요합니다.

전체 설계·공통 조건·업로드 순서: [ARCHITECTURES.md](../ARCHITECTURES.md). 검증 범위: [VALIDATION.md](../VALIDATION.md).
