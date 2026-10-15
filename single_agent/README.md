# 단일 에이전트 실행과 평가

한 역할의 Solar 모델이 정책·상권·통계·법령 도구를 직접 선택하고, 조회 결과를 확인한 뒤 필요한 만큼 반복하여 답합니다. 다른 에이전트에 위임하지 않습니다. 기존 `single_agent_test`와 별도로 동작합니다.

## 구조

```text
START → agent ── 도구 호출 있음 → tools ──→ agent
           │
           └─ 최종 답변 또는 실행 오류 → finish → END
```

| 파일 | 역할 |
|---|---|
| `prompts.py` | 한 역할의 업무 지침. 문항별 정답이나 정해진 조회 경로는 포함하지 않음 |
| `state.py` | 메시지·조회 근거·최종 답변·실행 상태·사용량 |
| `graph.py` | 판단 → 도구 실행 → 재판단을 연결하는 LangGraph |
| `service.py` | `SingleAgent(settings).answer(inputs)`. 질문마다 새 상태·실행 예산 생성 |
| `main.py` | 질문 하나 입력해서 실행 |
| `evaluate.py` | 36문항 검증, 로컬 일괄 실행, LangSmith 실험, 늦게 도착한 점수 조회 |
| `reporting.py` | 질문별 JSON·CSV와 전체·유형별 요약 저장 |

모델 설정, 읽기 전용 DB, 12개 도구, 실행 제한과 평가 연결은 `common`에서 가져옵니다. 한 질문에서 도구 여러 개를 요청할 수 있지만 현재 도구 실행기는 순차 실행합니다.

## 실행 위치와 설정

아래 명령은 **`main_project` 폴더에서 가상환경을 활성화한 상태**로 실행합니다.

```bat
cd /d C:\Users\USER\bigdata_sbiz\mas_project\main_project
```

설정은 `main_project/.env`에서 읽습니다. 기본 `LANGSMITH_TRACING=false`이므로 일반 실행은 추적을 전송하지 않습니다. 다른 환경 파일은 `--env 경로`로 지정합니다.

- `DATABASE_URL`: 읽기 전용으로 조회할 DB 연결값
- `UPSTAGE_API_KEY`: 답변 모델 호출 키
- `UPSTAGE_MODEL`: 기본 `solar-pro4`
- `EXPERIMENT_DATE`: 기본 `2026-10-13`
- `EVAL_DATASET_NAME`: 기본 `sbiz_eval_36`
- `LANGSMITH_API_KEY`: LangSmith 기록에 사용하는 별도 키
- `LANGSMITH_TRACING`: LangSmith 실험 실행 시 `true` 필요

OpenAI 채점 키는 LangSmith의 **Provider secrets**에 등록합니다. 로컬 Upstage 키나 LangSmith 키와 용도가 다릅니다.

## 1. 비용 없이 먼저 검사

```bat
python -m single_agent.evaluate
python -m common.check
python -m unittest discover -s tests -v
```

첫 명령은 기본 엑셀 `예비실험_질문36문항_v0.6.xlsx`를 읽고 검증만 합니다. 모델 생성·DB 조회·LangSmith 업로드를 하지 않으며 키도 필요하지 않습니다. 오프라인 테스트 역시 모델을 실제 호출하지 않습니다.

DB 대표 조회까지 검사하려면 다음 명령을 사용합니다. 읽기 전용 조회이며 모델 호출은 없습니다.

```bat
python -m common.check --live
```

## 2. 질문 하나 실행

```bat
python -m single_agent.main "서울 청운효자동에 카페가 얼마나 있어?"
```

질문을 생략하면 입력창이 나옵니다. 모델 호출 비용이 발생합니다. 답변·근거·사용량을 파일에 남기려면 경로를 지정합니다.

```bat
python -m single_agent.main "서울 청운효자동에 카페가 얼마나 있어?" --output outputs\single_question.json
```

`--output`의 동일한 파일 경로를 다시 사용하면 해당 파일을 덮어씁니다.

## 3. 엑셀 질문을 로컬에서 일괄 실행

```bat
python -m single_agent.evaluate --local-run --limit 2
python -m single_agent.evaluate --local-run
```

첫 명령은 엑셀 앞 2문항, 두 번째는 전체 36문항을 실행합니다. **모델 호출 비용은 발생하지만 LangSmith 추적·데이터셋 업로드·서버 채점은 하지 않습니다.** 문항마다 상태를 초기화하며 실패 문항도 결과에 남깁니다.

## 4. LangSmith에 답변과 채점 기록

처음 데이터셋을 준비할 때는 `python -m single_agent.evaluate --upload-only`로 질문·정답근거만 먼저 올릴 수 있습니다. 모델 실행은 없으며, 업로드 후 기존 평가기 두 개를 연결한 다음 `--run`을 실행합니다. 단일·멀티 구조의 평가·보고서 구현은 이제 `common/evaluation`을 함께 사용합니다.

먼저 `.env`의 `LANGSMITH_TRACING=true`와 유효한 `LANGSMITH_API_KEY`를 설정합니다.

```bat
python -m single_agent.evaluate --run --sync --limit 2
```

- `--sync`: 엑셀 전체 36문항을 지정 데이터셋에 추가·갱신합니다. 기존 데이터셋·평가기 연결·관련 없는 문항은 삭제하지 않습니다.
- `--limit 2`: 실제 답변 생성은 앞 2문항만 수행합니다. 데이터셋을 2문항으로 줄이는 옵션이 아닙니다.
- 데이터셋이 이미 준비되어 있고 엑셀 내용과 같으면 `--sync`를 생략합니다.
- `--sync`는 문항의 질문·필수항목·정답근거를 변경할 수 있으므로 의도한 데이터셋 이름인지 확인합니다.

**새 `sbiz_eval_36` 데이터셋에는 파일럿의 평가기 연결이 자동 복사되지 않습니다.** LangSmith에서 기존 `requirement_coverage`, `evidence_consistency` 평가기 두 개의 대상 데이터셋에 `sbiz_eval_36`을 연결해야 이후 실행을 서버에서 채점할 수 있습니다. 기존 데이터셋에 이미 연결된 평가기는 동기화로 삭제하지 않습니다.

전체 실험 실행:

```bat
python -m single_agent.evaluate --run
```

이 명령은 답변을 **새로 생성**합니다. 앞서 `--local-run`으로 36문항을 실행했더라도 그 답변을 재사용하지 않으므로 모델 비용이 다시 발생합니다. 연결된 서버 평가기 호출 비용도 별도입니다.

에이전트에는 질문만 전달합니다. 엑셀의 `required_items`와 `reference_evidence`는 평가용 출력에, 유형·분야는 분석용 메타정보에 저장하며 답변 모델에 전달하지 않습니다.

## 결과 파일과 늦게 도착한 점수

일괄 실행마다 새 폴더를 만듭니다.

```text
outputs/single_agent_날짜_고유번호/
  report.json       질문별 결과·실험 설정·전체 및 유형별 요약
  questions.csv     질문별 답변·시간·토큰·점수
  summary.json      전체 및 질문 유형별 요약
```

LangSmith 채점은 답변 실행 이후에 완료될 수 있습니다. 초기 보고서의 점수가 비어 있으면 저장된 `report.json` 경로로 다시 조회합니다.

```bat
python -m single_agent.evaluate --refresh "outputs\실제결과폴더\report.json"
```

이 명령은 기존 run의 피드백을 한 번 조회하여 새 결과 폴더에 저장합니다. **모델 호출·답변 재생성·평가기 재실행은 하지 않습니다.** 아직 점수가 없으면 빈값을 유지합니다. 평가기가 연결되지 않았거나 인증에 실패한 경우에도 점수가 없을 수 있어 기다리는 것만으로 해결되지는 않습니다.

| 보고서 항목 | 의미 |
|---|---|
| `completed` | 실행에서 근거 조회 후 답변까지 완료. 정답 판정은 아님 |
| `partial` | 근거 부족·조회 오류·실행 한도 등의 이유로 제한된 답변 |
| `failed` | 최종 답변을 완료하지 못한 실행 |
| `requirement_coverage` | 요구사항 충족률. 실제 채점값이 없으면 빈칸 |
| `evidence_consistency` | 근거 일치율. 실제 채점값이 없으면 빈칸 |
| `elapsed_seconds` | 답변 실행 시간. 채점 대기 시간 제외 |
| `input/output/total_tokens` | 답변 생성에 사용한 모델 사용량. 채점 모델 제외 |
| `usage_complete` | 사용량이 모든 모델 응답에 대해 확인되었는지 여부 |

전체·유형별 요약은 실패를 포함한 문항 수, 완료율과 관측된 점수의 분모를 함께 표시합니다. **미채점을 0점으로 계산하지 않습니다.** 토큰 평균은 사용량이 완전한 문항만 사용하므로 관측 수와 누락 수를 함께 확인합니다. 조회 가능한 근거가 있다는 사실만으로 답변 내용의 정확성이 보장되지는 않습니다.

## 실행 제한과 현재 검증 범위

기본 한도는 질문당 모델 12회·도구 16회·180초이며 `.env`에서 조정합니다. 마지막 모델 호출과 일부 시간을 최종 답변 작성용으로 남깁니다. 시간 제한은 다음 호출 시작을 제어하는 방식이므로 이미 진행 중인 조회를 즉시 취소하는 강제 종료 장치는 아닙니다.

본 비교 실험 전에는 모델·도구·데이터 기준시점·예산·평가기 지침을 고정해야 합니다. 단일·독립 복합·선행 결과 활용 복합 문항은 모두 같은 그래프를 사용하며 문항별 정답을 보고 경로를 선택하지 않습니다.

오프라인 테스트 통과와 데이터 조회 성공은 36문항 답변 품질 검증과 구분합니다. 실제 정확도·완료율·시간·토큰은 실행한 보고서와 LangSmith 채점 결과로 확인합니다.

### 개발 검증 기록 (2026-10-14)

- 공통·단일 그래프·평가 자동 테스트 74개 통과, 36문항 및 12개 도구 스키마 검사 완료.
- 실제 Solar/로컬 복원 DB로 상권 단일, 상권+통계 복합, 통계→상권, 정책, 법령, 정책→법령 질문을 실행했습니다.
- 상권·상권+통계·정책·법령 대표 실행은 답변 완료. 통계→상권은 잘못된 인자 후 재조회하여 값을 확보했지만 오류 이력 때문에 partial로 기록했습니다.
- 정책→법령은 12회 모델 호출 한도에서 부분 답변으로 종료했습니다. 업종별 인원 기준은 조회했으나 소기업 매출기준까지 완결하지 못했습니다. 모든 문항의 정답을 보장하는 상태가 아닙니다.
- 개발 중 지침을 수정했고 일부 검사 실행이 겹쳤으므로 이 기록의 시간·토큰을 정식 구조 비교 결과로 사용하지 않습니다. 대표 결과는 outputs/smoke_*.json에 남겼습니다.
- 새 36문항 LangSmith 데이터셋 업로드·평가기 연결·전체 채점은 이번 개발 검증에서 실행하지 않았습니다.
