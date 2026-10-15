# 소상공인 QA 비교 실험 — 공통 기반

`single_agent_test`는 파일럿으로 보존하고, 본 실험은 이 디렉터리에서 구성합니다.
현재 **단일 에이전트와 여섯 멀티에이전트 그래프**를 구현했습니다. 구조별 차이·폴더명·실행 명령·공정 비교 조건은 [ARCHITECTURES.md](ARCHITECTURES.md), 개발 검증 결과는 [VALIDATION.md](VALIDATION.md)를 참고하세요. 전체 36문항 평가는 사용자가 실행할 단계입니다.

각 구조의 `--run`은 답변 실행과 LangSmith 기록, 로컬 결과 저장을 마치면 종료합니다. 서버 채점 점수는 자동 조회하지 않습니다. LangSmith에서 채점 완료를 확인한 뒤 결과를 한 번에 수집합니다. 로컬 **[7개 구조 비교표](outputs/comparison.html)**에는 저장된 실행 통계만 반영되며, 미조회 점수는 빈칸입니다. 수동으로 한 번 조회하려면 `python -m common.evaluation.comparison --refresh`를 사용합니다. 조건이 다른 실행은 별도 표로 구분하며 CSV·JSON도 함께 저장합니다.

## 구성

```text
common/
  config.py                 설정·실험 기준일
  model.py                  Solar 모델 생성 (호출 전까지 과금 없음)
  check.py                  문항·도구 스키마·실제 조회 검사
  data/database.py          PostgreSQL 읽기 전용 접근
  tools/
    codes.py                상권용 지역·업종 코드 검색
    market.py               여러 지역의 같은 월 상권 조회
    statistics.py           두 조사 통계 목록·조회
    policy.py               정책 검색·문서 근거 조회
    law.py                  법령 조문·항호목·부칙·별표 조회
    calculator.py           비교·차이·비율 계산
  harness/
    context.py              공통 지침과 질문 내 이력 구성
    limits.py               질문 전체의 호출·시간 예산
    runtime.py              모델·도구 실행, 동일 조회 결과 재사용
    safe_tools.py           도구 본문 오류를 추적 콜백 이전에 정리
    tracing.py              모델 응답별 토큰 집계
  evaluation/
    dataset.py              Excel 읽기·검증·명시적 데이터셋 동기화
    runner.py               구조별 답변 함수의 공통 평가 실행
tests/                      외부 호출 없는 unittest
```

## 설정과 실행

기본 설정은 `main_project/.env`에서 읽습니다. 기존 파일럿의 DB·Upstage·LangSmith 연결값만 새 로컬 `.env`에 재사용했고, 파일럿은 수정하지 않았습니다. 새 `.env`의 추적은 기본 `false`입니다. 실험 기록을 전송할 때 `LANGSMITH_TRACING=true`로 바꾸세요. 키는 `.gitignore`에서 제외하며 코드·문서에 기록하지 않습니다.

현재는 파일럿의 가상환경으로 검증했습니다. 새 환경을 만들 경우 `requirements.txt`를 설치합니다. 아래 명령은 **main_project 폴더에서, 의존성이 설치된 가상환경을 활성화한 상태**에서 실행합니다.

```bat
python -m unittest discover -s tests -v
python -m common.check
python -u -m common.check --live
```

- 첫 번째: 로컬 테스트. 네트워크·LLM 호출 없음.
- 두 번째: 36문항 읽기 및 12개 도구 스키마 검사. DB 연결 없음.
- 세 번째: 설정된 DB의 읽기 전용 대표 조회. 모델 호출·DB 변경·LangSmith 업로드 없음.
- 다른 환경 파일을 쓰려면 `--env 경로`를 명시합니다. 다른 프로젝트 `.env`를 자동 탐색하지 않습니다.

## 공통 도구 12개

| 도구 | 입력과 범위 |
|---|---|
| `search_region_codes` | 지역명, AD_CL/SG_CL/SI_CL, 필요 시 상위 지역 |
| `search_industry_codes` | 카페 등 업종명. 상권 분류 코드이며 KSIC와 다름 |
| `lookup_market_summary` | 확인한 지역코드 목록·업종코드·월. 월 생략 시 최신 공통월 |
| `inspect_statistics` | `small_business` 또는 `traditional_markets`의 지역·업종·연도 |
| `lookup_statistics` | 조사명·지역명 목록·연도·산업 범위. 연도 생략 시 최신 공통연도 |
| `search_policies` | 짧은 키워드 목록 모두 포함하는 공고 후보 |
| `get_policy_document` | 검색한 문서 ID의 메타정보·원문 청크, 페이지·키워드 필터 |
| `search_law_articles` | 키워드·실험 기준일·선택적 법령명 |
| `get_law_article` | 검색한 mst·article_seq의 조문 및 항·호·목 |
| `get_law_addenda` | 선택한 법령 버전의 부칙 |
| `get_law_appendices` | 선택한 법령 버전의 별표. 본문 없는 링크는 별도로 표시 |
| `calculate_values` | 숫자 문자열 두 개의 차이·절대차이·비율·증감률·대소 비교 |

도구 공통 반환은 `status`, `data`, `source`, `conditions`이며 필요에 따라 단위·시점·페이지 정보가 추가됩니다. `found`는 조회 결과 존재를 의미하며 모든 법적 조건이나 지표가 완전하다는 뜻은 아닙니다. `ambiguous`는 동명이칭·불확실 매핑·중복 등으로 추가 확인이 필요합니다. `not_found`와 조회 `error`는 구분합니다. NULL/NaN을 0으로 바꾸지 않습니다.

### 실제 데이터에 맞춘 범위

- 상권: `market.simple_analysis`, `market.industry_search`; 지역은 `stats.sgis_adm_code_map`의 `code_system=sbiz365_adm`, **ext_adm_cd**를 사용합니다. SGIS 자체 코드를 상권 코드로 전용하지 않습니다.
- 통계: 현재 36문항에 필요한 `stats.kosis_small_business_survey_01`, `stats.kosis_traditional_market_survey_01` 두 표를 지원합니다. 전체 통계 테이블을 자동으로 조회하는 Text-to-SQL은 아닙니다.
- 정책: `announcement.policy_document`, `announcement.policy_chunk`의 키워드 검색. 검색 후보의 지역·대상·제외조건·기간은 상세 근거에서 확인해야 합니다. `active`는 수집 상태이며 현재 신청 가능을 뜻하지 않습니다. 임베딩 검색·자동 모집상태 판정은 구현하지 않았습니다.
- 법령: `law.law_master/article/clause/addenda/appendix`. 지정 기준일 이전 최신 **수집 버전**을 조회합니다. 완전한 법령 연혁이나 미수집 개정까지 보장하지 않습니다. 부칙·별표의 본문·누락 상태를 구분합니다.
- 페이지의 `has_more`, `next_offset` 등을 확인하고 필요하면 후속 조회합니다. 큰 문서를 조용히 잘라서 완전한 근거인 것처럼 반환하지 않습니다.

## LangGraph에 연결할 계약

각 구조의 `graph.py`는 아래 공통 기능을 사용합니다. 아래는 공통 연결 설명이며 실제 단일 에이전트 구현은 `single_agent/graph.py`에 있습니다.

```python
from common.config import load_settings
from common.data.database import Database
from common.tools import create_tools
from common.harness import AgentRuntime

settings = load_settings()
db = Database(settings.database_url, settings.statement_timeout_ms)
tools = create_tools(db)

# 매 질문 시작 시 새 runtime. 여러 문항에서 같은 runtime을 재사용하지 않음.
runtime = AgentRuntime(settings, tools)
# agent 노드: runtime.invoke_model(role_prompt, state['messages'], config=config)
# tools 노드: runtime.execute_tools(last_ai_message.tool_calls, config=config)
# graph는 새 메시지를 MessagesState에 누적하고 다음 노드를 결정.
```

- `common`은 역할 분담·그래프·프롬프트를 정하지 않습니다. 역할 지침은 각 구조가 전달합니다.
- 멀티에이전트는 역할마다 runtime을 만들고 같은 `RunBudget`과 `UsageLedger`를 전달해 **질문 전체 예산**을 공유할 수 있습니다. 각 runtime의 메시지·캐시는 역할 내부용입니다. 하나의 runtime을 여러 스레드가 동시에 호출하는 용도는 아닙니다.
- 초기값은 도구 16회·모델 12회·180초입니다. 개발용 시작값이므로 본 비교 전에 고정하세요. 마지막 모델 호출 1회 및 일부 시간을 답변 작성용으로 남깁니다.
- 시간 제한은 **협력적 제한**입니다. 새로운 호출 전 확인과 모델/DB 요청 타임아웃을 사용하지만 실행 중인 작업을 강제 종료하지 않습니다. 시간 초과 시 graph가 중단 상태·확보 근거를 기록해야 하며 최종 LLM 답변이 항상 보장되지는 않습니다.
- `model_calls`, `tool_calls`는 실패한 시도를 포함하는 예산 카운터입니다. 실제 반환된 모델 응답 수는 `responses`에 기록합니다. 컨텍스트 상한 초과도 시도에 포함될 수 있습니다.
- 컨텍스트는 문자 수 상한이며 토큰 수 상한이 아닙니다. 도구 설명의 토큰도 별도로 필요합니다. 상한 초과 시 근거를 조용히 삭제하지 않고 명시적 오류를 반환합니다.
- 도구 입력 검증과 DB 오류를 정리하지만 도구 입력·정상 반환값 자체는 추적 대상입니다. 비밀값을 질문·도구 인자·정상 결과에 넣지 않습니다.
- 내부 자동 재시도는 기본 0회입니다. 재시도는 graph가 같은 예산 안에서 명시적으로 요청합니다. 성공/없음/모호함의 동일 도구 호출은 질문 내 결과를 재사용합니다. 오류는 캐시하지 않습니다.

## Excel과 평가

```python
from common.evaluation import load_examples, sync_dataset, run_experiment

examples = load_examples('../예비실험/예비실험_질문36문항_v0.6.xlsx')
# client는 호출자가 명시적으로 생성한 LangSmith Client.
plan = sync_dataset(client, 'sbiz_eval_36', examples)             # 읽기 전용 계획
# 실제 업로드 필요 시: sync_dataset(client, 'sbiz_eval_36', examples, apply=True)
# 그래프 구현 후: run_experiment(answer_question, dataset_name='sbiz_eval_36',
#                               experiment_prefix='single_agent', client=client)
```

- agent에는 질문만 전달합니다. `required_items`, `reference_evidence`는 outputs에, 문항 ID·유형·분야는 metadata에만 저장합니다.
- 동기화는 문항 ID로 추가/갱신하고 기존 데이터셋·평가기 연결을 보존합니다. 기존 문항 삭제는 하지 않습니다. 새 데이터셋에는 평가기 2개를 별도로 연결해야 합니다.
- 답변 함수는 텍스트, `final_answer`, 또는 마지막 AI 메시지가 있는 상태를 반환합니다. 도구 결과로 끝났거나 도구 호출이 미완료면 채점용 최종 답변으로 인정하지 않습니다.
- `run_experiment`는 명시적으로 호출했을 때 답변 모델과 연결된 서버 평가기를 실행하므로 비용이 발생할 수 있습니다. 새 36문항 데이터셋의 업로드/평가기 연결과 전체 채점은 별도입니다.
- 서버 채점은 비동기입니다. 답변 실행 완료가 평가기 완료를 뜻하지 않습니다.
- 시간과 토큰은 답변 실행의 루트 run 기준입니다. 자식 run의 토큰을 다시 더하지 않으며 Astra 채점 사용량은 별도입니다. 사용량이 없으면 0으로 단정하지 않습니다.
- 정책 추천은 엑셀 예시와 다른 유효한 답이 있을 수 있습니다. 기존 근거일치율 평가기를 36문항에 사용하기 전에 대체 근거의 인정/판정 보류 규칙을 확정해야 합니다. 이 코드가 평가기 프롬프트를 자동 변경하지는 않습니다.

## 검증 기록 (2026-10-14)

- 실제 로컬 복원 DB **sbiz_experiment**, 트랜잭션 `read_only=on` 확인.
- 36문항 로컬 검증, 공통 도구 12개 스키마 생성 확인.
- 오프라인 자동 테스트 **48개 통과**, Solar 모델 설정 및 12개 도구 연결 확인 (모델 실행 없음).
- 청운효자동/사직동/삼청동 카페 85/171/68개 (202605) 조회.
- 서울/부산 소상공인 기업체 1,222,271/373,750개, 전통시장 194/158개 (2024) 조회.
- 정책 검색·원문 청크, 상가건물 임대차보호법 제10조 항호목, 부칙 및 소기업 매출기준 별표 조회 확인.
- SQL 인자 바인딩, 동명 지역, 중복/부분 누락, NULL/NaN, 0 나눗셈, 예산, 비밀값 오류 처리, 정답 분리, 동기화 보존 동작을 오프라인 테스트.
- 위 목록은 최초 공통 도구 검증 기록입니다. 이후 그래프·통합 검증은 VALIDATION.md에 별도로 기록합니다. **36문항 전체 품질 점수는 아직 측정하지 않았습니다.**
