"""공통 기능 검사 CLI. 모델 호출·DB 갱신·데이터셋 전송은 하지 않는다."""
import argparse
import json
from pathlib import Path
from common.config import PROJECT_ROOT, load_settings
from common.data.database import Database
from common.evaluation import load_examples
from common.tools import create_tools


def live_checks(db):
    """실제 데이터에서 대표 조건을 조회한다. 정답을 모델에 전달하지 않는다."""
    from common.tools.codes import find_regions, find_industries
    from common.tools.market import get_market_summary
    from common.tools.statistics import get_statistics
    from common.tools.policy import search_policies, get_policy_document
    from common.tools.law import search_law_articles, get_law_article
    results = []
    readonly = db.fetch_one("SELECT current_setting('transaction_read_only') AS read_only")
    if readonly["read_only"] != "on":
        raise RuntimeError("읽기 전용 연결 확인 실패")
    regions = find_regions(db, "청운효자동", province="서울")
    industry = find_industries(db, "카페")
    if regions["status"] != "found" or industry["status"] != "found":
        raise RuntimeError("코드 검색 검증 실패")
    market = get_market_summary(db, "AD_CL", [regions["data"][0]["region_code"]],
                                industry["data"][0]["industry_code"])
    results.append({"category": "market", "status": market["status"],
                    "period": market["conditions"]["year_month"]})
    for indicator in ("small_business", "traditional_markets"):
        stats = get_statistics(db, indicator, ["서울", "부산"])
        results.append({"category": indicator, "status": stats["status"],
                        "period": stats["conditions"]["year"]})
    policies = search_policies(db, ["소담스퀘어", "강원"])
    if not policies["data"]:
        raise RuntimeError("정책 검색 검증 실패")
    detail = get_policy_document(db, policies["data"][0]["document_id"])
    results.append({"category": "policy", "status": detail["status"],
                    "chunks": len(detail["data"]["chunks"])})
    laws = search_law_articles(db, ["갱신"], "2026-10-13", "상가건물 임대차보호법")
    if not laws["data"]:
        raise RuntimeError("법령 검색 검증 실패")
    row = laws["data"][0]
    detail = get_law_article(db, str(row["mst"]), row["article_seq"], "2026-10-13")
    results.append({"category": "law", "status": detail["status"],
                    "clauses": len(detail["data"]["clauses"])})
    if any(result["status"] != "found" for result in results):
        raise RuntimeError("일부 분야의 대표 조회가 완료되지 않았습니다.")
    return results


def main():
    parser = argparse.ArgumentParser(description="공통 기능 검증: 모델 호출·데이터 변경 없음")
    parser.add_argument("--env", type=Path)
    parser.add_argument("--live", action="store_true", help="설정된 DB에서 읽기 전용 조회")
    parser.add_argument("--workbook", type=Path,
                        default=PROJECT_ROOT.parent / "예비실험" / "예비실험_질문36문항_v0.6.xlsx")
    args = parser.parse_args()
    examples = load_examples(args.workbook)
    # DB 없음으로 도구 스키마만 검사. 모든 도구는 실제 실행 시에만 DB에 접속.
    tools = create_tools(None)
    for tool in tools:
        tool.tool_call_schema.model_json_schema()
    print(f"Validated: {len(examples)} questions, {len(tools)} tool schemas")
    if args.live:
        settings = load_settings(args.env)
        db = Database(settings.database_url, settings.statement_timeout_ms)
        print(json.dumps(live_checks(db), ensure_ascii=False, indent=2))
    print("완료: 모델 호출·DB 변경·LangSmith 업로드 없음")


if __name__ == "__main__":
    main()
