"""Bounded, explicitly labelled views. Original tool records remain untouched.

No LLM summary or invented facts: text selections carry offsets into the original
field, and omitted list records are counted. Numeric values and provenance in
retained records are unchanged. A view is never proof of exhaustive coverage.
"""
from copy import deepcopy
import json
import re


def dumps(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def candidate_view(value, *, final=False):
    """Search hits are navigation, not evidence: keep every title/id, omit previews.

    Full records remain in the run. Detailed article text is never affected.
    """
    if isinstance(value, list):
        return [candidate_view(item, final=final) for item in value]
    if not isinstance(value, dict):
        return deepcopy(value)
    data = value.get("data")
    if isinstance(data, dict) and isinstance(data.get("document"), dict) and isinstance(data.get("chunks"), list):
        # A fetched detail page is evidence, not another search preview. Repeated
        # header-only excerpts make workers keep paginating past available facts.
        return {**{k: deepcopy(v) for k, v in value.items() if k != "data"}, "data": {
            **{k: deepcopy(v) for k, v in data.items() if k != "chunks"},
            "policy_chunks": deepcopy(data["chunks"])}}
    if isinstance(data, dict) and isinstance(data.get("article"), dict) and isinstance(data.get("clauses"), list):
        # Paragraphs are logical conditions, not an arbitrary top-k list. Keep
        # every retrieved paragraph/item verbatim, removing only null metadata.
        return {**{k: deepcopy(v) for k, v in value.items() if k != "data"}, "data": {
            "article": {k: deepcopy(v) for k, v in data["article"].items() if v is not None and k != "loaded_at"},
            "legal_clauses": [{k: deepcopy(v) for k, v in clause.items() if v is not None} for clause in data["clauses"]]},
            "_legal_view": "조회된 모든 항·호·목의 내용과 순서를 보존했습니다. null 필드와 적재시각만 제외했습니다."}
    if value.get("detail_required") is True and isinstance(value.get("data"), list):
        rows = value["data"]
        if rows and all(isinstance(row, dict) and "article_seq" in row and "mst" in row for row in rows):
            if final:
                return {**{k: deepcopy(v) for k, v in value.items() if k != "data"},
                        "_candidate_view": "검색 후보는 최종 답변 근거가 아니므로 목록을 생략했습니다. 별도로 조회한 상세 조문만 근거로 사용하세요."}
            fields = {"mst", "law_nm", "article_seq", "article_title", "article_label", "enforce_date"}
            return {**{k: deepcopy(v) for k, v in value.items() if k != "data"},
                    "data": [{k: deepcopy(v) for k, v in row.items() if k in fields} for row in rows],
                    "_candidate_view": "모든 후보의 제목·식별자 목록입니다. 미리보기·반복 메타데이터는 생략했습니다. get_law_article로 실제 조문을 읽으세요."}
    return {key: candidate_view(item, final=final) for key, item in value.items()}


def _score(value, terms):
    text = dumps(value).casefold()
    return sum(text.count(term) for term in terms)


def compact_view(value, max_chars, question, *, minimal=False):
    original = dumps(value)
    if len(original) <= max_chars and not minimal:
        return deepcopy(value)
    terms = list(dict.fromkeys(re.findall(r"[\w가-힣]{2,}", question.casefold())))
    protected = {"question", "instruction", "source", "source_url", "conditions", "args",
                 "status", "stop_reason", "error_type", "unit", "as_of", "depends_on",
                 "deliverables", "required_sources", "missing_sources", "execution_contract", "planning_contract"}
    protected.update({"article_title", "article_label", "article_seq", "law_nm", "mst"})
    protected.update({"legal_clauses", "article_content", "policy_chunks"})

    def shrink(item, text_limit, list_limit, key=""):
        if key in protected or key.endswith(("_id", "_code", "_date")):
            return deepcopy(item)
        if isinstance(item, str) and len(item) > text_limit:
            # Keep a leading passage for context plus a query-matching passage.
            width = max(40, text_limit // 2)
            start = max(((item.casefold().find(t), t) for t in terms if t in item.casefold()),
                        key=lambda pair: len(pair[1]), default=(0, ""))[0]
            start = max(0, start - width // 3)
            ranges = [(0, width)]
            if start > width:
                ranges.append((start, min(len(item), start + width)))
            else:
                ranges = [(0, min(len(item), text_limit))]
            return {"verbatim_excerpts": [{"start": a, "end": b, "text": item[a:b]} for a, b in ranges],
                    "original_chars": len(item), "omitted_chars": len(item) - sum(b-a for a,b in ranges)}
        if isinstance(item, dict):
            return {k: shrink(v, text_limit, list_limit, k) for k, v in item.items()}
        if isinstance(item, list):
            selected = list(range(len(item)))
            # Keep registry indices and dependency IDs stable; only select data lists.
            article_candidates = bool(item) and all(isinstance(row, dict) and "article_seq" in row and "article_title" in row for row in item)
            if not article_candidates and key not in {"original_evidence", "original_evidence_indices", "depends_on"} and len(item) > list_limit:
                selected = sorted(sorted(selected, key=lambda i: (-_score(item[i], terms), i))[:list_limit])
            result = [shrink(item[i], text_limit, list_limit) for i in selected]
            if len(selected) < len(item):
                result.append({"_selection": "omitted_records", "count": len(item)-len(selected),
                               "kept_original_indices": selected})
            return result
        return item

    levels = ((100, 1),) if minimal else ((2000, 6), (1000, 4), (500, 3), (250, 2), (100, 1))
    for text_limit, list_limit in levels:
        view = shrink(value, text_limit, list_limit)
        if isinstance(view, dict):
            view["_context_selection"] = {
                "partial_view": True, "original_chars": len(original),
                "notice": "원문 일부 발췌입니다. 생략된 내용을 확인한 것으로 판단하지 마세요. "
                          "ID로 원문을 다시 조회할 수 있으며 전체 원문은 실행 기록에 보존됩니다.",
            }
        if minimal and len(original) <= len(dumps(view)) and len(original) <= max_chars:
            return deepcopy(value)
        if len(dumps(view)) <= max_chars:
            return view
    # No silent truncation of identifiers, conditions, numbers or user question.
    from common.harness.context import ContextLimitExceeded
    raise ContextLimitExceeded("필수 식별자와 근거 목록이 전달 예산을 초과했습니다.")


def report_payload(reports, original=()):
    """Normalize all roles to one evidence registry instead of repeated reports."""
    unique, index = [], {}
    def key(entry):
        return dumps({k: v for k, v in entry.items() if k != "tool_call_id"})
    for entry in [*original, *(e for r in reports.values() for e in r.get("evidence", []))]:
        identity = key(entry)
        if identity not in index:
            index[identity] = len(unique)
            unique.append(deepcopy(entry))
    summaries = {}
    for task_id, report in reports.items():
        summaries[task_id] = {k: deepcopy(v) for k,v in report.items() if k != "evidence"}
        summaries[task_id]["original_evidence_indices"] = [index[key(e)] for e in report.get("evidence", [])]
    return {"reports": summaries, "original_evidence": unique}
