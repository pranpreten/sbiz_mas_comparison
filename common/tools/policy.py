"""Read-only policy discovery and paginated evidence retrieval.

Keyword matches are candidates, never proof of eligibility or open applications.
No embeddings, hidden model calls, or reference-answer data are used.
"""
from __future__ import annotations

from typing import Any

SOURCE = "announcement.policy_document + announcement.policy_chunk"


def _terms(keywords: list[str]) -> list[str]:
    if not isinstance(keywords, list) or not 1 <= len(keywords) <= 8:
        raise ValueError("keywords must contain 1 to 8 literal search terms")
    if any(not isinstance(t, str) or not t.strip() or len(t.strip()) > 100 for t in keywords):
        raise ValueError("Each keyword must be nonblank and at most 100 characters")
    return list(dict.fromkeys(t.strip() for t in keywords))


def _pattern(term: str) -> str:
    return "%" + term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"


def _limit(value: int, maximum: int = 20) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise ValueError(f"limit must be between 1 and {maximum}")
    return value


def _result(rows: Any, conditions: dict, **extra: Any) -> dict:
    return {"status": "found" if rows else "not_found", "data": rows,
            "source": SOURCE, "conditions": conditions, **extra}


def search_policies(db: Any, keywords: list[str], limit: int = 5,
                    include_inactive: bool = False) -> dict:
    """Find documents matching ALL keywords across title, metadata or chunks.

    Use short keywords, not a full question. Search results require detail review;
    active is a stored ingestion status and does not guarantee current availability.
    """
    terms = _terms(keywords)
    _limit(limit)
    clauses, params = [], []
    for term in terms:
        clauses.append("""(d.title ILIKE %s OR d.doc_meta::text ILIKE %s OR
            EXISTS (SELECT 1 FROM announcement.policy_chunk c
                    WHERE c.document_id = d.document_id
                      AND c.expired = d.expired AND c.chunk_text ILIKE %s))""")
        params.extend([_pattern(term)] * 3)
    if not include_inactive:
        clauses.append("d.expired = 'active'")
    sql = """SELECT d.document_id, d.title, d.source_url, d.expired,
           d.version, d.government_level, d.doc_meta, d.updated_at, d.ingested_at
           FROM announcement.policy_document d WHERE """ + " AND ".join(clauses) + """
           ORDER BY d.updated_at DESC, d.document_id LIMIT %s"""
    rows = db.fetch_all(sql, tuple(params + [limit + 1]))
    return _result(rows[:limit], {"keywords": terms, "match": "all",
                   "include_inactive": include_inactive, "limit": limit},
                   has_more=len(rows) > limit,
                   caution="검색 후보입니다. 대상·지역·제외조건·접수기간을 상세 근거로 확인하세요. active는 현재 신청 가능을 보장하지 않습니다.")


def get_policy_document(db: Any, document_id: str, keywords: list[str] | None = None,
                        offset: int = 0, limit: int = 10) -> dict:
    """Return metadata and distinct full chunks, paginated without silent clipping.

    With keywords, return chunks containing ANY term. Without keywords, walk the
    entire document in order. Use next_offset until has_more is false.
    """
    if not isinstance(document_id, str) or not document_id.strip() or len(document_id) > 120:
        raise ValueError("document_id must be a retrieved document ID")
    _limit(limit, 30)
    if isinstance(offset, bool) or not isinstance(offset, int) or not 0 <= offset <= 10000:
        raise ValueError("offset must be between 0 and 10000")
    terms = _terms(keywords) if keywords is not None else []
    conditions = {"document_id": document_id, "keywords": terms, "offset": offset, "limit": limit}
    doc = db.fetch_one("""SELECT document_id, title, source_url, expired, version,
        government_level, doc_meta, updated_at, ingested_at,
        raw_json->>'bizOutlCn' AS business_outline,
        raw_json->>'useUsgVl' AS loan_purpose,
        raw_json->>'gdsOperPdCn' AS operation_period
        FROM announcement.policy_document WHERE document_id = %s""", (document_id,))
    if doc is None:
        return _result(None, conditions)
    filters, params = ["document_id = %s", "expired = %s"], [document_id, doc["expired"]]
    if terms:
        filters.append("(" + " OR ".join("chunk_text ILIKE %s" for _ in terms) + ")")
        params.extend(_pattern(t) for t in terms)
    # Duplicate ingestion chunks with identical text must not consume the page.
    sql = """WITH distinct_chunks AS (
        SELECT chunk_id, chunk_index, chunk_text, chunk_meta,
               row_number() OVER (PARTITION BY chunk_text ORDER BY chunk_index, chunk_id) AS duplicate_rank
        FROM announcement.policy_chunk WHERE """ + " AND ".join(filters) + """
        ) SELECT chunk_id, chunk_index, chunk_text, chunk_meta FROM distinct_chunks
        WHERE duplicate_rank = 1 ORDER BY chunk_index, chunk_id LIMIT %s OFFSET %s"""
    rows = db.fetch_all(sql, tuple(params + [limit + 1, offset]))
    more = len(rows) > limit
    return _result({"document": doc, "chunks": rows[:limit]}, conditions,
                   has_more=more, next_offset=offset + limit if more else None,
                   evidence_complete=not terms and offset == 0 and not more,
                   caution="청크 필터/페이지는 문서 전체가 아닐 수 있습니다. 첨부자료 수집 누락 여부와 현재 접수 상태는 별도 확인이 필요합니다.")


def create_tools(db: Any) -> list:
    """Bind the shared database without exposing it as a model argument."""
    from langchain_core.tools import tool

    @tool("search_policies")
    def search(keywords: list[str], limit: int = 5, include_inactive: bool = False) -> dict:
        """정책·지원사업을 짧은 키워드 모두 포함 조건으로 검색. 결과는 후보이며 신청자격 판정이 아님."""
        return search_policies(db, keywords, limit, include_inactive)

    @tool("get_policy_document")
    def detail(document_id: str, keywords: list[str] = [],
               offset: int = 0, limit: int = 4) -> dict:
        """정책 ID의 메타정보와 원문 청크 조회. keywords는 문자열 배열의 OR 필터이며 [] 또는 생략 시 전체 순차 조회. has_more면 next_offset으로 계속 조회."""
        # 모델에는 nullable union 대신 단일 배열 스키마를 제공한다. 기본 배열은 수정하지 않는다.
        return get_policy_document(db, document_id, keywords or None, offset, limit)

    return [search, detail]
