"""Date-bound retrieval from the collected law snapshot, including subclauses.

The snapshot is not a complete historical archive. Latest stored version on/before
as_of is selected; source dates are returned, never labeled live current law.
"""
from __future__ import annotations

from datetime import date
from typing import Any

from .policy import _limit, _pattern, _terms

SOURCE = "law.law_master + law.law_article + law.law_clause"
CAUTION = "수집된 법령 중 기준일 이전 최신 버전입니다. 완전한 연혁·미수집 개정 및 경과조치까지 보장하지 않습니다. 연결 조문·부칙이 필요하면 추가 조회하세요."
VERSIONS = """WITH versions AS (
    SELECT m.*, row_number() OVER (PARTITION BY law_id
        ORDER BY enforce_date DESC, promul_date DESC NULLS LAST, mst DESC) AS version_rank
    FROM law.law_master m WHERE enforce_date <= %s
), selected AS (SELECT * FROM versions WHERE version_rank = 1)
"""


def _as_of(value: str) -> str:
    if not isinstance(value, str) or len(value) != 10:
        raise ValueError("as_of must be an explicit ISO date YYYY-MM-DD")
    return date.fromisoformat(value).isoformat()


def _source_url(row: dict) -> str:
    # Stable version-specific official viewer; no source_file filesystem leakage.
    return "https://www.law.go.kr/LSW/lsInfoP.do?lsiSeq=" + str(row["mst"])


def _result(data: Any, conditions: dict, **extra: Any) -> dict:
    return {"status": "found" if data else "not_found", "data": data,
            "source": SOURCE, "conditions": conditions, "caution": CAUTION, **extra}


def search_law_articles(db: Any, keywords: list[str], as_of: str,
                        law_name: str | None = None, limit: int = 8) -> dict:
    """Search ALL literal terms across article header/content and nested clauses.

    law_name narrows law titles; results retain mst + article_seq for exact detail
    retrieval. An empty result is not evidence that no legal restriction exists.
    """
    day, terms = _as_of(as_of), _terms(keywords)
    _limit(limit)
    filters = ["NOT COALESCE(a.is_heading, false)",
               "(a.article_enforce_date IS NULL OR a.article_enforce_date <= %s)"]
    params: list = [day, day]
    if law_name is not None:
        _terms([law_name])
        filters.append("m.law_nm ILIKE %s")
        params.append(_pattern(law_name.strip()))
    for term in terms:
        filters.append("""(concat_ws(' ', a.article_title, a.article_content) ILIKE %s
            OR EXISTS (SELECT 1 FROM law.law_clause c
                WHERE c.mst = a.mst AND c.article_seq = a.article_seq AND c.content ILIKE %s))""")
        params.extend([_pattern(term)] * 2)
    sql = VERSIONS + """SELECT m.mst, m.law_id, m.law_nm, m.law_type,
        m.enforce_date, m.promul_date, m.loaded_at, a.article_seq, a.article_label,
        a.article_title, a.article_enforce_date, left(a.article_content, 1200) AS preview,
        length(a.article_content) > 1200 AS preview_truncated
        FROM selected m JOIN law.law_article a ON a.mst = m.mst WHERE """ + " AND ".join(filters) + """
        ORDER BY m.law_nm, a.article_seq LIMIT %s"""
    rows = db.fetch_all(sql, tuple(params + [limit + 1]))
    for row in rows[:limit]:
        row["source_url"] = _source_url(row)
    return _result(rows[:limit], {"keywords": terms, "as_of": day, "law_name": law_name, "limit": limit},
                   has_more=len(rows) > limit, detail_required=True)


def get_law_article(db: Any, mst: str, article_seq: int, as_of: str,
                    clause_offset: int = 0, clause_limit: int = 100) -> dict:
    """Get exact article plus ordered full paragraph/item/subitem text.

    Only the selected stored version for as_of is accepted. The article header is
    insufficient by itself: clauses, references, and relevant addenda matter.
    """
    day = _as_of(as_of)
    if not isinstance(mst, str) or not mst.isdigit() or len(mst) > 30:
        raise ValueError("mst must be the numeric version ID returned by search")
    if isinstance(article_seq, bool) or not isinstance(article_seq, int) or article_seq < 1:
        raise ValueError("article_seq must be a positive retrieved sequence")
    _limit(clause_limit, 200)
    if isinstance(clause_offset, bool) or not isinstance(clause_offset, int) or not 0 <= clause_offset <= 10000:
        raise ValueError("clause_offset must be between 0 and 10000")
    conditions = {"mst": mst, "article_seq": article_seq, "as_of": day,
                  "clause_offset": clause_offset, "clause_limit": clause_limit}
    article = db.fetch_one(VERSIONS + """SELECT m.mst, m.law_id, m.law_nm,
        m.law_type, m.enforce_date, m.promul_date, m.promul_no, m.loaded_at,
        m.article_enforce_note, a.article_seq, a.article_key, a.article_label,
        a.article_title, a.article_content, a.article_reference,
        a.article_enforce_date, a.article_rev_note
        FROM selected m JOIN law.law_article a ON a.mst = m.mst
        WHERE m.mst = %s AND a.article_seq = %s AND NOT COALESCE(a.is_heading, false)
          AND (a.article_enforce_date IS NULL OR a.article_enforce_date <= %s)""",
        (day, mst, article_seq, day))
    if article is None:
        return _result(None, conditions, reason="해당 기준일의 선택된 수집 버전에 조문이 없습니다. 검색으로 버전·조문 번호를 다시 확인하세요.")
    article["source_url"] = _source_url(article)
    clauses = db.fetch_all("""SELECT clause_seq, depth_level, depth_nm, para_no,
        item_no, item_branch_no, subitem_no, content, para_rev_type, para_rev_note
        FROM law.law_clause WHERE mst = %s AND article_seq = %s
        ORDER BY clause_seq LIMIT %s OFFSET %s""", (mst, article_seq, clause_limit + 1, clause_offset))
    more = len(clauses) > clause_limit
    return _result({"article": article, "clauses": clauses[:clause_limit]}, conditions,
                   has_more=more, next_clause_offset=clause_offset + clause_limit if more else None,
                   article_complete=clause_offset == 0 and not more)


def get_law_addenda(db: Any, mst: str, as_of: str, offset: int = 0, limit: int = 5) -> dict:
    """Retrieve full stored addenda for transition/application-date checks."""
    day = _as_of(as_of)
    if not isinstance(mst, str) or not mst.isdigit() or len(mst) > 30:
        raise ValueError("mst must be numeric")
    _limit(limit)
    if isinstance(offset, bool) or not isinstance(offset, int) or not 0 <= offset <= 10000:
        raise ValueError("offset must be between 0 and 10000")
    rows = db.fetch_all(VERSIONS + """SELECT m.mst, m.law_id, m.law_nm, m.enforce_date,
        a.addenda_key, a.addenda_seq, a.addenda_promul_date, a.addenda_promul_no,
        a.addenda_content FROM selected m JOIN law.law_addenda a ON a.mst = m.mst
        WHERE m.mst = %s ORDER BY a.addenda_seq DESC LIMIT %s OFFSET %s""",
        (day, mst, limit + 1, offset))
    for row in rows[:limit]:
        row["source_url"] = _source_url(row)
    more = len(rows) > limit
    return _result(rows[:limit], {"mst": mst, "as_of": day, "offset": offset, "limit": limit},
                   has_more=more, next_offset=offset + limit if more else None)


def get_law_appendices(db: Any, mst: str, as_of: str, offset: int = 0, limit: int = 5) -> dict:
    """Retrieve industry sales-threshold tables and other annexes, if collected.

    Links alone are not table evidence. Missing appendix_content is explicitly
    marked and must not be replaced with remembered numeric thresholds.
    """
    day = _as_of(as_of)
    if not isinstance(mst, str) or not mst.isdigit() or len(mst) > 30:
        raise ValueError("mst must be numeric")
    _limit(limit)
    if isinstance(offset, bool) or not isinstance(offset, int) or not 0 <= offset <= 10000:
        raise ValueError("offset must be between 0 and 10000")
    rows = db.fetch_all(VERSIONS + """SELECT m.mst, m.law_id, m.law_nm, m.enforce_date,
        a.appendix_key, a.appendix_seq, a.appendix_no, a.appendix_branch_no,
        a.appendix_type, a.appendix_title, a.appendix_content, a.file_link, a.pdf_link,
        a.image_links FROM selected m JOIN law.law_appendix a ON a.mst = m.mst
        WHERE m.mst = %s ORDER BY a.appendix_seq LIMIT %s OFFSET %s""",
        (day, mst, limit + 1, offset))
    for row in rows[:limit]:
        row["source_url"] = _source_url(row)
        row["text_available"] = bool((row.get("appendix_content") or "").strip())
    more = len(rows) > limit
    return _result(rows[:limit], {"mst": mst, "as_of": day, "offset": offset, "limit": limit},
                   has_more=more, next_offset=offset + limit if more else None,
                   note="text_available=false이면 파일 링크만 수집된 상태입니다. 숫자 기준을 추정하지 말고 원문 확인 필요로 처리하세요.")


def create_tools(db: Any) -> list:
    from langchain_core.tools import tool

    @tool("search_law_articles")
    def search(keywords: list[str], as_of: str, law_name: str | None = None, limit: int = 8) -> dict:
        """법령 조문과 항·호를 키워드 모두 포함 조건으로 검색. as_of는 실험 기준일 YYYY-MM-DD. mst/article_seq로 상세 조회 필수."""
        return search_law_articles(db, keywords, as_of, law_name, limit)

    @tool("get_law_article")
    def detail(mst: str, article_seq: int, as_of: str,
               clause_offset: int = 0, clause_limit: int = 100) -> dict:
        """검색한 법령 버전·조문의 본문과 항·호·목 전체를 순서대로 조회. has_more면 next_clause_offset으로 이어서 조회."""
        return get_law_article(db, mst, article_seq, as_of, clause_offset, clause_limit)

    @tool("get_law_addenda")
    def addenda(mst: str, as_of: str, offset: int = 0, limit: int = 5) -> dict:
        """법령의 적용시점·경과조치 확인용 부칙 원문 조회. has_more면 next_offset으로 이어서 조회."""
        return get_law_addenda(db, mst, as_of, offset, limit)

    @tool("get_law_appendices")
    def appendices(mst: str, as_of: str, offset: int = 0, limit: int = 5) -> dict:
        """법령 별표(업종별 매출기준 등) 조회. text_available=false면 원문 확인 필요이며 값을 추정하지 않는다."""
        return get_law_appendices(db, mst, as_of, offset, limit)

    return [search, detail, addenda, appendices]
