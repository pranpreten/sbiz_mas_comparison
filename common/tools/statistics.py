"""Typed statistics tools for the two surveys needed by the 36-question pilot.

The registry is a schema allowlist, not an answer lookup. Values, dimensions and
available years are always obtained from the database.
"""
from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import TYPE_CHECKING

from .codes import _text

if TYPE_CHECKING:
    from common.data.database import Database

DATASETS = {
    "small_business": {
        "table": "stats.kosis_small_business_survey_01",
        "title": "소상공인실태조사: 지역·산업별 기업체 수 및 종사자 수",
        "region_column": "sido_nm", "metrics": ["기업체 수", "종사자수"],
    },
    "traditional_markets": {
        "table": "stats.kosis_traditional_market_survey_01",
        "title": "전통시장·상점가 실태조사: 지역별 전통시장 수",
        "region_column": "region_nm", "metrics": ["전체"],
    },
}


def _dataset(indicator: str) -> dict:
    if indicator not in DATASETS:
        raise ValueError("indicator는 small_business 또는 traditional_markets여야 합니다.")
    return DATASETS[indicator]


def list_statistics(db: Database, indicator: str) -> dict:
    """Discover region/industry names and years in a fixed, documented dataset."""
    spec = _dataset(indicator)
    table, region = spec["table"], spec["region_column"]
    regions = db.fetch_all(f"SELECT DISTINCT {region} AS name FROM {table} WHERE {region} IS NOT NULL ORDER BY name LIMIT 100")
    years = db.fetch_all(f"SELECT DISTINCT prd_de AS year FROM {table} WHERE prd_de ~ '^[0-9]{{4}}$' ORDER BY year DESC LIMIT 100")
    industries = db.fetch_all(f"SELECT DISTINCT induty_nm AS name FROM {table} ORDER BY name LIMIT 100") if indicator == "small_business" else []
    return {"status": "found" if regions else "not_found", "source": table,
            "conditions": {"indicator": indicator},
            "data": {"title": spec["title"], "regions": [r["name"] for r in regions],
                     "years": [r["year"] for r in years], "metrics": spec["metrics"],
                     "industries": [r["name"] for r in industries]},
            "note": "이 도구가 지원하는 통계는 두 조사입니다. 다른 지표를 이 통계로 대체하지 마세요."}


def get_statistics(db: Database, indicator: str, regions: list[str],
                   year: str | None = None, industry: str = "전산업") -> dict:
    """Read named regions at one common year; partial data is explicitly marked."""
    spec = _dataset(indicator)
    if not isinstance(regions, list) or not 1 <= len(regions) <= 20:
        raise ValueError("regions는 1~20개 지역명 목록이어야 합니다.")
    names = list(dict.fromkeys(_text(r, "region") for r in regions))
    industry = _text(industry, "industry")
    if year is not None and (not isinstance(year, str) or not re.fullmatch(r"[0-9]{4}", year)):
        raise ValueError("year는 YYYY 형식이어야 합니다.")
    if indicator == "traditional_markets" and industry != "전산업":
        raise ValueError("전통시장 수에는 업종 필터를 적용할 수 없습니다.")
    table, region_column = spec["table"], spec["region_column"]
    candidates = db.fetch_all(f"SELECT DISTINCT {region_column} AS name FROM {table} WHERE {region_column} IS NOT NULL ORDER BY name LIMIT 100")
    available = [r["name"] for r in candidates]
    resolved, unresolved = [], {}
    for name in names:
        # Survey region labels differ (서울특별시 vs 서울). Only strip known
        # administrative suffixes, not arbitrary words from a user's question.
        suffixes = {"특별시", "광역시", "특별자치시", "특별자치도", "도"}
        matches = [name] if name in available else [
            r for r in available if r.startswith(name)
            or (name.startswith(r) and name[len(r):] in suffixes)]
        if len(matches) != 1:
            unresolved[name] = matches
        else:
            resolved.append(matches[0])
    conditions = {"indicator": indicator, "requested_regions": names, "regions": resolved,
                  "year": year, "industry": industry if indicator == "small_business" else None}
    if unresolved:
        return {"status": "ambiguous" if any(unresolved.values()) else "not_found", "data": [],
                "source": table, "conditions": conditions, "region_candidates": unresolved}
    resolved = list(dict.fromkeys(resolved))
    conditions["regions"] = resolved
    metrics = spec["metrics"]
    where = f"{region_column} = ANY(%s) AND category_nm = ANY(%s)"
    params: list = [resolved, metrics]
    if indicator == "small_business":
        where += " AND induty_nm = %s"
        params.append(industry)
    requested_year = year
    if year is None:
        latest = db.fetch_one(f"""
            SELECT prd_de FROM {table}
            WHERE {where} AND prd_de ~ '^[0-9]{{4}}$' AND value IS NOT NULL
              AND value::text NOT IN ('NaN', 'Infinity', '-Infinity')
            GROUP BY prd_de HAVING COUNT(DISTINCT ({region_column}, category_nm)) = %s
            ORDER BY prd_de DESC LIMIT 1
        """, tuple(params + [len(resolved) * len(metrics)]))
        year = latest["prd_de"] if latest else None
    conditions.update(year=year, period_selection="explicit" if requested_year else "latest_common_year")
    if year is None:
        return {"status": "not_found", "data": [], "source": table, "conditions": conditions,
                "note": "요청한 모든 지역·지표의 값이 있는 공통연도가 없습니다."}
    industry_select = "induty_nm AS industry," if indicator == "small_business" else ""
    rows = db.fetch_all(f"""
        SELECT id, {region_column} AS region, {industry_select}
               category_nm AS metric, unit_nm AS unit, prd_de AS year, value
        FROM {table} WHERE {where} AND prd_de = %s
        ORDER BY {region_column}, category_nm, id LIMIT %s
    """, tuple(params + [year, len(resolved) * len(metrics) * 2 + 1]))
    for row in rows:
        if row["value"] is not None:
            try:
                finite = Decimal(str(row["value"])).is_finite()
            except InvalidOperation:
                finite = False
            if not finite:
                row["value"] = None
                row["missing_reason"] = "non_finite_value"
    keys = [(r["region"], r["metric"]) for r in rows]
    duplicates = len(keys) != len(set(keys))
    complete = set(keys) == {(r, m) for r in resolved for m in metrics} and all(r["value"] is not None for r in rows)
    return {"status": "ambiguous" if duplicates else "found" if complete else "not_found",
            "data": rows, "source": table, "conditions": conditions,
            "complete": complete, "duplicate_rows": duplicates,
            "note": "단위는 각 행의 unit을 사용합니다. 결측을 0으로 바꾸지 말고 같은 연도·산업 범위끼리 비교하세요. 최신은 저장된 자료 기준입니다."}


def create_tools(db: Database) -> list:
    from langchain_core.tools import tool

    @tool
    def inspect_statistics(indicator: str) -> dict:
        """통계 지역명·업종·수록연도 확인. small_business=소상공인 업체/종사자, traditional_markets=전통시장 수."""
        return list_statistics(db, indicator)

    @tool
    def lookup_statistics(indicator: str, regions: list[str], year: str | None = None,
                          industry: str = "전산업") -> dict:
        """지역별 통계 수치 조회. small_business=기업체 수와 종사자 수, traditional_markets=전통시장 수.
        regions는 서울/부산 등 지역명. year 생략 시 모든 요청 지역·지표에 값이 있는 최신 공통연도.
        산업 범위 기본 전산업. 연도 비교는 각 연도를 지정해 호출. 평균 종사자 수는 종사자/기업체로 계산.
        """
        return get_statistics(db, indicator, regions, year, industry)

    return [inspect_statistics, lookup_statistics]
