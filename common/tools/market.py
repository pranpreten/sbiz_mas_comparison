"""Bounded, parameterized market queries with explicit month and sales semantics."""
from __future__ import annotations

import re
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from common.data.database import Database


def _validate(region_type: str, region_codes: list[str], industry_code: str,
              year_month: str | None) -> list[str]:
    if region_type not in {"AD_CL", "SG_CL", "SI_CL"}:
        raise ValueError("region_type은 AD_CL, SG_CL, SI_CL 중 하나여야 합니다.")
    if not isinstance(region_codes, list) or not 1 <= len(region_codes) <= 20:
        raise ValueError("region_codes는 1~20개 코드 목록이어야 합니다.")
    if any(not isinstance(code, str) or not re.fullmatch(r"[0-9]{2,10}", code) for code in region_codes):
        raise ValueError("조회로 확인한 숫자 지역코드가 필요합니다.")
    if not isinstance(industry_code, str) or not re.fullmatch(r"[A-Z][0-9]{5}", industry_code):
        raise ValueError("조회로 확인한 상권 소분류 업종코드가 필요합니다.")
    if year_month is not None:
        if not isinstance(year_month, str) or not re.fullmatch(r"[0-9]{6}", year_month):
            raise ValueError("year_month는 YYYYMM 형식이어야 합니다.")
        datetime.strptime(year_month, "%Y%m")
    return list(dict.fromkeys(region_codes))


def get_market_summary(db: Database, region_type: str, region_codes: list[str],
                       industry_code: str, year_month: str | None = None) -> dict:
    """Read all requested regions at one month; omitted month means latest common month."""
    codes = _validate(region_type, region_codes, industry_code, year_month)
    requested_month = year_month
    if year_month is None:
        latest = db.fetch_one("""
            SELECT crtr_ym FROM market.simple_analysis
            WHERE rgn_cd = %s AND crtr_shape_cd = ANY(%s) AND crtr_upjong_cd = %s
              AND crtr_ym ~ '^[0-9]{6}$'
            GROUP BY crtr_ym HAVING COUNT(DISTINCT crtr_shape_cd) = %s
            ORDER BY crtr_ym DESC LIMIT 1
        """, (region_type, codes, industry_code, len(codes)))
        year_month = latest["crtr_ym"] if latest else None
    conditions = {"region_type": region_type, "region_codes": codes,
                  "industry_code": industry_code, "year_month": year_month,
                  "period_selection": "explicit" if requested_month else "latest_common_month"}
    base = {"source": "market.simple_analysis", "conditions": conditions,
            "units": {"store_count": "개", "monthly_average_sales": "만원"},
            "note": "월평균 매출이며 총매출이 아닙니다. NULL은 결측이며 0이 아닙니다. 최신은 저장된 데이터 기준입니다."}
    if year_month is None:
        return {**base, "status": "not_found", "data": [], "missing_regions": codes}
    rows = db.fetch_all("""
        SELECT crtr_ym, rgn_cd, crtr_shape_cd, crtr_upjong_cd,
               sido_nm, sido_nm_std, sgg_nm, admi_nm, tpbiz_clscd_nm, is_substituted,
               CASE rgn_cd WHEN 'AD_CL' THEN amt_dong WHEN 'SG_CL' THEN amt_sgg
                   WHEN 'SI_CL' THEN amt_sido END AS monthly_average_sales,
               CASE rgn_cd WHEN 'AD_CL' THEN upso_count_dong WHEN 'SG_CL' THEN upso_count_sgg
                   WHEN 'SI_CL' THEN upso_count_sido END AS store_count
        FROM market.simple_analysis
        WHERE rgn_cd = %s AND crtr_shape_cd = ANY(%s) AND crtr_upjong_cd = %s AND crtr_ym = %s
        ORDER BY crtr_shape_cd LIMIT %s
    """, (region_type, codes, industry_code, year_month, len(codes) * 2 + 1))
    seen = [row["crtr_shape_cd"] for row in rows]
    duplicate = len(seen) != len(set(seen))
    missing = [code for code in codes if code not in seen]
    return {**base, "status": "ambiguous" if duplicate else "not_found" if missing else "found",
            "data": rows, "missing_regions": missing,
            "duplicate_rows": duplicate,
            "substituted_rows": any(row.get("is_substituted") is True for row in rows)}


def create_tools(db: Database) -> list:
    from langchain_core.tools import tool

    @tool
    def lookup_market_summary(region_type: str, region_codes: list[str], industry_code: str,
                              year_month: str | None = None) -> dict:
        """검증된 코드로 업소수·월평균 매출(만원)을 조회. 여러 지역은 같은 기준월로 비교.
        year_month=YYYYMM; 생략하면 요청 지역 모두 수록된 최신 공통월. 코드는 검색 도구로 먼저 확인.
        """
        return get_market_summary(db, region_type, region_codes, industry_code, year_month)

    return [lookup_market_summary]
