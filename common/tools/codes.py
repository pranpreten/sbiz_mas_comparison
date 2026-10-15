"""Resolve names against the restored database; never manufacture administrative codes."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from common.data.database import Database


def _text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 100:
        raise ValueError(f"{name}: 1~100자 문자열이 필요합니다.")
    return value.strip()


def _pattern(value: str) -> str:
    return "%" + value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"


def _limit(limit: int) -> int:
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 30:
        raise ValueError("limit은 1~30 정수여야 합니다.")
    return limit


def find_regions(db: Database, name: str, region_type: str = "AD_CL",
                 province: str | None = None, district: str | None = None,
                 limit: int = 10) -> dict:
    """Search verified sbiz365 code mappings; retain mapping method and namesakes."""
    name, limit = _text(name, "name"), _limit(limit)
    levels = {"AD_CL": "emd", "SG_CL": "sigungu", "SI_CL": "sido"}
    if region_type not in levels:
        raise ValueError("region_type은 AD_CL, SG_CL, SI_CL 중 하나여야 합니다.")
    where = ["code_system = 'sbiz365_adm'", "sgis_adm_level = %s",
             "(ext_adm_nm ILIKE %s OR sgis_adm_nm ILIKE %s)", "ext_adm_cd IS NOT NULL"]
    params: list = [levels[region_type], _pattern(name), _pattern(name)]
    if province is not None:
        where.append("split_part(sgis_adm_nm, ' ', 1) ILIKE %s")
        params.append(_pattern(_text(province, "province")))
    if district is not None:
        where.append("sgis_adm_nm ILIKE %s")
        params.append(_pattern(_text(district, "district")))
    rows = db.fetch_all(f"""
        SELECT DISTINCT ext_adm_cd AS region_code, ext_adm_nm AS region_name,
               sgis_adm_nm AS full_name, sgis_adm_level AS region_level, match_method
        FROM stats.sgis_adm_code_map WHERE {' AND '.join(where)}
        ORDER BY region_code, full_name LIMIT %s
    """, tuple(params + [limit + 1]))
    for row in rows:
        row["region_type"] = region_type
    uncertain_mapping = any(row.get("match_method") == "derived" for row in rows)
    return {"status": "ambiguous" if len(rows) > 1 or uncertain_mapping else "found" if rows else "not_found",
            "data": rows[:limit], "truncated": len(rows) > limit,
            "source": "stats.sgis_adm_code_map (code_system=sbiz365_adm)",
            "conditions": {"name": name, "region_type": region_type,
                           "province": province, "district": district},
            "note": "상권용 ext_adm_cd입니다. SGIS 코드와 혼동하지 마세요. derived 매핑은 행정구역 변경 등이 있어 범위를 추가 확인해야 합니다. 수록월은 상권 조회에서 확인합니다."}


def find_industries(db: Database, name: str, limit: int = 10) -> dict:
    """Find market industries; exact names are preferred, otherwise show candidates."""
    name, limit = _text(name, "name"), _limit(limit)
    rows = db.fetch_all("""
        SELECT industry_code, industry_name, large_name, middle_name,
               source_crtr_ym, exclusion_text, review_status
        FROM market.industry_search
        WHERE is_current = true AND industry_name = %s
        ORDER BY industry_code LIMIT %s
    """, (name, limit + 1))
    if not rows:
        rows = db.fetch_all("""
            SELECT industry_code, industry_name, large_name, middle_name,
                   source_crtr_ym, exclusion_text, review_status
            FROM market.industry_search
            WHERE is_current = true AND (industry_name ILIKE %s OR search_text ILIKE %s)
            ORDER BY industry_code LIMIT %s
        """, (_pattern(name), _pattern(name), limit + 1))
    return {"status": "ambiguous" if len(rows) > 1 else "found" if rows else "not_found",
            "data": rows[:limit], "truncated": len(rows) > limit,
            "source": "market.industry_search", "conditions": {"name": name},
            "note": "업종 후보의 제외 범위를 확인하세요. KSIC와 상권 업종코드는 서로 다릅니다."}


def create_tools(db: Database) -> list:
    from langchain_core.tools import tool

    @tool
    def search_region_codes(name: str, region_type: str = "AD_CL",
                            province: str | None = None, district: str | None = None,
                            limit: int = 10) -> dict:
        """상권 지역코드 검색. name에는 청운효자동/서울 등 지역명만, province에는 서울 등 상위지역을 넣는다.
        행정동 AD_CL, 시군구 SG_CL, 시도 SI_CL. ambiguous면 후보를 구분하고 임의 선택하지 않는다.
        """
        return find_regions(db, name, region_type, province, district, limit)

    @tool
    def search_industry_codes(name: str, limit: int = 10) -> dict:
        """카페 등 업종명으로 상권 업종코드를 검색한다. 정확한 이름 우선, 후보와 제외 범위 확인."""
        return find_industries(db, name, limit)

    return [search_region_codes, search_industry_codes]
