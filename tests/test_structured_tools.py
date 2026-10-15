"""Contract and boundary tests; no credentials, network or paid model calls."""
import unittest
from decimal import Decimal
from unittest.mock import Mock

from common.tools import calculator, codes, market, statistics


class StructuredToolsTests(unittest.TestCase):
    def test_codes_query_is_parameterized_and_namesakes_not_selected(self):
        db = Mock()
        db.fetch_all.return_value = [{"region_code": "11110530"}, {"region_code": "12270545"}]
        result = codes.find_regions(db, "사직동' OR 1=1 --")
        self.assertEqual(result["status"], "ambiguous")
        sql, params = db.fetch_all.call_args.args
        self.assertNotIn("OR 1=1", sql)
        self.assertIn("OR 1=1", params[1])

    def test_literal_wildcards_and_bounded_limit(self):
        db = Mock()
        db.fetch_all.return_value = []
        codes.find_regions(db, "a%b_c")
        self.assertEqual(db.fetch_all.call_args.args[1][1], r"%a\%b\_c%")
        with self.assertRaises(ValueError):
            codes.find_regions(db, "서울", limit=0)

    def test_industry_exact_match_prevents_cafe_study_cafe_confusion(self):
        db = Mock()
        db.fetch_all.return_value = [{"industry_code": "I21201", "industry_name": "카페"}]
        self.assertEqual(codes.find_industries(db, "카페")["status"], "found")
        self.assertEqual(db.fetch_all.call_count, 1)

    def test_market_latest_period_requires_all_requested_regions(self):
        db = Mock()
        db.fetch_one.return_value = {"crtr_ym": "202605"}
        db.fetch_all.return_value = [
            {"crtr_shape_cd": "11", "store_count": 5, "monthly_average_sales": None},
            {"crtr_shape_cd": "26", "store_count": 5, "monthly_average_sales": 8}]
        result = market.get_market_summary(db, "SI_CL", ["11", "26"], "I21201")
        self.assertEqual(db.fetch_one.call_args.args[1][-1], 2)
        self.assertIn("COUNT(DISTINCT crtr_shape_cd)", db.fetch_one.call_args.args[0])
        self.assertEqual(result["conditions"]["year_month"], "202605")
        self.assertIsNone(result["data"][0]["monthly_average_sales"])

    def test_market_missing_region_and_duplicates_not_success(self):
        db = Mock()
        db.fetch_all.return_value = [{"crtr_shape_cd": "11"}]
        result = market.get_market_summary(db, "SI_CL", ["11", "26"], "I21201", "202605")
        self.assertEqual(result["status"], "not_found")
        self.assertEqual(result["missing_regions"], ["26"])
        db.fetch_all.return_value = [{"crtr_shape_cd": "11"}, {"crtr_shape_cd": "11"}]
        self.assertEqual(market.get_market_summary(db, "SI_CL", ["11"], "I21201", "202605")["status"], "ambiguous")

    def test_bad_dates_and_codes_rejected_before_database(self):
        db = Mock()
        for month in ["202613", "2026-05", "２０２６０５"]:
            with self.assertRaises(ValueError):
                market.get_market_summary(db, "SI_CL", ["11"], "I21201", month)
        with self.assertRaises(ValueError):
            market.get_market_summary(db, "SI_CL", ["11; DROP"], "I21201")
        db.fetch_all.assert_not_called()

    def test_stats_common_year_and_aliases(self):
        db = Mock()
        db.fetch_one.return_value = {"prd_de": "2024"}
        db.fetch_all.side_effect = [
            [{"name": "서울특별시"}, {"name": "부산광역시"}],
            [{"region": r, "metric": m, "value": Decimal("1"), "unit": "개"}
             for r in ["서울특별시", "부산광역시"] for m in ["기업체 수", "종사자수"]]]
        result = statistics.get_statistics(db, "small_business", ["서울", "부산"])
        self.assertEqual(result["status"], "found")
        self.assertEqual(result["conditions"]["year"], "2024")
        self.assertEqual(db.fetch_one.call_args.args[1][-1], 4)

    def test_stats_null_and_nan_never_become_zero(self):
        for value in (None, Decimal("NaN"), Decimal("Infinity")):
            db = Mock()
            db.fetch_all.side_effect = [[{"name": "서울"}], [{"region": "서울", "metric": "전체", "value": value}]]
            result = statistics.get_statistics(db, "traditional_markets", ["서울"], "2024")
            self.assertEqual(result["status"], "not_found")
            self.assertIsNone(result["data"][0]["value"])

    def test_stats_unavailable_region_is_not_silently_ignored(self):
        db = Mock()
        db.fetch_all.return_value = [{"name": "서울"}]
        result = statistics.get_statistics(db, "traditional_markets", ["서울", "미지지역"])
        self.assertEqual(result["status"], "not_found")
        self.assertEqual(db.fetch_all.call_count, 1)

    def test_statistics_accepts_full_province_from_previous_tool(self):
        db = Mock()
        db.fetch_all.side_effect = [[{"name": "서울"}], [{"region": "서울", "metric": "전체", "value": 194}]]
        result = statistics.get_statistics(db, "traditional_markets", ["서울특별시"], "2024")
        self.assertEqual(result["status"], "found")
        self.assertEqual(result["conditions"]["regions"], ["서울"])

    def test_derived_region_code_is_flagged_for_review(self):
        db = Mock()
        db.fetch_all.return_value = [{"region_code": "12", "match_method": "derived"}]
        self.assertEqual(codes.find_regions(db, "광주", "SI_CL")["status"], "ambiguous")

    def test_stats_table_name_injection_rejected(self):
        with self.assertRaises(ValueError):
            statistics.get_statistics(Mock(), "stats.foo; drop table x", ["서울"])

    def test_calculation_decimal_tie_zero_and_nonfinite(self):
        self.assertEqual(calculator.calculate("difference", "0.3", "0.1")["data"]["result"], "0.2")
        self.assertEqual(calculator.calculate("compare", "5", "5")["data"]["result"], "equal")
        self.assertEqual(calculator.calculate("ratio", "12", "0")["status"], "error")
        self.assertEqual(calculator.calculate("ratio", "525105", "373750")["status"], "found")
        for bad in ["NaN", "Infinity", "1e1000", "__import__('os')"]:
            with self.assertRaises(ValueError):
                calculator.calculate("ratio", bad, "2")

    def test_tool_schemas_hide_database_and_names_unique(self):
        tools = [t for module in (codes, market, statistics, calculator) for t in module.create_tools(Mock())]
        self.assertEqual(len(tools), len({t.name for t in tools}))
        for t in tools:
            self.assertNotIn("db", t.args)


if __name__ == "__main__":
    unittest.main()
