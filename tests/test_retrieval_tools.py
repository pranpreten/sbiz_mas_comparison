"""Offline checks for evidence isolation, query binding and pagination."""
import unittest

from common.tools.policy import search_policies, get_policy_document
from common.tools.law import search_law_articles, get_law_article, get_law_addenda, get_law_appendices


class FakeDB:
    def __init__(self, all_rows=None, one=None):
        self.rows = all_rows or []
        self.one = one
        self.calls = []

    def fetch_all(self, query, params=()):
        self.calls.append((query, params))
        return self.rows

    def fetch_one(self, query, params=()):
        self.calls.append((query, params))
        return self.one


class RetrievalTests(unittest.TestCase):
    def test_policy_literal_keywords_parameterized_and_dedup_documents(self):
        db = FakeDB([{"document_id": "A:1"}, {"document_id": "A:2"}])
        result = search_policies(db, ["서울", "20%_지원'"], limit=1)
        sql, args = db.calls[0]
        self.assertNotIn("20%_지원'", sql)
        self.assertIn("%20\\%\\_지원'%", args)
        self.assertIn("EXISTS", sql)
        self.assertIn("d.expired = 'active'", sql)
        self.assertTrue(result["has_more"])
        self.assertEqual(len(result["data"]), 1)

    def test_policy_filtered_page_is_not_full_document(self):
        db = FakeDB([{"chunk_id": "1"}, {"chunk_id": "2"}], {"expired": "active"})
        result = get_policy_document(db, "B:1", ["자격"], offset=3, limit=1)
        self.assertFalse(result["evidence_complete"])
        self.assertEqual(result["next_offset"], 4)
        self.assertIn("PARTITION BY chunk_text", db.calls[1][0])
        self.assertEqual(db.calls[1][1][-2:], (2, 3))

    def test_missing_document_is_not_empty_found(self):
        db = FakeDB()
        result = get_policy_document(db, "missing")
        self.assertEqual(result["status"], "not_found")
        self.assertEqual(len(db.calls), 1)

    def test_law_selection_precedes_keyword_filter_and_searches_clauses(self):
        db = FakeDB()
        result = search_law_articles(db, ["갱신"], "2026-10-14", "임대차")
        sql, args = db.calls[0]
        self.assertIn("PARTITION BY law_id", sql)
        self.assertLess(sql.index("enforce_date <= %s"), sql.index("m.law_nm ILIKE"))
        self.assertIn("law.law_clause", sql)
        self.assertEqual(args[:2], ("2026-10-14", "2026-10-14"))
        self.assertEqual(result["status"], "not_found")

    def test_law_returns_clause_structure_and_marks_incomplete(self):
        db = FakeDB([{"clause_seq": 1, "content": "조건"}, {"clause_seq": 2, "content": "예외"}],
                    {"mst": "123", "law_nm": "법", "enforce_date": "2026-01-01", "article_content": "제1조"})
        result = get_law_article(db, "123", 1, "2026-10-14", clause_limit=1)
        self.assertFalse(result["article_complete"])
        self.assertEqual(result["next_clause_offset"], 1)
        self.assertEqual(result["data"]["clauses"][0]["content"], "조건")
        self.assertIn("lsiSeq=123", result["data"]["article"]["source_url"])

    def test_old_or_unknown_law_version_does_not_fetch_clauses(self):
        db = FakeDB()
        result = get_law_article(db, "123", 1, "2026-10-14")
        self.assertEqual(result["status"], "not_found")
        self.assertEqual(len(db.calls), 1)

    def test_addenda_are_bound_to_selected_version(self):
        db = FakeDB()
        get_law_addenda(db, "123", "2026-10-14")
        self.assertIn("JOIN law.law_addenda", db.calls[0][0])
        self.assertEqual(db.calls[0][1][:2], ("2026-10-14", "123"))

    def test_invalid_inputs_never_query(self):
        db = FakeDB()
        for args in [([],), ([None],), ([" "],), (["x"] * 9,)]:
            with self.assertRaises(ValueError):
                search_policies(db, *args)
        with self.assertRaises(ValueError):
            search_law_articles(db, ["임대"], "today")
        with self.assertRaises(ValueError):
            get_law_article(db, "123", 1, "2026-10-14", clause_offset=-1)
        with self.assertRaises(ValueError):
            get_policy_document(db, "A:1", limit=500)
        self.assertEqual(db.calls, [])

    def test_appendix_link_is_not_numeric_evidence(self):
        db = FakeDB([{"mst": "123", "appendix_content": None, "pdf_link": "https://www.law.go.kr/example"}])
        result = get_law_appendices(db, "123", "2026-10-14")
        self.assertFalse(result["data"][0]["text_available"])
        self.assertIn("law.law_appendix", db.calls[0][0])


if __name__ == "__main__":
    unittest.main()
