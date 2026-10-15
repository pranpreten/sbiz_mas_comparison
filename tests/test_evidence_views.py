import json
from copy import deepcopy
import unittest
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from common.orchestration.evidence import compact_view, report_payload, dumps, candidate_view
from common.orchestration import Context
from common.config import Settings
from test_orchestration import ScriptedFactory, ai, report


class EvidenceViewsTests(unittest.TestCase):
    def test_policy_detail_pages_keep_conditions_after_repeated_headers(self):
        chunks = [{"chunk_id": str(i), "chunk_text": "Repeated header " * 40 + " eligibility condition " + str(i)} for i in range(3)]
        raw = {"source": "fixture.policy", "data": {"document": {"document_id": "test"}, "chunks": chunks}}
        view = compact_view(candidate_view(raw), 5000, "Repeated header", minimal=True)
        self.assertEqual(view["data"]["policy_chunks"], chunks)
        self.assertEqual(raw["data"]["chunks"], chunks)

    def test_legal_items_are_never_top_k_filtered_or_text_truncated(self):
        clauses = [{"clause_seq": i, "content": "condition " + str(i) + " x" * 100,
                    "para_no": "1", "item_no": str(i), "nullable": None} for i in range(15)]
        raw = {"source": "fixture.law", "data": {"article": {"mst": "123", "article_content": "Definition"},
                                                   "clauses": clauses}}
        view = compact_view(candidate_view(raw), 7000, "first condition", minimal=True)
        self.assertEqual([r["content"] for r in view["data"]["legal_clauses"]], [r["content"] for r in clauses])
        self.assertEqual([r["clause_seq"] for r in view["data"]["legal_clauses"]], list(range(15)))

    def test_final_view_discards_search_navigation_but_preserves_actual_articles(self):
        search = {"source": "fixture.law", "detail_required": True, "data": [
            {"mst": "123", "article_seq": 1, "article_title": "Definition", "preview": "not evidence"}]}
        self.assertNotIn("data", candidate_view(search, final=True))
        detail = {"source": "fixture.law", "data": {"article": {"mst": "123", "article_seq": 1},
                   "clauses": [{"clause_seq": 1, "content": "actual condition"}]}}
        self.assertEqual(candidate_view(detail, final=True)["data"]["legal_clauses"][0]["content"], "actual condition")

    def test_candidate_navigation_keeps_all_ids_without_duplicate_preview_metadata(self):
        value = {"source": "fixture", "detail_required": True, "data": [
            {"mst": "123", "article_seq": i, "article_title": "Definition " + str(i),
             "preview": "preview " * 1000, "loaded_at": "internal timestamp"} for i in range(12)]}
        before = deepcopy(value)
        view = candidate_view(value)
        self.assertEqual(len(view["data"]), 12)
        self.assertNotIn("preview", view["data"][0])
        self.assertEqual(value, before)
        self.assertIn("_candidate_view", view)
        detail = {**value, "detail_required": False}
        self.assertEqual(candidate_view(detail), detail)

    def test_legal_candidate_titles_and_ids_survive_compaction(self):
        value = {"source": "fixture.law", "data": [
            {"mst": "123", "article_seq": i, "article_title": "Definition " + str(i),
             "preview": "long preview " * 1000} for i in range(12)]}
        view = compact_view(value, 9000, "eligibility")
        self.assertEqual([r["article_seq"] for r in view["data"]], list(range(12)))
        self.assertEqual(view["data"][-1]["article_title"], "Definition 11")
        self.assertLessEqual(len(dumps(view)), 9000)

    def test_protected_predecessor_floor_is_reserved_before_other_results(self):
        predecessor = {"question": "criteria", "conditions": {"scope": "p" * 9000},
                       "data": "long source text " * 3000}
        history = [HumanMessage(content=dumps(predecessor)), *[
            HumanMessage(content=dumps({"source": "fixture", "data": "passage " * 500})) for _ in range(8)]]
        factory = ScriptedFactory([ai("continued")])
        ctx = Context(Settings(), tools=[], model_factory=factory)
        ctx._invoke("worker:law", "read reference", history)
        sent = factory.calls[0]["messages"][1:]
        self.assertEqual(json.loads(sent[0].content)["conditions"], predecessor["conditions"])
        self.assertLessEqual(sum(len(m.content) for m in sent), 28000)

    def test_short_results_do_not_starve_large_predecessor_context(self):
        predecessor = {"question": "legal criteria", "original_evidence": [
            {"source": "fixture", "conditions": {"document_id": str(i), "scope": "x" * 600},
             "data": "evidence passage " * 2000} for i in range(10)]}
        history = [HumanMessage(content=dumps(predecessor)), *[
            HumanMessage(content=dumps({"status": "not_found", "data": []})) for _ in range(8)]]
        factory = ScriptedFactory([ai("continued")])
        ctx = Context(Settings(), tools=[], model_factory=factory)
        ctx._invoke("worker:law", "continue dependent research", history)
        body = json.loads(factory.calls[0]["messages"][1].content)
        self.assertEqual(len(body["original_evidence"]), 10)
        self.assertEqual(body["original_evidence"][9]["conditions"]["document_id"], "9")
        self.assertEqual(ctx.metrics()["model_calls"], 1)

    def test_compaction_keeps_tool_protocol_and_predecessor_evidence(self):
        raw = {"source": "policy.notice", "data": {"document_id": "p1",
               "text": "지원대상 및 제외조건 확인. " * 6000}}
        history = [HumanMessage(content=dumps({"question": "지원대상 법적 정의"})),
                   AIMessage(content="", tool_calls=[{"name": "lookup", "args": {},
                       "id": "policy-1", "type": "tool_call"}]),
                   ToolMessage(content=dumps(raw), tool_call_id="policy-1")]
        before = deepcopy(history)
        factory = ScriptedFactory([ai("발췌 확인")])
        ctx = Context(Settings(), tools=[], model_factory=factory)
        ctx._invoke("worker:law", "정책 대상의 법령 조회", history)
        sent = factory.calls[0]["messages"]
        sent_ai = next(m for m in sent if m.type == "ai")
        sent_tool = next(m for m in sent if m.type == "tool")
        self.assertEqual(sent_ai.tool_calls[0]["id"], sent_tool.tool_call_id)
        self.assertEqual(json.loads(sent_tool.content)["data"]["document_id"], "p1")
        self.assertTrue(json.loads(sent_tool.content)["_context_selection"]["partial_view"])
        self.assertEqual(history, before)

    def test_small_numeric_record_and_provenance_are_unchanged(self):
        data = {"source": "stats.table", "conditions": {"year": 2024},
                "data": [{"region_code": "11", "value": 194, "unit": "개"}]}
        self.assertEqual(compact_view(data, 2000, "서울 시장"), data)

    def test_long_text_selection_is_verbatim_explicit_and_does_not_mutate(self):
        text = "첫 문단\n" * 900 + "지원대상은 소상공인. 제외조건도 확인해야 합니다.\n" + "다른 문단\n" * 900
        data = {"source": "policy", "conditions": {"document_id": "doc1"},
                "data": {"document_id": "doc1", "chunk_text": text}}
        before = deepcopy(data)
        view = compact_view(data, 2500, "지원대상")
        self.assertLessEqual(len(dumps(view)), 2500)
        self.assertEqual(data, before)
        self.assertTrue(view["_context_selection"]["partial_view"])
        self.assertEqual(view["conditions"], data["conditions"])
        for excerpt in view["data"]["chunk_text"]["verbatim_excerpts"]:
            self.assertEqual(excerpt["text"], text[excerpt["start"]:excerpt["end"]])

    def test_registry_deduplicates_across_reports_without_losing_references(self):
        a, b = report("a"), report("b")
        b["evidence"][0]["tool_call_id"] = "another"
        packed = report_payload({"a": a, "b": b})
        self.assertEqual(len(packed["original_evidence"]), 1)
        self.assertEqual(packed["reports"]["b"]["original_evidence_indices"], [0])
        self.assertNotIn("evidence", packed["reports"]["a"])
        self.assertTrue(a["evidence"])

    def test_large_policy_reports_reach_final_with_bounded_view_raw_kept(self):
        reports = {}
        for i in range(8):
            r = report(str(i))
            r["evidence"][0]["result"]["data"] = {
                "document_id": str(i), "chunk_text": "신청 지원 조건은 본문 확인. " * 3000}
            reports[str(i)] = r
        original = deepcopy(reports)
        factory = ScriptedFactory([ai("확인한 발췌 범위의 답변")])
        ctx = Context(Settings(), tools=[], model_factory=factory)
        result = ctx.synthesize("신청 지원", reports)
        self.assertTrue(result["final_answer"])
        body = factory.calls[0]["messages"][1].content
        self.assertLessEqual(len(body), 28000)
        parsed = json.loads(body)
        self.assertEqual(len(parsed["original_evidence"]), 8)
        self.assertEqual(reports, original)
        self.assertTrue(any(e["event"] == "context_selected" for e in ctx.events_snapshot()))


if __name__ == "__main__":
    unittest.main()
