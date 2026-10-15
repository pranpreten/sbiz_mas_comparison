"""No-network regressions for policy -> legal evidence handoff and truncation."""
import json
import unittest
from langchain_core.tools import tool
from common.config import Settings
from common.orchestration import Context
from common.orchestration.context import validate_plan, OrchestrationError
from test_orchestration import ScriptedFactory, ai, task
from mas_sequential.graph import build_graph
from common.orchestration.tool_arguments import normalize_calls


@tool
def get_policy_document(document_id: str) -> dict:
    """Synthetic policy, not an evaluation answer."""
    return {"status": "found", "source": "fixture.policy", "data": {"eligible": "test-business", "law": "Fixture Act"}}


@tool
def search_law_articles(law_name: str) -> dict:
    """Find a synthetic article identifier."""
    return {"status": "found", "source": "fixture.law", "data": {"mst": "fixture", "article_seq": 87, "law": law_name}}


@tool
def get_law_article(mst: str, article_seq: int) -> dict:
    """Read synthetic legal conditions."""
    return {"status": "found", "source": "fixture.law", "data": {"condition": "synthetic criterion", "article_seq": article_seq}}


def call(name, args, number):
    return ai("", [{"name": name, "args": args, "id": "call_" + str(number)}])


@tool
def keyword_lookup(keywords: list[str]) -> dict:
    """Synthetic strict array tool."""
    return {"status": "found", "source": "fixture", "data": keywords}


class DependencyResearchTests(unittest.TestCase):
    def test_policy_reads_third_tool_result_before_completion(self):
        factory = ScriptedFactory([
            call("get_policy_document", {"document_id": "fixture"}, i) for i in range(3)
        ] + [ai("Eligibility confirmed from the last detail.")])
        ctx = Context(Settings(), tools=[get_policy_document], model_factory=factory)
        result = ctx.work("query", {**task(), "required_sources": ["policy"],
                          "deliverables": ["eligibility"]}, max_steps=10)
        self.assertEqual(result["status"], "success")
        self.assertEqual(ctx.metrics()["model_calls"], 4)
        self.assertEqual(ctx.metrics()["tool_calls"], 3)
        # Policy follow-up must not inject the legal search instructions.
        self.assertNotIn("위임 법령 검색", factory.calls[-1]["messages"][-1].content)

    def test_workers_cannot_spend_reserved_draft_review_and_final_calls(self):
        factory = ScriptedFactory([call("get_policy_document", {"document_id": "fixture"}, 1),
                                   ai("draft"), ai("review"), ai("final")])
        ctx = Context(Settings(max_model_calls=4), tools=[get_policy_document], model_factory=factory)
        ctx.worker_tail_calls = 3
        result = ctx.work("query", task(), max_steps=6)
        self.assertEqual(result["stop_reason"], "budget_exceeded")
        self.assertEqual(ctx.metrics()["model_calls"], 1)
        reports = {"t1": result}
        self.assertEqual(ctx.text("query", "draft", reports), "draft")
        self.assertEqual(ctx.text("query", "review", reports), "review")
        self.assertEqual(ctx.synthesize("query", reports)["final_answer"], "final")
        self.assertEqual(ctx.metrics()["model_calls"], 4)

    def test_scalar_array_argument_is_repaired_once_with_auditable_input(self):
        factory = ScriptedFactory([call("keyword_lookup", {"keywords": "one keyword"}, 1), ai("found")])
        ctx = Context(Settings(), tools=[keyword_lookup], model_factory=factory)
        result = ctx.work("query", task(), max_steps=2)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["evidence"][0]["args"], {"keywords": ["one keyword"]})
        self.assertEqual(result["evidence"][0]["requested_args"], {"keywords": "one keyword"})
        self.assertEqual(ctx.metrics()["tool_calls"], 1)

    def test_adapter_does_not_split_text_or_guess_broken_json(self):
        for original, expected in [('a,b', ['a,b']), ('["a", "b"]', ['a', 'b']), ('["a"', '["a"'), ('[1]', '[1]')]:
            calls, _ = normalize_calls([{"name": "keyword_lookup", "id": "a", "args": {"keywords": original}}],
                                       {"keyword_lookup": keyword_lookup})
            self.assertEqual(calls[0]["args"]["keywords"], expected)

    def test_repeated_identical_bad_request_stops_before_another_tool_attempt(self):
        factory = ScriptedFactory([call("keyword_lookup", {"keywords": 123}, 1),
                                   call("keyword_lookup", {"keywords": 123}, 2)])
        ctx = Context(Settings(), tools=[keyword_lookup], model_factory=factory)
        result = ctx.work("query", task(), max_steps=6)
        self.assertEqual(result["stop_reason"], "repeated_tool_error")
        self.assertEqual(ctx.metrics()["tool_calls"], 1)

    def test_policy_only_evidence_cannot_complete_a_legal_task(self):
        work = {**task(), "deliverables": ["legal conditions"], "required_sources": ["law"]}
        factory = ScriptedFactory([
            call("get_policy_document", {"document_id": "fixture"}, 1),
            ai("The policy does not list the legal conditions."),
            call("get_law_article", {"mst": "fixture", "article_seq": 87}, 2),
            ai("Found the legal conditions."),
        ])
        ctx = Context(Settings(), tools=[get_policy_document, get_law_article], model_factory=factory)
        result = ctx.work("Who qualifies and by what legal criteria?", work, max_steps=6)
        self.assertEqual(result["status"], "success")
        self.assertEqual(ctx.metrics()["model_calls"], 4)
        self.assertIn('"missing_sources": ["law"]', factory.calls[2]["messages"][-1].content)
        self.assertEqual(result["evidence"][-1]["tool"], "get_law_article")

    def test_truncated_scoped_work_is_not_success_even_with_some_evidence(self):
        factory = ScriptedFactory([call("get_policy_document", {"document_id": "fixture"}, 1)])
        ctx = Context(Settings(), tools=[get_policy_document], model_factory=factory)
        result = ctx.work("question", {**task(), "deliverables": ["policy and legal criteria"],
                          "required_sources": ["policy", "law"]}, max_steps=1)
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["missing_sources"], ["law"])
        self.assertEqual(len(result["evidence"]), 1)

    def test_sequential_dependency_preserves_policy_and_allows_followup_law_search(self):
        tasks = [{**task("policy"), "domain": "policy", "deliverables": ["eligible entity"], "required_sources": ["policy"]},
                 {**task("law", ["policy"]), "domain": "law", "deliverables": ["entity legal conditions"], "required_sources": ["law"]}]
        factory = ScriptedFactory([
            ai(json.dumps({"tasks": tasks})),
            call("get_policy_document", {"document_id": "fixture"}, 1), ai("test-business"),
            call("search_law_articles", {"law_name": "Fixture Act"}, 2),
            call("get_law_article", {"mst": "fixture", "article_seq": 87}, 3),
            call("search_law_articles", {"law_name": "Fixture Decree"}, 4),
            call("get_law_article", {"mst": "fixture", "article_seq": 88}, 5),
            ai("Both act and delegated conditions read."), ai("Combined answer"),
        ])
        ctx = Context(Settings(), tools=[get_policy_document, search_law_articles, get_law_article], model_factory=factory)
        result = build_graph(ctx).invoke({"question": "Identify the eligible entity, then its legal criteria.", "tasks": [], "reports": {}, "status": "running"})
        self.assertEqual(result["status"], "success")
        self.assertEqual(len(result["reports"]["law"]["evidence"]), 4)
        first_law_input = json.loads(factory.calls[3]["messages"][1].content)
        self.assertEqual(first_law_input["task"]["depends_on"], ["policy"])
        self.assertEqual(first_law_input["original_evidence"][0]["result"]["data"]["eligible"], "test-business")
        self.assertEqual(ctx.metrics()["model_calls"], 9)

    def test_plan_contract_rejects_unknown_source_and_preserves_deliverables(self):
        scoped = {**task(), "deliverables": ["criteria"], "required_sources": ["law"]}
        self.assertEqual(validate_plan(json.dumps({"tasks": [scoped]}))[0], scoped)
        scoped["required_sources"] = ["made-up"]
        with self.assertRaises(OrchestrationError):
            validate_plan(json.dumps({"tasks": [scoped]}))

    def test_sequential_can_read_delegated_conditions_after_sixth_turn(self):
        scoped = {**task("law"), "deliverables": ["delegated criteria"], "required_sources": ["law"]}
        factory = ScriptedFactory([ai(json.dumps({"tasks": [scoped]}))] + [
            call("get_law_article", {"mst": "fixture", "article_seq": i}, i)
            for i in range(1, 8)
        ] + [ai("All delegated conditions read."), ai("Combined answer")])
        ctx = Context(Settings(max_model_calls=20), tools=[get_law_article], model_factory=factory)
        result = build_graph(ctx).invoke({"question": "Legal criteria", "tasks": [], "reports": {}, "status": "running"})
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["reports"]["law"]["stop_reason"], "worker_answer")
        self.assertEqual(ctx.metrics()["tool_calls"], 7)

    def test_policy_worker_defers_legal_lookup_instead_of_spending_downstream_budget(self):
        factory = ScriptedFactory([
            call("get_policy_document", {"document_id": "fixture"}, 1),
            call("get_law_article", {"mst": "fixture", "article_seq": 87}, 2),
        ])
        ctx = Context(Settings(), tools=[get_policy_document, get_law_article], model_factory=factory)
        result = ctx.work("compound question", {**task(), "required_sources": ["policy"],
                          "deliverables": ["eligible entity"]}, max_steps=6)
        self.assertEqual(result["stop_reason"], "scope_handoff")
        self.assertEqual(ctx.metrics()["tool_calls"], 1)
        self.assertEqual(result["evidence"][0]["tool"], "get_policy_document")


if __name__ == "__main__":
    unittest.main()
