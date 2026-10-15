"""Provider-free LangGraph topology tests for three static MAS mechanisms."""
from copy import deepcopy
import json
import os
from threading import Barrier, Lock
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from common.config import Settings
from common.agents import MultiAgent
from common.orchestration import Context

from mas_central_parallel.graph import build_graph as central_graph
from mas_sequential.graph import build_graph as sequential_graph
from mas_review.graph import build_graph as review_graph


def initial():
    return {"question": "선행 비교 후 선택한 대상에 대한 근거를 알려줘", "reports": {},
            "tasks": [], "final_answer": "", "status": "running", "stop_reason": ""}


def task(identifier, dependencies=None):
    return {"id": identifier, "instruction": "조사 " + identifier, "depends_on": dependencies or []}


class FakeContext:
    def __init__(self, tasks, *, synchronize=False, fail_text_at=None):
        self.tasks = tasks
        self.plan_calls = []
        self.independent_plans = []
        self.work_calls = []
        self.text_calls = []
        self.synthesis_calls = []
        self.events = []
        self.barrier = Barrier(len(tasks)) if synchronize else None
        self.lock = Lock()
        self.fail_text_at = fail_text_at

    def plan(self, question, instruction, prior_reports=None, max_tasks=4, independent=False):
        self.plan_calls.append((question, instruction, prior_reports, max_tasks))
        self.independent_plans.append(independent)
        return deepcopy(self.tasks)

    def work(self, question, task, prior_reports=None, max_steps=2):
        with self.lock:
            self.work_calls.append((task["id"], deepcopy(prior_reports), max_steps))
            self.events.append("work:" + task["id"])
        if self.barrier:
            self.barrier.wait(timeout=3)
        return {"task_id": task["id"], "answer": "결론 " + task["id"],
                "evidence": [{"source": "raw_table", "data": {"target": task["id"], "value": 85}}],
                "status": "success", "stop_reason": "final_answer"}

    def text(self, question, instruction, reports=None, final=False):
        self.text_calls.append((instruction, deepcopy(reports), final))
        self.events.append("text:" + str(len(self.text_calls)))
        if len(self.text_calls) == self.fail_text_at:
            raise RuntimeError("secret-key-must-not-leak")
        return "통합초안표식" if len(self.text_calls) == 1 else "검토지적표식"

    def synthesize(self, question, reports, instruction=""):
        self.synthesis_calls.append((deepcopy(reports), instruction))
        self.events.append("synthesize")
        return {"final_answer": "최종 답변", "status": "success", "stop_reason": "final_answer"}

    def metrics(self):
        return {"model_calls": 1 + len(self.work_calls) + len(self.text_calls) + len(self.synthesis_calls)}


class StaticGraphsTests(unittest.TestCase):
    def test_facade_exports_context_events_instead_of_empty_graph_placeholder(self):
        ctx = FakeContext([task("a")])
        recorded = [{"role": "worker:a", "event": "tool_result", "status": "found"}]
        ctx.events_snapshot = lambda: deepcopy(recorded)
        with patch("common.orchestration.Context", return_value=ctx):
            result = MultiAgent("mas_central_parallel", Settings(), tools=[]).answer(
                {"messages": [["user", "질문"]], "reference_evidence": "gold-is-not-context"})
        self.assertEqual(result["events"], recorded)
        self.assertEqual(result["architecture"], "mas_central_parallel")
        self.assertEqual(ctx.plan_calls[0][0], "질문")
        self.assertNotIn("gold-is-not-context", str(ctx.plan_calls))

    def test_each_graph_integrates_real_shared_context_with_fake_model(self):
        @tool
        def read_fact() -> dict:
            """Synthetic read-only fact."""
            return {"status": "found", "source": "test_table", "data": {"count": 85}, "conditions": {}}

        for builder, extra in ((central_graph, []), (sequential_graph, []),
                               (review_graph, ["초안", "검토"])):
            with self.subTest(builder=builder), patch.dict(os.environ, {
                    "LANGSMITH_TRACING": "false", "LANGCHAIN_TRACING_V2": "false"}):
                scripted = iter([
                    AIMessage(content=json.dumps({"tasks": [task("a")]})),
                    AIMessage(content="", tool_calls=[{"name": "read_fact", "args": {}, "id": "fact-1", "type": "tool_call"}]),
                    AIMessage(content="85개를 확인했습니다."),
                    *[AIMessage(content=text) for text in extra], AIMessage(content="최종 85개"),
                ])

                class Model:
                    def bind(self, **kwargs):
                        return self

                    def bind_tools(self, tools):
                        return self

                    def invoke(self, messages, config=None):
                        return next(scripted)

                ctx = Context(Settings(), tools=[read_fact], model_factory=lambda settings, timeout=None: Model())
                result = builder(ctx).invoke(initial())
                self.assertEqual(result["status"], "success", (result, ctx.events_snapshot()))
                self.assertEqual(result["final_answer"], "최종 85개")
                self.assertEqual(result["metrics"]["model_calls"], 4 + len(extra))
                self.assertEqual(result["metrics"]["tool_calls"], 1)

    def test_central_fanout_is_concurrent_and_fanin_once_without_cross_reports(self):
        ctx = FakeContext([task("a"), task("b")], synchronize=True)
        result = central_graph(ctx).invoke(initial())
        self.assertEqual(set(result["reports"]), {"a", "b"})
        self.assertTrue(all(report["status"] == "success" for report in result["reports"].values()))
        self.assertEqual(len(ctx.synthesis_calls), 1)
        self.assertEqual(set(ctx.synthesis_calls[0][0]), {"a", "b"})
        self.assertTrue(all(call[1] is None for call in ctx.work_calls))
        self.assertEqual(result["final_answer"], "최종 답변")

    def test_central_rejects_dependent_plan_instead_of_silently_ignoring_edges(self):
        ctx = FakeContext([task("a"), task("b", ["a"])])
        result = central_graph(ctx).invoke(initial())
        self.assertEqual(result["status"], "error")
        self.assertEqual(ctx.work_calls, [])
        self.assertEqual(ctx.synthesis_calls, [])

    def test_sequential_topological_order_passes_raw_evidence_not_only_summary(self):
        ctx = FakeContext([task("next", ["first"]), task("first")])
        result = sequential_graph(ctx).invoke(initial())
        self.assertEqual([call[0] for call in ctx.work_calls], ["first", "next"])
        predecessor = ctx.work_calls[1][1]["first"]
        self.assertEqual(predecessor["answer"], "결론 first")
        self.assertEqual(predecessor["evidence"][0]["data"]["value"], 85)
        self.assertEqual(set(result["reports"]), {"first", "next"})
        self.assertEqual(len(ctx.synthesis_calls), 1)

    def test_sequential_cycle_or_missing_dependency_stops_before_work(self):
        for tasks in ([task("a", ["b"]), task("b", ["a"])], [task("a", ["missing"])]):
            with self.subTest(tasks=tasks):
                ctx = FakeContext(tasks)
                result = sequential_graph(ctx).invoke(initial())
                self.assertEqual(result["status"], "error")
                self.assertEqual(ctx.work_calls, [])

    def test_review_runs_after_integrated_draft_and_before_single_revision(self):
        ctx = FakeContext([task("a"), task("b")], synchronize=True)
        result = review_graph(ctx).invoke(initial())
        self.assertEqual(len(ctx.text_calls), 2)
        self.assertEqual(len(ctx.synthesis_calls), 1)
        self.assertEqual(ctx.events[-3:], ["text:1", "text:2", "synthesize"])
        self.assertIn("통합초안표식", ctx.text_calls[1][0])
        self.assertEqual(set(ctx.text_calls[1][1]), {"a", "b"})
        self.assertIn("evidence", ctx.text_calls[1][1]["a"])
        self.assertTrue(all(call[2] is False for call in ctx.text_calls))
        self.assertIn("통합초안표식", ctx.synthesis_calls[0][1])
        self.assertIn("검토지적표식", ctx.synthesis_calls[0][1])
        self.assertEqual(result["final_answer"], "최종 답변")

    def test_review_uses_same_central_plan_and_worker_caps(self):
        central, review = FakeContext([task("a")]), FakeContext([task("a")])
        central_graph(central).invoke(initial())
        review_graph(review).invoke(initial())
        self.assertEqual(central.plan_calls, review.plan_calls)
        self.assertEqual(central.independent_plans, [True])
        self.assertEqual(review.independent_plans, [True])
        self.assertEqual(central.work_calls, review.work_calls)

    def test_review_failure_is_explicit_and_draft_not_published_as_final(self):
        for fail_at in (1, 2):
            with self.subTest(fail_at=fail_at):
                ctx = FakeContext([task("a")], fail_text_at=fail_at)
                result = review_graph(ctx).invoke(initial())
                self.assertEqual(result["status"], "incomplete")
                self.assertEqual(result["final_answer"], "최종 답변")
                self.assertEqual(len(ctx.synthesis_calls), 1)
                self.assertIn("error", result["stop_reason"])
                self.assertNotEqual(result["final_answer"], result.get("draft"))
                self.assertNotIn("secret-key", str(result))

    def test_duplicate_task_ids_cannot_overwrite_report(self):
        for builder in (central_graph, sequential_graph, review_graph):
            with self.subTest(builder=builder):
                ctx = FakeContext([task("same"), task("same")])
                result = builder(ctx).invoke(initial())
                self.assertEqual(result["status"], "error")
                self.assertEqual(ctx.work_calls, [])


if __name__ == "__main__":
    unittest.main()
