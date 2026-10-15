"""Topology tests with a deterministic context; no DB or paid model calls."""
import copy
import unittest

from mas_decentralized.graph import build_graph as decentralized
from mas_hierarchical.graph import build_graph as hierarchical
from mas_dynamic.graph import build_graph as dynamic
from common.orchestration import OrchestrationError


def task(task_id, instruction, domain="general", depends_on=None):
    return {"id": task_id, "instruction": instruction, "domain": domain,
            "depends_on": depends_on or []}


class FakeContext:
    def __init__(self, plans=()):
        self.plans = list(plans)
        self.plan_calls = []
        self.work_calls = []
        self.synth_calls = []

    def plan(self, question, instruction, prior_reports=None, max_tasks=4):
        self.plan_calls.append({"instruction": instruction, "prior": copy.deepcopy(prior_reports or {})})
        result = self.plans.pop(0)
        if isinstance(result, Exception):
            raise result
        return copy.deepcopy(result)

    def work(self, question, task, prior_reports=None, max_steps=2):
        self.work_calls.append({"task": copy.deepcopy(task), "prior": copy.deepcopy(prior_reports or {}),
                                "max_steps": max_steps})
        return {"task_id": task["id"], "instruction": task["instruction"],
                "answer": "evidence-derived answer", "status": "success", "stop_reason": "",
                "evidence": [{"source": "test", "raw": task["id"]}]}

    def synthesize(self, question, reports, instruction=""):
        self.synth_calls.append(copy.deepcopy(reports))
        return {"final_answer": "combined", "status": "success", "stop_reason": "", "metrics": {}}


def run_graph(builder, context):
    return builder(context).invoke({"question": "A compound request", "reports": {}, "tasks": [],
                                    "final_answer": "", "status": "running", "stop_reason": ""},
                                   config={"recursion_limit": 60})


class AdaptiveTopologyTests(unittest.TestCase):
    def test_decentralized_first_round_independent_second_receives_all_peers(self):
        ctx = FakeContext()
        result = run_graph(decentralized, ctx)
        first = [call for call in ctx.work_calls if call["task"]["id"].endswith("_r1")]
        second = [call for call in ctx.work_calls if call["task"]["id"].endswith("_r2")]
        self.assertEqual(len(first), 4)
        self.assertEqual(len(second), 4)
        self.assertTrue(all(not call["prior"] for call in first))
        first_ids = {call["task"]["id"] for call in first}
        for call in second:
            self.assertEqual(set(call["prior"]), first_ids)
            self.assertTrue(all(report["evidence"] for report in call["prior"].values()))
        self.assertEqual(len(result["reports"]), 8)
        self.assertEqual(len(ctx.synth_calls), 1)
        self.assertEqual(ctx.plan_calls, [])

    def test_hierarchy_upper_plan_lower_plans_then_subordinate_workers(self):
        ctx = FakeContext([
            [task("a", "find region", "statistics"), task("b", "find program", "policy", ["a"])],
            [task("x", "query data", "statistics"), task("y", "compare data", "statistics", ["x"])],
            [task("z", "search program for selected region", "policy")],
        ])
        result = run_graph(hierarchical, ctx)
        self.assertEqual(len(ctx.plan_calls), 3)
        self.assertEqual(len(ctx.work_calls), 3)
        self.assertIn("상위 조정자", ctx.plan_calls[0]["instruction"])
        self.assertIn("하위 조정자", ctx.plan_calls[1]["instruction"])
        self.assertEqual(ctx.work_calls[1]["task"]["depends_on"], ["data_1"])
        self.assertIn("data_1", ctx.work_calls[1]["prior"])
        self.assertEqual(set(ctx.plan_calls[2]["prior"]), {"data_1", "data_2"})
        self.assertEqual(set(result["reports"]), {"data_1", "data_2", "documents_1"})

    def test_hierarchy_orders_document_before_data_when_upper_dependency_requires(self):
        ctx = FakeContext([
            [task("data", "query selected scope", "market", ["doc"]), task("doc", "find scope", "law")],
            [task("law_work", "read scope", "law")],
            [task("market_work", "query selected scope", "market")],
        ])
        result = run_graph(hierarchical, ctx)
        self.assertEqual(result["group_order"], ["documents", "data"])
        self.assertIn("documents_1", ctx.plan_calls[2]["prior"])

    def test_dynamic_replans_from_intermediate_evidence_and_accepts_finish(self):
        ctx = FakeContext([
            [task("first", "compare regions", "statistics")],
            [task("second", "query selected region", "market", ["cycle1_1"])],
            [],
        ])
        result = run_graph(dynamic, ctx)
        self.assertEqual(len(ctx.work_calls), 2)
        self.assertIn("cycle1_1", ctx.plan_calls[1]["prior"])
        self.assertEqual(ctx.plan_calls[1]["prior"]["cycle1_1"]["evidence"][0]["raw"], "cycle1_1")
        self.assertEqual(result["allocation_stop"], "planner_finished")
        self.assertEqual(set(result["reports"]), {"cycle1_1", "cycle2_1"})

    def test_dynamic_dependency_wave_and_duplicate_plan_stop(self):
        ctx = FakeContext([
            [task("a", "read", "statistics"), task("b", "use read", "market", ["a"])],
            [task("new_id", "read", "statistics")],
        ])
        result = run_graph(dynamic, ctx)
        self.assertEqual(len(ctx.work_calls), 2)
        self.assertEqual(ctx.work_calls[1]["task"]["depends_on"], ["cycle1_1"])
        self.assertIn("cycle1_1", ctx.work_calls[1]["prior"])
        self.assertEqual(result["allocation_stop"], "duplicate_plan")
        self.assertEqual(result["status"], "incomplete")

    def test_dynamic_has_fixed_three_cycle_limit(self):
        ctx = FakeContext([[task(str(i), f"distinct work {i}")] for i in range(5)])
        result = run_graph(dynamic, ctx)
        self.assertEqual(len(ctx.plan_calls), 3)
        self.assertEqual(len(ctx.work_calls), 3)
        self.assertEqual(result["allocation_stop"], "max_cycles")
        self.assertEqual(result["status"], "incomplete")

    def test_dynamic_planner_budget_failure_preserves_reserved_synthesis(self):
        ctx = FakeContext([[task("a", "retrieve")], OrchestrationError("budget_exceeded")])
        result = run_graph(dynamic, ctx)
        self.assertEqual(len(ctx.synth_calls), 1)
        self.assertIn("cycle1_1", ctx.synth_calls[0])
        self.assertEqual(result["final_answer"], "combined")
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["allocation_stop"], "planning_error:budget_exceeded")

    def test_hierarchy_later_manager_failure_preserves_collected_evidence(self):
        ctx = FakeContext([
            [task("a", "find region", "statistics"), task("b", "find program", "policy", ["a"])],
            [task("x", "query data", "statistics")],
            OrchestrationError("context_limit"),
        ])
        result = run_graph(hierarchical, ctx)
        self.assertEqual(len(ctx.synth_calls), 1)
        self.assertIn("data_1", ctx.synth_calls[0])
        self.assertEqual(result["final_answer"], "combined")
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["stop_reason"], "planning_error:context_limit")


if __name__ == "__main__":
    unittest.main()
