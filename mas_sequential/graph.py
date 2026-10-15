"""Planner -> ordered worker loop -> synthesis; no question-type keyword router."""
from langgraph.graph import StateGraph, START, END
from common.orchestration import MASState, Context
from mas_central_parallel.graph import validate_tasks, failure_reason
from .prompts import PLAN, SYNTHESIZE


class SequentialState(MASState):
    task_index: int


def topological_tasks(tasks):
    tasks = validate_tasks(tasks)
    known = {task["id"] for task in tasks}
    if any(dep not in known for task in tasks for dep in task.get("depends_on", [])):
        raise ValueError("Task depends on an unknown ID")
    ordered, completed = [], set()
    while len(ordered) < len(tasks):
        ready = [task for task in tasks if task["id"] not in completed
                 and set(task.get("depends_on", [])) <= completed]
        if not ready:
            raise ValueError("Cyclic task dependencies")
        # Stable order for independent tasks preserves the planner's order.
        task = ready[0]
        ordered.append(task)
        completed.add(task["id"])
    return ordered


def build_graph(ctx: Context):
    def planner(state):
        try:
            tasks = topological_tasks(ctx.plan(state["question"], PLAN, max_tasks=3))
            return {"tasks": tasks, "task_index": 0}
        except Exception as exc:
            return {"status": "error", "stop_reason": failure_reason("planning_error", exc)}

    def worker(state):
        index = state["task_index"]
        task = state["tasks"][index]
        try:
            report = ctx.work(state["question"], task, prior_reports=state["reports"], max_steps=10)
        except Exception as exc:
            report = {"task_id": task["id"], "answer": "", "evidence": [], "status": "error",
                      "stop_reason": failure_reason("worker_error", exc)}
        return {"reports": {task["id"]: report}, "task_index": index + 1}

    graph = StateGraph(SequentialState)
    graph.add_node("planner", planner)
    graph.add_node("worker", worker)
    graph.add_node("synthesizer", lambda state: ctx.synthesize(
        state["question"], state["reports"], instruction=SYNTHESIZE))
    graph.add_node("finish", lambda state: {"metrics": ctx.metrics()})
    graph.add_edge(START, "planner")
    graph.add_conditional_edges("planner", lambda state: "finish" if state.get("status") == "error" else "worker",
                                {"finish": "finish", "worker": "worker"})
    graph.add_conditional_edges("worker", lambda state: "worker" if state["task_index"] < len(state["tasks"]) else "synthesizer",
                                {"worker": "worker", "synthesizer": "synthesizer"})
    graph.add_edge("synthesizer", "finish")
    graph.add_edge("finish", END)
    return graph.compile()
