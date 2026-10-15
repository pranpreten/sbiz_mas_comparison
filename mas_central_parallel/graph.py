"""START -> planner -> Send(worker...) -> synthesizer -> finish -> END."""
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from common.orchestration import MASState, Context, OrchestrationError
from .prompts import PLAN, SYNTHESIZE


class ParallelState(MASState):
    task: dict


def failure_reason(prefix, exc):
    return prefix + ":" + (exc.code if isinstance(exc, OrchestrationError) else type(exc).__name__)


def validate_tasks(tasks, *, independent=False):
    """Validate bounded task identities; never silently drop model-planned work."""
    if not isinstance(tasks, list) or not 1 <= len(tasks) <= 3:
        raise ValueError("Plan must contain between one and three tasks")
    seen = set()
    for task in tasks:
        if (not isinstance(task, dict) or not isinstance(task.get("id"), str)
                or not task["id"].strip() or task["id"] in seen
                or not isinstance(task.get("instruction"), str) or not task["instruction"].strip()):
            raise ValueError("Tasks need unique IDs and nonempty instructions")
        deps = task.get("depends_on", [])
        if not isinstance(deps, list) or any(not isinstance(dep, str) for dep in deps):
            raise ValueError("Task dependencies must be ID strings")
        if independent and deps:
            raise ValueError("Independent workers cannot consume predecessor results")
        seen.add(task["id"])
    return tasks


def planning_nodes(ctx: Context):
    """Shared by central baseline and the post-draft review extension."""
    def planner(state):
        try:
            tasks = validate_tasks(ctx.plan(state["question"], PLAN, max_tasks=3, independent=True), independent=True)
            return {"tasks": tasks}
        except Exception as exc:
            return {"status": "error", "stop_reason": failure_reason("planning_error", exc)}

    def dispatch(state):
        if state.get("status") == "error":
            return "finish"
        return [Send("worker", {"question": state["question"], "task": task}) for task in state["tasks"]]

    def worker(state):
        task = state["task"]
        try:
            # No other worker's report is present in this branch's input.
            report = ctx.work(state["question"], task, max_steps=6)
        except Exception as exc:
            report = {"task_id": task["id"], "answer": "", "evidence": [],
                      "status": "error", "stop_reason": failure_reason("worker_error", exc)}
        return {"reports": {task["id"]: report}}

    return planner, dispatch, worker


def build_graph(ctx: Context):
    planner, dispatch, worker = planning_nodes(ctx)
    graph = StateGraph(ParallelState)
    graph.add_node("planner", planner)
    graph.add_node("worker", worker)
    graph.add_node("synthesizer", lambda state: ctx.synthesize(
        state["question"], state["reports"], instruction=SYNTHESIZE))
    graph.add_node("finish", lambda state: {"metrics": ctx.metrics()})
    graph.add_edge(START, "planner")
    graph.add_conditional_edges("planner", dispatch, ["worker", "finish"])
    graph.add_edge("worker", "synthesizer")
    graph.add_edge("synthesizer", "finish")
    graph.add_edge("finish", END)
    return graph.compile()
