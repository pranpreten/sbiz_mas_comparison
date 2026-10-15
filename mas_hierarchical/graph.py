"""A real two-level delegation tree with evidence passed between workgroups."""
from typing import TypedDict
from langgraph.graph import END, START, StateGraph
from common.orchestration import MASState, OrchestrationError


class HierarchyState(MASState):
    group_order: list[str]
    group_index: int
    group_tasks: dict
    hierarchy_stop: str


def _ordered(tasks):
    """Stable dependency order; external dependencies are already in prior reports."""
    pending, result = list(tasks), []
    ids = {task["id"] for task in tasks}
    while pending:
        completed = {task["id"] for task in result}
        ready = next((task for task in pending
                      if set(task.get("depends_on", [])) & ids <= completed), None)
        if ready is None:
            raise ValueError("Coordinator returned cyclic group tasks.")
        result.append(ready)
        pending.remove(ready)
    return result


def _groups(task):
    domain = task.get("domain", "general")
    if domain in {"market", "statistics"}:
        return ["data"]
    if domain in {"policy", "law"}:
        return ["documents"]
    return ["data", "documents"]


def build_graph(ctx):
    graph = StateGraph(HierarchyState)

    def upper_coordinator(state):
        tasks = _ordered(ctx.plan(state["question"],
            "상위 조정자입니다. 원 질문을 최대 두 개의 작업그룹 과제로 나누세요. "
            "상권·통계는 market/statistics, 정책·법령은 policy/law 도메인을 지정하고 "
            "선행 결과가 필요한 그룹 사이의 depends_on을 명시하세요. "
            "하위 작업자는 각 그룹 조정자가 다시 배정합니다.", max_tasks=2))
        order = []
        for task in tasks:
            for group in _groups(task):
                if group not in order:
                    order.append(group)
        return {"tasks": tasks, "group_order": order, "group_index": 0, "group_tasks": {}}

    def route_group(state):
        index = state["group_index"]
        if index >= len(state["group_order"]):
            return "upper_synthesis"
        return state["group_order"][index] + "_coordinator"

    def manager(group):
        def run(state):
            assigned = [task for task in state["tasks"] if group in _groups(task)]
            instruction = "\n".join(task["instruction"] for task in assigned)
            label = "상권·통계 자료" if group == "data" else "정책·법령 문서"
            try:
                tasks = ctx.plan(state["question"],
                    f"{label} 작업그룹의 하위 조정자입니다. 상위 조정자가 맡긴 과제: {instruction}\n"
                    "기존 작업그룹의 원본 근거를 사용하고, 이 과제를 최대 두 개 하위 작업으로 분해하세요. "
                    "서로 의존하는 작업에는 depends_on을 설정하세요. 이미 충분하면 빈 목록도 가능합니다.",
                    prior_reports=state["reports"], max_tasks=2)
            except OrchestrationError as exc:
                if not state["reports"]:
                    raise
                return {"hierarchy_stop": "planning_error:" + exc.code}
            tasks = _ordered(tasks)
            remap = {task["id"]: f"{group}_{i + 1}" for i, task in enumerate(tasks)}
            renamed = [{**task, "id": remap[task["id"]],
                        "depends_on": [remap.get(dep, dep) for dep in task.get("depends_on", [])]}
                       for task in tasks]
            return {"group_tasks": {**state.get("group_tasks", {}), group: renamed}}
        return run

    def worker(group, index):
        def run(state):
            tasks = state["group_tasks"].get(group, [])
            if index >= len(tasks):
                return {}
            task = tasks[index]
            report = ctx.work(state["question"], task, prior_reports=state["reports"], max_steps=6)
            return {"reports": {task["id"]: {**report, "task_id": task["id"]}}}
        return run

    graph.add_node("upper_coordinator", upper_coordinator)
    graph.add_edge(START, "upper_coordinator")
    graph.add_conditional_edges("upper_coordinator", route_group)
    for group in ("data", "documents"):
        graph.add_node(f"{group}_coordinator", manager(group))
        graph.add_node(f"{group}_worker_1", worker(group, 0))
        graph.add_node(f"{group}_worker_2", worker(group, 1))
        graph.add_conditional_edges(f"{group}_coordinator", lambda state, group=group:
            "upper_synthesis" if state.get("hierarchy_stop") else f"{group}_worker_1")
        graph.add_edge(f"{group}_worker_1", f"{group}_worker_2")
        graph.add_edge(f"{group}_worker_2", "advance_group")
    graph.add_node("advance_group", lambda state: {"group_index": state["group_index"] + 1})
    graph.add_conditional_edges("advance_group", route_group)
    def upper_synthesis(state):
        stop = state.get("hierarchy_stop", "")
        result = ctx.synthesize(state["question"], state["reports"],
            instruction="상위 조정자로서 하위 작업그룹의 근거를 종합하고 선행 결과와 후속 조회의 연결을 확인하세요. "
                        f"미완료 종료 사유가 있으면 숨기지 마세요: {stop}")
        if stop and result.get("status") != "error":
            result = {**result, "status": "incomplete", "stop_reason": stop}
        return result

    graph.add_node("upper_synthesis", upper_synthesis)
    graph.add_edge("upper_synthesis", END)
    return graph.compile()
