"""Bounded allocate → work → inspect/reallocate loop, with explicit early finish."""
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from common.orchestration import MASState, OrchestrationError
from common.orchestration.evidence import report_payload

MAX_CYCLES = 3


class DynamicState(MASState):
    cycle: int
    seen_instructions: list[str]
    allocation_stop: str
    active_task: dict
    previous_evidence_count: int


def _signature(task):
    return task.get("domain", "general") + ":" + " ".join(task["instruction"].split()).casefold()


def build_graph(ctx):
    graph = StateGraph(DynamicState)

    def allocate(state):
        cycle = state.get("cycle", 0)
        budget = getattr(ctx, "budget", None)
        if budget is not None and budget.should_finalize:
            return {"tasks": [], "allocation_stop": "budget_reserved_for_synthesis"}
        if cycle >= MAX_CYCLES:
            return {"tasks": [], "allocation_stop": "max_cycles"}
        prior = state["reports"]
        evidence_count = len(report_payload(prior)["original_evidence"])
        if cycle > 0 and evidence_count <= state.get("previous_evidence_count", -1):
            return {"tasks": [], "allocation_stop": "no_new_evidence"}
        instruction = (
            "동적 조정자입니다. 원 질문을 처리할 다음 작업을 최대 두 개 배정하세요. "
            "도메인과 depends_on을 명시하고 같은 계획을 반복하지 마세요. "
            + ("중간 답변과 원본 도구 근거를 검사해 남은 누락에 맞춰 작업·담당 관점을 재배정하세요. "
               "답변 근거가 충분하면 tasks를 빈 목록으로 반환해 종료하세요."
               if prior else "첫 조회에서 얻을 근거를 계획하세요. 정답이나 필수항목은 제공되지 않습니다."))
        try:
            planned = ctx.plan(state["question"], instruction, prior_reports=prior, max_tasks=2)
        except OrchestrationError as exc:
            if not prior:
                raise
            # Preserve the reserved synthesis call and all already collected
            # evidence when a later planner hits budget/context/model failure.
            return {"tasks": [], "allocation_stop": "planning_error:" + exc.code}
        seen = list(state.get("seen_instructions", []))
        fresh = []
        accepted = set(seen)
        for task in planned:
            signature = _signature(task)
            if signature not in accepted:
                fresh.append(task)
                accepted.add(signature)
        # Dropping a duplicate dependency would make its dependent task unsafe.
        removed_ids = {task["id"] for task in planned if task not in fresh}
        fresh = [task for task in fresh if not (set(task.get("depends_on", [])) & removed_ids)]
        remap = {task["id"]: f"cycle{cycle + 1}_{i + 1}" for i, task in enumerate(fresh)}
        tasks = [{**task, "id": remap[task["id"]],
                  "depends_on": [remap.get(dep, dep) for dep in task.get("depends_on", [])]}
                 for task in fresh]
        return {"tasks": tasks, "cycle": cycle + 1,
                "previous_evidence_count": evidence_count,
                "seen_instructions": seen + [_signature(task) for task in fresh],
                "allocation_stop": "" if tasks else "duplicate_plan" if planned else "planner_finished"}

    def route_allocated(state):
        return "dispatch_ready_workers" if state["tasks"] else "final_synthesis"

    def dispatch(state):
        pending = [task for task in state["tasks"] if task["id"] not in state["reports"]]
        if not pending:
            return "inspect_and_reallocate"
        ready = [task for task in pending if all(dep in state["reports"] for dep in task.get("depends_on", []))]
        if not ready:
            raise OrchestrationError("동적 작업의 선행 결과를 찾을 수 없습니다.")
        return [Send("assigned_worker", {**state, "active_task": task}) for task in ready]

    def worker(state):
        task = state["active_task"]
        report = ctx.work(state["question"], task, prior_reports=state["reports"], max_steps=6)
        return {"reports": {task["id"]: {**report, "task_id": task["id"]}}}

    def final_synthesis(state):
        stop = state.get("allocation_stop", "")
        result = ctx.synthesize(state["question"], state["reports"],
            instruction="중간 근거에 따라 재배정한 전체 작업 결과를 종합하세요. "
                        f"작업 배정 종료 사유: {stop}. "
                        "반복 한도나 미해결 내용은 숨기지 마세요.")
        if stop != "planner_finished" and result.get("status") != "error":
            result = {**result, "status": "incomplete", "stop_reason": "allocation_stop:" + stop}
        return result

    graph.add_node("initial_allocation", allocate)
    graph.add_node("inspect_and_reallocate", allocate)
    graph.add_node("dispatch_ready_workers", lambda state: {})
    graph.add_node("assigned_worker", worker)
    graph.add_node("final_synthesis", final_synthesis)
    graph.add_edge(START, "initial_allocation")
    for node in ("initial_allocation", "inspect_and_reallocate"):
        graph.add_conditional_edges(node, route_allocated)
    graph.add_conditional_edges("dispatch_ready_workers", dispatch)
    graph.add_edge("assigned_worker", "dispatch_ready_workers")
    graph.add_edge("final_synthesis", END)
    return graph.compile()
