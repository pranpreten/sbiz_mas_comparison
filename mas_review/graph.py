"""Identical central fanout, then draft -> review -> final revision exactly once."""
from langgraph.graph import StateGraph, START, END
from common.orchestration import Context
from mas_central_parallel.graph import ParallelState, planning_nodes, failure_reason
from .prompts import DRAFT, REVIEW, REVISE


class ReviewState(ParallelState):
    draft: str
    review_notes: str


def build_graph(ctx: Context):
    # Draft, review and final revision must all fit the same question budget.
    ctx.worker_tail_calls = 3
    planner, dispatch, worker = planning_nodes(ctx)

    def draft(state):
        try:
            text = ctx.text(state["question"], DRAFT, reports=state["reports"])
            if not text.strip():
                raise ValueError("Empty integrated draft")
            return {"draft": text}
        except Exception as exc:
            return {"status": "error", "stop_reason": failure_reason("draft_error", exc)}

    def reviewer(state):
        try:
            text = ctx.text(state["question"], REVIEW + "\n[통합 초안]\n" + state["draft"],
                            reports=state["reports"])
            if not text.strip():
                raise ValueError("Empty review")
            return {"review_notes": text}
        except Exception as exc:
            return {"status": "error", "stop_reason": failure_reason("review_error", exc)}

    def reviser(state):
        instruction = REVISE + "\n[통합 초안]\n" + state["draft"] + "\n[검토 결과]\n" + state["review_notes"]
        return ctx.synthesize(state["question"], state["reports"], instruction=instruction)

    def fallback(state):
        reason = state.get("stop_reason", "review_incomplete")
        result = ctx.synthesize(state["question"], state["reports"],
            instruction="초안 또는 검토 단계가 완료되지 않았습니다. 확보한 원본 근거로만 답하고 "
                        "검토를 마쳤다고 표현하지 마세요. 중단 사유: " + reason)
        if result.get("status") != "error":
            result.update(status="incomplete", stop_reason=reason)
        return result

    graph = StateGraph(ReviewState)
    for name, node in [("planner", planner), ("worker", worker), ("draft", draft),
                       ("reviewer", reviewer), ("reviser", reviser)]:
        graph.add_node(name, node)
    graph.add_node("finish", lambda state: {"metrics": ctx.metrics()})
    graph.add_node("fallback", fallback)
    graph.add_edge(START, "planner")
    graph.add_conditional_edges("planner", dispatch, ["worker", "finish"])
    graph.add_edge("worker", "draft")
    graph.add_conditional_edges("draft", lambda state: "fallback" if state.get("status") == "error" else "reviewer",
                                {"fallback": "fallback", "reviewer": "reviewer"})
    graph.add_conditional_edges("reviewer", lambda state: "fallback" if state.get("status") == "error" else "reviser",
                                {"fallback": "fallback", "reviser": "reviser"})
    graph.add_edge("fallback", "finish")
    graph.add_edge("reviser", "finish")
    graph.add_edge("finish", END)
    return graph.compile()
