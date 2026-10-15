"""START → agent ⇄ tools → finish → END. 동일 역할의 모델만 반복 호출한다."""
import json
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableConfig
from common.evaluation.runner import extract_final_answer
from common.harness.limits import BudgetExceeded
from common.harness.context import ContextLimitExceeded
from .state import AgentState
from .prompts import SYSTEM_PROMPT


def build_graph(runtime):
    def agent(state: AgentState, config: RunnableConfig):
        try:
            role = SYSTEM_PROMPT + ("\n[이번 실행의 기준일] " + str(runtime.settings.experiment_date)
                + "\n질문에 다른 법적 판단 시점이 없으면 모든 법령 도구의 as_of에 이 날짜를 그대로 넣으세요.")
            response = runtime.invoke_model(role, state["messages"], config=config)
            final_mode = runtime.last_model_final
        except (BudgetExceeded, ContextLimitExceeded) as exc:
            return {"status": "error", "stop_reason": type(exc).__name__}
        except Exception as exc:
            # 외부 SDK 오류 원문은 키/접속정보를 포함할 수 있어 상태에 보관하지 않는다.
            return {"status": "error", "stop_reason": "model_error:" + type(exc).__name__}
        if getattr(response, "invalid_tool_calls", None):
            return {"messages": [response], "status": "error", "stop_reason": "invalid_tool_calls"}
        if response.tool_calls:
            if final_mode:
                return {"messages": [response], "status": "error", "stop_reason": "tool_call_during_finalization"}
            return {"messages": [response], "status": "running"}
        try:
            answer = extract_final_answer({"messages": [response]})
        except ValueError:
            return {"messages": [response], "status": "error", "stop_reason": "invalid_final_answer"}
        evidence = state.get("evidence", [])
        found = any(e["result"].get("status") == "found" and
                    e["result"].get("source") != "decimal_calculator" for e in evidence)
        errors = any(e["result"].get("status") == "error" for e in evidence)
        # 조회 성공이 답변의 정확성을 보장하지는 않는다. 품질은 별도 평가기로 판정한다.
        status = "success" if found and not errors and not final_mode else "incomplete"
        reason = "final_answer" if status == "success" else "limited_or_missing_evidence"
        return {"messages": [response], "final_answer": answer, "status": status, "stop_reason": reason}

    def tools(state: AgentState, config: RunnableConfig):
        calls = state["messages"][-1].tool_calls
        replies = runtime.execute_tools(calls, config=config)
        evidence = list(state.get("evidence", []))
        for call, reply in zip(calls, replies):
            evidence.append({"tool": call["name"], "args": call["args"],
                             "tool_call_id": call["id"], "result": json.loads(reply.content)})
        return {"messages": replies, "evidence": evidence}

    def route(state: AgentState):
        if state.get("status") != "running":
            return "finish"
        return "tools"

    def finish(state: AgentState):
        return {"metrics": runtime.metrics()}

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent)
    graph.add_node("tools", tools)
    graph.add_node("finish", finish)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route, {"tools": "tools", "finish": "finish"})
    graph.add_edge("tools", "agent")
    graph.add_edge("finish", END)
    return graph.compile()
