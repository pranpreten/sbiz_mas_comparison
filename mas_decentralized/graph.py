"""Four equal peers: independent evidence gathering, exchange, then complement.

No peer allocates work to another. The final writer only combines their reports.
"""
from langgraph.graph import END, START, StateGraph
from common.orchestration import MASState

PEERS = {
    "policy": "지원사업과 신청 조건",
    "market": "상권·지역·업종 수치",
    "statistics": "통계·비교·계산",
    "law": "법령·조문·법적 요건",
}


def build_graph(ctx):
    graph = StateGraph(MASState)

    def peer_node(domain, round_number):
        def run(state):
            prior = {} if round_number == 1 else {
                key: report for key, report in state["reports"].items() if key.endswith("_r1")}
            task_id = f"peer_{domain}_r{round_number}"
            if round_number == 2 and prior.get(f"peer_{domain}_r1", {}).get("stop_reason") == "not_applicable":
                return {"reports": {task_id: {**prior[f"peer_{domain}_r1"], "task_id": task_id}}}
            instruction = (
                f"동등한 동료로서 {PEERS[domain]} 관점에서 원 질문에 필요한 근거를 조사하세요. "
                "우선 담당 관점에 직접 관련된 요청만 조사하세요. 관련 요청이 없고 후속 역할도 없으면 NOT_APPLICABLE만 출력하세요. "
                "다른 분야의 독립 요청을 대신 조사하지 마세요. 담당 요청의 선행 조건을 확인하는 데 필요한 경우에는 공통 도구를 사용할 수 있습니다. "
                + ("다른 동료의 결과를 아직 받지 않은 독립 조사입니다."
                   if round_number == 1 else
                   "모든 동료의 첫 조사 결과와 원본 도구 근거를 읽고 누락·모순을 보완하세요. "
                   "이미 확인한 동일 조회를 반복하지 말고 선행 결과를 다음 조회 조건에 사용하세요."))
            report = ctx.work(state["question"], {"id": task_id, "instruction": instruction,
                              "domain": domain, "depends_on": []}, prior_reports=prior,
                              max_steps=1 if round_number == 1 else 6)
            return {"reports": {task_id: {**report, "task_id": task_id}}}
        return run

    first, second = [], []
    for domain in PEERS:
        first_name, second_name = f"{domain}_round1", f"{domain}_round2"
        graph.add_node(first_name, peer_node(domain, 1))
        graph.add_node(second_name, peer_node(domain, 2))
        graph.add_edge(START, first_name)
        first.append(first_name)
        second.append(second_name)
    graph.add_node("exchange_peer_evidence", lambda state: {})
    graph.add_edge(first, "exchange_peer_evidence")
    for name in second:
        graph.add_edge("exchange_peer_evidence", name)

    def final_writer(state):
        # Round two supersedes round one's unfinished discovery, while retaining
        # both rounds' raw evidence for audit and synthesis.
        reports = dict(state["reports"])
        for domain in PEERS:
            first_id, last_id = f"peer_{domain}_r1", f"peer_{domain}_r2"
            if first_id in reports and last_id in reports:
                reports[last_id] = {**reports[last_id], "evidence": [
                    *reports[first_id].get("evidence", []), *reports[last_id].get("evidence", [])]}
                del reports[first_id]
        return ctx.synthesize(state["question"], reports,
                              instruction="동등한 동료들의 두 차례 조사 근거를 종합하세요. 업무 재배분 없이 최종 답변만 작성하세요.")

    graph.add_node("final_writer", final_writer)
    graph.add_edge(second, "final_writer")
    graph.add_edge("final_writer", END)
    return graph.compile()
