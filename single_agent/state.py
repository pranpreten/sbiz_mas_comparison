from langgraph.graph import MessagesState


class AgentState(MessagesState):
    evidence: list[dict]
    final_answer: str
    status: str
    stop_reason: str
    metrics: dict
