"""공통 도구를 사용하는 단일 에이전트. 문항마다 새 runtime과 상태를 만든다."""
from langchain_core.messages import HumanMessage
from common.data.database import Database
from common.evaluation.runner import question_inputs
from common.harness import AgentRuntime
from common.model import create_model
from common.tools import create_tools
from .graph import build_graph


class SingleAgent:
    def __init__(self, settings, *, tools=None, model_factory=None, database=None):
        self.settings = settings
        self.database = database
        if tools is None:
            self.database = database or Database(settings.database_url, settings.statement_timeout_ms)
            tools = create_tools(self.database)
        self.tools = tools
        self.model_factory = model_factory or create_model

    def answer(self, inputs):
        clean = question_inputs(inputs)
        runtime = AgentRuntime(self.settings, self.tools, model_factory=self.model_factory)
        graph = build_graph(runtime)
        state = {"messages": [HumanMessage(content=clean["messages"][0][1])],
                 "evidence": [], "final_answer": "", "status": "running", "stop_reason": ""}
        try:
            return graph.invoke(state, config={
                "recursion_limit": self.settings.max_model_calls * 2 + 5,
                "run_name": "single_agent_qa", "tags": ["single_agent", "sbiz"],
                "metadata": {"architecture": "single_agent", "model": self.settings.upstage_model,
                             "experiment_date": str(self.settings.experiment_date)},
            })
        except Exception as exc:
            # 그래프 실행 실패를 정상 답변으로 집계하지 않는다.
            return {"messages": [], "evidence": [], "final_answer": "", "status": "error",
                    "stop_reason": "graph_error:" + type(exc).__name__, "metrics": runtime.metrics()}
