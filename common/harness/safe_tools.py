"""도구 본문 오류를 LangChain 오류 콜백이 기록하기 전에 정리한다."""
from langchain_core.tools import StructuredTool
from langchain_core.runnables import RunnableConfig


class SafeStructuredTool(StructuredTool):
    def _run(self, *args, config: RunnableConfig, run_manager=None, **kwargs):
        try:
            return super()._run(*args, config=config, run_manager=run_manager, **kwargs)
        except Exception as exc:
            return {"status": "error", "error_type": type(exc).__name__,
                    "message": "도구 입력 또는 조회에 실패했습니다. 조건을 확인해 주세요."}


def safe_tool(tool):
    if isinstance(tool, SafeStructuredTool):
        return tool
    if not isinstance(tool, StructuredTool) or tool.func is None:
        raise TypeError("공통 실행기는 동기 StructuredTool을 사용합니다.")
    return SafeStructuredTool(name=tool.name, description=tool.description,
                              args_schema=tool.args_schema, func=tool.func,
                              handle_validation_error="도구 입력 형식이 잘못되었습니다.")
