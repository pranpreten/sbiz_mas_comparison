"""모든 실험 구조가 같은 도구 목록을 사용한다."""


def create_tools(db):
    """도구 객체 생성만 수행한다. 실제 DB 조회는 invoke 시점에 일어난다."""
    from . import calculator, codes, law, market, policy, statistics
    from common.harness.safe_tools import safe_tool

    result = []
    for module in (codes, market, statistics, policy, law, calculator):
        result.extend(module.create_tools(db))
    names = [tool.name for tool in result]
    if len(names) != len(set(names)):
        raise ValueError("공통 도구 이름이 중복됩니다.")
    return [safe_tool(tool) for tool in result]
