"""Small deterministic calculator: no eval, Python expressions or external calls."""
from __future__ import annotations

from decimal import Decimal, InvalidOperation, localcontext


def calculate(operation: str, left: str, right: str, unit: str = "") -> dict:
    """Decimal difference, absolute difference, ratio, percentage change or comparison.

    For ratio and percentage_change the right operand is the denominator/base.
    The caller must match periods, units and populations before calculation.
    """
    if operation not in {"difference", "absolute_difference", "ratio", "percentage_change", "compare"}:
        raise ValueError("지원 연산: difference, absolute_difference, ratio, percentage_change, compare")
    if not isinstance(unit, str) or len(unit) > 100:
        raise ValueError("unit은 100자 이하 문자열이어야 합니다.")
    numbers = []
    for value in (left, right):
        if not isinstance(value, str) or len(value) > 100:
            raise ValueError("숫자를 100자 이하 문자열로 입력하세요.")
        try:
            number = Decimal(value)
        except InvalidOperation as exc:
            raise ValueError("숫자만 입력하세요. 쉼표·단위·수식은 허용하지 않습니다.") from exc
        if not number.is_finite() or (number and not -100 <= number.adjusted() <= 100):
            raise ValueError("유한한 범위의 숫자가 필요합니다.")
        numbers.append(number)
    a, b = numbers
    conditions = {"operation": operation, "left": left, "right": right, "unit": unit}
    if operation in {"ratio", "percentage_change"} and b == 0:
        return {"status": "error", "data": None, "source": "decimal_calculator",
                "conditions": conditions, "error": "zero_denominator"}
    with localcontext() as ctx:
        ctx.prec = 32
        if operation == "compare":
            result = "equal" if a == b else "left_greater" if a > b else "right_greater"
        elif operation == "difference":
            result = str(a - b)
        elif operation == "absolute_difference":
            result = str(abs(a - b))
        elif operation == "ratio":
            result = str(a / b)
        else:
            result = str((a - b) / b * 100)
    return {"status": "found", "data": {"result": result}, "source": "decimal_calculator",
            "conditions": conditions,
            "note": "입력값의 출처·기준시점·단위가 비교 가능한지는 별도 확인해야 합니다. 비율은 최대 32자리 정밀도입니다."}


def create_tools(db=None) -> list:
    from langchain_core.tools import tool

    @tool
    def calculate_values(operation: str, left: str, right: str, unit: str = "") -> dict:
        """조회한 숫자 두 개 계산. 숫자는 단위·쉼표 없는 문자열.
        difference=left-right, absolute_difference=절대 차이, compare=대소 비교,
        ratio=left/right, percentage_change=(left-right)/right*100. 같은 시점·범위·단위를 먼저 확인.
        """
        return calculate(operation, left, right, unit)

    return [calculate_values]
