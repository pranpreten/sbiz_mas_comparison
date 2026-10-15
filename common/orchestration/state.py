"""Parallel graph updates merge reports by task ID rather than replacing peers."""
from copy import deepcopy
from operator import add
from typing import Annotated, TypedDict


def merge_reports(left: dict | None, right: dict | None) -> dict:
    # A later revision intentionally replaces the same task; other tasks remain.
    return {**deepcopy(left or {}), **deepcopy(right or {})}


class MASState(TypedDict, total=False):
    question: str
    tasks: list[dict]
    reports: Annotated[dict[str, dict], merge_reports]
    final_answer: str
    status: str
    stop_reason: str
    metrics: dict
    events: Annotated[list[dict], add]
