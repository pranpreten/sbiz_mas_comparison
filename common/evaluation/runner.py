"""Evaluate any architecture through one input/output contract.

No graphs, model instances or credentials are created on import. Server-side
evaluator completion is separate from completion of the answer experiment.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Mapping


def _field(obj: Any, key: str, default: Any = None) -> Any:
    return obj.get(key, default) if isinstance(obj, Mapping) else getattr(obj, key, default)


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "\n".join(block["text"] for block in content
            if isinstance(block, dict) and block.get("type") in {"text", "output_text"}
            and isinstance(block.get("text"), str)).strip()
    return ""


def extract_final_answer(result: Any) -> str:
    """Accept plain text, explicit final_answer, or a completed AI message list.

    Never fall back to an earlier AI/tool message: a trailing tool response or
    unfinished tool call means execution has not produced a final answer.
    """
    if isinstance(result, str):
        answer = result.strip()
    elif isinstance(result, Mapping) and "final_answer" in result:
        answer = _text(result["final_answer"])
    elif isinstance(result, Mapping) and result.get("messages"):
        last = result["messages"][-1]
        role = _field(last, "type") or _field(last, "role")
        additional = _field(last, "additional_kwargs", {}) or {}
        if (role not in {"ai", "assistant"} or _field(last, "tool_calls")
                or _field(last, "invalid_tool_calls") or additional.get("tool_calls")):
            raise ValueError("Execution has no completed final AI answer")
        answer = _text(_field(last, "content"))
    else:
        raise ValueError("Target must return text, final_answer, or final AI messages")
    if not answer:
        raise ValueError("Final answer is empty")
    # Some providers emit protocol tokens as plain text after tools are unbound.
    # Reject the whole completion, including truncated markers; never strip,
    # parse, or execute it as an unregistered tool call or grade it as an answer.
    if "<|tool_call" in answer.lower():
        raise ValueError("Final answer contains an unfinished tool-call protocol")
    return answer


def question_inputs(inputs: Mapping[str, Any]) -> dict[str, Any]:
    """Allow only the single user question into the evaluated target."""
    messages = inputs.get("messages")
    if not isinstance(messages, list) or len(messages) != 1:
        raise ValueError("Expected exactly one user question in inputs.messages")
    message = messages[0]
    if isinstance(message, (list, tuple)) and len(message) == 2:
        role, content = message
    else:
        role = _field(message, "role") or _field(message, "type")
        content = _field(message, "content")
    if role not in {"user", "human"} or not isinstance(content, str) or not content.strip():
        raise ValueError("Expected a nonempty user question")
    return {"messages": [["user", content]]}


def make_target(answer_question: Callable[[dict[str, Any]], Any]) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Adapter from any agent implementation to the existing UI evaluator format."""
    def target(inputs: dict[str, Any]) -> dict[str, Any]:
        result = answer_question(deepcopy(question_inputs(inputs)))
        return {"messages": [{"type": "ai", "content": extract_final_answer(result)}]}
    return target


@dataclass
class ExperimentRun:
    experiment_name: str
    rows: list[Any]
    sdk_results: Any

    @property
    def answer_errors(self) -> int:
        return sum(bool(_field(row["run"], "error")) for row in self.rows)

    def metrics(self) -> list[dict[str, Any]]:
        """Root answer-run metrics only; absent tokens remain None, not zero.

        Do not add child totals: LangSmith root usage is already aggregated.
        These metrics do not assert that asynchronous server graders finished.
        """
        metrics = []
        for row in self.rows:
            run = row["run"]
            start, end = _field(run, "start_time"), _field(run, "end_time")
            metrics.append({"run_id": str(_field(run, "id")),
                "latency_seconds": (end - start).total_seconds() if start and end else None,
                "input_tokens": _field(run, "prompt_tokens"),
                "output_tokens": _field(run, "completion_tokens"),
                "total_tokens": _field(run, "total_tokens"),
                "error": _field(run, "error")})
        return metrics


def run_experiment(answer_question: Callable[[dict[str, Any]], Any], *, dataset_name: str,
    experiment_prefix: str, client: Any = None, metadata: dict[str, Any] | None = None,
    max_concurrency: int = 1, num_repetitions: int = 1, experiment: Any = None) -> ExperimentRun:
    """Explicitly run a paid-capable answer experiment on an existing dataset.

    UI-bound evaluators run server-side; do not configure duplicate local judges.
    The caller owns model/graph configuration and per-question state isolation.
    """
    if max_concurrency < 1 or num_repetitions < 1:
        raise ValueError("Concurrency and repetitions must be positive")
    if not dataset_name.strip() or not experiment_prefix.strip():
        raise ValueError("Dataset name and experiment prefix are required")
    if client is None:
        from langsmith import Client
        client = Client()
    examples = list(client.list_examples(dataset_name=dataset_name))
    if not examples:
        raise ValueError("Dataset has no examples")
    for example in examples:
        question_inputs(example.inputs)
        reference = example.outputs or {}
        items = reference.get("required_items")
        evidence = reference.get("reference_evidence")
        if not isinstance(items, list) or not items or any(not isinstance(x, str) or not x.strip() for x in items):
            raise ValueError("Every example needs nonempty required_items")
        if not isinstance(evidence, str) or not evidence.strip():
            raise ValueError("Every example needs reference_evidence")
    results = client.evaluate(make_target(answer_question), data=examples,
        **({"experiment": experiment} if experiment is not None else {"experiment_prefix": experiment_prefix}),
        metadata=metadata or {},
        max_concurrency=max_concurrency, num_repetitions=num_repetitions,
        blocking=True, error_handling="log")
    return ExperimentRun(results.experiment_name, list(results), results)
