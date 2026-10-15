"""Read the question workbook and explicitly sync references to LangSmith.

Validation is local: ``python -m common.evaluation.dataset workbook.xlsx``.
This command never uploads anything. Call sync_dataset(..., apply=True) only
when intentionally updating the named remote dataset. Existing datasets and
their evaluator bindings are preserved; examples are never deleted.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import posixpath
import re
from typing import Any, Iterable
from uuid import UUID, uuid5
from xml.etree import ElementTree as ET
from zipfile import ZipFile

SHEET_NAME = "질문·확인할 내용"
HEADERS = ("문항 ID", "질문 유형", "분야", "질문", "확인할 내용", "정답 근거")
_NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


@dataclass(frozen=True)
class EvaluationExample:
    question_id: str
    question_type: str
    domain: str
    question: str
    required_items: tuple[str, ...]
    reference_evidence: str

    def __post_init__(self) -> None:
        for field in ("question_id", "question_type", "domain", "question", "reference_evidence"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Missing {field}")
        if not self.required_items or any(not isinstance(x, str) or not x.strip() for x in self.required_items):
            raise ValueError(f"Missing required_items: {self.question_id}")

    @property
    def inputs(self) -> dict[str, Any]:
        # Analysis labels and gold answers must not enter the agent's input.
        return {"messages": [["user", self.question]]}

    @property
    def outputs(self) -> dict[str, Any]:
        return {"required_items": list(self.required_items), "reference_evidence": self.reference_evidence}

    @property
    def metadata(self) -> dict[str, str]:
        return {"question_id": self.question_id, "question_type": self.question_type, "domain": self.domain}


def _required_items(value: str) -> tuple[str, ...]:
    # Numbered lines start a new item; wrapped unnumbered lines continue it.
    items: list[str] = []
    for line in value.splitlines():
        line = line.strip()
        if not line:
            continue
        match = re.match(r"^(?:\d+[.)]|[-•])\s*(.+)$", line)
        if match:
            items.append(match.group(1).strip())
        elif items:
            items[-1] += " " + line
        else:
            items.append(line)
    return tuple(items)


def _xlsx_rows(path: Path, sheet_name: str) -> list[tuple[int, list[str]]]:
    """Read ordinary OOXML cells without requiring an Excel process or library."""
    with ZipFile(path) as archive:
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        sheet = next((s for s in workbook.findall("s:sheets/s:sheet", _NS) if s.get("name") == sheet_name), None)
        if sheet is None:
            raise ValueError(f"Sheet not found: {sheet_name}")
        relations = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        rel_id = sheet.get(f"{{{_REL_NS}}}id")
        relation = next(r for r in relations if r.get("Id") == rel_id)
        if relation.get("TargetMode") == "External":
            raise ValueError("External worksheet references are unsupported")
        target = relation.get("Target", "")
        member = target.lstrip("/") if target.startswith("/") else posixpath.normpath("xl/" + target)
        shared: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            shared = ["".join(t.text or "" for t in node.findall(".//s:t", _NS)) for node in root]
        root = ET.fromstring(archive.read(member))
        rows = []
        for row in root.findall("s:sheetData/s:row", _NS):
            values = [""] * len(HEADERS)
            for cell in row.findall("s:c", _NS):
                address = cell.get("r", "")
                col = re.sub(r"\d", "", address)
                if col not in "ABCDEF" or len(col) != 1:
                    continue
                if cell.find("s:f", _NS) is not None:
                    raise ValueError(f"Formula in evaluation source cell {address}; use explicit values")
                kind = cell.get("t")
                if kind == "inlineStr":
                    text = "".join(t.text or "" for t in cell.findall(".//s:t", _NS))
                else:
                    text = cell.findtext("s:v", default="", namespaces=_NS)
                    if kind == "s":
                        text = shared[int(text)]
                    elif kind == "e":
                        raise ValueError(f"Excel error in {address}")
                values[ord(col) - ord("A")] = text.strip()
            if any(values):
                rows.append((int(row.get("r", "0")), values))
        return rows


def load_examples(path: str | Path, *, sheet_name: str = SHEET_NAME) -> list[EvaluationExample]:
    rows = _xlsx_rows(Path(path), sheet_name)
    if not rows or tuple(rows[0][1]) != HEADERS:
        raise ValueError(f"Expected headers: {HEADERS}")
    examples = []
    for row_number, values in rows[1:]:
        try:
            examples.append(EvaluationExample(*values[:4], _required_items(values[4]), values[5]))
        except ValueError as error:
            raise ValueError(f"Workbook row {row_number}: {error}") from error
    return validate_examples(examples)


def validate_examples(examples: Iterable[EvaluationExample]) -> list[EvaluationExample]:
    examples = list(examples)
    if not examples:
        raise ValueError("Dataset has no examples")
    seen: set[str] = set()
    for example in examples:
        if example.question_id in seen:
            raise ValueError(f"Duplicate question_id: {example.question_id}")
        seen.add(example.question_id)
    return examples


def sync_dataset(client: Any, dataset_name: str, examples: Iterable[EvaluationExample], *, apply: bool = False) -> dict[str, Any]:
    """Preview/create/update examples by question_id, retaining evaluator bindings.

    Preview performs remote reads only, using the explicitly supplied client.
    Existing examples without matching IDs stay untouched. This function does
    not configure evaluators, call a model, delete data, or replace the dataset.
    """
    if not dataset_name.strip():
        raise ValueError("dataset_name is required")
    examples = validate_examples(examples)
    exists = client.has_dataset(dataset_name=dataset_name)
    dataset = client.read_dataset(dataset_name=dataset_name) if exists else None
    remote = list(client.list_examples(dataset_id=dataset.id)) if exists else []
    by_id: dict[str, Any] = {}
    for item in remote:
        question_id = (item.metadata or {}).get("question_id")
        if question_id is None:
            continue
        if question_id in by_id:
            raise ValueError(f"Duplicate remote question_id: {question_id}")
        by_id[question_id] = item
    plan = {"dataset_name": dataset_name, "create_dataset": not exists,
            "create": [], "update": [], "unchanged": [], "applied": apply}
    for example in examples:
        old = by_id.get(example.question_id)
        if old is None:
            action = "create"
        elif old.inputs != example.inputs or old.outputs != example.outputs or any(
            (old.metadata or {}).get(key) != value for key, value in example.metadata.items()
        ):
            action = "update"
        else:
            action = "unchanged"
        plan[action].append(example.question_id)
    if not apply:
        return plan
    if dataset is None:
        dataset = client.create_dataset(dataset_name=dataset_name, description="Shared agent comparison questions and evaluation-only references")
    for example in examples:
        if example.question_id in plan["create"]:
            client.create_example(dataset_id=dataset.id,
                example_id=uuid5(UUID(str(dataset.id)), example.question_id),
                inputs=example.inputs, outputs=example.outputs, metadata=example.metadata)
        elif example.question_id in plan["update"]:
            old = by_id[example.question_id]
            client.update_example(old.id, inputs=example.inputs, outputs=example.outputs,
                metadata={**(old.metadata or {}), **example.metadata})
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate question workbook locally; no upload or model calls")
    parser.add_argument("workbook", type=Path)
    args = parser.parse_args()
    examples = load_examples(args.workbook)
    print(f"Validated {len(examples)} questions; no remote changes made.")


if __name__ == "__main__":
    main()
