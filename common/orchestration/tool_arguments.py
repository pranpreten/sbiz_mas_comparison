"""Lossless schema-based adapters; never infer codes or split user text."""
from copy import deepcopy
import json


def normalize_calls(calls, tools):
    normalized = deepcopy(calls)
    changes = []
    for call in normalized:
        tool = tools.get(call.get("name"))
        if tool is None or not isinstance(call.get("args"), dict):
            continue
        schema = tool.args_schema.model_json_schema()
        for key, prop in schema.get("properties", {}).items():
            value = call["args"].get(key)
            if prop.get("type") != "array" or prop.get("items", {}).get("type") != "string" or not isinstance(value, str):
                continue
            # A scalar string denotes ONE item. Never split commas or invent
            # identifiers. JSON-encoded arrays are decoded only when valid.
            if value.lstrip().startswith("["):
                try:
                    parsed = json.loads(value)
                except ValueError:
                    continue
                if not isinstance(parsed, list) or any(not isinstance(x, str) for x in parsed):
                    continue
            else:
                parsed = [value]
            call["args"][key] = parsed
            changes.append({"tool": call["name"], "field": key, "tool_call_id": call["id"]})
    return normalized, changes
