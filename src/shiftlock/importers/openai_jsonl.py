"""Import OpenAI chat-completions JSONL: the Batch API shape or the fine-tuning shape."""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, SchemaError

from shiftlock.domain import TestCase, ToolCall

ROLES = {"system", "developer", "user", "assistant", "tool"}
PERSONAL_DATA_PATTERNS = [
    re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
    re.compile(r"\+?\d[\d\s().-]{8,}\d"),
]


@dataclass
class ImportResult:
    cases: list[TestCase] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)

    @property
    def flagged(self) -> list[TestCase]:
        return [case for case in self.cases if case.has_personal_data]


def import_openai_jsonl(path: Path) -> ImportResult:
    result = ImportResult()
    seen_ids: set[str] = set()
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                case = parse_record(json.loads(line), default_id=f"line-{line_number}")
            except (json.JSONDecodeError, ValueError) as error:
                result.issues.append(f"line {line_number}: {error}")
                continue
            if case.id in seen_ids:
                result.issues.append(f"line {line_number}: duplicate id {case.id!r}")
                continue
            seen_ids.add(case.id)
            result.cases.append(case)
    return result


def parse_record(record: Any, default_id: str) -> TestCase:
    if not isinstance(record, dict):
        raise ValueError("each line must be a JSON object")
    body = record.get("body", record)
    if not isinstance(body, dict):
        raise ValueError("'body' must be an object")
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("'messages' must be a non-empty list")
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in ROLES:
            raise ValueError(f"message with invalid role: {message!r:.80}")

    expected_text: str | None = None
    expected_tool_calls: list[ToolCall] | None = None
    if "body" not in record and messages[-1]["role"] == "assistant":
        final = messages.pop()
        expected_text = final.get("content") if isinstance(final.get("content"), str) else None
        if final.get("tool_calls"):
            expected_tool_calls = [parse_tool_call(call) for call in final["tool_calls"]]
        if not messages:
            raise ValueError("no input messages before the expected assistant reply")

    tools = body.get("tools") or []
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "function" or "name" not in tool.get("function", {}):
            raise ValueError("tools must use the OpenAI function shape")

    response_format = body.get("response_format")
    output_schema = None
    if isinstance(response_format, dict) and response_format.get("type") == "json_schema":
        output_schema = response_format.get("json_schema", {}).get("schema")
        try:
            Draft202012Validator.check_schema(output_schema or {})
        except SchemaError as error:
            raise ValueError(f"invalid JSON schema in response_format: {error.message}") from error

    return TestCase(
        id=str(record.get("custom_id") or body.get("metadata", {}).get("id") or default_id),
        messages=messages,
        tools=tools,
        tool_choice=body.get("tool_choice"),
        response_format=response_format,
        output_schema=output_schema,
        temperature=body.get("temperature"),
        max_tokens=body.get("max_completion_tokens") or body.get("max_tokens"),
        expected_text=expected_text,
        expected_tool_calls=expected_tool_calls,
        has_personal_data=contains_personal_data(messages),
    )


def parse_tool_call(call: dict[str, Any]) -> ToolCall:
    function = call.get("function", {})
    raw_arguments = function.get("arguments", "")
    if not isinstance(raw_arguments, str):
        raw_arguments = json.dumps(raw_arguments)
    return ToolCall(
        name=function.get("name", ""), arguments=parse_arguments(raw_arguments), raw_arguments=raw_arguments
    )


def parse_arguments(raw_arguments: str) -> dict[str, Any] | None:
    try:
        arguments = json.loads(raw_arguments or "{}")
    except json.JSONDecodeError:
        return None
    return arguments if isinstance(arguments, dict) else None


def contains_personal_data(messages: list[dict[str, Any]]) -> bool:
    text = json.dumps([message.get("content") for message in messages])
    return any(pattern.search(text) for pattern in PERSONAL_DATA_PATTERNS)
