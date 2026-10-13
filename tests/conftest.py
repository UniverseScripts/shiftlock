import json
import logging
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from shiftlock.config import Config
from shiftlock.domain import Completion, Outcome, TestCase, ToolCall
from shiftlock.errors import ProviderError

Responder = Callable[[TestCase, str], Completion | ProviderError]


class FakeProvider:
    name = "fake"

    def __init__(self, responder: Responder):
        self.responder = responder
        self.calls: list[tuple[str, str]] = []

    async def complete(self, case: TestCase, model: str, max_tokens: int) -> Completion:
        self.calls.append((case.id, model))
        response = self.responder(case, model)
        if isinstance(response, ProviderError):
            raise response
        return response


def completion(text: str | None = None, tool_calls: list[ToolCall] | None = None, **fields: object) -> Completion:
    values: dict[str, object] = {
        "api": "fake",
        "text": text,
        "tool_calls": tool_calls or [],
        "input_tokens": 100,
        "output_tokens": 20,
        "latency_ms": 500,
        "outcome": Outcome.COMPLETE,
    }
    values.update(fields)
    return Completion.model_validate(values)


def tool_call(name: str, **arguments: object) -> ToolCall:
    return ToolCall(name=name, arguments=dict(arguments), raw_arguments=json.dumps(arguments))


SCHEMA = {
    "type": "object",
    "properties": {"intent": {"type": "string", "enum": ["refund", "other"]}},
    "required": ["intent"],
    "additionalProperties": False,
}
LOOKUP_TOOL = {
    "type": "function",
    "function": {
        "name": "lookup_order",
        "parameters": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
        },
    },
}


def schema_case(case_id: str = "c1") -> TestCase:
    return TestCase(
        id=case_id,
        messages=[{"role": "user", "content": "I want my money back"}],
        response_format={"type": "json_schema", "json_schema": {"name": "intent", "schema": SCHEMA}},
        output_schema=SCHEMA,
    )


def tool_case(case_id: str = "t1", expected: list[ToolCall] | None = None) -> TestCase:
    return TestCase(
        id=case_id,
        messages=[{"role": "user", "content": "Where is order A-100?"}],
        tools=[LOOKUP_TOOL],
        expected_tool_calls=expected,
    )


def make_config(**overrides: object) -> Config:
    values: dict[str, object] = {
        "version": 1,
        "suite": ".shiftlock/suites/default.jsonl",
        "baseline": {"provider": "openai", "model": "base-model"},
        "candidate": {"provider": "openai", "model": "new-model"},
        "thresholds": {"min_cases": 1},
    }
    values.update(overrides)
    return Config.model_validate(values)


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture(autouse=True)
def reset_logging() -> Iterator[None]:
    yield
    logging.getLogger().handlers = []
