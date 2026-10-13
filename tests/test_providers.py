import asyncio
from types import SimpleNamespace
from typing import Any

from pydantic import SecretStr

from shiftlock.domain import Outcome
from shiftlock.providers.anthropic_messages import AnthropicMessages, translate_messages, translate_tool_choice
from shiftlock.providers.openai_chat import OpenAIChat
from tests.conftest import LOOKUP_TOOL, SCHEMA, schema_case, tool_case


class Recorder:
    def __init__(self, response: Any):
        self.response = response
        self.request: dict[str, Any] = {}

    async def create(self, **request: Any) -> Any:
        self.request = request
        return self.response


def test_translate_messages_moves_system_and_merges_tool_results() -> None:
    system, messages = translate_messages(
        [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Where is A-100?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_1", "function": {"name": "lookup_order", "arguments": '{"order_id": "A-100"}'}}
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "shipped"},
            {"role": "user", "content": "Thanks"},
        ]
    )
    assert system == "Be brief."
    assert [message["role"] for message in messages] == ["user", "assistant", "user"]
    assert messages[1]["content"][0] == {
        "type": "tool_use",
        "id": "call_1",
        "name": "lookup_order",
        "input": {"order_id": "A-100"},
    }
    assert [block["type"] for block in messages[2]["content"]] == ["tool_result", "text"]


def test_translate_tool_choice() -> None:
    assert translate_tool_choice("auto") is None
    assert translate_tool_choice("required") == {"type": "any"}
    assert translate_tool_choice({"type": "function", "function": {"name": "x"}}) == {"type": "tool", "name": "x"}


def test_anthropic_request_and_response_mapping() -> None:
    adapter = AnthropicMessages(SecretStr("test"))
    response = SimpleNamespace(
        content=[SimpleNamespace(type="tool_use", name="lookup_order", input={"order_id": "A-100"})],
        usage=SimpleNamespace(input_tokens=50, output_tokens=7),
        stop_reason="tool_use",
    )
    recorder = Recorder(response)
    adapter._client = SimpleNamespace(messages=recorder)  # type: ignore[assignment]
    completion = asyncio.run(adapter.complete(tool_case(), "claude-x", 256))
    assert recorder.request["tools"] == [
        {"name": "lookup_order", "input_schema": LOOKUP_TOOL["function"]["parameters"]}
    ]
    assert recorder.request["max_tokens"] == 256
    assert completion.tool_calls[0].arguments == {"order_id": "A-100"}
    assert (completion.input_tokens, completion.outcome) == (50, Outcome.COMPLETE)

    asyncio.run(adapter.complete(schema_case(), "claude-x", 256))
    assert recorder.request["output_config"] == {"format": {"type": "json_schema", "schema": SCHEMA}}

    asyncio.run(adapter.complete(schema_case().model_copy(update={"temperature": 0}), "claude-x", 256))
    assert recorder.request["extra_body"] == {"temperature": 0}
    assert "temperature" not in recorder.request


def test_openai_replays_request_and_detects_truncation() -> None:
    adapter = OpenAIChat(SecretStr("test"))
    message = SimpleNamespace(content='{"intent": "ref', refusal=None, tool_calls=None)
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="length")],
        usage=SimpleNamespace(prompt_tokens=40, completion_tokens=5),
    )
    recorder = Recorder(response)
    adapter._client = SimpleNamespace(chat=SimpleNamespace(completions=recorder))  # type: ignore[assignment]
    completion = asyncio.run(adapter.complete(schema_case(), "gpt-x", 128))
    assert recorder.request["response_format"]["type"] == "json_schema"
    assert recorder.request["max_completion_tokens"] == 128
    assert completion.outcome is Outcome.TRUNCATED
