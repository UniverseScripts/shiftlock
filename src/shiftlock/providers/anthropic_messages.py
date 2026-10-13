"""Anthropic Messages adapter. Test cases are stored in OpenAI chat shape and translated here."""

import json
import time
from typing import Any

import anthropic
from pydantic import SecretStr

from shiftlock.domain import Completion, Outcome, TestCase, ToolCall
from shiftlock.providers.base import provider_error


class AnthropicMessages:
    name = "anthropic"

    def __init__(self, api_key: SecretStr, workspace_id: str | None = None):
        self._client = anthropic.AsyncAnthropic(
            api_key=api_key.get_secret_value(),
            default_headers={"anthropic-workspace-id": workspace_id} if workspace_id else None,
            max_retries=0,
            timeout=120,
        )

    async def complete(self, case: TestCase, model: str, max_tokens: int) -> Completion:
        system, messages = translate_messages(case.messages)
        request: dict[str, Any] = {"model": model, "messages": messages, "max_tokens": case.max_tokens or max_tokens}
        if system:
            request["system"] = system
        if case.tools:
            request["tools"] = [translate_tool(tool) for tool in case.tools]
            tool_choice = translate_tool_choice(case.tool_choice)
            if tool_choice:
                request["tool_choice"] = tool_choice
        if case.output_schema is not None:
            request["output_config"] = {"format": {"type": "json_schema", "schema": case.output_schema}}
        if case.temperature is not None:
            # The SDK no longer exposes temperature; send it anyway so the API decides, as in production.
            request["extra_body"] = {"temperature": case.temperature}

        started = time.perf_counter()
        try:
            response = await self._client.messages.create(**request)
        except anthropic.APIStatusError as error:
            raise provider_error(error.message, error.status_code, error.response.headers.get("retry-after")) from None
        except anthropic.APIConnectionError as error:
            raise provider_error(str(error), None, None) from None
        latency_ms = round((time.perf_counter() - started) * 1000)

        texts = [block.text for block in response.content if block.type == "text"]
        tool_calls = [
            ToolCall(name=block.name, arguments=dict(block.input), raw_arguments=json.dumps(block.input))
            for block in response.content
            if block.type == "tool_use"
        ]
        outcomes = {"max_tokens": Outcome.TRUNCATED, "refusal": Outcome.REFUSED}
        return Completion(
            api="anthropic.messages",
            text="".join(texts) if texts else None,
            tool_calls=tool_calls,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_ms=latency_ms,
            outcome=outcomes.get(response.stop_reason or "", Outcome.COMPLETE),
        )


def translate_messages(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    system_parts: list[str] = []
    translated: list[dict[str, Any]] = []
    for message in messages:
        role = message["role"]
        if role in ("system", "developer"):
            system_parts.append(content_text(message.get("content")))
            continue
        if role == "tool":
            block = {
                "type": "tool_result",
                "tool_use_id": message.get("tool_call_id", ""),
                "content": content_text(message.get("content")),
            }
            append_blocks(translated, "user", [block])
            continue
        blocks: list[dict[str, Any]] = []
        text = content_text(message.get("content"))
        if text:
            blocks.append({"type": "text", "text": text})
        for call in message.get("tool_calls") or []:
            function = call.get("function", {})
            arguments = function.get("arguments") or "{}"
            blocks.append(
                {
                    "type": "tool_use",
                    "id": call.get("id", ""),
                    "name": function.get("name", ""),
                    "input": json.loads(arguments) if isinstance(arguments, str) else arguments,
                }
            )
        append_blocks(translated, role, blocks)
    return "\n\n".join(part for part in system_parts if part), translated


def append_blocks(messages: list[dict[str, Any]], role: str, blocks: list[dict[str, Any]]) -> None:
    if messages and messages[-1]["role"] == role:
        messages[-1]["content"].extend(blocks)
    else:
        messages.append({"role": role, "content": blocks})


def content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return ""


def translate_tool(tool: dict[str, Any]) -> dict[str, Any]:
    function = tool["function"]
    translated = {
        "name": function["name"],
        "input_schema": function.get("parameters") or {"type": "object", "properties": {}},
    }
    if function.get("description"):
        translated["description"] = function["description"]
    return translated


def translate_tool_choice(tool_choice: str | dict[str, Any] | None) -> dict[str, Any] | None:
    if tool_choice in (None, "auto"):
        return None
    if tool_choice == "required":
        return {"type": "any"}
    if tool_choice == "none":
        return {"type": "none"}
    if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        return {"type": "tool", "name": tool_choice["function"]["name"]}
    return None
