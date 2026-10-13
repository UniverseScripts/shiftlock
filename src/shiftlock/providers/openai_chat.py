import time
from typing import Any

import openai
from pydantic import SecretStr

from shiftlock.domain import Completion, Outcome, TestCase, ToolCall
from shiftlock.importers.openai_jsonl import parse_arguments
from shiftlock.providers.base import provider_error


class OpenAIChat:
    name = "openai"

    def __init__(self, api_key: SecretStr, base_url: str | None = None):
        self._client = openai.AsyncOpenAI(
            api_key=api_key.get_secret_value(), base_url=base_url, max_retries=0, timeout=120
        )

    async def complete(self, case: TestCase, model: str, max_tokens: int) -> Completion:
        request: dict[str, Any] = {
            "model": model,
            "messages": case.messages,
            "max_completion_tokens": case.max_tokens or max_tokens,
        }
        if case.tools:
            request["tools"] = case.tools
        if case.tool_choice is not None:
            request["tool_choice"] = case.tool_choice
        if case.response_format is not None:
            request["response_format"] = case.response_format
        if case.temperature is not None:
            request["temperature"] = case.temperature

        started = time.perf_counter()
        try:
            response = await self._client.chat.completions.create(**request)
        except openai.APIStatusError as error:
            raise provider_error(error.message, error.status_code, error.response.headers.get("retry-after")) from None
        except openai.APIConnectionError as error:
            raise provider_error(str(error), None, None) from None
        latency_ms = round((time.perf_counter() - started) * 1000)

        choice = response.choices[0]
        message = choice.message
        tool_calls = [
            ToolCall(
                name=call.function.name,
                arguments=parse_arguments(call.function.arguments),
                raw_arguments=call.function.arguments,
            )
            for call in message.tool_calls or []
            if call.type == "function"
        ]
        if choice.finish_reason == "length":
            outcome = Outcome.TRUNCATED
        elif message.refusal:
            outcome = Outcome.REFUSED
        else:
            outcome = Outcome.COMPLETE
        return Completion(
            api="openai.chat.completions",
            text=message.content,
            tool_calls=tool_calls,
            input_tokens=response.usage.prompt_tokens if response.usage else None,
            output_tokens=response.usage.completion_tokens if response.usage else None,
            latency_ms=latency_ms,
            outcome=outcome,
        )
