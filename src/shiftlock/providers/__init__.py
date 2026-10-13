import os

from shiftlock.providers.base import Provider


def create_provider(name: str) -> Provider:
    from shiftlock.security import read_api_key

    if name == "openai":
        from shiftlock.providers.openai_chat import OpenAIChat

        return OpenAIChat(read_api_key("openai"))
    if name == "anthropic":
        from shiftlock.providers.anthropic_messages import AnthropicMessages

        return AnthropicMessages(read_api_key("anthropic"), os.environ.get("ANTHROPIC_WORKSPACE_ID") or None)
    raise ValueError(f"unknown provider {name!r}")
