from typing import Protocol

from shiftlock.domain import Completion, TestCase
from shiftlock.errors import ProviderError
from shiftlock.security import redact

RETRYABLE_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504, 529}


class Provider(Protocol):
    name: str

    async def complete(self, case: TestCase, model: str, max_tokens: int) -> Completion: ...


def provider_error(message: str, status_code: int | None, retry_after_header: str | None) -> ProviderError:
    retry_after = None
    if retry_after_header:
        try:
            retry_after = float(retry_after_header)
        except ValueError:
            retry_after = None
    retryable = status_code is None or status_code in RETRYABLE_STATUS_CODES
    label = f"HTTP {status_code}" if status_code else "connection error"
    return ProviderError(
        f"{label}: {redact(message)[:300]}",
        status_code=status_code,
        retryable=retryable,
        retry_after=retry_after,
    )
