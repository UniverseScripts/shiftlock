from enum import StrEnum
from typing import Any

from pydantic import BaseModel

RULE_VERSION = "v1"


class Role(StrEnum):
    BASELINE = "baseline"
    CANDIDATE = "candidate"


class Outcome(StrEnum):
    COMPLETE = "complete"
    TRUNCATED = "truncated"
    REFUSED = "refused"
    ERROR = "error"


class Decision(StrEnum):
    GO = "go"
    CONDITIONAL = "conditional"
    NO_GO = "no-go"
    INCONCLUSIVE = "inconclusive"


class ToolCall(BaseModel):
    name: str
    arguments: dict[str, Any] | None
    raw_arguments: str


class TestCase(BaseModel):
    __test__ = False

    id: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] = []
    tool_choice: str | dict[str, Any] | None = None
    response_format: dict[str, Any] | None = None
    output_schema: dict[str, Any] | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    expected_text: str | None = None
    expected_tool_calls: list[ToolCall] | None = None
    has_personal_data: bool = False


class Completion(BaseModel):
    api: str
    text: str | None = None
    tool_calls: list[ToolCall] = []
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: int
    outcome: Outcome
    error: str | None = None
    status_code: int | None = None
    attempts: int = 1


class CaseResult(BaseModel):
    case_id: str
    sample: int
    role: Role
    provider: str
    model: str
    completion: Completion
    cost_usd: float | None = None


class CheckResult(BaseModel):
    case_id: str
    sample: int
    role: Role
    check: str
    blocking: bool
    passed: bool
    detail: str = ""


class Verdict(BaseModel):
    decision: Decision
    reasons: list[str]
    usable_cases: int
    total_cases: int
    regressions: list[str]
    advisory_regressions: list[str]
    baseline_pass_rate: float | None
    candidate_pass_rate: float | None
    error_rate: float
    cost_change: float | None
    latency_change: float | None
    rule_version: str = RULE_VERSION
