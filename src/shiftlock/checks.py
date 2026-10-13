import json
import re
from typing import Any

from jsonschema import Draft202012Validator

from shiftlock.config import CheckConfig
from shiftlock.domain import CaseResult, CheckResult, Outcome, Role, TestCase

UNSCORED_OUTCOMES = {Outcome.ERROR, Outcome.TRUNCATED}

Check = tuple[str, bool, bool, str]


def evaluate_all(cases: list[TestCase], results: list[CaseResult], checks: list[CheckConfig]) -> list[CheckResult]:
    cases_by_id = {case.id: case for case in cases}
    baseline_results = {(result.case_id, result.sample): result for result in results if result.role is Role.BASELINE}
    scored: list[CheckResult] = []
    for result in results:
        if result.completion.outcome in UNSCORED_OUTCOMES:
            continue
        case = cases_by_id[result.case_id]
        baseline = baseline_results.get((result.case_id, result.sample))
        for name, blocking, passed, detail in evaluate(case, result, baseline, checks):
            scored.append(
                CheckResult(
                    case_id=result.case_id,
                    sample=result.sample,
                    role=result.role,
                    check=name,
                    blocking=blocking,
                    passed=passed,
                    detail=detail,
                )
            )
    return scored


def evaluate(case: TestCase, result: CaseResult, baseline: CaseResult | None, checks: list[CheckConfig]) -> list[Check]:
    completion = result.completion
    outcomes: list[Check] = []
    wants_json = case.output_schema is not None or (case.response_format or {}).get("type") == "json_object"
    parsed: Any = None
    if wants_json:
        parsed, error = parse_json(completion.text)
        outcomes.append(("json_valid", True, error is None, error or ""))
    if case.output_schema is not None:
        outcomes.append(("schema_match", True, *schema_match(parsed, case.output_schema)))
    if case.tools:
        outcomes.append(("tool_calls_valid", True, *tool_calls_valid(case, result)))
        if case.expected_tool_calls is None and baseline is not None and result.role is Role.CANDIDATE:
            outcomes.append(("same_tools_as_baseline", False, *same_tools(result, baseline)))
    if case.expected_tool_calls is not None:
        outcomes.append(("tool_call_match", True, *tool_call_match(case, result)))
    for check in checks:
        outcomes.append((check_name(check), check.blocking, *configured_check(check, result)))
    return outcomes


def parse_json(text: str | None) -> tuple[Any, str | None]:
    try:
        return json.loads(text or ""), None
    except json.JSONDecodeError as error:
        return None, f"not JSON: {error.msg} at character {error.pos}"


def schema_match(parsed: Any, schema: dict[str, Any]) -> tuple[bool, str]:
    if parsed is None:
        return False, "no JSON to validate"
    error = next(iter(Draft202012Validator(schema).iter_errors(parsed)), None)
    if error is None:
        return True, ""
    path = "/".join(str(part) for part in error.absolute_path) or "(root)"
    return False, f"{path}: {error.message[:200]}"


def tool_calls_valid(case: TestCase, result: CaseResult) -> tuple[bool, str]:
    schemas = {tool["function"]["name"]: tool["function"].get("parameters") for tool in case.tools}
    calls = result.completion.tool_calls
    must_call = case.tool_choice == "required" or isinstance(case.tool_choice, dict)
    if must_call and not calls:
        return False, "a tool call was required but none was made"
    for call in calls:
        if call.name not in schemas:
            return False, f"called unknown tool {call.name!r}"
        if call.arguments is None:
            return False, f"{call.name}: arguments are not a JSON object"
        schema = schemas[call.name]
        if schema:
            passed, detail = schema_match(call.arguments, schema)
            if not passed:
                return False, f"{call.name}: {detail}"
    return True, ""


def tool_call_match(case: TestCase, result: CaseResult) -> tuple[bool, str]:
    expected = [(call.name, call.arguments) for call in case.expected_tool_calls or []]
    actual = [(call.name, call.arguments) for call in result.completion.tool_calls]
    if [name for name, _ in expected] != [name for name, _ in actual]:
        return False, f"expected tools {[name for name, _ in expected]}, got {[name for name, _ in actual]}"
    for (name, expected_arguments), (_, actual_arguments) in zip(expected, actual, strict=True):
        if expected_arguments != actual_arguments:
            return False, f"{name}: arguments differ from the expected call"
    return True, ""


def same_tools(result: CaseResult, baseline: CaseResult) -> tuple[bool, str]:
    expected = [call.name for call in baseline.completion.tool_calls]
    actual = [call.name for call in result.completion.tool_calls]
    return expected == actual, "" if expected == actual else f"baseline called {expected}, candidate called {actual}"


def check_name(check: CheckConfig) -> str:
    value = f"{check.value:g}" if isinstance(check.value, float) else check.value
    return f"{check.kind}:{value}"


def configured_check(check: CheckConfig, result: CaseResult) -> tuple[bool, str]:
    text = result.completion.text or ""
    match check.kind:
        case "contains":
            passed = str(check.value) in text
            return passed, "" if passed else "text not found"
        case "not_contains":
            passed = str(check.value) not in text
            return passed, "" if passed else "forbidden text found"
        case "regex":
            passed = re.search(str(check.value), text) is not None
            return passed, "" if passed else "pattern did not match"
        case "max_latency_ms":
            passed = result.completion.latency_ms <= float(check.value)
            return passed, f"{result.completion.latency_ms} ms"
        case "max_cost_usd":
            if result.cost_usd is None:
                return True, "cost unknown"
            return result.cost_usd <= float(check.value), f"${result.cost_usd:.6f}"
    raise AssertionError(f"unhandled check kind {check.kind}")
