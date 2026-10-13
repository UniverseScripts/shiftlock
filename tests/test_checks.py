from shiftlock.checks import evaluate
from shiftlock.config import CheckConfig
from shiftlock.domain import CaseResult, Completion, Role
from tests.conftest import completion, schema_case, tool_call, tool_case


def result(completion_value: Completion, role: Role = Role.CANDIDATE, cost: float | None = None) -> CaseResult:
    return CaseResult(
        case_id="c", sample=0, role=role, provider="fake", model="m", completion=completion_value, cost_usd=cost
    )


def outcomes(checks: list[tuple[str, bool, bool, str]]) -> dict[str, bool]:
    return {name: passed for name, _, passed, _ in checks}


def test_schema_checks() -> None:
    case = schema_case()
    assert outcomes(evaluate(case, result(completion('{"intent": "refund"}')), None, [])) == {
        "json_valid": True,
        "schema_match": True,
    }
    assert outcomes(evaluate(case, result(completion('{"intent": "angry"}')), None, [])) == {
        "json_valid": True,
        "schema_match": False,
    }
    assert outcomes(evaluate(case, result(completion("Sure! refund")), None, [])) == {
        "json_valid": False,
        "schema_match": False,
    }


def test_tool_calls_validated_against_tool_schema() -> None:
    case = tool_case()
    good = result(completion(tool_calls=[tool_call("lookup_order", order_id="A-100")]))
    missing_argument = result(completion(tool_calls=[tool_call("lookup_order")]))
    unknown_tool = result(completion(tool_calls=[tool_call("delete_order", order_id="A-100")]))
    assert outcomes(evaluate(case, good, None, []))["tool_calls_valid"]
    assert not outcomes(evaluate(case, missing_argument, None, []))["tool_calls_valid"]
    assert not outcomes(evaluate(case, unknown_tool, None, []))["tool_calls_valid"]


def test_required_tool_call_must_happen() -> None:
    case = tool_case().model_copy(update={"tool_choice": "required"})
    assert not outcomes(evaluate(case, result(completion("I can't look that up")), None, []))["tool_calls_valid"]


def test_expected_tool_call_match() -> None:
    case = tool_case(expected=[tool_call("lookup_order", order_id="A-100")])
    assert outcomes(
        evaluate(case, result(completion(tool_calls=[tool_call("lookup_order", order_id="A-100")])), None, [])
    )["tool_call_match"]
    assert not outcomes(
        evaluate(case, result(completion(tool_calls=[tool_call("lookup_order", order_id="B-200")])), None, [])
    )["tool_call_match"]


def test_candidate_compared_to_baseline_tools_is_advisory() -> None:
    case = tool_case()
    baseline = result(completion(tool_calls=[tool_call("lookup_order", order_id="A")]), role=Role.BASELINE)
    candidate = result(completion("no tool"))
    checks = evaluate(case, candidate, baseline, [])
    advisory = [check for check in checks if check[0] == "same_tools_as_baseline"]
    assert advisory == [
        ("same_tools_as_baseline", False, False, "baseline called ['lookup_order'], candidate called []")
    ]


def test_configured_checks() -> None:
    case = schema_case()
    checks = [
        CheckConfig(kind="contains", value="refund"),
        CheckConfig(kind="not_contains", value="sorry"),
        CheckConfig(kind="regex", value=r'"intent":\s*"\w+"'),
        CheckConfig(kind="max_latency_ms", value=400),
        CheckConfig(kind="max_cost_usd", value=0.01),
    ]
    found = outcomes(evaluate(case, result(completion('{"intent": "refund"}'), cost=0.02), None, checks))
    assert found["contains:refund"] and found["not_contains:sorry"] and found['regex:"intent":\\s*"\\w+"']
    assert not found["max_latency_ms:400"]
    assert not found["max_cost_usd:0.01"]
