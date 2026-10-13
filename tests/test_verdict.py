from shiftlock.checks import evaluate_all
from shiftlock.config import Thresholds
from shiftlock.domain import CaseResult, Completion, Decision, Outcome, Role, TestCase, Verdict
from shiftlock.verdict import decide
from tests.conftest import completion, schema_case

GOOD = '{"intent": "refund"}'
BAD = "refund please"


def verdict_for(
    pairs: list[tuple[Completion, Completion]],
    thresholds: Thresholds | None = None,
    costs: tuple[float | None, float | None] = (0.01, 0.01),
) -> Verdict:
    cases: list[TestCase] = []
    results: list[CaseResult] = []
    for index, (baseline, candidate) in enumerate(pairs):
        case = schema_case(f"c{index}")
        cases.append(case)
        for role, value, cost in ((Role.BASELINE, baseline, costs[0]), (Role.CANDIDATE, candidate, costs[1])):
            results.append(
                CaseResult(
                    case_id=case.id, sample=0, role=role, provider="fake", model="m", completion=value, cost_usd=cost
                )
            )
    checks = evaluate_all(cases, results, [])
    return decide(cases, results, checks, thresholds or Thresholds(min_cases=2), samples_per_case=1)


def test_go_when_nothing_regresses() -> None:
    verdict = verdict_for([(completion(GOOD), completion(GOOD))] * 2)
    assert verdict.decision is Decision.GO
    assert verdict.regressions == []


def test_single_blocking_regression_is_no_go_even_with_few_cases() -> None:
    verdict = verdict_for([(completion(GOOD), completion(BAD))], thresholds=Thresholds(min_cases=20))
    assert verdict.decision is Decision.NO_GO
    assert verdict.regressions == ["c0: json_valid", "c0: schema_match"]


def test_failure_on_both_models_is_not_a_regression() -> None:
    verdict = verdict_for([(completion(BAD), completion(BAD)), (completion(GOOD), completion(GOOD))])
    assert verdict.decision is Decision.GO


def test_too_few_usable_cases_is_inconclusive() -> None:
    truncated = completion(GOOD, outcome=Outcome.TRUNCATED)
    verdict = verdict_for([(completion(GOOD), completion(GOOD)), (completion(GOOD), truncated)])
    assert verdict.decision is Decision.INCONCLUSIVE
    assert verdict.usable_cases == 1


def test_high_error_rate_is_inconclusive() -> None:
    failed = Completion(api="fake", latency_ms=0, outcome=Outcome.ERROR, error="HTTP 500")
    pairs = [(completion(GOOD), completion(GOOD))] * 4 + [(completion(GOOD), failed)]
    verdict = verdict_for(pairs, thresholds=Thresholds(min_cases=2, max_error_rate=0.05))
    assert verdict.decision is Decision.INCONCLUSIVE


def test_cost_increase_is_conditional() -> None:
    verdict = verdict_for([(completion(GOOD), completion(GOOD))] * 2, costs=(0.010, 0.013))
    assert verdict.decision is Decision.CONDITIONAL
    assert verdict.cost_change is not None and round(verdict.cost_change, 2) == 0.30


def test_latency_increase_is_conditional() -> None:
    verdict = verdict_for([(completion(GOOD, latency_ms=1000), completion(GOOD, latency_ms=1500))] * 2)
    assert verdict.decision is Decision.CONDITIONAL


def test_unknown_cost_is_reported_not_guessed() -> None:
    verdict = verdict_for([(completion(GOOD), completion(GOOD))] * 2, costs=(None, None))
    assert verdict.cost_change is None
    assert verdict.decision is Decision.GO


def test_candidate_rejecting_a_request_the_baseline_accepted_is_no_go() -> None:
    rejected = Completion(
        api="fake", latency_ms=0, outcome=Outcome.ERROR, status_code=400, error="HTTP 400: temperature is not supported"
    )
    verdict = verdict_for([(completion(GOOD), rejected), (completion(GOOD), completion(GOOD))])
    assert verdict.decision is Decision.NO_GO
    assert verdict.regressions == ["c0: request_rejected"]
    assert "temperature is not supported" in verdict.reasons[0]


def test_server_errors_are_not_treated_as_rejections() -> None:
    failed = Completion(api="fake", latency_ms=0, outcome=Outcome.ERROR, status_code=500, error="HTTP 500")
    verdict = verdict_for([(completion(GOOD), failed), (completion(GOOD), completion(GOOD))])
    assert verdict.decision is Decision.INCONCLUSIVE
