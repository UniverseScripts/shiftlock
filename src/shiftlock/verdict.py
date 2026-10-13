"""Verdict rules v1 (strict). Order matters: no-go beats inconclusive beats conditional beats go."""

from collections import defaultdict
from collections.abc import Callable
from statistics import median

from shiftlock.checks import UNSCORED_OUTCOMES
from shiftlock.config import Thresholds
from shiftlock.domain import CaseResult, CheckResult, Decision, Outcome, Role, TestCase, Verdict

REJECTED_REQUEST_CODES = {400, 422}
Measure = Callable[[list[CaseResult]], float | None]


def decide(
    cases: list[TestCase],
    results: list[CaseResult],
    check_results: list[CheckResult],
    thresholds: Thresholds,
    samples_per_case: int,
) -> Verdict:
    results_by_case: dict[str, list[CaseResult]] = defaultdict(list)
    for result in results:
        results_by_case[result.case_id].append(result)
    usable = {
        case.id
        for case in cases
        if len(results_by_case[case.id]) == 2 * samples_per_case
        and all(result.completion.outcome not in UNSCORED_OUTCOMES for result in results_by_case[case.id])
    }

    outcomes: dict[tuple[str, str, Role], list[bool]] = defaultdict(list)
    blocking: dict[str, bool] = {}
    for check in check_results:
        if check.case_id in usable:
            outcomes[(check.case_id, check.check, check.role)].append(check.passed)
            blocking[check.check] = check.blocking

    rejected = rejected_by_candidate(results_by_case)
    regressions = sorted(f"{case_id}: request_rejected" for case_id in rejected)
    advisory_regressions: list[str] = []
    for case_id, check_name, role in sorted(outcomes):
        if role is not Role.CANDIDATE:
            continue
        baseline_passed = all(outcomes.get((case_id, check_name, Role.BASELINE), [True]))
        if baseline_passed and not all(outcomes[(case_id, check_name, role)]):
            (regressions if blocking[check_name] else advisory_regressions).append(f"{case_id}: {check_name}")

    usable_results = [result for result in results if result.case_id in usable]
    error_rate = (
        sum(result.completion.outcome is Outcome.ERROR for result in results) / len(results) if results else 1.0
    )
    cost_change = relative_change(usable_results, total_cost)
    latency_change = relative_change(usable_results, median_latency)

    reasons: list[str] = []
    if rejected:
        example = next(
            result.completion.error
            for case_id in sorted(rejected)
            for result in results_by_case[case_id]
            if result.role is Role.CANDIDATE and result.completion.error
        )
        reasons.append(f"candidate rejected {len(rejected)} request(s) the baseline accepted, e.g. {example}")
    if len(regressions) > len(rejected):
        reasons.append(
            f"{len(regressions) - len(rejected)} blocking check(s) passed on the baseline but failed on the candidate"
        )
    if len(usable) < thresholds.min_cases:
        reasons.append(f"only {len(usable)} usable case(s); at least {thresholds.min_cases} are needed")
    if error_rate > thresholds.max_error_rate:
        example = next(result.completion.error for result in results if result.completion.error)
        reasons.append(f"{error_rate:.0%} of calls failed (limit {thresholds.max_error_rate:.0%}), e.g. {example}")
    if advisory_regressions:
        reasons.append(f"{len(advisory_regressions)} advisory check(s) regressed")
    if cost_change is not None and cost_change > thresholds.max_cost_increase:
        reasons.append(f"cost up {cost_change:.0%} (limit {thresholds.max_cost_increase:.0%})")
    if latency_change is not None and latency_change > thresholds.max_latency_increase:
        reasons.append(f"median latency up {latency_change:.0%} (limit {thresholds.max_latency_increase:.0%})")

    if regressions:
        decision = Decision.NO_GO
    elif len(usable) < thresholds.min_cases or error_rate > thresholds.max_error_rate:
        decision = Decision.INCONCLUSIVE
    elif reasons:
        decision = Decision.CONDITIONAL
    else:
        decision = Decision.GO
        reasons.append("no regressions, and cost and latency are within limits")

    return Verdict(
        decision=decision,
        reasons=reasons,
        usable_cases=len(usable),
        total_cases=len(cases),
        regressions=regressions,
        advisory_regressions=advisory_regressions,
        baseline_pass_rate=pass_rate(check_results, usable, Role.BASELINE),
        candidate_pass_rate=pass_rate(check_results, usable, Role.CANDIDATE),
        error_rate=error_rate,
        cost_change=cost_change,
        latency_change=latency_change,
    )


def rejected_by_candidate(results_by_case: dict[str, list[CaseResult]]) -> set[str]:
    """Cases the baseline answered but the candidate refused as an invalid request, e.g. a removed parameter."""
    rejected = set()
    for case_id, case_results in results_by_case.items():
        baseline_ok = all(
            result.completion.outcome is not Outcome.ERROR for result in case_results if result.role is Role.BASELINE
        )
        candidate_rejected = any(
            result.completion.status_code in REJECTED_REQUEST_CODES
            for result in case_results
            if result.role is Role.CANDIDATE
        )
        if baseline_ok and candidate_rejected:
            rejected.add(case_id)
    return rejected


def total_cost(results: list[CaseResult]) -> float | None:
    costs = [result.cost_usd for result in results]
    if not costs or any(cost is None for cost in costs):
        return None
    return sum(cost for cost in costs if cost is not None)


def median_latency(results: list[CaseResult]) -> float | None:
    return median(result.completion.latency_ms for result in results) if results else None


def relative_change(results: list[CaseResult], measure: Measure) -> float | None:
    baseline = measure([result for result in results if result.role is Role.BASELINE])
    candidate = measure([result for result in results if result.role is Role.CANDIDATE])
    if baseline is None or candidate is None or baseline == 0:
        return None
    return candidate / baseline - 1


def pass_rate(check_results: list[CheckResult], usable: set[str], role: Role) -> float | None:
    relevant = [check.passed for check in check_results if check.role is role and check.case_id in usable]
    return sum(relevant) / len(relevant) if relevant else None
