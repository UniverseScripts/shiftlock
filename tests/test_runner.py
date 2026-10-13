import asyncio

import pytest

from shiftlock.domain import Completion, Outcome, Role, TestCase
from shiftlock.errors import ProviderError
from shiftlock.pricing import Price
from shiftlock.runner import RunOutcome, run_suite
from tests.conftest import FakeProvider, completion, make_config, schema_case

PRICES = {
    "openai/base-model": Price(input_per_million=1, output_per_million=1, source="test", as_of="2026-10-13"),
    "openai/new-model": Price(input_per_million=1, output_per_million=1, source="test", as_of="2026-10-13"),
}


async def no_sleep(_: float) -> None:
    return None


def run(provider: FakeProvider, cases: list[TestCase], **config_overrides: object) -> RunOutcome:
    config = make_config(**config_overrides)
    return asyncio.run(run_suite(cases, config, {Role.BASELINE: provider, Role.CANDIDATE: provider}, PRICES, no_sleep))


def test_runs_every_case_sample_and_role() -> None:
    provider = FakeProvider(lambda case, model: completion('{"intent": "refund"}'))
    outcome = run(provider, [schema_case("a"), schema_case("b")], samples_per_case=2)
    assert len(outcome.results) == 8
    assert sorted(set(provider.calls)) == [
        ("a", "base-model"),
        ("a", "new-model"),
        ("b", "base-model"),
        ("b", "new-model"),
    ]


def test_retries_transient_errors_then_records_attempts() -> None:
    failures = iter([ProviderError("HTTP 429", status_code=429, retryable=True, retry_after=1)])

    def responder(case: TestCase, model: str) -> Completion | ProviderError:
        if model == "new-model":
            failure = next(failures, None)
            if failure:
                return failure
        return completion('{"intent": "refund"}')

    outcome = run(FakeProvider(responder), [schema_case()])
    candidate = next(result for result in outcome.results if result.role is Role.CANDIDATE)
    assert candidate.completion.attempts == 2
    assert candidate.completion.outcome is Outcome.COMPLETE


def test_permanent_error_is_recorded_not_raised() -> None:
    provider = FakeProvider(lambda case, model: ProviderError("HTTP 400: bad", status_code=400, retryable=False))
    outcome = run(provider, [schema_case()])
    assert all(result.completion.outcome is Outcome.ERROR for result in outcome.results)


def test_auth_failure_stops_the_run() -> None:
    provider = FakeProvider(lambda case, model: ProviderError("HTTP 401", status_code=401, retryable=False))
    with pytest.raises(ProviderError):
        run(provider, [schema_case()])


def test_cost_cap_stops_before_overspending() -> None:
    provider = FakeProvider(lambda case, model: completion('{"intent": "refund"}', input_tokens=100, output_tokens=400))
    cases = [schema_case(f"c{index}") for index in range(10)]
    outcome = run(provider, cases, max_cost_usd=0.003, max_tokens=500, concurrency=1)
    assert outcome.cost_cap_reached
    assert 0 < len(provider.calls) < 20
    assert outcome.spent_usd <= 0.003
