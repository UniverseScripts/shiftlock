import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from shiftlock.config import Config
from shiftlock.domain import CaseResult, Completion, Outcome, Role, TestCase
from shiftlock.errors import ProviderError
from shiftlock.estimate import worst_case_cost
from shiftlock.pricing import PriceTable, cost_usd, price_key
from shiftlock.providers.base import Provider

MAX_ATTEMPTS = 4
MAX_BACKOFF_SECONDS = 30.0

logger = logging.getLogger(__name__)
Sleep = Callable[[float], Awaitable[None]]


@dataclass
class RunOutcome:
    results: list[CaseResult]
    cost_cap_reached: bool
    spent_usd: float


class Budget:
    """Reserves the worst-case cost before each call so concurrent calls cannot overshoot the cap."""

    def __init__(self, limit: float | None):
        self.limit = limit
        self.committed = 0.0
        self.reached = False

    def reserve(self, amount: float) -> bool:
        if self.limit is not None and self.committed + amount > self.limit:
            self.reached = True
            return False
        self.committed += amount
        return True

    def settle(self, reserved: float, actual: float | None) -> None:
        self.committed += (reserved if actual is None else actual) - reserved


async def run_suite(
    cases: list[TestCase],
    config: Config,
    providers: dict[Role, Provider],
    prices: PriceTable,
    sleep: Sleep = asyncio.sleep,
) -> RunOutcome:
    semaphore = asyncio.Semaphore(config.concurrency)
    budget = Budget(config.max_cost_usd)
    results: list[CaseResult] = []

    async def run_one(case: TestCase, sample: int, role: Role) -> None:
        target = config.baseline if role is Role.BASELINE else config.candidate
        async with semaphore:
            reserved = worst_case_cost(case, target.provider, target.model, config, prices) or 0.0
            if not budget.reserve(reserved):
                return
            completion = await complete_with_retries(providers[role], case, target.model, config.max_tokens, sleep)
            cost = cost_usd(
                prices.get(price_key(target.provider, target.model)),
                completion.input_tokens,
                completion.output_tokens,
            )
            budget.settle(reserved, cost)
        results.append(
            CaseResult(
                case_id=case.id,
                sample=sample,
                role=role,
                provider=target.provider,
                model=target.model,
                completion=completion,
                cost_usd=cost,
            )
        )
        logger.info("case %s sample %d %s: %s", case.id, sample, role.value, completion.outcome.value)

    try:
        async with asyncio.TaskGroup() as group:
            for case in cases:
                for sample in range(config.samples_per_case):
                    for role in Role:
                        group.create_task(run_one(case, sample, role))
    except ExceptionGroup as group_error:
        raise group_error.exceptions[0] from None

    results.sort(key=lambda result: (result.case_id, result.sample, result.role.value))
    return RunOutcome(results=results, cost_cap_reached=budget.reached, spent_usd=budget.committed)


async def complete_with_retries(
    provider: Provider, case: TestCase, model: str, max_tokens: int, sleep: Sleep
) -> Completion:
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            completion = await provider.complete(case, model, max_tokens)
            return completion.model_copy(update={"attempts": attempt})
        except ProviderError as error:
            if error.is_auth_failure:
                raise
            if not error.retryable or attempt == MAX_ATTEMPTS:
                return Completion(
                    api=provider.name,
                    latency_ms=0,
                    outcome=Outcome.ERROR,
                    error=str(error),
                    status_code=error.status_code,
                    attempts=attempt,
                )
            await sleep(min(error.retry_after or 2.0**attempt, MAX_BACKOFF_SECONDS))
    raise AssertionError("unreachable")
