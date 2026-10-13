import json
from dataclasses import dataclass, field

from shiftlock.config import Config
from shiftlock.domain import TestCase
from shiftlock.pricing import PriceTable, cost_usd, price_key

CHARACTERS_PER_TOKEN = 4


@dataclass
class Estimate:
    calls: int = 0
    input_tokens: int = 0
    max_output_tokens: int = 0
    max_cost_usd: float | None = 0.0
    unpriced_models: list[str] = field(default_factory=list)


def estimate_input_tokens(case: TestCase) -> int:
    return len(json.dumps([case.messages, case.tools, case.output_schema])) // CHARACTERS_PER_TOKEN + 1


def worst_case_cost(case: TestCase, provider: str, model: str, config: Config, prices: PriceTable) -> float | None:
    return cost_usd(
        prices.get(price_key(provider, model)), estimate_input_tokens(case), case.max_tokens or config.max_tokens
    )


def estimate_run(cases: list[TestCase], config: Config, prices: PriceTable) -> Estimate:
    """Rough upper bound: input tokens from text length, output tokens at the max_tokens limit."""
    estimate = Estimate()
    for target in (config.baseline, config.candidate):
        if price_key(target.provider, target.model) not in prices:
            estimate.unpriced_models.append(price_key(target.provider, target.model))
        for case in cases:
            estimate.calls += config.samples_per_case
            estimate.input_tokens += estimate_input_tokens(case) * config.samples_per_case
            estimate.max_output_tokens += (case.max_tokens or config.max_tokens) * config.samples_per_case
            call_cost = worst_case_cost(case, target.provider, target.model, config, prices)
            if call_cost is None or estimate.max_cost_usd is None:
                estimate.max_cost_usd = None
            else:
                estimate.max_cost_usd += call_cost * config.samples_per_case
    return estimate
