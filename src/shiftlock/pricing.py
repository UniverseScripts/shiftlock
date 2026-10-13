from datetime import date
from importlib.resources import files
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

from shiftlock.errors import ConfigError


class Price(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_per_million: float
    output_per_million: float
    source: str
    as_of: date


PriceTable = dict[str, Price]


def price_key(provider: str, model: str) -> str:
    return f"{provider}/{model}"


def load_prices(override: Path | None = None) -> PriceTable:
    raw = yaml.safe_load(files("shiftlock").joinpath("data/prices.yaml").read_text(encoding="utf-8")) or {}
    if override:
        if not override.exists():
            raise ConfigError(f"price file {override} not found")
        raw.update(yaml.safe_load(override.read_text(encoding="utf-8")) or {})
    try:
        return {key: Price.model_validate(value) for key, value in raw.items()}
    except ValidationError as error:
        raise ConfigError(f"invalid price entry: {error}") from error


def cost_usd(price: Price | None, input_tokens: int | None, output_tokens: int | None) -> float | None:
    if price is None or input_tokens is None or output_tokens is None:
        return None
    return (input_tokens * price.input_per_million + output_tokens * price.output_per_million) / 1_000_000
