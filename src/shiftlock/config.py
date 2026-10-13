from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from shiftlock.errors import ConfigError

CONFIG_FILE = Path("shiftlock.yaml")

ProviderName = Literal["openai", "anthropic"]
CheckKind = Literal["contains", "not_contains", "regex", "max_latency_ms", "max_cost_usd"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelTarget(StrictModel):
    provider: ProviderName
    model: str = Field(min_length=1)


class CheckConfig(StrictModel):
    kind: CheckKind
    value: str | float
    blocking: bool = False


class Thresholds(StrictModel):
    min_cases: int = Field(default=20, ge=1)
    max_error_rate: float = Field(default=0.10, ge=0, le=1)
    max_cost_increase: float = Field(default=0.20, ge=0)
    max_latency_increase: float = Field(default=0.20, ge=0)


class Config(StrictModel):
    version: Literal[1]
    suite: Path
    baseline: ModelTarget
    candidate: ModelTarget
    checks: list[CheckConfig] = []
    thresholds: Thresholds = Thresholds()
    samples_per_case: int = Field(default=1, ge=1, le=10)
    concurrency: int = Field(default=4, ge=1, le=32)
    max_tokens: int = Field(default=1024, ge=1)
    max_cost_usd: float | None = Field(default=None, gt=0)
    prices: Path | None = None


TEMPLATE = """\
version: 1
suite: .shiftlock/suites/default.jsonl
baseline:
  provider: anthropic
  model: replace-with-current-model-id
candidate:
  provider: anthropic
  model: replace-with-replacement-model-id
samples_per_case: 1
concurrency: 4
max_tokens: 1024
max_cost_usd: 5
thresholds:
  min_cases: 20
  max_error_rate: 0.10
  max_cost_increase: 0.20
  max_latency_increase: 0.20
checks: []
"""


def load_config(path: Path = CONFIG_FILE) -> Config:
    if not path.exists():
        raise ConfigError(f"{path} not found. Run `shiftlock init` first.")
    try:
        return Config.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    except yaml.YAMLError as error:
        raise ConfigError(f"{path} is not valid YAML: {error}") from error
    except ValidationError as error:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in issue['loc'])}: {issue['msg']}" for issue in error.errors()
        )
        raise ConfigError(f"{path} is invalid: {problems}") from error
