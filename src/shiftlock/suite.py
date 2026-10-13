from pathlib import Path

from pydantic import ValidationError

from shiftlock.domain import TestCase
from shiftlock.errors import ConfigError
from shiftlock.security import WORKSPACE, write_private_file

SUITES = WORKSPACE / "suites"


def save_suite(cases: list[TestCase], name: str) -> Path:
    path = SUITES / f"{name}.jsonl"
    write_private_file(path, "".join(case.model_dump_json() + "\n" for case in cases))
    return path


def load_suite(path: Path) -> list[TestCase]:
    if not path.exists():
        raise ConfigError(f"suite {path} not found. Run `shiftlock import` first.")
    try:
        return [
            TestCase.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
        ]
    except ValidationError as error:
        raise ConfigError(f"suite {path} is corrupt: {error.error_count()} invalid records") from error
