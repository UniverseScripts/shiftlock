"""Opt-in smoke test against a real provider. Spends API credit.

Run with: SHIFTLOCK_LIVE_PROVIDER=anthropic SHIFTLOCK_LIVE_BASELINE=<model> SHIFTLOCK_LIVE_CANDIDATE=<model>
          uv run --env-file .env pytest -m live
"""

import os
import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from shiftlock import cli
from shiftlock.domain import Decision
from shiftlock.report import load_decision

pytestmark = pytest.mark.live
EXAMPLE = Path(__file__).parent.parent / "examples" / "sample.jsonl"


def test_sample_suite_against_real_models(workspace: Path) -> None:
    provider = os.environ.get("SHIFTLOCK_LIVE_PROVIDER", "anthropic")
    baseline = os.environ.get("SHIFTLOCK_LIVE_BASELINE")
    candidate = os.environ.get("SHIFTLOCK_LIVE_CANDIDATE")
    if not baseline or not candidate:
        pytest.skip("set SHIFTLOCK_LIVE_BASELINE and SHIFTLOCK_LIVE_CANDIDATE")
    shutil.copy(EXAMPLE, "sample.jsonl")
    Path("shiftlock.yaml").write_text(
        "version: 1\n"
        "suite: .shiftlock/suites/default.jsonl\n"
        f"baseline: {{provider: {provider}, model: {baseline}}}\n"
        f"candidate: {{provider: {provider}, model: {candidate}}}\n"
        "max_tokens: 300\n"
        "thresholds: {min_cases: 4}\n",
        encoding="utf-8",
    )
    runner = CliRunner()
    assert runner.invoke(cli.app, ["import", "sample.jsonl"]).exit_code == 0
    result = runner.invoke(cli.app, ["run", "--yes", "--fail-on", ""])
    assert result.exit_code == 0, result.output
    _, decision = load_decision(None)
    assert decision["verdict"]["decision"] in {item.value for item in Decision}
    assert decision["verdict"]["error_rate"] == 0
