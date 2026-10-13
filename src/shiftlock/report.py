import json
import secrets
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jinja2 import Environment, PackageLoader, select_autoescape

from shiftlock import __version__
from shiftlock.config import Config
from shiftlock.domain import CaseResult, CheckResult, Role, Verdict
from shiftlock.security import WORKSPACE, write_private_file

RUNS = WORKSPACE / "runs"


def new_run_id() -> str:
    return f"{datetime.now(UTC):%Y%m%d-%H%M%S}-{secrets.token_hex(3)}"


def write_run(
    run_id: str,
    config: Config,
    results: list[CaseResult],
    check_results: list[CheckResult],
    verdict: Verdict,
    cost_cap_reached: bool,
) -> Path:
    directory = RUNS / run_id
    decision = {
        "run_id": run_id,
        "shiftlock_version": __version__,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "baseline": config.baseline.model_dump(),
        "candidate": config.candidate.model_dump(),
        "samples_per_case": config.samples_per_case,
        "cost_cap_reached": cost_cap_reached,
        "verdict": verdict.model_dump(mode="json"),
    }
    write_private_file(directory / "raw.jsonl", "".join(result.model_dump_json() + "\n" for result in results))
    write_private_file(directory / "scores.jsonl", "".join(check.model_dump_json() + "\n" for check in check_results))
    write_private_file(directory / "decision.json", json.dumps(decision, indent=2) + "\n")
    write_private_file(directory / "report.html", render_html(decision, results, check_results, verdict))
    return directory


def render_html(
    decision: dict[str, Any], results: list[CaseResult], check_results: list[CheckResult], verdict: Verdict
) -> str:
    environment = Environment(
        loader=PackageLoader("shiftlock", "templates"), autoescape=select_autoescape(default=True)
    )
    outputs = {(result.case_id, result.sample, result.role): result.completion for result in results}
    failures = [
        {
            "case_id": check.case_id,
            "sample": check.sample,
            "check": check.check,
            "blocking": check.blocking,
            "detail": check.detail,
            "baseline": outputs.get((check.case_id, check.sample, Role.BASELINE)),
            "candidate": outputs.get((check.case_id, check.sample, Role.CANDIDATE)),
        }
        for check in check_results
        if check.role is Role.CANDIDATE and not check.passed
    ]
    return environment.get_template("report.html.j2").render(decision=decision, verdict=verdict, failures=failures)


def load_decision(run_id: str | None) -> tuple[str, dict[str, Any]]:
    runs = sorted(path for path in RUNS.glob("*") if (path / "decision.json").exists()) if RUNS.exists() else []
    if not runs:
        raise FileNotFoundError("no runs found. Run `shiftlock run` first.")
    directory = RUNS / run_id if run_id else runs[-1]
    return directory.name, json.loads((directory / "decision.json").read_text(encoding="utf-8"))
