import asyncio
import secrets
import traceback
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer

from shiftlock.checks import evaluate_all
from shiftlock.config import CONFIG_FILE, TEMPLATE, Config, load_config
from shiftlock.domain import Decision, Role, TestCase
from shiftlock.errors import ConfigError, ExitCode, ShiftlockError, SuiteImportError
from shiftlock.estimate import estimate_run
from shiftlock.importers.openai_jsonl import import_openai_jsonl
from shiftlock.pricing import PriceTable, load_prices
from shiftlock.providers import create_provider
from shiftlock.providers.base import Provider
from shiftlock.report import load_decision, new_run_id, write_run
from shiftlock.runner import run_suite
from shiftlock.security import configure_logging, ensure_workspace, redact
from shiftlock.suite import load_suite, save_suite
from shiftlock.verdict import decide

app = typer.Typer(
    help="Check whether your own prompts still work before you switch LLM models.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)
debug_enabled = False

DECISION_EXIT_CODES = {
    Decision.GO: ExitCode.GO,
    Decision.NO_GO: ExitCode.NO_GO,
    Decision.CONDITIONAL: ExitCode.CONDITIONAL,
    Decision.INCONCLUSIVE: ExitCode.INCONCLUSIVE,
}
ConfigOption = Annotated[Path, typer.Option("--config", "-c", help="Path to shiftlock.yaml.")]


class PersonalData(StrEnum):
    ASK = "ask"
    KEEP = "keep"
    DROP = "drop"


@app.callback()
def options(
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show progress logs.")] = False,
    debug: Annotated[bool, typer.Option("--debug", help="Show debug logs and redacted tracebacks.")] = False,
) -> None:
    global debug_enabled
    debug_enabled = debug
    configure_logging(verbose, debug)


@app.command()
def init() -> None:
    """Create shiftlock.yaml and the private .shiftlock workspace."""
    ensure_workspace()
    if CONFIG_FILE.exists():
        typer.echo(f"{CONFIG_FILE} already exists; left unchanged.")
    else:
        CONFIG_FILE.write_text(TEMPLATE, encoding="utf-8")
        typer.echo(f"Created {CONFIG_FILE}. Set the baseline and candidate model IDs next.")


@app.command("import")
def import_suite(
    file: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="OpenAI chat-completions JSONL.")],
    name: Annotated[str, typer.Option(help="Suite name.")] = "default",
    personal_data: Annotated[
        PersonalData, typer.Option(help="What to do with cases that look like they contain personal data.")
    ] = PersonalData.ASK,
) -> None:
    """Import test cases from a Batch-API or fine-tuning JSONL file. Nothing leaves your machine."""
    result = import_openai_jsonl(file)
    for issue in result.issues:
        typer.echo(f"skipped {issue}", err=True)
    if not result.cases:
        raise SuiteImportError(f"no valid cases in {file}")
    cases = result.cases
    if result.flagged:
        keep = personal_data is PersonalData.KEEP or (
            personal_data is PersonalData.ASK
            and typer.confirm(
                f"{len(result.flagged)} case(s) look like they contain emails or phone numbers. Keep them?"
            )
        )
        if not keep:
            cases = [case for case in cases if not case.has_personal_data]
    ensure_workspace()
    path = save_suite(cases, name)
    typer.echo(f"Imported {len(cases)} case(s) to {path} ({len(result.issues)} skipped).")


@app.command()
def estimate(config_path: ConfigOption = CONFIG_FILE) -> None:
    """Estimate calls, tokens and worst-case cost without calling any API."""
    config = load_config(config_path)
    print_estimate(config, load_suite(config.suite), load_prices(config.prices))


def print_estimate(config: Config, cases: list[TestCase], prices: PriceTable) -> None:
    result = estimate_run(cases, config, prices)
    typer.echo(f"{len(cases)} case(s), {result.calls} call(s)")
    typer.echo(f"~{result.input_tokens:,} input tokens, up to {result.max_output_tokens:,} output tokens")
    if result.max_cost_usd is None:
        typer.echo(f"Worst-case cost: unknown (no price data for {', '.join(result.unpriced_models)})")
    else:
        typer.echo(f"Worst-case cost: ${result.max_cost_usd:.4f}")


@app.command()
def run(
    config_path: ConfigOption = CONFIG_FILE,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip the cost confirmation.")] = False,
    fail_on: Annotated[
        str, typer.Option(help="Comma-separated verdicts that return a non-zero exit code.")
    ] = "no-go,conditional,inconclusive",
) -> None:
    """Run the suite on both models, score the outputs and write a report."""
    config = load_config(config_path)
    cases = load_suite(config.suite)
    prices = load_prices(config.prices)
    failing = {Decision(value.strip()) for value in fail_on.split(",") if value.strip()}
    unpriced = estimate_run(cases, config, prices).unpriced_models
    if config.max_cost_usd is not None and unpriced:
        raise ConfigError(f"max_cost_usd needs prices for {', '.join(unpriced)}; add them under `prices:`")

    print_estimate(config, cases, prices)
    if not yes and not typer.confirm("Run now? This calls the provider APIs with your keys"):
        raise typer.Exit(ExitCode.GO)

    providers: dict[str, Provider] = {}
    for target in (config.baseline, config.candidate):
        providers.setdefault(target.provider, create_provider(target.provider))
    by_role = {Role.BASELINE: providers[config.baseline.provider], Role.CANDIDATE: providers[config.candidate.provider]}

    outcome = asyncio.run(run_suite(cases, config, by_role, prices))
    check_results = evaluate_all(cases, outcome.results, config.checks)
    verdict = decide(cases, outcome.results, check_results, config.thresholds, config.samples_per_case)
    directory = write_run(new_run_id(), config, outcome.results, check_results, verdict, outcome.cost_cap_reached)

    typer.echo(f"\nVerdict: {verdict.decision.value.upper()}")
    for reason in verdict.reasons:
        typer.echo(f"  - {reason}")
    if outcome.cost_cap_reached:
        typer.echo(f"  - stopped early: max_cost_usd ${config.max_cost_usd} reached")
    typer.echo(f"Report: {directory / 'report.html'}")

    if outcome.cost_cap_reached and verdict.decision is not Decision.NO_GO:
        raise typer.Exit(ExitCode.COST_CAP)
    raise typer.Exit(DECISION_EXIT_CODES[verdict.decision] if verdict.decision in failing else ExitCode.GO)


@app.command()
def report(run_id: Annotated[str | None, typer.Argument(help="Run ID; defaults to the latest run.")] = None) -> None:
    """Show the verdict of a previous run."""
    try:
        found_id, decision = load_decision(run_id)
    except (FileNotFoundError, OSError) as error:
        raise ConfigError(str(error)) from error
    verdict = decision["verdict"]
    typer.echo(f"Run {found_id}: {verdict['decision'].upper()}")
    for reason in verdict["reasons"]:
        typer.echo(f"  - {reason}")
    typer.echo(f"Report: .shiftlock/runs/{found_id}/report.html")


def main() -> None:
    try:
        app()
    except ShiftlockError as error:
        typer.echo(f"error: {redact(str(error))}", err=True)
        raise SystemExit(error.exit_code) from None
    except Exception as error:
        error_id = secrets.token_hex(4)
        typer.echo(f"unexpected error [{error_id}]: {type(error).__name__}: {redact(str(error))[:300]}", err=True)
        if debug_enabled:
            typer.echo(redact(traceback.format_exc()), err=True)
        raise SystemExit(ExitCode.UNEXPECTED_ERROR) from None
