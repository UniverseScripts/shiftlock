import json
import os
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from shiftlock import cli
from shiftlock.domain import Completion, TestCase
from shiftlock.errors import ExitCode, ProviderError
from shiftlock.providers.base import provider_error
from tests.conftest import FakeProvider, completion, tool_call

EXAMPLE = Path(__file__).parent.parent / "examples" / "sample.jsonl"
FAKE_KEY = "sk-test-DO-NOT-LEAK-1234567890abcdef"
runner = CliRunner()


def answer(case: TestCase, model: str) -> Completion | ProviderError:
    if case.tools:
        order_id = "B-7731" if case.id.startswith("line") else "A-1042"
        return completion(tool_calls=[tool_call("lookup_order", order_id=order_id)])
    intent = "refund" if "money back" in json.dumps(case.messages) else "shipping"
    if model == "new-model" and intent == "shipping":
        return completion(f"<script>alert(1)</script> It is {intent}")
    return completion(json.dumps({"intent": intent}))


def prepare(workspace: Path, monkeypatch: pytest.MonkeyPatch, responder: object = answer) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_KEY)
    monkeypatch.setattr(cli, "create_provider", lambda name: FakeProvider(responder))  # type: ignore[arg-type]
    assert runner.invoke(cli.app, ["init"]).exit_code == 0
    config = Path("shiftlock.yaml").read_text(encoding="utf-8")
    config = config.replace("provider: anthropic", "provider: openai")
    config = config.replace("replace-with-current-model-id", "base-model").replace(
        "replace-with-replacement-model-id", "new-model"
    )
    config = config.replace("max_cost_usd: 5\n", "").replace("min_cases: 20", "min_cases: 2")
    Path("shiftlock.yaml").write_text(config, encoding="utf-8")
    result = runner.invoke(cli.app, ["import", str(EXAMPLE)])
    assert result.exit_code == 0, result.output


def test_end_to_end_no_go_with_escaped_report(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    prepare(workspace, monkeypatch)
    result = runner.invoke(cli.app, ["run", "--yes"])
    assert result.exit_code == ExitCode.NO_GO, result.output
    assert "NO-GO" in result.output
    run_directory = next(Path(".shiftlock/runs").iterdir())
    decision = json.loads((run_directory / "decision.json").read_text(encoding="utf-8"))
    assert decision["verdict"]["regressions"] == ["intent-shipping: json_valid", "intent-shipping: schema_match"]
    html = (run_directory / "report.html").read_text(encoding="utf-8")
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html
    assert (Path(".shiftlock") / ".gitignore").read_text() == "*\n"


def test_fail_on_controls_exit_code(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    prepare(workspace, monkeypatch)
    assert runner.invoke(cli.app, ["run", "--yes", "--fail-on", "inconclusive"]).exit_code == ExitCode.GO


def test_estimate_and_report_commands(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    prepare(workspace, monkeypatch)
    estimate = runner.invoke(cli.app, ["estimate"])
    assert "4 case(s), 8 call(s)" in estimate.output
    assert "unknown" in estimate.output
    runner.invoke(cli.app, ["run", "--yes"])
    assert "NO-GO" in runner.invoke(cli.app, ["report"]).output


def test_cost_cap_requires_prices(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    prepare(workspace, monkeypatch)
    Path("shiftlock.yaml").write_text(Path("shiftlock.yaml").read_text() + "max_cost_usd: 1\n", encoding="utf-8")
    result = runner.invoke(cli.app, ["run", "--yes"])
    assert isinstance(result.exception, cli.ConfigError)


def test_api_key_never_leaks(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def leaky(case: TestCase, model: str) -> Completion | ProviderError:
        return provider_error(f"Incorrect API key provided: {FAKE_KEY}", 400, None)

    prepare(workspace, monkeypatch, leaky)
    result = runner.invoke(cli.app, ["--debug", "run", "--yes"])
    stored = "".join(path.read_text(encoding="utf-8") for path in Path(".shiftlock").rglob("*") if path.is_file())
    assert "[REDACTED]" in stored
    assert FAKE_KEY not in result.output + stored


def test_unexpected_error_is_redacted(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    prepare(workspace, monkeypatch)

    async def explode(*_: object, **__: object) -> None:
        raise RuntimeError(f"boom Authorization: Bearer {FAKE_KEY}")

    monkeypatch.setattr(cli, "run_suite", explode)
    monkeypatch.setattr(sys, "argv", ["shiftlock", "--debug", "run", "--yes"])
    with pytest.raises(SystemExit) as exit_info:
        cli.main()
    captured = capsys.readouterr()
    assert exit_info.value.code == ExitCode.UNEXPECTED_ERROR
    assert "unexpected error [" in captured.err
    assert FAKE_KEY not in captured.out + captured.err


def test_missing_key_is_a_clear_error(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from shiftlock.errors import MissingKeyError
    from shiftlock.security import read_api_key

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(MissingKeyError, match="ANTHROPIC_API_KEY is not set"):
        read_api_key("anthropic")


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits are not enforced on Windows")
def test_outputs_are_private(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    prepare(workspace, monkeypatch)
    runner.invoke(cli.app, ["run", "--yes"])
    for path in Path(".shiftlock").rglob("*"):
        assert path.stat().st_mode & 0o077 == 0, path
