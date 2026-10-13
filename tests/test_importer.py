import json
from pathlib import Path

from shiftlock.importers.openai_jsonl import import_openai_jsonl
from tests.conftest import LOOKUP_TOOL, SCHEMA


def write_lines(path: Path, records: list[object]) -> Path:
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
    return path


def test_batch_shape_keeps_request_settings(tmp_path: Path) -> None:
    record = {
        "custom_id": "ticket-1",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "any",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [LOOKUP_TOOL],
            "tool_choice": "required",
            "response_format": {"type": "json_schema", "json_schema": {"name": "x", "schema": SCHEMA}},
            "temperature": 0,
            "max_completion_tokens": 300,
        },
    }
    result = import_openai_jsonl(write_lines(tmp_path / "batch.jsonl", [record]))
    case = result.cases[0]
    assert result.issues == []
    assert case.id == "ticket-1"
    assert case.output_schema == SCHEMA
    assert case.tool_choice == "required"
    assert (case.temperature, case.max_tokens) == (0, 300)
    assert case.expected_tool_calls is None


def test_fine_tune_shape_uses_final_assistant_message_as_expectation(tmp_path: Path) -> None:
    record = {
        "messages": [
            {"role": "user", "content": "Where is order A-100?"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "1",
                        "type": "function",
                        "function": {"name": "lookup_order", "arguments": '{"order_id": "A-100"}'},
                    }
                ],
            },
        ],
        "tools": [LOOKUP_TOOL],
    }
    case = import_openai_jsonl(write_lines(tmp_path / "tune.jsonl", [record])).cases[0]
    assert [message["role"] for message in case.messages] == ["user"]
    assert case.expected_tool_calls is not None
    assert case.expected_tool_calls[0].arguments == {"order_id": "A-100"}


def test_bad_lines_are_reported_with_line_numbers(tmp_path: Path) -> None:
    path = tmp_path / "mixed.jsonl"
    path.write_text(
        '{"messages": [{"role": "user", "content": "ok"}]}\n'
        "not json\n"
        '{"messages": []}\n'
        '{"messages": [{"role": "robot", "content": "x"}]}\n',
        encoding="utf-8",
    )
    result = import_openai_jsonl(path)
    assert len(result.cases) == 1
    assert [issue.split(":")[0] for issue in result.issues] == ["line 2", "line 3", "line 4"]


def test_invalid_schema_and_duplicate_ids_are_rejected(tmp_path: Path) -> None:
    bad_schema = {
        "messages": [{"role": "user", "content": "x"}],
        "response_format": {"type": "json_schema", "json_schema": {"schema": {"type": 5}}},
    }
    duplicate = {"custom_id": "a", "body": {"messages": [{"role": "user", "content": "x"}]}}
    result = import_openai_jsonl(write_lines(tmp_path / "x.jsonl", [bad_schema, duplicate, duplicate]))
    assert len(result.cases) == 1
    assert "invalid JSON schema" in result.issues[0]
    assert "duplicate id" in result.issues[1]


def test_personal_data_is_flagged(tmp_path: Path) -> None:
    records = [
        {"messages": [{"role": "user", "content": "email me at jane.doe@example.com"}]},
        {"messages": [{"role": "user", "content": "call +1 (555) 010-9999"}]},
        {"messages": [{"role": "user", "content": "order 12345 please"}]},
    ]
    result = import_openai_jsonl(write_lines(tmp_path / "pii.jsonl", records))
    assert [case.has_personal_data for case in result.cases] == [True, True, False]
