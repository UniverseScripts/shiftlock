# shiftlock

Find out whether your own prompts, tool calls and JSON outputs still work before you switch LLM
models: a forced deprecation, a version bump or a provider change.

shiftlock replays your real requests on the current model (baseline) and the replacement
(candidate), runs deterministic checks, and returns one of four verdicts: **go**, **conditional**,
**no-go** or **inconclusive**. It also writes an HTML report showing every case that got worse.

> Status: early alpha (v0.1). It supports OpenAI and Anthropic, and imports OpenAI-style JSONL.

## Quick start

```bash
uv tool install .                # from a clone; not on PyPI yet
shiftlock init                   # creates shiftlock.yaml and .shiftlock/
shiftlock import requests.jsonl  # your own requests; nothing leaves your machine
# edit shiftlock.yaml: set the baseline and candidate model IDs
shiftlock estimate               # calls, tokens and worst-case cost, no API calls
export ANTHROPIC_API_KEY=...     # and/or OPENAI_API_KEY
export ANTHROPIC_WORKSPACE_ID=...  # only if your key isn't scoped to a workspace
shiftlock run                    # asks before spending, then prints the verdict
shiftlock report                 # show the latest verdict again
```

`examples/sample.jsonl` holds four synthetic cases you can try it on.

## Input

The input is a JSONL file in one of two OpenAI chat-completions shapes:

- **Batch API:** `{"custom_id": ..., "body": {"messages": [...], "tools": [...], "response_format": {...}}}`.
  Settings such as `temperature`, `tool_choice` and `max_completion_tokens` are replayed as
  given.
- **Fine-tuning:** `{"messages": [...], "tools": [...]}`. A final assistant message becomes the
  expected answer, and its tool calls are compared exactly.

Anthropic runs use the same file: shiftlock translates messages, tools and JSON schemas for you.

## Checks

| Check | When it runs | Effect |
|---|---|---|
| `json_valid`, `schema_match` | The request had a JSON schema or JSON mode | Blocking |
| `tool_calls_valid` | The request offered tools; arguments must match the tool's schema | Blocking |
| `tool_call_match` | The case has an expected tool call | Blocking |
| `same_tools_as_baseline` | The request offered tools, with no expected call | Advisory |
| `contains`, `not_contains`, `regex`, `max_latency_ms`, `max_cost_usd` | You add them in `shiftlock.yaml` | Advisory unless `blocking: true` |

## Verdict rules (v1, strict)

1. **no-go:** any blocking check that passed on the baseline fails on the candidate. One case is
   enough.
2. **inconclusive:**
   - fewer than `min_cases` usable cases (default 20);
   - or more than `max_error_rate` of calls failed (default 10%).
   - Truncated and errored outputs don't count as usable.
3. **conditional:** an advisory check regressed, or cost or median latency rose by more than 20%.
4. **go:** none of the above.

Exit codes:
- `0` go
- `1` no-go
- `2` conditional
- `3` inconclusive
- `4` cost cap reached
- `10` and above: errors

`--fail-on` chooses which verdicts fail CI.

## Cost

Customers bring their own keys, so every run spends credit on your provider account.
- `shiftlock estimate` gives a worst-case figure without calling any API.
- `max_cost_usd` stops the run before a call could exceed the cap.
- Prices come from `src/shiftlock/data/prices.yaml`, which is hand-maintained. Each entry needs
  a source URL and the date it was checked.
- If a model isn't listed, cost shows as "unknown" rather than being guessed. Add prices under
  `prices:` in `shiftlock.yaml`.

## Privacy

There is no telemetry. Keys are read only from environment variables and are redacted from all
output. See [SECURITY.md](SECURITY.md).

## Development

```bash
uv sync
uv run ruff check . && uv run ruff format --check .
uv run mypy src
uv run pytest            # offline; `-m live` runs the opt-in tests that spend API credit
```

## License

MIT
