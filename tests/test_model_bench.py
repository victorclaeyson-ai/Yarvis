"""The model-switching measurement (ADR 0019), on fake API responses: no network, no cost.

``Any`` here is for JSON request and event payloads, which have no schema of their own.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import anthropic
import httpx2
import pytest

from fakes import posix_mode
from test_anthropic import sse
from yarvis import model_bench
from yarvis.llm.anthropic import DEFAULT_MODEL
from yarvis.model_bench import (
    ADVISOR_BETA,
    APPROACHES,
    BIG_MAX_TOKENS,
    EASY_PROMPTS,
    HARD_PROMPTS,
    OPUS,
    SONNET,
    THINK_HARDER,
    Budget,
    ModelBench,
    report,
    save,
    summarise,
)
from yarvis.security.secrets import KEYRING_SERVICE

if TYPE_CHECKING:
    from pathlib import Path

    from fakes import FakeKeyring, Invoke

TEST_KEY = "sk-ant-api03-" + "k" * 40


def start(model: str, usage: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "message_start",
        "message": {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": usage,
        },
    }


def end(reason: str = "end_turn", output: int = 100) -> list[dict[str, Any]]:
    return [
        {
            "type": "message_delta",
            "delta": {"stop_reason": reason, "stop_sequence": None},
            "usage": {"output_tokens": output},
        },
        {"type": "message_stop"},
    ]


def answer(model: str, text: str, *, advisor: str | None = None) -> bytes:
    usage: dict[str, Any] = {"input_tokens": 2000, "output_tokens": 1}
    if advisor is not None:  # the advisor's own tokens, billed at its model
        usage["iterations"] = [
            {
                "type": "advisor_message",
                "model": advisor,
                "input_tokens": 3000,
                "output_tokens": 500,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            }
        ]
    return sse(
        start(model, usage),
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}},
        {"type": "content_block_stop", "index": 0},
        *end(),
    )


def hand_over() -> bytes:
    """Haiku calls think_harder and says nothing."""
    use: dict[str, object] = {
        "type": "tool_use",
        "id": "toolu_1",
        "name": THINK_HARDER,
        "input": {},
    }
    return sse(
        start(DEFAULT_MODEL, {"input_tokens": 2000, "output_tokens": 1}),
        {"type": "content_block_start", "index": 0, "content_block": use},
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": "{}"},
        },
        {"type": "content_block_stop", "index": 0},
        *end("tool_use", 10),
    )


class FakeApi:
    """Answers like the real models would, and keeps every request."""

    def __init__(self, *, fail_model: str | None = None) -> None:
        self.requests: list[tuple[dict[str, Any], dict[str, str]]] = []
        self.fail_model = fail_model

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        self.requests.append((body, dict(request.headers)))
        model = body["model"]
        if model == self.fail_model:
            error = {"type": "error", "error": {"type": "overloaded_error", "message": "busy"}}
            return httpx2.Response(529, json=error)
        prompt = body["messages"][0]["content"]
        hard = prompt in HARD_PROMPTS
        tools = {tool.get("name") for tool in body.get("tools", [])}
        if model == DEFAULT_MODEL and THINK_HARDER in tools and hard:
            content = hand_over()
        elif model == DEFAULT_MODEL and "advisor" in tools and hard:
            advisor = next(t["model"] for t in body["tools"] if t.get("name") == "advisor")
            content = answer(model, "Let me think about that. Here is the plan.", advisor=advisor)
        elif model == DEFAULT_MODEL:
            content = answer(model, "Lisbon.")
        else:
            content = answer(model, f"A careful answer from {model}.")
        return httpx2.Response(200, content=content, headers={"content-type": "text/event-stream"})


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        self.now += 0.1  # every look at the clock is a tenth of a second later
        return self.now


def client(api: FakeApi) -> anthropic.AsyncAnthropic:
    return anthropic.AsyncAnthropic(
        api_key=TEST_KEY,
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(api)),
    )


async def test_every_prompt_goes_through_every_approach() -> None:
    api = FakeApi()
    runs = await ModelBench(client(api), budget=Budget(10), clock=Clock()).run()
    by_kind = {a.key: a.kind for a in APPROACHES}
    direct = sum(kind == "direct" for kind in by_kind.values())
    assert len(runs) == len(APPROACHES) * len(HARD_PROMPTS) + (len(APPROACHES) - direct) * len(
        EASY_PROMPTS
    )  # "think hard" is never used for easy questions
    for run in runs:
        assert run.error is None
        kind = by_kind[run.approach]
        expected = run.hard and kind != "haiku"  # A and B escalate on hard prompts only
        assert run.escalated is expected, (run.approach, run.prompt)


async def test_a_re_run_speaks_first_and_answers_with_the_bigger_model() -> None:
    api = FakeApi()
    rerun = next(a for a in APPROACHES if a.key == "rerun-sonnet")
    runs = await ModelBench(client(api), budget=Budget(10), clock=Clock()).run([rerun])
    hard = next(run for run in runs if run.hard)
    assert hard.answer == f"A careful answer from {SONNET}."
    assert hard.first_word_ms is not None
    assert hard.first_word_ms < hard.total_ms  # "Let me think about that." comes first
    assert {usage.model for usage in hard.usages} == {DEFAULT_MODEL, SONNET}
    big = next(body for body, _ in api.requests if body["model"] == SONNET)
    assert big["max_tokens"] == BIG_MAX_TOKENS
    assert "tools" not in big
    easy = next(run for run in runs if not run.hard)
    assert easy.answer == "Lisbon."
    assert not easy.escalated


async def test_the_advisor_is_asked_once_and_billed_at_its_own_price() -> None:
    api = FakeApi()
    advisor = next(a for a in APPROACHES if a.key == "advisor-opus-5-5")
    runs = await ModelBench(client(api), budget=Budget(10), clock=Clock()).run([advisor])
    body, headers = next(
        (b, h) for b, h in api.requests if b["messages"][0]["content"] == HARD_PROMPTS[0]
    )
    tool = body["tools"][0]
    assert tool == {
        "type": "advisor_20260301",
        "name": "advisor",
        "model": OPUS,
        "max_uses": 1,
        "max_tokens": 2048,
    }
    assert ADVISOR_BETA in headers["anthropic-beta"]
    hard = next(run for run in runs if run.hard)
    assert [usage.model for usage in hard.usages] == [DEFAULT_MODEL, OPUS]
    haiku_only = 2000 * 0.10 / 1e6 + 100 * 0.50 / 1e6
    opus = 3000 * 4.0 / 1e6 + 500 * 20.0 / 1e6
    assert hard.cost_usd == pytest.approx(haiku_only + opus)


async def test_the_budget_stops_the_measurement() -> None:
    budget = Budget(0.3)
    runs = await ModelBench(client(FakeApi()), budget=budget, clock=Clock()).run()
    assert 0 < len(runs) < len(APPROACHES) * len(HARD_PROMPTS)
    assert budget.spent_usd <= 0.3
    assert not budget.has_room()


async def test_a_failed_request_is_recorded_and_the_rest_goes_on() -> None:
    api = FakeApi(fail_model=OPUS)
    runs = await ModelBench(client(api), budget=Budget(10), clock=Clock()).run()
    failed = [run for run in runs if run.error is not None]
    assert failed
    assert all(run.cost_usd == 0 for run in failed)
    assert any(run.approach == "direct-sonnet" and run.error is None for run in runs)


async def test_the_report_puts_the_answers_side_by_side(tmp_path: Path) -> None:
    budget = Budget(10)
    runs = await ModelBench(client(FakeApi()), budget=budget, clock=Clock()).run()
    text = report(runs, budget.spent_usd)
    assert "| Approach | First word" in text
    for prompt in HARD_PROMPTS:
        assert f". {prompt}" in text
    assert text.count("### C: think hard -> Opus 5.5") == len(HARD_PROMPTS)
    summary = {s.label: s for s in summarise(runs)}
    assert summary["B: re-run on Sonnet 5.5"].escalated_hard == "8/8"
    assert summary["B: re-run on Sonnet 5.5"].escalated_easy == "0/4"
    numbers, answers = save(runs, budget.spent_usd, tmp_path / "bench")
    assert posix_mode(numbers) in (None, 0o600)
    assert json.loads(numbers.read_text(encoding="utf-8"))["runs"]
    assert TEST_KEY not in numbers.read_text(encoding="utf-8") + answers.read_text(encoding="utf-8")


def test_yarvis_bench_switching_runs_within_its_budget(
    invoke: Invoke,
    fake_keyring: FakeKeyring,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_keyring.store[(KEYRING_SERVICE, "anthropic")] = TEST_KEY
    api = FakeApi()
    used: list[str] = []

    def fake_client(key: Any) -> anthropic.AsyncAnthropic:  # Any: a SecretStr from the keychain
        used.append(key.get_secret_value())
        return client(api)

    monkeypatch.setattr(model_bench, "client_for", fake_client)
    result = invoke("bench", "--switching", "--out", str(tmp_path / "out"), "--budget", "0.5")
    assert result.exit_code == 0, result.output
    assert used == [TEST_KEY]  # read from the keychain by Yarvis itself
    assert TEST_KEY not in result.output
    assert "Spent $" in result.output
    assert (tmp_path / "out" / "answers.md").exists()
