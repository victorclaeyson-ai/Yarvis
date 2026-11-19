"""Which way to a bigger model? A one-off measurement for ADR 0019 (backlog task 10).

Why it exists: Haiku 5.5 answers fast, a bigger model answers better. Three ways of
reaching the bigger model are compared on the same prompts:
- A, Anthropic's advisor tool: Haiku consults the bigger model inside one request.
- B, a re-run: Haiku hands the question back and code asks the bigger model.
- C, the user's choice: "think hard" sends the question to the bigger model at once.

For each it measures the time to the first word, the time to the full answer and the cost,
and keeps the answers to the bench's own prompts so a person can judge their quality.

What it must never do: print, log or store the API key (it only passes it to the client);
store anything but its own prompts, answers and numbers, in the folder it is given; or
spend more than its budget, counted with the prices in ``usage.py``.
"""

from __future__ import annotations

import json
import logging
import statistics
import time
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Final, Literal

import anthropic

from yarvis.agent.prompts import system_prompt
from yarvis.llm.anthropic import DEFAULT_MODEL
from yarvis.paths import ensure_private_dir, write_private_file
from yarvis.usage import ClaudeUsage

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from anthropic.types.beta import (
        BetaMessage,
        BetaMessageParam,
        BetaTextBlockParam,
        BetaToolUnionParam,
        BetaUsage,
    )
    from pydantic import SecretStr

logger = logging.getLogger(__name__)

SONNET: Final = "claude-sonnet-5-5"
OPUS: Final = "claude-opus-5-5"
OPUS_READABLE: Final = "claude-opus-4-8"
"""An advisor whose advice comes back as text (5.x advisors return it encrypted)."""
ADVISOR_BETA: Final = "advisor-tool-2026-03-01"
VOICE_MAX_TOKENS: Final = 1024
BIG_MAX_TOKENS: Final = 4096
"""Bigger models think before they answer; 1,024 tokens could end an answer early."""
ADVISOR_MAX_TOKENS: Final = 2048
RESERVE_USD: Final = 0.15
"""What one more run could cost at most: no run starts unless it fits in the budget."""
THINK_HARDER: Final = "think_harder"

HARD_PROMPTS: Final = (
    "I have five exams in seventeen days: calculus, physics, programming, chemistry and "
    "linear algebra. How should I split my study time?",
    "If I put two hundred euros a month into savings at four percent a year, how much do I "
    "have after ten years?",
    "Should I take the train or drive from Leuven to Amsterdam on a Friday afternoon? "
    "Think about cost and time.",
    "Explain why the sky is blue but sunsets are red, so a ten-year-old understands.",
    "A bat and a ball cost one euro ten together, and the bat costs one euro more than the "
    "ball. How much is the ball?",
    "What are the pros and cons of buying versus renting an apartment in Leuven as a student?",
    "How many weekdays are there from the third of March to the fourteenth of April 2027, "
    "both days included?",
    "My laptop gets slow after an hour of use. Walk me through finding out why.",
)
EASY_PROMPTS: Final = (
    "What's the capital of Portugal?",
    "How do you say good morning in Dutch?",
    "How many minutes are in three hours?",
    "Who wrote Hamlet?",
)

_ADVISOR_NOTE: Final = (
    "\n\nFor a question that needs careful reasoning (calculations, planning, comparisons, "
    'explanations in several steps), first say one short sentence such as "Let me think '
    'about that.", then consult the advisor, then answer. Answer simple questions directly.'
)
_RERUN_NOTE: Final = (
    "\n\nFor a question that needs careful reasoning (calculations, planning, comparisons, "
    f"explanations in several steps), call {THINK_HARDER} and say nothing else. Answer "
    "simple questions directly."
)
_THINK_HARDER_TOOL: Final[BetaToolUnionParam] = {
    "name": THINK_HARDER,
    "description": "Hand a question that needs careful reasoning to a bigger, slower model.",
    "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
}

Kind = Literal["haiku", "advisor", "rerun", "direct"]


@dataclass(frozen=True, slots=True)
class Approach:
    """One way of answering, as measured."""

    key: str
    label: str
    kind: Kind
    big: str | None = None
    """The bigger model (None: Haiku alone)."""


APPROACHES: Final = (
    Approach("haiku", "Haiku 5.5 alone", "haiku"),
    Approach("advisor-opus-5-5", "A: Haiku + advisor Opus 5.5", "advisor", OPUS),
    Approach("advisor-opus-4-8", "A: Haiku + advisor Opus 4.8", "advisor", OPUS_READABLE),
    Approach("rerun-sonnet", "B: re-run on Sonnet 5.5", "rerun", SONNET),
    Approach("rerun-opus", "B: re-run on Opus 5.5", "rerun", OPUS),
    Approach("direct-sonnet", "C: think hard -> Sonnet 5.5", "direct", SONNET),
    Approach("direct-opus", "C: think hard -> Opus 5.5", "direct", OPUS),
)


@dataclass(frozen=True, slots=True)
class Run:
    """One prompt answered one way."""

    approach: str
    prompt: str
    hard: bool
    first_word_ms: float | None
    """Until something could be spoken: the first text, or (B) the moment code would say
    "Let me think about that." None: nothing was said."""
    total_ms: float
    answer: str
    escalated: bool
    """The bigger model was involved (always for C, by Haiku's choice for A and B)."""
    cost_usd: float
    error: str | None = None
    usages: tuple[ClaudeUsage, ...] = field(default=())


REQUEST_TIMEOUT_S: Final = 180.0
"""Bigger models think before answering; the voice client's 30 seconds is too short here."""


def client_for(key: SecretStr) -> anthropic.AsyncAnthropic:
    """A client for the measurement. The key goes to the client only, never to output."""
    return anthropic.AsyncAnthropic(
        api_key=key.get_secret_value(), max_retries=1, timeout=REQUEST_TIMEOUT_S
    )


class Budget:
    """Money the measurement may still spend, counted with ``usage.py`` prices."""

    def __init__(self, limit_usd: float) -> None:
        self.limit_usd = limit_usd
        self.spent_usd = 0.0

    def has_room(self) -> bool:
        """Whether one more run fits, with ``RESERVE_USD`` to spare."""
        return self.spent_usd + RESERVE_USD <= self.limit_usd

    def add(self, cost_usd: float) -> None:
        """Count what a run cost."""
        self.spent_usd += cost_usd


@dataclass(slots=True)
class _Streamed:
    text: str = ""
    first_text_s: float | None = None
    tool_start_s: float | None = None
    tool_names: list[str] = field(default_factory=list[str])
    usages: list[ClaudeUsage] = field(default_factory=list[ClaudeUsage])


def _big(approach: Approach) -> str:
    """The bigger model of an approach that has one."""
    if approach.big is None:
        msg = f"{approach.key} has no bigger model"
        raise ValueError(msg)
    return approach.big


def _usages(message: BetaMessage) -> list[ClaudeUsage]:
    """The executor's tokens and, separately, each advisor's (billed at its own model)."""
    usage: BetaUsage = message.usage
    found = [
        ClaudeUsage(
            model=message.model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_write_tokens=usage.cache_creation_input_tokens or 0,
            cache_read_tokens=usage.cache_read_input_tokens or 0,
        )
    ]
    for iteration in usage.iterations or []:
        if iteration.type == "advisor_message":
            found.append(
                ClaudeUsage(
                    model=iteration.model,
                    input_tokens=iteration.input_tokens,
                    output_tokens=iteration.output_tokens,
                    cache_write_tokens=iteration.cache_creation_input_tokens,
                    cache_read_tokens=iteration.cache_read_input_tokens,
                )
            )
    return found


class ModelBench:
    """Runs the prompts through every approach, within a budget."""

    def __init__(
        self,
        client: anthropic.AsyncAnthropic,
        *,
        budget: Budget,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._client = client
        self._budget = budget
        self._clock = clock
        base = system_prompt("Yarvis")
        self._system = base.stable + base.volatile

    async def run(
        self,
        approaches: Sequence[Approach] = APPROACHES,
        *,
        on_progress: Callable[[str], None] | None = None,
    ) -> list[Run]:
        """Every prompt, every approach (C only for hard prompts), in turn, until done or broke."""
        runs: list[Run] = []
        prompts = [(p, True) for p in HARD_PROMPTS] + [(p, False) for p in EASY_PROMPTS]
        for prompt, hard in prompts:
            for approach in approaches:
                if approach.kind == "direct" and not hard:
                    continue  # "think hard" is the user's choice: not for easy questions
                if not self._budget.has_room():
                    if on_progress is not None:
                        on_progress(f"Budget reached (${self._budget.spent_usd:.2f}): stopped.")
                    return runs
                run = await self._one(approach, prompt, hard=hard)
                self._budget.add(run.cost_usd)
                runs.append(run)
                if on_progress is not None:
                    on_progress(
                        f"{approach.label}: {run.total_ms / 1000:.1f} s, ${run.cost_usd:.4f} "
                        f"(spent ${self._budget.spent_usd:.2f})"
                    )
        return runs

    async def _one(self, approach: Approach, prompt: str, *, hard: bool) -> Run:
        start = self._clock()
        try:
            return await self._answer(approach, prompt, hard=hard, start=start)
        except anthropic.APIError as exc:
            logger.warning("%s failed: %s", approach.key, type(exc).__name__)
            return Run(
                approach.key,
                prompt,
                hard,
                None,
                (self._clock() - start) * 1000,
                "",
                escalated=False,
                cost_usd=0.0,
                error=type(exc).__name__,
            )

    async def _answer(self, approach: Approach, prompt: str, *, hard: bool, start: float) -> Run:
        if approach.kind == "haiku":
            first = await self._stream(DEFAULT_MODEL, self._system, prompt, VOICE_MAX_TOKENS)
            return self._run(
                approach, prompt, first, first, hard=hard, start=start, escalated=False
            )
        if approach.kind == "direct":
            big = await self._stream(_big(approach), self._system, prompt, BIG_MAX_TOKENS)
            return self._run(approach, prompt, big, big, hard=hard, start=start, escalated=True)
        if approach.kind == "advisor":
            advisor: BetaToolUnionParam = {
                "type": "advisor_20260301",
                "name": "advisor",
                "model": _big(approach),
                "max_uses": 1,
                "max_tokens": ADVISOR_MAX_TOKENS,
            }
            first = await self._stream(
                DEFAULT_MODEL,
                self._system + _ADVISOR_NOTE,
                prompt,
                BIG_MAX_TOKENS,
                tools=[advisor],
                betas=[ADVISOR_BETA],
            )
            consulted = len(first.usages) > 1  # an advisor iteration was billed
            return self._run(
                approach, prompt, first, first, hard=hard, start=start, escalated=consulted
            )
        first = await self._stream(
            DEFAULT_MODEL,
            self._system + _RERUN_NOTE,
            prompt,
            VOICE_MAX_TOKENS,
            tools=[_THINK_HARDER_TOOL],
        )
        if THINK_HARDER not in first.tool_names:
            return self._run(
                approach, prompt, first, first, hard=hard, start=start, escalated=False
            )
        big = await self._stream(_big(approach), self._system, prompt, BIG_MAX_TOKENS)
        big.usages = [*first.usages, *big.usages]
        # The user hears "Let me think about that." the moment Haiku asks for help.
        spoken = first.first_text_s or first.tool_start_s
        heard = _Streamed(first_text_s=spoken)
        return self._run(approach, prompt, heard, big, hard=hard, start=start, escalated=True)

    def _run(
        self,
        approach: Approach,
        prompt: str,
        first: _Streamed,
        answer: _Streamed,
        *,
        hard: bool,
        start: float,
        escalated: bool,
    ) -> Run:
        cost = sum(usage.token_cost() or 0.0 for usage in answer.usages)
        return Run(
            approach.key,
            prompt,
            hard,
            None if first.first_text_s is None else (first.first_text_s - start) * 1000,
            (self._clock() - start) * 1000,
            answer.text.strip(),
            escalated=escalated,
            cost_usd=cost,
            usages=tuple(answer.usages),
        )

    async def _stream(
        self,
        model: str,
        system: str,
        prompt: str,
        max_tokens: int,
        *,
        tools: Sequence[BetaToolUnionParam] = (),
        betas: Sequence[str] = (),
    ) -> _Streamed:
        streamed = _Streamed()
        system_blocks: list[BetaTextBlockParam] = [{"type": "text", "text": system}]
        messages: list[BetaMessageParam] = [{"role": "user", "content": prompt}]
        async with self._client.beta.messages.stream(
            model=model,
            max_tokens=max_tokens,
            system=system_blocks,
            messages=messages,
            tools=list(tools) or anthropic.omit,
            betas=list(betas) or anthropic.omit,
        ) as stream:
            async for event in stream:
                if event.type == "text":
                    if streamed.first_text_s is None and event.text.strip():
                        streamed.first_text_s = self._clock()
                    streamed.text += event.text
                elif (
                    event.type == "content_block_start"
                    and event.content_block.type == "tool_use"
                    and streamed.tool_start_s is None
                ):
                    streamed.tool_start_s = self._clock()
            final = await stream.get_final_message()
        streamed.tool_names = [b.name for b in final.content if b.type == "tool_use"]
        streamed.usages = _usages(final)
        return streamed


# --------------------------------------------------------------------------- the report


@dataclass(frozen=True, slots=True)
class Summary:
    """One approach, over the prompts it answered."""

    label: str
    first_word_hard_ms: float | None
    total_hard_ms: float | None
    cost_usd: float
    escalated_hard: str
    escalated_easy: str


def summarise(runs: Sequence[Run], approaches: Sequence[Approach] = APPROACHES) -> list[Summary]:
    """Medians on the hard prompts, total cost, and how often the bigger model was used."""
    summaries: list[Summary] = []
    for approach in approaches:
        mine = [run for run in runs if run.approach == approach.key and run.error is None]
        hard = [run for run in mine if run.hard]
        easy = [run for run in mine if not run.hard]
        firsts = [run.first_word_ms for run in hard if run.first_word_ms is not None]
        summaries.append(
            Summary(
                approach.label,
                statistics.median(firsts) if firsts else None,
                statistics.median(run.total_ms for run in hard) if hard else None,
                sum(run.cost_usd for run in mine),
                f"{sum(run.escalated for run in hard)}/{len(hard)}",
                f"{sum(run.escalated for run in easy)}/{len(easy)}" if easy else "-",
            )
        )
    return summaries


def _ms(value: float | None) -> str:
    return "-" if value is None else f"{value / 1000:.1f} s"


def report(
    runs: Sequence[Run], spent_usd: float, approaches: Sequence[Approach] = APPROACHES
) -> str:
    """Markdown: the summary table, then each hard prompt's answers side by side."""
    lines = [
        "# Model switching: measurement",
        "",
        f"Spent: ${spent_usd:.2f}. Times are from this computer; first word = when something",
        'could be spoken (for B: the moment code says "Let me think about that.").',
        "",
        "| Approach | First word (hard, median) | Full answer (hard, median) | Cost | "
        "Bigger model used: hard | easy |",
        "|---|---|---|---|---|---|",
    ]
    for s in summarise(runs, approaches):
        lines.append(
            f"| {s.label} | {_ms(s.first_word_hard_ms)} | {_ms(s.total_hard_ms)} | "
            f"${s.cost_usd:.3f} | {s.escalated_hard} | {s.escalated_easy} |"
        )
    labels = {a.key: a.label for a in approaches}
    for number, prompt in enumerate(HARD_PROMPTS, start=1):
        lines += ["", f"## {number}. {prompt}", ""]
        for run in (r for r in runs if r.prompt == prompt):
            took = f"first word {_ms(run.first_word_ms)}, all {_ms(run.total_ms)}"
            used = "bigger model" if run.escalated else "Haiku only"
            lines.append(f"### {labels.get(run.approach, run.approach)} ({took}, {used})")
            lines.append("")
            body = run.answer or f"(no answer: {run.error})"
            lines += [f"> {line}" if line else ">" for line in body.splitlines()]
            lines.append("")
    return "\n".join(lines) + "\n"


def save(runs: Sequence[Run], spent_usd: float, folder: Path) -> tuple[Path, Path]:
    """``results.json`` (every run, every number) and ``answers.md``, both private."""
    ensure_private_dir(folder)
    numbers = folder / "results.json"
    answers = folder / "answers.md"
    data = {"spent_usd": spent_usd, "runs": [asdict(run) for run in runs]}
    write_private_file(numbers, json.dumps(data, indent=2, ensure_ascii=False), overwrite=True)
    write_private_file(answers, report(runs, spent_usd), overwrite=True)
    return numbers, answers
