"""What the cloud services cost and how fast Yarvis answers, counted and kept per day.

Only numbers are kept: tokens, web searches, seconds of speech streamed to AssemblyAI, and
the time from the end of your words to Yarvis's first word. Never what was said.

Costs are estimates at the providers' list prices in US dollars (the currency both bill
in), as of ``PRICES_AS_OF``. Your invoice is what counts: a discount, a price change or a
model Yarvis has no price for makes it differ. A model without a known price is counted
(``Day.unpriced``) so the dashboard can say its total is too low.

The numbers live in ``usage.json`` in Yarvis's private data folder, one entry per day,
for ``KEEP_DAYS`` days. ``yarvis usage`` shows them and ``yarvis usage clear`` deletes them.
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import TYPE_CHECKING, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from yarvis.paths import write_private_file

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence
    from pathlib import Path

    from yarvis.paths import AppPaths

logger = logging.getLogger(__name__)

PRICES_AS_OF: Final = "19 November 2026"
USAGE_FILE: Final = "usage.json"
KEEP_DAYS: Final = 366
MAX_LATENCIES: Final = 500
"""Latencies kept per day (the most recent ones): plenty for a median."""
MAX_FILE_BYTES: Final = 4_000_000
CHARS_PER_TOKEN: Final = 4
"""For a reply cut off mid-stream, whose output tokens were never reported."""


def local_today() -> date:
    """Today's date where the user is (days are counted from midnight to midnight)."""
    return datetime.now().astimezone().date()


# --------------------------------------------------------------------------- prices


@dataclass(frozen=True, slots=True)
class TokenPrices:
    """US dollars per million tokens."""

    input: float
    output: float
    cache_write: float
    cache_read: float


@dataclass(frozen=True, slots=True)
class ModelPrices:
    """A model's prices; some models cost more for very long prompts."""

    base: TokenPrices
    long: TokenPrices | None = None
    long_above: int = 100_000
    """Prompts with more input tokens than this are billed at ``long`` prices."""

    def of(self, input_tokens: int) -> TokenPrices:
        """The prices that apply to a prompt of ``input_tokens`` tokens."""
        if self.long is not None and input_tokens > self.long_above:
            return self.long
        return self.base


_OPUS_45: Final = ModelPrices(TokenPrices(5.0, 25.0, 6.25, 0.50))
_SONNET_45: Final = ModelPrices(TokenPrices(3.0, 15.0, 3.75, 0.30))

CLAUDE_PRICES: Final[Mapping[str, ModelPrices]] = {
    "claude-haiku-5-5": ModelPrices(
        TokenPrices(0.10, 0.50, 0.125, 0.01), long=TokenPrices(0.50, 2.50, 0.625, 0.05)
    ),
    "claude-sonnet-5-5": ModelPrices(TokenPrices(2.0, 10.0, 2.50, 0.10)),
    "claude-sonnet-5": ModelPrices(TokenPrices(2.0, 10.0, 2.50, 0.20)),
    "claude-opus-5-5": ModelPrices(TokenPrices(4.0, 20.0, 5.0, 0.20)),
    "claude-opus-5": ModelPrices(TokenPrices(5.0, 25.0, 6.25, 0.50)),
    "claude-haiku-4-5": ModelPrices(TokenPrices(1.0, 5.0, 1.25, 0.10)),
    "claude-sonnet-4-6": _SONNET_45,
    "claude-sonnet-4-5": _SONNET_45,
    "claude-opus-4-8": _OPUS_45,
    "claude-opus-4-7": _OPUS_45,
    "claude-opus-4-6": _OPUS_45,
    "claude-opus-4-5": _OPUS_45,
}
"""Anthropic's list prices (5-minute cache writes), by model id."""

WEB_SEARCH_USD: Final = 0.01
"""Claude's web search: $10 per 1,000 searches, on top of the tokens."""

SPEECH_USD_PER_HOUR: Final[Mapping[str, float]] = {"universal-3-6-pro": 0.45}
"""AssemblyAI streaming, billed per hour of session (the time the connection is open)."""

_DATED: Final = re.compile(r"-\d{8}")


def model_prices(model: str) -> ModelPrices | None:
    """The prices of ``model``, also when it carries a date ("claude-haiku-4-5-20251001")."""
    if (prices := CLAUDE_PRICES.get(model)) is not None:
        return prices
    for name, prices in CLAUDE_PRICES.items():
        rest = model.removeprefix(name)
        if rest != model and _DATED.fullmatch(rest):
            return prices
    return None


def speech_cost(model: str, seconds: float) -> float | None:
    """What ``seconds`` of an AssemblyAI session cost (None: no known price)."""
    per_hour = SPEECH_USD_PER_HOUR.get(model)
    return None if per_hour is None else per_hour * max(seconds, 0.0) / 3600


# --------------------------------------------------------------------------- one request


@dataclass(frozen=True, slots=True)
class ClaudeUsage:
    """What one request to Claude used (as reported by the API, or estimated if cut off)."""

    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    web_searches: int = 0

    @property
    def prompt_tokens(self) -> int:
        """Every token of the prompt, cached or not."""
        return self.input_tokens + self.cache_write_tokens + self.cache_read_tokens

    def token_cost(self) -> float | None:
        """The tokens' cost in US dollars (None: no known price for the model)."""
        prices = model_prices(self.model)
        if prices is None:
            return None
        rate = prices.of(self.prompt_tokens)
        total = (
            self.input_tokens * rate.input
            + self.output_tokens * rate.output
            + self.cache_write_tokens * rate.cache_write
            + self.cache_read_tokens * rate.cache_read
        )
        return total / 1_000_000

    def search_cost(self) -> float:
        """The web searches' cost in US dollars."""
        return self.web_searches * WEB_SEARCH_USD


def estimated_tokens(chars: int) -> int:
    """Output tokens of ``chars`` characters of reply (a rough estimate)."""
    return math.ceil(chars / CHARS_PER_TOKEN)


# --------------------------------------------------------------------------- one day


class Day(BaseModel):
    """One day's totals (also the format of ``usage.json``)."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    turns: int = Field(default=0, ge=0)
    requests: int = Field(default=0, ge=0)
    """Requests to Claude, including drafts started at a pause and then dropped."""
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    web_searches: int = Field(default=0, ge=0)
    speech_seconds: float = Field(default=0.0, ge=0)
    claude_usd: float = Field(default=0.0, ge=0)
    claude_big_usd: float = Field(default=0.0, ge=0)
    """Part of ``claude_usd`` spent on the bigger model: what its daily cap reads (ADR 0019)."""
    search_usd: float = Field(default=0.0, ge=0)
    speech_usd: float = Field(default=0.0, ge=0)
    voice_chars: int = Field(default=0, ge=0)
    """Characters spoken by a cloud voice (Cartesia, ADR 0020)."""
    voice_usd: float = Field(default=0.0, ge=0)
    unpriced: int = Field(default=0, ge=0)
    """Requests to a model (or sessions of a speech model) without a known price."""
    latencies_ms: tuple[int, ...] = ()
    """Time to the first word of spoken turns, the most recent last."""

    @field_validator("latencies_ms")
    @classmethod
    def _recent(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(ms for ms in value if ms >= 0)[-MAX_LATENCIES:]

    @property
    def total_usd(self) -> float:
        """Claude, web searches and speech together."""
        return self.claude_usd + self.search_usd + self.speech_usd + self.voice_usd

    def plus(self, other: Day) -> Day:
        """Both days' numbers added up."""
        return Day(
            turns=self.turns + other.turns,
            requests=self.requests + other.requests,
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            web_searches=self.web_searches + other.web_searches,
            speech_seconds=self.speech_seconds + other.speech_seconds,
            claude_usd=self.claude_usd + other.claude_usd,
            claude_big_usd=self.claude_big_usd + other.claude_big_usd,
            search_usd=self.search_usd + other.search_usd,
            speech_usd=self.speech_usd + other.speech_usd,
            voice_chars=self.voice_chars + other.voice_chars,
            voice_usd=self.voice_usd + other.voice_usd,
            unpriced=self.unpriced + other.unpriced,
            latencies_ms=(*self.latencies_ms, *other.latencies_ms),
        )

    def with_claude(self, usage: ClaudeUsage, *, big: bool = False) -> Day:
        """This day plus one request to Claude (``big``: to the bigger model, ADR 0019)."""
        cost = usage.token_cost()
        return self.plus(
            Day(
                requests=1,
                input_tokens=usage.prompt_tokens,
                output_tokens=usage.output_tokens,
                web_searches=usage.web_searches,
                claude_usd=cost or 0.0,
                claude_big_usd=(cost or 0.0) if big else 0.0,
                search_usd=usage.search_cost(),
                unpriced=int(cost is None),
            )
        )

    def with_speech(self, model: str, seconds: float) -> Day:
        """This day plus one AssemblyAI session of ``seconds``."""
        cost = speech_cost(model, seconds)
        return self.plus(
            Day(
                speech_seconds=max(seconds, 0.0),
                speech_usd=cost or 0.0,
                unpriced=int(cost is None),
            )
        )

    def with_voice(self, characters: int, usd_per_million: float) -> Day:
        """This day plus ``characters`` spoken by a cloud voice, at the user's plan price."""
        characters = max(characters, 0)
        return self.plus(
            Day(voice_chars=characters, voice_usd=characters * usd_per_million / 1_000_000)
        )

    def with_turn(self, latency_ms: int | None) -> Day:
        """This day plus one finished turn (``latency_ms``: spoken turns only)."""
        latencies = () if latency_ms is None else (max(latency_ms, 0),)
        return self.plus(Day(turns=1, latencies_ms=latencies))


def total(days: Iterable[Day]) -> Day:
    """Several days added up."""
    result = Day()
    for day in days:
        result = result.plus(day)
    return result


def percentile(values: Sequence[int], fraction: float) -> int | None:
    """The nearest-rank percentile (``fraction`` 0.5: the median); None without values."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(fraction * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


# --------------------------------------------------------------------------- the file


class _File(BaseModel):
    model_config = ConfigDict(extra="ignore")

    version: Literal[1] = 1
    days: dict[date, Day] = Field(default_factory=dict[date, Day])


class UsageBook:
    """The days on record: ``usage.json`` in the data folder, or memory only (``path`` None).

    Several Yarvis processes may add to it: each adds what it counted since its last save
    to what the file holds at that moment.
    """

    def __init__(self, path: Path | None) -> None:
        self.path = path
        self._memory: dict[date, Day] = {}

    @classmethod
    def of(cls, paths: AppPaths) -> UsageBook:
        """Where this installation keeps it."""
        return cls(paths.data_dir / USAGE_FILE)

    def days(self) -> dict[date, Day]:
        """Every day on record, oldest first (empty if there is none, or it is unreadable)."""
        if self.path is None:
            return dict(sorted(self._memory.items()))
        try:
            if self.path.stat().st_size > MAX_FILE_BYTES:
                logger.warning("%s is too large to be Yarvis's: ignored", self.path.name)
                return {}
            raw = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return {}
        except (OSError, UnicodeDecodeError) as exc:
            logger.warning("Could not read %s: %s", self.path.name, type(exc).__name__)
            return {}
        try:
            return dict(sorted(_File.model_validate_json(raw).days.items()))
        except ValidationError:
            logger.warning("%s is damaged: its numbers start again from zero", self.path.name)
            return {}

    def add(self, counted: Mapping[date, Day]) -> dict[date, Day]:
        """Add ``counted`` to the record and keep the latest ``KEEP_DAYS``; returns the record."""
        days = self.days()
        for day, numbers in counted.items():
            days[day] = days.get(day, Day()).plus(numbers)
        kept = dict(sorted(days.items())[-KEEP_DAYS:])
        if self.path is None:
            self._memory = kept
        else:
            content = _File(days=kept).model_dump_json(exclude_defaults=True)
            write_private_file(self.path, _pretty(content), overwrite=True)
        return kept

    def clear(self) -> bool:
        """Delete the record; False if there was none."""
        if self.path is None:
            had = bool(self._memory)
            self._memory = {}
            return had
        try:
            self.path.unlink()
        except FileNotFoundError:
            return False
        return True


def _pretty(content: str) -> str:
    """One day per line: readable, and a diff of two days stays small."""
    data = json.loads(content)
    days = data.get("days", {})
    lines = [f"  {json.dumps(day)}: {json.dumps(numbers)}" for day, numbers in days.items()]
    if not lines:
        return '{"version": 1, "days": {}}\n'
    body = ",\n".join(lines)
    return f'{{"version": 1, "days": {{\n{body}\n}}}}\n'
