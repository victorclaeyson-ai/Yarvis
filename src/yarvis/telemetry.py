"""The dashboard's numbers: the state of every connection, the cost and the speed.

The cloud clients report what they used (``Reporter``); the turn reports how fast the first
word came; ``Telemetry`` adds it all up per day, keeps it in the ``UsageBook`` and tells the
``Dashboard`` (the app window) whenever something changed. Everything runs on the
assistant's event loop; a dashboard that lives on another thread must hand the updates
over itself (the app's event pump does).
"""

from __future__ import annotations

import logging
import math
import sys
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Final, Literal, Protocol, runtime_checkable

from yarvis.security.redact import default_redactor
from yarvis.usage import Day, UsageBook, local_today

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from datetime import date

    from yarvis.config.models import Config
    from yarvis.pipeline import TurnTimings
    from yarvis.tools.timers import Alarm
    from yarvis.tools.weather import Outlook
    from yarvis.usage import ClaudeUsage

logger = logging.getLogger(__name__)

type LinkState = Literal["ok", "off", "warn", "error"]

CLAUDE: Final = "claude"
SPEECH: Final = "speech"
MAX_DETAIL_CHARS: Final = 140
RECENT_TURNS: Final = 24
"""Latencies shown in the dashboard's little chart."""


@dataclass(frozen=True, slots=True)
class Link:
    """One connection as the dashboard shows it ("Claude: haiku-5-5", all well)."""

    key: str
    label: str
    state: LinkState
    detail: str = ""


@dataclass(frozen=True, slots=True)
class UsageUpdate:
    """Today's numbers, and the cost of the turn that just ended (if one did)."""

    today: Day
    turn_usd: float | None = None


class Dashboard(Protocol):
    """Shows the connections, the numbers, the timers and the weather (the app window)."""

    def links(self, links: Sequence[Link]) -> None:
        """The connections changed (all of them, in order)."""
        ...

    def usage(self, update: UsageUpdate) -> None:
        """Today's numbers changed."""
        ...

    def timers(self, alarms: Sequence[Alarm]) -> None:
        """The running timers and reminders changed (all of them, soonest first)."""
        ...

    def weather(self, outlook: Outlook) -> None:
        """The weather at home, refreshed."""
        ...


class Reporter(Protocol):
    """What a cloud client tells the dashboard."""

    def claude(self, usage: ClaudeUsage) -> None:
        """A request to Claude ended (finished, failed after starting, or cut off)."""
        ...

    def speech(self, model: str, seconds: float) -> None:
        """An AssemblyAI session closed after ``seconds``."""
        ...

    def spoken(self, model: str, characters: int) -> None:
        """A cloud voice spoke ``characters`` (Cartesia bills per character)."""
        ...

    def health(self, key: str, state: LinkState, detail: str) -> None:
        """How a connection is doing now (``key``: ``CLAUDE`` or ``SPEECH``)."""
        ...


@runtime_checkable
class Reports(Protocol):
    """A component that can report to the dashboard (the Claude and AssemblyAI clients)."""

    def report_to(self, reporter: Reporter) -> None:
        """Send reports to ``reporter`` from now on."""
        ...


class Telemetry:
    """Counts usage per day, keeps it, and keeps the dashboard up to date (a ``Reporter``)."""

    def __init__(
        self,
        book: UsageBook | None = None,
        *,
        dashboard: Dashboard | None = None,
        today: Callable[[], date] = local_today,
        voice_usd_per_million: float = 0.0,
    ) -> None:
        self._book = book or UsageBook(None)
        self._voice_usd_per_million = voice_usd_per_million
        """The cloud voice's price per million characters on the user's plan (0: free)."""
        self._big_unpriced = False
        """The bigger model answered at an unknown price: its cap is closed (fail closed)."""
        self._dashboard = dashboard
        self._today = today
        self._links: dict[str, Link] = {}
        self._unsaved: dict[date, Day] = {}
        self._saved: dict[date, Day] = self._book.days()
        self._turn_usd = 0.0

    # ------------------------------------------------------------------ connections

    def start(self, links: Sequence[Link]) -> None:
        """Show the connections as they are at start, and today's numbers so far."""
        self._links = {link.key: _clean(link) for link in links}
        self._show_links()
        self._show_usage()

    @property
    def links(self) -> tuple[Link, ...]:
        """Every connection, in the order they were given."""
        return tuple(self._links.values())

    def health(self, key: str, state: LinkState, detail: str) -> None:
        """Update one connection (shown only if something changed)."""
        link = self._links.get(key)
        if link is None:
            return
        updated = _clean(replace(link, state=state, detail=detail))
        if updated != link:
            self._links[key] = updated
            self._show_links()

    # ------------------------------------------------------------------ usage

    def claude(self, usage: ClaudeUsage, *, big: bool = False) -> None:
        """Count a request to Claude (``big``: made by the bigger model's client)."""
        before = self._pending()
        self._count(before.with_claude(usage, big=big))
        self._turn_usd += self._pending().total_usd - before.total_usd
        if big and usage.token_cost() is None and not self._big_unpriced:
            self._big_unpriced = True
            logger.warning(
                "The bigger model %s has no known price: Yarvis stops using it today. "
                "Set llm.big_model to a model it knows (see yarvis usage).",
                usage.model,
            )

    def for_bigger_model(self) -> Reporter:
        """What the bigger model's client reports to: its spend counts towards its cap.

        Counted by which client made the request, never by the model name in the answer,
        so an unusual id can't slip past the cap (ADR 0019).
        """
        return _BiggerModel(self)

    def big_model_usd_today(self) -> float:
        """Today's spend on the bigger model (infinite if its price is unknown: closed)."""
        if self._big_unpriced:
            return math.inf
        return self.today.claude_big_usd

    def speech(self, model: str, seconds: float) -> None:
        """Count an AssemblyAI session."""
        before = self._pending()
        self._count(before.with_speech(model, seconds))
        self._turn_usd += self._pending().total_usd - before.total_usd

    def spoken(self, model: str, characters: int) -> None:
        """Count characters spoken by the cloud voice."""
        before = self._pending()
        self._count(before.with_voice(characters, self._voice_usd_per_million))
        self._turn_usd += self._pending().total_usd - before.total_usd

    def turn_finished(self, timings: TurnTimings) -> None:
        """A reply was given: count it, keep the numbers, show the turn's cost.

        Only spoken turns count towards the voice-to-voice latency.
        """
        spoken = timings.speech_end is not None
        self._count(self._pending().with_turn(timings.voice_to_voice_ms if spoken else None))
        turn_usd, self._turn_usd = self._turn_usd, 0.0
        self.save()
        self._show_usage(turn_usd)

    @property
    def today(self) -> Day:
        """Today's numbers: kept ones plus those not kept yet."""
        day = self._today()
        return self._saved.get(day, Day()).plus(self._unsaved.get(day, Day()))

    def save(self) -> None:
        """Add what was counted since the last save to the ``UsageBook``."""
        if not self._unsaved:
            return
        unsaved, self._unsaved = self._unsaved, {}
        try:
            self._saved = self._book.add(unsaved)
        except OSError as exc:
            logger.warning("Could not keep the usage numbers: %s", exc)
            for day, numbers in unsaved.items():  # kept in memory, tried again next time
                self._unsaved[day] = numbers.plus(self._unsaved.get(day, Day()))

    def close(self) -> None:
        """Keep what is left (Yarvis is closing)."""
        self.save()

    # ------------------------------------------------------------------ helpers

    def _pending(self) -> Day:
        return self._unsaved.get(self._today(), Day())

    def _count(self, numbers: Day) -> None:
        self._unsaved[self._today()] = numbers

    def _show_links(self) -> None:
        if self._dashboard is not None:
            self._dashboard.links(self.links)

    def _show_usage(self, turn_usd: float | None = None) -> None:
        if self._dashboard is not None:
            self._dashboard.usage(UsageUpdate(today=self.today, turn_usd=turn_usd))


def _clean(link: Link) -> Link:
    """Details may quote an error message: no secrets, and short."""
    detail = " ".join(default_redactor.redact(link.detail).split())
    if len(detail) > MAX_DETAIL_CHARS:
        detail = detail[: MAX_DETAIL_CHARS - 1].rstrip() + "…"
    return replace(link, detail=detail)


# --------------------------------------------------------------------------- at start


@dataclass(frozen=True, slots=True)
class Setup:
    """What is connected when Yarvis starts (gathered by ``runtime.run_assistant``)."""

    model: str
    """Claude's model id."""
    hearing: str | None
    """The speech recognition engine on this computer ("parakeet"; None: typing only)."""
    cloud_hearing: bool
    """AssemblyAI hears the user (the engine on this computer is the backup)."""
    voice: str | None
    """The voice (None: muted)."""
    wake_word: bool
    spotify: bool
    """Spotify tools are on offer."""
    spotify_login: bool
    """...with the user's own login (playlists, playback)."""
    personal: bool
    """Calendar and email tools are on offer (a Mac)."""
    skills: int
    profile: bool
    cloud_voice: bool = False
    """Cartesia speaks, with Kokoro as backup (ADR 0020)."""
    memory: int | None = None
    """How many facts Yarvis remembers (None: memory is not on offer)."""
    mcp: Sequence[Link] = ()
    platform: str = sys.platform


def starting_links(config: Config, setup: Setup) -> list[Link]:
    """Every connection Yarvis has or could have, in the order the dashboard lists them."""
    links = [Link(CLAUDE, "Claude", "ok", short_model(setup.model)), _hearing(config, setup)]
    links.append(
        Link("voice", "Voice", "ok", _voice(setup))
        if setup.voice
        else Link("voice", "Voice", "off", "muted")
    )
    links.append(
        Link("wake", "Wake word", "ok", "“Hey Yarvis”, heard on this computer")
        if setup.wake_word
        else Link("wake", "Wake word", "off", "off")
    )
    links.append(_spotify(setup))
    links.append(_personal(config, setup))
    links.append(
        Link("weather", "Weather", "ok", "Open-Meteo")
        if config.tools.weather
        else Link("weather", "Weather", "off", "off in the config")
    )
    links.append(
        Link("search", "Web search", "ok", "Claude's web search")
        if config.tools.web_search
        else Link("search", "Web search", "off", "off in the config")
    )
    links.append(
        Link("skills", "Skills", "ok", f"{setup.skills} added")
        if setup.skills
        else Link("skills", "Skills", "off", "none added (yarvis skills add)")
    )
    links.append(
        Link("profile", "Profile", "ok", "read before every question")
        if setup.profile
        else Link("profile", "Profile", "off", "empty (yarvis profile edit)")
    )
    if setup.memory is not None:
        links.append(
            Link("memory", "Memory", "ok", f"{_facts(setup.memory)}, read before every question")
            if setup.memory
            else Link("memory", "Memory", "off", "nothing remembered yet")
        )
    links += setup.mcp
    return links


def _hearing(config: Config, setup: Setup) -> Link:
    if setup.hearing is None:
        return Link(SPEECH, "Speech recognition", "off", "off: typing only")
    engine = setup.hearing[:1].upper() + setup.hearing[1:]
    if setup.cloud_hearing:
        return Link(SPEECH, "Speech recognition", "ok", cloud_hearing(config.voice.cloud_region))
    return Link(SPEECH, "Speech recognition", "ok", f"{engine}, on this computer")


def cloud_hearing(region: str) -> str:
    """How the dashboard says that AssemblyAI is listening."""
    return f"AssemblyAI, {'EU' if region == 'eu' else 'global'} servers"


def short_model(model: str) -> str:
    """A model id as the dashboard shows it ("haiku-5-5")."""
    return model.removeprefix("claude-")


def _spotify(setup: Setup) -> Link:
    if not setup.spotify:
        return Link("spotify", "Spotify", "off", "not set up (yarvis spotify login)")
    if setup.spotify_login:
        return Link("spotify", "Spotify", "ok", "your account")
    return Link("spotify", "Spotify", "warn", "not logged in: yarvis spotify login")


def _personal(config: Config, setup: Setup) -> Link:
    if setup.personal:
        return Link("accounts", "Calendar & mail", "ok", "the Mac's own accounts")
    if setup.platform != "darwin":
        return Link("accounts", "Calendar & mail", "off", "on a Mac only, for now")
    if not (config.tools.calendar or config.tools.mail):
        return Link("accounts", "Calendar & mail", "off", "off in the config")
    return Link("accounts", "Calendar & mail", "off", "not available")


def _facts(count: int) -> str:
    """'1 fact', '3 facts'."""
    return f"{count} fact{'' if count == 1 else 's'}"


class _BiggerModel:
    """A ``Reporter`` that marks every Claude request as the bigger model's."""

    def __init__(self, telemetry: Telemetry) -> None:
        self._telemetry = telemetry

    def claude(self, usage: ClaudeUsage) -> None:
        self._telemetry.claude(usage, big=True)

    def speech(self, model: str, seconds: float) -> None:
        self._telemetry.speech(model, seconds)

    def health(self, key: str, state: LinkState, detail: str) -> None:
        self._telemetry.health(key, state, detail)

    def spoken(self, model: str, characters: int) -> None:
        self._telemetry.spoken(model, characters)


def _voice(setup: Setup) -> str:
    return "Cartesia · Kokoro as backup" if setup.cloud_voice else f"Kokoro · {setup.voice}"
