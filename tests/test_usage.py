"""Cost and speed: prices, the daily record, telemetry and ``yarvis usage``."""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import date, timedelta
from typing import TYPE_CHECKING

import pytest

from yarvis.config import Config
from yarvis.pipeline import TurnTimings
from yarvis.telemetry import (
    CLAUDE,
    SPEECH,
    Link,
    Setup,
    Telemetry,
    UsageUpdate,
    starting_links,
)
from yarvis.usage import (
    KEEP_DAYS,
    MAX_FILE_BYTES,
    MAX_LATENCIES,
    ClaudeUsage,
    Day,
    UsageBook,
    estimated_tokens,
    local_today,
    model_prices,
    percentile,
    speech_cost,
    total,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from fakes import Invoke
    from yarvis.paths import AppPaths
    from yarvis.tools.timers import Alarm
    from yarvis.tools.weather import Outlook

DAY = date(2026, 11, 17)


# --------------------------------------------------------------------------- prices


def test_haiku_costs_ten_cents_per_million_input_tokens() -> None:
    usage = ClaudeUsage("claude-haiku-5-5", input_tokens=50_000, output_tokens=1_000_000)
    assert usage.token_cost() == pytest.approx(0.005 + 0.50)


def test_long_prompts_cost_more_on_haiku() -> None:
    usage = ClaudeUsage("claude-haiku-5-5", input_tokens=60_000, cache_read_tokens=60_000)
    assert usage.prompt_tokens == 120_000
    assert usage.token_cost() == pytest.approx((60_000 * 0.50 + 60_000 * 0.05) / 1e6)


def test_every_kind_of_token_is_priced() -> None:
    usage = ClaudeUsage(
        "claude-sonnet-5-5",
        input_tokens=1_000,
        output_tokens=1_000,
        cache_write_tokens=1_000,
        cache_read_tokens=1_000,
    )
    assert usage.token_cost() == pytest.approx((2 + 10 + 2.5 + 0.10) / 1_000)


@pytest.mark.parametrize(
    ("model", "known"),
    [
        ("claude-opus-5-5", True),
        ("claude-opus-5", True),
        ("claude-haiku-4-5-20251001", True),  # an id with a date
        ("claude-opus-5-6", False),  # a newer model is not priced as an older one
        ("claude-haiku-4-5-latest", False),
        ("gpt-5", False),
    ],
)
def test_model_ids_are_matched_exactly_or_with_a_date(model: str, known: bool) -> None:
    assert (model_prices(model) is not None) is known


def test_opus_5_5_is_not_priced_as_opus_5() -> None:
    assert ClaudeUsage("claude-opus-5-5", output_tokens=1_000_000).token_cost() == 20.0
    assert ClaudeUsage("claude-opus-5", output_tokens=1_000_000).token_cost() == 25.0


def test_unknown_models_have_no_price_but_searches_do() -> None:
    usage = ClaudeUsage("claude-future-9", input_tokens=500, web_searches=3)
    assert usage.token_cost() is None
    assert usage.search_cost() == pytest.approx(0.03)


def test_assemblyai_costs_45_cents_an_hour() -> None:
    assert speech_cost("universal-3-6-pro", 3600) == pytest.approx(0.45)
    assert speech_cost("universal-3-6-pro", -1) == 0
    assert speech_cost("some-other-model", 60) is None


def test_estimated_tokens_round_up() -> None:
    assert estimated_tokens(0) == 0
    assert estimated_tokens(9) == 3


def test_percentiles() -> None:
    assert percentile([], 0.5) is None
    assert percentile([900], 0.9) == 900
    values = list(range(1, 101))
    assert percentile(values, 0.5) == 50
    assert percentile(values, 0.9) == 90
    assert percentile([300, 100, 200], 0.5) == 200


# --------------------------------------------------------------------------- days


def test_a_day_adds_requests_sessions_and_turns() -> None:
    day = (
        Day()
        .with_claude(ClaudeUsage("claude-haiku-5-5", input_tokens=10_000, web_searches=1))
        .with_claude(ClaudeUsage("claude-future-9", input_tokens=10))
        .with_speech("universal-3-6-pro", 36.0)
        .with_speech("unknown-model", 5.0)
        .with_turn(850)
        .with_turn(None)
    )
    assert day.requests == 2
    assert day.input_tokens == 10_010
    assert day.unpriced == 2
    assert day.turns == 2
    assert day.latencies_ms == (850,)
    assert day.speech_seconds == pytest.approx(41.0)
    assert day.claude_usd == pytest.approx(0.001)
    assert day.search_usd == pytest.approx(0.01)
    assert day.speech_usd == pytest.approx(0.0045)
    assert day.total_usd == pytest.approx(0.0155)


def test_a_day_keeps_only_recent_latencies() -> None:
    day = Day(latencies_ms=tuple(range(MAX_LATENCIES + 10)))
    assert len(day.latencies_ms) == MAX_LATENCIES
    assert day.latencies_ms[-1] == MAX_LATENCIES + 9
    merged = day.plus(Day(latencies_ms=(7,)))
    assert merged.latencies_ms[-1] == 7
    assert len(merged.latencies_ms) == MAX_LATENCIES


def test_total_of_days() -> None:
    assert total([]) == Day()
    assert total([Day(turns=1), Day(turns=2)]).turns == 3


# --------------------------------------------------------------------------- the book


def test_the_book_adds_up_and_reads_back(tmp_path: Path) -> None:
    book = UsageBook(tmp_path / "data" / "usage.json")
    assert book.days() == {}
    book.add({DAY: Day(turns=1, claude_usd=0.5)})
    record = book.add({DAY: Day(turns=2), DAY - timedelta(days=1): Day(turns=4)})
    assert record[DAY].turns == 3
    assert list(record) == [DAY - timedelta(days=1), DAY]
    assert UsageBook(book.path).days() == record
    assert book.path is not None
    text = book.path.read_text(encoding="utf-8")
    assert text.count("\n") == 4  # one line per day
    assert json.loads(text)["days"]["2026-11-17"]["turns"] == 3


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission bits")
def test_the_book_is_private(tmp_path: Path) -> None:
    book = UsageBook(tmp_path / "usage.json")
    book.add({DAY: Day(turns=1)})
    assert book.path is not None
    assert book.path.stat().st_mode & 0o077 == 0


def test_the_book_keeps_a_year(tmp_path: Path) -> None:
    book = UsageBook(tmp_path / "usage.json")
    book.add({DAY - timedelta(days=n): Day(turns=1) for n in range(KEEP_DAYS + 5)})
    days = book.days()
    assert len(days) == KEEP_DAYS
    assert min(days) == DAY - timedelta(days=KEEP_DAYS - 1)


def test_two_processes_add_to_each_other(tmp_path: Path) -> None:
    path = tmp_path / "usage.json"
    UsageBook(path).add({DAY: Day(turns=1)})
    UsageBook(path).add({DAY: Day(turns=1)})
    assert UsageBook(path).days()[DAY].turns == 2


@pytest.mark.parametrize(
    "content",
    [
        "not json",
        '{"version": 2, "days": {}}',
        '{"version": 1, "days": {"yesterday": {}}}',
        '{"version": 1, "days": {"2026-11-17": {"turns": -1}}}',
    ],
)
def test_a_damaged_book_starts_again(
    tmp_path: Path, content: str, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "usage.json"
    path.write_text(content, encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        assert UsageBook(path).days() == {}
    assert "damaged" in caplog.text


def test_unknown_fields_are_ignored(tmp_path: Path) -> None:
    path = tmp_path / "usage.json"
    path.write_text('{"version": 1, "days": {"2026-11-17": {"turns": 2, "new": 1}}}', "utf-8")
    assert UsageBook(path).days()[DAY].turns == 2


def test_a_huge_or_unreadable_book_is_ignored(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    huge = tmp_path / "huge.json"
    huge.write_bytes(b" " * (MAX_FILE_BYTES + 1))
    folder = tmp_path / "folder.json"
    folder.mkdir()
    binary = tmp_path / "binary.json"
    binary.write_bytes(b"\xff\xfe\x00")
    with caplog.at_level(logging.WARNING):
        assert UsageBook(huge).days() == {}
        assert UsageBook(folder).days() == {}
        assert UsageBook(binary).days() == {}
    assert "too large" in caplog.text
    assert "Could not read" in caplog.text


def test_an_empty_record_is_still_a_valid_file(tmp_path: Path) -> None:
    book = UsageBook(tmp_path / "usage.json")
    assert book.add({}) == {}
    assert book.path is not None
    assert json.loads(book.path.read_text(encoding="utf-8")) == {"version": 1, "days": {}}


def test_clearing_the_book(tmp_path: Path) -> None:
    book = UsageBook(tmp_path / "usage.json")
    assert not book.clear()
    book.add({DAY: Day(turns=1)})
    assert book.clear()
    assert book.days() == {}


def test_a_book_in_memory() -> None:
    book = UsageBook(None)
    assert not book.clear()
    book.add({DAY: Day(turns=1)})
    assert book.days()[DAY].turns == 1
    assert book.clear()
    assert book.days() == {}


def test_the_book_lives_in_the_data_folder(paths: AppPaths) -> None:
    assert UsageBook.of(paths).path == paths.data_dir / "usage.json"


def test_local_today_is_a_date() -> None:
    assert isinstance(local_today(), date)


# --------------------------------------------------------------------------- telemetry


class RecordingDashboard:
    def __init__(self) -> None:
        self.link_updates: list[tuple[Link, ...]] = []
        self.usage_updates: list[UsageUpdate] = []
        self.timer_updates: list[list[str]] = []
        self.outlooks: list[Outlook] = []

    def links(self, links: Sequence[Link]) -> None:
        self.link_updates.append(tuple(links))

    def usage(self, update: UsageUpdate) -> None:
        self.usage_updates.append(update)

    def timers(self, alarms: Sequence[Alarm]) -> None:
        self.timer_updates.append([alarm.label for alarm in alarms])

    def weather(self, outlook: Outlook) -> None:
        self.outlooks.append(outlook)


def spoken(ms: int) -> TurnTimings:
    return TurnTimings(speech_end=10.0, first_audio=10.0 + ms / 1000)


LINKS = [Link(CLAUDE, "Claude", "ok", "haiku-5-5"), Link(SPEECH, "Speech recognition", "ok")]


def test_telemetry_shows_links_and_today_at_start() -> None:
    book = UsageBook(None)
    book.add({DAY: Day(turns=5, claude_usd=0.01)})
    dashboard = RecordingDashboard()
    telemetry = Telemetry(book, dashboard=dashboard, today=lambda: DAY)
    telemetry.start(LINKS)
    assert dashboard.link_updates == [tuple(LINKS)]
    assert dashboard.usage_updates == [UsageUpdate(today=Day(turns=5, claude_usd=0.01))]


def test_health_changes_are_shown_once_and_scrubbed() -> None:
    dashboard = RecordingDashboard()
    telemetry = Telemetry(dashboard=dashboard, today=lambda: DAY)
    telemetry.start(LINKS)
    telemetry.health(CLAUDE, "ok", "haiku-5-5")  # no change: nothing shown
    telemetry.health("nonexistent", "error", "ignored")
    leaked = "sk-ant-api03-" + "a" * 30
    telemetry.health(CLAUDE, "error", f"Claude said no ({leaked}) " + "x" * 300)
    assert len(dashboard.link_updates) == 2
    claude = telemetry.links[0]
    assert claude.state == "error"
    assert leaked not in claude.detail
    assert "[REDACTED]" in claude.detail
    assert len(claude.detail) <= 140
    assert claude.detail.endswith("…")


def test_a_turn_shows_its_own_cost_including_dropped_drafts() -> None:
    dashboard = RecordingDashboard()
    telemetry = Telemetry(dashboard=dashboard, today=lambda: DAY)
    telemetry.start(LINKS)
    dropped = ClaudeUsage("claude-haiku-5-5", input_tokens=5_000, output_tokens=20)
    used = ClaudeUsage("claude-haiku-5-5", input_tokens=5_000, output_tokens=40, web_searches=1)
    telemetry.claude(dropped)
    telemetry.claude(used)
    telemetry.speech("universal-3-6-pro", 8.0)
    telemetry.turn_finished(spoken(900))
    update = dashboard.usage_updates[-1]
    expected = (10_000 * 0.10 + 60 * 0.50) / 1e6 + 0.01 + 0.45 * 8 / 3600
    assert update.turn_usd == pytest.approx(expected)
    assert update.today.requests == 2
    assert update.today.latencies_ms == (900,)
    telemetry.turn_finished(TurnTimings(transcribed=1.0, first_audio=1.5))  # typed
    second = dashboard.usage_updates[-1]
    assert second.turn_usd == 0.0
    assert second.today.turns == 2
    assert second.today.latencies_ms == (900,)  # typed turns are not voice-to-voice


def test_numbers_are_kept_at_each_turn_and_at_close(tmp_path: Path) -> None:
    book = UsageBook(tmp_path / "usage.json")
    telemetry = Telemetry(book, today=lambda: DAY)
    telemetry.claude(ClaudeUsage("claude-haiku-5-5", input_tokens=100))
    assert book.days() == {}
    telemetry.turn_finished(spoken(700))
    assert book.days()[DAY].turns == 1
    telemetry.speech("universal-3-6-pro", 3.0)
    telemetry.close()
    telemetry.close()  # nothing left: nothing written
    kept = UsageBook(book.path).days()[DAY]
    assert kept.speech_seconds == 3.0
    assert telemetry.today == kept


def test_a_failed_save_is_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    book = UsageBook(tmp_path / "usage.json")
    telemetry = Telemetry(book, today=lambda: DAY)
    telemetry.claude(ClaudeUsage("claude-haiku-5-5", input_tokens=100))

    def full(_counted: object) -> dict[date, Day]:
        raise OSError("disk full")

    monkeypatch.setattr(book, "add", full)
    with caplog.at_level(logging.WARNING):
        telemetry.turn_finished(spoken(800))
    assert "disk full" in caplog.text
    assert telemetry.today.turns == 1  # still counted
    monkeypatch.undo()
    telemetry.close()
    assert book.days()[DAY].requests == 1


def test_days_change_at_midnight() -> None:
    today = [DAY]
    telemetry = Telemetry(today=lambda: today[0])
    telemetry.turn_finished(spoken(500))
    today[0] = DAY + timedelta(days=1)
    assert telemetry.today == Day()
    telemetry.turn_finished(spoken(600))
    assert telemetry.today.latencies_ms == (600,)


# --------------------------------------------------------------------------- links at start


def setup(**overrides: object) -> Setup:
    values: dict[str, object] = {
        "model": "claude-haiku-5-5",
        "hearing": "parakeet",
        "cloud_hearing": False,
        "voice": "bm_george",
        "wake_word": True,
        "spotify": True,
        "spotify_login": True,
        "personal": True,
        "skills": 2,
        "profile": True,
        "platform": "darwin",
    }
    values.update(overrides)
    return Setup(**values)  # pyright: ignore[reportArgumentType]


def details(links: Sequence[Link]) -> dict[str, tuple[str, str]]:
    return {link.key: (link.state, link.detail) for link in links}


def test_everything_connected() -> None:
    mcp = Link("mcp:time", "MCP · time", "ok", "2 tools")
    links = starting_links(Config(), setup(mcp=[mcp]))
    assert [link.key for link in links] == [
        "claude", "speech", "voice", "wake", "spotify", "accounts", "weather", "search",
        "skills", "profile", "mcp:time",
    ]  # fmt: skip
    shown = details(links)
    assert shown["claude"] == ("ok", "haiku-5-5")
    assert shown["speech"] == ("ok", "Parakeet, on this computer")
    assert shown["voice"] == ("ok", "Kokoro · bm_george")
    assert shown["skills"] == ("ok", "2 added")
    assert all(state == "ok" for state, _ in shown.values())


def test_assemblyai_is_named_with_its_servers() -> None:
    eu = details(starting_links(Config(), setup(cloud_hearing=True)))
    assert eu["speech"] == ("ok", "AssemblyAI, EU servers")
    world = Config.model_validate({"voice": {"cloud_region": "global"}})
    assert details(starting_links(world, setup(cloud_hearing=True)))["speech"][1] == (
        "AssemblyAI, global servers"
    )


def test_what_is_off_says_how_to_turn_it_on() -> None:
    config = Config.model_validate({"tools": {"weather": False, "web_search": False}})
    shown = details(
        starting_links(
            config,
            setup(
                hearing=None,
                voice=None,
                wake_word=False,
                spotify=False,
                personal=False,
                skills=0,
                profile=False,
                platform="win32",
            ),
        )
    )
    assert shown["speech"] == ("off", "off: typing only")
    assert shown["voice"] == ("off", "muted")
    assert shown["wake"][0] == "off"
    assert shown["spotify"] == ("off", "not set up (yarvis spotify login)")
    assert shown["accounts"] == ("off", "on a Mac only, for now")
    assert shown["weather"][0] == shown["search"][0] == "off"
    assert "yarvis skills add" in shown["skills"][1]
    assert "yarvis profile edit" in shown["profile"][1]


def test_spotify_without_a_login_is_a_warning() -> None:
    assert details(starting_links(Config(), setup(spotify_login=False)))["spotify"][0] == "warn"


def test_calendar_and_mail_on_a_mac() -> None:
    off = Config.model_validate({"tools": {"calendar": False, "mail": False}})
    assert details(starting_links(off, setup(personal=False)))["accounts"] == (
        "off",
        "off in the config",
    )
    assert details(starting_links(Config(), setup(personal=False)))["accounts"] == (
        "off",
        "not available",
    )


# --------------------------------------------------------------------------- yarvis usage


def test_usage_with_nothing_counted(invoke: Invoke) -> None:
    result = invoke("usage")
    assert result.exit_code == 0, result.output
    assert "Nothing counted yet" in result.output


def test_usage_shows_days_and_totals(invoke: Invoke, paths: AppPaths) -> None:
    today = local_today()
    book = UsageBook.of(paths)
    book.add(
        {
            today: Day(turns=3, claude_usd=0.0042, speech_usd=0.002, latencies_ms=(800, 900)),
            today - timedelta(days=1): Day(turns=1, search_usd=0.01, latencies_ms=(1200,)),
            today - timedelta(days=30): Day(turns=99),  # outside the window
        }
    )
    result = invoke("usage")
    assert result.exit_code == 0, result.output
    assert today.strftime("%a %d %b") in result.output
    assert "Together" in result.output
    assert "$0.0042" in result.output
    assert "p50 0.90 s, p90 1.20 s" in result.output
    assert "99" not in result.output
    assert "list prices" in result.output
    wide = invoke("usage", "--days", "31")
    assert "99" in wide.output


def test_usage_warns_about_unpriced_models(invoke: Invoke, paths: AppPaths) -> None:
    UsageBook.of(paths).add({local_today(): Day(turns=1, unpriced=2)})
    assert "too low" in invoke("usage").output


def test_usage_clear(invoke: Invoke, paths: AppPaths) -> None:
    UsageBook.of(paths).add({local_today(): Day(turns=1)})
    assert invoke("usage", "clear", stdin="n\n").exit_code == 1
    assert UsageBook.of(paths).days()
    result = invoke("usage", "clear", "--yes")
    assert result.exit_code == 0
    assert "Deleted" in result.output
    assert UsageBook.of(paths).days() == {}
    assert "nothing to delete" in invoke("usage", "clear", "--yes").output


def test_usage_file_holds_numbers_only(paths: AppPaths) -> None:
    telemetry = Telemetry(UsageBook.of(paths))
    telemetry.claude(ClaudeUsage("claude-haiku-5-5", input_tokens=100, output_tokens=10))
    telemetry.turn_finished(spoken(640))
    path = UsageBook.of(paths).path
    assert path is not None
    numbers = json.loads(path.read_text(encoding="utf-8"))["days"]
    (day,) = numbers.values()
    assert all(isinstance(value, int | float | list) for value in day.values())
    assert os.path.getsize(path) < 1_000  # noqa: PTH202
