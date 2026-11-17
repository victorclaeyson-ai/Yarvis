from __future__ import annotations

import json
import re
import threading
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import pytest

from fakes import VALID_KEY, FakeKeyring, Invoke
from test_runtime_and_run import install_models
from yarvis.app.bridge import AppApi, AppUi, CommandQueue, EventPump, script_for
from yarvis.app.window import AppError, index_html
from yarvis.assistant import AssistantControls
from yarvis.pipeline import TurnResult, TurnTimings
from yarvis.security.secrets import KEYRING_SERVICE
from yarvis.telemetry import Link, UsageUpdate
from yarvis.tools.timers import Alarm
from yarvis.tools.weather import Day as WeatherDay
from yarvis.tools.weather import Outlook
from yarvis.usage import PRICES_AS_OF, Day

if TYPE_CHECKING:
    from yarvis.cli.services import Services
    from yarvis.paths import AppPaths


# --------------------------------------------------------------------------- the page


def test_page_has_a_strict_content_security_policy() -> None:
    html = index_html()
    policy = re.search(r'Content-Security-Policy"\s+content="([^"]+)"', html)
    assert policy is not None
    rules = policy.group(1)
    for rule in (
        "default-src 'none'",
        "connect-src 'none'",
        "base-uri 'none'",
        "form-action 'none'",
    ):
        assert rule in rules
    assert "http" not in rules  # no remote origins at all


def test_page_never_interprets_text_as_html() -> None:
    html = index_html()
    for dangerous in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval("):
        assert dangerous not in html
    assert "new Function" not in html
    assert "textContent" in html


def test_policy_allows_what_pywebview_needs() -> None:
    # pywebview builds window.pywebview.api with new Function(); without 'unsafe-eval' the
    # talk button silently does nothing (the bug that motivated this test).
    assert "script-src 'unsafe-inline' 'unsafe-eval'" in index_html()


def test_page_loads_nothing_remote() -> None:
    assert not re.search(r"""(src|href)\s*=\s*["']https?://""", index_html())


# --------------------------------------------------------------------------- events to the page


def test_script_is_safe_json() -> None:
    events = [{"type": "delta", "text": "</script>\u2028<b>hi</b>"}]
    script = script_for(events)
    assert script.startswith("window.yarvis && window.yarvis.receive(")
    payload = script.removeprefix("window.yarvis && window.yarvis.receive(").removesuffix(");")
    assert json.loads(payload) == events
    assert "\u2028" not in script


def test_app_ui_events() -> None:
    events: list[dict[str, Any]] = []
    ui = AppUi(events.append)
    ui.notice("Loading")
    ui.ready(voice=True, wake_word=True)
    ui.woke()
    ui.listening()
    ui.speech_started()
    for value in (0.01, 0.02, 0.03, 5.0):
        ui.level(value)
    ui.on_transcript("hi")
    ui.on_reply_delta("Hello")
    ui.on_audio_started()
    timings = TurnTimings(speech_end=1.0, llm_started=1.2, first_token=1.5, first_audio=1.9)
    ui.reply_finished(TurnResult("hi", "Hello", timings))
    ui.nothing_heard()
    ui.error("broken")
    ui.stopped()
    ui.interrupted()
    ui.said("Good evening.")
    ui.alert("Your timer is done.")
    ui.activity("set a 10-minute timer")
    kinds = [event["type"] for event in events]
    assert kinds == [
        "notice", "state", "wake", "state", "speech", "level", "level", "user", "state",
        "delta", "state", "done", "notice", "error", "stopped", "interrupted", "said", "alert",
        "activity",
    ]  # fmt: skip
    assert events[-3] == {"type": "said", "text": "Good evening."}
    assert events[-1] == {"type": "activity", "text": "set a 10-minute timer"}
    assert events[1] == {"type": "state", "state": "idle", "voice": True, "wake": True}
    levels = [event["value"] for event in events if event["type"] == "level"]
    assert levels == [0.02, 1.0]  # every second frame, clamped
    done = next(event for event in events if event["type"] == "done")
    assert done["ms"] == 900
    assert "Claude 0.30" in done["timing"]
    json.dumps(events)  # everything must be JSON-serialisable


def test_dashboard_events_carry_data_only() -> None:
    events: list[dict[str, Any]] = []
    ui = AppUi(events.append)
    ui.links([Link("claude", "Claude", "ok", "haiku-5-5"), Link("wake", "Wake word", "off")])
    day = Day(
        turns=3, claude_usd=0.002, speech_usd=0.001, latencies_ms=tuple(range(100, 3100, 100))
    )
    ui.usage(UsageUpdate(today=day, turn_usd=0.0007))
    ui.usage(UsageUpdate(today=Day()))
    links, usage, empty = events
    assert links == {
        "type": "links",
        "links": [
            {"key": "claude", "label": "Claude", "state": "ok", "detail": "haiku-5-5"},
            {"key": "wake", "label": "Wake word", "state": "off", "detail": ""},
        ],
    }
    assert usage["turn_usd"] == 0.0007
    assert usage["as_of"] == PRICES_AS_OF
    today = usage["today"]
    assert today["turns"] == 3
    assert today["total_usd"] == pytest.approx(0.003)
    assert (today["p50_ms"], today["p90_ms"]) == (1500, 2700)
    assert today["recent_ms"] == list(range(700, 3100, 100))  # the last 24
    assert empty["turn_usd"] is None
    assert empty["today"]["p50_ms"] is None
    assert empty["today"]["recent_ms"] == []
    json.dumps(events)


def test_timers_and_weather_events() -> None:
    events: list[dict[str, Any]] = []
    ui = AppUi(events.append)
    due = datetime(2026, 11, 17, 14, 0, tzinfo=UTC)
    ui.timers(
        [
            Alarm(1, "timer", "pasta " * 20, due, 600.0),
            Alarm(2, "reminder", "call back", due + timedelta(hours=4)),
        ]
    )
    ui.weather(
        Outlook(
            place="Leuven, Belgium",
            temperature=9.3,
            feels_like=6.8,
            condition="overcast",
            wind_kmh=14.2,
            today=WeatherDay("light rain", 4.1, 10.2, 40.0),
            tomorrow=None,
        )
    )
    timers, weather = events
    first, second = timers["timers"]
    assert first == {
        "id": 1,
        "kind": "timer",
        "label": ("pasta " * 20)[:60],
        "due_ms": round(due.timestamp() * 1000),
        "seconds": 600.0,
    }
    assert second["kind"] == "reminder"
    assert second["seconds"] is None
    assert weather["weather"]["today"] == {
        "condition": "light rain",
        "low": 4.1,
        "high": 10.2,
        "rain": 40.0,
    }
    assert weather["weather"]["tomorrow"] is None
    json.dumps(events)


def test_page_handles_the_dashboard_events() -> None:
    html = index_html()
    for handler in ("links(e)", "usage(e)", "timers(e)", "weather(e)"):
        assert handler in html
    for element in ('id="links"', 'id="panel"', 'id="spark"', 'id="cost"', 'id="linklist"'):
        assert element in html


# --------------------------------------------------------------------------- commands from the page


async def test_api_validates_input_and_queues_commands() -> None:
    commands = CommandQueue()
    api = AppApi(commands, AssistantControls(), name="Yarvis", voice=True)
    assert api.hello() == {
        "name": "Yarvis",
        "voice": True,
        "user": None,
        "place": None,
        "asks": True,
    }
    told = AppApi(
        commands,
        AssistantControls(),
        name="Y",
        voice=False,
        user="Victor",
        place="Leuven",
        asks=False,
    )
    assert told.hello() == {
        "name": "Y", "voice": False, "user": "Victor", "place": "Leuven", "asks": False,
    }  # fmt: skip
    api.talk()
    api.send("  hello \n  there ")
    api.send(42)
    api.send("   ")
    api.send("x" * 5000)
    commands.close()
    received = [line async for line in commands.lines()]
    assert received == ["", "hello there", "x" * 2000]


async def test_talk_is_ignored_in_text_mode() -> None:
    commands = CommandQueue()
    AppApi(commands, AssistantControls(), name="Y", voice=False).talk()
    commands.close()
    assert [line async for line in commands.lines()] == []


async def test_commands_cross_threads() -> None:
    commands = CommandQueue()
    threading.Thread(target=lambda: (commands.put("from a thread"), commands.close())).start()
    assert [line async for line in commands.lines()] == ["from a thread"]


def test_api_exposes_only_intended_methods() -> None:
    public = {name for name in dir(AppApi) if not name.startswith("_")}
    assert public == {"hello", "talk", "send", "stop"}


def test_api_stop_reaches_the_controls() -> None:
    class Spy(AssistantControls):
        stops = 0

        def stop(self) -> None:
            self.stops += 1

    controls = Spy()
    AppApi(CommandQueue(), controls, name="Y", voice=True).stop()
    assert controls.stops == 1


# --------------------------------------------------------------------------- event pump


class FakeWindow:
    def __init__(self) -> None:
        self.scripts: list[str] = []
        self.ran = threading.Event()

    def run_js(self, script: str) -> None:
        self.scripts.append(script)
        self.ran.set()


def test_pump_waits_for_the_page_then_batches() -> None:
    window = FakeWindow()
    pump = EventPump()
    pump.start(window)
    pump.post({"type": "notice", "text": "one"})
    pump.post({"type": "notice", "text": "two"})
    assert not window.ran.wait(0.1)  # page not loaded yet
    pump.page_loaded()
    assert window.ran.wait(2)
    pump.stop()
    delivered = [e["text"] for s in window.scripts for e in json.loads(s[s.index("(") + 1 : -2])]
    assert delivered == ["one", "two"]


def test_pump_stop_without_events() -> None:
    window = FakeWindow()
    pump = EventPump()
    pump.start(window)
    pump.stop()
    assert window.scripts == []


def test_pump_delivers_events_queued_before_stop() -> None:
    window = FakeWindow()
    pump = EventPump()
    pump.post({"type": "notice", "text": "last words"})
    pump.start(window)
    pump.stop()
    assert len(window.scripts) == 1
    assert "last words" in window.scripts[0]


# --------------------------------------------------------------------------- `yarvis app`


@pytest.fixture
def ready_app(
    services: Services, fake_keyring: FakeKeyring, paths: AppPaths
) -> list[dict[str, Any]]:
    fake_keyring.store[(KEYRING_SERVICE, "anthropic")] = VALID_KEY
    install_models(paths)
    calls: list[dict[str, Any]] = []

    def fake_runner(config: object, **kwargs: Any) -> None:
        calls.append(kwargs)

    services.app_runner = fake_runner
    return calls


def test_app_command_opens_the_window(invoke: Invoke, ready_app: list[dict[str, Any]]) -> None:
    result = invoke("app")
    assert result.exit_code == 0, result.output
    assert "Opening" in result.output
    assert len(ready_app) == 1
    assert ready_app[0]["voice_input"] is True
    assert ready_app[0]["api_key"].get_secret_value() == VALID_KEY


def test_app_command_text_mode(invoke: Invoke, ready_app: list[dict[str, Any]]) -> None:
    assert invoke("app", "--text", "--mute").exit_code == 0
    assert ready_app[0]["voice_input"] is False
    assert ready_app[0]["voice_output"] is False


def test_app_command_reports_gui_failures(
    invoke: Invoke, services: Services, ready_app: object
) -> None:
    def broken(config: object, **kwargs: Any) -> None:
        raise AppError("could not open the window: no GUI backend")

    services.app_runner = broken
    result = invoke("app")
    assert result.exit_code == 1
    assert "pywebview[qt]" in result.output
    assert "yarvis run" in result.output


def test_app_command_needs_a_key(invoke: Invoke) -> None:
    assert invoke("app").exit_code == 1
