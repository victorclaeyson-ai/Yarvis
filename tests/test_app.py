from __future__ import annotations

import json
import re
import threading
from typing import TYPE_CHECKING, Any

import pytest

from fakes import ASSEMBLYAI_KEY, VALID_KEY, FakeKeyring, Invoke
from test_runtime_and_run import install_models
from yarvis.app.bridge import AppApi, AppUi, CommandQueue, EventPump, script_for
from yarvis.app.window import AppError, index_html, uses_cloud_stt
from yarvis.assistant import AssistantControls
from yarvis.config import Config
from yarvis.pipeline import TurnResult, TurnTimings
from yarvis.security.secrets import KEYRING_SERVICE, SecretName, SecretStore

if TYPE_CHECKING:
    from yarvis.cli.app import Services
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


# --------------------------------------------------------------------------- commands from the page


async def test_api_validates_input_and_queues_commands() -> None:
    commands = CommandQueue()
    api = AppApi(
        commands,
        AssistantControls(),
        name="Yarvis",
        voice=True,
        details={"llm": "claude-haiku-5-5"},
    )
    assert api.hello() == {"name": "Yarvis", "voice": True, "details": {"llm": "claude-haiku-5-5"}}
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


def test_the_window_says_when_assemblyai_hears(fake_keyring: FakeKeyring) -> None:
    secrets = SecretStore(backend=fake_keyring, environ={})
    assert not uses_cloud_stt(Config(), secrets)
    secrets.set(SecretName.ASSEMBLYAI, ASSEMBLYAI_KEY)
    assert uses_cloud_stt(Config(), secrets)
    assert not uses_cloud_stt(Config.model_validate({"voice": {"cloud_stt": False}}), secrets)
