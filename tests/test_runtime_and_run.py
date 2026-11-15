from __future__ import annotations

import asyncio
import io
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Self

import numpy as np
import pytest
from pydantic import SecretStr
from rich.console import Console

from fakes import VALID_KEY, FakeKeyring, FakeLLM, FakeSTT, FakeTTS, FakeWakeWord, Invoke
from test_computer import FakeRunner
from test_listen_and_assistant import RecordingUi, lines
from test_speech_engines import frames, speech_clip
from test_wake_and_barge_in import EventUi
from yarvis.cli.app import Services
from yarvis.cli.ui import ConsoleUi, format_timings, stdin_lines
from yarvis.config import Config
from yarvis.config.models import VoiceConfig
from yarvis.llm.base import ToolCall
from yarvis.models import (
    KOKORO_FILES,
    PARAKEET_FILES,
    WAKE_WORD_FILES,
    WHISPER_REVISIONS,
    ModelFile,
    ModelIntegrityError,
    kokoro_dir,
    parakeet_dir,
    wake_word_dir,
    whisper_dir,
)
from yarvis.paths import is_private
from yarvis.pipeline import TurnResult, TurnTimings
from yarvis.runtime import Backends, run_assistant
from yarvis.security.secrets import KEYRING_SERVICE
from yarvis.speech.silero import SileroVAD
from yarvis.tools.computer import Apps
from yarvis.tools.desktop import computer_tools
from yarvis.tools.spotify import SpotifyKeys

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable
    from pathlib import Path
    from types import TracebackType

    from yarvis.paths import AppPaths
    from yarvis.speech.base import AudioArray


class LoadableSTT(FakeSTT):
    loaded = False

    def load(self) -> None:
        self.loaded = True


class LoadableTTS(FakeTTS):
    loaded = False

    def load(self) -> None:
        self.loaded = True


class FakeMic:
    def __init__(self, script: list[AudioArray] | None = None) -> None:
        self.opened = 0
        self.flushes = 0
        self._script = script
        self._pending: list[AudioArray] = []

    async def __aenter__(self) -> Self:
        self.opened += 1
        # The user asks one question; after that the microphone hears nothing (the stream
        # ends), so a follow-up listen returns without speech.
        first = self._script if self._script is not None else frames(speech_clip())
        self._pending = list(first) if self.opened == 1 else []
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        pass

    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> AudioArray:
        await asyncio.sleep(0)  # like real hardware: one frame at a time, safe to cancel
        if not self._pending:
            raise StopAsyncIteration
        return self._pending.pop(0)

    def flush(self) -> None:
        self.flushes += 1


class FakeSink:
    def __init__(self, device_name: str | None = None) -> None:
        self.device_name = device_name
        self.played = 0
        self.closed = False

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.closed = True

    async def play(self, audio: AudioArray) -> None:
        self.played += 1

    async def drain(self) -> None:
        pass

    def stop(self) -> None:
        pass


class FakeBackends:
    def __init__(self, llm: FakeLLM) -> None:
        self.llm = llm
        self.stt = LoadableSTT("hey yarvis what time is it")
        self.tts = LoadableTTS()
        self.mic = FakeMic()
        self.sink = FakeSink()
        self.wake = FakeWakeWord()
        self.keys: list[str] = []
        self.stt_engines: list[str] = []

    def build(self) -> Backends:
        def make_llm(config: Config, key: SecretStr) -> FakeLLM:
            self.keys.append(key.get_secret_value())
            return self.llm

        def make_stt(config: Config, cache: Path) -> LoadableSTT:
            self.stt_engines.append(config.voice.stt)
            return self.stt

        return Backends(
            llm=make_llm,
            stt=make_stt,
            tts=lambda config, cache: self.tts,
            vad=SileroVAD,
            microphone=lambda config: self.mic,
            speaker=lambda config, rate: self.sink,
            wake_word=lambda config, cache: self.wake,
            computer=lambda config, spotify: [],
        )


NO_WAKE_WORD = Config.model_validate({"voice": {"wake_word": False}})


async def test_full_voice_turn_through_the_real_wiring(tmp_path: Path) -> None:
    fakes = FakeBackends(FakeLLM(["It is half past one."]))
    ui = RecordingUi()
    loading: list[str] = []
    await run_assistant(
        NO_WAKE_WORD,
        api_key=SecretStr("k"),
        cache_dir=tmp_path,
        lines=lines("", "q"),
        ui=ui,
        backends=fakes.build(),
        on_loading=loading.append,
    )
    assert fakes.stt.loaded and fakes.tts.loaded
    assert not fakes.wake.loaded
    assert loading == ["Loading speech models…", "Warming up…"]
    assert fakes.mic.opened == 2  # the question, then the follow-up window (silence)
    warm_up, *real = fakes.stt.received
    assert warm_up == 8_000  # half a second of silence at startup
    assert len(real) == 1  # the speculation during the final pause, reused at end of turn
    assert real[0] > 2 * 16_000
    # "hey yarvis" is cut from the transcript: Claude only sees the request itself
    assert ui.events[:4] == ["ready:True", "listening", "speech", "you:what time is it"]
    assert "reply:It is half past one." in ui.events
    assert fakes.tts.spoken == ["Ready.", "It is half past one."]
    assert fakes.sink.played == 1 and fakes.sink.closed
    assert fakes.keys == ["k"]
    assert fakes.llm.warm_ups == 2  # at startup, and again when speech started


async def test_wake_word_through_the_real_wiring(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    fakes = FakeBackends(FakeLLM(["Half past one."]))
    trigger = np.full(512, 0.01, np.float32)
    trigger[0] = FakeWakeWord.TRIGGER
    quiet = [np.full(512, 0.001, np.float32)] * 20
    fakes.mic = FakeMic([*quiet, trigger, *frames(speech_clip())])
    fakes.sink = FakeSink(device_name="AirPods Pro")
    ui = EventUi()

    async def quit_after_reply() -> AsyncIterator[str]:
        await ui.replied.wait()
        yield "q"

    caplog.set_level("INFO")
    await run_assistant(
        Config(),
        api_key=SecretStr("k"),
        cache_dir=tmp_path,
        lines=quit_after_reply(),
        ui=ui,
        backends=fakes.build(),
    )
    assert fakes.wake.loaded
    assert fakes.mic.opened == 1  # one stream for the whole session
    assert ui.events[:3] == ["ready:True:wake", "woke", "listening"]
    assert "you:what time is it" in ui.events  # "hey yarvis" is cut from the transcript
    assert "reply:Half past one." in ui.events
    assert fakes.sink.played == 2  # the chime, then the reply
    assert "Barge-in: voice (output device: AirPods Pro)" in caplog.text
    assert any(text.startswith(("Good ", "Still up")) for text in fakes.tts.spoken)  # greetings
    assert "The microphone stays open" in caplog.text


async def test_text_and_mute_mode_load_nothing(tmp_path: Path) -> None:
    fakes = FakeBackends(FakeLLM(["Hello."]))
    ui = RecordingUi()
    await run_assistant(
        Config(),
        api_key=SecretStr("k"),
        cache_dir=tmp_path,
        lines=lines("hi"),
        ui=ui,
        backends=fakes.build(),
        voice_input=False,
        voice_output=False,
    )
    assert not fakes.stt.loaded and not fakes.tts.loaded
    assert ui.events == ["ready:False", "you:hi", "reply:Hello.", "ready:False"]


# --------------------------------------------------------------------------- CLI: run / models


def install_models(
    paths: AppPaths, *, whisper: bool = True, wake_word: bool = True, parakeet: bool = False
) -> None:
    paths.ensure_all()
    sets: list[tuple[Path, tuple[ModelFile, ...]]] = [(kokoro_dir(paths.cache_dir), KOKORO_FILES)]
    if wake_word:
        sets.append((wake_word_dir(paths.cache_dir), WAKE_WORD_FILES))
    if parakeet:
        sets.append((parakeet_dir(paths.cache_dir), PARAKEET_FILES))
    for directory, models in sets:
        directory.mkdir(parents=True, exist_ok=True)
        for model in models:
            with (directory / model.name).open("wb") as handle:
                handle.truncate(model.size)  # sparse file with the right size
    if whisper:
        model = VoiceConfig().stt_model
        snapshot = (
            whisper_dir(paths.cache_dir) / f"models--Systran--faster-whisper-{model}" / "snapshots"
        )
        (snapshot / WHISPER_REVISIONS[model]).mkdir(parents=True, exist_ok=True)
        (snapshot / WHISPER_REVISIONS[model] / "model.bin").write_bytes(b"x")


@pytest.fixture
def ready(services: Services, fake_keyring: FakeKeyring, paths: AppPaths) -> FakeBackends:
    fake_keyring.store[(KEYRING_SERVICE, "anthropic")] = VALID_KEY
    install_models(paths)
    fakes = FakeBackends(FakeLLM(["Good afternoon."]))
    services.backends = fakes.build
    return fakes


def test_run_text_mode(invoke: Invoke, services: Services, ready: FakeBackends) -> None:
    services.lines = lambda: lines("hello", "q")
    result = invoke("run", "--text", "--mute")
    assert result.exit_code == 0, result.output
    assert "Good afternoon." in result.output
    assert "Bye." in result.output
    assert ready.keys == [VALID_KEY]


def test_run_voice_mode(invoke: Invoke, services: Services, ready: FakeBackends) -> None:
    services.lines = lambda: lines("", "q")
    result = invoke("run")
    assert result.exit_code == 0, result.output
    assert "You: what time is it" in result.output
    assert "to first word" in result.output


def test_run_requires_a_key(invoke: Invoke, paths: AppPaths) -> None:
    result = invoke("run", "--text", "--mute")
    assert result.exit_code == 1
    assert "yarvis key set anthropic" in result.output
    assert all(is_private(directory) for directory in paths.all_dirs())


def test_run_rejects_ollama_for_now(invoke: Invoke, paths: AppPaths) -> None:
    invoke("config", "init")
    paths.config_file.write_text('[llm]\nprovider = "ollama"\n')
    result = invoke("run", "--text", "--mute")
    assert result.exit_code == 1
    assert "Phase 3" in result.output


def test_run_reports_invalid_config(invoke: Invoke, paths: AppPaths) -> None:
    invoke("config", "init")
    paths.config_file.write_text("[voice]\nvoice = 3\n")
    assert invoke("run").exit_code == 1


def test_run_announces_whisper_download(
    invoke: Invoke, services: Services, ready: FakeBackends, paths: AppPaths
) -> None:
    for item in whisper_dir(paths.cache_dir).rglob("model.bin"):
        item.unlink()
    services.lines = lambda: lines("q")
    assert "downloading the speech recognition model" in invoke("run").output


def test_run_uses_parakeet_on_apple_silicon(
    invoke: Invoke, services: Services, ready: FakeBackends, paths: AppPaths
) -> None:
    services.parakeet_ok = lambda: True
    install_models(paths, whisper=False, wake_word=False, parakeet=True)
    services.lines = lambda: lines("", "q")
    result = invoke("run")
    assert result.exit_code == 0, result.output
    assert ready.stt_engines == ["parakeet"]
    assert "Parakeet" not in result.output  # already downloaded: nothing to ask


def test_run_offers_to_download_parakeet(
    invoke: Invoke, services: Services, ready: FakeBackends
) -> None:
    services.parakeet_ok = lambda: True
    downloaded: list[str] = []
    services.downloader = recording_downloader(downloaded)
    services.lines = lambda: lines("q")
    declined = invoke("run", stdin="n\n")
    assert declined.exit_code == 0, declined.output
    assert "GPU (2.5 GB, one time" in declined.output
    assert "Using Whisper for speech recognition instead" in declined.output
    assert downloaded == []
    accepted = invoke("run", "--yes")
    assert accepted.exit_code == 0, accepted.output
    assert downloaded == [model.name for model in PARAKEET_FILES]
    assert ready.stt_engines == ["faster-whisper", "parakeet"]


def test_text_mode_needs_no_speech_recognition(
    invoke: Invoke, services: Services, ready: FakeBackends
) -> None:
    services.parakeet_ok = lambda: True
    services.lines = lambda: lines("q")
    result = invoke("run", "--text", "--mute")
    assert result.exit_code == 0, result.output
    assert "Parakeet" not in result.output


def test_models_download_fetches_parakeet(
    invoke: Invoke, services: Services, paths: AppPaths
) -> None:
    install_models(paths, whisper=False)
    services.parakeet_ok = lambda: True
    fakes = FakeBackends(FakeLLM())
    services.backends = fakes.build
    downloaded: list[str] = []
    services.downloader = recording_downloader(downloaded)
    assert invoke("models", "download", "--yes").exit_code == 0
    assert downloaded == [model.name for model in PARAKEET_FILES]
    assert not fakes.stt.loaded  # Whisper is not needed
    assert invoke("models", "download", stdin="n\n").exit_code == 1


def test_kokoro_download_flow(
    invoke: Invoke, services: Services, fake_keyring: FakeKeyring, paths: AppPaths
) -> None:
    fake_keyring.store[(KEYRING_SERVICE, "anthropic")] = VALID_KEY
    fakes = FakeBackends(FakeLLM(["Hi."]))
    services.backends = fakes.build
    services.lines = lambda: lines("q")
    downloaded: list[str] = []

    def fake_download(model: ModelFile, directory: Path, progress: Callable[[int], None]) -> Path:
        downloaded.append(model.name)
        progress(model.size)
        return directory / model.name

    services.downloader = fake_download
    declined = invoke("run", "--text", stdin="n\n")
    assert declined.exit_code == 1
    assert "cannot speak" in declined.output
    accepted = invoke("run", "--text", "--yes")
    assert accepted.exit_code == 0, accepted.output
    assert downloaded == [model.name for model in KOKORO_FILES]


def recording_downloader(
    downloaded: list[str],
) -> Callable[[ModelFile, Path, Callable[[int], None]], Path]:
    def fake_download(model: ModelFile, directory: Path, progress: Callable[[int], None]) -> Path:
        downloaded.append(model.name)
        progress(model.size)
        return directory / model.name

    return fake_download


def test_wake_word_download_flow(
    invoke: Invoke, services: Services, fake_keyring: FakeKeyring, paths: AppPaths
) -> None:
    fake_keyring.store[(KEYRING_SERVICE, "anthropic")] = VALID_KEY
    install_models(paths, wake_word=False)
    fakes = FakeBackends(FakeLLM())
    services.backends = fakes.build
    services.lines = lambda: lines("q")
    downloaded: list[str] = []
    services.downloader = recording_downloader(downloaded)

    typed = invoke("run", "--text")  # no microphone, so no wake word and no question
    assert typed.exit_code == 0, typed.output
    assert "wake word" not in typed.output
    skipped = invoke("run", "--no-wake-word")
    assert skipped.exit_code == 0 and "wake word" not in skipped.output
    declined = invoke("run", stdin="n\n")
    assert declined.exit_code == 0, declined.output
    assert "non-commercial" in declined.output  # the licence is stated before downloading
    assert "Continuing without the wake word" in declined.output
    assert not fakes.wake.loaded
    assert downloaded == []
    accepted = invoke("run", "--yes")
    assert accepted.exit_code == 0, accepted.output
    assert downloaded == [model.name for model in WAKE_WORD_FILES]
    assert fakes.wake.loaded


def test_models_download_includes_the_wake_word(
    invoke: Invoke, services: Services, paths: AppPaths
) -> None:
    install_models(paths, wake_word=False)
    services.backends = FakeBackends(FakeLLM()).build
    downloaded: list[str] = []
    services.downloader = recording_downloader(downloaded)
    assert invoke("models", "download", "--yes").exit_code == 0
    assert downloaded == [model.name for model in WAKE_WORD_FILES]

    downloaded.clear()
    for model in WAKE_WORD_FILES:
        (wake_word_dir(paths.cache_dir) / model.name).unlink(missing_ok=True)
    declined = invoke("models", "download", stdin="n\n")
    assert declined.exit_code == 1
    assert downloaded == []


def test_kokoro_download_failure(invoke: Invoke, services: Services) -> None:
    def failing(model: ModelFile, directory: Path, progress: Callable[[int], None]) -> Path:
        raise ModelIntegrityError("hash mismatch")

    services.downloader = failing
    result = invoke("models", "download", "--yes")
    assert result.exit_code == 1
    assert "hash mismatch" in result.output


def test_models_download(invoke: Invoke, services: Services, paths: AppPaths) -> None:
    install_models(paths, whisper=False)
    fakes = FakeBackends(FakeLLM())
    services.backends = fakes.build
    result = invoke("models", "download")
    assert result.exit_code == 0, result.output
    assert fakes.stt.loaded
    assert "ready" in result.output


def test_models_download_invalid_config(invoke: Invoke, paths: AppPaths) -> None:
    invoke("config", "init")
    paths.config_file.write_text("[nope]\n")
    assert invoke("models", "download").exit_code == 1


# --------------------------------------------------------------------------- console UI


def test_format_timings() -> None:
    timings = TurnTimings(
        speech_end=10.0, turn_detected=10.7, stt_started=10.7, transcribed=10.85,
        llm_started=10.85, first_token=11.2, first_sentence=11.3, first_audio=11.4,
    )  # fmt: skip
    line = format_timings(timings)
    assert line == (
        "1.40 s to first word  (end of turn 0.70 · speech-to-text 0.15 · Claude 0.35 · voice 0.10)"
    )
    assert format_timings(TurnTimings()) is None
    assert format_timings(TurnTimings(llm_started=1.0, first_token=1.5)) == "timings  (Claude 0.50)"


def test_console_ui_streams_and_escapes_markup() -> None:
    out = io.StringIO()
    ui = ConsoleUi(Console(file=out, width=120, color_system=None), name="Yarvis")
    ui.notice("Loading…")
    ui.ready(voice=True)
    ui.listening()
    ui.speech_started()
    ui.on_transcript("[bold]not markup[/]")
    ui.on_reply_delta("Hello ")
    ui.on_reply_delta("[there]")
    ui.reply_finished(
        TurnResult("x", "Hello [there]", TurnTimings(llm_started=1.0, first_token=1.2))
    )
    ui.ready(voice=False)
    ui.nothing_heard()
    ui.on_reply_delta("partial")
    ui.error("broken [pipe]")
    ui.ready(voice=True, wake_word=True)
    ui.woke()
    ui.on_reply_delta("cut")
    ui.interrupted()
    ui.said("Good [evening].")
    ui.alert("Your [pasta] timer is done.")
    ui.activity("set a [10]-minute timer")
    text = out.getvalue()
    for expected in (
        "Loading…", "Enter = talk", "listening", "You: [bold]not markup[/]",
        "Yarvis: Hello [there]", "Claude 0.20", "type a message", "Didn't catch that",
        "partial\n", "✗ broken [pipe]", "Say “Hey Yarvis” · Enter = talk", "cut\n",
        "↳ interrupted", "Yarvis: Good [evening].", "⏰ Your [pasta] timer is done.",
        "• set a [10]-minute timer",
    ):  # fmt: skip
        assert expected in text


async def test_stdin_lines(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO("one\ntwo\n"))
    assert [line async for line in stdin_lines()] == ["one\n", "two\n"]


def test_default_backends_factory_is_lazy() -> None:
    # Services.default() must not import heavy speech libraries.
    assert Services.default().backends == Backends.real


async def test_tools_through_the_real_wiring(tmp_path: Path) -> None:
    llm = FakeLLM([ToolCall("t1", "set_timer", {"minutes": 1})], ["Done."])
    fakes = FakeBackends(llm)
    ui = RecordingUi()
    await run_assistant(
        Config(),
        api_key=SecretStr("k"),
        cache_dir=tmp_path,
        lines=lines("set a timer for a minute", "q"),
        ui=ui,
        backends=fakes.build(),
        voice_input=False,
        voice_output=False,
    )
    assert "did:set a 1-minute timer" in ui.events
    assert "reply:Done." in ui.events
    system = llm.calls[0][1]
    assert "What you can do: set, list and cancel timers and reminders." in system
    assert llm.offered[0] == ["set_timer", "set_reminder", "list_timers", "cancel_timer"]


async def test_tools_can_be_switched_off(tmp_path: Path) -> None:
    llm = FakeLLM(["No can do."])
    fakes = FakeBackends(llm)
    config = Config.model_validate({"tools": {"timers": False, "computer": False}})
    await run_assistant(
        config,
        api_key=SecretStr("k"),
        cache_dir=tmp_path,
        lines=lines("set a timer", "q"),
        ui=RecordingUi(),
        backends=fakes.build(),
        voice_input=False,
        voice_output=False,
    )
    assert llm.offered == [[]]
    assert "You cannot act on the computer yet" in llm.calls[0][1]


async def test_computer_tools_are_described_to_the_model(tmp_path: Path) -> None:
    llm = FakeLLM(["Sure."])

    def mac(config: Config, spotify: SecretStr | None) -> list[Any]:
        return computer_tools(FakeRunner(), apps=Apps([tmp_path]), platform="darwin")

    def linux_tools(config: Config, spotify: SecretStr | None) -> list[Any]:
        return computer_tools(FakeRunner(), platform="linux")

    backends = replace(FakeBackends(llm).build(), computer=mac)
    await run_assistant(
        Config(),
        api_key=SecretStr("k"),
        cache_dir=tmp_path,
        lines=lines("hi", "q"),
        ui=RecordingUi(),
        backends=backends,
        voice_input=False,
        voice_output=False,
    )
    assert "open and quit apps, set the volume and play, pause or skip music" in llm.calls[0][1]
    assert "quit_app" in llm.offered[0]
    linux = replace(backends, computer=linux_tools)
    llm.replies.append(["Sure."])
    await run_assistant(
        Config(),
        api_key=SecretStr("k"),
        cache_dir=tmp_path,
        lines=lines("hi", "q"),
        ui=RecordingUi(),
        backends=linux,
        voice_input=False,
        voice_output=False,
    )


async def test_spotify_credentials_reach_the_computer_tools(tmp_path: Path) -> None:
    seen: list[SpotifyKeys | None] = []

    def computer(config: Config, spotify: SpotifyKeys | None) -> list[Any]:
        seen.append(spotify)
        return []

    backends = replace(FakeBackends(FakeLLM(["Hi."])).build(), computer=computer)
    await run_assistant(
        Config(),
        api_key=SecretStr("k"),
        cache_dir=tmp_path,
        lines=lines("hi", "q"),
        ui=RecordingUi(),
        backends=backends,
        voice_input=False,
        voice_output=False,
        spotify=SpotifyKeys(SecretStr("id:secret")),
    )
    assert [s.app.get_secret_value() for s in seen if s] == ["id:secret"]


def test_run_passes_stored_spotify_credentials(
    invoke: Invoke, services: Services, ready: FakeBackends, fake_keyring: FakeKeyring
) -> None:
    fake_keyring.store[(KEYRING_SERVICE, "spotify")] = "a" * 32 + ":" + "b" * 32
    seen: list[SpotifyKeys | None] = []
    real = ready.build

    def build() -> Backends:
        def computer(config: Config, spotify: SpotifyKeys | None) -> list[Any]:
            seen.append(spotify)
            return []

        return replace(real(), computer=computer)

    services.backends = build
    services.lines = lambda: lines("hello", "q")
    assert invoke("run", "--text", "--mute").exit_code == 0
    assert seen and seen[0] is not None
    assert seen[0].login is None and seen[0].save_login is None  # not logged in

    fake_keyring.store[(KEYRING_SERVICE, "spotify-login")] = "r" * 40
    seen.clear()
    services.lines = lambda: lines("q")
    assert invoke("run", "--text", "--mute").exit_code == 0
    keys = seen[0]
    assert keys is not None and keys.login is not None and keys.save_login is not None
    assert keys.login.get_secret_value() == "r" * 40
    keys.save_login("s" * 40)  # Spotify renewed it: the keychain keeps the new one
    assert fake_keyring.store[(KEYRING_SERVICE, "spotify-login")] == "s" * 40
