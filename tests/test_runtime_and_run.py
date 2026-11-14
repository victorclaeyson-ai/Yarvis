from __future__ import annotations

import io
from typing import TYPE_CHECKING, Self

import pytest
from pydantic import SecretStr
from rich.console import Console

from fakes import VALID_KEY, FakeKeyring, FakeLLM, FakeSTT, FakeTTS, Invoke
from test_listen_and_assistant import RecordingUi, lines
from test_speech_engines import frames, speech_clip
from yarvis.cli.app import Services
from yarvis.cli.ui import ConsoleUi, format_timings, stdin_lines
from yarvis.config import Config
from yarvis.models import (
    KOKORO_FILES,
    WHISPER_REVISIONS,
    ModelFile,
    ModelIntegrityError,
    kokoro_dir,
    whisper_dir,
)
from yarvis.paths import is_private
from yarvis.pipeline import TurnResult, TurnTimings
from yarvis.runtime import Backends, run_assistant
from yarvis.security.secrets import KEYRING_SERVICE
from yarvis.speech.silero import SileroVAD

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
    def __init__(self) -> None:
        self.opened = 0

    async def __aenter__(self) -> Self:
        self.opened += 1
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        pass

    def __aiter__(self) -> AsyncIterator[AudioArray]:
        # The user asks one question; after that the microphone hears nothing (the stream
        # ends), so a follow-up listen returns without speech.
        speech = frames(speech_clip()) if self.opened == 1 else []

        async def gen() -> AsyncIterator[AudioArray]:
            for frame in speech:
                yield frame

        return gen()


class FakeSink:
    def __init__(self) -> None:
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
        self.keys: list[str] = []

    def build(self) -> Backends:
        def make_llm(config: Config, key: SecretStr) -> FakeLLM:
            self.keys.append(key.get_secret_value())
            return self.llm

        return Backends(
            llm=make_llm,
            stt=lambda config, cache: self.stt,
            tts=lambda config, cache: self.tts,
            vad=SileroVAD,
            microphone=lambda config: self.mic,
            speaker=lambda config, rate: self.sink,
        )


async def test_full_voice_turn_through_the_real_wiring(tmp_path: Path) -> None:
    fakes = FakeBackends(FakeLLM(["It is half past one."]))
    ui = RecordingUi()
    loading: list[str] = []
    await run_assistant(
        Config(),
        api_key=SecretStr("k"),
        cache_dir=tmp_path,
        lines=lines("", "q"),
        ui=ui,
        backends=fakes.build(),
        on_loading=loading.append,
    )
    assert fakes.stt.loaded and fakes.tts.loaded
    assert loading == ["Loading speech models…", "Warming up…"]
    assert fakes.mic.opened == 2  # the question, then the follow-up window (silence)
    warm_up, *real = fakes.stt.received
    assert warm_up == 8_000  # half a second of silence at startup
    assert len(real) == 1  # the speculation during the final pause, reused at end of turn
    assert real[0] > 2 * 16_000
    assert ui.events[:4] == ["ready:True", "listening", "speech", "you:hey yarvis what time is it"]
    assert "reply:It is half past one." in ui.events
    assert fakes.tts.spoken == ["Ready.", "It is half past one."]
    assert fakes.sink.played == 1 and fakes.sink.closed
    assert fakes.keys == ["k"]
    assert fakes.llm.warm_ups == 2  # at startup, and again when speech started


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


def install_models(paths: AppPaths, *, whisper: bool = True) -> None:
    paths.ensure_all()
    directory = kokoro_dir(paths.cache_dir)
    directory.mkdir(parents=True)
    for model in KOKORO_FILES:
        with (directory / model.name).open("wb") as handle:
            handle.truncate(model.size)  # sparse file with the right size
    if whisper:
        snapshot = (
            whisper_dir(paths.cache_dir) / "models--Systran--faster-whisper-base.en" / "snapshots"
        )
        (snapshot / WHISPER_REVISIONS["base.en"]).mkdir(parents=True)
        (snapshot / WHISPER_REVISIONS["base.en"] / "model.bin").write_bytes(b"x")


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
    assert "You: hey yarvis what time is it" in result.output
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


def test_kokoro_download_failure(invoke: Invoke, services: Services) -> None:
    def failing(model: ModelFile, directory: Path, progress: Callable[[int], None]) -> Path:
        raise ModelIntegrityError("SHA-256 mismatch")

    services.downloader = failing
    result = invoke("models", "download", "--yes")
    assert result.exit_code == 1
    assert "SHA-256 mismatch" in result.output


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
    text = out.getvalue()
    for expected in (
        "Loading…", "Enter = talk", "listening", "You: [bold]not markup[/]",
        "Yarvis: Hello [there]", "Claude 0.20", "type a message", "Didn't catch that",
        "partial\n", "✗ broken [pipe]",
    ):  # fmt: skip
        assert expected in text


async def test_stdin_lines(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO("one\ntwo\n"))
    assert [line async for line in stdin_lines()] == ["one\n", "two\n"]


def test_default_backends_factory_is_lazy() -> None:
    # Services.default() must not import heavy speech libraries.
    assert Services.default().backends == Backends.real
