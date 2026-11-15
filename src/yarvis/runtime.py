"""Build the real components from the configuration and run the assistant.

Every external dependency is created through ``Backends``, so tests can swap in fakes and
run the full wiring without hardware, models or network.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, Self

import numpy as np

from yarvis.agent.conversation import Conversation
from yarvis.agent.greetings import Greeter
from yarvis.agent.prompts import system_prompt
from yarvis.agent.toolbox import Toolbox
from yarvis.assistant import Assistant, AssistantControls, TurnTaking
from yarvis.audio.sounds import chime
from yarvis.ears import MicrophoneEars, SharedMicrophone, WakeWord, barge_in_mode
from yarvis.pipeline import Turn
from yarvis.security.gate import Gate
from yarvis.speech.base import AudioSink, SpeechToText, TextToSpeech
from yarvis.speech.endpointing import EndpointerSettings
from yarvis.speech.turn_taking import DEFAULT_WAKE_NAMES
from yarvis.tools.desktop import abilities
from yarvis.tools.timers import Alarms, timer_tools

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable
    from pathlib import Path
    from types import TracebackType

    from pydantic import SecretStr

    from yarvis.assistant import AssistantUi
    from yarvis.config.models import Config
    from yarvis.ears import FrameSource
    from yarvis.llm.base import LLMProvider
    from yarvis.speech.base import VoiceActivityDetector
    from yarvis.tools.base import Tool
    from yarvis.tools.ducking import Ducker
    from yarvis.tools.spotify import SpotifyKeys


class LoadableSTT(SpeechToText, Protocol):
    """Speech-to-text that can be loaded ahead of the first turn."""

    def load(self) -> None:
        """Load the model (blocking)."""
        ...


class LoadableTTS(TextToSpeech, Protocol):
    """Text-to-speech that can be loaded ahead of the first turn."""

    def load(self) -> None:
        """Load the model (blocking)."""
        ...


class LoadableWakeWord(WakeWord, Protocol):
    """Wake-word detector that can be loaded ahead of time."""

    def load(self) -> None:
        """Load the model (blocking)."""
        ...


class Sink(AudioSink, Protocol):
    """A speaker: used as ``async with``, then given audio to play."""

    device_name: str | None
    """The output device actually used (decides how barge-in works)."""

    async def __aenter__(self) -> Self: ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class Backends:
    """Factories for every component that touches hardware, models or the network."""

    llm: Callable[[Config, SecretStr], LLMProvider]
    stt: Callable[[Config, Path], LoadableSTT]
    tts: Callable[[Config, Path], LoadableTTS]
    vad: Callable[[], VoiceActivityDetector]
    microphone: Callable[[Config], FrameSource]
    speaker: Callable[[Config, int], Sink]
    wake_word: Callable[[Config, Path], LoadableWakeWord]
    computer: Callable[[Config, SpotifyKeys | None], list[Tool[Any]]]
    """The tools that act on this computer (given the Spotify keys, if any)."""
    ducker: Callable[[Config], Ducker | None] = lambda _config: None
    """Turns music down during conversations (None: nothing to turn down)."""

    @classmethod
    def real(cls) -> Backends:  # pragma: no cover - exercised on real machines, not in CI
        """The production implementations, imported lazily (they are heavy)."""
        from yarvis.audio.microphone import Microphone  # noqa: PLC0415
        from yarvis.audio.speaker import Speaker  # noqa: PLC0415
        from yarvis.llm.anthropic import AnthropicProvider  # noqa: PLC0415
        from yarvis.speech.choice import make_stt, parakeet_available  # noqa: PLC0415
        from yarvis.speech.kokoro import KokoroTTS  # noqa: PLC0415
        from yarvis.speech.silero import SileroVAD  # noqa: PLC0415
        from yarvis.speech.wakeword import WakeWordDetector  # noqa: PLC0415
        from yarvis.tools.desktop import computer_tools, music_ducker  # noqa: PLC0415
        from yarvis.tools.spotify import SpotifyAccount, SpotifyCatalog  # noqa: PLC0415

        return cls(
            llm=lambda config, key: AnthropicProvider(key, model=config.llm.model),
            stt=lambda config, cache: make_stt(
                config.voice, cache, parakeet_ok=parakeet_available()
            ),
            tts=lambda config, cache: KokoroTTS(
                cache_dir=cache, voice=config.voice.voice, speed=config.voice.speech_rate
            ),
            vad=SileroVAD,
            microphone=lambda config: Microphone(device=config.voice.input_device),
            speaker=lambda config, rate: Speaker(
                sample_rate=rate, device=config.voice.output_device
            ),
            wake_word=lambda config, cache: WakeWordDetector(
                cache_dir=cache, threshold=config.voice.wake_word_threshold
            ),
            ducker=music_ducker,
            computer=lambda config, spotify: computer_tools(
                browser=config.tools.browser,
                typing=config.tools.typing,
                spotify=SpotifyCatalog.from_secret(spotify.app) if spotify else None,
                spotify_account=SpotifyAccount.from_keys(spotify) if spotify else None,
            ),
        )


async def run_assistant(
    config: Config,
    *,
    api_key: SecretStr,
    cache_dir: Path,
    lines: AsyncIterator[str],
    ui: AssistantUi,
    backends: Backends,
    voice_input: bool = True,
    voice_output: bool = True,
    on_loading: Callable[[str], None] | None = None,
    controls: AssistantControls | None = None,
    spotify: SpotifyKeys | None = None,
) -> None:
    """Assemble the pipeline and run the interactive loop until the user quits."""
    llm = backends.llm(config, api_key)
    announcements: asyncio.Queue[str] = asyncio.Queue()
    alarms = Alarms(announcements.put_nowait) if config.tools.timers else None
    tools, abilities = _tools(config, alarms, backends, spotify)
    gate = Gate(
        confirm_external=config.security.confirm_tier2,
        allow_dangerous=config.security.allow_tier3,
    )
    conversation = Conversation(
        llm,
        system_prompt=lambda: system_prompt(
            config.assistant.name, user_name=config.assistant.user_name, abilities=abilities
        ),
        max_tokens=config.llm.max_tokens,
        toolbox=Toolbox(tools, gate, on_activity=ui.activity) if tools else None,
    )
    stt = backends.stt(config, cache_dir) if voice_input else None
    tts = backends.tts(config, cache_dir) if voice_output else None
    wake_word = (
        backends.wake_word(config, cache_dir) if voice_input and config.voice.wake_word else None
    )

    loading = [engine for engine in (stt, tts, wake_word) if engine is not None]
    if loading:
        if on_loading is not None:
            on_loading("Loading speech models…")
        await asyncio.gather(*(asyncio.to_thread(engine.load) for engine in loading))
    if on_loading is not None:
        on_loading("Warming up…")
    await warm_up(llm, stt, tts)

    async with AsyncExitStack() as stack:
        if alarms is not None:
            stack.callback(alarms.close)
        speaker = None
        if tts is not None:
            speaker = await stack.enter_async_context(backends.speaker(config, tts.sample_rate))
        turn = Turn(
            conversation,
            stt=stt,
            tts=tts,
            speaker=speaker,
            observer=ui,
            chime=chime(tts.sample_rate) if tts is not None and config.voice.wake_chime else None,
            names=(config.assistant.name, *DEFAULT_WAKE_NAMES),
        )

        greeter = Greeter(config.assistant.user_name) if wake_word is not None else None
        if greeter is not None:
            # Ready in a second or two, so "Hey Yarvis" is answered without delay.
            preparing = asyncio.create_task(turn.prepare(greeter.phrases()))
            preparing.add_done_callback(_log_failure)
            stack.callback(preparing.cancel)

        ears: MicrophoneEars | None = None
        if stt is not None:
            settings = EndpointerSettings(end_of_turn_ms=config.voice.end_of_turn_ms)

            background: set[asyncio.Task[None]] = set()

            def speech_started() -> None:
                ui.speech_started()
                # The user is talking: make sure the connection to Claude is hot by the time
                # the transcript is ready.
                task = asyncio.create_task(llm.warm_up())
                background.add(task)
                task.add_done_callback(background.discard)

            barge_in = (
                barge_in_mode(
                    config.voice.barge_in,
                    output_device=speaker.device_name,
                    wake_word=wake_word is not None,
                )
                if speaker is not None
                else "off"  # nothing is spoken, so there is nothing to interrupt
            )
            microphone = await stack.enter_async_context(
                SharedMicrophone(
                    lambda: backends.microphone(config), always_on=wake_word is not None
                )
            )
            ears = MicrophoneEars(
                microphone,
                vad=backends.vad(),
                settings=settings,
                wake_word=wake_word,
                barge_in=barge_in,
                barge_in_vad=backends.vad() if barge_in == "voice" else None,
                on_speech_start=speech_started,
                on_level=ui.level,
            )
            _log_hearing(wake_word=wake_word is not None, barge_in=barge_in, speaker=speaker)

        assistant = Assistant(
            turn=turn,
            lines=lines,
            ui=ui,
            ears=ears,
            turn_taking=TurnTaking(
                speculate=config.voice.speculative,
                thinking_pause_ms=config.voice.thinking_pause_ms,
                follow_up_s=config.voice.follow_up_s,
            ),
            greeter=greeter,
            announcements=announcements,
            ducker=backends.ducker(config) if voice_input and config.voice.duck_music else None,
        )
        if conversation.toolbox is not None:
            conversation.toolbox.on_playback = assistant.media_started
        if controls is not None:
            controls.attach(assistant)
        await assistant.run()


def _tools(
    config: Config, alarms: Alarms | None, backends: Backends, spotify: SpotifyKeys | None
) -> tuple[list[Tool[Any]], list[str]]:
    """The enabled tools, and what they let Yarvis do (in words, for the system prompt)."""
    tools: list[Tool[Any]] = []
    said: list[str] = []
    if alarms is not None:
        tools += timer_tools(alarms)
        said.append("set, list and cancel timers and reminders")
    if config.tools.computer and (computer := backends.computer(config, spotify)):
        tools += computer
        said += abilities(computer)
    return tools, said


def _log_failure(task: asyncio.Task[None]) -> None:
    if not task.cancelled() and (error := task.exception()) is not None:
        logger.warning("Could not prepare the greetings: %s", error)


def _log_hearing(*, wake_word: bool, barge_in: str, speaker: Sink | None) -> None:
    if wake_word:
        logger.info(
            "Wake word on: listening for 'Hey Yarvis' on this device. The microphone stays "
            "open; audio is never stored or sent."
        )
    device = speaker.device_name if speaker is not None else None
    logger.info("Barge-in: %s (output device: %s)", barge_in, device or "unknown")


WARM_UP_AUDIO_SAMPLES = 8_000  # half a second of silence


async def warm_up(llm: LLMProvider, stt: LoadableSTT | None, tts: LoadableTTS | None) -> None:
    """Run every engine once so the first real question is as fast as the rest.

    The first inference of an ONNX/CTranslate2 model and the first HTTPS request are much
    slower than later ones (memory allocation, kernel selection, TLS handshake).
    """
    jobs: list[Awaitable[object]] = [llm.warm_up()]
    if stt is not None:
        jobs.append(stt.transcribe(np.zeros(WARM_UP_AUDIO_SAMPLES, dtype=np.float32)))
    if tts is not None:
        jobs.append(tts.synthesize("Ready."))
    await asyncio.gather(*jobs)
