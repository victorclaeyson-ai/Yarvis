"""Build the real components from the configuration and run the assistant.

Every external dependency is created through ``Backends``, so tests can swap in fakes and
run the full wiring without hardware, models or network.
"""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, Self

from yarvis.agent.conversation import Conversation
from yarvis.agent.prompts import system_prompt
from yarvis.assistant import Assistant, AssistantControls
from yarvis.listen import listen
from yarvis.pipeline import Turn
from yarvis.speech.base import AudioSink, SpeechToText, TextToSpeech
from yarvis.speech.endpointing import EndpointerSettings

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable
    from pathlib import Path
    from types import TracebackType

    from pydantic import SecretStr

    from yarvis.assistant import AssistantUi
    from yarvis.config.models import Config
    from yarvis.listen import Utterance
    from yarvis.llm.base import LLMProvider
    from yarvis.speech.base import AudioArray, VoiceActivityDetector


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


class _AsyncContext(Protocol):
    async def __aenter__(self) -> Self: ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None: ...


class FrameSource(_AsyncContext, Protocol):
    """A microphone: used as ``async with``, then iterated for frames."""

    def __aiter__(self) -> AsyncIterator[AudioArray]:
        """Iterate frames."""
        ...


class Sink(_AsyncContext, AudioSink, Protocol):
    """A speaker: used as ``async with``, then given audio to play."""


@dataclass(frozen=True, slots=True)
class Backends:
    """Factories for every component that touches hardware, models or the network."""

    llm: Callable[[Config, SecretStr], LLMProvider]
    stt: Callable[[Config, Path], LoadableSTT]
    tts: Callable[[Config, Path], LoadableTTS]
    vad: Callable[[], VoiceActivityDetector]
    microphone: Callable[[Config], FrameSource]
    speaker: Callable[[Config, int], Sink]

    @classmethod
    def real(cls) -> Backends:  # pragma: no cover - exercised on real machines, not in CI
        """The production implementations, imported lazily (they are heavy)."""
        from yarvis.audio.microphone import Microphone  # noqa: PLC0415
        from yarvis.audio.speaker import Speaker  # noqa: PLC0415
        from yarvis.llm.anthropic import AnthropicProvider  # noqa: PLC0415
        from yarvis.speech.kokoro import KokoroTTS  # noqa: PLC0415
        from yarvis.speech.silero import SileroVAD  # noqa: PLC0415
        from yarvis.speech.whisper import WhisperSTT  # noqa: PLC0415

        return cls(
            llm=lambda config, key: AnthropicProvider(key, model=config.llm.model),
            stt=lambda config, cache: WhisperSTT(
                config.voice.stt_model, cache_dir=cache, vocabulary=config.voice.vocabulary
            ),
            tts=lambda config, cache: KokoroTTS(
                cache_dir=cache, voice=config.voice.voice, speed=config.voice.speech_rate
            ),
            vad=SileroVAD,
            microphone=lambda config: Microphone(device=config.voice.input_device),
            speaker=lambda config, rate: Speaker(
                sample_rate=rate, device=config.voice.output_device
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
) -> None:
    """Assemble the pipeline and run the interactive loop until the user quits."""
    conversation = Conversation(
        backends.llm(config, api_key),
        system_prompt=lambda: system_prompt(config.assistant.name),
        max_tokens=config.llm.max_tokens,
    )
    stt = backends.stt(config, cache_dir) if voice_input else None
    tts = backends.tts(config, cache_dir) if voice_output else None

    loading = [engine for engine in (stt, tts) if engine is not None]
    if loading:
        if on_loading is not None:
            on_loading("Loading speech models…")
        await asyncio.gather(*(asyncio.to_thread(engine.load) for engine in loading))

    async with AsyncExitStack() as stack:
        speaker = None
        if tts is not None:
            speaker = await stack.enter_async_context(backends.speaker(config, tts.sample_rate))
        turn = Turn(conversation, stt=stt, tts=tts, speaker=speaker, observer=ui)

        listener: Callable[[], Awaitable[Utterance | None]] | None = None
        if stt is not None:
            vad = backends.vad()
            settings = EndpointerSettings(end_of_turn_ms=config.voice.end_of_turn_ms)

            async def listen_once() -> Utterance | None:
                async with backends.microphone(config) as microphone:
                    return await listen(
                        aiter(microphone),
                        vad,
                        settings,
                        on_speech_start=ui.speech_started,
                        on_level=ui.level,
                    )

            listener = listen_once

        assistant = Assistant(turn=turn, lines=lines, ui=ui, listen=listener)
        if controls is not None:
            controls.attach(assistant)
        await assistant.run()
