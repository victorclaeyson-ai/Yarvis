"""Build the real components from the configuration and run the assistant.

Every external dependency is created through ``Backends``, so tests can swap in fakes and
run the full wiring without hardware, models or network.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, Self

import numpy as np

from yarvis.agent.conversation import Conversation
from yarvis.agent.greetings import Greeter
from yarvis.agent.prompts import system_prompt
from yarvis.agent.switching import THINK_HARDER, Switching
from yarvis.agent.toolbox import Toolbox
from yarvis.assistant import Assistant, AssistantControls, TurnTaking
from yarvis.audio.sounds import chime
from yarvis.ears import MicrophoneEars, SharedMicrophone, WakeWord, barge_in_mode
from yarvis.listen import AudioTap
from yarvis.llm.anthropic import DEFAULT_MODEL
from yarvis.pipeline import Turn
from yarvis.security.gate import Gate
from yarvis.speech.base import AudioSink, SpeechToText, TextToSpeech
from yarvis.speech.cartesia import CartesiaTTS, cloud_voice
from yarvis.speech.choice import describe as describe_stt
from yarvis.speech.endpointing import EndpointerSettings
from yarvis.speech.turn_taking import DEFAULT_WAKE_NAMES
from yarvis.telemetry import Reports, Setup, Telemetry, starting_links
from yarvis.tools.desktop import abilities
from yarvis.tools.memory import memory_ability, memory_tools
from yarvis.tools.skills import UseSkill
from yarvis.tools.timers import Alarms, timer_tools
from yarvis.tools.weather import Weather, watch_weather

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Sequence
    from contextlib import AbstractAsyncContextManager
    from pathlib import Path
    from types import TracebackType

    from pydantic import SecretStr

    from yarvis.assistant import AssistantUi
    from yarvis.config.models import Config
    from yarvis.ears import FrameSource
    from yarvis.llm.base import LLMProvider
    from yarvis.memory import MemoryBook
    from yarvis.speech.base import VoiceActivityDetector
    from yarvis.telemetry import Dashboard, Link
    from yarvis.tools.base import Tool
    from yarvis.tools.ducking import Ducker
    from yarvis.tools.spotify import SpotifyKeys
    from yarvis.usage import UsageBook


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


class McpTools(Protocol):
    """The tools of the running MCP servers (see ``tools.mcp.McpHub``)."""

    @property
    def tools(self) -> Sequence[Tool[Any]]:
        """The approved, allowlisted tools."""
        ...

    @property
    def servers(self) -> Sequence[str]:
        """The servers they come from."""
        ...

    @property
    def links(self) -> Sequence[Link]:
        """How every configured server is doing, for the dashboard."""
        ...


@dataclass(frozen=True, slots=True)
class _NoMcpTools:
    tools: tuple[Tool[Any], ...] = ()
    servers: tuple[str, ...] = ()
    links: tuple[Link, ...] = ()


@asynccontextmanager
async def no_mcp(_config: Config) -> AsyncGenerator[McpTools]:
    """No MCP servers (the default for tests)."""
    yield _NoMcpTools()


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
    personal: Callable[[Config], list[Tool[Any]]] = lambda _config: []
    """Calendars and email (on a Mac, through its own accounts)."""
    mcp: Callable[[Config], AbstractAsyncContextManager[McpTools]] = no_mcp
    """Starts the configured MCP servers for as long as Yarvis runs."""
    skills: Callable[[Config], list[Tool[Any]]] = lambda _config: []
    """The Agent Skills the user added."""
    profile: Callable[[], str] = lambda: ""
    """What the user wrote about themselves, read again before every request."""
    memory: Callable[[], MemoryBook | None] = lambda: None
    """The facts Yarvis remembers about the user (None: no memory)."""
    usage: Callable[[], UsageBook | None] = lambda: None
    """Where the day's numbers are kept (None: in memory only)."""

    @classmethod
    def real(cls) -> Backends:  # pragma: no cover - exercised on real machines, not in CI
        """The production implementations, imported lazily (they are heavy)."""
        from yarvis.audio.microphone import Microphone  # noqa: PLC0415
        from yarvis.audio.speaker import Speaker  # noqa: PLC0415
        from yarvis.llm.anthropic import AnthropicProvider  # noqa: PLC0415
        from yarvis.memory import MemoryBook  # noqa: PLC0415
        from yarvis.paths import AppPaths  # noqa: PLC0415
        from yarvis.profile import ProfileReader, profile_path  # noqa: PLC0415
        from yarvis.security.secrets import SecretStore  # noqa: PLC0415
        from yarvis.speech.choice import (  # noqa: PLC0415
            cloud_key,
            make_stt,
            parakeet_available,
            spoken_names,
        )
        from yarvis.speech.kokoro import KokoroTTS  # noqa: PLC0415
        from yarvis.speech.silero import SileroVAD  # noqa: PLC0415
        from yarvis.speech.wakeword import WakeWordDetector  # noqa: PLC0415
        from yarvis.tools.desktop import (  # noqa: PLC0415
            computer_tools,
            music_ducker,
            personal_tools,
        )
        from yarvis.tools.mcp import Approvals, McpHub  # noqa: PLC0415
        from yarvis.tools.mcp_launch import Launcher  # noqa: PLC0415
        from yarvis.tools.skills import SkillLibrary, user_skills  # noqa: PLC0415
        from yarvis.tools.spotify import SpotifyAccount, SpotifyCatalog  # noqa: PLC0415
        from yarvis.usage import UsageBook  # noqa: PLC0415

        paths = AppPaths.default()
        launcher = Launcher(
            SecretStore().get_extra, work_dir=paths.data_dir / "mcp", log_dir=paths.log_dir
        )
        return cls(
            llm=lambda config, key: AnthropicProvider(
                key, model=config.llm.model, web_search=config.tools.web_search
            ),
            stt=lambda config, cache: make_stt(
                config.voice,
                cache,
                parakeet_ok=parakeet_available(),
                cloud=cloud_key(config.voice, SecretStore()),
                terms=spoken_names(config),
            ),
            tts=lambda config, cache: cloud_voice(
                config.voice,
                KokoroTTS(
                    cache_dir=cache, voice=config.voice.voice, speed=config.voice.speech_rate
                ),
                SecretStore(),
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
            personal=personal_tools,
            mcp=lambda config: McpHub(config.mcp, Approvals.of(paths), launcher),
            skills=lambda config: (
                user_skills(SkillLibrary.of(paths)) if config.tools.skills else []
            ),
            profile=ProfileReader(profile_path(paths)),
            memory=lambda: MemoryBook.of(paths),
            usage=lambda: UsageBook.of(paths),
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
    dashboard: Dashboard | None = None,
) -> None:
    """Assemble the pipeline and run the interactive loop until the user quits.

    The day's usage is counted and kept whatever the interface; ``dashboard`` shows it.
    """
    llm, big_llm = _models(config, api_key, backends)
    announcements: asyncio.Queue[str] = asyncio.Queue()
    alarms = (
        Alarms(announcements.put_nowait, on_change=dashboard.timers if dashboard else None)
        if config.tools.timers
        else None
    )
    stt = backends.stt(config, cache_dir) if voice_input else None
    tts = backends.tts(config, cache_dir) if voice_output else None
    wake_word = (
        backends.wake_word(config, cache_dir) if voice_input and config.voice.wake_word else None
    )

    telemetry = _telemetry(backends, dashboard, config, llm, stt, tts, big_llm=big_llm)

    async with AsyncExitStack() as stack:
        stack.callback(telemetry.close)
        if alarms is not None:
            stack.callback(alarms.close)
        engines = [engine for engine in (stt, tts, wake_word) if engine is not None]
        mcp = await _load(stack, engines, backends.mcp(config), on_loading)
        memory = backends.memory()
        tools, abilities = _tools(config, alarms, backends, spotify, mcp, memory=memory)
        setup = Setup(
            model=config.llm.model or DEFAULT_MODEL,
            hearing=describe_stt(config.voice) if stt is not None else None,
            cloud_hearing=isinstance(stt, AudioTap),
            voice=config.voice.voice if tts is not None else None,
            cloud_voice=isinstance(tts, CartesiaTTS),
            wake_word=wake_word is not None,
            spotify=any(tool.name.startswith("spotify_") for tool in tools),
            spotify_login=spotify is not None and spotify.login is not None,
            personal=any(tool.name.startswith(("calendar_", "mail_")) for tool in tools),
            skills=sum(len(t.skill_names) for t in tools if isinstance(t, UseSkill)),
            profile=bool(backends.profile()),
            memory=len(memory.facts()) if memory is not None else None,
            mcp=mcp.links,
        )
        _show_dashboard(stack, telemetry, starting_links(config, setup), tools, dashboard)
        conversation = _conversation(
            config,
            llm,
            tools,
            abilities,
            profile=backends.profile,
            memory=memory,
            on_activity=ui.activity,
            switching=_switching(config, big_llm, telemetry),
        )
        if on_loading is not None:
            on_loading("Warming up…")
        await warm_up(llm, stt, tts, big_llm)

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
            on_finished=telemetry.turn_finished,
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
                tap=stt if isinstance(stt, AudioTap) else None,
            )
            _log_hearing(
                wake_word=wake_word is not None,
                barge_in=barge_in,
                speaker=speaker,
                cloud=isinstance(stt, AudioTap),
            )

        assistant = Assistant(
            turn=turn,
            lines=lines,
            ui=ui,
            ears=ears,
            turn_taking=TurnTaking(
                speculate=config.voice.speculative,
                thinking_pause_ms=config.voice.thinking_pause_ms,
                follow_up_s=config.voice.follow_up_s,
                stay_open_s=config.voice.stay_open_s,
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


async def _load(
    stack: AsyncExitStack,
    engines: Sequence[LoadableSTT | LoadableTTS | LoadableWakeWord],
    mcp: AbstractAsyncContextManager[McpTools],
    on_loading: Callable[[str], None] | None,
) -> McpTools:
    """Load the speech models and, meanwhile, start the MCP servers (kept on ``stack``)."""
    if engines and on_loading is not None:
        on_loading("Loading speech models…")
    loading = asyncio.gather(*(asyncio.to_thread(engine.load) for engine in engines))
    try:
        return await stack.enter_async_context(mcp)
    finally:
        await loading


def _conversation(
    config: Config,
    llm: LLMProvider,
    tools: list[Tool[Any]],
    abilities: list[str],
    *,
    profile: Callable[[], str],
    memory: MemoryBook | None,
    on_activity: Callable[[str], None],
    switching: Switching | None = None,
) -> Conversation:
    """The conversation with Claude, its tools behind the permission gate."""
    gate = Gate(
        confirm_external=config.security.confirm_tier2,
        allow_dangerous=config.security.allow_tier3,
    )
    return Conversation(
        llm,
        system_prompt=lambda: system_prompt(
            config.assistant.name,
            user_name=config.assistant.user_name,
            location=config.assistant.location,
            abilities=abilities,
            web_search=_web_search(config) and bool(tools),
            profile=profile(),
            memory=_remembered(memory),
        ),
        max_tokens=config.llm.max_tokens,
        toolbox=Toolbox(tools, gate, on_activity=on_activity) if tools else None,
        idle_reset_s=config.conversation.idle_reset_minutes * 60,
        switching=switching,
    )


def _models(
    config: Config, api_key: SecretStr, backends: Backends
) -> tuple[LLMProvider, LLMProvider | None]:
    """The voice model, and the bigger model for hard questions (None: switching is off)."""
    llm = backends.llm(config, api_key)
    if config.llm.switching == "off":
        return llm, None
    big = config.llm.model_copy(update={"model": config.llm.big_model})
    return llm, backends.llm(config.model_copy(update={"llm": big}), api_key)


def _switching(
    config: Config, big_llm: LLMProvider | None, telemetry: Telemetry
) -> Switching | None:
    """How hard questions reach the bigger model, within its daily budget (in code)."""
    if big_llm is None:
        return None
    cap = config.llm.big_model_daily_usd
    return Switching(
        big_llm,
        auto=config.llm.switching == "auto",
        within_budget=lambda: telemetry.big_model_usd_today() < cap,
    )


def _remembered(memory: MemoryBook | None) -> list[tuple[int, str]]:
    """The facts for the system prompt, newest first (read again only after a change)."""
    if memory is None:
        return []
    return [(fact.id, fact.text) for fact in reversed(memory.facts())]


def _tools(
    config: Config,
    alarms: Alarms | None,
    backends: Backends,
    spotify: SpotifyKeys | None,
    mcp: McpTools,
    *,
    memory: MemoryBook | None = None,
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
    if personal := backends.personal(config):
        tools += personal
        said += abilities(personal)
    if memory is not None:
        tools += memory_tools(memory, auto=config.memory.auto)
        said.append(memory_ability(auto=config.memory.auto))
    if config.tools.weather:
        tools.append(Weather(config.assistant.location))
        said.append("tell the weather, now and for the coming week")
    if skills := backends.skills(config):
        tools += skills
        said.append("follow the skills the user added (use_skill loads one when it applies)")
    if mcp.tools:
        tools += mcp.tools
        said.append(
            f"use the tools of {', '.join(mcp.servers)}, connected through MCP (each tool "
            "says what it does)"
        )
    if _web_search(config) and tools:  # it is offered alongside the other tools
        said.append(
            "search the web for anything current: news, sports results, prices, opening "
            "hours, facts you are unsure of"
        )
    if config.llm.switching == "auto" and tools:  # without tools the prompt says it can't act
        said.append(
            f"hand questions that need careful reasoning to a bigger model ({THINK_HARDER})"
        )
    return tools, said


def _telemetry(
    backends: Backends,
    dashboard: Dashboard | None,
    config: Config,
    *parts: LLMProvider | LoadableSTT | LoadableTTS | None,
    big_llm: LLMProvider | None = None,
) -> Telemetry:
    """Counts the day's usage; the cloud clients among ``parts`` report to it.

    The bigger model's client reports apart, so its spend counts towards its cap.
    """
    telemetry = Telemetry(
        backends.usage(),
        dashboard=dashboard,
        voice_usd_per_million=config.voice.cartesia_usd_per_million_chars,
    )
    for part in parts:
        if isinstance(part, Reports):
            part.report_to(telemetry)
    if isinstance(big_llm, Reports):
        big_llm.report_to(telemetry.for_bigger_model())
    return telemetry


def _show_dashboard(
    stack: AsyncExitStack,
    telemetry: Telemetry,
    links: Sequence[Link],
    tools: Sequence[Tool[Any]],
    dashboard: Dashboard | None,
) -> None:
    """Show the connections and today's numbers; keep the home weather fresh (if any)."""
    telemetry.start(links)
    weather = next((tool for tool in tools if isinstance(tool, Weather)), None)
    if dashboard is None or weather is None or not weather.home:
        return
    task = asyncio.create_task(watch_weather(weather, dashboard.weather))
    stack.callback(task.cancel)


def _web_search(config: Config) -> bool:
    """Claude's own web search: only with Anthropic, and only if switched on."""
    return config.tools.web_search and config.llm.provider == "anthropic"


def _log_failure(task: asyncio.Task[None]) -> None:
    if not task.cancelled() and (error := task.exception()) is not None:
        logger.warning("Could not prepare the greetings: %s", error)


def _log_hearing(*, wake_word: bool, barge_in: str, speaker: Sink | None, cloud: bool) -> None:
    if wake_word:
        logger.info(
            "Wake word on: listening for 'Hey Yarvis' on this device. The microphone stays "
            "open; audio is never stored, and nothing is sent until you speak to Yarvis."
        )
    if cloud:
        logger.info("Speech recognition: AssemblyAI while you talk, this computer as backup")
    device = speaker.device_name if speaker is not None else None
    logger.info("Barge-in: %s (output device: %s)", barge_in, device or "unknown")


WARM_UP_AUDIO_SAMPLES = 8_000  # half a second of silence


async def warm_up(
    llm: LLMProvider,
    stt: LoadableSTT | None,
    tts: LoadableTTS | None,
    big_llm: LLMProvider | None = None,
) -> None:
    """Run every engine once so the first real question is as fast as the rest.

    The first inference of an ONNX/CTranslate2 model and the first HTTPS request are much
    slower than later ones (memory allocation, kernel selection, TLS handshake).
    """
    jobs: list[Awaitable[object]] = [llm.warm_up()]
    if big_llm is not None:  # its first answer should not pay for a TLS handshake either
        jobs.append(big_llm.warm_up())
    if stt is not None:
        jobs.append(stt.transcribe(np.zeros(WARM_UP_AUDIO_SAMPLES, dtype=np.float32)))
    if tts is not None:
        jobs.append(tts.synthesize("Ready."))
    await asyncio.gather(*jobs)
