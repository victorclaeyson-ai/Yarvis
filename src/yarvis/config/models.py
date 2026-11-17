"""Typed, validated configuration models.

All models are immutable and reject unknown keys, so a typo in ``config.toml`` is an error
instead of a silently ignored setting.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from yarvis.tools.browser import BrowserName  # noqa: TC001 - pydantic needs it at runtime

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class AssistantConfig(_Section):
    """Identity and language of the assistant."""

    name: str = Field(default="Yarvis", min_length=1, max_length=32)
    user_name: str | None = Field(
        default=None,
        min_length=1,
        max_length=40,
        pattern=r"^[^\W\d_][^\W\d_ .'-]*(?:[ .'-][^\W\d_]+)*$",
        description="What Yarvis calls you, e.g. in greetings. Stays on this computer.",
    )
    location: str | None = Field(
        default=None,
        min_length=2,
        max_length=80,
        pattern=r"^[^\W\d_][\w .,'()-]*$",
        description=(
            "Where you live, e.g. 'Leuven, Belgium': the weather and searches default to it."
        ),
    )
    language: str = Field(default="en", pattern=r"^[a-z]{2}(-[A-Z]{2})?$")


class LLMConfig(_Section):
    """Which language model backs the assistant."""

    provider: Literal["anthropic", "ollama"] = "anthropic"
    model: str | None = Field(
        default=None,
        min_length=1,
        description="Model id; None selects the provider's recommended default.",
    )
    max_tokens: int = Field(default=1024, ge=1, le=8192)
    ollama_url: str = Field(default="http://127.0.0.1:11434", pattern=r"^https?://")


SttModel = Literal["tiny.en", "base.en", "small.en"]
BargeIn = Literal["auto", "voice", "wake_word", "off"]
VocabularyWord = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=40)
]


class VoiceConfig(_Section):
    """Audio devices and speech backends.

    Only implemented backends are accepted; more (Piper) arrive in later phases.
    """

    cloud_stt: bool = Field(
        default=True,
        description=(
            "Recognize speech with AssemblyAI (the most accurate) when its API key is stored. "
            "Only what you say after the wake word is sent; the engine below is the backup."
        ),
    )
    cloud_region: Literal["eu", "global"] = Field(
        default="eu",
        description=(
            "Where AssemblyAI processes your speech. 'eu': servers in the EU only, and never "
            "used to train models; 'global': the nearest servers."
        ),
    )
    stt: Literal["auto", "parakeet", "faster-whisper"] = Field(
        default="auto",
        description=(
            "Speech recognition on this computer (and the backup for AssemblyAI). 'auto': "
            "Parakeet, on the GPU of Apple Silicon Macs and on the processor elsewhere; "
            "Whisper if it cannot run."
        ),
    )
    stt_model: SttModel = Field(default="small.en", description="The Whisper model.")
    stt_device: Literal["cpu", "mps"] = Field(
        default="cpu",
        description="'mps' runs Whisper on the Apple GPU (experimental; try `yarvis bench`).",
    )
    vocabulary: tuple[VocabularyWord, ...] = Field(
        default=("Yarvis",),
        max_length=50,
        description=(
            "Names and places to spell correctly (AssemblyAI and Whisper; Parakeet takes no hints)."
        ),
    )
    tts: Literal["kokoro"] = "kokoro"
    voice: str = Field(default="bm_george", pattern=r"^[a-z]{2}_[a-z]+$")
    speech_rate: float = Field(default=1.0, ge=0.5, le=2.0)
    end_of_turn_ms: int = Field(
        default=900,
        ge=200,
        le=3000,
        description="Silence that ends a sentence that sounds finished.",
    )
    thinking_pause_ms: int = Field(
        default=2000,
        ge=500,
        le=6000,
        description="Silence allowed when your sentence sounds unfinished ('and…', 'um').",
    )
    follow_up_s: float = Field(
        default=15.0,
        ge=0,
        le=60,
        description=(
            "After Yarvis answers, anything you say this long afterwards goes to Yarvis, "
            "without the wake word. Then say 'Hey Yarvis' again."
        ),
    )
    stay_open_s: float = Field(
        default=0.0,
        ge=0,
        le=3600,
        description=(
            "Off by default (0). Keep listening this long after follow_up_s, but only for "
            "requests that sound meant for Yarvis (its name, or a command such as 'open "
            "Spotify'); other talk is dropped on this computer and never reaches Claude. It "
            "can miss a real request that does not sound like one."
        ),
    )
    speculative: bool = Field(
        default=True,
        description="Start transcribing and asking Claude during pauses (much faster replies).",
    )
    wake_word: bool = Field(
        default=True,
        description=(
            "Wake up on 'Hey Yarvis'. Detection runs on this computer and no audio leaves it "
            "until you say it, but the microphone stays open while Yarvis runs."
        ),
    )
    wake_word_threshold: float = Field(
        default=0.5,
        ge=0.1,
        le=0.95,
        description="Higher means fewer accidental wake-ups; lower means it wakes more easily.",
    )
    wake_chime: bool = Field(default=True, description="Play a short sound when woken.")
    duck_music: bool = Field(
        default=True,
        description="Turn Spotify down while you talk to Yarvis, so it hears you, not the song.",
    )
    barge_in: BargeIn = Field(
        default="auto",
        description=(
            "Interrupting Yarvis while it talks. 'voice': just start talking (needs headphones, "
            "or it hears itself). 'wake_word': say 'Hey Yarvis' (works on speakers). 'auto' "
            "picks by output device."
        ),
    )
    input_device: str | int | None = None
    output_device: str | int | None = None

    @model_validator(mode="after")
    def _thinking_pause_is_longer(self) -> VoiceConfig:
        if self.thinking_pause_ms < self.end_of_turn_ms:
            msg = "thinking_pause_ms must be at least end_of_turn_ms"
            raise ValueError(msg)
        return self


class ToolsConfig(_Section):
    """What Yarvis may do. Every tool also has a permission tier (design doc §5.3)."""

    timers: bool = Field(default=True, description="Timers and reminders, said out loud.")
    computer: bool = Field(
        default=True,
        description="Apps, websites and tabs, volume, music and Spotify (macOS).",
    )
    browser: BrowserName | Literal["auto"] = Field(
        default="auto", description="Browser to control; 'auto' = the one that is open."
    )
    typing: bool = Field(
        default=True,
        description="Type into apps (pressing Enter always needs your confirmation).",
    )
    web_search: bool = Field(
        default=True,
        description=(
            "Let Claude search the web for current information (Anthropic's web search, "
            "about 1 cent per search)."
        ),
    )
    weather: bool = Field(
        default=True, description="Weather forecasts from Open-Meteo (free, no account)."
    )
    calendar: bool = Field(
        default=True,
        description="Read your calendars and add events, through the Mac's accounts (macOS).",
    )
    mail: bool = Field(
        default=True,
        description=(
            "Read your email and send it, through the Mac's Mail app (macOS). Sending always "
            "needs your confirmation."
        ),
    )


class SecurityConfig(_Section):
    """Security policy. Defaults are the safe choice; see design doc §5.3."""

    allow_tier3: bool = Field(
        default=False,
        description="Allow dangerous tools (shell, delete, purchases). Strongly discouraged.",
    )
    confirm_tier2: bool = Field(
        default=True,
        description="Require confirmation for external or hard-to-undo actions.",
    )


class LoggingConfig(_Section):
    """Log verbosity and destinations."""

    level: LogLevel = "INFO"
    file: bool = True


class Config(_Section):
    """Root configuration object."""

    assistant: AssistantConfig = AssistantConfig()
    llm: LLMConfig = LLMConfig()
    voice: VoiceConfig = VoiceConfig()
    tools: ToolsConfig = ToolsConfig()
    security: SecurityConfig = SecurityConfig()
    logging: LoggingConfig = LoggingConfig()

    def security_relaxations(self) -> list[str]:
        """Human-readable list of every setting that weakens the default security policy."""
        relaxations: list[str] = []
        if self.security.allow_tier3:
            relaxations.append("security.allow_tier3 = true (dangerous tools enabled)")
        if not self.security.confirm_tier2:
            relaxations.append(
                "security.confirm_tier2 = false (no confirmation for external actions)"
            )
        if (
            self.llm.provider == "ollama"
            and self.llm.ollama_url.startswith("http://")
            and not (_is_loopback_url(self.llm.ollama_url))
        ):
            relaxations.append("llm.ollama_url uses unencrypted HTTP to a non-local host")
        return relaxations


def _is_loopback_url(url: str) -> bool:
    host = url.split("://", 1)[1].split("/", 1)[0].rsplit(":", 1)[0].strip("[]")
    return host in {"localhost", "127.0.0.1", "::1"}
