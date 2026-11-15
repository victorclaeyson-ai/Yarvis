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

    stt: Literal["auto", "parakeet", "faster-whisper"] = Field(
        default="auto",
        description=(
            "Speech recognition. 'auto': Parakeet on the GPU of Apple Silicon Macs (the most "
            "accurate), Whisper elsewhere."
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
        description="Names and places Whisper should spell correctly (Parakeet needs no hints).",
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
            "When Yarvis asks you something, anything you say this long afterwards is your "
            "answer. Without stay_open_s, this is how long a conversation stays open."
        ),
    )
    stay_open_s: float = Field(
        default=300.0,
        ge=0,
        le=3600,
        description=(
            "After your last request, keep listening this long for requests meant for Yarvis "
            "(its name, or a command such as 'open Spotify'). Everything else, like talking "
            "to someone else, is ignored on this computer and never sent. 0 = off. Say "
            "'thanks' or 'that's all' to end it sooner."
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
