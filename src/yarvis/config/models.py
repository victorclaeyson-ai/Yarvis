"""Typed, validated configuration models.

All models are immutable and reject unknown keys, so a typo in ``config.toml`` is an error
instead of a silently ignored setting.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class AssistantConfig(_Section):
    """Identity and language of the assistant."""

    name: str = Field(default="Yarvis", min_length=1, max_length=32)
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
VocabularyWord = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=40)
]


class VoiceConfig(_Section):
    """Audio devices and speech backends.

    Only implemented backends are accepted; more (MLX Whisper, Piper) arrive in later phases.
    """

    stt: Literal["auto", "faster-whisper"] = "auto"
    stt_model: SttModel = "base.en"
    stt_device: Literal["cpu", "mps"] = Field(
        default="cpu",
        description="'mps' runs Whisper on the Apple GPU (experimental; try `yarvis bench`).",
    )
    vocabulary: tuple[VocabularyWord, ...] = Field(
        default=("Yarvis",),
        max_length=50,
        description="Names and places speech recognition should spell correctly.",
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
        default=5.0,
        ge=0,
        le=30,
        description="After a spoken reply, keep listening this long (0 = off).",
    )
    speculative: bool = Field(
        default=True,
        description="Start transcribing and asking Claude during pauses (much faster replies).",
    )
    input_device: str | int | None = None
    output_device: str | int | None = None

    @model_validator(mode="after")
    def _thinking_pause_is_longer(self) -> VoiceConfig:
        if self.thinking_pause_ms < self.end_of_turn_ms:
            msg = "thinking_pause_ms must be at least end_of_turn_ms"
            raise ValueError(msg)
        return self


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
