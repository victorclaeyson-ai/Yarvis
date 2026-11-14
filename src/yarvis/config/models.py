"""Typed, validated configuration models.

All models are immutable and reject unknown keys, so a typo in ``config.toml`` is an error
instead of a silently ignored setting.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

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


class VoiceConfig(_Section):
    """Audio devices and speech backends."""

    stt: Literal["auto", "mlx-whisper", "faster-whisper"] = "auto"
    tts: Literal["kokoro", "piper"] = "kokoro"
    voice: str | None = None
    input_device: str | int | None = None
    output_device: str | int | None = None


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
