"""Load ``config.toml`` and turn validation problems into readable errors."""

from __future__ import annotations

import tomllib
from typing import TYPE_CHECKING, Final

from pydantic import ValidationError

from yarvis.config.models import Config
from yarvis.paths import write_private_file
from yarvis.security.redact import default_redactor

if TYPE_CHECKING:
    from pathlib import Path

    from yarvis.security.redact import Redactor

DEFAULT_CONFIG_TOML: Final = """\
# Yarvis configuration.
#
# Secrets never go in this file. Store your API key in the OS keychain with:
#     yarvis key set anthropic

[assistant]
name = "Yarvis"
language = "en"

[llm]
provider = "anthropic"      # "anthropic" or "ollama" (fully offline)
# model = "..."             # leave unset to use the recommended default
max_tokens = 1024

[voice]
stt = "auto"                # picks the fastest local engine for your platform
tts = "kokoro"
# input_device = "MacBook Air Microphone"
# output_device = "MacBook Air Speakers"

[security]
allow_tier3 = false         # dangerous tools (shell, delete, purchases): keep this off
confirm_tier2 = true        # confirm external actions (send message, create event)

[logging]
level = "INFO"
file = true
"""


class ConfigError(Exception):
    """The configuration file is missing, unreadable or invalid."""


def load_config(path: Path, *, redactor: Redactor = default_redactor) -> Config:
    """Load and validate the configuration at ``path``.

    A missing file is not an error: defaults are returned.

    Raises:
        ConfigError: If the file cannot be parsed, fails validation, or appears to contain
            a secret.
    """
    if not path.exists():
        return Config()
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        msg = f"cannot read {path}: {exc.strerror}"
        raise ConfigError(msg) from exc
    return parse_config(text, source=str(path), redactor=redactor)


def parse_config(
    text: str, *, source: str = "<string>", redactor: Redactor = default_redactor
) -> Config:
    """Parse and validate TOML configuration text."""
    if redactor.contains_secret(text):
        msg = (
            f"{source} appears to contain an API key or token. Remove it and store it in "
            "the OS keychain instead: yarvis key set <name>"
        )
        raise ConfigError(msg)
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        msg = f"{source} is not valid TOML: {exc}"
        raise ConfigError(msg) from exc
    try:
        return Config.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(_format_validation_error(source, exc)) from exc


def write_default_config(path: Path, *, overwrite: bool = False) -> None:
    """Write the commented default configuration with private permissions.

    Raises:
        FileExistsError: If ``path`` exists and ``overwrite`` is False.
    """
    write_private_file(path, DEFAULT_CONFIG_TOML, overwrite=overwrite)


def _format_validation_error(source: str, exc: ValidationError) -> str:
    lines = [f"{source} has {exc.error_count()} invalid setting(s):"]
    for error in exc.errors(include_url=False, include_input=False):
        location = ".".join(str(part) for part in error["loc"]) or "<root>"
        lines.append(f"  - {location}: {error['msg']}")
    return "\n".join(lines)
