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
# user_name = "Victor"      # what Yarvis calls you ("Good evening, Victor.")
# location = "Leuven, Belgium"  # where you live: the weather and searches default to it
language = "en"

[llm]
provider = "anthropic"      # "anthropic" ("ollama" for fully offline arrives in Phase 3)
# model = "claude-sonnet-5-5"  # default: claude-haiku-5-5, the fastest
max_tokens = 1024

[voice]
cloud_stt = true            # AssemblyAI, if you stored its key (`yarvis key set assemblyai`)
cloud_region = "eu"         # AssemblyAI servers: "eu" (EU only, never used for training), "global"
stt = "auto"                # On this computer (and backup): Parakeet, or Whisper if it cannot run
stt_model = "small.en"      # Whisper: "tiny.en" (fastest), "base.en", "small.en" (most accurate)
stt_device = "cpu"          # Whisper on "mps" = Apple GPU (experimental); see `yarvis bench`
vocabulary = ["Yarvis"]     # Names to spell right, e.g. ["Yarvis", "Zedelgem", "Leuven"]
tts = "kokoro"
voice = "bm_george"         # British male. Try "af_heart", "am_michael", "bf_emma"
speech_rate = 1.0
end_of_turn_ms = 900        # silence that ends a sentence that sounds finished
thinking_pause_ms = 2000    # ...and one that sounds unfinished ("and", "um", "the")
follow_up_s = 15            # after an answer, anything you say this long goes to Yarvis
stay_open_s = 0             # then keep listening this long, only for what sounds meant for
                            # Yarvis ("open Spotify", "Yarvis, ..."); 0 = off, say "Hey Yarvis"
speculative = true          # start working during pauses: much faster, costs a few extra tokens
wake_word = true            # "Hey Yarvis", detected on this computer (the mic stays open)
wake_word_threshold = 0.5   # higher = fewer accidental wake-ups, lower = wakes more easily
wake_chime = true           # short sound when Yarvis wakes up
duck_music = true           # turn Spotify down while you talk to Yarvis
barge_in = "auto"           # interrupt it mid-sentence: "voice" (headphones), "wake_word",
                            # "auto" (voice on headphones, wake word on speakers) or "off"
# input_device = "MacBook Air Microphone"
# output_device = "MacBook Air Speakers"

[tools]
timers = true               # timers and reminders, announced out loud
computer = true             # apps, websites and tabs, volume, music and Spotify (macOS)
browser = "auto"            # or "Brave Browser", "Google Chrome", "Safari", ...
typing = true               # type into apps; pressing Enter always asks you first
web_search = true           # Claude searches the web for current info (~1 cent per search)
weather = true              # forecasts from Open-Meteo (free, no account)
calendar = true             # your calendars, via System Settings → Internet Accounts (macOS)
mail = true                 # your email, via the Mail app; sending always asks you first
skills = true               # Agent Skills you added (yarvis skills add <folder>)

# MCP servers: only the tools you list are used, each with a tier (0 read only,
# 1 reversible, 2 asks you first, 3 never). Then run: yarvis mcp approve <name>
# [[mcp.servers]]
# name = "files"
# command = ["npx", "-y", "@modelcontextprotocol/server-filesystem@<version>", "/Users/me/Notes"]
# secrets = []              # filled from the keychain: yarvis mcp secret <name> <SECRET>
#   [mcp.servers.tools]
#   read_text_file = 0
#   write_file = 2

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
