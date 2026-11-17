"""The user's own description of themselves: ``profile.md``, known at every request.

Like a ``CLAUDE.md`` for a voice assistant (ADR 0014): who you are, what you are working
on, how you like to be spoken to, and your own rules. You write it; Yarvis reads it again
before every request, so an edit counts from your next question, no restart needed.

What it is not:

* **Not a way around the gate.** The prompt says so, and it does not matter anyway: the
  confirmations and refusals are decided by code (design doc §5.3), not by the model.
* **Not a place for secrets.** It is sent to the language model with every request, so a
  profile that looks like it holds an API key or token is not used at all.
* **Not unbounded.** It costs tokens on every request: past ``MAX_PROFILE_CHARS`` it is
  not used either (``yarvis doctor`` and ``yarvis profile show`` say why).

``<!-- comments -->`` and the template's empty bullet points are left out of what is sent.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Final

from yarvis.paths import write_private_file
from yarvis.security.redact import Redactor, default_redactor

if TYPE_CHECKING:
    from pathlib import Path

    from yarvis.paths import AppPaths

logger = logging.getLogger(__name__)

PROFILE_FILE: Final = "profile.md"
MAX_PROFILE_CHARS: Final = 6_000
"""About 1,500 tokens: a page. Sent with every request, so it should stay short."""

TEMPLATE: Final = """\
# About me

<!-- Yarvis knows this at every question. Keep it to about a page: it is sent to Claude
     with each request. Never put passwords or keys in it.
     Text inside these comment brackets is not sent. -->

## Who I am
-

## What I'm working on (most important first)
1.

## How to talk to me
-

## My rules
<!-- For example: "Always read an email back to me before sending it." Yarvis's own
     safety checks always apply on top of these, whatever this file says. -->
-
"""

_BOM: Final = "\N{ZERO WIDTH NO-BREAK SPACE}"
_COMMENT: Final = re.compile(r"<!--.*?-->", re.DOTALL)
_PLACEHOLDER: Final = re.compile(r"^\s*(?:[-*+]|\d+[.)])?\s*$")
_HEADING: Final = re.compile(r"^\s*(#{1,6})\s")


class ProfileError(Exception):
    """The profile cannot be used (the message says why and what to do)."""


def profile_path(paths: AppPaths) -> Path:
    """Where the profile lives: next to ``config.toml``."""
    return paths.config_dir / PROFILE_FILE


def create_profile(path: Path) -> bool:
    """Write the template if there is no profile yet. Returns True if it was created."""
    if path.exists():
        return False
    write_private_file(path, TEMPLATE)
    return True


def profile_text(raw: str) -> str:
    """What is sent: no comments, no empty template lines, no headings without content."""
    lines = [line.rstrip() for line in _COMMENT.sub("", raw.lstrip(_BOM)).splitlines()]
    lines = [line for line in lines if line == "" or not _PLACEHOLDER.match(line)]
    lines = [line for index, line in enumerate(lines) if not _empty_section(lines, index)]
    kept: list[str] = []
    for line in lines:
        if line or (kept and kept[-1]):  # at most one blank line in a row
            kept.append(line)
    return "\n".join(kept).strip()


def _empty_section(lines: list[str], index: int) -> bool:
    """Whether ``lines[index]`` is a heading with nothing written under it."""
    level = _level(lines[index])
    if not level:
        return False
    for line in lines[index + 1 :]:
        if (other := _level(line)) and other <= level:
            return True
        if line and not other:
            return False
    return True


def _level(line: str) -> int:
    """The level of a Markdown heading (``## Title`` is 2), or 0."""
    match = _HEADING.match(line)
    return len(match.group(1)) if match else 0


def load_profile(path: Path, *, redactor: Redactor = default_redactor) -> str:
    """The profile as it is sent ("" if there is none, or nothing is written in it yet).

    Raises:
        ProfileError: If it cannot be used: unreadable, too long, or holding a secret.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""
    except UnicodeDecodeError:
        msg = f"{path.name} must be saved as UTF-8 text"
        raise ProfileError(msg) from None
    except OSError as exc:
        msg = f"cannot read {path.name}: {exc.strerror}"
        raise ProfileError(msg) from None
    text = profile_text(raw)
    if redactor.contains_secret(text):
        msg = (
            f"{path.name} looks like it holds an API key or token: remove it (keys belong "
            "in the keychain)"
        )
        raise ProfileError(msg)
    if len(text) > MAX_PROFILE_CHARS:
        msg = (
            f"{path.name} is {len(text):,} characters: keep it under {MAX_PROFILE_CHARS:,} "
            "(it is sent with every request)"
        )
        raise ProfileError(msg)
    return text


class ProfileReader:
    """Reads the profile when asked, again only after the file changed.

    A profile that cannot be used is logged once (per change) and left out: Yarvis keeps
    working without it.
    """

    def __init__(self, path: Path, *, redactor: Redactor = default_redactor) -> None:
        self.path = path
        self._redactor = redactor
        self._seen: tuple[float, int] | None = None
        self._text = ""

    def __call__(self) -> str:
        """The current profile text ("" if there is none or it cannot be used)."""
        try:
            stat = self.path.stat()
        except OSError:
            self._seen, self._text = None, ""
            return ""
        seen = (stat.st_mtime, stat.st_size)
        if seen != self._seen:
            self._seen = seen
            try:
                self._text = load_profile(self.path, redactor=self._redactor)
            except ProfileError as exc:
                logger.warning("Your profile is not used: %s", exc)
                self._text = ""
            else:
                logger.info("Profile: %d characters", len(self._text))
        return self._text
