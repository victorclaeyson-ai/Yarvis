"""Control the computer: apps, websites, volume and music.

The model never gets near a shell:

* commands run with absolute paths, as argument lists (no shell), with a scrubbed
  environment (no API keys) and a timeout;
* apps are looked up among the installed ones, so a "name" can never be a path or a flag;
* AppleScript receives values as ``argv`` (data), never spliced into the script text;
* websites must be ``https://`` URLs without embedded credentials.
"""

from __future__ import annotations

import asyncio
import difflib
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal
from urllib.parse import urlsplit

from pydantic import Field

from yarvis.tools.base import Arguments, Tier, Tool, ToolError

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterable, Sequence

OPEN: Final = "/usr/bin/open"
OSASCRIPT: Final = "/usr/bin/osascript"
XDG_OPEN: Final = "/usr/bin/xdg-open"
COMMAND_TIMEOUT_S: Final = 8.0
APP_DIRS: Final = (
    Path("/Applications"),
    Path("/Applications/Utilities"),
    Path("/System/Applications"),
    Path("/System/Applications/Utilities"),
    Path.home() / "Applications",
)

type Runner = Callable[[Sequence[str]], Awaitable[tuple[int, str]]]
"""Runs a command (argument list) and returns (exit code, output)."""


async def run_command(argv: Sequence[str]) -> tuple[int, str]:
    """Run a program (no shell) with a minimal environment; return (exit code, output)."""
    env = {"PATH": "/usr/bin:/bin", "HOME": str(Path.home()), "LANG": "en_US.UTF-8"}
    process = await asyncio.create_subprocess_exec(
        *argv,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env=env,
    )
    try:
        async with asyncio.timeout(COMMAND_TIMEOUT_S):
            output, _ = await process.communicate()
    except TimeoutError:
        process.kill()
        await process.wait()
        raise
    return process.returncode or 0, output.decode(errors="replace").strip()[:500]


class Apps:
    """The applications installed on this Mac, found by (part of) their name."""

    def __init__(self, directories: Iterable[Path] = APP_DIRS) -> None:
        self._directories = tuple(directories)

    def names(self) -> dict[str, Path]:
        """Installed apps by name (``Safari`` → ``/Applications/Safari.app``)."""
        found: dict[str, Path] = {}
        for directory in self._directories:
            try:
                entries = list(directory.iterdir())
            except OSError:
                continue
            for entry in entries:
                if entry.suffix == ".app" and entry.is_dir():
                    found.setdefault(entry.stem, entry)
        return found

    def find(self, name: str) -> tuple[str, Path]:
        """The app called ``name`` ("chrome" finds "Google Chrome").

        Raises:
            ToolError: If there is no such app, or the name is ambiguous.
        """
        apps = self.names()
        wanted = name.casefold()
        exact = [app for app in apps if app.casefold() == wanted]
        partial = [app for app in apps if wanted in app.casefold()]
        matches = exact or partial
        if len(matches) == 1:
            return matches[0], apps[matches[0]]
        if matches:
            options = ", ".join(sorted(matches)[:4])
            msg = f"Several apps match {name!r}: {options}."
            raise ToolError(msg)
        close = difflib.get_close_matches(name, list(apps), n=3, cutoff=0.6)
        hint = f" Did you mean {' or '.join(close)}?" if close else ""
        msg = f"No app called {name!r} is installed.{hint}"
        raise ToolError(msg)


class AppArgs(Arguments):
    """An app name."""

    name: str = Field(
        min_length=1, max_length=60, pattern=r"^[\w .&'+-]+$", description="e.g. 'Safari'."
    )


class VolumeArgs(Arguments):
    """Arguments of ``set_volume``."""

    percent: int = Field(ge=0, le=100, description="Output volume, 0 (silent) to 100.")


MediaAction = Literal["play", "pause", "toggle", "next", "previous"]
Player = Literal["auto", "Music", "Spotify"]


class MediaArgs(Arguments):
    """Arguments of ``media``."""

    action: MediaAction
    player: Player = Field(default="auto", description="Which app; 'auto' picks the active one.")


class UrlArgs(Arguments):
    """Arguments of ``open_website``."""

    url: str = Field(min_length=8, max_length=2000, description="A full https:// URL.")


class _Command[A: Arguments](Tool[A]):
    def __init__(self, run: Runner) -> None:
        self._run = run

    async def _check(self, argv: Sequence[str], failure: str) -> str:
        code, output = await self._run(argv)
        if code != 0:
            raise ToolError(failure)
        return output


class OpenApp(_Command[AppArgs]):
    """Launch or switch to an app."""

    name = "open_app"
    description = "Open (or switch to) an installed app on the Mac."
    tier = Tier.REVERSIBLE
    Args = AppArgs

    def __init__(self, run: Runner, apps: Apps) -> None:
        super().__init__(run)
        self._apps = apps

    async def run(self, args: AppArgs) -> str:
        """Open it."""
        app, path = self._apps.find(args.name)
        await self._check([OPEN, "-a", str(path)], f"{app} would not open.")
        return f"{app} is open."

    def describe(self, args: AppArgs) -> str:
        """'open Safari'."""
        return f"open {args.name}"


_QUIT_SCRIPT: Final = (
    "on run argv",
    "set theApp to item 1 of argv",
    "if application theApp is running then",
    "tell application theApp to quit",
    'return "quit"',
    "end if",
    'return "not running"',
    "end run",
)


class QuitApp(_Command[AppArgs]):
    """Quit an app (unsaved work may be lost, so the user confirms first)."""

    name = "quit_app"
    description = "Quit an app on the Mac. The user is asked to confirm first."
    tier = Tier.EXTERNAL
    Args = AppArgs

    def __init__(self, run: Runner, apps: Apps) -> None:
        super().__init__(run)
        self._apps = apps

    async def run(self, args: AppArgs) -> str:
        """Quit it."""
        app, _ = self._apps.find(args.name)
        output = await self._check(osascript(_QUIT_SCRIPT, app), f"{app} would not quit.")
        return f"{app} was not running." if output == "not running" else f"{app} has quit."

    def describe(self, args: AppArgs) -> str:
        """'quit Safari'."""
        return f"quit {args.name}"


class SetVolume(_Command[VolumeArgs]):
    """Change the output volume."""

    name = "set_volume"
    description = "Set the Mac's output volume in percent."
    tier = Tier.REVERSIBLE
    Args = VolumeArgs

    async def run(self, args: VolumeArgs) -> str:
        """Set it."""
        script = (
            "on run argv",
            "set theVolume to (item 1 of argv) as integer",
            "set volume output volume theVolume",
            "end run",
        )
        await self._check(osascript(script, str(args.percent)), "The volume could not be set.")
        return f"Volume is {args.percent}%."

    def describe(self, args: VolumeArgs) -> str:
        """'set the volume to 40%'."""
        return f"set the volume to {args.percent}%"


_MEDIA_COMMANDS: Final[dict[MediaAction, str]] = {
    "play": "play",
    "pause": "pause",
    "toggle": "playpause",
    "next": "next track",
    "previous": "previous track",
}


class Media(_Command[MediaArgs]):
    """Play, pause and skip in Music or Spotify."""

    name = "media"
    description = (
        "Control music playback (play, pause, toggle, next, previous) in Music or Spotify."
    )
    tier = Tier.REVERSIBLE
    Args = MediaArgs

    def __init__(self, run: Runner, apps: Apps) -> None:
        super().__init__(run)
        self._apps = apps

    async def run(self, args: MediaArgs) -> str:
        """Do it."""
        player = args.player if args.player != "auto" else await self._active_player()
        # Both the player and the command come from fixed lists, never from model text.
        script = f'tell application "{player}" to {_MEDIA_COMMANDS[args.action]}'
        await self._check([OSASCRIPT, "-e", script], f"{player} did not respond.")
        return f"{player}: {args.action}."

    def describe(self, args: MediaArgs) -> str:
        """'pause the music'."""
        verbs = {"toggle": "play or pause", "next": "skip to the next track"}
        verb = verbs.get(args.action, args.action)
        return f"{verb} the music" if args.action != "next" else verb

    async def _active_player(self) -> Literal["Music", "Spotify"]:
        for player in ("Spotify", "Music"):
            code, output = await self._run([OSASCRIPT, "-e", f'application "{player}" is running'])
            if code == 0 and output == "true":
                return player
        return "Spotify" if "Spotify" in self._apps.names() else "Music"


class OpenWebsite(_Command[UrlArgs]):
    """Open a website in the default browser."""

    name = "open_website"
    description = "Open an https:// website in the default browser."
    tier = Tier.REVERSIBLE
    Args = UrlArgs

    def __init__(self, run: Runner, *, opener: str = OPEN) -> None:
        super().__init__(run)
        self._opener = opener

    async def run(self, args: UrlArgs) -> str:
        """Open it."""
        url = safe_url(args.url)
        await self._check([self._opener, url], "The browser would not open it.")
        return f"Opened {urlsplit(url).hostname}."

    def describe(self, args: UrlArgs) -> str:
        """'open example.com'."""
        host = urlsplit(args.url).hostname or args.url
        return f"open {host}"


def safe_url(url: str) -> str:
    """Accept only plain ``https://host/...`` URLs.

    Raises:
        ToolError: For other schemes, missing hosts or embedded credentials.
    """
    parts = urlsplit(url)
    if parts.scheme != "https" or not parts.hostname:
        msg = "Only https:// websites can be opened."
        raise ToolError(msg)
    if parts.username or parts.password or any(c.isspace() for c in url):
        msg = "That does not look like a normal web address."
        raise ToolError(msg)
    return url


def osascript(script: Sequence[str], *values: str) -> list[str]:
    """An ``osascript`` command line: the script as fixed lines, values as data (argv)."""
    argv = [OSASCRIPT]
    for line in script:
        argv += ["-e", line]
    return [*argv, *values]
