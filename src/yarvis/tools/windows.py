"""Control a Windows PC: apps, websites, volume, music and Spotify.

The same rules as on the Mac (see ``computer``):

* apps are looked up among the Start menu's apps (``Get-StartApps``), so a "name" can never
  be a path or a command, and they are opened by the ID Windows gave them;
* PowerShell only runs fixed commands; the one number put into a command (a process ID) is
  one Windows itself listed;
* websites must be ``https://`` URLs without credentials;
* only media and volume keys are ever pressed.

Quitting an app asks it to close its main window, as clicking ✕ would (it may still ask
to save); it needs the user's "yes".
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, cast
from urllib.parse import urlsplit

from yarvis.tools.base import Arguments, Tier, Tool, ToolError
from yarvis.tools.computer import (
    AppArgs,
    MediaAction,
    MediaArgs,
    OpenWebsite,
    UrlArgs,
    VolumeArgs,
    pick_app,
    run_command,
    safe_url,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from yarvis.tools.computer import Runner

logger = logging.getLogger(__name__)

type Launcher = Callable[[str], Awaitable[None]]
"""Opens a target the way double-clicking it would (an app ID or a website)."""
type KeyPresser = Callable[[Sequence[int]], Awaitable[None]]
"""Presses (and releases) virtual keys, in order."""

POWERSHELL: Final = str(
    Path(os.environ.get("SYSTEMROOT", r"C:\Windows"))
    / "System32"
    / "WindowsPowerShell"
    / "v1.0"
    / "powershell.exe"
)
_UTF8: Final = "[Console]::OutputEncoding = [Text.Encoding]::UTF8; "
_START_APPS: Final = _UTF8 + "Get-StartApps | Select-Object Name, AppID | ConvertTo-Json -Compress"
_WINDOWS: Final = (
    _UTF8 + "Get-Process | Where-Object { $_.MainWindowHandle -ne 0 } | "
    "Select-Object Id, ProcessName, Description | ConvertTo-Json -Compress"
)
_APPS_CACHE_S: Final = 300.0

# Virtual-key codes (WinUser.h).
VK_VOLUME_DOWN: Final = 0xAE
VK_VOLUME_UP: Final = 0xAF
VK_MEDIA_NEXT_TRACK: Final = 0xB0
VK_MEDIA_PREV_TRACK: Final = 0xB1
VK_MEDIA_PLAY_PAUSE: Final = 0xB3
_MEDIA_KEYS: Final[dict[MediaAction, int]] = {
    "play": VK_MEDIA_PLAY_PAUSE,
    "pause": VK_MEDIA_PLAY_PAUSE,
    "toggle": VK_MEDIA_PLAY_PAUSE,
    "next": VK_MEDIA_NEXT_TRACK,
    "previous": VK_MEDIA_PREV_TRACK,
}
_VOLUME_STEPS: Final = 50  # each volume key moves Windows' volume by 2%


def powershell(command: str) -> list[str]:
    """A PowerShell command line for one of the fixed commands in this module."""
    return [POWERSHELL, "-NoProfile", "-NonInteractive", "-Command", command]


class StartMenu:
    """The apps in the Start menu (desktop and Store apps alike), by name."""

    def __init__(self, run: Runner, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._run = run
        self._clock = clock
        self._apps: dict[str, str] = {}
        self._until = 0.0

    async def names(self) -> dict[str, str]:
        """Start menu apps: name → the ID Windows opens them by."""
        if self._apps and self._clock() < self._until:
            return self._apps
        code, output = await self._run(powershell(_START_APPS))
        if code != 0:
            msg = "Windows did not list its apps."
            raise ToolError(msg)
        apps: dict[str, str] = {}
        for item in _objects(output):
            name, app_id = item.get("Name"), item.get("AppID")
            if isinstance(name, str) and isinstance(app_id, str) and name and app_id:
                apps.setdefault(name, app_id)
        self._apps, self._until = apps, self._clock() + _APPS_CACHE_S
        return apps

    async def find(self, name: str) -> tuple[str, str]:
        """(name, ID) of the app called ``name``."""
        apps = await self.names()
        app = pick_app(name, apps, what="is in the Start menu")
        return app, apps[app]


class _WindowsTool[A: Arguments](Tool[A]):
    def __init__(self, run: Runner, *, launch: Launcher, press: KeyPresser) -> None:
        self._run = run
        self._launch = launch
        self._press = press


class WindowsOpenApp(_WindowsTool[AppArgs]):
    """Open an app from the Start menu."""

    name = "open_app"
    description = "Open (or switch to) an installed app on the PC."
    tier = Tier.REVERSIBLE
    Args = AppArgs

    def __init__(
        self, run: Runner, start_menu: StartMenu, *, launch: Launcher, press: KeyPresser
    ) -> None:
        super().__init__(run, launch=launch, press=press)
        self._start_menu = start_menu

    async def run(self, args: AppArgs) -> str:
        """Open it."""
        app, app_id = await self._start_menu.find(args.name)
        try:
            await self._launch(f"shell:AppsFolder\\{app_id}")
        except OSError as exc:
            msg = f"{app} would not open."
            raise ToolError(msg) from exc
        return f"{app} is open."

    def describe(self, args: AppArgs) -> str:
        """'open Spotify'."""
        return f"open {args.name}"


class WindowsQuitApp(_WindowsTool[AppArgs]):
    """Close an app's window (unsaved work may be lost, so the user confirms first)."""

    name = "quit_app"
    description = "Quit an app on the PC. The user is asked to confirm first."
    tier = Tier.EXTERNAL
    Args = AppArgs

    async def run(self, args: AppArgs) -> str:
        """Ask it to close, as clicking its ✕ would."""
        windows = await self._windows()
        app = pick_app(args.name, windows, what="has a window open")
        code, _ = await self._run(
            powershell(f"[void](Get-Process -Id {windows[app]:d}).CloseMainWindow()")
        )
        if code != 0:
            msg = f"{app} would not quit."
            raise ToolError(msg)
        return f"{app} was asked to close."

    def describe(self, args: AppArgs) -> str:
        """'quit Spotify'."""
        return f"quit {args.name}"

    async def _windows(self) -> dict[str, int]:
        """Apps with a window: their description ("Google Chrome") or name → process ID."""
        code, output = await self._run(powershell(_WINDOWS))
        if code != 0:
            msg = "Windows did not list the open apps."
            raise ToolError(msg)
        found: dict[str, int] = {}
        for item in _objects(output):
            pid = item.get("Id")
            if not isinstance(pid, int) or pid <= 0:
                continue
            for label in (item.get("Description"), item.get("ProcessName")):
                if isinstance(label, str) and label:
                    found.setdefault(label, pid)
        return found


class WindowsSetVolume(_WindowsTool[VolumeArgs]):
    """Set the volume with the volume keys (2% per key, as Windows moves it)."""

    name = "set_volume"
    description = "Set the PC's output volume in percent."
    tier = Tier.REVERSIBLE
    Args = VolumeArgs

    async def run(self, args: VolumeArgs) -> str:
        """All the way down, then up to the level."""
        steps = round(args.percent / 2)
        await self._press([VK_VOLUME_DOWN] * _VOLUME_STEPS + [VK_VOLUME_UP] * steps)
        return f"Volume is {steps * 2}%."

    def describe(self, args: VolumeArgs) -> str:
        """'set the volume to 40%'."""
        return f"set the volume to {args.percent}%"


class WindowsMedia(_WindowsTool[MediaArgs]):
    """The media keys: whichever app is playing (Spotify, a browser video) responds."""

    name = "media"
    description = (
        "Control whatever is playing on the PC (Spotify, a video): play or pause (the same "
        "key: it toggles), next, previous."
    )
    tier = Tier.REVERSIBLE
    Args = MediaArgs

    async def run(self, args: MediaArgs) -> str:
        """Press the key."""
        await self._press([_MEDIA_KEYS[args.action]])
        return "Done."

    def starts_playback(self, args: MediaArgs) -> bool:
        """Play/pause is one key: it may start music; so may next and previous."""
        return args.action != "pause"

    def describe(self, args: MediaArgs) -> str:
        """'play or pause the music'."""
        return {
            "next": "skip to the next track",
            "previous": "go back a track",
        }.get(args.action, "play or pause the music")


class WindowsOpenWebsite(OpenWebsite):
    """Open a website in the default browser."""

    def __init__(self, run: Runner, *, launch: Launcher) -> None:
        super().__init__(run)
        self._launch = launch

    async def run(self, args: UrlArgs) -> str:
        """Open it."""
        url = safe_url(args.url)
        try:
            await self._launch(url)
        except OSError as exc:
            msg = "The browser would not open it."
            raise ToolError(msg) from exc
        return f"Opened {urlsplit(url).hostname}."


def windows_tools(
    run: Runner = run_command,
    *,
    launch: Launcher | None = None,
    press: KeyPresser | None = None,
) -> list[Tool[Any]]:
    """The tools that act on a Windows PC."""
    launch = launch or start_file
    press = press or press_keys
    return [
        WindowsOpenApp(run, StartMenu(run), launch=launch, press=press),
        WindowsQuitApp(run, launch=launch, press=press),
        WindowsSetVolume(run, launch=launch, press=press),
        WindowsMedia(run, launch=launch, press=press),
        WindowsOpenWebsite(run, launch=launch),
    ]


def _objects(output: str) -> list[dict[str, object]]:
    """``ConvertTo-Json`` output: a list of objects, or one object on its own."""
    try:
        parsed: object = json.loads(output) if output.strip() else []
    except ValueError:
        logger.info("PowerShell gave an unreadable answer")
        return []
    items = cast("list[object]", parsed) if isinstance(parsed, list) else [parsed]
    return [cast("dict[str, object]", item) for item in items if isinstance(item, dict)]


# --------------------------------------------------------------------------- Windows itself


async def start_file(target: str) -> None:  # pragma: no cover - Windows only
    """Open ``target`` (an app ID, a website) the way double-clicking it would."""
    if sys.platform != "win32":
        msg = "Only on Windows."
        raise OSError(msg)
    await asyncio.to_thread(os.startfile, target)


async def press_keys(codes: Sequence[int]) -> None:  # pragma: no cover - Windows only
    """Press and release each virtual key in turn."""
    if sys.platform != "win32":
        msg = "Only on Windows."
        raise OSError(msg)
    import ctypes  # noqa: PLC0415 - Windows only

    user32 = ctypes.windll.user32
    key_up = 0x0002

    def press() -> None:
        for code in codes:
            user32.keybd_event(code, 0, 0, 0)
            user32.keybd_event(code, 0, key_up, 0)

    await asyncio.to_thread(press)


async def start_spotify() -> None:  # pragma: no cover - Windows only
    """Start the Spotify app minimized (installed from spotify.com), or else normally."""
    if sys.platform != "win32":
        return
    app = Path(os.environ.get("APPDATA", "")) / "Spotify" / "Spotify.exe"
    if not app.is_file():
        await start_file("spotify:")  # the Microsoft Store version: no way to start it minimized
        return

    def start() -> None:
        subprocess.Popen(  # noqa: S603 - a fixed program, fixed arguments
            [str(app), "--minimized"],
            creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
        )

    await asyncio.to_thread(start)
