"""Which computer tools exist on this machine, and how to describe them to the model."""

from __future__ import annotations

import os
import socket
import sys
from functools import partial
from typing import TYPE_CHECKING, Any, Literal

from yarvis.tools.browser import Browsers, browser_tools
from yarvis.tools.calendars import EventKitStore, calendar_tools
from yarvis.tools.computer import (
    XDG_OPEN,
    Apps,
    Media,
    OpenApp,
    OpenWebsite,
    QuitApp,
    SetVolume,
    run_command,
    start_hidden,
)
from yarvis.tools.ducking import BrowserDucker, Duckers, SpotifyDucker
from yarvis.tools.mail import MAIL_TIMEOUT_S, MacMail, mail_tools
from yarvis.tools.spotify import spotify_tools
from yarvis.tools.spotify_remote import SpotifyRemote, computer_name
from yarvis.tools.typing import typing_tools
from yarvis.tools.windows import start_spotify, windows_tools

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from yarvis.config.models import Config
    from yarvis.tools.base import Tool
    from yarvis.tools.browser import BrowserName
    from yarvis.tools.calendars import CalendarStore
    from yarvis.tools.computer import Runner
    from yarvis.tools.ducking import Ducker
    from yarvis.tools.spotify import SpotifyAccount, SpotifyCatalog


def computer_tools(
    run: Runner = run_command,
    *,
    apps: Apps | None = None,
    platform: str = sys.platform,
    browser: BrowserName | Literal["auto"] = "auto",
    typing: bool = True,
    spotify: SpotifyCatalog | None = None,
    spotify_account: SpotifyAccount | None = None,
    node: str | None = None,
) -> list[Tool[Any]]:
    """The tools that act on this computer, given what is installed.

    ``node`` is this computer's network name, to find its Spotify app among the user's
    devices (default: the real one).
    """
    name = computer_name(node or socket.gethostname())
    if platform == "win32":
        tools = windows_tools(run)
        if spotify is not None:
            remote = _remote(spotify_account, name, start_spotify)
            tools += spotify_tools(None, spotify, spotify_account, remote=remote)
        return tools
    if platform == "darwin":
        apps = apps or Apps()
        tools: list[Tool[Any]] = [
            OpenApp(run, apps),
            QuitApp(run, apps),
            SetVolume(run),
            Media(run, apps),
        ]
        browsers = Browsers(run, apps, browser)
        tools += browser_tools(run, browsers) if browsers.available() else [OpenWebsite(run)]
        if "Spotify" in apps.names():

            async def start() -> None:
                await start_hidden(run, "Spotify")

            remote = _remote(spotify_account, name, start)
            tools += spotify_tools(run, spotify, spotify_account, remote=remote)
        if typing:
            tools += typing_tools(run, apps)
        return tools
    if platform.startswith("linux") and os.access(XDG_OPEN, os.X_OK):
        return [OpenWebsite(run, opener=XDG_OPEN)]
    return []


def personal_tools(
    config: Config,
    *,
    platform: str = sys.platform,
    run: Runner | None = None,
    store: CalendarStore | None = None,
) -> list[Tool[Any]]:
    """Calendars and email, through the Mac's own apps and accounts."""
    if platform != "darwin":
        return []
    tools: list[Tool[Any]] = []
    if config.tools.calendar:
        tools += calendar_tools(store or EventKitStore())
    if config.tools.mail:
        tools += mail_tools(MacMail(run or partial(run_command, timeout_s=MAIL_TIMEOUT_S)))
    return tools


def _remote(
    account: SpotifyAccount | None, computer: str, start: Callable[[], Awaitable[None]]
) -> SpotifyRemote | None:
    """Spotify's Web API player, once the user has logged in."""
    return None if account is None else SpotifyRemote(account, computer=computer, start_app=start)


def music_ducker(
    config: Config,
    *,
    apps: Apps | None = None,
    platform: str = sys.platform,
    run: Runner = run_command,
) -> Ducker | None:
    """Turns Spotify and browser videos down during conversations, on a Mac."""
    if platform != "darwin" or not config.tools.computer:
        return None
    apps = apps or Apps()
    duckers: list[Ducker] = []
    if "Spotify" in apps.names():
        duckers.append(SpotifyDucker(run))
    browsers = Browsers(run, apps, config.tools.browser)
    if browsers.available():
        duckers.append(BrowserDucker(run, browsers))
    return Duckers(*duckers) if duckers else None


def abilities(tools: Sequence[Tool[Any]]) -> list[str]:
    """What the computer tools let Yarvis do, in words, for the system prompt."""
    names = {tool.name for tool in tools}
    said: list[str] = []
    if "open_app" in names:
        said.append("open and quit apps, set the volume and play, pause or skip music")
    if "browser_tabs" in names:
        said.append(
            "control the web browser: list, switch, open and close tabs in every window, open "
            "websites, search the web in a new tab for the user, go back, forward or reload, "
            "read the current page, follow its links, press its buttons and play or pause "
            "its videos"
        )
    elif "open_website" in names:
        said.append("open websites")
    if "spotify_play" in names:
        said.append(
            "play any song, album, artist or playlist on Spotify, tell what is playing, and set "
            "Spotify's own volume, shuffle, repeat and position"
        )
    if "spotify_my_playlists" in names:
        said.append("list and play the user's own Spotify playlists")
    elif "spotify_play" in names:
        said.append(
            "not yet the user's own Spotify playlists: for those they first run "
            "'yarvis spotify login' in the terminal"
        )
    if "type_text" in names:
        said.append("type text into apps (pressing Enter needs the user's confirmation)")
    if "calendar_events" in names:
        said.append(
            "read every calendar the user has on this Mac (Google, Outlook, iCloud, ...) and "
            "add events (the user confirms)"
        )
    if "mail_list" in names:
        said.append(
            "list, search and read the user's email in all their accounts, star them or mark "
            "them read, send emails or replies and move emails to the trash (sending and "
            "trashing need the user's confirmation; names are looked up in their contacts). "
            "Say emails as who, what and the gist, never the addresses"
        )
    return said
