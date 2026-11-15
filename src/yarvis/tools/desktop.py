"""Which computer tools exist on this machine, and how to describe them to the model."""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING, Any, Literal

from yarvis.tools.browser import Browsers, browser_tools
from yarvis.tools.computer import (
    XDG_OPEN,
    Apps,
    Media,
    OpenApp,
    OpenWebsite,
    QuitApp,
    SetVolume,
    run_command,
)
from yarvis.tools.spotify import spotify_tools
from yarvis.tools.typing import typing_tools

if TYPE_CHECKING:
    from collections.abc import Sequence

    from yarvis.tools.base import Tool
    from yarvis.tools.browser import BrowserName
    from yarvis.tools.computer import Runner
    from yarvis.tools.spotify import SpotifyCatalog


def computer_tools(
    run: Runner = run_command,
    *,
    apps: Apps | None = None,
    platform: str = sys.platform,
    browser: BrowserName | Literal["auto"] = "auto",
    typing: bool = True,
    spotify: SpotifyCatalog | None = None,
) -> list[Tool[Any]]:
    """The tools that act on this computer, given what is installed."""
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
            tools += spotify_tools(run, spotify)
        if typing:
            tools += typing_tools(run, apps)
        return tools
    if platform.startswith("linux") and os.access(XDG_OPEN, os.X_OK):
        return [OpenWebsite(run, opener=XDG_OPEN)]
    return []


def abilities(tools: Sequence[Tool[Any]]) -> list[str]:
    """What the computer tools let Yarvis do, in words, for the system prompt."""
    names = {tool.name for tool in tools}
    said: list[str] = []
    if "open_app" in names:
        said.append("open and quit apps, set the volume and play, pause or skip music")
    if "browser_tabs" in names:
        said.append(
            "control the web browser: list, switch, open and close tabs, search the web in a "
            "new tab for the user, go back, forward or reload"
        )
    elif "open_website" in names:
        said.append("open websites")
    if "spotify_play" in names:
        said.append(
            "play any song, album, artist or playlist on Spotify, tell what is playing, and set "
            "Spotify's own volume, shuffle, repeat and position"
        )
    if "type_text" in names:
        said.append("type text into apps (pressing Enter needs the user's confirmation)")
    return said
