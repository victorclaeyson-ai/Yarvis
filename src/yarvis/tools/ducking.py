"""Turn the music down while Yarvis listens and talks, and back up afterwards.

Music playing during a conversation hurts in two ways: speech recognition hears the
lyrics as words, and in barge-in mode the singing can "interrupt" Yarvis. So when a
conversation starts, Spotify is turned down to a fraction of its volume, and when it
ends, the volume goes back, unless the user changed it in the meantime.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Final, Protocol

from yarvis.tools.browser import javascript_command
from yarvis.tools.computer import osascript

if TYPE_CHECKING:
    from yarvis.tools.browser import Browsers
    from yarvis.tools.computer import Runner

logger = logging.getLogger(__name__)

DUCK_PERCENT: Final = 25
"""The music plays at this share of its volume during a conversation."""
_TOLERANCE: Final = 2  # Spotify reports a volume it was just set to give or take one

_DUCK: Final = (
    "on run argv",
    "set theShare to (item 1 of argv) as integer",
    'if application "Spotify" is not running then return ""',
    'tell application "Spotify"',
    'if player state is not playing then return ""',
    "set theOriginal to sound volume",
    "set theLowered to (theOriginal * theShare div 100)",
    "if theLowered < 5 then set theLowered to 5",
    'if theLowered >= theOriginal then return ""',
    "set sound volume to theLowered",
    "end tell",
    'return (theOriginal as text) & "," & (theLowered as text)',
    "end run",
)

_RESTORE: Final = (
    "on run argv",
    "set theOriginal to (item 1 of argv) as integer",
    "set theLowered to (item 2 of argv) as integer",
    "set theTolerance to (item 3 of argv) as integer",
    'if application "Spotify" is not running then return ""',
    'tell application "Spotify"',
    "set theVolume to sound volume",
    "if theVolume >= (theLowered - theTolerance) and theVolume <= (theLowered + theTolerance) then",
    "set sound volume to theOriginal",
    "end if",
    "end tell",
    'return ""',
    "end run",
)


class Ducker(Protocol):
    """Lowers other audio for the length of a conversation."""

    async def duck(self) -> None:
        """Turn the music down (if any is playing)."""
        ...

    async def restore(self) -> None:
        """Turn it back up, unless the user has set a volume since."""
        ...


class SpotifyDucker:
    """``Ducker`` for the Spotify Mac app. Never raises: music is a nicety."""

    def __init__(self, run: Runner, *, share: int = DUCK_PERCENT) -> None:
        self._run = run
        self._share = share
        self._saved: tuple[int, int] | None = None

    async def duck(self) -> None:
        """Remember Spotify's volume and lower it."""
        if self._saved is not None:
            return
        try:
            code, output = await self._run(osascript(_DUCK, str(self._share)))
        except Exception as exc:  # noqa: BLE001 - never let the music break a conversation
            logger.debug("Could not turn the music down: %s", exc)
            return
        original, _, lowered = output.partition(",")
        if code == 0 and original.isdigit() and lowered.isdigit():
            self._saved = (int(original), int(lowered))
            logger.info("Music turned down while we talk")

    async def restore(self) -> None:
        """Put the volume back if it is still where Yarvis left it."""
        if self._saved is None:
            return
        original, lowered = self._saved
        self._saved = None
        try:
            await self._run(osascript(_RESTORE, str(original), str(lowered), str(_TOLERANCE)))
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not turn the music back up: %s", exc)


_PAGE_DOWN: Final = (
    "(() => { const playing = [...document.querySelectorAll('video, audio')]"
    ".filter(m => !m.paused && !m.muted && m.volume > 0);"
    " for (const m of playing) { m.dataset.yarvisVolume = String(m.volume);"
    f" m.volume = m.volume * {DUCK_PERCENT / 100}; }}"
    " return String(playing.length); })()"
)
_PAGE_UP: Final = (
    "(() => { for (const m of document.querySelectorAll('[data-yarvis-volume]')) {"
    " const before = parseFloat(m.dataset.yarvisVolume);"
    f" if (Math.abs(m.volume - before * {DUCK_PERCENT / 100}) < 0.02) m.volume = before;"
    " delete m.dataset.yarvisVolume; } return 'ok'; })()"
)


class BrowserDucker:
    """``Ducker`` for a video or song playing in the browser's current tab.

    Uses the same fixed-JavaScript route as reading pages, so it only works once the user
    allowed JavaScript from Apple Events; otherwise it quietly does nothing.
    """

    def __init__(self, run: Runner, browsers: Browsers) -> None:
        self._run = run
        self._browsers = browsers
        self._lowered = False

    async def duck(self) -> None:
        """Turn down what plays in the current tab."""
        if self._lowered:
            return
        try:
            browser = await self._browsers.pick()
            code, output = await self._run(javascript_command(browser, _PAGE_DOWN))
        except Exception as exc:  # noqa: BLE001 - never let a video break a conversation
            logger.debug("Could not turn the video down: %s", exc)
            return
        if code == 0 and output.isdigit() and int(output) > 0:
            self._lowered = True
            logger.info("Video turned down while we talk")

    async def restore(self) -> None:
        """Turn it back up, unless the user changed the volume since."""
        if not self._lowered:
            return
        self._lowered = False
        try:
            browser = await self._browsers.pick()
            await self._run(javascript_command(browser, _PAGE_UP))
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not turn the video back up: %s", exc)


class Duckers:
    """Several ``Ducker`` at once (Spotify and the browser)."""

    def __init__(self, *duckers: Ducker) -> None:
        self._duckers = duckers

    async def duck(self) -> None:
        """Turn them all down, side by side."""
        await asyncio.gather(*(ducker.duck() for ducker in self._duckers))

    async def restore(self) -> None:
        """Turn them all back up."""
        await asyncio.gather(*(ducker.restore() for ducker in self._duckers))
