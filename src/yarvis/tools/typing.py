"""Typing into the app in front.

Text is pasted (clipboard + Command-V), not typed key by key: that works with every
keyboard layout (AZERTY included), and a paste cannot trigger single-key shortcuts. The
previous clipboard is put back afterwards. Pressing Enter is a separate action that needs
the user's "yes", because in a chat window it sends.

Some apps are off-limits: terminals (typing there runs commands), and apps that hold
passwords or system settings.

macOS asks once for permission: System Settings → Privacy & Security → Accessibility.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Final

from pydantic import Field

from yarvis.tools.base import Arguments, Tier, Tool, ToolError
from yarvis.tools.computer import OPEN, osascript

if TYPE_CHECKING:
    from yarvis.tools.computer import Apps, Runner

OFF_LIMITS: Final = frozenset(
    name.casefold()
    for name in (
        "Terminal", "iTerm2", "iTerm", "Warp", "Ghostty", "kitty", "Alacritty", "WezTerm",
        "Hyper", "Tabby", "Script Editor", "Automator", "Shortcuts", "Keychain Access",
        "Passwords", "System Settings", "System Preferences", "1Password", "1Password 7",
        "Bitwarden", "Activity Monitor", "Console",
    )
)  # fmt: skip
_FIELD: Final = "\x1f"
_NO_PERMISSION: Final = (
    "macOS has not allowed typing yet. In System Settings → Privacy & Security → "
    "Accessibility, turn on the app Yarvis runs from (e.g. Terminal), then try again."
)

_FRONTMOST: Final = (
    'tell application "System Events"',
    "set p to first application process whose frontmost is true",
    "return (name of p) & (character id 31) & (unix id of p)",
    "end tell",
)
_PASTE: Final = (
    "on run argv",
    "set saved to missing value",
    "try",
    "set saved to the clipboard",
    "end try",
    "set theText to item 1 of argv",
    "set the clipboard to theText",
    'tell application "System Events" to keystroke "v" using command down',
    "delay 0.4",
    "if saved is not missing value then set the clipboard to saved",
    "end run",
)
_ENTER: Final = ('tell application "System Events" to key code 36',)  # Return, any layout


class TypeArgs(Arguments):
    """Arguments of ``type_text``."""

    text: str = Field(min_length=1, max_length=2000)
    app: str = Field(
        default="",
        max_length=60,
        pattern=r"^[\w .&'+-]*$",
        description="App to type into (it is brought to the front); empty = the app in front.",
    )


class EnterArgs(Arguments):
    """Arguments of ``press_enter``."""

    app: str = Field(
        min_length=1, max_length=60, pattern=r"^[\w .&'+-]+$", description="The app in front."
    )


class _Keyboard[A: Arguments](Tool[A]):
    def __init__(self, run: Runner, apps: Apps) -> None:
        self._run = run
        self._apps = apps

    async def _front(self) -> str:
        """The app in front, if Yarvis may type into it."""
        code, output = await self._run(osascript(_FRONTMOST))
        if code != 0:
            raise ToolError(
                _NO_PERMISSION if _denied(output) else "Could not see which app is in front."
            )
        name, _, pid = output.partition(_FIELD)
        if pid.strip() == str(os.getpid()):
            msg = "Yarvis's own window is in front. Say which app to type into."
            raise ToolError(msg)
        if name.casefold() in OFF_LIMITS:
            msg = f"For safety, Yarvis does not type into {name}."
            raise ToolError(msg)
        return name

    async def _keys(self, argv: list[str]) -> None:
        code, output = await self._run(argv)
        if code != 0:
            raise ToolError(
                _NO_PERMISSION if _denied(output) else "The keystrokes did not go through."
            )


class TypeText(_Keyboard[TypeArgs]):
    """Type (paste) text."""

    name = "type_text"
    description = (
        "Type text into an app, at the cursor (without pressing Enter). Give the app's name "
        "to bring it to the front first."
    )
    tier = Tier.REVERSIBLE
    Args = TypeArgs

    async def run(self, args: TypeArgs) -> str:
        """Bring the app forward if asked, check it, paste."""
        if args.app:
            app, path = self._apps.find(args.app)
            code, _ = await self._run([OPEN, "-a", str(path)])
            if code != 0:
                msg = f"{app} would not open."
                raise ToolError(msg)
        front = await self._front()
        await self._keys(osascript(_PASTE, args.text))
        return f"Typed {len(args.text)} characters into {front}."

    def describe(self, args: TypeArgs) -> str:
        """'type “Hi Thomas” into Messages'."""
        preview = args.text if len(args.text) <= 60 else args.text[:57] + "…"  # noqa: PLR2004
        return f"type “{preview}” into {args.app or 'the app in front'}"


class PressEnter(_Keyboard[EnterArgs]):
    """Press Enter (sends, in chat apps)."""

    name = "press_enter"
    description = (
        "Press Enter in the app in front (in chat apps this sends the message). "
        "The user is asked to confirm first."
    )
    tier = Tier.EXTERNAL
    Args = EnterArgs

    async def run(self, args: EnterArgs) -> str:
        """Check it is the app the user agreed to, then press Enter."""
        front = await self._front()
        said, actual = args.app.casefold(), front.casefold()
        if said not in actual and actual not in said:
            msg = f"{front} is in front, not {args.app}, so Enter was not pressed."
            raise ToolError(msg)
        await self._keys(osascript(_ENTER))
        return f"Pressed Enter in {front}."

    def describe(self, args: EnterArgs) -> str:
        """'press Enter in Messages'."""
        return f"press Enter in {args.app}"


def _denied(output: str) -> bool:
    lowered = output.lower()
    return (
        "not allowed" in lowered or "1002" in lowered or "-1719" in lowered or "-25211" in lowered
    )


def typing_tools(run: Runner, apps: Apps) -> list[Tool[Any]]:
    """Typing tools."""
    return [TypeText(run, apps), PressEnter(run, apps)]
