"""Control the web browser: tabs in every window, websites, searching, and the current page.

Brave, Chrome, Edge, Vivaldi and Chromium share one AppleScript dictionary; Safari has its
own. The browser's name comes from a fixed list and is the only thing ever written into a
script; URLs, searches, numbers and JavaScript travel as data (``argv``).

Reading and clicking on pages uses small JavaScript programs that are fixed in this file:
the model only ever chooses an element by the number the page listing gave it, and the
code checks the element still has the label the model expects. Following a link is just
opening its https:// address; pressing a button can do anything, so it is asked first.
Private (incognito) windows are never listed or read.

Everything a website writes (titles, text, link labels) is untrusted content.
"""

from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, Literal, Self, cast, get_args
from urllib.parse import quote_plus, urlsplit

from pydantic import Field, model_validator

from yarvis.tools.base import Arguments, Tier, Tool, ToolError
from yarvis.tools.computer import osascript, safe_url, short_url

if TYPE_CHECKING:
    from yarvis.tools.computer import Apps, Runner

BrowserName = Literal[
    "Brave Browser", "Google Chrome", "Microsoft Edge", "Vivaldi", "Chromium", "Safari"
]
BROWSERS: Final[tuple[BrowserName, ...]] = get_args(BrowserName)
PS: Final = "/bin/ps"
_FIELD: Final = "\x1f"
_RECORD: Final = "\x1e"
_MAX_TITLE: Final = 80
_MAX_LISTED_TABS: Final = 60
_LABEL_OVERLAP: Final = 0.6
_MIN_WORD: Final = 4  # shorter title words ("the", "on") are no use for matching by ear
_NO_WINDOW: Final = "<no window>"
_PRIVATE: Final = "<private>"


class Browsers:
    """Picks the browser to control: the configured one, else the one in use."""

    def __init__(self, run: Runner, apps: Apps, preferred: BrowserName | Literal["auto"]) -> None:
        self._run = run
        self._apps = apps
        self._preferred: BrowserName | Literal["auto"] = preferred

    def available(self) -> bool:
        """Whether any supported browser is installed."""
        installed = self._apps.names()
        return any(browser in installed for browser in BROWSERS)

    async def pick(self) -> BrowserName:
        """The browser to act on."""
        installed = self._apps.names()
        preferred = self._preferred
        if preferred != "auto":
            if preferred not in installed:
                msg = f"{preferred} is not installed."
                raise ToolError(msg)
            return preferred
        _, processes = await self._run([PS, "-axo", "comm="])
        for browser in BROWSERS:
            if f"/{browser}.app/" in processes:
                return browser
        for browser in BROWSERS:
            if browser in installed:
                return browser
        msg = "No supported browser is installed."
        raise ToolError(msg)


# ----------------------------------------------------------------------------- AppleScript


def _tell(browser: BrowserName, *lines: str) -> tuple[str, ...]:
    return (f'tell application "{browser}"', *lines, "end tell")


def _tabs_script(browser: BrowserName) -> tuple[str, ...]:
    """Every tab of every window, front window first: window, tab, active, URL, title."""
    safari = browser == "Safari"
    title, active = ("name", "index of current tab") if safari else ("title", "active tab index")
    # Chromium browsers say which windows are private; those are skipped.
    private = "false" if safari else '(mode of window w is "incognito")'
    return (
        f'if application "{browser}" is not running then return ""',
        "set f to character id 31",
        "set r to character id 30",
        'set out to ""',
        *_tell(
            browser,
            "repeat with w from 1 to (count of windows)",
            "try",
            f"if not {private} then",
            f"set theTitles to {title} of every tab of window w",
            "set theURLs to URL of every tab of window w",
            f"set activeIndex to {active} of window w",
            "repeat with t from 1 to (count of theTitles)",
            "try",  # one odd tab (no address yet) must not hide the others
            "set out to out & (w as text) & f & (t as text) & f & ((t = activeIndex) as text) "
            "& f & (item t of theURLs) & f & (item t of theTitles) & r",
            "end try",
            "end repeat",
            "end if",
            "end try",  # a window without tabs (settings, a pop-up) is skipped
            "end repeat",
        ),
        "return out",
    )


def _switch_script(browser: BrowserName) -> tuple[str, ...]:
    select = (
        "set current tab of window w to tab t of window w"
        if browser == "Safari"
        else "set active tab index of window w to t"
    )
    return (
        "on run argv",
        "set w to (item 1 of argv) as integer",
        "set t to (item 2 of argv) as integer",
        *_tell(browser, select, "set index of window w to 1", "activate"),
        "end run",
    )


def _open_script(browser: BrowserName) -> tuple[str, ...]:
    """Open ``theURL`` in a new tab, or an empty new tab when it is ""."""
    if browser == "Safari":
        body = (
            "if (count of windows) is 0 then",
            'if theURL is "" then',
            "make new document",
            "else",
            "make new document with properties {URL:theURL}",
            "end if",
            'else if theURL is "" then',
            "tell front window to set current tab to (make new tab)",
            "else",
            "tell front window to set current tab to (make new tab with properties {URL:theURL})",
            "end if",
            "activate",
        )
    else:
        body = (
            "if (count of windows) is 0 then",
            "make new window",
            'if theURL is not "" then set URL of active tab of front window to theURL',
            'else if theURL is "" then',
            "tell front window to make new tab",
            "else",
            "tell front window to make new tab with properties {URL:theURL}",
            "end if",
            "activate",
        )
    return ("on run argv", "set theURL to item 1 of argv", *_tell(browser, *body), "end run")


def _close_script(browser: BrowserName) -> tuple[str, ...]:
    current = "current tab" if browser == "Safari" else "active tab"
    return (
        "on run argv",
        "set w to (item 1 of argv) as integer",
        "set t to (item 2 of argv) as integer",
        *_tell(
            browser,
            "if w is 0 then",
            f"close {current} of front window",
            "else",
            "close tab t of window w",
            "end if",
        ),
        "end run",
    )


_GO_COMMANDS: Final = {"back": "go back", "forward": "go forward", "reload": "reload"}


def _go_script(
    browser: BrowserName, action: Literal["back", "forward", "reload"]
) -> tuple[str, ...]:
    if browser == "Safari":
        if action != "reload":
            msg = "Safari cannot go back or forward by script; use Brave or Chrome for that."
            raise ToolError(msg)
        line = "set URL of current tab of front window to (URL of current tab of front window)"
        return _tell(browser, line)
    return _tell(browser, f"tell active tab of front window to {_GO_COMMANDS[action]}")


def _javascript_script(browser: BrowserName) -> tuple[str, ...]:
    """Run the JavaScript in ``argv`` in the current tab of the front window."""
    if browser == "Safari":
        run = "set theResult to do JavaScript theCode in current tab of front window"
        private: tuple[str, ...] = ()
    else:
        run = "set theResult to execute active tab of front window javascript theCode"
        private = (f'if mode of front window is "incognito" then return "{_PRIVATE}"',)
    return (
        "on run argv",
        "set theCode to item 1 of argv",
        *_tell(browser, f'if (count of windows) is 0 then return "{_NO_WINDOW}"', *private, run),
        'if theResult is missing value then return ""',
        "return theResult as text",
        "end run",
    )


def javascript_command(browser: BrowserName, code: str) -> list[str]:
    """The ``osascript`` command that runs fixed ``code`` in the browser's current tab.

    Prints ``<no window>`` or ``<private>`` instead when there is no tab to run it in.
    """
    return osascript(_javascript_script(browser), code)


def _navigate_script(browser: BrowserName) -> tuple[str, ...]:
    tab = "current tab" if browser == "Safari" else "active tab"
    return (
        "on run argv",
        "set theURL to item 1 of argv",
        *_tell(browser, f"set URL of {tab} of front window to theURL"),
        "end run",
    )


# ----------------------------------------------------------------------------- JavaScript

# Shared helpers: what counts as a link (a real https:// address, not "#" or a script) and
# the label a person would read on an element.
_JS_HELPERS: Final = r"""
const clean = s => (s || '').replace(/\s+/g, ' ').trim();
const isLink = e => e.tagName === 'A' && /^https?:/i.test(e.href)
  && !(e.getAttribute('href') || '').startsWith('#') && e.getAttribute('role') !== 'button';
const labelOf = e => clean(isLink(e)
  ? (e.innerText || e.getAttribute('aria-label') || e.title)
  : (e.getAttribute('aria-label') || e.innerText || e.title || e.value)).slice(0, 90);
"""

_READ_PAGE_JS: Final = (
    "(() => {"
    + _JS_HELPERS
    + r"""
for (const e of document.querySelectorAll('[data-yarvis-n]')) e.removeAttribute('data-yarvis-n');
const found = [];
const seen = new Set();
const nodes = document.querySelectorAll('a[href], button, [role="button"], [role="link"], '
  + 'input[type="submit"], input[type="button"]');
// Menus, headers and sidebars come last, so the page's own content (search results,
// videos, articles) makes the list even on busy sites such as YouTube.
const CHROME = 'nav, header, aside, footer, [role="navigation"], [role="banner"], '
  + '[role="contentinfo"], [role="complementary"], #masthead, #guide, ytd-masthead, '
  + 'tp-yt-app-drawer, ytd-mini-guide-renderer';
for (let i = 0; i < nodes.length && found.length < 1500; i++) {
  const e = nodes[i];
  const box = e.getBoundingClientRect();
  if (box.width < 2 || box.height < 2) continue;
  const text = labelOf(e);
  if (!text) continue;
  const key = (isLink(e) ? e.href : 'button') + '|' + text;
  if (seen.has(key)) continue;
  seen.add(key);
  found.push({e, text, link: isLink(e), seen: box.bottom > 0 && box.top < innerHeight,
    chrome: !!e.closest(CHROME), i});
}
found.sort((a, b) => (b.seen - a.seen) || (a.chrome - b.chrome) || (a.i - b.i));
const items = found.slice(0, 50).map((x, k) => {
  x.e.setAttribute('data-yarvis-n', String(k + 1));
  return [k + 1, x.link ? 'link' : 'button', x.text];
});
const text = clean(document.body ? document.body.innerText : '').slice(0, 2500);
return JSON.stringify({title: clean(document.title).slice(0, 150), url: location.href,
  text, items});
})()"""
)


def _element_js(number: int) -> str:
    """Describe element ``number`` from the last listing ("" if it is gone)."""
    return (
        "(() => {"
        + _JS_HELPERS
        + f"const e = document.querySelector('[data-yarvis-n=\"{number:d}\"]');"
        + r"""
if (!e) return '';
return JSON.stringify({link: isLink(e), text: labelOf(e), href: isLink(e) ? e.href : ''});
})()"""
    )


def _press_js(number: int) -> str:
    return (
        f"(() => {{ const e = document.querySelector('[data-yarvis-n=\"{number:d}\"]');"
        " if (!e) return ''; e.scrollIntoView({block: 'center'}); e.click(); return 'ok'; })()"
    )


def _media_js(action: Literal["play", "pause", "toggle"]) -> str:
    return (
        "(() => { const all = [...document.querySelectorAll('video, audio')];"
        " const m = all.find(x => !x.paused) || all.find(x => x.currentSrc || x.src);"
        " if (!m) return 'none';"
        f" const want = '{action}';"
        " if (want === 'play' || (want === 'toggle' && m.paused)) { m.play()?.catch(() => {}); }"
        " else { m.pause(); }"
        " return m.paused ? 'paused' : 'playing'; })()"
    )


# ----------------------------------------------------------------------------- tools


@dataclass(frozen=True, slots=True)
class Tab:
    """One open tab. ``number`` counts across windows, front window first."""

    number: int
    window: int
    index: int
    active: bool
    url: str
    title: str

    @property
    def site(self) -> str:
        """The host name, for listings."""
        return urlsplit(self.url).hostname or self.url[:40]


def _parse_tabs(output: str) -> list[Tab]:
    """Parse the tab listing; malformed records (a title containing separators) are skipped."""
    tabs: list[Tab] = []
    for record in output.split(_RECORD):
        fields = record.split(_FIELD, 4)
        if len(fields) != 5 or not (fields[0].isdigit() and fields[1].isdigit()):  # noqa: PLR2004
            continue
        window, index, active, url, title = fields
        tabs.append(
            Tab(len(tabs) + 1, int(window), int(index), active == "true", url, title.strip())
        )
    return tabs


def _best_match(spoken: str, tabs: list[Tab]) -> Tab:
    """The tab whose title or site contains ``spoken``, else the closest-sounding one."""
    wanted = spoken.casefold()
    for tab in tabs:
        if wanted in tab.title.casefold() or wanted in tab.url.casefold():
            return tab
    names: dict[str, Tab] = {}
    for tab in reversed(tabs):  # the first tab wins a tie
        names[tab.title.casefold()] = tab
        names[tab.site.removeprefix("www.").split(".")[0]] = tab  # "github", "youtube"
        names.update({w: tab for w in tab.title.casefold().split() if len(w) >= _MIN_WORD})
    close = difflib.get_close_matches(wanted.replace(" ", ""), list(names), n=1, cutoff=0.7)
    if not close:
        msg = f"No tab matches {spoken!r}."
        raise ToolError(msg)
    return names[close[0]]


class NoArgs(Arguments):
    """No arguments."""


class SwitchArgs(Arguments):
    """Arguments of ``browser_switch_tab``: a number or some text."""

    number: int | None = Field(
        default=None, ge=1, le=500, description="Tab number from browser_tabs."
    )
    match: str | None = Field(
        default=None,
        min_length=1,
        max_length=100,
        description="Text in the tab's title or address.",
    )

    @model_validator(mode="after")
    def _one(self) -> Self:
        if (self.number is None) == (self.match is None):
            msg = "give either number or match"
            raise ValueError(msg)
        return self


class OpenArgs(Arguments):
    """Arguments of ``browser_open``."""

    url: str | None = Field(
        default=None,
        min_length=8,
        max_length=2000,
        description="A full https:// URL. Leave it out for an empty new tab.",
    )
    in_current_tab: bool = Field(
        default=False,
        description="Load the website in the current tab instead (e.g. a tab just opened).",
    )


class SearchArgs(Arguments):
    """Arguments of ``browser_search``."""

    query: str = Field(min_length=1, max_length=300)


class CloseArgs(Arguments):
    """Arguments of ``browser_close_tab``."""

    number: int | None = Field(
        default=None, ge=1, le=500, description="Tab number from browser_tabs; omit for this tab."
    )


class GoArgs(Arguments):
    """Arguments of ``browser_go``."""

    action: Literal["back", "forward", "reload"]


class ElementArgs(Arguments):
    """An element from the last ``browser_read_page`` listing."""

    number: int = Field(ge=1, le=50, description="The item's number in the page listing.")
    label: str = Field(
        min_length=1, max_length=100, description="The item's text, as the listing showed it."
    )


class MediaArgs(Arguments):
    """Arguments of ``browser_media``."""

    action: Literal["play", "pause", "toggle"]


class _BrowserTool[A: Arguments](Tool[A]):
    def __init__(self, run: Runner, browsers: Browsers) -> None:
        self._run = run
        self._browsers = browsers

    async def _script(self, lines: tuple[str, ...], *values: str, failure: str) -> str:
        code, output = await self._run(osascript(lines, *values))
        if code != 0:
            raise ToolError(failure)
        return output

    async def _tabs(self, browser: BrowserName) -> list[Tab]:
        output = await self._script(_tabs_script(browser), failure=f"{browser} did not answer.")
        return _parse_tabs(output)

    async def _tab(self, browser: BrowserName, number: int) -> Tab:
        tabs = await self._tabs(browser)
        if number > len(tabs):
            msg = f"There are only {len(tabs)} tabs."
            raise ToolError(msg)
        return tabs[number - 1]

    async def _javascript(self, browser: BrowserName, code: str) -> str:
        """Run fixed JavaScript in the current tab and return its (string) result."""
        status, output = await self._run(javascript_command(browser, code))
        if status != 0:
            if "JavaScript" in output:
                menu = "Develop" if browser == "Safari" else "View > Developer"
                msg = (
                    f"{browser} does not allow it yet. In {browser}, turn on {menu} > "
                    "Allow JavaScript from Apple Events, then ask again."
                )
                raise ToolError(msg)
            msg = f"{browser} did not answer."
            raise ToolError(msg)
        if output == _NO_WINDOW:
            msg = f"{browser} has no open window."
            raise ToolError(msg)
        if output == _PRIVATE:
            msg = "That is a private window; Yarvis does not look at those."
            raise ToolError(msg)
        return output

    async def _element(self, browser: BrowserName, args: ElementArgs) -> tuple[bool, str, str]:
        """(is a link, its label, its address) after checking it is the expected item."""
        output = await self._javascript(browser, _element_js(args.number))
        details = _json_object(output)
        if details is None:
            msg = "The page has changed since it was read. Read it again first."
            raise ToolError(msg)
        text = str(details.get("text", ""))
        if not _same_label(args.label, text):
            msg = f"Item {args.number} is now {text!r}. Read the page again first."
            raise ToolError(msg)
        return details.get("link") is True, text, str(details.get("href", ""))


def _same_label(wanted: str, actual: str) -> bool:
    """Whether the model's label still describes the element (it may shorten or re-case)."""
    a, b = _words(wanted), _words(actual)
    if not a or not b:
        return False
    if f" {a} " in f" {b} " or f" {b} " in f" {a} ":  # whole words: not "Unsubscribe"
        return True
    overlap = len(set(a.split()) & set(b.split())) / len(set(a.split()))
    return overlap >= _LABEL_OVERLAP


def _words(text: str) -> str:
    return " ".join(re.sub(r"[^\w]+", " ", text.casefold()).split())


def _json_object(text: str) -> dict[str, Any] | None:
    try:
        value: object = json.loads(text)
    except ValueError:
        return None
    return cast("dict[str, Any]", value) if isinstance(value, dict) else None


class BrowserTabs(_BrowserTool[NoArgs]):
    """List the open tabs."""

    name = "browser_tabs"
    description = (
        "List the open browser tabs in every window (number, title, site). The numbers "
        "count across windows, front window first."
    )
    tier = Tier.READ
    Args = NoArgs
    untrusted_output = True

    async def run(self, args: NoArgs) -> str:
        """List them."""
        browser = await self._browsers.pick()
        tabs = await self._tabs(browser)
        if not tabs:
            return f"{browser} has no open tabs (private windows are not listed)."
        windows = len({tab.window for tab in tabs})
        lines = [f"{browser}: {len(tabs)} tabs in {windows} window(s)."]
        for tab in tabs[:_MAX_LISTED_TABS]:
            if windows > 1 and (tab.number == 1 or tab.window != tabs[tab.number - 2].window):
                lines.append("Front window:" if tab.window == 1 else "Another window:")
            current = " [current]" if tab.active and tab.window == 1 else ""
            lines.append(f"{tab.number}. {tab.title[:_MAX_TITLE]} ({tab.site}){current}")
        if len(tabs) > _MAX_LISTED_TABS:
            lines.append(f"…and {len(tabs) - _MAX_LISTED_TABS} more.")
        return "\n".join(lines)

    def describe(self, args: NoArgs) -> str:
        """'look at your tabs'."""
        return "look at your browser tabs"


class BrowserSwitchTab(_BrowserTool[SwitchArgs]):
    """Go to another tab (in any window)."""

    name = "browser_switch_tab"
    description = "Switch to a browser tab by number, or by text in its title or address."
    tier = Tier.REVERSIBLE
    Args = SwitchArgs
    untrusted_output = True

    async def run(self, args: SwitchArgs) -> str:
        """Switch."""
        browser = await self._browsers.pick()
        if args.match is None:
            tab = await self._tab(browser, cast("int", args.number))
        else:
            tab = _best_match(args.match, await self._tabs(browser))
        await self._script(
            _switch_script(browser),
            str(tab.window),
            str(tab.index),
            failure="The tab did not switch.",
        )
        return f"Switched to tab {tab.number}: {tab.title[:_MAX_TITLE]}"

    def describe(self, args: SwitchArgs) -> str:
        """'switch to tab 3' / 'switch to the Gmail tab'."""
        return f"switch to tab {args.number}" if args.number else f"switch to the {args.match} tab"


class BrowserOpen(_BrowserTool[OpenArgs]):
    """Open a new tab, empty or with a website."""

    name = "browser_open"
    description = (
        "Open an https:// website in a new browser tab (or in the current tab), or open an "
        "empty new tab when no URL is given."
    )
    tier = Tier.REVERSIBLE
    Args = OpenArgs
    opens_links = True

    async def run(self, args: OpenArgs) -> str:
        """Open it."""
        url = "" if args.url is None else safe_url(args.url)
        browser = await self._browsers.pick()
        if url and args.in_current_tab:
            await self._script(_navigate_script(browser), url, failure="The page did not open.")
            return f"Opened {urlsplit(url).hostname} in the current tab."
        await self._script(_open_script(browser), url, failure=f"{browser} would not open it.")
        if not url:
            return f"Opened a new tab in {browser}."
        return f"Opened {urlsplit(url).hostname} in {browser}."

    def sends_data_out(self, args: OpenArgs) -> bool:
        """An address can carry data to its site; an empty tab cannot."""
        return args.url is not None

    def describe(self, args: OpenArgs) -> str:
        """'open kuleuven.be' / 'open a new tab'."""
        if args.url is None:
            return "open a new tab"
        return f"open {short_url(args.url)}"


class BrowserSearch(_BrowserTool[SearchArgs]):
    """Search the web in the browser (for the user to read)."""

    name = "browser_search"
    description = "Search the web in a new browser tab, for the user to look at."
    tier = Tier.REVERSIBLE
    Args = SearchArgs

    async def run(self, args: SearchArgs) -> str:
        """Search."""
        browser = await self._browsers.pick()
        engine = (
            "https://search.brave.com/search?q="
            if browser == "Brave Browser"
            else "https://www.google.com/search?q="
        )
        url = engine + quote_plus(args.query)
        await self._script(_open_script(browser), url, failure=f"{browser} would not search.")
        return f"Searching for '{args.query}' in {browser}."

    def describe(self, args: SearchArgs) -> str:
        """'search for "weather Leuven"'."""
        return f"search for “{args.query}”"


class BrowserCloseTab(_BrowserTool[CloseArgs]):
    """Close a tab."""

    name = "browser_close_tab"
    description = "Close a browser tab by its number from browser_tabs (or the current tab)."
    tier = Tier.REVERSIBLE
    Args = CloseArgs

    async def run(self, args: CloseArgs) -> str:
        """Close it."""
        browser = await self._browsers.pick()
        if args.number is None:
            where = ("0", "0")
        else:
            tab = await self._tab(browser, args.number)
            where = (str(tab.window), str(tab.index))
        await self._script(_close_script(browser), *where, failure="The tab did not close.")
        return f"Closed {'tab ' + str(args.number) if args.number else 'the current tab'}."

    def describe(self, args: CloseArgs) -> str:
        """'close tab 2' / 'close this tab'."""
        return f"close tab {args.number}" if args.number else "close this tab"


class BrowserGo(_BrowserTool[GoArgs]):
    """Back, forward, reload."""

    name = "browser_go"
    description = "Go back, go forward, or reload in the current browser tab."
    tier = Tier.REVERSIBLE
    Args = GoArgs

    async def run(self, args: GoArgs) -> str:
        """Do it."""
        browser = await self._browsers.pick()
        await self._script(_go_script(browser, args.action), failure=f"{browser} did not respond.")
        return f"Done: {args.action}."

    def describe(self, args: GoArgs) -> str:
        """'go back'."""
        return "reload the page" if args.action == "reload" else f"go {args.action}"


class BrowserReadPage(_BrowserTool[NoArgs]):
    """Read the current page: its text, links and buttons."""

    name = "browser_read_page"
    description = (
        "Read the page in the current browser tab: its title, the start of its text, and "
        "numbered links and buttons (visible ones first) to use with browser_open_link and "
        "browser_press. Read again after the page changes."
    )
    tier = Tier.READ
    Args = NoArgs
    untrusted_output = True

    async def run(self, args: NoArgs) -> str:
        """Read it."""
        browser = await self._browsers.pick()
        page = _json_object(await self._javascript(browser, _READ_PAGE_JS))
        if page is None:
            msg = "The page could not be read."
            raise ToolError(msg)
        site = urlsplit(str(page.get("url", ""))).hostname or "this page"
        lines = [f"Page: {page.get('title', '')} ({site})", f"Text: {page.get('text', '')}"]
        items = cast("list[Any]", page.get("items") or [])
        if items:
            lines.append("Links and buttons:")
            lines += [f"{n}. {kind}: {text}" for n, kind, text in items]
        return "\n".join(lines)

    def describe(self, args: NoArgs) -> str:
        """'read the page'."""
        return "read the current page"


class BrowserOpenLink(_BrowserTool[ElementArgs]):
    """Follow a link from the page listing (opens its https:// address)."""

    name = "browser_open_link"
    description = (
        "Open a link from the browser_read_page listing in the current tab, e.g. a video "
        "or an article. Give its number and its text."
    )
    tier = Tier.REVERSIBLE
    Args = ElementArgs
    untrusted_output = True

    async def run(self, args: ElementArgs) -> str:
        """Open it."""
        browser = await self._browsers.pick()
        link, text, href = await self._element(browser, args)
        if not link:
            msg = f"Item {args.number} is a button, not a link: use browser_press."
            raise ToolError(msg)
        url = safe_url(href)  # never javascript:, file: or http: (the page wrote the href)
        await self._script(_navigate_script(browser), url, failure="The link did not open.")
        return f"Opened “{text}”."

    def starts_playback(self, args: ElementArgs) -> bool:
        """A link can be a video that starts by itself."""
        return True

    def describe(self, args: ElementArgs) -> str:
        """'open the link “…”'."""
        return f"open the link “{args.label}”"


class BrowserPress(_BrowserTool[ElementArgs]):
    """Press a button on the page. Buttons can send, buy or delete, so this is asked first."""

    name = "browser_press"
    description = (
        "Press a button from the browser_read_page listing. Give its number and its text. "
        "The user is asked first."
    )
    tier = Tier.EXTERNAL
    Args = ElementArgs
    untrusted_output = True

    async def run(self, args: ElementArgs) -> str:
        """Press it."""
        browser = await self._browsers.pick()
        _, text, _ = await self._element(browser, args)
        if await self._javascript(browser, _press_js(args.number)) != "ok":
            msg = "The page has changed since it was read. Read it again first."
            raise ToolError(msg)
        return f"Pressed “{text}”."

    def describe(self, args: ElementArgs) -> str:
        """'press “Subscribe” on the page'."""
        return f"press “{args.label}” on the page"


class BrowserMedia(_BrowserTool[MediaArgs]):
    """Play or pause the video or audio on the current page."""

    name = "browser_media"
    description = "Play or pause the video or audio on the page in the current browser tab."
    tier = Tier.REVERSIBLE
    Args = MediaArgs

    async def run(self, args: MediaArgs) -> str:
        """Play or pause."""
        browser = await self._browsers.pick()
        state = await self._javascript(browser, _media_js(args.action))
        if state == "none":
            msg = "There is no video or audio on this page."
            raise ToolError(msg)
        return "Playing." if state == "playing" else "Paused."

    def starts_playback(self, args: MediaArgs) -> bool:
        """Play and toggle (may) start the video; pause does not."""
        return args.action != "pause"

    def describe(self, args: MediaArgs) -> str:
        """'pause the video'."""
        return {"play": "play", "pause": "pause", "toggle": "play or pause"}[args.action] + (
            " the video"
        )


def browser_tools(run: Runner, browsers: Browsers) -> list[Tool[Any]]:
    """All browser tools."""
    return [
        BrowserTabs(run, browsers),
        BrowserSwitchTab(run, browsers),
        BrowserOpen(run, browsers),
        BrowserSearch(run, browsers),
        BrowserCloseTab(run, browsers),
        BrowserGo(run, browsers),
        BrowserReadPage(run, browsers),
        BrowserOpenLink(run, browsers),
        BrowserPress(run, browsers),
        BrowserMedia(run, browsers),
    ]
