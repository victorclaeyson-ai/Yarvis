"""Your calendars, through the Mac's own calendar store (EventKit).

Every calendar added to the Mac (Google, Outlook, iCloud, a subscribed timetable) is read
the same way, with repeating events expanded. macOS signs in to those accounts and keeps
their tokens: Yarvis never sees a password or a token, and needs no developer account.
The first use asks the user, in a macOS dialog, to let their terminal see the calendars.

Events are written by other people too (invitations), so what is read is outside content.
Adding an event needs the user's "yes": it may land in a shared calendar.
"""

from __future__ import annotations

import asyncio
import importlib
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from typing import TYPE_CHECKING, Any, Final, Protocol

from pydantic import Field

from yarvis.tools.base import Arguments, Tier, Tool, ToolError

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

MAX_DAYS: Final = 31
MAX_EVENTS: Final = 60
_REACH_DAYS: Final = 2 * 366
_ACCESS_WAIT_S: Final = 40.0
_NO_ACCESS: Final = (
    "Yarvis may not see the calendars. The user can allow it in System Settings → Privacy "
    "& Security → Calendars (turn on the app Yarvis is started from, with full access)."
)
_WEEKDAYS: Final = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
_MONTHS: Final = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)  # fmt: skip

# EventKit's EKAuthorizationStatus values.
_FULL_ACCESS: Final = 3


@dataclass(frozen=True, slots=True)
class Event:
    """One event (or one occurrence of a repeating event)."""

    title: str
    start: datetime
    end: datetime
    all_day: bool = False
    location: str = ""
    calendar: str = ""


@dataclass(frozen=True, slots=True)
class CalendarInfo:
    """A calendar on this Mac."""

    id: str
    title: str
    account: str
    writable: bool

    @property
    def label(self) -> str:
        """How the user knows it: "Home (victor@gmail.com)"."""
        return f"{self.title} ({self.account})" if self.account else self.title


@dataclass(frozen=True, slots=True)
class NewEvent:
    """An event to add."""

    title: str
    start: datetime
    end: datetime
    all_day: bool
    location: str
    calendar_id: str


class CalendarStore(Protocol):
    """Where the events live."""

    async def events(self, start: datetime, end: datetime) -> list[Event]:
        """Events overlapping [start, end), sorted by start."""
        ...

    async def calendars(self) -> list[CalendarInfo]:
        """Every calendar."""
        ...

    async def default_calendar(self) -> CalendarInfo | None:
        """Where new events go unless told otherwise."""
        ...

    async def add(self, event: NewEvent) -> Event:
        """Save a new event.

        Raises:
            ToolError: If it could not be saved.
        """
        ...


class EventKitStore:
    """The Mac's calendars, through PyObjC's EventKit bindings.

    EventKit calls block, so they run on one dedicated thread.
    """

    def __init__(self, *, eventkit: Any = None, foundation: Any = None) -> None:  # noqa: ANN401
        self._ek = eventkit
        self._ns = foundation
        self._store: Any = None
        self._thread = ThreadPoolExecutor(max_workers=1, thread_name_prefix="eventkit")

    async def events(self, start: datetime, end: datetime) -> list[Event]:
        """Events overlapping [start, end), repeating ones expanded."""
        return await self._call(self._events, start, end)

    async def calendars(self) -> list[CalendarInfo]:
        """Every event calendar."""
        return await self._call(self._calendars)

    async def default_calendar(self) -> CalendarInfo | None:
        """The calendar the Mac puts new events in."""
        return await self._call(self._default)

    async def add(self, event: NewEvent) -> Event:
        """Save a new event."""
        return await self._call(self._add, event)

    async def _call[R](self, function: Callable[..., R], *args: object) -> R:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._thread, function, *args)

    # Everything below runs on the EventKit thread.

    def _modules(self) -> tuple[Any, Any]:
        if self._ek is None:  # pragma: no cover - only on a Mac, with PyObjC installed
            # Imported by name: PyObjC exists only on macOS, and type checkers run on both.
            self._ek = importlib.import_module("EventKit")
            self._ns = importlib.import_module("Foundation")
        return self._ek, self._ns

    def _ready(self) -> Any:  # noqa: ANN401 - an EKEventStore
        if self._store is not None:
            return self._store
        ek, _ = self._modules()
        status = int(ek.EKEventStore.authorizationStatusForEntityType_(ek.EKEntityTypeEvent))
        if status != _FULL_ACCESS:
            if status != 0:  # denied, restricted or write-only: only the user can change it
                raise ToolError(_NO_ACCESS)
            self._ask(ek)
        self._store = ek.EKEventStore.alloc().init()
        return self._store

    @staticmethod
    def _ask(ek: Any) -> None:  # noqa: ANN401
        """Show macOS's own dialog; the user decides."""
        answered = threading.Event()
        granted: list[bool] = []

        def done(ok: object, _error: object) -> None:
            granted.append(bool(ok))
            answered.set()

        logger.info("Asking macOS for access to the calendars")
        ek.EKEventStore.alloc().init().requestFullAccessToEventsWithCompletion_(done)
        if not answered.wait(_ACCESS_WAIT_S) or not granted[0]:
            raise ToolError(_NO_ACCESS)

    def _events(self, start: datetime, end: datetime) -> list[Event]:
        store = self._ready()
        _, ns = self._modules()
        predicate = store.predicateForEventsWithStartDate_endDate_calendars_(
            _nsdate(ns, start), _nsdate(ns, end), None
        )
        found = [_event(item) for item in store.eventsMatchingPredicate_(predicate) or ()]
        return sorted(found, key=lambda event: (event.start, event.title))

    def _calendars(self) -> list[CalendarInfo]:
        store = self._ready()
        ek, _ = self._modules()
        return [_calendar(item) for item in store.calendarsForEntityType_(ek.EKEntityTypeEvent)]

    def _default(self) -> CalendarInfo | None:
        found = self._ready().defaultCalendarForNewEvents()
        return None if found is None else _calendar(found)

    def _add(self, new: NewEvent) -> Event:
        store = self._ready()
        ek, ns = self._modules()
        calendar = store.calendarWithIdentifier_(new.calendar_id)
        if calendar is None:
            msg = "That calendar is no longer on this Mac."
            raise ToolError(msg)
        event = ek.EKEvent.eventWithEventStore_(store)
        event.setTitle_(new.title)
        event.setStartDate_(_nsdate(ns, new.start))
        event.setEndDate_(_nsdate(ns, new.end))
        event.setAllDay_(new.all_day)
        if new.location:
            event.setLocation_(new.location)
        event.setCalendar_(calendar)
        saved, _error = store.saveEvent_span_commit_error_(event, ek.EKSpanThisEvent, True, None)  # noqa: FBT003
        if not saved:
            msg = "The calendar did not accept the event (it may be read-only)."
            raise ToolError(msg)
        return _event(event)


def _nsdate(ns: Any, moment: datetime) -> Any:  # noqa: ANN401
    return ns.NSDate.dateWithTimeIntervalSince1970_(moment.timestamp())


def _local(nsdate: Any) -> datetime:  # noqa: ANN401
    return datetime.fromtimestamp(float(nsdate.timeIntervalSince1970()), tz=UTC).astimezone()


def _text(value: object) -> str:
    return "" if value is None else str(value)


def _event(item: Any) -> Event:  # noqa: ANN401 - an EKEvent
    calendar = item.calendar()
    return Event(
        title=_text(item.title()) or "(no title)",
        start=_local(item.startDate()),
        end=_local(item.endDate()),
        all_day=bool(item.isAllDay()),
        location=_text(item.location()),
        calendar="" if calendar is None else _text(calendar.title()),
    )


def _calendar(item: Any) -> CalendarInfo:  # noqa: ANN401 - an EKCalendar
    source = item.source()
    return CalendarInfo(
        id=_text(item.calendarIdentifier()),
        title=_text(item.title()),
        account="" if source is None else _text(source.title()),
        writable=bool(item.allowsContentModifications()),
    )


# --------------------------------------------------------------------------- the tools


class EventsArgs(Arguments):
    """Arguments of ``calendar_events``."""

    start: date | None = Field(default=None, description="First day (YYYY-MM-DD). Empty: today.")
    days: int = Field(default=1, ge=1, le=MAX_DAYS, description="How many days, from start.")


class AddArgs(Arguments):
    """Arguments of ``calendar_add``."""

    title: str = Field(min_length=1, max_length=120)
    start: datetime = Field(description="Local date and time, e.g. 2026-11-20T15:00.")
    minutes: int = Field(default=60, ge=5, le=7 * 24 * 60, description="How long it lasts.")
    all_day: bool = Field(default=False, description="An all-day event on the start date.")
    location: str = Field(default="", max_length=200)
    calendar: str = Field(
        default="",
        max_length=120,
        description="Which calendar, by name or account. Empty: the default one.",
    )


class _CalendarTool[A: Arguments](Tool[A]):
    timeout_s = _ACCESS_WAIT_S + 5  # the first call may wait for the user's answer in a dialog

    def __init__(self, store: CalendarStore, *, today: Callable[[], date] = date.today) -> None:
        self._store = store
        self._today = today

    def _within_reach(self, day: date) -> None:
        if abs((day - self._today()).days) > _REACH_DAYS:
            msg = "That is too far away; ask about a date within two years."
            raise ToolError(msg)


class CalendarEvents(_CalendarTool[EventsArgs]):
    """What is in the calendars."""

    name = "calendar_events"
    description = (
        "The events in all the user's calendars (Google, Outlook, iCloud, ...) for a day "
        "or a range of days, repeating events included."
    )
    tier = Tier.READ
    Args = EventsArgs
    untrusted_output = True  # invitations are written by other people

    async def run(self, args: EventsArgs) -> str:
        """Look them up."""
        first = args.start or self._today()
        self._within_reach(first)
        start = _midnight(first)
        events = await self._store.events(start, _midnight(first + timedelta(days=args.days)))
        return _agenda(events, first, args.days, self._today().year)

    def describe(self, args: EventsArgs) -> str:
        """'check your calendar for today'."""
        when = "today" if args.start is None else _day(args.start, self._today().year)
        if args.days > 1:
            return f"check your calendar for {args.days} days from {when}"
        return f"check your calendar for {when}"


class CalendarAdd(_CalendarTool[AddArgs]):
    """Add an event (after the user's "yes")."""

    name = "calendar_add"
    description = (
        "Add an event to one of the user's calendars. The user is asked to confirm first; "
        "say which calendar only if they named one."
    )
    tier = Tier.EXTERNAL
    Args = AddArgs

    async def prepare(self, args: AddArgs) -> AddArgs:
        """Settle which calendar, so the question names it."""
        self._within_reach(_local_time(args.start).date())
        calendar = await self._pick(args.calendar)
        return args.model_copy(update={"calendar": calendar.label})

    async def run(self, args: AddArgs) -> str:
        """Add it."""
        start = _local_time(args.start)
        self._within_reach(start.date())
        calendar = await self._pick(args.calendar)
        if args.all_day:
            start = _midnight(start.date())
            end = start + timedelta(days=max(1, args.minutes // (24 * 60)))
        else:
            end = start + timedelta(minutes=args.minutes)
        new = NewEvent(args.title, start, end, args.all_day, args.location, calendar.id)
        added = await self._store.add(new)
        year = self._today().year
        return (
            f"Added to {calendar.label}: {_line(added, year)} on {_day(added.start.date(), year)}."
        )

    def describe(self, args: AddArgs) -> str:
        """'add “Dentist” to Home (victor@gmail.com) on Thursday 20 November, 15:00 to 16:00'."""
        start = _local_time(args.start)
        year = self._today().year
        where = args.calendar or "your calendar"
        if args.all_day:
            when = f"on {_day(start.date(), year)}, all day"
        else:
            end = start + timedelta(minutes=args.minutes)
            same_day = end.date() == start.date()
            until = f"{end:%H:%M}" if same_day else f"{_day(end.date(), year)} {end:%H:%M}"
            when = f"on {_day(start.date(), year)}, {start:%H:%M} to {until}"
        at = f", at {args.location}" if args.location else ""
        return f"add “{args.title}” to {where} {when}{at}"

    async def _pick(self, wanted: str) -> CalendarInfo:
        """The calendar called ``wanted`` (its label, title or account), or the default."""
        writable = [c for c in await self._store.calendars() if c.writable]
        if not wanted:
            default = await self._store.default_calendar()
            if default is not None and default.writable:
                return default
            if len(writable) == 1:
                return writable[0]
            msg = f"Ask the user which calendar: {_options(writable)}."
            raise ToolError(msg)
        key = wanted.casefold()
        exact = [c for c in writable if key in (c.label.casefold(), c.title.casefold())]
        partial = [c for c in writable if key in c.label.casefold()]
        matches = exact or partial
        if len(matches) == 1:
            return matches[0]
        if not writable:
            msg = "None of the calendars on this Mac can be written to."
            raise ToolError(msg)
        msg = f"No single calendar matches {wanted!r}. The calendars: {_options(writable)}."
        raise ToolError(msg)


def calendar_tools(store: CalendarStore) -> list[Tool[Any]]:
    """Reading and adding events."""
    return [CalendarEvents(store), CalendarAdd(store)]


# --------------------------------------------------------------------------- words


def _midnight(day: date) -> datetime:
    return datetime.combine(day, time()).astimezone()


def _local_time(moment: datetime) -> datetime:
    """A time the model gave: without a zone, it is the user's local time."""
    return moment.astimezone()


def _day(day: date, this_year: int | None = None) -> str:
    """'Thursday 20 November' (with the year only when it is not this year)."""
    year = "" if day.year == this_year else f" {day.year}"
    return f"{_WEEKDAYS[day.weekday()]} {day.day} {_MONTHS[day.month - 1]}{year}"


def _line(event: Event, this_year: int) -> str:
    if event.all_day:
        when = "all day"
    elif event.end.date() == event.start.date():
        when = f"{event.start:%H:%M} to {event.end:%H:%M}"
    else:
        when = f"{event.start:%H:%M} until {_day(event.end.date(), this_year)} {event.end:%H:%M}"
    at = f", at {event.location[:80]}" if event.location else ""
    calendar = f" ({event.calendar[:40]})" if event.calendar else ""
    return f"{when} {event.title[:120]}{at}{calendar}"


def _agenda(events: list[Event], first: date, days: int, this_year: int) -> str:
    by_day: dict[date, list[Event]] = {}
    for event in events[:MAX_EVENTS]:
        by_day.setdefault(max(event.start.date(), first), []).append(event)
    lines: list[str] = []
    for offset in range(days):
        day = first + timedelta(days=offset)
        if day in by_day:
            lines.append(f"{_day(day, this_year)}:")
            lines += [f"- {_line(event, this_year)}" for event in by_day[day]]
        elif days <= 7:  # noqa: PLR2004 - a week is short enough to name every empty day
            lines.append(f"{_day(day, this_year)}: nothing.")
    if not by_day and days > 7:  # noqa: PLR2004
        lines.append(f"Nothing in the {days} days from {_day(first, this_year)}.")
    if len(events) > MAX_EVENTS:
        lines.append(f"(and {len(events) - MAX_EVENTS} more events)")
    return "\n".join(lines)


def _options(calendars: list[CalendarInfo]) -> str:
    return "; ".join(c.label for c in calendars[:12]) or "none"
