"""Calendars: the tools with a fake store, and the EventKit store with fake EventKit."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import date, datetime
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, ClassVar

import pytest

from test_computer import FakeRunner
from yarvis.config.models import Config
from yarvis.tools import calendars
from yarvis.tools.base import Tier, ToolError
from yarvis.tools.calendars import (
    AddArgs,
    CalendarAdd,
    CalendarEvents,
    CalendarInfo,
    Event,
    EventKitStore,
    EventsArgs,
    NewEvent,
    calendar_tools,
)
from yarvis.tools.desktop import abilities, personal_tools

if TYPE_CHECKING:
    from collections.abc import Callable

TODAY = date(2026, 11, 20)  # a Friday
HOME = CalendarInfo("h", "Home", "victortje10@gmail.com", writable=True)
WORK = CalendarInfo("w", "Work", "victorclaeyson@gmail.com", writable=True)
KUL = CalendarInfo("k", "Calendar", "KU Leuven", writable=True)
HOLIDAYS = CalendarInfo("x", "Holidays in Belgium", "victortje10@gmail.com", writable=False)


def naive(year: int, month: int, day: int, hour: int, minute: int) -> datetime:
    """A time as the model gives it: no zone, meaning the user's local time."""
    return datetime(year, month, day, hour, minute)  # noqa: DTZ001


def at(day: int, hour: int, minute: int = 0, month: int = 11) -> datetime:
    return datetime(2026, month, day, hour, minute).astimezone()


@dataclass
class FakeStore:
    found: list[Event] = field(default_factory=list[Event])
    cals: list[CalendarInfo] = field(default_factory=lambda: [HOME, WORK, KUL, HOLIDAYS])
    default: CalendarInfo | None = HOME
    asked: list[tuple[datetime, datetime]] = field(default_factory=list[tuple[datetime, datetime]])
    added: list[NewEvent] = field(default_factory=list[NewEvent])

    async def events(self, start: datetime, end: datetime) -> list[Event]:
        self.asked.append((start, end))
        return [e for e in self.found if e.start < end and e.end > start]

    async def calendars(self) -> list[CalendarInfo]:
        return self.cals

    async def default_calendar(self) -> CalendarInfo | None:
        return self.default

    async def add(self, event: NewEvent) -> Event:
        self.added.append(event)
        return Event(event.title, event.start, event.end, event.all_day, event.location, "Home")


def events_tool(store: FakeStore) -> CalendarEvents:
    return CalendarEvents(store, today=lambda: TODAY)


def add_tool(store: FakeStore) -> CalendarAdd:
    return CalendarAdd(store, today=lambda: TODAY)


# --------------------------------------------------------------------------- reading


async def test_todays_events_read_like_an_agenda() -> None:
    store = FakeStore(
        [
            Event("Analyse I", at(20, 9), at(20, 11), location="200C 00.01", calendar="KULeuven"),
            Event("Dentist", at(20, 15), at(20, 16)),
            Event("Mom's birthday", at(20, 0), at(21, 0), all_day=True, calendar="Birthdays"),
        ]
    )
    tool = events_tool(store)
    said = await tool.run(EventsArgs())
    assert said == (
        "Friday 20 November:\n"
        "- 09:00 to 11:00 Analyse I, at 200C 00.01 (KULeuven)\n"
        "- 15:00 to 16:00 Dentist\n"
        "- all day Mom's birthday (Birthdays)"
    )
    assert store.asked == [(at(20, 0), at(21, 0))]  # local midnight to midnight
    assert tool.untrusted_output  # invitations are written by other people
    assert tool.tier is Tier.READ
    assert tool.describe(EventsArgs()) == "check your calendar for today"


async def test_a_week_names_its_empty_days() -> None:
    store = FakeStore([Event("Late shift", at(21, 22), at(22, 2))])
    said = await events_tool(store).run(EventsArgs(start=date(2026, 11, 21), days=3))
    assert said == (
        "Saturday 21 November:\n"
        "- 22:00 until Sunday 22 November 02:00 Late shift\n"
        "Sunday 22 November: nothing.\n"
        "Monday 23 November: nothing."
    )


async def test_a_long_range_only_names_busy_days() -> None:
    store = FakeStore([Event("Exam", at(5, 9, month=12), at(5, 12, month=12))])
    tool = events_tool(store)
    said = await tool.run(EventsArgs(days=31))
    assert said == "Saturday 5 December:\n- 09:00 to 12:00 Exam"
    assert await events_tool(FakeStore()).run(EventsArgs(days=10)) == (
        "Nothing in the 10 days from Friday 20 November."
    )
    assert tool.describe(EventsArgs(start=date(2027, 1, 4), days=10)) == (
        "check your calendar for 10 days from Monday 4 January 2027"
    )


async def test_an_event_that_started_earlier_is_listed_on_the_first_day() -> None:
    store = FakeStore([Event("Ski trip", at(15, 0), at(25, 0), all_day=True)])
    said = await events_tool(store).run(EventsArgs())
    assert said == "Friday 20 November:\n- all day Ski trip"


async def test_very_busy_calendars_are_cut_short() -> None:
    many = [Event(f"Slot {i}", at(20, 8), at(20, 9)) for i in range(calendars.MAX_EVENTS + 5)]
    said = await events_tool(FakeStore(many)).run(EventsArgs())
    assert said.endswith("(and 5 more events)")
    assert said.count("- 08:00") == calendars.MAX_EVENTS


async def test_dates_far_away_are_refused() -> None:
    with pytest.raises(ToolError, match="too far away"):
        await events_tool(FakeStore()).run(EventsArgs(start=date(2030, 1, 1)))


# --------------------------------------------------------------------------- adding


async def test_adding_asks_first_and_names_the_calendar() -> None:
    store = FakeStore()
    tool = add_tool(store)
    asked = AddArgs(title="Dentist", start=naive(2026, 11, 26, 15, 0), location="Tiensestraat 1")
    assert tool.tier is Tier.EXTERNAL
    assert tool.describe(asked) == (
        "add “Dentist” to your calendar on Thursday 26 November, 15:00 to 16:00, at Tiensestraat 1"
    )
    ready = await tool.prepare(asked)
    assert ready.calendar == "Home (victortje10@gmail.com)"
    assert tool.describe(ready).startswith("add “Dentist” to Home (victortje10@gmail.com) on")
    said = await tool.run(ready)
    assert store.added == [
        NewEvent("Dentist", at(26, 15), at(26, 16), all_day=False, location="Tiensestraat 1",
                 calendar_id="h"),
    ]  # fmt: skip
    assert said == (
        "Added to Home (victortje10@gmail.com): 15:00 to 16:00 Dentist, at Tiensestraat 1 (Home) "
        "on Thursday 26 November."
    )


async def test_all_day_and_overnight_events() -> None:
    store = FakeStore()
    tool = add_tool(store)
    trip = AddArgs(title="Trip", start=naive(2026, 12, 1, 10, 0), all_day=True, minutes=3 * 1440)
    assert tool.describe(trip) == "add “Trip” to your calendar on Tuesday 1 December, all day"
    await tool.run(await tool.prepare(trip))
    assert store.added[0].start == at(1, 0, month=12)
    assert store.added[0].end == at(4, 0, month=12)
    assert store.added[0].all_day
    party = AddArgs(title="Party", start=naive(2026, 12, 31, 22, 0), minutes=240)
    assert tool.describe(party) == (
        "add “Party” to your calendar on Thursday 31 December, 22:00 to Friday 1 January 2027 02:00"
    )


@pytest.mark.parametrize(
    ("wanted", "label"),
    [
        ("work", "Work (victorclaeyson@gmail.com)"),
        ("Calendar (KU Leuven)", "Calendar (KU Leuven)"),
        ("KU Leuven", "Calendar (KU Leuven)"),
        ("victorclaeyson", "Work (victorclaeyson@gmail.com)"),
    ],
)
async def test_calendars_are_found_by_name_or_account(wanted: str, label: str) -> None:
    ready = await add_tool(FakeStore()).prepare(
        AddArgs(title="x", start=naive(2026, 11, 26, 9, 0), calendar=wanted)
    )
    assert ready.calendar == label


@pytest.mark.parametrize(
    ("store", "wanted", "message"),
    [
        (FakeStore(), "gmail", "No single calendar matches 'gmail'"),
        (FakeStore(), "Holidays", "No single calendar matches 'Holidays'"),  # read-only
        (FakeStore(default=None), "", "Ask the user which calendar: Home"),
        (FakeStore(default=HOLIDAYS), "", "Ask the user which calendar"),
        (FakeStore(cals=[HOLIDAYS]), "x", "None of the calendars on this Mac can be written to."),
    ],
)
async def test_unclear_calendars_are_asked_about(
    store: FakeStore, wanted: str, message: str
) -> None:
    with pytest.raises(ToolError, match=message):
        await add_tool(store).prepare(
            AddArgs(title="x", start=naive(2026, 11, 26, 9, 0), calendar=wanted)
        )


async def test_the_only_writable_calendar_is_the_default() -> None:
    store = FakeStore(cals=[HOLIDAYS, WORK], default=None)
    ready = await add_tool(store).prepare(AddArgs(title="x", start=naive(2026, 11, 26, 9, 0)))
    assert ready.calendar == WORK.label


async def test_adding_far_away_is_refused_before_asking() -> None:
    with pytest.raises(ToolError, match="too far away"):
        await add_tool(FakeStore()).prepare(AddArgs(title="x", start=naive(2031, 1, 1, 9, 0)))
    with pytest.raises(ToolError, match="too far away"):
        await add_tool(FakeStore()).run(
            AddArgs(title="x", start=naive(2031, 1, 1, 9, 0), calendar=HOME.label)
        )


# --------------------------------------------------------------------------- EventKit


def stamp(moment: datetime) -> float:
    return moment.timestamp()


class NSDate:
    def __init__(self, seconds: float) -> None:
        self.seconds = seconds

    @classmethod
    def dateWithTimeIntervalSince1970_(cls, seconds: float) -> NSDate:  # noqa: N802
        return cls(seconds)

    def timeIntervalSince1970(self) -> float:  # noqa: N802
        return self.seconds


class Source:
    def __init__(self, title: str) -> None:
        self._title = title

    def title(self) -> str:
        return self._title


class EKCalendar:
    def __init__(self, ident: str, title: str, source: str | None, *, writable: bool) -> None:
        self._id, self._title, self._writable = ident, title, writable
        self._source = None if source is None else Source(source)

    def calendarIdentifier(self) -> str:  # noqa: N802
        return self._id

    def title(self) -> str:
        return self._title

    def source(self) -> Source | None:
        return self._source

    def allowsContentModifications(self) -> bool:  # noqa: N802
        return self._writable


class EKEvent:
    def __init__(self, title: str | None = None, start: float = 0, end: float = 0) -> None:
        self._title: str | None = title
        self._start, self._end = NSDate(start), NSDate(end)
        self._all_day = False
        self._location: str | None = None
        self._calendar: EKCalendar | None = None

    @classmethod
    def eventWithEventStore_(cls, store: object) -> EKEvent:  # noqa: N802
        return cls()

    def title(self) -> str | None:
        return self._title

    def startDate(self) -> NSDate:  # noqa: N802
        return self._start

    def endDate(self) -> NSDate:  # noqa: N802
        return self._end

    def isAllDay(self) -> bool:  # noqa: N802
        return self._all_day

    def location(self) -> str | None:
        return self._location

    def calendar(self) -> EKCalendar | None:
        return self._calendar

    def setTitle_(self, value: str) -> None:  # noqa: N802
        self._title = value

    def setStartDate_(self, value: NSDate) -> None:  # noqa: N802
        self._start = value

    def setEndDate_(self, value: NSDate) -> None:  # noqa: N802
        self._end = value

    def setAllDay_(self, value: bool) -> None:  # noqa: N802
        self._all_day = value

    def setLocation_(self, value: str) -> None:  # noqa: N802
        self._location = value

    def setCalendar_(self, value: EKCalendar) -> None:  # noqa: N802
        self._calendar = value


GMAIL = EKCalendar("g", "Home", "victortje10@gmail.com", writable=True)
LOCAL = EKCalendar("l", "Local", None, writable=False)


class EKEventStore:
    status: ClassVar[int] = 3
    answer: ClassVar[bool | None] = True
    requests: ClassVar[int] = 0
    accept: ClassVar[bool] = True
    events: ClassVar[list[EKEvent]] = []
    saved: ClassVar[list[EKEvent]] = []

    @classmethod
    def authorizationStatusForEntityType_(cls, kind: int) -> int:  # noqa: N802
        return cls.status

    @classmethod
    def alloc(cls) -> EKEventStore:
        return cls()

    def init(self) -> EKEventStore:
        return self

    def requestFullAccessToEventsWithCompletion_(  # noqa: N802
        self, done: Callable[[object, object], None]
    ) -> None:
        type(self).requests += 1
        if type(self).answer is not None:
            done(type(self).answer, None)

    def predicateForEventsWithStartDate_endDate_calendars_(  # noqa: N802
        self, start: NSDate, end: NSDate, only: object
    ) -> tuple[float, float]:
        return start.seconds, end.seconds

    def eventsMatchingPredicate_(self, span: tuple[float, float]) -> list[EKEvent]:  # noqa: N802
        return [
            e
            for e in self.events
            if e.startDate().seconds < span[1] and e.endDate().seconds > span[0]
        ]

    def calendarsForEntityType_(self, kind: int) -> list[EKCalendar]:  # noqa: N802
        return [GMAIL, LOCAL]

    def defaultCalendarForNewEvents(self) -> EKCalendar | None:  # noqa: N802
        return GMAIL if type(self).status == 3 else None

    def calendarWithIdentifier_(self, ident: str) -> EKCalendar | None:  # noqa: N802
        return {"g": GMAIL, "l": LOCAL}.get(ident)

    def saveEvent_span_commit_error_(  # noqa: N802
        self, event: EKEvent, span: int, commit: bool, error: None
    ) -> tuple[bool, None]:
        self.saved.append(event)
        return type(self).accept, None


@pytest.fixture
def kit() -> type[EKEventStore]:
    class Store(EKEventStore):
        events: ClassVar[list[EKEvent]] = []
        saved: ClassVar[list[EKEvent]] = []

    return Store


def eventkit(store: type[EKEventStore]) -> EventKitStore:
    ek = SimpleNamespace(
        EKEventStore=store, EKEvent=EKEvent, EKEntityTypeEvent=0, EKSpanThisEvent=0
    )
    return EventKitStore(eventkit=ek, foundation=SimpleNamespace(NSDate=NSDate))


async def test_eventkit_reads_events_and_calendars(kit: type[EKEventStore]) -> None:
    lecture = EKEvent("Analyse I", stamp(at(20, 9)), stamp(at(20, 11)))
    lecture.setCalendar_(GMAIL)
    lecture.setLocation_("200C")
    untitled = EKEvent(None, stamp(at(20, 7)), stamp(at(20, 8)))
    kit.events = [lecture, untitled, EKEvent("Tomorrow", stamp(at(21, 9)), stamp(at(21, 10)))]
    store = eventkit(kit)
    found = await store.events(at(20, 0), at(21, 0))
    assert found == [
        Event("(no title)", at(20, 7), at(20, 8)),
        Event("Analyse I", at(20, 9), at(20, 11), location="200C", calendar="Home"),
    ]
    assert await store.calendars() == [
        CalendarInfo("g", "Home", "victortje10@gmail.com", writable=True),
        CalendarInfo("l", "Local", "", writable=False),
    ]
    assert await store.default_calendar() == CalendarInfo(
        "g", "Home", "victortje10@gmail.com", writable=True
    )
    assert kit.requests == 0  # access was already given


async def test_eventkit_adds_events(kit: type[EKEventStore]) -> None:
    store = eventkit(kit)
    new = NewEvent("Dentist", at(26, 15), at(26, 16), all_day=False, location="", calendar_id="g")
    added = await store.add(new)
    assert added == Event("Dentist", at(26, 15), at(26, 16), calendar="Home")
    assert kit.saved[0].location() is None  # no empty location written
    with pytest.raises(ToolError, match="no longer on this Mac"):
        await store.add(NewEvent("x", at(26, 15), at(26, 16), all_day=True, location="",
                                 calendar_id="gone"))  # fmt: skip
    kit.accept = False
    with pytest.raises(ToolError, match="did not accept"):
        await store.add(NewEvent("x", at(26, 15), at(26, 16), all_day=True, location="Home",
                                 calendar_id="l"))  # fmt: skip


async def test_eventkit_asks_macos_the_first_time(kit: type[EKEventStore]) -> None:
    kit.status = 0
    store = eventkit(kit)
    assert await store.calendars()
    assert kit.requests == 1
    assert await store.default_calendar() is None
    assert kit.requests == 1  # asked once


@pytest.mark.parametrize(("status", "answer"), [(2, True), (0, False), (0, None)])
async def test_eventkit_without_access_explains_how_to_allow_it(
    kit: type[EKEventStore], status: int, answer: bool | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(calendars, "_ACCESS_WAIT_S", 0.01)
    kit.status, kit.answer = status, answer
    with pytest.raises(ToolError, match="Privacy & Security → Calendars"):
        await eventkit(kit).calendars()


async def test_eventkit_runs_on_its_own_thread(kit: type[EKEventStore]) -> None:
    store = eventkit(kit)
    names = await asyncio.gather(*(store.calendars() for _ in range(3)))
    assert all(found == names[0] for found in names)


# --------------------------------------------------------------------------- wiring


def test_calendar_and_mail_come_with_the_mac() -> None:
    config = Config()
    store = FakeStore()
    names = [
        t.name for t in personal_tools(config, platform="darwin", run=FakeRunner(), store=store)
    ]
    assert names == ["calendar_events", "calendar_add", "mail_list", "mail_read", "mail_send",
                     "mail_reply"]  # fmt: skip
    assert personal_tools(config, platform="linux") == []
    off = Config.model_validate({"tools": {"calendar": False, "mail": False}})
    assert personal_tools(off, platform="darwin", run=FakeRunner(), store=store) == []
    said = " ".join(abilities(personal_tools(config, platform="darwin", run=FakeRunner(),
                                             store=store)))  # fmt: skip
    assert "read every calendar the user has on this Mac" in said
    assert "never the addresses" in said


def test_calendar_tools_share_one_store() -> None:
    tools: list[Any] = calendar_tools(FakeStore())
    assert [t.name for t in tools] == ["calendar_events", "calendar_add"]
    assert all(t.timeout_s >= 40 for t in tools)  # room for the user to answer macOS's dialog
