from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from yarvis.tools.base import ToolError
from yarvis.tools.timers import (
    MAX_ACTIVE,
    Alarms,
    CancelArgs,
    CancelTimer,
    ListTimers,
    NoArgs,
    ReminderArgs,
    SetReminder,
    SetTimer,
    TimerArgs,
    duration,
    timer_tools,
)

START = datetime(2026, 11, 15, 14, 0, tzinfo=UTC)


class FakeTime:
    """A clock that only moves when the alarms sleep (plus optional jumps)."""

    def __init__(self) -> None:
        self.now = START
        self.jumps: list[timedelta] = []

    def __call__(self) -> datetime:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds) + (self.jumps.pop(0) if self.jumps else timedelta())
        await asyncio.sleep(0)


def alarms(said: list[str], time: FakeTime) -> Alarms:
    return Alarms(said.append, now=time, sleep=time.sleep)


async def settle() -> None:
    for _ in range(50):
        await asyncio.sleep(0)


@pytest.mark.parametrize(
    ("seconds", "text"),
    [(30, "30-second"), (60, "1-minute"), (600, "10-minute"), (5400, "1-hour-30-minute"),
     (7200, "2-hour"), (89.6, "1-minute-30-second"), (3630, "1-hour"), (0.2, "0-second")],
)  # fmt: skip
def test_duration(seconds: float, text: str) -> None:
    assert duration(seconds) == text


async def test_a_timer_is_announced_when_it_ends() -> None:
    said: list[str] = []
    time = FakeTime()
    service = alarms(said, time)
    service.start_timer(600)
    service.start_timer(90, "pasta")
    assert [alarm.label for alarm in service.active()] == ["pasta", ""]
    await settle()
    assert said == ["Your pasta timer is done.", "Your 10-minute timer is done."]
    assert service.active() == []


async def test_reminders_survive_the_laptop_sleeping() -> None:
    said: list[str] = []
    time = FakeTime()
    time.jumps = [timedelta(hours=2)]  # the clock jumps while asleep
    service = alarms(said, time)
    service.remind(START + timedelta(hours=1), "call mom")
    await settle()
    assert said == ["Reminder: call mom."]
    assert time.now < START + timedelta(hours=2, minutes=1)  # fired on waking, not later


async def test_reminders_must_be_in_the_next_week() -> None:
    service = alarms([], FakeTime())
    with pytest.raises(ToolError, match="already passed"):
        service.remind(START - timedelta(minutes=1), "x")
    with pytest.raises(ToolError, match="seven days"):
        service.remind(START + timedelta(days=8), "x")


async def test_cancel_and_close() -> None:
    said: list[str] = []
    service = alarms(said, FakeTime())
    timer = service.start_timer(60)
    assert service.cancel(timer.id) == timer
    with pytest.raises(ToolError, match="no timer or reminder number"):
        service.cancel(timer.id)
    service.start_timer(60)
    service.close()
    await settle()
    assert said == []
    assert service.active() == []


async def test_there_is_a_limit() -> None:
    service = alarms([], FakeTime())
    for _ in range(MAX_ACTIVE):
        service.start_timer(60)
    with pytest.raises(ToolError, match="cancel one first"):
        service.start_timer(60)
    service.close()


async def test_timer_tools() -> None:
    said: list[str] = []
    time = FakeTime()
    service = alarms(said, time)
    set_timer, set_reminder, list_timers, cancel_timer = timer_tools(service)
    assert isinstance(set_timer, SetTimer) and isinstance(set_reminder, SetReminder)
    assert isinstance(list_timers, ListTimers) and isinstance(cancel_timer, CancelTimer)

    assert await list_timers.run(NoArgs()) == "No timers or reminders are running."
    tea = TimerArgs(minutes=4.5, label="tea")
    assert set_timer.describe(tea) == "set a 4-minute-30-second tea timer"
    assert set_timer.describe(TimerArgs(minutes=10)) == "set a 10-minute timer"
    assert await set_timer.run(tea) == "Timer 1 started; it ends at 14:04:30."

    at = datetime(2026, 11, 15, 18, 0)  # noqa: DTZ001 - the model sends local times
    reminder = ReminderArgs(at=at, text="call mom")
    assert set_reminder.describe(reminder) == "remind you at 18:00: call mom"
    assert await set_reminder.run(reminder) == "Reminder 2 set for Sunday 18:00."
    listing = await list_timers.run(NoArgs())
    assert listing == "1: tea timer, 4 min 30 s left\n2: reminder 'call mom' at Sun 18:00"
    assert list_timers.describe(NoArgs()) == "check your timers"

    assert cancel_timer.describe(CancelArgs(id=1)) == "cancel timer 1"
    assert await cancel_timer.run(CancelArgs(id=1)) == "Cancelled timer 1."
    service.close()


def test_timer_arguments_are_validated() -> None:
    set_timer = SetTimer(alarms([], FakeTime()))
    for bad in ({"minutes": 0}, {"minutes": 2000}, {"minutes": 5, "label": "x" * 41}, {}):
        with pytest.raises(ValueError, match="validation error"):
            set_timer.parse(bad)
    assert set_timer.parse({"minutes": "5"}).minutes == 5
    spec = set_timer.spec()
    assert spec.name == "set_timer"
    assert set(spec.input_schema["properties"]) == {"minutes", "label"}
