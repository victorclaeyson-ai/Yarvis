"""Timers and reminders, announced out loud when they are due.

They live in memory while Yarvis runs (lasting reminders arrive with memory, Phase 3).
Waiting happens in short steps against the wall clock, so a reminder set for 18:00 still
fires on time after the laptop has been asleep.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Final, Literal

from pydantic import Field

from yarvis.tools.base import Arguments, Tier, Tool, ToolError

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)

MAX_ACTIVE: Final = 20
MAX_AHEAD: Final = timedelta(days=7)
_STEP_S: Final = 30.0


def local_now() -> datetime:
    """The current local time, time-zone aware."""
    return datetime.now().astimezone()


@dataclass(frozen=True, slots=True)
class Alarm:
    """A running timer or a reminder."""

    id: int
    kind: Literal["timer", "reminder"]
    label: str
    due: datetime
    seconds: float | None = None

    def announcement(self) -> str:
        """What Yarvis says when it is due."""
        if self.kind == "reminder":
            return f"Reminder: {self.label}."
        name = f"{self.label} timer" if self.label else f"{duration(self.seconds or 0)} timer"
        return f"Your {name} is done."


def duration(seconds: float) -> str:
    """'10-minute', '45-second', '1-hour-30-minute' (as in "a 10-minute timer").

    Seconds are only mentioned below an hour.
    """
    minutes, secs = divmod(round(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    parts = [f"{hours}-hour"] if hours else []
    parts += [f"{minutes}-minute"] if minutes else []
    parts += [f"{secs}-second"] if secs and not hours else []
    return "-".join(parts) or "0-second"


@dataclass(slots=True)
class Alarms:
    """The running timers and reminders. Due ones are handed to ``announce``."""

    announce: Callable[[str], None]
    now: Callable[[], datetime] = local_now
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep
    on_change: Callable[[list[Alarm]], None] | None = None
    """Told the running alarms whenever one is set, cancelled or goes off (the dashboard)."""
    _alarms: dict[int, Alarm] = field(default_factory=dict[int, Alarm])
    _tasks: dict[int, asyncio.Task[None]] = field(default_factory=dict[int, asyncio.Task[None]])
    _next_id: int = 1

    def start_timer(self, seconds: float, label: str = "") -> Alarm:
        """Start a timer."""
        due = self.now() + timedelta(seconds=seconds)
        return self._add("timer", label, due, seconds)

    def remind(self, at: datetime, text: str) -> Alarm:
        """Set a reminder (``at`` must be in the next seven days)."""
        now = self.now()
        if at <= now:
            msg = f"{at:%H:%M} has already passed."
            raise ToolError(msg)
        if at - now > MAX_AHEAD:
            msg = "Reminders can be at most seven days ahead for now."
            raise ToolError(msg)
        return self._add("reminder", text, at, None)

    def active(self) -> list[Alarm]:
        """All running alarms, soonest first."""
        return sorted(self._alarms.values(), key=lambda alarm: alarm.due)

    def cancel(self, alarm_id: int) -> Alarm:
        """Stop an alarm."""
        alarm = self._alarms.pop(alarm_id, None)
        if alarm is None:
            msg = f"There is no timer or reminder number {alarm_id}."
            raise ToolError(msg)
        self._tasks.pop(alarm_id).cancel()
        self._changed()
        return alarm

    def close(self) -> None:
        """Stop everything (Yarvis is quitting)."""
        for task in self._tasks.values():
            task.cancel()
        self._tasks.clear()
        self._alarms.clear()
        self._changed()

    def _add(
        self, kind: Literal["timer", "reminder"], label: str, due: datetime, seconds: float | None
    ) -> Alarm:
        if len(self._alarms) >= MAX_ACTIVE:
            msg = f"There are already {MAX_ACTIVE} timers and reminders; cancel one first."
            raise ToolError(msg)
        alarm = Alarm(self._next_id, kind, label, due, seconds)
        self._next_id += 1
        self._alarms[alarm.id] = alarm
        self._tasks[alarm.id] = asyncio.create_task(self._wait(alarm))
        logger.info("%s %d set for %s", kind.capitalize(), alarm.id, f"{due:%H:%M:%S}")
        self._changed()
        return alarm

    def _changed(self) -> None:
        if self.on_change is not None:
            self.on_change(self.active())

    async def _wait(self, alarm: Alarm) -> None:
        while (left := (alarm.due - self.now()).total_seconds()) > 0:
            await self.sleep(min(left, _STEP_S))
        self._alarms.pop(alarm.id, None)
        self._tasks.pop(alarm.id, None)
        logger.info("%s %d is due", alarm.kind.capitalize(), alarm.id)
        self._changed()
        self.announce(alarm.announcement())


# ----------------------------------------------------------------------------- tools


class TimerArgs(Arguments):
    """Arguments of ``set_timer``."""

    minutes: float = Field(gt=0, le=24 * 60, description="Length in minutes (0.5 = 30 seconds).")
    label: str = Field(default="", max_length=40, description="Optional name, e.g. 'pasta'.")


class ReminderArgs(Arguments):
    """Arguments of ``set_reminder``."""

    at: datetime = Field(description="Local date and time, ISO 8601, e.g. 2026-11-15T18:00.")
    text: str = Field(min_length=1, max_length=200, description="What to remind the user of.")


class CancelArgs(Arguments):
    """Arguments of ``cancel_timer``."""

    id: int = Field(ge=1, description="The number from list_timers.")


class NoArgs(Arguments):
    """No arguments."""


class SetTimer(Tool[TimerArgs]):
    """Start a countdown."""

    name = "set_timer"
    description = "Start a countdown timer. Yarvis announces it out loud when it is done."
    tier = Tier.REVERSIBLE
    Args = TimerArgs

    def __init__(self, alarms: Alarms) -> None:
        self._alarms = alarms

    async def run(self, args: TimerArgs) -> str:
        """Start it."""
        alarm = self._alarms.start_timer(args.minutes * 60, args.label)
        return f"Timer {alarm.id} started; it ends at {alarm.due:%H:%M:%S}."

    def describe(self, args: TimerArgs) -> str:
        """'set a 10-minute pasta timer'."""
        label = f" {args.label}" if args.label else ""
        return f"set a {duration(args.minutes * 60)}{label} timer"


class SetReminder(Tool[ReminderArgs]):
    """Remind the user at a time."""

    name = "set_reminder"
    description = (
        "Remind the user of something at a local date and time (within 7 days). "
        "Yarvis says it out loud then."
    )
    tier = Tier.REVERSIBLE
    Args = ReminderArgs

    def __init__(self, alarms: Alarms) -> None:
        self._alarms = alarms

    async def run(self, args: ReminderArgs) -> str:
        """Set it."""
        alarm = self._alarms.remind(self._local(args.at), args.text)
        return f"Reminder {alarm.id} set for {alarm.due:%A %H:%M}."

    def describe(self, args: ReminderArgs) -> str:
        """'remind you at 18:00: call mom'."""
        return f"remind you at {self._local(args.at):%H:%M}: {args.text}"

    def _local(self, at: datetime) -> datetime:
        return at if at.tzinfo is not None else at.replace(tzinfo=self._alarms.now().tzinfo)


class ListTimers(Tool[NoArgs]):
    """What is running."""

    name = "list_timers"
    description = "List running timers and reminders with their numbers and time left."
    tier = Tier.READ
    Args = NoArgs

    def __init__(self, alarms: Alarms) -> None:
        self._alarms = alarms

    async def run(self, args: NoArgs) -> str:
        """List them."""
        alarms = self._alarms.active()
        if not alarms:
            return "No timers or reminders are running."
        now = self._alarms.now()
        lines: list[str] = []
        for alarm in alarms:
            left = max(0, round((alarm.due - now).total_seconds()))
            if alarm.kind == "timer":
                what = f"{alarm.label} timer" if alarm.label else "timer"
                minutes, seconds = divmod(left, 60)
                lines.append(f"{alarm.id}: {what}, {minutes} min {seconds} s left")
            else:
                lines.append(f"{alarm.id}: reminder '{alarm.label}' at {alarm.due:%a %H:%M}")
        return "\n".join(lines)

    def describe(self, args: NoArgs) -> str:
        """'check your timers'."""
        return "check your timers"


class CancelTimer(Tool[CancelArgs]):
    """Stop a timer or reminder."""

    name = "cancel_timer"
    description = "Cancel a timer or reminder by its number (use list_timers to find it)."
    tier = Tier.REVERSIBLE
    Args = CancelArgs

    def __init__(self, alarms: Alarms) -> None:
        self._alarms = alarms

    async def run(self, args: CancelArgs) -> str:
        """Cancel it."""
        alarm = self._alarms.cancel(args.id)
        return f"Cancelled {alarm.kind} {alarm.id}."

    def describe(self, args: CancelArgs) -> str:
        """'cancel timer 2'."""
        return f"cancel timer {args.id}"


def timer_tools(alarms: Alarms) -> list[Tool[Any]]:
    """All timer and reminder tools, sharing ``alarms``."""
    return [SetTimer(alarms), SetReminder(alarms), ListTimers(alarms), CancelTimer(alarms)]
