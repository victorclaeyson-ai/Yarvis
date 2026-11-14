"""Core runtime: typed events and the event bus.

The turn state machine (design doc §3.2) is added in Phase 1.
"""

from yarvis.core.bus import (
    BusClosedError,
    BusFullError,
    EventBus,
    OverflowPolicy,
    Subscription,
)
from yarvis.core.events import AssistantStarted, Event, ShutdownRequested

__all__ = [
    "AssistantStarted",
    "BusClosedError",
    "BusFullError",
    "Event",
    "EventBus",
    "OverflowPolicy",
    "ShutdownRequested",
    "Subscription",
]
