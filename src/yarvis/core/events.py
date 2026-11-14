"""Typed events exchanged between Yarvis components.

Events are small immutable dataclasses. Components subscribe by type, and subscribing to a
base class also delivers its subclasses, so ``bus.subscribe(Event)`` sees everything.
Pipeline events (audio frames, transcripts, turn state) are added in Phase 1.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True, kw_only=True)
class Event:
    """Base class for every event. ``timestamp`` uses the monotonic clock (seconds)."""

    timestamp: float = field(default_factory=time.monotonic)


@dataclass(frozen=True, slots=True, kw_only=True)
class AssistantStarted(Event):
    """All components are running."""


@dataclass(frozen=True, slots=True, kw_only=True)
class ShutdownRequested(Event):
    """A component or the user asked Yarvis to stop."""

    reason: str
