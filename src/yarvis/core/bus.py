"""An in-process, typed publish/subscribe bus with explicit backpressure.

Every subscriber owns a bounded buffer and chooses what happens when it is full:

* ``OverflowPolicy.BLOCK`` — ``publish`` waits until there is room. Use for control events
  that must never be lost.
* ``OverflowPolicy.DROP_OLDEST`` — the oldest buffered event is discarded. Use for
  high-rate streams such as audio frames, where fresh data matters more than old data.

The bus is **not thread-safe**. Code running on other threads (for example PortAudio
callbacks) must hand events over with ``loop.call_soon_threadsafe(bus.publish_nowait, ev)``.

Example:
    >>> async def main(bus: EventBus) -> None:
    ...     with bus.subscribe(ShutdownRequested) as sub:
    ...         async for event in sub:
    ...             print(event.reason)
"""

from __future__ import annotations

import asyncio
from collections import deque
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Self

from yarvis.core.events import Event

if TYPE_CHECKING:
    from types import TracebackType

DEFAULT_MAXSIZE = 256


class OverflowPolicy(StrEnum):
    """What a subscription does when its buffer is full."""

    BLOCK = "block"
    DROP_OLDEST = "drop_oldest"


class BusClosedError(RuntimeError):
    """Raised when publishing to or subscribing on a closed bus."""


class BusFullError(RuntimeError):
    """Raised by ``publish_nowait`` when a blocking subscriber has no room."""


class Subscription[E: Event]:
    """A bounded stream of events of type ``E``. Iterate it with ``async for``.

    Closing a subscription stops delivery; events already buffered are still yielded,
    then iteration ends.
    """

    def __init__(
        self,
        bus: EventBus,
        event_type: type[E],
        *,
        maxsize: int,
        overflow: OverflowPolicy,
    ) -> None:
        if maxsize < 1:
            msg = "maxsize must be at least 1"
            raise ValueError(msg)
        self._bus = bus
        self.event_type = event_type
        self.maxsize = maxsize
        self.overflow = overflow
        self.dropped = 0
        self._items: deque[E] = deque()
        self._not_empty = asyncio.Event()
        self._not_full = asyncio.Event()
        self._not_full.set()
        self._closed = False

    @property
    def closed(self) -> bool:
        """Whether the subscription no longer receives new events."""
        return self._closed

    def __len__(self) -> int:
        return len(self._items)

    def accepts(self, event: Event) -> bool:
        """Return True if ``event`` matches this subscription's type."""
        return isinstance(event, self.event_type)

    def try_put(self, event: E) -> bool:
        """Buffer ``event`` without waiting.

        Returns:
            False only when the policy is BLOCK and the buffer is full.
        """
        if self._closed:
            return True
        if len(self._items) >= self.maxsize:
            if self.overflow is OverflowPolicy.BLOCK:
                self._not_full.clear()
                return False
            self._items.popleft()
            self.dropped += 1
        self._items.append(event)
        self._not_empty.set()
        return True

    async def put(self, event: E) -> None:
        """Buffer ``event``, waiting for room if the policy is BLOCK."""
        while not self.try_put(event):
            await self._not_full.wait()

    def close(self) -> None:
        """Stop receiving events and detach from the bus. Idempotent."""
        if self._closed:
            return
        self._closed = True
        self._not_empty.set()
        self._not_full.set()
        self._bus.detach(self)

    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> E:
        while not self._items:
            if self._closed:
                raise StopAsyncIteration
            self._not_empty.clear()
            await self._not_empty.wait()
        item = self._items.popleft()
        self._not_full.set()
        return item

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


class EventBus:
    """Routes events to every subscription whose type matches."""

    def __init__(self) -> None:
        # Heterogeneous: each subscription has its own event type, checked by ``accepts``.
        self._subscriptions: list[Subscription[Any]] = []
        self._closed = False

    @property
    def closed(self) -> bool:
        """Whether the bus has been shut down."""
        return self._closed

    @property
    def subscriber_count(self) -> int:
        """Number of active subscriptions."""
        return len(self._subscriptions)

    def subscribe[E: Event](
        self,
        event_type: type[E],
        *,
        maxsize: int = DEFAULT_MAXSIZE,
        overflow: OverflowPolicy = OverflowPolicy.BLOCK,
    ) -> Subscription[E]:
        """Create a subscription for ``event_type`` and all of its subclasses."""
        if self._closed:
            raise BusClosedError("cannot subscribe to a closed bus")
        subscription = Subscription(self, event_type, maxsize=maxsize, overflow=overflow)
        self._subscriptions.append(subscription)
        return subscription

    async def publish(self, event: Event) -> None:
        """Deliver ``event`` to every matching subscriber, honouring backpressure.

        Subscribers are served in subscription order; a full BLOCK subscriber delays
        delivery to the ones after it.
        """
        self._ensure_open()
        for subscription in tuple(self._subscriptions):
            if subscription.accepts(event):
                await subscription.put(event)

    def publish_nowait(self, event: Event) -> None:
        """Deliver ``event`` without waiting.

        Raises:
            BusFullError: If a BLOCK subscriber is full. Subscribers before it have
                already received the event.
        """
        self._ensure_open()
        for subscription in tuple(self._subscriptions):
            if subscription.accepts(event) and not subscription.try_put(event):
                msg = f"subscriber for {subscription.event_type.__name__} is full"
                raise BusFullError(msg)

    def detach(self, subscription: Subscription[Any]) -> None:
        """Remove ``subscription``. Called by ``Subscription.close``."""
        if subscription in self._subscriptions:
            self._subscriptions.remove(subscription)

    def close(self) -> None:
        """Close the bus and every subscription. Idempotent."""
        self._closed = True
        for subscription in tuple(self._subscriptions):
            subscription.close()

    def _ensure_open(self) -> None:
        if self._closed:
            raise BusClosedError("cannot publish to a closed bus")
