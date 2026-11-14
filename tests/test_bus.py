from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest
from hypothesis import given
from hypothesis import strategies as st

from yarvis.core import (
    AssistantStarted,
    BusClosedError,
    BusFullError,
    Event,
    EventBus,
    OverflowPolicy,
    ShutdownRequested,
    Subscription,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class Tick(Event):
    n: int


@dataclass(frozen=True, slots=True, kw_only=True)
class FastTick(Tick):
    pass


def drain_nowait(subscription: Subscription[Tick]) -> list[int]:
    """Collect every buffered tick without awaiting."""

    async def collect() -> list[int]:
        subscription.close()
        return [event.n async for event in subscription]

    return asyncio.run(collect())


async def test_events_are_routed_by_type() -> None:
    bus = EventBus()
    ticks = bus.subscribe(Tick)
    shutdowns = bus.subscribe(ShutdownRequested)
    await bus.publish(Tick(n=1))
    await bus.publish(ShutdownRequested(reason="bye"))
    assert (await anext(ticks)).n == 1
    assert (await anext(shutdowns)).reason == "bye"
    assert len(ticks) == 0
    assert len(shutdowns) == 0


async def test_subscribing_to_a_base_class_receives_subclasses() -> None:
    bus = EventBus()
    everything = bus.subscribe(Event)
    ticks = bus.subscribe(Tick)
    await bus.publish(FastTick(n=7))
    await bus.publish(AssistantStarted())
    assert len(everything) == 2
    assert len(ticks) == 1


async def test_block_policy_applies_backpressure() -> None:
    bus = EventBus()
    subscription = bus.subscribe(Tick, maxsize=1)
    await bus.publish(Tick(n=1))

    blocked = asyncio.create_task(bus.publish(Tick(n=2)))
    await asyncio.sleep(0)
    assert not blocked.done()

    assert (await anext(subscription)).n == 1
    await asyncio.wait_for(blocked, timeout=1)
    assert (await anext(subscription)).n == 2


async def test_drop_oldest_keeps_newest_events() -> None:
    bus = EventBus()
    subscription = bus.subscribe(Tick, maxsize=2, overflow=OverflowPolicy.DROP_OLDEST)
    for n in range(5):
        await bus.publish(Tick(n=n))
    assert subscription.dropped == 3
    subscription.close()
    assert [event.n async for event in subscription] == [3, 4]


@given(count=st.integers(0, 50), maxsize=st.integers(1, 10))
def test_drop_oldest_invariant(count: int, maxsize: int) -> None:
    bus = EventBus()
    subscription = bus.subscribe(Tick, maxsize=maxsize, overflow=OverflowPolicy.DROP_OLDEST)
    for n in range(count):
        bus.publish_nowait(Tick(n=n))
    kept = drain_nowait(subscription)
    assert kept == list(range(count))[-maxsize:] if count else kept == []
    assert subscription.dropped == max(0, count - maxsize)


async def test_publish_nowait_raises_when_blocking_subscriber_is_full() -> None:
    bus = EventBus()
    bus.subscribe(Tick, maxsize=1)
    bus.publish_nowait(Tick(n=1))
    with pytest.raises(BusFullError, match="Tick"):
        bus.publish_nowait(Tick(n=2))


async def test_closing_a_subscription_drains_then_stops() -> None:
    bus = EventBus()
    subscription = bus.subscribe(Tick)
    await bus.publish(Tick(n=1))
    subscription.close()
    subscription.close()  # idempotent
    await bus.publish(Tick(n=2))  # no longer delivered
    assert [event.n async for event in subscription] == [1]
    assert subscription.closed
    assert bus.subscriber_count == 0


async def test_close_wakes_a_waiting_consumer() -> None:
    bus = EventBus()
    subscription = bus.subscribe(Tick)
    consumer = asyncio.create_task(anext(subscription, None))
    await asyncio.sleep(0)
    subscription.close()
    assert await asyncio.wait_for(consumer, timeout=1) is None


async def test_context_manager_detaches() -> None:
    bus = EventBus()
    with bus.subscribe(Tick) as subscription:
        assert bus.subscriber_count == 1
    assert subscription.closed
    assert bus.subscriber_count == 0


async def test_closed_bus_rejects_use_and_ends_iteration() -> None:
    bus = EventBus()
    subscription = bus.subscribe(Tick)
    bus.close()
    assert bus.closed
    assert [event async for event in subscription] == []
    with pytest.raises(BusClosedError):
        await bus.publish(Tick(n=1))
    with pytest.raises(BusClosedError):
        bus.publish_nowait(Tick(n=1))
    with pytest.raises(BusClosedError):
        bus.subscribe(Tick)


def test_maxsize_must_be_positive() -> None:
    with pytest.raises(ValueError, match="maxsize"):
        EventBus().subscribe(Tick, maxsize=0)


def test_detaching_an_unknown_subscription_is_harmless() -> None:
    bus = EventBus()
    subscription = bus.subscribe(Tick)
    bus.detach(subscription)
    bus.detach(subscription)
    assert bus.subscriber_count == 0
