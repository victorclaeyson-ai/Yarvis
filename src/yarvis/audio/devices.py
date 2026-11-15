"""Discover audio input and output devices.

``sounddevice`` is imported lazily: importing it fails with ``OSError`` when the PortAudio
system library is missing, and Yarvis should report that clearly instead of crashing.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Protocol, cast

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from types import ModuleType


@dataclass(frozen=True, slots=True)
class AudioDevice:
    """A sound device as reported by PortAudio."""

    index: int
    name: str
    max_input_channels: int
    max_output_channels: int
    default_samplerate: float

    @property
    def is_input(self) -> bool:
        """Whether the device can record."""
        return self.max_input_channels > 0

    @property
    def is_output(self) -> bool:
        """Whether the device can play audio."""
        return self.max_output_channels > 0


class AudioBackendError(RuntimeError):
    """The audio subsystem is unavailable (e.g. PortAudio is not installed)."""


class AudioProbe(Protocol):
    """Read-only view of the system's audio devices."""

    def devices(self) -> Sequence[AudioDevice]:
        """All devices."""
        ...

    def default_input(self) -> AudioDevice | None:
        """The default recording device, or None if there is none."""
        ...

    def default_output(self) -> AudioDevice | None:
        """The default playback device, or None if there is none."""
        ...


# Words that appear in the names of private listening devices on macOS and Linux
# ("AirPods Pro", "External Headphones", "WH-1000XM5", "Jabra Evolve2 65 Headset").
_HEADPHONE_HINTS: Final = (
    "headphone",
    "headset",
    "airpods",
    "earpods",
    "earbud",
    "earphone",
    "buds",
    "beats",
    "wh-1000",
    "wf-1000",
    "jabra",
    "in-ear",
)


def looks_like_headphones(name: str | None) -> bool:
    """Guess whether an output device keeps the sound away from the microphone.

    Unknown devices count as speakers: the cautious answer, because voice barge-in on
    speakers would make Yarvis interrupt itself.
    """
    if not name:
        return False
    lowered = name.lower()
    return any(hint in lowered for hint in _HEADPHONE_HINTS)


def _import_sounddevice() -> ModuleType:
    return importlib.import_module("sounddevice")


def load_sounddevice() -> ModuleType:
    """Import ``sounddevice``, turning a missing PortAudio library into ``AudioBackendError``."""
    try:
        return _import_sounddevice()
    except OSError as exc:
        msg = "the PortAudio library is not installed"
        raise AudioBackendError(msg) from exc


class SoundDeviceProbe:
    """``AudioProbe`` implemented with the ``sounddevice`` package."""

    def __init__(self, loader: Callable[[], ModuleType] = _import_sounddevice) -> None:
        self._loader = loader

    def devices(self) -> Sequence[AudioDevice]:
        """All devices.

        Raises:
            AudioBackendError: If PortAudio is unavailable.
        """
        sd = self._load()
        try:
            raw = cast("Sequence[Mapping[str, object]]", sd.query_devices())
        except sd.PortAudioError as exc:
            raise AudioBackendError(str(exc)) from exc
        return tuple(_to_device(index, info) for index, info in enumerate(raw))

    def default_input(self) -> AudioDevice | None:
        """The default recording device, or None."""
        return self._default("input")

    def default_output(self) -> AudioDevice | None:
        """The default playback device, or None."""
        return self._default("output")

    def _default(self, kind: str) -> AudioDevice | None:
        sd = self._load()
        try:
            info = cast("Mapping[str, object]", sd.query_devices(kind=kind))
        except sd.PortAudioError:
            return None
        index = info.get("index", -1)
        return _to_device(index if isinstance(index, int) else -1, info)

    def _load(self) -> ModuleType:
        try:
            return self._loader()
        except OSError as exc:
            msg = "the PortAudio library is not installed"
            raise AudioBackendError(msg) from exc


def _to_device(index: int, info: Mapping[str, object]) -> AudioDevice:
    return AudioDevice(
        index=index,
        name=str(info.get("name", "unknown")),
        max_input_channels=_as_int(info.get("max_input_channels")),
        max_output_channels=_as_int(info.get("max_output_channels")),
        default_samplerate=_as_float(info.get("default_samplerate")),
    )


def _as_int(value: object) -> int:
    return value if isinstance(value, int) else 0


def _as_float(value: object) -> float:
    return float(value) if isinstance(value, int | float) else 0.0
