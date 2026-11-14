from __future__ import annotations

from types import ModuleType
from typing import TYPE_CHECKING, Any

import pytest

from yarvis.audio.devices import AudioBackendError, SoundDeviceProbe

if TYPE_CHECKING:
    from collections.abc import Callable


class PortAudioError(Exception):
    pass


RAW_DEVICES: list[dict[str, Any]] = [
    {
        "name": "Built-in Mic",
        "index": 0,
        "max_input_channels": 1,
        "max_output_channels": 0,
        "default_samplerate": 48000.0,
    },
    {
        "name": "Built-in Speakers",
        "index": 1,
        "max_input_channels": 0,
        "max_output_channels": 2,
        "default_samplerate": 44100,
    },
]


class FakeSoundDevice(ModuleType):
    PortAudioError = PortAudioError

    def __init__(self, query_devices: Callable[..., object]) -> None:
        super().__init__("sounddevice")
        self.query_devices = query_devices


def fake_sounddevice(*, has_input: bool = True, broken: bool = False) -> ModuleType:
    def query_devices(kind: str | None = None) -> object:
        if broken:
            raise PortAudioError("Error querying device")
        if kind is None:
            return RAW_DEVICES
        if kind == "input":
            if not has_input:
                raise PortAudioError("No input device")
            return RAW_DEVICES[0]
        return RAW_DEVICES[1]

    return FakeSoundDevice(query_devices)


def test_devices_are_converted() -> None:
    probe = SoundDeviceProbe(loader=fake_sounddevice)
    mic, speaker = probe.devices()
    assert mic.name == "Built-in Mic"
    assert mic.is_input
    assert not mic.is_output
    assert speaker.is_output
    assert speaker.default_samplerate == 44100.0


def test_defaults() -> None:
    probe = SoundDeviceProbe(loader=fake_sounddevice)
    default_in = probe.default_input()
    default_out = probe.default_output()
    assert default_in is not None
    assert default_in.index == 0
    assert default_out is not None
    assert default_out.name == "Built-in Speakers"


def test_missing_default_input_returns_none() -> None:
    probe = SoundDeviceProbe(loader=lambda: fake_sounddevice(has_input=False))
    assert probe.default_input() is None


def test_query_errors_become_backend_errors() -> None:
    probe = SoundDeviceProbe(loader=lambda: fake_sounddevice(broken=True))
    with pytest.raises(AudioBackendError, match="querying"):
        probe.devices()


def test_missing_portaudio_becomes_backend_error() -> None:
    def loader() -> ModuleType:
        raise OSError("PortAudio library not found")

    probe = SoundDeviceProbe(loader=loader)
    with pytest.raises(AudioBackendError, match="PortAudio"):
        probe.default_output()


def test_malformed_device_info_is_tolerated() -> None:
    def malformed(**_: object) -> object:
        return {"name": 5, "index": "x"}

    module = FakeSoundDevice(malformed)
    device = SoundDeviceProbe(loader=lambda: module).default_input()
    assert device is not None
    assert device.index == -1
    assert device.max_input_channels == 0
    assert device.default_samplerate == 0.0
