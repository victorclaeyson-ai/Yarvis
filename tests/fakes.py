"""In-memory fakes for external systems (keychain, audio hardware)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import keyring.errors

from yarvis.audio.devices import AudioBackendError, AudioDevice

if TYPE_CHECKING:
    from collections.abc import Sequence

VALID_KEY = "sk-ant-test-0123456789abcdefghij"  # gitleaks:allow


@dataclass
class FakeKeyring:
    """Dict-backed stand-in for the OS keychain."""

    store: dict[tuple[str, str], str] = field(default_factory=dict[tuple[str, str], str])
    available: bool = True
    fail_reads: bool = False

    def get_password(self, service: str, username: str) -> str | None:
        if self.fail_reads:
            raise keyring.errors.KeyringError("locked")
        return self.store.get((service, username))

    def set_password(self, service: str, username: str, password: str) -> None:
        self.store[(service, username)] = password

    def delete_password(self, service: str, username: str) -> None:
        try:
            del self.store[(service, username)]
        except KeyError:
            raise keyring.errors.PasswordDeleteError("not found") from None

    def is_available(self) -> bool:
        return self.available

    def describe(self) -> str:
        return "fake.Keyring"


MIC = AudioDevice(
    index=0,
    name="Test Microphone",
    max_input_channels=1,
    max_output_channels=0,
    default_samplerate=48_000.0,
)
SPEAKER = AudioDevice(
    index=1,
    name="Test Speakers",
    max_input_channels=0,
    max_output_channels=2,
    default_samplerate=48_000.0,
)


@dataclass
class FakeAudioProbe:
    """Configurable set of audio devices."""

    all_devices: Sequence[AudioDevice] = (MIC, SPEAKER)
    input_device: AudioDevice | None = MIC
    output_device: AudioDevice | None = SPEAKER
    broken: bool = False

    def devices(self) -> Sequence[AudioDevice]:
        self._check()
        return self.all_devices

    def default_input(self) -> AudioDevice | None:
        self._check()
        return self.input_device

    def default_output(self) -> AudioDevice | None:
        self._check()
        return self.output_device

    def _check(self) -> None:
        if self.broken:
            raise AudioBackendError("the PortAudio library is not installed")
