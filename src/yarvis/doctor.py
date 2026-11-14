"""``yarvis doctor``: check that the machine is ready to run Yarvis.

Each check is a pure function of a ``DoctorContext`` so it can be tested with fakes.
Checks never print secrets; at most they say where a secret was found.
"""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from yarvis.audio.devices import AudioBackendError, AudioDevice
from yarvis.config import Config, ConfigError, load_config
from yarvis.paths import is_private
from yarvis.security.secrets import SecretName, SecretSource

if TYPE_CHECKING:
    from collections.abc import Sequence

    from yarvis.audio.devices import AudioProbe
    from yarvis.paths import AppPaths
    from yarvis.security.secrets import SecretStore

MIN_PYTHON = (3, 12)


class Status(StrEnum):
    """Outcome of a single check."""

    OK = "ok"
    WARN = "warn"
    FAIL = "fail"


@dataclass(frozen=True, slots=True)
class CheckResult:
    """The result of one diagnostic check."""

    name: str
    status: Status
    detail: str
    hint: str | None = None


@dataclass(frozen=True, slots=True)
class DoctorContext:
    """Everything the checks need, injectable for tests."""

    paths: AppPaths
    secrets: SecretStore
    audio: AudioProbe
    python_version: tuple[int, int, int]
    system: str
    machine: str

    @classmethod
    def current(cls, *, paths: AppPaths, secrets: SecretStore, audio: AudioProbe) -> DoctorContext:
        """Build a context describing the running machine."""
        major, minor, micro = sys.version_info[:3]
        return cls(
            paths=paths,
            secrets=secrets,
            audio=audio,
            python_version=(major, minor, micro),
            system=platform.system(),
            machine=platform.machine(),
        )


def run_doctor(ctx: DoctorContext) -> list[CheckResult]:
    """Run every check in a stable order."""
    config_result, config = check_config(ctx)
    return [
        check_python(ctx),
        check_platform(ctx),
        config_result,
        check_storage(ctx),
        check_keychain(ctx),
        check_api_key(ctx, config),
        check_audio_input(ctx, config),
        check_audio_output(ctx, config),
    ]


def has_failures(results: Sequence[CheckResult]) -> bool:
    """Return True if any check failed."""
    return any(result.status is Status.FAIL for result in results)


# --------------------------------------------------------------------------- checks


def check_python(ctx: DoctorContext) -> CheckResult:
    """Python must be new enough."""
    version = ".".join(str(part) for part in ctx.python_version)
    if ctx.python_version[:2] >= MIN_PYTHON:
        return CheckResult("Python", Status.OK, version)
    required = ".".join(str(part) for part in MIN_PYTHON)
    return CheckResult(
        "Python",
        Status.FAIL,
        f"{version} is too old",
        hint=f"Install Python {required} or newer (uv python install {required}).",
    )


def check_platform(ctx: DoctorContext) -> CheckResult:
    """MacOS on Apple Silicon and Linux are supported."""
    label = f"{ctx.system} {ctx.machine}"
    if ctx.system == "Darwin":
        if ctx.machine == "arm64":
            return CheckResult("Platform", Status.OK, label)
        return CheckResult(
            "Platform",
            Status.WARN,
            label,
            hint="Intel Macs are not a supported target; local speech models will be slow.",
        )
    if ctx.system == "Linux":
        return CheckResult("Platform", Status.OK, label)
    return CheckResult(
        "Platform",
        Status.FAIL,
        f"{label} is not supported yet",
        hint="Yarvis supports macOS and Linux. Windows support is planned.",
    )


def check_config(ctx: DoctorContext) -> tuple[CheckResult, Config]:
    """The config file must be valid, private and free of secrets."""
    path = ctx.paths.config_file
    if not path.exists():
        result = CheckResult(
            "Config",
            Status.OK,
            "no config file, using defaults",
            hint="Create one with: yarvis config init",
        )
        return result, Config()
    try:
        config = load_config(path)
    except ConfigError as exc:
        return CheckResult("Config", Status.FAIL, str(exc)), Config()
    if not is_private(path):
        result = CheckResult(
            "Config",
            Status.WARN,
            f"{path} is readable or writable by other users",
            hint=f"chmod 600 '{path}'",
        )
        return result, config
    if relaxations := config.security_relaxations():
        result = CheckResult(
            "Config",
            Status.WARN,
            "security relaxed: " + "; ".join(relaxations),
            hint="Only relax security settings if you understand the risk (design doc §5).",
        )
        return result, config
    return CheckResult("Config", Status.OK, str(path)), config


def check_storage(ctx: DoctorContext) -> CheckResult:
    """Existing data directories must be private to the current user."""
    exposed = [path for path in ctx.paths.all_dirs() if path.exists() and not is_private(path)]
    if exposed:
        listed = ", ".join(str(path) for path in exposed)
        return CheckResult(
            "Storage",
            Status.WARN,
            f"accessible by other users: {listed}",
            hint="chmod 700 each listed directory.",
        )
    return CheckResult("Storage", Status.OK, "private")


def check_keychain(ctx: DoctorContext) -> CheckResult:
    """A secure OS keychain should be available for API keys."""
    if ctx.secrets.keychain_available:
        return CheckResult("Keychain", Status.OK, ctx.secrets.keychain_description)
    hint = "On Linux, install and unlock a Secret Service provider (e.g. GNOME Keyring)."
    return CheckResult(
        "Keychain",
        Status.WARN,
        "no secure keychain; API keys must come from environment variables",
        hint=hint,
    )


def check_api_key(ctx: DoctorContext, config: Config) -> CheckResult:
    """The configured LLM provider must have credentials."""
    if config.llm.provider == "ollama":
        return CheckResult("API key", Status.OK, "not needed (local Ollama)")
    name = SecretName.ANTHROPIC
    resolved = ctx.secrets.get(name)
    if resolved is None:
        hint = (
            f"Run: yarvis key set {name.value}"
            if ctx.secrets.keychain_available
            else f"Set the {name.env_var} environment variable."
        )
        return CheckResult("API key", Status.FAIL, f"no {name.value} key found", hint=hint)
    if resolved.source is SecretSource.KEYCHAIN:
        return CheckResult("API key", Status.OK, f"{name.value} key in keychain")
    return CheckResult("API key", Status.OK, f"{name.value} key from ${name.env_var}")


def check_audio_input(ctx: DoctorContext, config: Config) -> CheckResult:
    """A microphone must be available."""
    return _check_audio(ctx, config.voice.input_device, kind="input")


def check_audio_output(ctx: DoctorContext, config: Config) -> CheckResult:
    """A speaker must be available."""
    return _check_audio(ctx, config.voice.output_device, kind="output")


def _check_audio(ctx: DoctorContext, wanted: str | int | None, *, kind: str) -> CheckResult:
    label = "Microphone" if kind == "input" else "Speaker"
    try:
        if wanted is not None:
            device = find_device(ctx.audio.devices(), wanted, kind=kind)
            if device is None:
                return CheckResult(
                    label,
                    Status.FAIL,
                    f"configured {kind} device {wanted!r} not found",
                    hint="List devices with: python -m sounddevice",
                )
        else:
            device = ctx.audio.default_input() if kind == "input" else ctx.audio.default_output()
    except AudioBackendError as exc:
        return CheckResult(label, Status.FAIL, str(exc), hint=_portaudio_hint(ctx.system))
    if device is None:
        return CheckResult(label, Status.FAIL, f"no default {kind} device")
    return CheckResult(label, Status.OK, device.name)


def find_device(
    devices: Sequence[AudioDevice], wanted: str | int, *, kind: str
) -> AudioDevice | None:
    """Find a device by index or (case-insensitive) name that supports ``kind``."""
    for device in devices:
        capable = device.is_input if kind == "input" else device.is_output
        if not capable:
            continue
        if isinstance(wanted, int):
            if device.index == wanted:
                return device
        elif device.name.casefold() == wanted.casefold():
            return device
    return None


def _portaudio_hint(system: str) -> str:
    if system == "Linux":
        return "Install PortAudio: sudo apt install libportaudio2 (Debian/Ubuntu)."
    return "Reinstall Yarvis; the sounddevice wheel bundles PortAudio on macOS."
