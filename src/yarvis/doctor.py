"""``yarvis doctor``: check that the machine is ready to run Yarvis.

Each check is a pure function of a ``DoctorContext`` so it can be tested with fakes.
Checks never print secrets; at most they say where a secret was found.
"""

from __future__ import annotations

import importlib.util
import platform
import sys
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from yarvis.audio.devices import AudioBackendError, AudioDevice
from yarvis.config import Config, ConfigError, load_config
from yarvis.memory import MemoryBook, MemoryBookError
from yarvis.models import (
    KOKORO_FILES,
    WAKE_WORD_FILES,
    WHISPER_APPROX_MB,
    kokoro_dir,
    missing,
    parakeet_dir,
    parakeet_files,
    wake_word_dir,
    whisper_present,
)
from yarvis.paths import is_private
from yarvis.profile import ProfileError, load_profile, profile_path
from yarvis.security.secrets import SecretName, SecretSource
from yarvis.speech.choice import (
    choose_engine,
    parakeet_build,
    parakeet_package,
    parakeet_supported,
)
from yarvis.tools.mcp import Approvals
from yarvis.tools.mcp_launch import secret_key
from yarvis.tools.skills import SkillLibrary

if TYPE_CHECKING:
    from collections.abc import Sequence

    from yarvis.audio.devices import AudioProbe
    from yarvis.paths import AppPaths
    from yarvis.security.secrets import SecretStore

MIN_PYTHON = (3, 12)
MIN_MACOS = (14, 0)


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
    mac_version: str = ""
    parakeet_installed: bool = False

    @classmethod
    def current(cls, *, paths: AppPaths, secrets: SecretStore, audio: AudioProbe) -> DoctorContext:
        """Build a context describing the running machine."""
        major, minor, micro = sys.version_info[:3]
        build = parakeet_build(platform.system(), platform.machine())
        return cls(
            paths=paths,
            secrets=secrets,
            audio=audio,
            python_version=(major, minor, micro),
            system=platform.system(),
            machine=platform.machine(),
            mac_version=platform.mac_ver()[0],
            parakeet_installed=importlib.util.find_spec(parakeet_package(build)) is not None,
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
        check_speech_recognition(ctx, config),
        check_models(ctx, config),
        check_mcp(ctx, config),
        check_skills(ctx, config),
        check_profile(ctx),
        check_memory(ctx),
        check_voice(ctx, config),
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
    """MacOS on Apple Silicon and Linux are supported; Windows is getting there."""
    label = f"{ctx.system} {ctx.machine}"
    if ctx.system == "Darwin":
        if ctx.mac_version:
            label = f"macOS {ctx.mac_version} {ctx.machine}"
            if _version_tuple(ctx.mac_version) < MIN_MACOS:
                return CheckResult(
                    "Platform",
                    Status.FAIL,
                    label,
                    hint="Yarvis needs macOS 14 (Sonoma) or newer for its audio libraries.",
                )
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
    if ctx.system == "Windows":
        return CheckResult(
            "Platform",
            Status.WARN,
            f"{label} (early support)",
            hint="Talking, timers, the weather, web search, apps, volume and Spotify work; "
            "mail, the calendar, typing and browser tabs are macOS-only for now.",
        )
    return CheckResult(
        "Platform",
        Status.FAIL,
        f"{label} is not supported yet",
        hint="Yarvis supports macOS, Linux and (early) Windows.",
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
    if not is_private(path, windows=_windows(ctx)):
        result = CheckResult(
            "Config",
            Status.WARN,
            f"{path} is readable or writable by other users",
            hint=_keep_private(ctx, f"chmod 600 '{path}'"),
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
    folders = dict.fromkeys(ctx.paths.all_dirs())  # on Windows, config and data share one
    windows = _windows(ctx)
    exposed = [path for path in folders if path.exists() and not is_private(path, windows=windows)]
    if exposed:
        listed = ", ".join(str(path) for path in exposed)
        return CheckResult(
            "Storage",
            Status.WARN,
            f"accessible by other users: {listed}",
            hint=_keep_private(ctx, "chmod 700 each listed directory."),
        )
    return CheckResult("Storage", Status.OK, "private")


def _windows(ctx: DoctorContext) -> bool:
    return ctx.system == "Windows"


def _keep_private(ctx: DoctorContext, unix_hint: str) -> str:
    if _windows(ctx):
        return "Keep Yarvis's folders inside your user profile (unset YARVIS_HOME)."
    return unix_hint


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


def check_speech_recognition(ctx: DoctorContext, config: Config) -> CheckResult:
    """Which engine hears you: AssemblyAI (with a key) or the one on this computer."""
    label = "Speech recognition"
    parakeet_ok = parakeet_supported(ctx.system, ctx.machine, installed=ctx.parakeet_installed)
    engine = choose_engine(config.voice, parakeet_ok=parakeet_ok)
    local = "Parakeet" if engine == "parakeet" else f"Whisper {config.voice.stt_model}"
    if not config.voice.cloud_stt:
        return CheckResult(label, Status.OK, f"{local} on this computer (cloud_stt is off)")
    resolved = ctx.secrets.get(SecretName.ASSEMBLYAI)
    if resolved is None:
        return CheckResult(
            label,
            Status.OK,
            f"{local} on this computer",
            hint="For the most accurate transcripts: yarvis key set assemblyai",
        )
    where = "EU servers" if config.voice.cloud_region == "eu" else "nearest servers"
    return CheckResult(label, Status.OK, f"AssemblyAI ({where}), {local} as backup")


def check_models(ctx: DoctorContext, config: Config) -> CheckResult:
    """Speech models should be downloaded before the first conversation."""
    cache = ctx.paths.cache_dir
    todo: list[str] = []
    megabytes = 0
    if pending := missing(KOKORO_FILES, kokoro_dir(cache)):
        todo.append("Kokoro voice")
        megabytes += sum(model.size for model in pending) // 1_000_000
    parakeet_ok = parakeet_supported(ctx.system, ctx.machine, installed=ctx.parakeet_installed)
    if choose_engine(config.voice, parakeet_ok=parakeet_ok) == "parakeet":
        build = parakeet_build(ctx.system, ctx.machine)
        if pending := missing(parakeet_files(build), parakeet_dir(cache, build)):
            todo.append("Parakeet speech recognition")
            megabytes += sum(model.size for model in pending) // 1_000_000
    elif not whisper_present(cache, config.voice.stt_model):
        todo.append(f"Whisper {config.voice.stt_model}")
        megabytes += WHISPER_APPROX_MB[config.voice.stt_model]
    if config.voice.wake_word and (pending := missing(WAKE_WORD_FILES, wake_word_dir(cache))):
        todo.append("wake word")
        megabytes += max(1, sum(model.size for model in pending) // 1_000_000)
    if not todo:
        return CheckResult("Speech models", Status.OK, "downloaded")
    return CheckResult(
        "Speech models",
        Status.WARN,
        f"not downloaded yet: {', '.join(todo)} (~{megabytes} MB)",
        hint="Run: yarvis models download (or let yarvis run do it)",
    )


def _version_tuple(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split(".") if part.isdigit())


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


def check_mcp(ctx: DoctorContext, config: Config) -> CheckResult:
    """Every configured MCP server should have its secrets and approved tools."""
    label = "MCP servers"
    servers = config.mcp.servers
    if not servers:
        return CheckResult(label, Status.OK, "none configured")
    approved = Approvals.of(ctx.paths).load()
    for server in servers:
        needed = (*server.secrets, *((server.token,) if server.token else ()))
        for name in needed:
            if ctx.secrets.get_extra(secret_key(server.name, name)) is None:
                return CheckResult(
                    label,
                    Status.WARN,
                    f"{server.name}: secret {name} is not set",
                    hint=f"Run: yarvis mcp secret {server.name} {name}",
                )
        pending = [name for name in server.tools if name not in approved.get(server.name, {})]
        if pending:
            return CheckResult(
                label,
                Status.WARN,
                f"{server.name}: not approved yet: {', '.join(pending)}",
                hint=f"Run: yarvis mcp approve {server.name}",
            )
    names = ", ".join(server.name for server in servers)
    return CheckResult(label, Status.OK, f"{names} (definitions are checked at every start)")


def check_skills(ctx: DoctorContext, config: Config) -> CheckResult:
    """The skills the user added should still be the ones they approved."""
    label = "Skills"
    if not config.tools.skills:
        return CheckResult(label, Status.OK, "off (tools.skills = false)")
    skills, problems = SkillLibrary.of(ctx.paths).skills()
    if problems:
        return CheckResult(label, Status.WARN, "; ".join(problems))
    if not skills:
        return CheckResult(label, Status.OK, "none", hint="Add one: yarvis skills add <folder>")
    return CheckResult(label, Status.OK, ", ".join(skill.name for skill in skills))


def check_profile(ctx: DoctorContext) -> CheckResult:
    """The user's profile, if written, should be usable (short, no secrets)."""
    label = "Profile"
    try:
        text = load_profile(profile_path(ctx.paths))
    except ProfileError as exc:
        return CheckResult(label, Status.WARN, f"not used: {exc}", hint="Run: yarvis profile edit")
    if not text:
        return CheckResult(
            label,
            Status.OK,
            "not written yet",
            hint="Tell Yarvis about yourself: yarvis profile edit",
        )
    return CheckResult(label, Status.OK, f"{len(text):,} characters, sent with every request")


def check_memory(ctx: DoctorContext) -> CheckResult:
    """The facts Yarvis remembers should be readable (ADR 0017)."""
    label = "Memory"
    try:
        facts = MemoryBook.of(ctx.paths).load()
    except MemoryBookError as exc:
        return CheckResult(label, Status.WARN, str(exc), hint="See them: yarvis memory list")
    if not facts:
        return CheckResult(label, Status.OK, "nothing remembered yet")
    count = len(facts)
    noun = "fact" if count == 1 else "facts"
    return CheckResult(
        label,
        Status.OK,
        f"{count} {noun}, the newest sent with every request",
        hint="yarvis memory list",
    )


def check_voice(ctx: DoctorContext, config: Config) -> CheckResult:
    """With Cartesia chosen, its key should be stored (Kokoro speaks otherwise, ADR 0020)."""
    label = "Voice"
    if config.voice.tts != "cartesia":
        return CheckResult(label, Status.OK, "Kokoro, on this computer")
    if ctx.secrets.get(SecretName.CARTESIA) is None:
        return CheckResult(
            label,
            Status.WARN,
            "Cartesia chosen, but no key stored: Kokoro speaks",
            hint="Run: yarvis key set cartesia",
        )
    return CheckResult(
        label,
        Status.OK,
        f"Cartesia ({config.voice.cartesia_model}), Kokoro as backup",
        hint=(
            "Each sentence spoken is sent to Cartesia. Opt out of training at "
            "https://play.cartesia.ai/privacy/data-controls"
        ),
    )
