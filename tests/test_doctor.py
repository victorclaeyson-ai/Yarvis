from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from fakes import MIC, SPEAKER, VALID_KEY, FakeAudioProbe, FakeKeyring
from yarvis.config import Config, parse_config, write_default_config
from yarvis.doctor import (
    CheckResult,
    DoctorContext,
    Status,
    check_api_key,
    check_audio_input,
    check_audio_output,
    check_config,
    check_keychain,
    check_platform,
    check_python,
    check_storage,
    find_device,
    has_failures,
    run_doctor,
)
from yarvis.security.secrets import KEYRING_SERVICE

if TYPE_CHECKING:
    from yarvis.paths import AppPaths
    from yarvis.security.secrets import SecretStore


@pytest.fixture
def ctx(paths: AppPaths, secrets: SecretStore, audio: FakeAudioProbe) -> DoctorContext:
    return DoctorContext(
        paths=paths,
        secrets=secrets,
        audio=audio,
        python_version=(3, 12, 4),
        system="Darwin",
        machine="arm64",
    )


def test_current_context_describes_this_machine(
    paths: AppPaths, secrets: SecretStore, audio: FakeAudioProbe
) -> None:
    ctx = DoctorContext.current(paths=paths, secrets=secrets, audio=audio)
    assert ctx.python_version >= (3, 12, 0)
    assert ctx.system


def test_healthy_machine_passes(ctx: DoctorContext, fake_keyring: FakeKeyring) -> None:
    fake_keyring.store[(KEYRING_SERVICE, "anthropic")] = VALID_KEY
    results = run_doctor(ctx)
    assert [result.status for result in results] == [Status.OK] * 8
    assert not has_failures(results)


def test_old_python_fails(ctx: DoctorContext) -> None:
    result = check_python(replace(ctx, python_version=(3, 11, 9)))
    assert result.status is Status.FAIL
    assert result.hint is not None


@pytest.mark.parametrize(
    ("system", "machine", "status"),
    [
        ("Darwin", "arm64", Status.OK),
        ("Darwin", "x86_64", Status.WARN),
        ("Linux", "aarch64", Status.OK),
        ("Windows", "AMD64", Status.FAIL),
    ],
)
def test_platform_support(ctx: DoctorContext, system: str, machine: str, status: Status) -> None:
    assert check_platform(replace(ctx, system=system, machine=machine)).status is status


def test_missing_config_uses_defaults(ctx: DoctorContext) -> None:
    result, config = check_config(ctx)
    assert result.status is Status.OK
    assert config == Config()


def test_valid_config_is_ok(ctx: DoctorContext) -> None:
    write_default_config(ctx.paths.config_file)
    result, _ = check_config(ctx)
    assert result.status is Status.OK
    assert str(ctx.paths.config_file) in result.detail


def test_invalid_config_fails(ctx: DoctorContext) -> None:
    write_default_config(ctx.paths.config_file)
    ctx.paths.config_file.write_text("[llm]\nprovider = 3\n")
    result, config = check_config(ctx)
    assert result.status is Status.FAIL
    assert config == Config()


def test_world_readable_config_warns(ctx: DoctorContext) -> None:
    write_default_config(ctx.paths.config_file)
    ctx.paths.config_file.chmod(0o644)
    result, _ = check_config(ctx)
    assert result.status is Status.WARN
    assert result.hint is not None
    assert "chmod 600" in result.hint


def test_relaxed_security_warns(ctx: DoctorContext) -> None:
    write_default_config(ctx.paths.config_file)
    ctx.paths.config_file.write_text("[security]\nallow_tier3 = true\n")
    result, _ = check_config(ctx)
    assert result.status is Status.WARN
    assert "allow_tier3" in result.detail


def test_exposed_storage_warns(ctx: DoctorContext) -> None:
    ctx.paths.data_dir.mkdir(parents=True, mode=0o755)
    ctx.paths.data_dir.chmod(0o755)
    result = check_storage(ctx)
    assert result.status is Status.WARN
    assert str(ctx.paths.data_dir) in result.detail


def test_missing_keychain_warns(ctx: DoctorContext, fake_keyring: FakeKeyring) -> None:
    fake_keyring.available = False
    assert check_keychain(ctx).status is Status.WARN


def test_missing_api_key_fails_with_keychain_hint(ctx: DoctorContext) -> None:
    result = check_api_key(ctx, Config())
    assert result.status is Status.FAIL
    assert result.hint == "Run: yarvis key set anthropic"


def test_missing_api_key_without_keychain_suggests_env_var(
    ctx: DoctorContext, fake_keyring: FakeKeyring
) -> None:
    fake_keyring.available = False
    result = check_api_key(ctx, Config())
    assert result.hint is not None
    assert "ANTHROPIC_API_KEY" in result.hint


def test_api_key_from_environment(ctx: DoctorContext, environ: dict[str, str]) -> None:
    environ["ANTHROPIC_API_KEY"] = VALID_KEY
    result = check_api_key(ctx, Config())
    assert result.status is Status.OK
    assert "$ANTHROPIC_API_KEY" in result.detail
    assert VALID_KEY not in result.detail


def test_ollama_needs_no_key(ctx: DoctorContext) -> None:
    config = parse_config('[llm]\nprovider = "ollama"\n')
    assert check_api_key(ctx, config).status is Status.OK


def test_broken_audio_fails_with_platform_hint(ctx: DoctorContext, audio: FakeAudioProbe) -> None:
    audio.broken = True
    mac = check_audio_input(ctx, Config())
    linux = check_audio_output(replace(ctx, system="Linux"), Config())
    assert mac.status is Status.FAIL
    assert linux.hint is not None
    assert "libportaudio2" in linux.hint


def test_no_default_microphone_fails(ctx: DoctorContext, audio: FakeAudioProbe) -> None:
    audio.input_device = None
    assert check_audio_input(ctx, Config()).status is Status.FAIL


def test_configured_device_is_found_by_name_or_index(ctx: DoctorContext) -> None:
    by_name = parse_config('[voice]\ninput_device = "test microphone"\n')
    by_index = parse_config("[voice]\noutput_device = 1\n")
    assert check_audio_input(ctx, by_name).detail == MIC.name
    assert check_audio_output(ctx, by_index).detail == SPEAKER.name


def test_configured_device_missing_fails(ctx: DoctorContext) -> None:
    config = parse_config('[voice]\ninput_device = "Test Speakers"\n')  # exists, but output-only
    result = check_audio_input(ctx, config)
    assert result.status is Status.FAIL
    assert "not found" in result.detail


def test_find_device_skips_wrong_kind() -> None:
    assert find_device([MIC, SPEAKER], 0, kind="output") is None
    assert find_device([MIC, SPEAKER], 0, kind="input") == MIC


def test_has_failures() -> None:
    ok = CheckResult("a", Status.OK, "")
    warn = CheckResult("b", Status.WARN, "")
    fail = CheckResult("c", Status.FAIL, "")
    assert not has_failures([ok, warn])
    assert has_failures([ok, fail])
