from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from fakes import ASSEMBLYAI_KEY, MIC, SPEAKER, VALID_KEY, FakeAudioProbe, FakeKeyring
from test_runtime_and_run import install_models
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
    check_models,
    check_platform,
    check_python,
    check_speech_recognition,
    check_storage,
    find_device,
    has_failures,
    run_doctor,
)
from yarvis.security.secrets import KEYRING_SERVICE

if TYPE_CHECKING:
    from pathlib import Path

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
    install_models(ctx.paths)
    results = run_doctor(ctx)
    assert [result.status for result in results] == [Status.OK] * 13
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
        ("Windows", "AMD64", Status.WARN),
        ("FreeBSD", "amd64", Status.FAIL),
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


def test_on_windows_the_user_profile_is_private(
    ctx: DoctorContext, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    windows = replace(ctx, system="Windows", machine="AMD64")
    shared = replace(ctx.paths, config_dir=ctx.paths.data_dir)  # as on Windows
    windows = replace(windows, paths=shared)
    shared.data_dir.mkdir(parents=True, mode=0o755)
    write_default_config(shared.config_file)
    monkeypatch.setenv("HOME", str(tmp_path))  # the paths are inside the user's profile
    assert check_storage(windows).status is Status.OK
    assert check_config(windows)[0].status is Status.OK
    monkeypatch.setenv("HOME", str(tmp_path / "someone-else"))
    result = check_storage(windows)
    assert result.status is Status.WARN
    assert result.detail.count(str(shared.data_dir)) == 1  # listed once
    assert result.hint == "Keep Yarvis's folders inside your user profile (unset YARVIS_HOME)."
    assert "chmod" not in (check_config(windows)[0].hint or "")


def test_windows_is_early_support(ctx: DoctorContext) -> None:
    result = check_platform(replace(ctx, system="Windows", machine="AMD64"))
    assert result.status is Status.WARN
    assert result.hint is not None
    assert "Spotify work" in result.hint


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


def test_missing_models_warn(ctx: DoctorContext) -> None:
    result = check_models(ctx, Config())
    assert result.status is Status.WARN
    assert "Kokoro voice" in result.detail
    assert "Whisper small.en" in result.detail
    install_models(ctx.paths)
    assert check_models(ctx, Config()).status is Status.OK


def test_apple_silicon_needs_parakeet_not_whisper(ctx: DoctorContext) -> None:
    mac = replace(ctx, parakeet_installed=True)
    install_models(ctx.paths)  # Whisper is there, Parakeet is not
    result = check_models(mac, Config())
    assert result.status is Status.WARN
    assert "Parakeet speech recognition" in result.detail
    assert "Whisper" not in result.detail
    assert "(~2471 MB)" in result.detail
    install_models(ctx.paths, whisper=False, wake_word=False, parakeet=True)
    assert check_models(mac, Config()).status is Status.OK
    whisper = Config.model_validate({"voice": {"stt": "faster-whisper"}})
    assert check_models(mac, whisper).status is Status.OK


def test_a_windows_pc_needs_the_onnx_build_of_parakeet(ctx: DoctorContext) -> None:
    pc = replace(ctx, system="Windows", machine="AMD64", parakeet_installed=True)
    install_models(ctx.paths, parakeet=True)  # the Mac's build is there: not the PC's
    result = check_models(pc, Config())
    assert result.status is Status.WARN
    assert "Parakeet speech recognition" in result.detail
    assert "(~2512 MB)" in result.detail


@pytest.mark.parametrize(
    ("version", "status"), [("14.5", Status.OK), ("26.1", Status.OK), ("13.6.1", Status.FAIL)]
)
def test_macos_version(ctx: DoctorContext, version: str, status: Status) -> None:
    result = check_platform(replace(ctx, mac_version=version))
    assert result.status is status
    assert f"macOS {version}" in result.detail


def test_missing_wake_word_model_warns_only_when_enabled(ctx: DoctorContext) -> None:
    install_models(ctx.paths, wake_word=False)
    result = check_models(ctx, Config())
    assert result.status is Status.WARN
    assert "wake word" in result.detail
    off = Config.model_validate({"voice": {"wake_word": False}})
    assert check_models(ctx, off).status is Status.OK


def test_speech_recognition_names_the_engine_that_hears(
    ctx: DoctorContext, fake_keyring: FakeKeyring
) -> None:
    on_the_mac = check_speech_recognition(replace(ctx, parakeet_installed=True), Config())
    assert on_the_mac.status is Status.OK
    assert on_the_mac.detail == "Parakeet on this computer"
    assert on_the_mac.hint == "For the most accurate transcripts: yarvis key set assemblyai"
    fake_keyring.store[(KEYRING_SERVICE, "assemblyai")] = ASSEMBLYAI_KEY
    cloud = check_speech_recognition(ctx, Config())
    assert cloud.detail == "AssemblyAI (EU servers), Whisper small.en as backup"
    nearest = Config.model_validate({"voice": {"cloud_region": "global"}})
    assert "nearest servers" in check_speech_recognition(ctx, nearest).detail
    off = Config.model_validate({"voice": {"cloud_stt": False}})
    assert check_speech_recognition(ctx, off).detail.endswith("(cloud_stt is off)")
