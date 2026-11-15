from __future__ import annotations

import stat
from typing import TYPE_CHECKING

import pytest

from yarvis.config import (
    DEFAULT_CONFIG_TOML,
    Config,
    ConfigError,
    load_config,
    parse_config,
    write_default_config,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_defaults_are_secure() -> None:
    config = Config()
    assert not config.security.allow_tier3
    assert config.security.confirm_tier2
    assert config.security_relaxations() == []


def test_default_template_matches_model_defaults() -> None:
    assert parse_config(DEFAULT_CONFIG_TOML) == Config()


def test_missing_file_gives_defaults(tmp_path: Path) -> None:
    assert load_config(tmp_path / "absent.toml") == Config()


def test_values_are_loaded(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text('[assistant]\nname = "Yeti"\nlanguage = "nl-BE"\n[llm]\nprovider = "ollama"\n')
    config = load_config(path)
    assert config.assistant.name == "Yeti"
    assert config.assistant.language == "nl-BE"
    assert config.llm.provider == "ollama"


def test_unknown_keys_are_rejected_with_location() -> None:
    with pytest.raises(ConfigError, match=r"llm\.modle: Extra inputs are not permitted"):
        parse_config('[llm]\nmodle = "typo"\n', source="config.toml")


def test_invalid_values_list_every_problem() -> None:
    with pytest.raises(ConfigError) as info:
        parse_config('[llm]\nprovider = "skynet"\nmax_tokens = 0\n')
    message = str(info.value)
    assert "2 invalid setting(s)" in message
    assert "llm.provider" in message
    assert "llm.max_tokens" in message


def test_invalid_toml_is_reported() -> None:
    with pytest.raises(ConfigError, match="not valid TOML"):
        parse_config("[llm\n")


def test_secrets_in_config_are_refused() -> None:
    with pytest.raises(ConfigError, match="OS keychain"):
        parse_config('[llm]\nmodel = "sk-ant-api03-abcdefghijklmnop"\n')  # gitleaks:allow


def test_unreadable_file_is_reported(tmp_path: Path) -> None:
    directory = tmp_path / "config.toml"
    directory.mkdir()
    with pytest.raises(ConfigError, match="cannot read"):
        load_config(directory)


def test_config_is_immutable() -> None:
    config = Config()
    with pytest.raises(ValueError, match="frozen"):
        config.assistant.name = "Changed"


def test_write_default_config_is_private(tmp_path: Path) -> None:
    path = tmp_path / "cfg" / "config.toml"
    write_default_config(path)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert load_config(path) == Config()
    with pytest.raises(FileExistsError):
        write_default_config(path)


@pytest.mark.parametrize(
    ("toml", "expected"),
    [
        ("[security]\nallow_tier3 = true\n", "allow_tier3"),
        ("[security]\nconfirm_tier2 = false\n", "confirm_tier2"),
        ('[llm]\nprovider = "ollama"\nollama_url = "http://192.168.1.20:11434"\n', "unencrypted"),
    ],
)
def test_security_relaxations_are_reported(toml: str, expected: str) -> None:
    relaxations = parse_config(toml).security_relaxations()
    assert len(relaxations) == 1
    assert expected in relaxations[0]


@pytest.mark.parametrize(
    "url",
    ["http://localhost:11434", "http://127.0.0.1:11434/", "http://[::1]:11434", "https://gpu.lan"],
)
def test_local_or_encrypted_ollama_is_fine(url: str) -> None:
    config = parse_config(f'[llm]\nprovider = "ollama"\nollama_url = "{url}"\n')
    assert config.security_relaxations() == []


def test_vocabulary() -> None:
    assert Config().voice.vocabulary == ("Yarvis",)
    config = parse_config('[voice]\nvocabulary = [" Zedelgem ", "Leuven"]\n')
    assert config.voice.vocabulary == ("Zedelgem", "Leuven")
    with pytest.raises(ConfigError, match="vocabulary"):
        parse_config('[voice]\nvocabulary = [""]\n')
    with pytest.raises(ConfigError, match="vocabulary"):
        parse_config(f"[voice]\nvocabulary = [{', '.join(['"x"'] * 51)}]\n")


def test_thinking_pause_must_not_be_shorter_than_end_of_turn() -> None:
    with pytest.raises(ConfigError, match="thinking_pause_ms"):
        parse_config("[voice]\nend_of_turn_ms = 1500\nthinking_pause_ms = 1000\n")
    config = parse_config("[voice]\nfollow_up_s = 0\n")
    assert config.voice.follow_up_s == 0


def test_wake_word_and_barge_in_settings() -> None:
    voice = Config().voice
    assert voice.wake_word and voice.wake_chime
    assert voice.wake_word_threshold == 0.5
    assert voice.barge_in == "auto"
    custom = parse_config(
        '[voice]\nwake_word = false\nbarge_in = "voice"\nwake_word_threshold = 0.7'
    )
    assert not custom.voice.wake_word
    assert custom.voice.barge_in == "voice"
    with pytest.raises(ConfigError, match="barge_in"):
        parse_config('[voice]\nbarge_in = "always"')
    with pytest.raises(ConfigError, match="wake_word_threshold"):
        parse_config("[voice]\nwake_word_threshold = 1.0")


def test_user_name() -> None:
    assert Config().assistant.user_name is None
    for name in ("Victor", "Jean-Luc", "Anne Marie", "O'Brien", "Zoë"):
        assert parse_config(f'[assistant]\nuser_name = "{name}"').assistant.user_name == name
    for bad in ("x1", " ", "Victor; rm -rf", "a" * 41):
        with pytest.raises(ConfigError, match="user_name"):
            parse_config(f'[assistant]\nuser_name = "{bad}"')


def test_follow_up_window() -> None:
    assert Config().voice.follow_up_s == 15
    assert parse_config("[voice]\nfollow_up_s = 0").voice.follow_up_s == 0
    with pytest.raises(ConfigError, match="follow_up_s"):
        parse_config("[voice]\nfollow_up_s = 61")


def test_staying_open() -> None:
    assert Config().voice.stay_open_s == 300  # five minutes after the last request
    assert parse_config("[voice]\nstay_open_s = 0").voice.stay_open_s == 0
    with pytest.raises(ConfigError, match="stay_open_s"):
        parse_config("[voice]\nstay_open_s = 3601")


def test_tools_are_on_by_default_and_can_be_switched_off() -> None:
    assert Config().tools.timers and Config().tools.computer
    config = parse_config("[tools]\ntimers = false\ncomputer = false")
    assert not config.tools.timers and not config.tools.computer


def test_browser_choice() -> None:
    assert Config().tools.browser == "auto"
    assert Config().tools.typing
    assert parse_config('[tools]\nbrowser = "Brave Browser"').tools.browser == "Brave Browser"
    with pytest.raises(ConfigError, match="browser"):
        parse_config('[tools]\nbrowser = "Netscape"')
