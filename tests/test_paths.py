from __future__ import annotations

import stat
from typing import TYPE_CHECKING

import pytest

from yarvis.paths import (
    HOME_ENV_VAR,
    AppPaths,
    ensure_private_dir,
    is_private,
    write_private_file,
)

if TYPE_CHECKING:
    from pathlib import Path


def mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def test_home_env_var_places_everything_under_one_root(tmp_path: Path) -> None:
    paths = AppPaths.default({HOME_ENV_VAR: str(tmp_path)})
    assert all(directory.parent == tmp_path for directory in paths.all_dirs())
    assert paths.config_file == tmp_path / "config" / "config.toml"


def test_platform_default_paths_mention_app_name() -> None:
    paths = AppPaths.default({})
    assert all("yarvis" in str(directory) for directory in paths.all_dirs())


def test_ensure_private_dir_tightens_existing_permissions(tmp_path: Path) -> None:
    directory = tmp_path / "open"
    directory.mkdir(mode=0o755)
    ensure_private_dir(directory)
    assert mode(directory) == 0o700
    assert is_private(directory)


def test_is_private_detects_group_access(tmp_path: Path) -> None:
    file = tmp_path / "f"
    file.write_text("x")
    file.chmod(0o640)
    assert not is_private(file)


def test_write_private_file_creates_0600(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "secret.toml"
    write_private_file(target, "hello")
    assert target.read_text() == "hello"
    assert mode(target) == 0o600
    assert mode(target.parent) == 0o700


def test_write_private_file_refuses_to_overwrite_by_default(tmp_path: Path) -> None:
    target = tmp_path / "file"
    write_private_file(target, "one")
    with pytest.raises(FileExistsError):
        write_private_file(target, "two")
    write_private_file(target, "two", overwrite=True)
    assert target.read_text() == "two"


def test_write_private_file_does_not_follow_symlinks(tmp_path: Path) -> None:
    victim = tmp_path / "victim"
    victim.write_text("original")
    link = tmp_path / "link"
    link.symlink_to(victim)
    with pytest.raises(OSError, match=r"symbolic links|Too many levels"):
        write_private_file(link, "attacker", overwrite=True)
    assert victim.read_text() == "original"


def test_ensure_all_creates_private_directories(tmp_path: Path) -> None:
    paths = AppPaths.under(tmp_path / "deep" / "home")
    paths.ensure_all()
    assert all(is_private(directory) for directory in paths.all_dirs())
