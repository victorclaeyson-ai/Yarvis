from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from fakes import POSIX_ONLY, posix_mode, symlink_or_skip
from yarvis import paths as yarvis_paths
from yarvis.paths import (
    HOME_ENV_VAR,
    AppPaths,
    ensure_private_dir,
    is_private,
    write_private_file,
)

if TYPE_CHECKING:
    from pathlib import Path


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
    assert posix_mode(directory) in {0o700, None}
    assert is_private(directory)


@POSIX_ONLY
def test_is_private_detects_group_access(tmp_path: Path) -> None:
    file = tmp_path / "f"
    file.write_text("x")
    file.chmod(0o640)
    assert not is_private(file)


def test_write_private_file_creates_0600(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "secret.toml"
    write_private_file(target, "hello")
    assert target.read_text() == "hello"
    assert posix_mode(target) in {0o600, None}
    assert posix_mode(target.parent) in {0o700, None}


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
    symlink_or_skip(link, victim)
    with pytest.raises(OSError, match=r"symbolic links|Too many levels"):
        write_private_file(link, "attacker", overwrite=True)
    assert victim.read_text() == "original"


def test_without_o_nofollow_symlinks_are_refused_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(yarvis_paths, "_NO_FOLLOW", 0)  # as on Windows
    victim = tmp_path / "victim"
    victim.write_text("original")
    link = tmp_path / "link"
    symlink_or_skip(link, victim)
    with pytest.raises(OSError, match="symbolic links"):
        write_private_file(link, "attacker", overwrite=True)
    assert victim.read_text() == "original"


def test_on_windows_private_means_inside_the_user_profile(tmp_path: Path) -> None:
    inside = ensure_private_dir(tmp_path / "AppData" / "Local" / "yarvis")
    inside.chmod(0o777)  # Windows reports every folder like this
    assert is_private(inside, windows=True, home=tmp_path)
    assert not is_private(inside, windows=True, home=tmp_path / "other-user")
    assert not is_private(inside, windows=False)  # on a Mac or Linux, 0o777 is open to all


def test_ensure_all_creates_private_directories(tmp_path: Path) -> None:
    paths = AppPaths.under(tmp_path / "deep" / "home")
    paths.ensure_all()
    assert all(is_private(directory) for directory in paths.all_dirs())
