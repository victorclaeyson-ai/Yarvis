"""Filesystem locations used by Yarvis, created with private permissions.

Everything Yarvis stores (config, memory, logs) lives in per-user directories that are
readable only by the current user. ``YARVIS_HOME`` overrides all locations at once, which
is useful for portable installs and for tests.
"""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from platformdirs import PlatformDirs

if TYPE_CHECKING:
    from collections.abc import Mapping

APP_NAME = "yarvis"
HOME_ENV_VAR = "YARVIS_HOME"
PRIVATE_DIR_MODE = 0o700
PRIVATE_FILE_MODE = 0o600


@dataclass(frozen=True, slots=True)
class AppPaths:
    """The set of directories Yarvis reads from and writes to."""

    config_dir: Path
    data_dir: Path
    log_dir: Path
    cache_dir: Path

    @property
    def config_file(self) -> Path:
        """Path of the main TOML configuration file."""
        return self.config_dir / "config.toml"

    @classmethod
    def default(cls, environ: Mapping[str, str] | None = None) -> AppPaths:
        """Resolve platform-appropriate paths, honouring ``YARVIS_HOME`` if set."""
        env = os.environ if environ is None else environ
        if home := env.get(HOME_ENV_VAR):
            return cls.under(Path(home).expanduser())
        dirs = PlatformDirs(APP_NAME, appauthor=False)
        return cls(
            config_dir=dirs.user_config_path,
            data_dir=dirs.user_data_path,
            log_dir=dirs.user_log_path,
            cache_dir=dirs.user_cache_path,
        )

    @classmethod
    def under(cls, root: Path) -> AppPaths:
        """Place every directory under a single root (portable mode)."""
        return cls(
            config_dir=root / "config",
            data_dir=root / "data",
            log_dir=root / "logs",
            cache_dir=root / "cache",
        )

    def all_dirs(self) -> tuple[Path, ...]:
        """Every directory, in a stable order."""
        return (self.config_dir, self.data_dir, self.log_dir, self.cache_dir)


def ensure_private_dir(path: Path) -> Path:
    """Create ``path`` (and parents) if needed and restrict it to the current user."""
    path.mkdir(mode=PRIVATE_DIR_MODE, parents=True, exist_ok=True)
    path.chmod(PRIVATE_DIR_MODE)
    return path


def is_private(path: Path) -> bool:
    """Return True if neither group nor others have any access to ``path``."""
    mode = path.stat().st_mode
    return stat.S_IMODE(mode) & 0o077 == 0


def write_private_file(path: Path, content: str, *, overwrite: bool = False) -> None:
    """Write ``content`` to ``path`` with mode 0600, never following an existing symlink.

    Raises:
        FileExistsError: If the file exists and ``overwrite`` is False.
    """
    ensure_private_dir(path.parent)
    flags = os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW
    flags |= os.O_TRUNC if overwrite else os.O_EXCL
    fd = os.open(path, flags, PRIVATE_FILE_MODE)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(content)
    path.chmod(PRIVATE_FILE_MODE)
