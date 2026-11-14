from __future__ import annotations

import hashlib
import stat
from contextlib import contextmanager
from typing import TYPE_CHECKING

import httpx2
import pytest

from yarvis.models import (
    KOKORO_FILES,
    WHISPER_APPROX_MB,
    WHISPER_REVISIONS,
    ModelFile,
    ModelIntegrityError,
    _https_stream,  # pyright: ignore[reportPrivateUsage]
    _require_https,  # pyright: ignore[reportPrivateUsage]
    download,
    is_present,
    missing,
    whisper_dir,
    whisper_present,
)

if TYPE_CHECKING:
    from collections.abc import Generator, Iterator
    from pathlib import Path

PAYLOAD = b"pretend this is a neural network" * 100


def model_for(data: bytes) -> ModelFile:
    return ModelFile(
        "m.onnx", "https://example.com/m.onnx", hashlib.sha256(data).hexdigest(), len(data)
    )


class FakeResponse:
    def __init__(self, data: bytes) -> None:
        self.data = data

    def iter_bytes(self) -> Iterator[bytes]:
        for start in range(0, len(self.data), 256):
            yield self.data[start : start + 256]


def opener_for(data: bytes):
    @contextmanager
    def opener(url: str) -> Generator[FakeResponse]:
        assert url.startswith("https://")
        yield FakeResponse(data)

    return opener


def test_download_verifies_and_installs_privately(tmp_path: Path) -> None:
    progress: list[int] = []
    path = download(
        model_for(PAYLOAD), tmp_path, opener=opener_for(PAYLOAD), progress=progress.append
    )
    assert path.read_bytes() == PAYLOAD
    assert sum(progress) == len(PAYLOAD)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert is_present(model_for(PAYLOAD), tmp_path)
    assert missing((model_for(PAYLOAD),), tmp_path) == []


@pytest.mark.parametrize(
    ("served", "match"),
    [
        (PAYLOAD[:-1] + b"X", "SHA-256 mismatch"),
        (PAYLOAD[:-10], "expected"),
        (PAYLOAD + b"extra bytes", "larger than expected"),
    ],
)
def test_bad_downloads_never_become_visible(tmp_path: Path, served: bytes, match: str) -> None:
    with pytest.raises(ModelIntegrityError, match=match):
        download(model_for(PAYLOAD), tmp_path, opener=opener_for(served))
    assert list(tmp_path.iterdir()) == []


def test_pinned_models_are_complete() -> None:
    assert set(WHISPER_REVISIONS) == set(WHISPER_APPROX_MB)
    assert all(len(revision) == 40 for revision in WHISPER_REVISIONS.values())
    for model in KOKORO_FILES:
        assert model.url.startswith("https://github.com/")
        assert len(model.sha256) == 64


def test_whisper_present(tmp_path: Path) -> None:
    assert not whisper_present(tmp_path, "base.en")
    snapshot = (
        whisper_dir(tmp_path)
        / "models--Systran--faster-whisper-base.en"
        / "snapshots"
        / WHISPER_REVISIONS["base.en"]
    )
    snapshot.mkdir(parents=True)
    (snapshot / "model.bin").write_bytes(b"x")
    assert whisper_present(tmp_path, "base.en")


def test_plain_http_is_refused() -> None:
    with pytest.raises(ModelIntegrityError, match="non-HTTPS"):
        _require_https(httpx2.Request("GET", "http://example.com/model.onnx"))
    _require_https(httpx2.Request("GET", "https://example.com/model.onnx"))


def test_https_stream_reports_misuse() -> None:
    stream = _https_stream("https://example.com")
    with pytest.raises(RuntimeError, match="not started"):
        stream.iter_bytes()
    stream.__exit__(None, None, None)
