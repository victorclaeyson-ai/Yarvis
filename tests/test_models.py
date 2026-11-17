from __future__ import annotations

import hashlib
from contextlib import contextmanager
from typing import TYPE_CHECKING

import httpx2
import pytest

from fakes import posix_mode
from yarvis.models import (
    KOKORO_FILES,
    PARAKEET_FILES,
    WAKE_WORD_FILES,
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
    assert posix_mode(path) in {0o600, None}
    assert is_present(model_for(PAYLOAD), tmp_path)
    assert missing((model_for(PAYLOAD),), tmp_path) == []


@pytest.mark.parametrize(
    ("served", "match"),
    [
        (PAYLOAD[:-1] + b"X", "hash mismatch"),
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
    for model in (*KOKORO_FILES, *WAKE_WORD_FILES):
        assert model.url.startswith("https://github.com/")
        assert len(model.digest) == 64
        assert model.algorithm == "sha256"
    for model in PARAKEET_FILES:  # pinned to one commit, never to a branch
        assert model.url.startswith("https://huggingface.co/mlx-community/")
        assert "/resolve/8ae155301e23d820d82aa60d24817c900e69e487/" in model.url
        assert model.url.endswith(f"/{model.name}")
    weights = PARAKEET_FILES[-1]
    assert weights.algorithm == "sha256" and len(weights.digest) == 64


def test_git_blob_ids_are_checked(tmp_path: Path) -> None:
    """Small files kept in git are verified against their git object id."""
    data = b'{"hello": "world"}\n'
    blob = hashlib.sha1(b"blob %d\0" % len(data) + data, usedforsecurity=False).hexdigest()
    config = ModelFile("config.json", "https://example.com/c", blob, len(data), "git-blob")
    assert download(config, tmp_path, opener=opener_for(data)).read_bytes() == data
    tampered = ModelFile("other.json", "https://example.com/c", blob, len(data), "git-blob")
    with pytest.raises(ModelIntegrityError, match="hash mismatch"):
        download(tampered, tmp_path, opener=opener_for(data.replace(b"world", b"evil!")))


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
