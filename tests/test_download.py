from __future__ import annotations

import hashlib
import io
import os
import stat
import sys
import zipfile

import httpx
import pytest

from voice_agent_next.utils.download import (
    DownloadError,
    cache_dir,
    download,
    download_archive,
    extract_archive,
    hf_file,
)


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):  # type: ignore[no-untyped-def]
    monkeypatch.setenv("VAN_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("VAN_OFFLINE", raising=False)


def client_for(payload: bytes, status: int = 200) -> tuple[httpx.Client, list[str]]:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(status, content=payload)

    return httpx.Client(transport=httpx.MockTransport(handler)), calls


def test_download_caches_and_verifies_checksum() -> None:
    payload = b"model-bytes" * 1000
    digest = hashlib.sha256(payload).hexdigest()
    client, calls = client_for(payload)
    path = download("https://example.com/m/model.onnx", subdir="x", sha256=digest, client=client)
    assert path.read_bytes() == payload
    assert path.parent == cache_dir() / "x"
    again = download("https://example.com/m/model.onnx", subdir="x", sha256=digest, client=client)
    assert again == path
    assert len(calls) == 1  # cache hit
    assert not list(path.parent.glob("*.part"))


def test_download_rejects_bad_checksum_and_http_errors() -> None:
    client, _ = client_for(b"abc")
    with pytest.raises(DownloadError, match="checksum"):
        download("https://example.com/a.bin", sha256="0" * 64, client=client)
    assert not (cache_dir() / "a.bin").exists()
    client404, _ = client_for(b"", status=404)
    with pytest.raises(DownloadError, match="404"):
        download("https://example.com/b.bin", client=client404)


def test_offline_mode_refuses_network(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("VAN_OFFLINE", "1")
    client, calls = client_for(b"x")
    with pytest.raises(DownloadError, match="VAN_OFFLINE"):
        download("https://example.com/c.bin", client=client)
    assert calls == []


def test_hf_file_url_fallback(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import voice_agent_next.utils.download as dl

    seen: dict[str, object] = {}

    def fake_download(url: str, **kwargs: object):  # type: ignore[no-untyped-def]
        seen["url"] = url
        seen.update(kwargs)
        return cache_dir() / "fake"

    monkeypatch.setattr(dl, "is_installed", lambda name: False)
    monkeypatch.setattr(dl, "download", fake_download)
    hf_file("org/repo", "sub/file.onnx", revision="v1")
    assert seen["url"] == "https://huggingface.co/org/repo/resolve/v1/sub/file.onnx"
    assert seen["filename"] == "file.onnx"
    assert seen["subdir"] == "hf/org/repo/v1"  # revisions never share cache entries


def _zip(entries: dict[str, tuple[bytes, int]]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, (data, mode) in entries.items():
            info = zipfile.ZipInfo(name)
            info.external_attr = mode << 16
            zf.writestr(info, data)
    return buf.getvalue()


def test_download_archive_extracts_zip_files_and_keeps_executable_bits() -> None:
    payload = _zip(
        {
            "runner/server": (b"#!/bin/sh\n", stat.S_IFREG | 0o755),
            "runner/lib/libx.so": (b"lib", stat.S_IFREG | 0o644),
        }
    )
    digest = hashlib.sha256(payload).hexdigest()
    client, calls = client_for(payload)
    url = "https://example.com/runner.zip"
    root = download_archive(url, sha256=digest, subdir="z", client=client)
    assert root == cache_dir() / "z" / "runner"
    assert (root / "lib" / "libx.so").read_bytes() == b"lib"
    if sys.platform != "win32":
        assert os.access(root / "server", os.X_OK)
        assert not os.access(root / "lib" / "libx.so", os.X_OK)
    # cached: no second download, and the archive itself is gone
    assert download_archive(url, sha256=digest, subdir="z") == root
    assert len(calls) == 1 and not (cache_dir() / "z" / "runner.zip").exists()


@pytest.mark.parametrize(
    ("name", "mode"),
    [
        ("../evil", stat.S_IFREG | 0o644),
        ("/abs", stat.S_IFREG | 0o644),
        ("link", stat.S_IFLNK | 0o777),
    ],
)
def test_zip_extraction_refuses_unsafe_members(tmp_path, name: str, mode: int) -> None:  # type: ignore[no-untyped-def]
    archive = tmp_path / "bad.zip"
    archive.write_bytes(_zip({name: (b"x", mode)}))
    with pytest.raises(DownloadError):
        extract_archive(archive, tmp_path / "out")
