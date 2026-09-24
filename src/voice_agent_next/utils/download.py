"""Model/asset downloads with a shared, cross-platform cache.

* Cache directory: ``$VAN_CACHE_DIR`` or the platform user cache dir
  (``~/.cache/voice-agent-next/models`` on Linux, ``~/Library/Caches/...`` on macOS,
  ``%LOCALAPPDATA%\\voice-agent-next\\Cache\\models`` on Windows).
* Writes are atomic (``.part`` file + ``os.replace``), so concurrent downloads of the
  same file from several processes are safe.
* ``VAN_OFFLINE=1`` forbids network access (missing files raise).
* Hugging Face files use ``huggingface_hub`` when installed (shared HF cache, auth),
  otherwise the public ``resolve`` URL.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import tempfile
from pathlib import Path

import httpx
import platformdirs

from ..errors import VoiceAgentError
from .deps import is_installed
from .log import logger

__all__ = ["DownloadError", "cache_dir", "download", "download_async", "hf_file"]


class DownloadError(VoiceAgentError):
    """A model/asset could not be downloaded or failed verification."""


def cache_dir() -> Path:
    """Root directory for downloaded models (created on demand)."""
    root = os.environ.get("VAN_CACHE_DIR")
    path = Path(root) if root else platformdirs.user_cache_path("voice-agent-next") / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _offline() -> bool:
    return os.environ.get("VAN_OFFLINE", "").lower() in ("1", "true", "yes")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(
    url: str,
    *,
    filename: str | None = None,
    subdir: str = "",
    sha256: str | None = None,
    force: bool = False,
    timeout: float = 60.0,
    client: httpx.Client | None = None,
) -> Path:
    """Download ``url`` into the cache (if not already there) and return the local path.

    Args:
        url: http(s) URL.
        filename: local file name (default: last URL path segment).
        subdir: sub-directory of the cache, e.g. ``"silero"``.
        sha256: expected hex digest; verified after download (and on cache hits).
        force: re-download even if the file exists.
        client: optional ``httpx.Client`` (tests inject a mock transport).
    """
    name = filename or url.rstrip("/").split("/")[-1].split("?")[0]
    target_dir = cache_dir() / subdir if subdir else cache_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / name
    if target.exists() and not force:
        if sha256 is None or _sha256(target) == sha256:
            return target
        logger.warning("checksum mismatch for cached %s; re-downloading", target)
    if _offline():
        raise DownloadError(f"{name} is not cached and VAN_OFFLINE is set (url: {url})")
    logger.info("downloading %s -> %s", url, target)
    own_client = client is None
    http = client or httpx.Client(follow_redirects=True, timeout=timeout)
    fd, tmp_name = tempfile.mkstemp(dir=target_dir, prefix=f".{name}.", suffix=".part")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as out, http.stream("GET", url) as resp:
            if resp.status_code != 200:
                raise DownloadError(f"GET {url} failed with HTTP {resp.status_code}")
            for chunk in resp.iter_bytes(1 << 16):
                out.write(chunk)
        if sha256 is not None:
            digest = _sha256(tmp)
            if digest != sha256:
                raise DownloadError(f"checksum mismatch for {url}: {digest} != {sha256}")
        os.replace(tmp, target)
    except httpx.HTTPError as exc:
        raise DownloadError(f"download of {url} failed: {exc}") from exc
    finally:
        tmp.unlink(missing_ok=True)
        if own_client:
            http.close()
    return target


async def download_async(url: str, **kwargs: object) -> Path:
    """:func:`download` in a worker thread (never blocks the event loop)."""
    return await asyncio.to_thread(download, url, **kwargs)  # type: ignore[arg-type]


def hf_file(
    repo_id: str,
    filename: str,
    *,
    revision: str = "main",
    sha256: str | None = None,
    repo_type: str = "model",
) -> Path:
    """Return a local path to ``filename`` from a Hugging Face repository."""
    if is_installed("huggingface_hub"):
        from huggingface_hub import hf_hub_download

        try:
            return Path(
                hf_hub_download(
                    repo_id,
                    filename,
                    revision=revision,
                    repo_type=repo_type,
                    local_files_only=_offline(),
                )
            )
        except Exception as exc:
            raise DownloadError(f"failed to fetch {repo_id}/{filename}: {exc}") from exc
    prefix = "" if repo_type == "model" else f"{repo_type}s/"
    url = f"https://huggingface.co/{prefix}{repo_id}/resolve/{revision}/{filename}"
    return download(
        url, filename=filename.split("/")[-1], subdir=f"hf/{repo_id}/{revision}", sha256=sha256
    )
