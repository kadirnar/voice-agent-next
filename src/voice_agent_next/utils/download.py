"""Model/asset downloads with a shared, cross-platform cache.

* Cache directory: ``$VAN_CACHE_DIR`` or the platform user cache dir
  (``~/.cache/voice-agent-next/models`` on Linux, ``~/Library/Caches/...`` on macOS,
  ``%LOCALAPPDATA%\\voice-agent-next\\Cache\\models`` on Windows).
* Writes are atomic (``.part`` file + ``os.replace``), so concurrent downloads of the
  same file from several processes are safe.
* ``VAN_OFFLINE=1`` forbids network access (missing files raise).
* Hugging Face files use ``huggingface_hub`` when installed (shared HF cache, auth),
  otherwise the public ``resolve`` URL.
* Archives (``.tar.bz2``, ``.tar.gz``, ``.tar.xz``, ``.tar``) are verified, extracted with
  path-traversal protection into a temporary directory and moved into place atomically
  (:func:`download_archive`, :func:`extract_archive`).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import posixpath
import shutil
import tarfile
import tempfile
from pathlib import Path, PurePosixPath, PureWindowsPath

import httpx
import platformdirs

from ..errors import VoiceAgentError
from .deps import is_installed
from .log import logger

__all__ = [
    "DownloadError",
    "cache_dir",
    "download",
    "download_archive",
    "download_async",
    "extract_archive",
    "hf_file",
]

_ARCHIVE_SUFFIXES = (".tar.bz2", ".tbz2", ".tar.gz", ".tgz", ".tar.xz", ".txz", ".tar")
_ARCHIVE_MARKER = ".van-archive.json"
"""Written into an extracted archive directory once extraction completed."""


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


def _archive_stem(filename: str) -> str:
    lower = filename.lower()
    for suffix in _ARCHIVE_SUFFIXES:
        if lower.endswith(suffix):
            return filename[: -len(suffix)]
    raise DownloadError(
        f"unsupported archive type: {filename} (expected one of {', '.join(_ARCHIVE_SUFFIXES)})"
    )


def _read_marker(directory: Path) -> dict[str, str] | None:
    try:
        data = json.loads((directory / _ARCHIVE_MARKER).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def download_archive(
    url: str,
    *,
    sha256: str | None = None,
    subdir: str = "",
    name: str | None = None,
    force: bool = False,
    timeout: float = 60.0,
    client: httpx.Client | None = None,
) -> Path:
    """Download a tar archive, extract it into the cache and return the extracted directory.

    The archive is verified against ``sha256``, extracted safely (:func:`extract_archive`)
    into ``<cache>/<subdir>/<name>`` and then deleted. ``name`` defaults to the archive file
    name without its extension; when the archive holds a single top-level directory, that
    directory's contents become ``<name>`` (``foo.tar.bz2`` holding ``foo/model.onnx`` gives
    ``<cache>/<subdir>/foo/model.onnx``). A marker file records the URL and digest, so later
    calls return the directory without touching the network (``VAN_OFFLINE=1`` works) and a
    changed ``sha256`` triggers a new download.

    Args:
        url: http(s) URL of a ``.tar.bz2``, ``.tar.gz``, ``.tar.xz`` or ``.tar`` file.
        sha256: expected hex digest of the archive.
        subdir: sub-directory of the cache, e.g. ``"sherpa-onnx"``.
        name: directory name in the cache (default: the archive name without extension).
        force: download and extract again even if the directory exists.
        client: optional ``httpx.Client`` (tests inject a mock transport).
    """
    filename = url.rstrip("/").split("/")[-1].split("?")[0]
    stem = name or _archive_stem(filename)
    root = cache_dir() / subdir if subdir else cache_dir()
    root.mkdir(parents=True, exist_ok=True)
    target = root / stem
    if target.is_dir() and not force:
        marker = _read_marker(target)
        if marker is not None and (sha256 is None or marker.get("sha256") == sha256):
            return target
        logger.warning("%s is incomplete or outdated; downloading %s again", target, url)
    archive = download(
        url,
        filename=filename,
        subdir=subdir,
        sha256=sha256,
        force=force,
        timeout=timeout,
        client=client,
    )
    tmp = Path(tempfile.mkdtemp(dir=root, prefix=f".{stem}.", suffix=".tmp"))
    try:
        logger.info("extracting %s -> %s", archive.name, target)
        extract_archive(archive, tmp)
        entries = list(tmp.iterdir())
        source = entries[0] if len(entries) == 1 and entries[0].is_dir() else tmp
        marker_data = {"url": url, "sha256": sha256 or "", "archive": filename}
        (source / _ARCHIVE_MARKER).write_text(json.dumps(marker_data), encoding="utf-8")
        if target.exists():
            shutil.rmtree(target)  # incomplete, outdated or forced: replace it
        try:
            os.replace(source, target)
        except OSError:
            # another process may have finished the same extraction first
            if _read_marker(target) is None:
                raise
    except (OSError, tarfile.TarError) as exc:
        raise DownloadError(f"cannot extract {archive} into {target}: {exc}") from exc
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        try:
            archive.unlink(missing_ok=True)  # the extracted copy is all we keep
        except OSError:  # e.g. still open by another process on Windows
            pass
    return target


def _unsafe_path(name: str) -> bool:
    """True for absolute paths, drive letters and ``..`` components (POSIX or Windows)."""
    if not name or name.startswith(("/", "\\")):
        return True
    win = PureWindowsPath(name)
    if win.drive or win.root:
        return True
    return ".." in PurePosixPath(name).parts or ".." in win.parts


def _check_member(member: tarfile.TarInfo) -> None:
    if _unsafe_path(member.name):
        raise DownloadError(f"unsafe path in archive: {member.name!r}")
    if member.issym() or member.islnk():
        link = member.linkname
        # symlinks resolve against their own directory, hard links against the archive root
        base = posixpath.dirname(member.name) if member.issym() else ""
        resolved = posixpath.normpath(posixpath.join(base, link.replace("\\", "/")))
        if not link or _unsafe_path(link) or _unsafe_path(resolved):
            raise DownloadError(f"unsafe link in archive: {member.name!r} -> {link!r}")
    elif not (member.isfile() or member.isdir()):
        raise DownloadError(f"unsupported member type in archive: {member.name!r}")


def extract_archive(archive: str | os.PathLike[str], destination: str | os.PathLike[str]) -> None:
    """Extract a tar archive (any compression) into ``destination``, refusing unsafe members.

    Absolute paths, ``..`` components, Windows drive letters, links pointing outside the
    archive and device files raise :class:`DownloadError` (on top of Python's ``"data"``
    extraction filter where available). Members are extracted in one sequential pass;
    on error the destination may hold a partial extraction, so extract into a temporary
    directory (as :func:`download_archive` does).
    """
    dest = Path(destination)
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:*") as tar:
        for member in tar:
            _check_member(member)
            if hasattr(tarfile, "data_filter"):  # Python >= 3.11.4
                tar.extract(member, dest, set_attrs=False, filter="data")
            else:  # pragma: no cover - the member was validated above
                tar.extract(member, dest, set_attrs=False)


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
