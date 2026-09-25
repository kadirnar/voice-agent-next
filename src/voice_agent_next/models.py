"""Model manager: a catalog of the model files local providers download, and cache tools.

Providers declare what they download with :func:`register_model` (at import time, next to
their own pinned URLs and digests). The catalog then answers what is available, what is
cached, and where; :func:`download_model`, :func:`verify_model`, :func:`plan_prune` and
:func:`disk_usage` back the ``van models`` command, and :func:`models_for_config` /
:func:`models_for_spec` list what a configuration needs, e.g. to pre-download
everything into a Docker image or onto an offline machine::

    from voice_agent_next import models

    for info in models.models_for_config("agent.yaml").models:
        models.download_model(info)

Files live in two places:

* the shared model cache (:func:`voice_agent_next.utils.download.cache_dir`,
  ``$VAN_CACHE_DIR``): single files (``"url"``), extracted archives (``"archive"``) and
  Hugging Face files fetched without ``huggingface_hub``;
* the Hugging Face cache (``$HF_HUB_CACHE``, ``$HF_HOME/hub`` or
  ``~/.cache/huggingface/hub``): Hugging Face files (``"hf"``) and repositories
  (``"hf-repo"``, e.g. faster-whisper). It is shared with other programs, so the model
  manager reports it but only deletes from it when asked to (``include_hf=True``).
"""

from __future__ import annotations

import fnmatch
import json
import os
import re
import shutil
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .errors import ConfigurationError, MissingDependencyError
from .utils.deps import is_installed
from .utils.download import (
    DownloadError,
    _offline,
    _sha256,
    cache_dir,
    download,
    download_archive,
    hf_file,
)

__all__ = [
    "FileCheck",
    "FileStatus",
    "ModelFile",
    "ModelInfo",
    "ModelStatus",
    "PruneItem",
    "Requirements",
    "UsageReport",
    "apply_prune",
    "catalog",
    "disk_usage",
    "download_model",
    "find_models",
    "get_model",
    "hf_cache_dir",
    "is_offline",
    "model_status",
    "models_for",
    "models_for_config",
    "models_for_spec",
    "plan_prune",
    "register_model",
    "resolve_models",
    "verify_model",
]

Source = Literal["url", "archive", "hf", "hf-repo"]
ModelProgress = Callable[["ModelFile", int, "int | None"], None]
"""``progress(file, downloaded_bytes, total_bytes_or_None)``."""

_KINDS = ("stt", "tts", "llm", "vad", "turn", "engine")
_ARCHIVE_MARKER = ".van-archive.json"  # written by utils.download.download_archive
_ARCHIVE_SUFFIXES = (".tar.bz2", ".tbz2", ".tar.gz", ".tgz", ".tar.xz", ".txz", ".tar", ".zip")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_HF_FALLBACK = "hf"
"""Cache sub-directory of Hugging Face files downloaded without ``huggingface_hub``."""


# ----------------------------------------------------------------------------- data model
@dataclass(frozen=True, slots=True)
class ModelFile:
    """One downloadable file (or archive, or Hugging Face repository) of a model.

    Build it with :meth:`from_url`, :meth:`from_archive`, :meth:`from_hf` or
    :meth:`from_hf_repo`; the fields mirror the arguments of the matching function in
    :mod:`voice_agent_next.utils.download`, so the provider and the model manager agree
    on where the file lives.
    """

    source: Source
    location: str
    """URL (``url``/``archive``) or Hugging Face repository id (``hf``/``hf-repo``)."""
    filename: str = ""
    """Cache file name (``url``), extracted directory name (``archive``) or file in the
    repository (``hf``)."""
    subdir: str = ""
    """Sub-directory of the model cache (``url``/``archive``)."""
    sha256: str | None = None
    """Digest of the file (``url``/``hf``) or of the archive (``archive``)."""
    size: int | None = None
    """Download size in bytes (informational)."""
    revision: str = "main"
    """Hugging Face revision (``hf``/``hf-repo``)."""
    patterns: tuple[str, ...] = ()
    """Files of the repository to fetch (``hf-repo``; empty: everything)."""
    required: tuple[str, ...] = ()
    """Paths that must exist inside the directory (``archive``/``hf-repo``)."""

    @classmethod
    def from_url(
        cls,
        url: str,
        *,
        subdir: str = "",
        filename: str | None = None,
        sha256: str | None = None,
        size: int | None = None,
    ) -> ModelFile:
        """A single file fetched with :func:`~voice_agent_next.utils.download.download`."""
        name = filename or url.rstrip("/").split("/")[-1].split("?")[0]
        return cls("url", url, name, subdir, sha256, size)

    @classmethod
    def from_archive(
        cls,
        url: str,
        *,
        subdir: str = "",
        name: str | None = None,
        sha256: str | None = None,
        size: int | None = None,
        required: Iterable[str] = (),
    ) -> ModelFile:
        """A tar archive fetched and extracted with ``download_archive``."""
        filename = url.rstrip("/").split("/")[-1].split("?")[0]
        stem = name or _archive_stem(filename)
        return cls("archive", url, stem, subdir, sha256, size, required=tuple(required))

    @classmethod
    def from_hf(
        cls,
        repo_id: str,
        filename: str,
        *,
        revision: str = "main",
        sha256: str | None = None,
        size: int | None = None,
    ) -> ModelFile:
        """A file fetched with :func:`~voice_agent_next.utils.download.hf_file`."""
        return cls("hf", repo_id, filename, "", sha256, size, revision)

    @classmethod
    def from_hf_repo(
        cls,
        repo_id: str,
        *,
        revision: str = "main",
        patterns: Iterable[str] = (),
        required: Iterable[str] = (),
        size: int | None = None,
    ) -> ModelFile:
        """(Part of) a Hugging Face repository in the HF cache (``snapshot_download``)."""
        return cls(
            "hf-repo",
            repo_id,
            "",
            "",
            None,
            size,
            revision,
            tuple(patterns),
            tuple(required),
        )

    @property
    def in_hf_cache(self) -> bool:
        """The file lives (or may live) in the Hugging Face cache."""
        return self.source in ("hf", "hf-repo")

    @property
    def label(self) -> str:
        """Short human-readable name."""
        if self.source == "hf-repo":
            return f"{self.location}@{self.revision}"
        if self.source == "hf":
            return f"{self.location}/{self.filename}"
        return self.filename


@dataclass(frozen=True, slots=True)
class ModelInfo:
    """A model a provider can download: ``create(kind, f"{provider}/{model}")`` uses it."""

    provider: str
    """Normalized provider name (``sherpa_onnx``)."""
    model: str
    """Model id as used in specs (``nemo-fastconformer-en-80ms``)."""
    kinds: tuple[str, ...]
    files: tuple[ModelFile, ...]
    license: str = ""
    languages: str = ""
    description: str = ""
    aliases: tuple[str, ...] = ()
    """Other model ids that select the same files."""
    size_hint: int | None = None
    """Total download size when the per-file sizes are unknown or incomplete."""

    @property
    def name(self) -> str:
        """Catalog name: ``<provider>/<model>`` (``sherpa-onnx/silero``)."""
        return f"{self.provider.replace('_', '-')}/{self.model}"

    @property
    def size(self) -> int | None:
        """Download size in bytes (``None``: unknown)."""
        if self.size_hint is not None:
            return self.size_hint
        sizes = [f.size for f in self.files]
        return None if any(s is None for s in sizes) else sum(s or 0 for s in sizes)

    @property
    def in_hf_cache(self) -> bool:
        return any(f.in_hf_cache for f in self.files)


_MODELS: dict[str, ModelInfo] = {}
_LOADED = False


def _normalize(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(".", "_")


def register_model(
    provider: str,
    model: str,
    *,
    kind: str | Sequence[str],
    files: Iterable[ModelFile],
    license: str = "",
    languages: str = "",
    description: str = "",
    aliases: Iterable[str] = (),
    size: int | None = None,
) -> ModelInfo:
    """Declare a downloadable model of ``provider`` (call it at module import time).

    Args:
        provider: provider name as registered with ``@register_provider``.
        model: model id as used in specs (``create(kind, f"{provider}/{model}")``).
        kind: component kind(s) the model serves (``"stt"``, ``"tts"``, ``"vad"``...).
        files: what to download (:class:`ModelFile`), in the order the provider fetches it.
        license: SPDX id or short license description.
        languages: e.g. ``"en"``, ``"multilingual (99)"``, ``"any"``.
        description: one line for ``van models list``.
        aliases: other model ids selecting the same files.
        size: total download size when the per-file sizes are unknown.
    """
    kinds = (kind,) if isinstance(kind, str) else tuple(kind)
    bad = [k for k in kinds if k not in _KINDS]
    if bad or not kinds:
        raise ValueError(f"unknown component kind(s) {bad or kinds!r}; expected {_KINDS}")
    info = ModelInfo(
        provider=_normalize(provider),
        model=model,
        kinds=kinds,
        files=tuple(files),
        license=license,
        languages=languages,
        description=description,
        aliases=tuple(aliases),
        size_hint=size,
    )
    if not info.files:
        raise ValueError(f"model {info.name} declares no files")
    _MODELS[info.name] = info
    return info


def _load_catalog() -> None:
    """Import every provider module once, so that their models are registered."""
    global _LOADED
    if _LOADED:
        return
    _LOADED = True
    from .registry import list_providers

    list_providers()


def catalog(
    *, provider: str | None = None, kind: str | None = None, load: bool = True
) -> list[ModelInfo]:
    """All registered models, optionally filtered by provider (or alias) and kind."""
    if load:
        _load_catalog()
    wanted = _resolve_provider(provider) if provider else None
    models = [
        m
        for m in _MODELS.values()
        if (wanted is None or m.provider == wanted) and (kind is None or kind in m.kinds)
    ]
    return sorted(models, key=lambda m: (m.provider, m.model))


def _resolve_provider(name: str) -> str:
    from .registry import _ALIASES

    key = _normalize(name)
    if any(m.provider == key for m in _MODELS.values()):
        return key
    return _ALIASES.get(key, key)


def _matches(info: ModelInfo, model: str) -> bool:
    return model == info.model or model in info.aliases


def find_models(name: str, *, kind: str | None = None) -> list[ModelInfo]:
    """Catalog entries matching ``name``: ``provider/model``, ``provider`` or a model id."""
    _load_catalog()
    provider, sep, model = name.strip().partition("/")
    if sep:
        wanted = _resolve_provider(provider)
        return [
            m for m in catalog(provider=wanted, kind=kind, load=False) if _matches(m, model.strip())
        ]
    by_model = [m for m in catalog(kind=kind, load=False) if _matches(m, name.strip())]
    by_provider = catalog(provider=name, kind=kind, load=False)
    return by_model + [m for m in by_provider if m not in by_model]


def get_model(name: str, *, kind: str | None = None) -> ModelInfo:
    """The one catalog entry named ``name`` (``provider/model`` or an unambiguous model id)."""
    found = find_models(name, kind=kind)
    if len(found) == 1:
        return found[0]
    if not found:
        raise ConfigurationError(
            f"unknown model {name!r}; run `van models list` to see the catalog"
        )
    names = ", ".join(m.name for m in found)
    raise ConfigurationError(f"{name!r} is ambiguous: {names} (use <provider>/<model>)")


def resolve_models(name: str) -> list[ModelInfo]:
    """Catalog models for a CLI argument: one model (``provider/model`` or an unambiguous
    model id), else the default model(s) of a provider or spec (``silero``, ``kokoro``)."""
    try:
        return [get_model(name)]
    except ConfigurationError as exc:
        req = models_for(name)
        if not req.models:
            raise exc from None
        return req.models


def is_offline() -> bool:
    """``VAN_OFFLINE`` is set: nothing may be downloaded."""
    return _offline()


# ----------------------------------------------------------------------------- locations
def hf_cache_dir() -> Path:
    """The Hugging Face hub cache, resolved like ``huggingface_hub`` does (not created)."""
    explicit = os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if explicit:
        return Path(explicit).expanduser()
    home = os.environ.get("HF_HOME")
    if home:
        return Path(home).expanduser() / "hub"
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return base / "huggingface" / "hub"


def _hf_repo_dir(repo_id: str) -> Path:
    return hf_cache_dir() / f"models--{repo_id.replace('/', '--')}"


def _hf_snapshot(repo_id: str, revision: str) -> Path | None:
    repo = _hf_repo_dir(repo_id)
    snapshots = repo / "snapshots"
    if not snapshots.is_dir():
        return None
    commit = revision
    ref = repo / "refs" / revision
    if ref.is_file():
        commit = ref.read_text(encoding="utf-8").strip()
    candidate = snapshots / commit
    if candidate.is_dir():
        return candidate
    return None


def _hf_fallback_path(f: ModelFile) -> Path:
    return (
        cache_dir()
        / _HF_FALLBACK
        / Path(*f.location.split("/"))
        / f.revision
        / f.filename.split("/")[-1]
    )


def _cache_path(f: ModelFile) -> Path:
    root = cache_dir()
    return (root / f.subdir if f.subdir else root) / f.filename


def _archive_stem(filename: str) -> str:
    lower = filename.lower()
    for suffix in _ARCHIVE_SUFFIXES:
        if lower.endswith(suffix):
            return filename[: -len(suffix)]
    raise ValueError(f"unsupported archive type: {filename}")


def _read_marker(directory: Path) -> dict[str, Any] | None:
    try:
        data = json.loads((directory / _ARCHIVE_MARKER).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _tree_size(path: Path) -> int:
    """Bytes used by ``path`` (a file or a directory; symlinks count once, by target)."""
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    seen: set[Path] = set()
    total = 0
    for dirpath, _dirs, files in os.walk(path):
        for name in files:
            p = Path(dirpath) / name
            try:
                real = p.resolve()
                if real in seen:
                    continue
                seen.add(real)
                total += real.stat().st_size
            except OSError:
                continue
    return total


def _snapshot_files(snapshot: Path) -> list[Path]:
    return sorted(p for p in snapshot.rglob("*") if p.is_file())


def _repo_files_complete(f: ModelFile, snapshot: Path) -> bool:
    names = [p.relative_to(snapshot).as_posix() for p in _snapshot_files(snapshot)]
    return bool(names) and all(any(fnmatch.fnmatch(n, r) for n in names) for r in f.required)


# ----------------------------------------------------------------------------- status
@dataclass(frozen=True, slots=True)
class FileStatus:
    """Where a :class:`ModelFile` is (or would be) on disk."""

    file: ModelFile
    path: Path
    """Local path: the file, the extracted directory, or the HF snapshot directory."""
    present: bool
    size_on_disk: int = 0
    in_hf_cache: bool = False
    """The file was found in the Hugging Face cache (not the model cache)."""


@dataclass(frozen=True, slots=True)
class ModelStatus:
    """Cache state of a model: ``cached`` (every file), ``partial`` or ``missing``."""

    model: ModelInfo
    files: tuple[FileStatus, ...]

    @property
    def state(self) -> Literal["cached", "partial", "missing"]:
        present = [f.present for f in self.files]
        if all(present):
            return "cached"
        return "partial" if any(present) else "missing"

    @property
    def cached(self) -> bool:
        return self.state == "cached"

    @property
    def size_on_disk(self) -> int:
        return sum(f.size_on_disk for f in self.files)

    @property
    def hf_size_on_disk(self) -> int:
        return sum(f.size_on_disk for f in self.files if f.in_hf_cache)


def file_status(f: ModelFile) -> FileStatus:
    """Locate one model file in the model cache or the Hugging Face cache."""
    if f.source == "url":
        path = _cache_path(f)
        present = path.is_file()
        return FileStatus(f, path, present, path.stat().st_size if present else 0)
    if f.source == "archive":
        path = _cache_path(f)
        marker = _read_marker(path) if path.is_dir() else None
        present = marker is not None and (f.sha256 is None or marker.get("sha256") == f.sha256)
        return FileStatus(f, path, present, _tree_size(path) if path.is_dir() else 0)
    if f.source == "hf":
        snapshot = _hf_snapshot(f.location, f.revision)
        if snapshot is not None and (snapshot / f.filename).is_file():
            path = snapshot / f.filename
            return FileStatus(f, path, True, _tree_size(path), in_hf_cache=True)
        path = _hf_fallback_path(f)
        present = path.is_file()
        return FileStatus(f, path, present, path.stat().st_size if present else 0)
    snapshot = _hf_snapshot(f.location, f.revision)
    if snapshot is None:
        return FileStatus(f, _hf_repo_dir(f.location), False, in_hf_cache=True)
    present = _repo_files_complete(f, snapshot)
    size = sum(_tree_size(p) for p in _snapshot_files(snapshot))
    return FileStatus(f, snapshot, present, size, in_hf_cache=True)


def model_status(info: ModelInfo) -> ModelStatus:
    """Which files of ``info`` are cached, where, and how much disk they use."""
    return ModelStatus(info, tuple(file_status(f) for f in info.files))


# ----------------------------------------------------------------------------- download
def download_model(
    info: ModelInfo,
    *,
    force: bool = False,
    progress: ModelProgress | None = None,
    client: Any = None,
) -> list[Path]:
    """Fetch every file of ``info`` that is not cached yet (blocking); return local paths.

    Files already in the cache are checked against their digest (as the providers do)
    and not downloaded again unless ``force``. With ``VAN_OFFLINE`` set, a missing file
    raises :class:`~voice_agent_next.utils.download.DownloadError`.

    Args:
        progress: ``progress(file, done, total)`` for direct downloads (Hugging Face
            downloads report through ``huggingface_hub`` instead).
        client: optional ``httpx.Client`` for direct downloads (tests).
    """
    paths: list[Path] = []
    for f in info.files:

        def report(done: int, total: int | None, f: ModelFile = f) -> None:
            if progress is not None:
                progress(f, done, total)

        if f.source == "url":
            paths.append(
                download(
                    f.location,
                    filename=f.filename,
                    subdir=f.subdir,
                    sha256=f.sha256,
                    force=force,
                    client=client,
                    progress=report,
                )
            )
        elif f.source == "archive":
            paths.append(
                download_archive(
                    f.location,
                    sha256=f.sha256,
                    subdir=f.subdir,
                    name=f.filename,
                    force=force,
                    client=client,
                    progress=report,
                )
            )
        elif f.source == "hf":
            status = file_status(f)
            if status.present and not force and not status.in_hf_cache:
                paths.append(status.path)  # fallback copy, re-verified by hf_file on use
                continue
            paths.append(hf_file(f.location, f.filename, revision=f.revision, sha256=f.sha256))
        else:
            paths.append(_download_hf_repo(f, force=force))
    return paths


def _download_hf_repo(f: ModelFile, *, force: bool) -> Path:
    if not force:
        status = file_status(f)
        if status.present:
            return status.path
    if not is_installed("huggingface_hub"):
        raise MissingDependencyError(
            f"downloading {f.location} needs huggingface_hub: pip install huggingface_hub "
            "(it comes with the faster-whisper extra)"
        )
    from huggingface_hub import snapshot_download

    try:
        return Path(
            snapshot_download(
                f.location,
                revision=f.revision,
                allow_patterns=list(f.patterns) or None,
                local_files_only=_offline(),
                force_download=force,
            )
        )
    except Exception as exc:
        raise DownloadError(f"failed to fetch {f.location}@{f.revision}: {exc}") from exc


# ----------------------------------------------------------------------------- verify
@dataclass(frozen=True, slots=True)
class FileCheck:
    """Result of verifying one cached file: ``ok``, ``unverified`` (no digest to check
    against, but present), ``missing`` or ``corrupt``."""

    file: ModelFile
    path: Path
    status: Literal["ok", "unverified", "missing", "corrupt"]
    detail: str = ""

    @property
    def good(self) -> bool:
        return self.status in ("ok", "unverified")


def _check_digest(f: ModelFile, path: Path) -> FileCheck:
    if f.size is not None and path.stat().st_size != f.size and f.sha256 is None:
        return FileCheck(f, path, "corrupt", f"size {path.stat().st_size} != {f.size}")
    if f.sha256 is None:
        return FileCheck(f, path, "unverified", "no sha256 declared")
    digest = _sha256(path)
    if digest != f.sha256:
        return FileCheck(f, path, "corrupt", f"sha256 {digest[:12]}... != {f.sha256[:12]}...")
    return FileCheck(f, path, "ok", "sha256 matches")


def _verify_file(f: ModelFile) -> FileCheck:
    status = file_status(f)
    path = status.path
    if f.source in ("url", "hf"):
        if not status.present:
            return FileCheck(f, path, "missing")
        return _check_digest(f, path.resolve())
    if f.source == "archive":
        if not path.is_dir():
            return FileCheck(f, path, "missing")
        marker = _read_marker(path)
        if marker is None:
            return FileCheck(f, path, "corrupt", "incomplete extraction (no marker)")
        if f.sha256 is not None and marker.get("sha256") != f.sha256:
            return FileCheck(f, path, "corrupt", "extracted from another archive version")
        missing = [
            part
            for rel in f.required
            for part in rel.split(",")
            if part and not (path / part).exists()
        ]
        if missing:
            return FileCheck(f, path, "corrupt", f"missing {', '.join(missing)}")
        detail = "archive digest recorded at extraction; files present"
        return FileCheck(f, path, "ok" if f.sha256 else "unverified", detail)
    if not status.present:
        if path.name.startswith("models--") or not path.exists():
            return FileCheck(f, path, "missing")
        return FileCheck(f, path, "corrupt", f"incomplete snapshot (needs {', '.join(f.required)})")
    # hf-repo: LFS blobs are stored under their sha256 in the HF cache
    checked = 0
    for p in _snapshot_files(path):
        # snapshots/<commit>/<file> links to blobs/<sha256> for LFS files (the blob may
        # itself link elsewhere, e.g. into a deduplicating store: hash the final target)
        name = Path(os.readlink(p)).name if p.is_symlink() else p.name
        if _HEX64.match(name):
            checked += 1
            digest = _sha256(p.resolve())
            if digest != name:
                rel = p.relative_to(path).as_posix()
                return FileCheck(f, path, "corrupt", f"{rel}: sha256 does not match its blob")
    if checked:
        return FileCheck(f, path, "ok", f"{checked} large file(s) match their sha256")
    return FileCheck(f, path, "unverified", "no content-addressed files to check")


def verify_model(info: ModelInfo) -> list[FileCheck]:
    """Check every file of ``info`` against its sha256 (or structure, for archives)."""
    return [_verify_file(f) for f in info.files]


# ----------------------------------------------------------------------------- prune
@dataclass(frozen=True, slots=True)
class PruneItem:
    """A file or directory :func:`plan_prune` proposes to delete."""

    path: Path
    reason: Literal["partial", "unused", "model"]
    size: int
    model: str | None = None
    """Catalog name, for ``reason == "model"``."""
    in_hf_cache: bool = False


def _last_used(path: Path) -> float:
    """Latest access or modification time of ``path`` or anything inside it."""
    try:
        st = path.stat()
    except OSError:
        return 0.0
    latest = max(st.st_atime, st.st_mtime)
    if path.is_dir():
        for p in path.rglob("*"):
            try:
                s = p.stat()
            except OSError:
                continue
            latest = max(latest, s.st_atime, s.st_mtime)
    return latest


def _is_partial(path: Path) -> bool:
    name = path.name
    if path.is_file() and name.startswith(".") and name.endswith(".part"):
        return True  # download(): .<name>.<random>.part
    return path.is_dir() and name.startswith(".") and name.endswith(".tmp")  # extraction


def _owned_paths(models: Iterable[ModelInfo]) -> set[Path]:
    owned: set[Path] = set()
    for info in models:
        for f in info.files:
            if f.source in ("url", "archive"):
                owned.add(_cache_path(f))
            elif f.source == "hf":
                owned.add(_hf_fallback_path(f))
    return owned


def _scan_cache(
    models: Sequence[ModelInfo],
) -> Iterator[tuple[Path, Literal["partial", "unused"]]]:
    """Partial downloads anywhere in the cache, and unknown entries in provider folders.

    A *provider folder* is the first path component of a cache location some catalog
    model uses (``silero``, ``kokoro``, ``sherpa-onnx``, ``hf``). Everything else at the
    top level (for example Kokoro's ``espeak-ng-data-*`` copy) is left alone.
    """
    root = cache_dir()
    owned = _owned_paths(models)
    ancestors: set[Path] = set()
    managed: set[Path] = set()
    for p in owned:
        rel = p.relative_to(root)
        if len(rel.parts) > 1:
            managed.add(root / rel.parts[0])
        ancestors.update(a for a in p.parents if a != root and root in a.parents)
    managed.add(root / _HF_FALLBACK)
    archive_dirs = {p for p in owned if p.is_dir()}

    def walk(directory: Path, in_managed: bool) -> Iterator[tuple[Path, Any]]:
        try:
            entries = sorted(directory.iterdir())
        except OSError:
            return
        for entry in entries:
            if entry in owned:
                continue
            if _is_partial(entry):
                yield entry, "partial"
            elif entry.is_symlink():
                continue
            elif entry.is_file() and entry.name.endswith(_ARCHIVE_SUFFIXES) and in_managed:
                # an archive whose extraction was interrupted, or not deleted (Windows)
                stem = _archive_stem(entry.name)
                reason = "partial" if (directory / stem) in archive_dirs else "unused"
                yield entry, reason
            elif entry in ancestors or (entry.is_dir() and entry in managed):
                yield from walk(entry, True)
            elif in_managed:
                yield entry, "unused"
            elif entry.is_dir() and entry not in archive_dirs:
                yield from walk(entry, False)

    yield from walk(root, False)


def plan_prune(
    *,
    partial: bool = True,
    unused: bool = True,
    models: Iterable[str | ModelInfo] = (),
    all_models: bool = False,
    include_hf: bool = False,
    older_than: float | None = None,
    partial_min_age: float = 3600.0,
) -> list[PruneItem]:
    """What ``van models prune`` would delete (nothing is deleted here).

    Args:
        partial: interrupted downloads/extractions older than ``partial_min_age`` seconds
            (younger ones may belong to a download in progress).
        unused: files in provider folders that no catalog model uses (old versions,
            outdated pins, custom downloads).
        models: catalog models (names or :class:`ModelInfo`) to remove.
        all_models: remove every cached catalog model.
        include_hf: also delete Hugging Face cache entries of the selected models
            (otherwise they are reported but kept: the HF cache is shared).
        older_than: only entries not used (accessed or modified) for this many seconds.
    """
    now = time.time()
    everything = catalog()
    items: list[PruneItem] = []
    if partial or unused:
        for path, reason in _scan_cache(everything):
            if not (partial if reason == "partial" else unused):
                continue
            last = _last_used(path)
            if reason == "partial" and now - last < partial_min_age:
                continue
            if reason == "unused" and older_than is not None and now - last < older_than:
                continue
            items.append(PruneItem(path, reason, _tree_size(path)))
    selected = [x for m in models for x in (resolve_models(m) if isinstance(m, str) else [m])]
    if all_models:
        selected = everything
    seen: set[Path] = set()
    for info in selected:
        status = model_status(info)
        for fs in status.files:
            if fs.size_on_disk == 0:
                continue
            path = fs.path
            if fs.in_hf_cache:
                if not include_hf:
                    continue
                if fs.file.source == "hf-repo":
                    path = _hf_repo_dir(fs.file.location)
            if path in seen:
                continue  # shared between models (e.g. Kokoro voice packs)
            if older_than is not None and now - _last_used(path) < older_than:
                continue
            seen.add(path)
            items.append(
                PruneItem(path, "model", _tree_size(path), info.name, in_hf_cache=fs.in_hf_cache)
            )
    return items


def apply_prune(items: Iterable[PruneItem]) -> int:
    """Delete what :func:`plan_prune` returned; return the bytes freed."""
    freed = 0
    for item in items:
        path = item.path
        try:
            if item.in_hf_cache and path.is_symlink():
                blob = path.resolve()
                path.unlink()
                blob.unlink(missing_ok=True)
            elif path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink(missing_ok=True)
        except OSError:
            continue
        freed += item.size
    return freed


# ----------------------------------------------------------------------------- disk usage
@dataclass
class UsageReport:
    """Disk usage of the model cache (and the catalog's Hugging Face cache entries)."""

    cache_dir: Path
    hf_cache_dir: Path
    by_provider: dict[str, int] = field(default_factory=dict)
    """Bytes of cached catalog models in the model cache, per provider."""
    hf_by_provider: dict[str, int] = field(default_factory=dict)
    """Bytes of catalog models in the Hugging Face cache, per provider."""
    unused: int = 0
    partial: int = 0
    other: int = 0
    """Everything else in the model cache (e.g. Kokoro's espeak-ng data copy)."""
    total: int = 0
    """Size of the model cache directory."""


def disk_usage() -> UsageReport:
    """Bytes used per provider, plus unused/partial/other files in the model cache."""
    root = cache_dir()
    report = UsageReport(root, hf_cache_dir())
    counted: set[Path] = set()
    accounted = 0
    for info in catalog():
        for fs in model_status(info).files:
            if fs.size_on_disk == 0 or fs.path in counted:
                continue
            counted.add(fs.path)
            target = report.hf_by_provider if fs.in_hf_cache else report.by_provider
            target[info.provider] = target.get(info.provider, 0) + fs.size_on_disk
            if not fs.in_hf_cache:
                accounted += fs.size_on_disk
    for item in plan_prune(partial_min_age=0.0):
        if item.reason == "partial":
            report.partial += item.size
        else:
            report.unused += item.size
    report.total = _tree_size(root)
    report.other = max(0, report.total - accounted - report.partial - report.unused)
    return report


# ----------------------------------------------------------------------------- requirements
@dataclass
class Requirements:
    """The catalog models a configuration needs, and notes about components that
    download something the catalog does not know (custom paths, cloud providers are
    silently skipped)."""

    models: list[ModelInfo] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def add(self, info: ModelInfo) -> None:
        if info not in self.models:
            self.models.append(info)

    def extend(self, other: Requirements) -> None:
        for m in other.models:
            self.add(m)
        self.notes.extend(n for n in other.notes if n not in self.notes)


def models_for_spec(kind: str, spec: Any) -> Requirements:
    """Catalog models a component spec (string, mapping or failover list) will download."""
    from .errors import ProviderNotFoundError
    from .registry import get_provider, parse_spec

    req = Requirements()
    if spec is None:
        return req
    if isinstance(spec, (list, tuple)):
        for s in spec:
            req.extend(models_for_spec(kind, s))
        return req
    if isinstance(spec, Mapping):
        if "fallback" in spec:
            return models_for_spec(kind, spec["fallback"])
        target = spec.get("provider") or spec.get("use")
        if not target:
            raise ConfigurationError(f"{kind} config needs a 'provider' key: {dict(spec)!r}")
        name, model = parse_spec(str(target))
        model = spec.get("model") or model
    elif isinstance(spec, str):
        name, model = parse_spec(spec)
    else:
        return req  # a component instance
    if kind == "turn" and name == "fused":  # FusedTurnDetector: its two detectors
        parts = spec if isinstance(spec, Mapping) else {}
        req.extend(models_for_spec("turn", parts.get("audio", "smart_turn")))
        req.extend(models_for_spec("turn", parts.get("text", "lm_turn")))
        return req
    try:
        provider = get_provider(kind, name)  # type: ignore[arg-type]
    except ProviderNotFoundError as exc:
        req.notes.append(str(exc))
        return req
    _load_catalog()
    model = model or provider.default_model
    known = catalog(provider=provider.name, kind=kind, load=False)
    found = [m for m in known if model is not None and _matches(m, str(model))]
    if found:
        for m in found:
            req.add(m)
    elif known:
        label = f"{provider.name}/{model}" if model else provider.name
        req.notes.append(
            f"{kind} {label}: not in the model catalog (local path or custom URL?); "
            "it is fetched on first use"
        )
    elif provider.local and provider.name != "mock":
        req.notes.append(
            f"{kind} {provider.name}: no catalog models (weights managed by the provider "
            "or an external server)"
        )
    return req


def models_for_config(config: Any) -> Requirements:
    """Catalog models an :class:`~voice_agent_next.config.AppConfig` (or a config file
    path, or a config mapping) needs."""
    from .config import AppConfig, load_config

    cfg = config if isinstance(config, AppConfig) else load_config(config)
    req = Requirements()
    for kind, spec in (
        ("engine", cfg.engine),
        ("stt", cfg.stt),
        ("llm", cfg.llm),
        ("tts", cfg.tts),
        ("vad", cfg.vad),
        ("turn", cfg.turn_detector),
    ):
        req.extend(models_for_spec(kind, spec))
    return req


def models_for(target: str) -> Requirements:
    """Resolve a ``--for`` target: a config file, ``preset:NAME``, ``kind=spec[,...]``, or a spec.

    A bare spec (``kokoro/v1.0-int8``, ``sherpa-onnx``) selects matching catalog models of
    every kind; without a model id, the provider's default model of each kind.
    ``preset:local-cpu`` selects every model of a preset (failover members included), e.g.
    to bake them into a Docker image.
    """
    if target.startswith("preset:"):
        from .presets import get_preset

        return models_for_config(get_preset(target.removeprefix("preset:").strip()).app_config())
    path = Path(target).expanduser()
    if path.suffix.lower() in (".yaml", ".yml", ".toml", ".json") or path.is_file():
        return models_for_config(path)
    if "=" in target:
        req = Requirements()
        for part in target.split(","):
            kind, sep, spec = part.partition("=")
            kind = {"turn_detector": "turn"}.get(kind.strip(), kind.strip())
            if not sep or kind not in _KINDS:
                raise ConfigurationError(
                    f"bad component {part!r}: expected <kind>=<spec> with kind in {_KINDS}"
                )
            req.extend(models_for_spec(kind, spec.strip()))
        return req
    from .registry import parse_spec

    name, model = parse_spec(target)
    _load_catalog()
    req = Requirements()
    provider_models = catalog(provider=name, load=False)
    kinds = sorted({k for m in provider_models for k in m.kinds}, key=_KINDS.index)
    if model is None:
        for kind in kinds:
            req.extend(models_for_spec(kind, name))
    else:
        for m in find_models(target):
            req.add(m)
    if not req.models and not req.notes:
        req.notes.append(f"{target}: no catalog models match")
    return req
