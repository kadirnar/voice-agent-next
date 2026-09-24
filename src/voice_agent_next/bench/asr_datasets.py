"""ASR evaluation data: pinned smoke subsets of public corpora and custom manifests.

**Built-in smoke subsets** (``van bench asr --dataset <name>``, see :func:`builtin_datasets`):

* ``librispeech-test-clean-smoke`` — 50 utterances of LibriSpeech test-clean (read English
  audiobooks, CC BY 4.0; 10 speakers × 5 utterances of 1.5–20 s);
* ``fleurs-{en,es,de,tr,zh}-smoke`` — 10 utterances per language of the FLEURS test split
  (read Wikipedia sentences, CC BY 4.0; distinct sentences of 2–20 s).

The subsets are defined in ``bench/data/asr_smoke.json``: archive URL (the FLEURS URLs pin
a Hugging Face commit), and for every utterance the archive member, its SHA-256, duration
and reference transcript. The source archives are large (LibriSpeech test-clean: 347 MB,
FLEURS test: 290–580 MB per language), so they are **streamed** and only the listed members
are kept: the members were chosen among the first files of each archive, so the download
stops after ~35 MB (LibriSpeech) or a few MB (FLEURS). Every member is verified against its
SHA-256 before it is written into the shared cache
(``<cache>/datasets/<name>/``; ``$VAN_CACHE_DIR``), after which runs need no network
(``VAN_OFFLINE=1`` works). ``benchmarks/tools/make_asr_smoke_subsets.py`` regenerates the
definitions.

**Custom data**: a manifest file of local audio + reference text (:func:`load_manifest`):

* JSON Lines (``.jsonl``; NeMo manifests work as they are) or a JSON list — one object per
  utterance with ``audio`` (also ``audio_filepath``, ``path``, ``file`` or ``wav``) and
  ``text`` (or ``transcript``/``sentence``), optional ``id``, ``language``, ``duration``;
* TSV/CSV with a header row naming the same columns.

Relative audio paths are resolved against the manifest's directory. WAV is read natively;
FLAC/OGG/MP3... need ``soundfile`` (extra ``bench``).
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import tarfile
import tempfile
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import IO, Any

import httpx
import numpy as np

from ..audio.frame import AudioFrame
from ..audio.wav import read_wav
from ..utils.deps import require
from ..utils.download import DownloadError, cache_dir
from ..utils.log import logger

__all__ = [
    "AsrDataset",
    "AsrUtterance",
    "builtin_datasets",
    "fetch_archive_members",
    "iter_tar_stream",
    "load_asr_dataset",
    "load_audio",
    "load_builtin_dataset",
    "load_manifest",
]

_CATALOG_FILE = Path(__file__).parent / "data" / "asr_smoke.json"
_AUDIO_KEYS = ("audio", "audio_filepath", "path", "file", "wav", "audio_path")
_TEXT_KEYS = ("text", "transcript", "transcription", "sentence", "reference")


@dataclass(frozen=True)
class AsrUtterance:
    """One evaluation item: an audio file and its reference transcript."""

    id: str
    audio: Path
    text: str
    language: str | None = None
    duration: float | None = None
    """Seconds (from the dataset definition; measured when the audio is loaded)."""
    sha256: str | None = None
    """SHA-256 of the audio file."""


@dataclass
class AsrDataset:
    """An ordered list of utterances plus provenance."""

    name: str
    utterances: list[AsrUtterance]
    language: str | None = None
    source: dict[str, Any] = field(default_factory=dict)
    """Where the data comes from: archive URL/revision, license, attribution, manifest path."""
    builtin: bool = False

    @property
    def sha256(self) -> str:
        """Hash of the definition: ids, references, languages and audio hashes, in order."""
        payload = json.dumps(
            [[u.id, u.text, u.language, u.sha256] for u in self.utterances],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @property
    def id(self) -> str:
        """``<name>@sha256:<first 12 hex digits>``."""
        return f"{self.name}@sha256:{self.sha256[:12]}"

    @property
    def total_duration(self) -> float | None:
        durations = [u.duration for u in self.utterances]
        if not durations or any(d is None for d in durations):
            return None
        return float(sum(d for d in durations if d is not None))

    def limit(self, n: int | None) -> AsrDataset:
        """The first ``n`` utterances (same name, new hash)."""
        if n is None or n >= len(self.utterances):
            return self
        return AsrDataset(
            self.name, self.utterances[:n], self.language, dict(self.source), self.builtin
        )

    def describe(self) -> dict[str, Any]:
        """Manifest entry: name, hash, provenance and one entry (with SHA-256) per file."""
        return {
            "name": self.name,
            "id": self.id,
            "sha256": self.sha256,
            "language": self.language,
            "builtin": self.builtin,
            "n": len(self.utterances),
            "audio_s": None if self.total_duration is None else round(self.total_duration, 3),
            "source": self.source,
            "items": [
                {"id": u.id, "sha256": u.sha256, "duration_s": u.duration, "language": u.language}
                for u in self.utterances
            ],
        }


# ------------------------------------------------------------------------ audio


def load_audio(path: str | os.PathLike[str]) -> AudioFrame:
    """Read an audio file as mono s16le at its native sample rate."""
    p = Path(path)
    if p.suffix.lower() == ".wav":
        try:
            return read_wav(p).to_mono()
        except Exception:  # e.g. float WAV: fall back to soundfile
            pass
    sf = require("soundfile", extra="bench")
    data, rate = sf.read(str(p), dtype="int16", always_2d=True)
    mono = data.mean(axis=1).round().astype(np.int16) if data.shape[1] > 1 else data[:, 0]
    return AudioFrame.from_numpy(np.ascontiguousarray(mono), int(rate))


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ------------------------------------------------------------ streamed archives


class _IterStream(io.RawIOBase):
    """A read-only file object over an iterator of byte chunks (counts bytes read)."""

    def __init__(self, chunks: Iterator[bytes]) -> None:
        self._chunks = chunks
        self._buf = b""
        self.bytes_read = 0

    def readable(self) -> bool:
        return True

    def readinto(self, b: Any) -> int:
        while not self._buf:
            try:
                self._buf = next(self._chunks)
            except StopIteration:
                return 0
        k = min(len(b), len(self._buf))
        b[:k] = self._buf[:k]
        self._buf = self._buf[k:]
        self.bytes_read += k
        return k


def iter_tar_stream(
    url: str,
    *,
    client: httpx.Client | None = None,
    timeout: float = 60.0,
    stats: dict[str, int] | None = None,
) -> Iterator[tuple[tarfile.TarInfo, IO[bytes] | None]]:
    """Stream a (compressed) tar archive over HTTP and yield ``(member, file)`` pairs.

    ``file`` is ``None`` for non-regular members and must be read before advancing.
    Stop iterating to stop the download (the connection is closed). ``stats["bytes"]``
    receives the number of (compressed) bytes downloaded.
    """
    own = client is None
    http = client or httpx.Client(follow_redirects=True, timeout=timeout)
    try:
        with http.stream("GET", url) as resp:
            if resp.status_code != 200:
                raise DownloadError(f"GET {url} failed with HTTP {resp.status_code}")
            raw = _IterStream(resp.iter_raw(1 << 16))
            with tarfile.open(fileobj=io.BufferedReader(raw, 1 << 16), mode="r|*") as tar:
                for member in tar:
                    if stats is not None:
                        stats["bytes"] = raw.bytes_read
                    yield member, tar.extractfile(member) if member.isfile() else None
            if stats is not None:
                stats["bytes"] = raw.bytes_read
    except httpx.HTTPError as exc:
        raise DownloadError(f"download of {url} failed: {exc}") from exc
    except tarfile.TarError as exc:
        raise DownloadError(f"cannot read the archive {url}: {exc}") from exc
    finally:
        if own:
            http.close()


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".part")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def fetch_archive_members(
    urls: Iterable[str],
    members: Mapping[str, tuple[str, str]],
    directory: Path,
    *,
    client: httpx.Client | None = None,
    progress: Callable[[str], None] | None = None,
) -> int:
    """Extract ``members`` (archive name -> (local file name, sha256)) of a streamed archive.

    The download stops as soon as every member was found. URLs are tried in order (mirrors).
    Returns the number of bytes downloaded. Raises :class:`DownloadError` if a member is
    missing or fails verification.
    """
    errors: list[str] = []
    for url in urls:
        wanted = dict(members)
        stats: dict[str, int] = {"bytes": 0}
        try:
            for member, fileobj in iter_tar_stream(url, client=client, stats=stats):
                spec = wanted.get(member.name)
                if spec is None or fileobj is None:
                    continue
                name, sha256 = spec
                data = fileobj.read()
                digest = hashlib.sha256(data).hexdigest()
                if digest != sha256:
                    raise DownloadError(
                        f"checksum mismatch for {member.name} in {url}: {digest} != {sha256}"
                    )
                _write_atomic(directory / name, data)
                del wanted[member.name]
                if not wanted:
                    break
        except DownloadError as exc:
            errors.append(str(exc))
            logger.warning("%s", exc)
            continue
        if wanted:
            errors.append(f"{url}: {len(wanted)} member(s) not found, e.g. {next(iter(wanted))}")
            continue
        if progress is not None:
            progress(f"downloaded {stats['bytes'] / 1e6:.1f} MB from {url}")
        return stats["bytes"]
    raise DownloadError("could not fetch the dataset: " + "; ".join(errors))


# --------------------------------------------------------------- built-in subsets


@cache
def _catalog() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(_CATALOG_FILE.read_text(encoding="utf-8"))
    return data


def builtin_datasets() -> dict[str, dict[str, Any]]:
    """Name -> definition of every built-in smoke subset."""
    datasets: dict[str, dict[str, Any]] = _catalog()["datasets"]
    return datasets


def _offline() -> bool:
    return os.environ.get("VAN_OFFLINE", "").lower() in ("1", "true", "yes")


def load_builtin_dataset(
    name: str,
    *,
    client: httpx.Client | None = None,
    progress: Callable[[str], None] | None = None,
) -> AsrDataset:
    """Fetch (first use) and return a built-in smoke subset."""
    catalog = builtin_datasets()
    if name not in catalog:
        raise ValueError(f"unknown dataset {name!r}; built-in: {', '.join(sorted(catalog))}")
    spec = catalog[name]
    directory = cache_dir() / "datasets" / name
    items: list[dict[str, Any]] = spec["items"]
    missing: dict[str, tuple[str, str]] = {}
    for item in items:
        path = directory / item["file"]
        if not path.exists() or _file_sha256(path) != item["sha256"]:
            missing[item["member"]] = (item["file"], item["sha256"])
    if missing:
        if _offline():
            raise DownloadError(f"dataset {name} is not cached and VAN_OFFLINE is set")
        archive = spec["archive"]
        if progress is not None:
            progress(
                f"fetching {len(missing)} file(s) of {name} (streaming {archive['urls'][0]})"
            )
        fetch_archive_members(
            archive["urls"], missing, directory, client=client, progress=progress
        )
    source = {k: v for k, v in spec.items() if k not in ("items",)}
    utterances = [
        AsrUtterance(
            id=item["id"],
            audio=directory / item["file"],
            text=item["text"],
            language=spec.get("language"),
            duration=item.get("duration"),
            sha256=item["sha256"],
        )
        for item in items
    ]
    return AsrDataset(name, utterances, spec.get("language"), source, builtin=True)


# -------------------------------------------------------------------- manifests


def _pick(row: Mapping[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return None


def _manifest_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8-sig")
    if suffix in (".tsv", ".csv"):
        delimiter = "\t" if suffix == ".tsv" else ","
        return [dict(r) for r in csv.DictReader(io.StringIO(text), delimiter=delimiter)]
    if suffix == ".json":
        data = json.loads(text)
        if isinstance(data, dict):
            data = data.get("items") or data.get("utterances") or []
        if not isinstance(data, list):
            raise ValueError(f"{path}: expected a JSON list of utterances")
        return [dict(r) for r in data]
    rows = []
    for n, line in enumerate(text.splitlines(), 1):
        if line.strip():
            try:
                rows.append(dict(json.loads(line)))
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path}:{n}: not a JSON object: {exc}") from exc
    return rows


def load_manifest(
    path: str | os.PathLike[str], *, language: str | None = None, name: str | None = None
) -> AsrDataset:
    """Load a custom dataset from a JSONL/JSON/TSV/CSV manifest (see the module docstring)."""
    manifest = Path(path)
    if not manifest.is_file():
        raise ValueError(f"dataset manifest not found: {manifest}")
    rows = _manifest_rows(manifest)
    if not rows:
        raise ValueError(f"{manifest}: no utterances")
    utterances: list[AsrUtterance] = []
    seen: set[str] = set()
    for n, row in enumerate(rows, 1):
        audio = _pick(row, _AUDIO_KEYS)
        text = _pick(row, _TEXT_KEYS)
        if audio is None or text is None:
            raise ValueError(
                f"{manifest}: item {n} needs an audio path ({'/'.join(_AUDIO_KEYS)}) and a "
                f"reference ({'/'.join(_TEXT_KEYS)})"
            )
        audio_path = Path(str(audio)).expanduser()
        if not audio_path.is_absolute():
            audio_path = manifest.parent / audio_path
        if not audio_path.is_file():
            raise ValueError(f"{manifest}: item {n}: audio file not found: {audio_path}")
        uid = str(row.get("id") or row.get("utt_id") or audio_path.stem)
        if uid in seen:
            uid = f"{uid}#{n}"
        seen.add(uid)
        duration = row.get("duration")
        utterances.append(
            AsrUtterance(
                id=uid,
                audio=audio_path,
                text=str(text),
                language=str(row.get("language") or language or "") or None,
                duration=float(duration) if duration not in (None, "") else None,
                sha256=_file_sha256(audio_path),
            )
        )
    languages = {u.language for u in utterances}
    return AsrDataset(
        name or manifest.stem,
        utterances,
        languages.pop() if len(languages) == 1 else None,
        {"manifest": str(manifest), "license": "user data"},
    )


def load_asr_dataset(
    spec: str,
    *,
    language: str | None = None,
    client: httpx.Client | None = None,
    progress: Callable[[str], None] | None = None,
) -> AsrDataset:
    """A built-in subset by name, or a custom manifest by path."""
    if spec in builtin_datasets():
        dataset = load_builtin_dataset(spec, client=client, progress=progress)
        if language:
            dataset.utterances = [
                AsrUtterance(u.id, u.audio, u.text, language, u.duration, u.sha256)
                for u in dataset.utterances
            ]
            dataset.language = language
        return dataset
    path = Path(spec)
    if path.suffix.lower() in (".jsonl", ".json", ".tsv", ".csv") or path.exists():
        return load_manifest(path, language=language)
    raise ValueError(
        f"unknown dataset {spec!r}: use a built-in subset ({', '.join(sorted(builtin_datasets()))}) "
        "or a manifest file (.jsonl/.json/.tsv/.csv)"
    )
