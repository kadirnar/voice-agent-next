"""Spoken-question datasets for the T5 speech-to-speech quality track.

**Built-in smoke subsets** (``van bench quality --dataset <name>``, see
:func:`builtin_quality_datasets`), pinned in ``bench/data/quality_smoke.json``:

* ``big-bench-audio-smoke`` — 48 questions of Artificial Analysis' Big Bench Audio (MIT):
  12 per category (``formal_fallacies``, ``navigate``, ``object_counting``,
  ``web_of_lies``), drawn with a fixed seed and ordered round-robin by category, so
  ``--limit 20`` keeps 5 per category (the research smoke tier). Closed answers:
  valid/invalid, yes/no, a number. One MP3 per question, fetched from a pinned commit.
* ``voicebench-<subset>-smoke`` — 20 items of a VoiceBench subset (Apache-2.0), drawn
  with a fixed seed from the first Parquet row group of the pinned revision:
  ``openbookqa`` (multiple choice A–D), ``sd-qa-usa`` (short factual answers with a
  reference), ``commoneval`` (open questions: judge only) and ``advbench`` (harmful
  requests: the agent should refuse).

Every audio file is verified against its SHA-256 before it is written to the shared cache
(``<cache>/datasets/<name>/``, ``$VAN_CACHE_DIR``); later runs are offline
(``VAN_OFFLINE=1`` works). Big Bench Audio files are downloaded one by one; for VoiceBench
only the audio column of one row group of one Parquet file is read, with HTTP range
requests (~15 MB per subset, needs ``pyarrow``: extra ``bench``).
``benchmarks/tools/make_quality_smoke_subsets.py`` regenerates the catalog.

**User data**: a JSONL/JSON/TSV/CSV manifest (:func:`load_quality_manifest`) with one
spoken question per row: ``audio`` (or ``audio_filepath``/``path``/``file``/``wav``), and
optionally ``id``, ``answer`` (or ``reference``), ``prompt`` (or ``question``/``text``: the
question as text, for the judge), ``category``, ``scoring`` (see
:mod:`voice_agent_next.bench.quality_scoring`) and ``choices`` (``{"A": "...", ...}``).
"""

from __future__ import annotations

import hashlib
import io
import json
import os
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

import httpx

from ..utils.deps import require
from ..utils.download import DownloadError, cache_dir
from .asr_datasets import _AUDIO_KEYS, _file_sha256, _manifest_rows, _pick, _write_atomic
from .quality_scoring import SCORINGS, infer_scoring, parse_choices

__all__ = [
    "HttpRangeFile",
    "QualityDataset",
    "QualityItem",
    "builtin_quality_datasets",
    "load_builtin_quality_dataset",
    "load_quality_dataset",
    "load_quality_manifest",
    "read_parquet_row_group",
]

_CATALOG_FILE = Path(__file__).parent / "data" / "quality_smoke.json"
_ANSWER_KEYS = ("answer", "reference", "official_answer", "target")
_PROMPT_KEYS = ("prompt", "question", "text", "transcript")


@dataclass(frozen=True)
class QualityItem:
    """One spoken question and how its answer is scored."""

    id: str
    audio: Path
    scoring: str
    """``choice``, ``yes_no``, ``valid_invalid``, ``number``, ``contains``, ``exact``,
    ``refusal`` or ``open`` (judge only)."""
    answer: str | None = None
    """Reference answer (closed questions)."""
    prompt: str | None = None
    """The question as text (shown to the judge; ``None`` when only audio exists)."""
    category: str | None = None
    choices: Mapping[str, str] | None = None
    """Multiple choice options (letter -> text), parsed from the prompt when omitted."""
    sha256: str | None = None
    duration: float | None = None

    def describe(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category,
            "scoring": self.scoring,
            "answer": self.answer,
            "sha256": self.sha256,
            "duration_s": self.duration,
        }


@dataclass
class QualityDataset:
    """An ordered list of spoken questions plus provenance."""

    name: str
    items: list[QualityItem]
    source: dict[str, Any] = field(default_factory=dict)
    builtin: bool = False

    @property
    def sha256(self) -> str:
        """Hash of ids, scoring, answers, prompts, categories and audio hashes, in order."""
        payload = json.dumps(
            [[i.id, i.scoring, i.answer, i.prompt, i.category, i.sha256] for i in self.items],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @property
    def id(self) -> str:
        return f"{self.name}@sha256:{self.sha256[:12]}"

    def limit(self, n: int | None) -> QualityDataset:
        if n is None or n >= len(self.items):
            return self
        return QualityDataset(self.name, self.items[:n], dict(self.source), self.builtin)

    def describe(self) -> dict[str, Any]:
        """Manifest entry: name, hash, provenance and one entry (with SHA-256) per file."""
        return {
            "name": self.name,
            "id": self.id,
            "sha256": self.sha256,
            "builtin": self.builtin,
            "n": len(self.items),
            "source": self.source,
            "items": [i.describe() for i in self.items],
        }


# ------------------------------------------------------------ HTTP range reads


class HttpRangeFile(io.RawIOBase):
    """A seekable, read-only file over HTTP range requests (for Parquet footers and row
    groups: only the bytes that are read are downloaded)."""

    def __init__(self, url: str, *, client: httpx.Client | None = None) -> None:
        self._own = client is None
        self._http = client or httpx.Client(follow_redirects=True, timeout=120.0)
        try:
            resp = self._http.get(url, headers={"Range": "bytes=0-0"})
        except httpx.HTTPError as exc:
            raise DownloadError(f"GET {url} failed: {exc}") from exc
        if resp.status_code != 206 or "content-range" not in resp.headers:
            raise DownloadError(f"{url}: range requests not supported (HTTP {resp.status_code})")
        self.url = str(resp.url)  # the redirect target (CDN), reused for every range
        self.size = int(resp.headers["content-range"].rsplit("/", 1)[1])
        self._pos = 0
        self.bytes_read = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self._pos, io.SEEK_END: self.size}[whence]
        self._pos = max(0, base + offset)
        return self._pos

    def readinto(self, b: Any) -> int:
        n = min(len(b), self.size - self._pos)
        if n <= 0:
            return 0
        rng = f"bytes={self._pos}-{self._pos + n - 1}"
        try:
            resp = self._http.get(self.url, headers={"Range": rng})
        except httpx.HTTPError as exc:
            raise DownloadError(f"range request to {self.url} failed: {exc}") from exc
        if resp.status_code != 206:
            raise DownloadError(f"range request to {self.url}: HTTP {resp.status_code}")
        data = resp.content[:n]
        b[: len(data)] = data
        self._pos += len(data)
        self.bytes_read += len(data)
        return len(data)

    def close(self) -> None:
        if self._own and not self.closed:
            self._http.close()
        super().close()


def read_parquet_row_group(
    url: str,
    row_group: int,
    columns: Iterable[str],
    *,
    client: httpx.Client | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Rows of one row group of a remote Parquet file (``columns`` only) and the number of
    bytes downloaded."""
    pq = require("pyarrow.parquet", extra="bench")
    with HttpRangeFile(url, client=client) as f:
        # pre_buffer=False: read only the column chunks asked for
        pf = pq.ParquetFile(f, pre_buffer=False)
        table = pf.read_row_group(row_group, columns=list(columns))
        return table.to_pylist(), f.bytes_read


# --------------------------------------------------------------- built-in subsets


@cache
def _catalog() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(_CATALOG_FILE.read_text(encoding="utf-8"))
    return data


def builtin_quality_datasets() -> dict[str, dict[str, Any]]:
    """Name -> definition of every built-in smoke subset."""
    datasets: dict[str, dict[str, Any]] = _catalog()["datasets"]
    return datasets


def _offline() -> bool:
    return os.environ.get("VAN_OFFLINE", "").lower() in ("1", "true", "yes")


def _fetch_files(
    spec: Mapping[str, Any],
    missing: list[dict[str, Any]],
    directory: Path,
    client: httpx.Client | None,
) -> int:
    base = spec["source"]["base_url"].rstrip("/")
    own = client is None
    http = client or httpx.Client(follow_redirects=True, timeout=120.0)
    total = 0
    try:
        for item in missing:
            url = f"{base}/{item['path']}"
            try:
                resp = http.get(url)
            except httpx.HTTPError as exc:
                raise DownloadError(f"GET {url} failed: {exc}") from exc
            if resp.status_code != 200:
                raise DownloadError(f"GET {url} failed with HTTP {resp.status_code}")
            digest = hashlib.sha256(resp.content).hexdigest()
            if digest != item["sha256"]:
                raise DownloadError(f"checksum mismatch for {url}: {digest} != {item['sha256']}")
            _write_atomic(directory / item["file"], resp.content)
            total += len(resp.content)
    finally:
        if own:
            http.close()
    return total


def _fetch_parquet(
    spec: Mapping[str, Any],
    missing: list[dict[str, Any]],
    directory: Path,
    client: httpx.Client | None,
) -> int:
    source = spec["source"]
    rows, nbytes = read_parquet_row_group(
        source["url"], int(source["row_group"]), [source.get("audio_column", "audio")],
        client=client,
    )  # fmt: skip
    column = source.get("audio_column", "audio")
    for item in missing:
        cell = rows[int(item["row"])][column]
        data = cell["bytes"] if isinstance(cell, Mapping) else cell
        digest = hashlib.sha256(data).hexdigest()
        if digest != item["sha256"]:
            raise DownloadError(
                f"checksum mismatch for row {item['row']} of {source['url']}: "
                f"{digest} != {item['sha256']}"
            )
        _write_atomic(directory / item["file"], data)
    return nbytes


def load_builtin_quality_dataset(
    name: str,
    *,
    client: httpx.Client | None = None,
    progress: Callable[[str], None] | None = None,
) -> QualityDataset:
    """Fetch (first use) and return a built-in smoke subset."""
    catalog = builtin_quality_datasets()
    if name not in catalog:
        raise ValueError(f"unknown dataset {name!r}; built-in: {', '.join(sorted(catalog))}")
    spec = catalog[name]
    directory = cache_dir() / "datasets" / name
    missing = [
        item
        for item in spec["items"]
        if not (directory / item["file"]).exists()
        or _file_sha256(directory / item["file"]) != item["sha256"]
    ]
    if missing:
        if _offline():
            raise DownloadError(f"dataset {name} is not cached and VAN_OFFLINE is set")
        kind = spec["source"]["type"]
        if progress is not None:
            progress(f"fetching {len(missing)} file(s) of {name} ({kind})")
        fetch = _fetch_parquet if kind == "parquet" else _fetch_files
        nbytes = fetch(spec, missing, directory, client)
        if progress is not None:
            progress(f"downloaded {nbytes / 1e6:.1f} MB for {name}")
    items = [
        QualityItem(
            id=item["id"],
            audio=directory / item["file"],
            scoring=item["scoring"],
            answer=item.get("answer"),
            prompt=item.get("prompt"),
            category=item.get("category"),
            choices=parse_choices(item.get("prompt")) if item["scoring"] == "choice" else None,
            sha256=item["sha256"],
            duration=item.get("duration"),
        )
        for item in spec["items"]
    ]
    source = {k: v for k, v in spec.items() if k != "items"}
    return QualityDataset(name, items, source, builtin=True)


# -------------------------------------------------------------------- manifests


def _parse_choices_field(value: Any) -> dict[str, str] | None:
    if value in (None, ""):
        return None
    if isinstance(value, str):
        value = json.loads(value)
    if isinstance(value, Mapping):
        return {str(k).strip().upper(): str(v) for k, v in value.items()}
    if isinstance(value, list):
        return {chr(ord("A") + i): str(v) for i, v in enumerate(value)}
    raise ValueError(f"choices must be a mapping or a list, got {type(value).__name__}")


def load_quality_manifest(
    path: str | os.PathLike[str], *, name: str | None = None
) -> QualityDataset:
    """Load spoken questions from a JSONL/JSON/TSV/CSV manifest (see the module docstring)."""
    manifest = Path(path)
    if not manifest.is_file():
        raise ValueError(f"dataset manifest not found: {manifest}")
    rows = _manifest_rows(manifest)
    if not rows:
        raise ValueError(f"{manifest}: no items")
    items: list[QualityItem] = []
    seen: set[str] = set()
    for n, row in enumerate(rows, 1):
        audio = _pick(row, _AUDIO_KEYS)
        if audio is None:
            raise ValueError(f"{manifest}: item {n} needs an audio path ({'/'.join(_AUDIO_KEYS)})")
        audio_path = Path(str(audio)).expanduser()
        if not audio_path.is_absolute():
            audio_path = manifest.parent / audio_path
        if not audio_path.is_file():
            raise ValueError(f"{manifest}: item {n}: audio file not found: {audio_path}")
        uid = str(row.get("id") or audio_path.stem)
        if uid in seen:
            uid = f"{uid}#{n}"
        seen.add(uid)
        answer = _pick(row, _ANSWER_KEYS)
        prompt = _pick(row, _PROMPT_KEYS)
        try:
            choices = _parse_choices_field(row.get("choices"))
        except ValueError as exc:
            raise ValueError(f"{manifest}: item {n}: {exc}") from exc
        if choices is None and prompt is not None:
            choices = parse_choices(str(prompt))
        scoring = str(row.get("scoring") or "") or infer_scoring(
            None if answer is None else str(answer), choices
        )
        if scoring not in SCORINGS:
            raise ValueError(
                f"{manifest}: item {n}: unknown scoring {scoring!r} ({', '.join(SCORINGS)})"
            )
        if scoring not in ("open", "refusal") and answer is None:
            raise ValueError(f"{manifest}: item {n}: scoring {scoring!r} needs an `answer`")
        duration = row.get("duration")
        items.append(
            QualityItem(
                id=uid,
                audio=audio_path,
                scoring=scoring,
                answer=None if answer is None else str(answer),
                prompt=None if prompt is None else str(prompt),
                category=str(row["category"]) if row.get("category") else None,
                choices=choices if scoring == "choice" else None,
                sha256=_file_sha256(audio_path),
                duration=float(duration) if duration not in (None, "") else None,
            )
        )
    return QualityDataset(
        name or manifest.stem, items, {"manifest": str(manifest), "license": "user data"}
    )


def load_quality_dataset(
    spec: str,
    *,
    client: httpx.Client | None = None,
    progress: Callable[[str], None] | None = None,
) -> QualityDataset:
    """A built-in subset by name, or a user manifest by path."""
    if spec in builtin_quality_datasets():
        return load_builtin_quality_dataset(spec, client=client, progress=progress)
    path = Path(spec)
    if path.suffix.lower() in (".jsonl", ".json", ".tsv", ".csv") or path.exists():
        return load_quality_manifest(path)
    names = ", ".join(sorted(builtin_quality_datasets()))
    raise ValueError(
        f"unknown dataset {spec!r}: use a built-in subset ({names}) "
        "or a manifest file (.jsonl/.json/.tsv/.csv)"
    )
