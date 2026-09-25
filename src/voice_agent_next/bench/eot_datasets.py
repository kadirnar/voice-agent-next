"""End-of-turn evaluation data: LiveKit's eot-bench (pinned) and custom manifests.

**eot-bench** (`livekit/eot-bench-data <https://huggingface.co/datasets/livekit/eot-bench-data>`_,
CC BY 4.0): real human-to-agent user turns in 14 languages (≤ 400 per language), each
annotated with every silence span of at least 100 ms. The **last** span is the true end of
the turn (label ``eot``); every earlier span is a mid-turn pause the agent must listen
through (label ``hold``). Rows also carry word timings and the prior conversation.

The built-in datasets ``eot-bench-<lang>`` pin one Parquet file per language at a fixed
Hugging Face revision (:data:`EOT_BENCH_REVISION`) with its size and SHA-256
(96–166 MB each; ``en``: 162 MB). On first use the file is downloaded into the shared cache,
verified, and unpacked into one WAV per turn plus ``turns.jsonl`` (this step needs
``pyarrow``: ``pip install 'voice-agent-next[bench]'``); the Parquet file is then deleted, so
later runs are offline (``VAN_OFFLINE=1`` works) and need no ``pyarrow``.

**Custom data**: a JSON Lines manifest with one turn per line — ``audio`` (WAV path,
relative to the manifest), ``silence_spans`` (``[{start, end}, ...]`` in seconds, the last
one being the end of the turn), optional ``id``, ``language``, ``words``
(``[{start, end, word}]``) and ``messages`` (``[{role, content}]``).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from ..audio.frame import AudioFrame
from ..audio.wav import read_wav, write_wav
from ..utils.deps import require
from ..utils.download import DownloadError, cache_dir, download
from ..utils.env import is_offline

__all__ = [
    "EOT_BENCH_FILES",
    "EOT_BENCH_REPO",
    "EOT_BENCH_REVISION",
    "EotDataset",
    "EotSpan",
    "EotTurn",
    "builtin_eot_datasets",
    "load_eot_dataset",
    "load_eot_manifest",
    "turn_from_row",
]

EOT_BENCH_REPO = "livekit/eot-bench-data"
EOT_BENCH_REVISION = "ca9d98a9686b920a2d8c9eb984224ba9be74e4dd"
"""Pinned commit of :data:`EOT_BENCH_REPO` (2026-07-22)."""
EOT_BENCH_FILES: dict[str, tuple[int, str]] = {
    "ar": (145145023, "ab08f587b30b19fc8f6846572ba131fffb7d481534728edcd3f4654d634eb30c"),
    "de": (138163610, "e02dd1afa056b594810689eac364b632c712c18f1e3d7468fd19d07ec6da5f3d"),
    "en": (162406142, "e475d435c8e693912243e8a810bf7e34a59e01e208551ce5b362c9c778b0fdc4"),
    "es": (166059372, "54da6201ad3aa303d4fb2e8416ded3fbfcdaaf090df66e48ee432c4bb75b782f"),
    "fr": (137615422, "daeb94120bbaa7879d8921effb25b434c9f62c687e007ff2708507f131b66157"),
    "hi": (132810327, "d02181dce0b75d0b203d61d1c519c8c20324c91297039122acfddcc6c235ff1f"),
    "id": (139854090, "cdc792a99e58c9718c9080c9606b33a1522aa293a0a56a506616fef43c0b4bb3"),
    "it": (133680197, "d1bf75b11b7bd265f61db2c7bb28ca018fe3e9519af3467a1588131f8777edd4"),
    "ja": (138452115, "347cc9d555a495099c0d061ef200e18924b3c6d1c019205325eba6a7b418a8f5"),
    "ko": (114700081, "6526150783f8be2ecaaac5c487db56e806df05250c956f8006b095419ae74127"),
    "nl": (124181863, "86f78d7b7c2a915b87488d3e0c1851638aa3e606a94a7d7198bf098eb24db092"),
    "pt": (134064610, "ab0446412c366903cecb7f1d095cff7e7e02fe45a73be08136b4f8049472695d"),
    "tr": (96458483, "a09c8587a99fe222dc593ed809ebc1f8e415aa62f832a1fbe3202c04d4609944"),
    "zh": (129303772, "5fd7a30919721832642bea7201ea3d9b09fa6ebce13ae7902ec21ad2dbc9f0f8"),
}
"""Language -> (size in bytes, SHA-256) of ``data/<lang>/validation-00000-of-00001.parquet``."""

MIN_SILENCE_SPAN = 0.1
"""Silence spans shorter than this are not decision points (eot-bench's default)."""
_TURNS_FILE = "turns.jsonl"
_COMPLETE_MARKER = ".complete"

SpanLabel = Literal["hold", "eot"]


@dataclass(frozen=True, slots=True)
class EotSpan:
    """A silence span of a user turn: a decision point for the end-of-turn policy."""

    index: int
    start: float
    end: float
    label: SpanLabel

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class EotTurn:
    """One complete user turn with its silence spans (the last one is the end of turn)."""

    id: str
    audio: Path
    language: str | None
    spans: tuple[EotSpan, ...]
    words: tuple[tuple[float, float, str], ...] = ()
    """``(start, end, word)`` timings of the user's words."""
    messages: tuple[tuple[str, str], ...] = ()
    """``(role, content)`` of the conversation before this turn."""
    sha256: str | None = None
    duration: float | None = None

    def load_audio(self) -> AudioFrame:
        return read_wav(self.audio).to_mono()


@dataclass
class EotDataset:
    name: str
    turns: list[EotTurn]
    source: dict[str, Any] = field(default_factory=dict)
    builtin: bool = False

    @property
    def sha256(self) -> str:
        """Hash over ids, languages, spans, words, messages and audio hashes, in order."""
        payload = json.dumps(
            [
                [t.id, t.language, [[s.start, s.end, s.label] for s in t.spans],
                 [list(w) for w in t.words], [list(m) for m in t.messages], t.sha256]
                for t in self.turns
            ],
            ensure_ascii=False,
            separators=(",", ":"),
        )  # fmt: skip
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @property
    def id(self) -> str:
        return f"{self.name}@sha256:{self.sha256[:12]}"

    @property
    def languages(self) -> list[str]:
        return sorted({t.language or "und" for t in self.turns})

    def limit(self, n: int | None) -> EotDataset:
        if n is None or n >= len(self.turns):
            return self
        return EotDataset(self.name, self.turns[:n], dict(self.source), self.builtin)

    def describe(self) -> dict[str, Any]:
        spans = [s for t in self.turns for s in t.spans]
        return {
            "name": self.name,
            "id": self.id,
            "sha256": self.sha256,
            "builtin": self.builtin,
            "turns": len(self.turns),
            "spans": {"hold": sum(s.label == "hold" for s in spans),
                      "eot": sum(s.label == "eot" for s in spans)},
            "languages": self.languages,
            "source": self.source,
        }  # fmt: skip


# ----------------------------------------------------------------------- parsing


def _spans(raw: Iterable[Mapping[str, Any]], min_silence: float) -> tuple[EotSpan, ...]:
    """Every span >= ``min_silence``; the dataset's **last** span is the end of turn."""
    items = [(float(s["start"]), float(s["end"])) for s in raw]
    if not items:
        return ()
    last = len(items) - 1
    return tuple(
        EotSpan(k, start, end, "eot" if k == last else "hold")
        for k, (start, end) in enumerate(items)
        if end - start >= min_silence - 1e-6
    )


def turn_from_row(
    row: Mapping[str, Any],
    audio: Path,
    *,
    sha256: str | None = None,
    min_silence: float = MIN_SILENCE_SPAN,
) -> EotTurn:
    """An :class:`EotTurn` from an eot-bench-style row (``silence_spans``, ``words``...)."""
    if "silence_spans" not in row:
        raise ValueError(f"turn {row.get('id')!r} has no `silence_spans`")
    words = tuple(
        (float(w["start"]), float(w["end"]), str(w["word"])) for w in row.get("words") or []
    )
    messages = tuple((str(m["role"]), str(m["content"])) for m in row.get("messages") or [])
    duration = row.get("duration")
    language = row.get("language")
    return EotTurn(
        id=str(row.get("id") or audio.stem),
        audio=audio,
        language=str(language).strip().lower() if language else None,
        spans=_spans(row["silence_spans"], min_silence),
        words=words,
        messages=messages,
        sha256=sha256,
        duration=float(duration) if duration not in (None, "") else None,
    )


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_eot_manifest(
    path: str | os.PathLike[str],
    *,
    name: str | None = None,
    min_silence: float = MIN_SILENCE_SPAN,
    source: dict[str, Any] | None = None,
    builtin: bool = False,
) -> EotDataset:
    """Load a JSON Lines manifest (see the module docstring)."""
    manifest = Path(path)
    if not manifest.is_file():
        raise ValueError(f"end-of-turn manifest not found: {manifest}")
    turns: list[EotTurn] = []
    for n, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as exc:
            raise ValueError(f"{manifest}:{n}: not a JSON object: {exc}") from exc
        audio = row.get("audio") or row.get("audio_filepath") or row.get("wav")
        if not audio:
            raise ValueError(f"{manifest}:{n}: needs an `audio` path")
        audio_path = Path(str(audio)).expanduser()
        if not audio_path.is_absolute():
            audio_path = manifest.parent / audio_path
        if not audio_path.is_file():
            raise ValueError(f"{manifest}:{n}: audio file not found: {audio_path}")
        sha = row.get("sha256") or _file_sha256(audio_path)
        turn = turn_from_row(row, audio_path, sha256=sha, min_silence=min_silence)
        if not any(s.label == "eot" for s in turn.spans):
            raise ValueError(
                f"{manifest}:{n}: the final silence span is shorter than {min_silence}s"
            )
        turns.append(turn)
    if not turns:
        raise ValueError(f"{manifest}: no turns")
    return EotDataset(
        name or manifest.stem,
        turns,
        source or {"manifest": str(manifest), "license": "user data"},
        builtin,
    )


# ------------------------------------------------------------------- eot-bench


def builtin_eot_datasets() -> list[str]:
    """Names of the built-in datasets: ``eot-bench-<lang>``."""
    return [f"eot-bench-{lang}" for lang in EOT_BENCH_FILES]


def _parquet_url(lang: str) -> str:
    return (
        f"https://huggingface.co/datasets/{EOT_BENCH_REPO}/resolve/{EOT_BENCH_REVISION}"
        f"/data/{lang}/validation-00000-of-00001.parquet"
    )


def _unpack_parquet(parquet: Path, directory: Path) -> int:
    """Write one WAV per row plus ``turns.jsonl``; returns the number of rows."""
    pq = require("pyarrow.parquet", extra="bench")
    table = pq.read_table(parquet)
    audio_dir = directory / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    rows = table.to_pylist()
    lines = []
    for row in rows:
        data = row["audio"]["bytes"]
        # normalize to 16-bit PCM WAV (the source is WAV; re-encode for a stable format)
        frame = read_wav(data).to_mono()
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in str(row["id"]))
        target = audio_dir / f"{safe}.wav"
        write_wav(target, frame)
        lines.append(
            json.dumps(
                {
                    "id": row["id"],
                    "audio": f"audio/{target.name}",
                    "sha256": _file_sha256(target),
                    "language": row.get("language"),
                    "duration": row.get("duration"),
                    "silence_spans": row.get("silence_spans") or [],
                    "words": row.get("words") or [],
                    "messages": row.get("messages") or [],
                },
                ensure_ascii=False,
            )
        )
    (directory / _TURNS_FILE).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(rows)


def _load_eot_bench(
    lang: str, *, min_silence: float, progress: Callable[[str], None] | None
) -> EotDataset:
    size, sha256 = EOT_BENCH_FILES[lang]
    name = f"eot-bench-{lang}"
    directory = cache_dir() / "datasets" / name
    marker = directory / _COMPLETE_MARKER
    source = {
        "repo": EOT_BENCH_REPO,
        "revision": EOT_BENCH_REVISION,
        "file": f"data/{lang}/validation-00000-of-00001.parquet",
        "sha256": sha256,
        "bytes": size,
        "license": "CC BY 4.0",
        "attribution": "LiveKit eot-bench (https://github.com/livekit/eot-bench)",
    }
    if not marker.exists() or marker.read_text(encoding="utf-8").strip() != sha256:
        if is_offline() and not (directory / "validation-00000-of-00001.parquet").exists():
            raise DownloadError(
                f"dataset {name} is not cached and offline mode is on (VAN_OFFLINE/HF_HUB_OFFLINE)"
            )
        if progress is not None:
            progress(
                f"fetching {name} ({size / 1e6:.0f} MB, {EOT_BENCH_REPO}@{EOT_BENCH_REVISION[:8]})"
            )
        parquet = download(
            _parquet_url(lang), subdir=f"datasets/{name}", sha256=sha256, timeout=300.0
        )
        if (directory / "audio").exists():
            shutil.rmtree(directory / "audio")
        count = _unpack_parquet(parquet, directory)
        marker.write_text(sha256 + "\n", encoding="utf-8")
        parquet.unlink(missing_ok=True)
        if progress is not None:
            progress(f"unpacked {count} turns into {directory}")
    return load_eot_manifest(
        directory / _TURNS_FILE, name=name, min_silence=min_silence, source=source, builtin=True
    )


def load_eot_dataset(
    spec: str,
    *,
    min_silence: float = MIN_SILENCE_SPAN,
    progress: Callable[[str], None] | None = None,
) -> EotDataset:
    """``eot-bench-<lang>`` (downloaded once) or a JSON Lines manifest path."""
    if spec.startswith("eot-bench-") and spec.removeprefix("eot-bench-") in EOT_BENCH_FILES:
        return _load_eot_bench(
            spec.removeprefix("eot-bench-"), min_silence=min_silence, progress=progress
        )
    path = Path(spec)
    if path.suffix.lower() == ".jsonl" or path.exists():
        return load_eot_manifest(path, min_silence=min_silence)
    raise ValueError(
        f"unknown end-of-turn dataset {spec!r}: use eot-bench-<lang> "
        f"({', '.join(EOT_BENCH_FILES)}) or a .jsonl manifest"
    )
