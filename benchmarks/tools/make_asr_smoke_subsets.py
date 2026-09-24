"""Regenerate ``src/voice_agent_next/bench/data/asr_smoke.json`` (T2 ASR smoke subsets).

Streams the source archives, applies the selection rules below to the files in archive
order and records every selected member with its SHA-256, duration and reference text.
Only the start of each archive is downloaded (~35 MB for LibriSpeech, a few MB per FLEURS
language). Needs ``soundfile`` (extra ``bench``) to read FLAC durations::

    uv run --extra bench python benchmarks/tools/make_asr_smoke_subsets.py

Selection rules (deterministic given the archive):

* LibriSpeech test-clean: the first 50 utterances of 1.5–20 s in archive order, at most 5
  per speaker. References from the chapter ``*.trans.txt`` (read to the end of the archive
  here; runs stop after the last selected file).
* FLEURS test (per language): the first 10 files of 2–20 s whose sentence was not selected
  yet (one recording per sentence); non-English sentences containing digits are skipped
  (the basic normalizer does not normalize numbers) and so are Chinese sentences
  containing Latin letters. References: ``raw_transcription`` of ``test.tsv``.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import httpx

from voice_agent_next.bench.asr_datasets import iter_tar_stream, load_audio

OUT = Path(__file__).resolve().parents[2] / "src/voice_agent_next/bench/data/asr_smoke.json"

LIBRISPEECH_URLS = [
    "https://www.openslr.org/resources/12/test-clean.tar.gz",
    "https://us.openslr.org/resources/12/test-clean.tar.gz",
    "https://openslr.elda.org/resources/12/test-clean.tar.gz",
]
FLEURS_REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
FLEURS_BASE = f"https://huggingface.co/datasets/google/fleurs/resolve/{FLEURS_REVISION}/data"
FLEURS_LANGUAGES = {
    "en": "en_us",
    "es": "es_419",
    "de": "de_de",
    "tr": "tr_tr",
    "zh": "cmn_hans_cn",
}
FLEURS_ATTRIBUTION = (
    "FLEURS (Conneau et al., 2022, arXiv:2205.12446), google/fleurs on the Hugging Face Hub"
)


def _duration(data: bytes, suffix: str, tmp: Path) -> float:
    path = tmp.with_suffix(suffix)
    path.write_bytes(data)
    try:
        return round(load_audio(path).duration, 3)
    finally:
        path.unlink()


def librispeech(tmp: Path) -> dict[str, Any]:
    per_speaker: dict[str, int] = defaultdict(int)
    speakers: list[str] = []
    selected: list[dict[str, Any]] = []
    texts: dict[str, str] = {}
    stats: dict[str, int] = {}
    streamed = 0
    pending_chapters: set[str] = set()
    for member, f in iter_tar_stream(LIBRISPEECH_URLS[0], stats=stats):
        if f is None:
            continue
        name = member.name
        if name.endswith(".trans.txt"):
            for line in f.read().decode("utf-8").splitlines():
                uid, _, text = line.partition(" ")
                texts[uid] = text.strip()
            pending_chapters.discard(Path(name).name.removesuffix(".trans.txt"))
            if len(selected) == 50 and not pending_chapters:
                break
            continue
        if not name.endswith(".flac") or len(selected) == 50:
            continue
        uid = Path(name).stem
        speaker, chapter = uid.split("-")[:2]
        if per_speaker[speaker] >= 5:
            continue
        data = f.read()
        duration = _duration(data, ".flac", tmp)
        if not 1.5 <= duration <= 20.0:
            continue
        if speaker not in speakers:
            speakers.append(speaker)
        per_speaker[speaker] += 1
        pending_chapters.add(f"{speaker}-{chapter}")
        if len(selected) == 49:
            streamed = stats["bytes"]
        selected.append(
            {
                "id": uid,
                "member": name,
                "file": f"{uid}.flac",
                "sha256": hashlib.sha256(data).hexdigest(),
                "bytes": len(data),
                "duration": duration,
            }
        )
    for item in selected:
        item["text"] = texts[item["id"]]
    print(
        f"librispeech: {len(selected)} items from {len(speakers)} speakers, {streamed / 1e6:.1f} MB streamed"
    )
    return {
        "description": f"LibriSpeech test-clean: 50 utterances (1.5-20 s) of {len(speakers)} speakers",
        "language": "en",
        "license": "CC-BY-4.0",
        "attribution": "LibriSpeech ASR corpus (Panayotov et al., ICASSP 2015), https://www.openslr.org/12",
        "archive": {
            "urls": LIBRISPEECH_URLS,
            "format": "tar.gz",
            "streamed_mb": round(streamed / 1e6, 1),
        },
        "selection": "first 50 utterances of 1.5-20 s in archive order, at most 5 per speaker",
        "items": selected,
    }


def fleurs(lang: str, code: str, tmp: Path) -> dict[str, Any]:
    tsv = httpx.get(f"{FLEURS_BASE}/{code}/test.tsv", follow_redirects=True, timeout=60)
    tsv.raise_for_status()
    rows = {}
    reader = csv.reader(io.StringIO(tsv.text), delimiter="\t", quoting=csv.QUOTE_NONE)
    for row in reader:
        rows[row[1]] = {"sentence": row[0], "raw": row[2], "samples": int(row[5]), "gender": row[6]}
    selected: list[dict[str, Any]] = []
    sentences: set[str] = set()
    stats: dict[str, int] = {}
    url = f"{FLEURS_BASE}/{code}/audio/test.tar.gz"
    for member, f in iter_tar_stream(url, stats=stats):
        if f is None:
            continue
        base = Path(member.name).name
        info = rows.get(base)
        if info is None or info["sentence"] in sentences:
            continue
        raw = info["raw"]
        if lang != "en" and any(c.isdigit() for c in raw):
            continue
        if lang == "zh" and any("a" <= c.lower() <= "z" for c in raw):
            continue
        data = f.read()
        duration = _duration(data, ".wav", tmp)
        if not 2.0 <= duration <= 20.0:
            continue
        sentences.add(info["sentence"])
        selected.append(
            {
                "id": f"{code}-{info['sentence']}-{Path(base).stem}",
                "member": member.name,
                "file": base,
                "sha256": hashlib.sha256(data).hexdigest(),
                "bytes": len(data),
                "duration": duration,
                "gender": info["gender"],
                "text": raw,
            }
        )
        if len(selected) == 10:
            break
    print(f"fleurs-{lang}: {len(selected)} items, {stats['bytes'] / 1e6:.1f} MB streamed")
    return {
        "description": f"FLEURS {code} test: 10 distinct sentences (2-20 s)",
        "language": lang,
        "license": "CC-BY-4.0",
        "attribution": FLEURS_ATTRIBUTION,
        "archive": {
            "urls": [url],
            "revision": FLEURS_REVISION,
            "format": "tar.gz",
            "streamed_mb": round(stats["bytes"] / 1e6, 1),
        },
        "selection": "first 10 files of 2-20 s with distinct sentences, in archive order"
        + ("; sentences with digits skipped" if lang != "en" else "")
        + ("; sentences with Latin letters skipped" if lang == "zh" else ""),
        "items": selected,
    }


def main() -> None:
    tmp = OUT.parent / ".subset-tmp"
    datasets: dict[str, Any] = {}
    only = set(sys.argv[1:])
    if not only or "librispeech" in only:
        datasets["librispeech-test-clean-smoke"] = librispeech(tmp)
    for lang, code in FLEURS_LANGUAGES.items():
        if not only or lang in only:
            datasets[f"fleurs-{lang}-smoke"] = fleurs(lang, code, tmp)
    if only and OUT.exists():
        old = json.loads(OUT.read_text(encoding="utf-8"))["datasets"]
        datasets = {**old, **datasets}
    catalog = {"schema": 1, "datasets": dict(sorted(datasets.items()))}
    OUT.write_text(json.dumps(catalog, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
