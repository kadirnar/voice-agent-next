"""Regenerate ``src/voice_agent_next/bench/data/quality_smoke.json`` (T5 quality smoke subsets).

Downloads the selected audio once (~25 MB of Big Bench Audio MP3s, one Parquet row group
of ~15 MB per VoiceBench subset) to record every file's SHA-256, size and duration. Needs
``soundfile`` and ``pyarrow`` (extra ``bench``)::

    uv run --extra bench python benchmarks/tools/make_quality_smoke_subsets.py

Selection rules (deterministic given the pinned revisions):

* Big Bench Audio: ``random.Random(0).sample`` of 12 ids per category (categories in
  alphabetical order, ids in ascending order before sampling), then ordered round-robin
  by category (formal_fallacies, navigate, object_counting, web_of_lies, formal_fallacies,
  ...), so the first 4k items hold k per category.
* VoiceBench: ``random.Random(0).sample`` of 20 rows of the first row group of the first
  Parquet file of the subset, in row order.
"""

from __future__ import annotations

import hashlib
import io
import json
import random
import sys
from pathlib import Path
from typing import Any

import httpx
import soundfile as sf

from voice_agent_next.bench.quality_datasets import read_parquet_row_group

OUT = Path(__file__).resolve().parents[2] / "src/voice_agent_next/bench/data/quality_smoke.json"

BBA_REPO = "ArtificialAnalysis/big_bench_audio"
BBA_REVISION = "af7bb9c25b015792583ca4da3ee27ec62cb79fe6"
BBA_BASE = f"https://huggingface.co/datasets/{BBA_REPO}/resolve/{BBA_REVISION}"
BBA_PER_CATEGORY = 12
BBA_SCORING = {
    "formal_fallacies": "valid_invalid",
    "navigate": "yes_no",
    "object_counting": "number",
    "web_of_lies": "yes_no",
}

VB_REPO = "hlt-lab/voicebench"
VB_REVISION = "b02edcef1330480be3a11bd6f7434ac32f05ad08"
VB_BASE = f"https://huggingface.co/datasets/{VB_REPO}/resolve/{VB_REVISION}"
VB_ITEMS = 20
VB_SUBSETS = {
    # name: (Parquet file, scoring, description)
    "openbookqa": (
        "openbookqa/test-00000-of-00001.parquet",
        "choice",
        "OpenBookQA science questions read aloud with four options (A-D)",
    ),
    "sd-qa-usa": (
        "sd-qa/usa-00000-of-00001.parquet",
        "contains",
        "SD-QA factual questions spoken by US-English speakers, short reference answers",
    ),
    "commoneval": (
        "commoneval/test-00000-of-00001.parquet",
        "open",
        "open questions from Common Voice speakers (judge only)",
    ),
    "advbench": (
        "advbench/test-00000-of-00001.parquet",
        "refusal",
        "AdvBench harmful requests read aloud: the agent should refuse",
    ),
}


def _duration(data: bytes) -> float:
    info = sf.info(io.BytesIO(data))
    return round(float(info.frames) / float(info.samplerate), 3)


def big_bench_audio(client: httpx.Client) -> dict[str, Any]:
    resp = client.get(f"{BBA_BASE}/metadata.jsonl")
    resp.raise_for_status()
    rows = [json.loads(line) for line in resp.text.splitlines() if line.strip()]
    by_category: dict[str, list[dict[str, Any]]] = {}
    for row in sorted(rows, key=lambda r: int(r["id"])):
        by_category.setdefault(row["category"], []).append(row)
    rng = random.Random(0)
    picked = {
        cat: sorted(rng.sample(by_category[cat], BBA_PER_CATEGORY), key=lambda r: int(r["id"]))
        for cat in sorted(by_category)
    }
    ordered = [picked[cat][k] for k in range(BBA_PER_CATEGORY) for cat in sorted(picked)]
    items = []
    total = 0
    for row in ordered:
        path = row["file_name"]
        resp = client.get(f"{BBA_BASE}/{path}")
        resp.raise_for_status()
        data = resp.content
        total += len(data)
        items.append(
            {
                "id": f"bba-{int(row['id']):04d}",
                "path": path,
                "file": Path(path).name,
                "sha256": hashlib.sha256(data).hexdigest(),
                "bytes": len(data),
                "duration": _duration(data),
                "category": row["category"],
                "scoring": BBA_SCORING[row["category"]],
                "answer": str(row["official_answer"]),
            }
        )
        print(f"  {items[-1]['id']} {row['category']:<17} {items[-1]['duration']:6.1f} s",
              file=sys.stderr)  # fmt: skip
    return {
        "description": (
            f"Big Bench Audio: {BBA_PER_CATEGORY} spoken reasoning questions per category "
            "(4 BIG-Bench Hard categories), seeded draw, round-robin by category"
        ),
        "language": "en",
        "license": "MIT",
        "attribution": (
            "Big Bench Audio (Artificial Analysis, 2024), ArtificialAnalysis/big_bench_audio "
            "on the Hugging Face Hub; questions from BIG-Bench Hard (Suzgun et al., 2022)"
        ),
        "source": {
            "type": "files",
            "repo": BBA_REPO,
            "revision": BBA_REVISION,
            "base_url": BBA_BASE,
            "download_mb": round(total / 1e6, 1),
        },
        "selection": (
            f"random.Random(0).sample of {BBA_PER_CATEGORY} ids per category (categories "
            "sorted, ids ascending), ordered round-robin by category"
        ),
        "items": items,
    }


def voicebench(client: httpx.Client, subset: str) -> dict[str, Any]:
    file, scoring, description = VB_SUBSETS[subset]
    url = f"{VB_BASE}/{file}"
    rows, nbytes = read_parquet_row_group(url, 0, ["audio", "prompt", "reference"]
                                          if scoring in ("choice", "contains")
                                          else ["audio", "prompt"], client=client)  # fmt: skip
    rng = random.Random(0)
    picked = sorted(rng.sample(range(len(rows)), VB_ITEMS))
    items = []
    for r in picked:
        row = rows[r]
        audio = row["audio"]
        data: bytes = audio["bytes"]
        suffix = Path(audio.get("path") or "x.wav").suffix or ".wav"
        item: dict[str, Any] = {
            "id": f"{subset}-{r:04d}",
            "row": r,
            "file": f"{subset}-{r:04d}{suffix}",
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
            "duration": _duration(data),
            "scoring": scoring,
            "prompt": row["prompt"],
        }
        if row.get("reference") is not None:
            item["answer"] = str(row["reference"])
        items.append(item)
    return {
        "description": f"VoiceBench {subset}: {VB_ITEMS} items, {description}",
        "language": "en",
        "license": "Apache-2.0",
        "attribution": (
            "VoiceBench (Chen et al., 2024, arXiv:2410.17196), hlt-lab/voicebench on the "
            "Hugging Face Hub"
        ),
        "source": {
            "type": "parquet",
            "repo": VB_REPO,
            "revision": VB_REVISION,
            "url": url,
            "row_group": 0,
            "audio_column": "audio",
            "download_mb": round(nbytes / 1e6, 1),
        },
        "selection": f"random.Random(0).sample of {VB_ITEMS} rows of row group 0, in row order",
        "items": items,
    }


def main() -> None:
    datasets: dict[str, Any] = {}
    with httpx.Client(follow_redirects=True, timeout=120.0) as client:
        print("big-bench-audio-smoke", file=sys.stderr)
        datasets["big-bench-audio-smoke"] = big_bench_audio(client)
        for subset in VB_SUBSETS:
            print(f"voicebench-{subset}-smoke", file=sys.stderr)
            datasets[f"voicebench-{subset}-smoke"] = voicebench(client, subset)
    OUT.write_text(
        json.dumps({"schema": 1, "datasets": datasets}, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
