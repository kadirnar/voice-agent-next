"""The recording's JSONL timeline is written off the event loop, in order (#141)."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from voice_agent_next.audio.frame import AudioFormat
from voice_agent_next.session import SessionRecorder
from voice_agent_next.utils import now


def fake_session() -> Any:
    fmt = AudioFormat(1000, 1)
    transport = SimpleNamespace(output_format=fmt, input_format=fmt)
    return SimpleNamespace(transport=transport, engine=SimpleNamespace(provider="m", model="m"))


class _SpyFile:
    def __init__(self, inner: Any, threads: set[int]) -> None:
        self._inner, self._threads = inner, threads

    def write(self, text: str) -> int:
        self._threads.add(threading.get_ident())
        return int(self._inner.write(text))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def lines_of(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]


def test_timeline_lines_are_written_off_the_calling_thread_in_order(tmp_path: Path) -> None:
    rec = SessionRecorder(tmp_path / "call.wav")
    session = fake_session()
    t0 = now()
    threads: set[int] = set()
    rec.session_started(session, t0)
    writer: Any = rec._timeline
    assert writer is not None
    writer._file = _SpyFile(writer._file, threads)
    for i in range(500):
        rec._log("session", "tick", {"i": i}, t=t0 + i / 1000)
    rec.session_closing(session, "done", t0 + 1.0)

    assert threads and threading.get_ident() not in threads
    lines = lines_of(tmp_path / "call.jsonl")
    assert lines[0]["event"] == "recording_started"
    assert [ln["data"]["i"] for ln in lines[1:-1]] == list(range(500))  # order kept
    assert lines[-1]["event"] == "session_closed"  # flushed and complete on close


def test_payloads_are_snapshotted_when_logged(tmp_path: Path) -> None:
    rec = SessionRecorder(tmp_path / "call.wav")
    session = fake_session()
    t0 = now()
    rec.session_started(session, t0)
    payload = {"text": "before"}
    rec._log("session", "item", payload, t=t0)
    payload["text"] = "after"  # mutated after logging: the line keeps what was logged
    rec.session_closing(session, "done", t0 + 0.5)
    lines = lines_of(tmp_path / "call.jsonl")
    assert [ln["data"] for ln in lines if ln["event"] == "item"] == [{"text": "before"}]
