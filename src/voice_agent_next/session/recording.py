"""Call recording: a stereo WAV (user left, agent right) and a JSONL event timeline.

:class:`SessionRecorder` records what the call *sounded like*: the user's audio as the
transport delivered it (before echo cancellation or other processors) and the agent's
audio at the time it was played — paused while playback was paused and cut where a
barge-in stopped it. Both channels share one clock (t = 0 when the transport opened), the
same clock as the ``t`` of every line of the timeline, so a transcript, an interruption or
a latency metric can be found in the waveform at a glance.

Both files are streamed to disk: only the last second or so of audio is held in memory,
the WAV header is kept valid after every write, timeline lines are encoded and written by
a background thread (in order, never on the event loop), and everything is flushed and
finalized when the session closes. It works with every transport and engine because it only uses
what the session itself sees.
"""

from __future__ import annotations

import dataclasses
import enum
import json
import math
import os
import queue
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ..audio.frame import AudioFrame
from ..audio.resample import Resampler, resample
from ..audio.timeline import TimelineTrack
from ..audio.wav import WavWriter
from ..chat import AudioContent, ChatMessage
from ..events import ResponseAudio
from ..utils.clock import now
from ..utils.ids import new_id
from ..utils.log import logger
from .taps import SessionTap

if TYPE_CHECKING:
    from ..events import EngineEvent
    from .session import AgentSession

__all__ = ["SESSION_EVENTS", "SessionRecorder", "to_jsonable"]

SESSION_EVENTS = (
    "agent_state_changed",
    "user_state_changed",
    "user_transcript",
    "agent_transcript",
    "conversation_item",
    "tool_call",
    "tool_result",
    "tool_filler",
    "tool_progress",
    "tool_cancelled",
    "interrupted",
    "agent_false_interruption",
    "agent_handoff",
    "metrics",
    "error",
)
"""Session events written to the timeline (``close`` becomes ``session_closed``)."""

_JITTER = 0.2
"""User audio arriving up to this much later than its predecessor ends is contiguous."""

_BACKLOG = 0.05
"""User audio placed this much later than it arrived was queued behind a stall (see
:meth:`SessionRecorder.user_audio`)."""


@dataclasses.dataclass(slots=True)
class _Gap:
    """Silence the recorder inserted into the user stream (the latest run of it)."""

    pos: int  # where it starts on the recording (output samples)
    n_out: int  # its length on the recording
    n_in: int  # its length in input samples
    in_end: int  # input samples of the stream up to its end


class _TimelineWriter:
    """Appends JSON lines to a file from a background thread, in the order they were
    submitted: encoding and file I/O stay off the event loop. Payloads must already be
    snapshots (:func:`to_jsonable` output); :meth:`close` drains, flushes and closes."""

    def __init__(self, path: Path) -> None:
        self._file = open(path, "w", encoding="utf-8")  # noqa: SIM115 - closed by the thread
        self._queue: queue.SimpleQueue[dict[str, Any] | None] = queue.SimpleQueue()
        self._failed = False
        self._thread = threading.Thread(
            target=self._run, name=f"van-timeline-{path.name}", daemon=True
        )
        self._thread.start()

    @property
    def closed(self) -> bool:
        """Everything was written and the file is closed (like ``IO.closed``)."""
        return bool(self._file.closed)

    def write(self, line: dict[str, Any]) -> None:
        self._queue.put(line)

    def close(self, timeout: float = 5.0) -> None:
        self._queue.put(None)
        self._thread.join(timeout)
        if self._thread.is_alive():
            logger.warning("the session timeline writer did not finish within %.0f s", timeout)

    def _run(self) -> None:
        try:
            while True:
                line = self._queue.get()
                while line is not None:
                    self._write(line)
                    try:
                        line = self._queue.get_nowait()
                    except queue.Empty:
                        break
                else:
                    return
                self._flush()  # the backlog is written: readers see complete lines
        finally:
            try:
                self._file.close()
            except Exception:
                logger.exception("closing the session timeline failed")

    def _write(self, line: dict[str, Any]) -> None:
        if self._failed:
            return
        try:
            self._file.write(json.dumps(line, ensure_ascii=False, default=str) + "\n")
        except Exception:
            logger.exception("writing the session timeline failed; timeline stopped")
            self._failed = True

    def _flush(self) -> None:
        if self._failed:
            return
        try:
            self._file.flush()
        except Exception:
            logger.exception("flushing the session timeline failed; timeline stopped")
            self._failed = True


class SessionRecorder(SessionTap):
    """Records an :class:`~voice_agent_next.session.AgentSession` to disk.

    Usually created by ``AgentSession(record="recordings/")``.

    Args:
        path: a directory (each session gets ``<YYYYmmdd-HHMMSS>-<id>.wav`` and ``.jsonl``)
            or the ``.wav`` file to write (the timeline goes next to it, as ``.jsonl``).
        audio_events: also log every agent audio chunk the engine produced
            (``response_audio``: ids and duration, not the samples). Off by default: it adds
            ~25 lines per second of speech.
        flush_delay: audio older than this many seconds is written out. It must exceed
            how late user audio can arrive and the output look-ahead; memory holds about
            this much audio.

    Attributes:
        wav_path: the stereo WAV (set once the session started).
        timeline_path: the JSONL timeline (set once the session started).
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        audio_events: bool = False,
        flush_delay: float = 1.0,
    ) -> None:
        if flush_delay <= _JITTER:
            raise ValueError(f"flush_delay must be > {_JITTER} s")
        self.path = Path(path)
        self.audio_events = audio_events
        self.flush_delay = flush_delay
        self.wav_path: Path | None = None
        self.timeline_path: Path | None = None
        self.sample_rate = 0
        self._session: AgentSession | None = None
        self._origin = 0.0
        self._wav: WavWriter | None = None
        self._timeline: _TimelineWriter | None = None
        self._closed = False
        self._wav_failed = False
        self._written = 0  # stereo samples written to the WAV
        # agent channel: already at the recording rate, placed by playback time
        self._agent = TimelineTrack(1)
        self._paused_at: int | None = None  # sample where playback is paused
        # user channel: one continuous stream (frames + silence for gaps), resampled
        self._user = TimelineTrack(1)
        self._user_rs: Resampler | None = None
        self._user_rate = 0
        self._user_skip = 0  # resampler delay still to drop
        self._user_delay = 0  # the resampler's delay (output samples)
        self._user_base = 0.0  # time (s) at which the current input stream starts
        self._user_in = 0  # input samples of the current stream so far
        self._gap: _Gap | None = None  # the latest silence inserted into the user stream

    @property
    def origin(self) -> float:
        """:func:`~voice_agent_next.utils.now` time of t = 0 (the transport opened)."""
        return self._origin

    # ------------------------------------------------------------------ setup
    def attach(self, session: AgentSession) -> SessionRecorder:
        """Record ``session`` (call before it starts)."""
        if self._session is not None:
            raise RuntimeError("a SessionRecorder records one session")
        self._session = session
        session.add_tap(self)
        for name in SESSION_EVENTS:
            session.on(name, self._make_handler(name))
        return self

    def _make_handler(self, name: str) -> Any:
        if name == "metrics":  # wall-clock timestamp: log at receipt; keep the metric type

            def on_metrics(payload: Any) -> None:
                self._log("session", name, payload, t=now(), drop=("timestamp",))

            return on_metrics

        def handler(payload: Any) -> None:
            self._log("session", name, payload)

        return handler

    def _paths(self) -> tuple[Path, Path]:
        if self.path.suffix.lower() == ".wav":
            return self.path, self.path.with_suffix(".jsonl")
        stem = f"{datetime.now():%Y%m%d-%H%M%S}-{new_id('')[:8]}"
        return self.path / f"{stem}.wav", self.path / f"{stem}.jsonl"

    # ------------------------------------------------------------------- hooks
    def session_started(self, session: AgentSession, t: float) -> None:
        transport = session.transport
        self._origin = t
        self.sample_rate = rate = transport.output_format.sample_rate
        self.wav_path, self.timeline_path = self._paths()
        self.wav_path.parent.mkdir(parents=True, exist_ok=True)
        self.timeline_path.parent.mkdir(parents=True, exist_ok=True)
        self._wav = WavWriter(self.wav_path, rate, 2)
        self._timeline = _TimelineWriter(self.timeline_path)
        self._agent = TimelineTrack(rate)
        self._user = TimelineTrack(rate)
        engine = session.engine
        self._write_line(
            {
                "t": 0.0,
                "source": "recording",
                "event": "recording_started",
                "data": {
                    "wall_time": time.time(),
                    "date": datetime.now(UTC).isoformat(),
                    "wav": self.wav_path.name,
                    "sample_rate": rate,
                    "channels": {"left": "user", "right": "agent"},
                    "engine": {"provider": engine.provider, "model": engine.model},
                    "transport": type(transport).__name__,
                    "input_format": str(transport.input_format),
                    "output_format": str(transport.output_format),
                },
            }
        )

    def session_closing(self, session: AgentSession, reason: str, t: float) -> None:
        if self._closed or self._wav is None:
            return
        self._closed = True
        end = self._pos(t)
        try:
            self._agent.truncate(end)  # scheduled but never played
            self._paused_at = None
            self._advance_user(t)
            if self._user_rs is not None:
                self._put_user(self._user_rs.flush())
            self._user.truncate(end)
            self._flush(end)
            self._write_line(
                {
                    "t": round(t - self._origin, 6),
                    "source": "recording",
                    "event": "session_closed",
                    "data": {"reason": reason, "duration": self._written / self.sample_rate},
                }
            )
        except Exception:
            logger.exception("finalizing the session recording failed")
        finally:
            try:
                self._wav.close()
            except Exception:
                logger.exception("closing the session recording failed")
            if self._timeline is not None:
                self._timeline.close()

    def user_audio(self, frame: AudioFrame, t: float) -> None:
        if self._closed or self._wav is None:
            return
        frame = frame.to_mono()
        if frame.sample_rate != self._user_rate:
            self._restart_user_stream(frame.sample_rate)
        rate = self._user_rate
        # the frame was captured from its timestamp (when the transport stamped it: that
        # holds even when the input loop falls behind), or at the latest during
        # [t - duration, t]; keep the stream contiguous unless it starts clearly later
        # than the previous one ended (a gap in the input)
        cursor = self._user_base + self._user_in / rate
        captured = t - frame.duration if frame.timestamp is None else min(frame.timestamp, t)
        start = captured - self._origin
        if start > cursor + _JITTER:
            self._push_silence(start - cursor)
        elif cursor - start > _BACKLOG:
            # it would end after it arrived: the "gap" before it was a backlog (a stalled
            # event loop or input loop delivers the queued real-time frames late, then all
            # at once), not a pause of the input: take that silence out again, or the rest
            # of the user channel would stay late by the length of the stall
            self._take_back_silence(cursor - start)
        self._push_user(frame.to_numpy())
        self._maybe_flush()

    def agent_audio(self, frame: AudioFrame, start: float) -> None:
        if self._closed or self._wav is None:
            return
        frame = frame.to_mono()
        if frame.sample_rate != self.sample_rate:  # not expected: the session resamples
            frame = resample(frame, self.sample_rate)
        pos = self._pos(start)
        if abs(pos - self._agent.end) <= 1:  # rounding: keep consecutive chunks seamless
            pos = self._agent.end
        self._agent.write(pos, frame.to_numpy())
        self._maybe_flush()

    def playback_paused(self, t: float) -> None:
        self._paused_at = self._pos(t)

    def playback_shifted(self, paused_at: float, delta: float) -> None:
        self._agent.insert_silence(self._pos(paused_at), round(delta * self.sample_rate))

    def playback_resumed(self) -> None:
        self._paused_at = None

    def playback_cleared(self, t: float) -> None:
        self._agent.truncate(self._pos(t))
        self._log("recording", "playback_cleared", None, t=t)

    def engine_event(self, event: EngineEvent) -> None:
        if isinstance(event, ResponseAudio) and not self.audio_events:
            return
        self._log("engine", event.type, event)

    # ----------------------------------------------------------------- timeline
    def _log(
        self,
        source: str,
        name: str,
        payload: Any,
        *,
        t: float | None = None,
        drop: tuple[str, ...] = ("timestamp", "type"),
    ) -> None:
        if self._timeline is None or self._closed:
            return
        if t is None:  # session and engine events carry their time on the now() clock
            ts = getattr(payload, "timestamp", None)
            t = ts if isinstance(ts, float) else now()
        line: dict[str, Any] = {"t": round(t - self._origin, 6), "source": source, "event": name}
        if payload is not None:
            line["data"] = to_jsonable(payload, drop=drop)
        try:
            self._write_line(line)
        except Exception:
            logger.exception("writing the session timeline failed")

    def _write_line(self, obj: dict[str, Any]) -> None:
        assert self._timeline is not None
        self._timeline.write(obj)  # encoded and written by the writer thread

    # -------------------------------------------------------------------- audio
    def _pos(self, t: float) -> int:
        return round((t - self._origin) * self.sample_rate)

    def _restart_user_stream(self, rate: int) -> None:
        """Start (or, if the input rate changed, restart) the resampled user stream."""
        if self._user_rs is not None:
            self._put_user(self._user_rs.flush())
        if self._user_rate:
            self._user_base += self._user_in / self._user_rate
        self._user.truncate(self._pos(self._origin + self._user_base))
        self._user_rate, self._user_in = rate, 0
        self._gap = None
        self._user_rs = None if rate == self.sample_rate else Resampler(rate, self.sample_rate)
        self._user_delay = self._user_skip = _resampler_delay(rate, self.sample_rate)

    def _push_user(self, samples: np.ndarray[Any, np.dtype[np.int16]]) -> None:
        if len(samples) == 0:
            return
        self._user_in += len(samples)
        frame = AudioFrame(samples.tobytes(), self._user_rate, 1)
        self._put_user(frame if self._user_rs is None else self._user_rs.push(frame))

    def _push_silence(self, seconds: float) -> None:
        """Extend the user stream with ``seconds`` of silence (a gap in the input),
        remembering where it is so that :meth:`_take_back_silence` can undo it."""
        n = round(seconds * self._user_rate)
        if n <= 0:
            return
        gap = self._gap
        extend = gap is not None and gap.in_end == self._user_in  # no audio since it
        if self._user_rs is not None and not extend:
            # write the silence directly, not through the resampler (which holds back part
            # of its output): the audio so far is flushed and a new resampler takes over
            self._put_user(self._user_rs.flush())
            self._user_rs = Resampler(self._user_rate, self.sample_rate)
            self._user_skip = self._user_delay
        self._user_in += n
        pos = self._user.end
        end = self._pos(self._origin + self._user_base + self._user_in / self._user_rate)
        self._user.write(pos, np.zeros(max(0, end - pos), dtype=np.int16))
        if gap is not None and extend:
            gap.n_in += n
            gap.n_out = self._user.end - gap.pos
            gap.in_end = self._user_in
        else:
            self._gap = _Gap(pos, self._user.end - pos, n, self._user_in)

    def _take_back_silence(self, seconds: float) -> None:
        """Remove up to ``seconds`` of the latest inserted silence that is still in memory
        (from its middle: resampler ringing at its edges stays)."""
        gap = self._gap
        if gap is None:
            return
        ratio = self.sample_rate / self._user_rate
        lo = max(gap.pos, self._user.start)
        hi = min(gap.pos + gap.n_out, self._user.end)
        n_in = min(gap.n_in, round(seconds * self._user_rate), math.floor((hi - lo) / ratio))
        if n_in <= 0:
            return
        n_out = min(hi - lo, round(n_in * ratio))
        self._user.remove(lo + (hi - lo - n_out) // 2, n_out)
        self._user_in -= n_in
        gap.n_in -= n_in
        gap.n_out -= n_out
        gap.in_end -= n_in

    def _put_user(self, frame: AudioFrame) -> None:
        samples = frame.to_numpy()
        if self._user_skip:
            drop = min(self._user_skip, len(samples))
            samples, self._user_skip = samples[drop:], self._user_skip - drop
        self._user.write(max(self._user.end, self._user.start), samples)

    def _advance_user(self, t: float) -> None:
        """No user audio arrived until ``t``: extend the stream with silence."""
        if self._user_rate == 0:
            self._restart_user_stream(self.sample_rate)
        cursor = self._user_base + self._user_in / self._user_rate
        gap = t - self._origin - cursor
        if gap > 0:
            self._push_silence(gap)

    def _maybe_flush(self) -> None:
        t = now()
        horizon = self._pos(t - self.flush_delay)
        if self._paused_at is not None:
            horizon = min(horizon, self._paused_at)  # paused audio may still move
        if horizon - self._written < self.sample_rate // 2:
            return
        # silence up to shortly after the horizon: user audio arriving from now on lands
        # later (it is placed no earlier than its arrival time minus _JITTER)
        self._advance_user(t - self.flush_delay + _JITTER)
        self._flush(min(horizon, self._user.end))

    def _flush(self, end: int) -> None:
        assert self._wav is not None
        n = end - self._written
        if n <= 0:
            return
        stereo = np.empty((n, 2), dtype=np.int16)
        stereo[:, 0] = self._user.pop(n)
        stereo[:, 1] = self._agent.pop(n)
        self._written = end
        if self._wav_failed:
            return
        try:
            self._wav.write(AudioFrame(stereo.tobytes(), self.sample_rate, 2))
        except Exception:
            logger.exception("writing the session recording failed; audio recording stopped")
            self._wav_failed = True


def _resampler_delay(in_rate: int, out_rate: int) -> int:
    """Output samples by which a streaming :class:`Resampler` lags its input (measured
    with an impulse, so it holds for every backend)."""
    if in_rate == out_rate:
        return 0
    rs = Resampler(in_rate, out_rate)
    n = in_rate // 10
    k = n // 2
    x = np.zeros(n, dtype=np.int16)
    x[k] = 30_000
    y = np.concatenate(
        [rs.push(AudioFrame(x.tobytes(), in_rate, 1)).to_numpy(), rs.flush().to_numpy()]
    )
    if len(y) == 0:
        return 0
    peak = int(np.argmax(np.abs(y.astype(np.int32))))
    return max(0, round(peak - k * out_rate / in_rate))


# ------------------------------------------------------------------ serialization
def to_jsonable(obj: Any, *, drop: tuple[str, ...] = ()) -> Any:
    """Convert event payloads (dataclasses, enums, chat items, audio, exceptions) to JSON
    values. ``drop`` lists top-level dataclass fields to leave out."""
    if obj is None or isinstance(obj, bool | int | str):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, enum.Enum):
        return obj.value
    if isinstance(obj, BaseException):
        return {"error": type(obj).__name__, "message": str(obj)}
    if isinstance(obj, AudioFrame):
        return {"audio_duration": round(obj.duration, 6), "sample_rate": obj.sample_rate}
    if isinstance(obj, AudioContent):
        return {"type": "audio", "transcript": obj.transcript, "duration": obj.frame.duration}
    if isinstance(obj, bytes | bytearray):
        return {"bytes": len(obj)}
    if isinstance(obj, ChatMessage):
        return {
            "type": "message",
            "id": obj.id,
            "role": obj.role,
            "text": obj.text,
            "content": [to_jsonable(c) for c in obj.content],
            "interrupted": obj.interrupted,
        }
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {
            f.name: to_jsonable(getattr(obj, f.name))
            for f in dataclasses.fields(obj)
            if f.name not in drop
        }
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple | set | frozenset):
        return [to_jsonable(v) for v in obj]
    return str(obj)
