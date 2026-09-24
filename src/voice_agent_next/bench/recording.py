"""Two-channel call recordings on one clock, plus Audacity-compatible label files.

A :class:`DuplexRecording` places user audio (left channel) and agent audio (right
channel) by the :func:`~voice_agent_next.utils.now` time at which each chunk was spoken or
played, relative to a common ``origin`` (t = 0 of the recording). Every user-perceived
metric is then read off this recording (research note 06, §8.1: "the recorded audio is
the ground truth; traces explain it").

Each channel is kept at its native sample rate for analysis; :meth:`DuplexRecording.stereo`
resamples the user channel (delay-compensated) only to write the stereo WAV.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from ..audio.frame import AudioFrame
from ..audio.resample import resample
from ..audio.wav import write_wav

__all__ = ["DuplexRecording", "Label", "read_labels", "write_labels"]


class _Track:
    """Mono int16 audio assembled from time-stamped segments (later segments overwrite)."""

    def __init__(self, sample_rate: int) -> None:
        self.sample_rate = sample_rate
        self._segments: list[tuple[int, npt.NDArray[np.int16]]] = []
        self._end = 0

    def add(self, samples: npt.NDArray[np.int16], offset: float) -> None:
        start = round(offset * self.sample_rate)
        if start < 0:  # drop audio from before the origin
            samples = samples[-start:]
            start = 0
        if len(samples) == 0:
            return
        self._segments.append((start, samples))
        self._end = max(self._end, start + len(samples))

    @property
    def num_samples(self) -> int:
        return self._end

    def render(self, num_samples: int | None = None) -> npt.NDArray[np.int16]:
        n = self._end if num_samples is None else num_samples
        out = np.zeros(n, dtype=np.int16)
        for start, samples in self._segments:
            if start >= n:
                continue
            chunk = samples[: n - start]
            out[start : start + len(chunk)] = chunk
        return out


class DuplexRecording:
    """User and agent audio on one clock.

    Args:
        origin: :func:`~voice_agent_next.utils.now` value of t = 0.
        user_rate: sample rate of the user (left) channel.
        agent_rate: sample rate of the agent (right) channel.
    """

    def __init__(self, origin: float, *, user_rate: int, agent_rate: int) -> None:
        self.origin = origin
        self._user = _Track(user_rate)
        self._agent = _Track(agent_rate)

    @property
    def user_rate(self) -> int:
        return self._user.sample_rate

    @property
    def agent_rate(self) -> int:
        return self._agent.sample_rate

    @property
    def duration(self) -> float:
        return max(
            self._user.num_samples / self.user_rate, self._agent.num_samples / self.agent_rate
        )

    def to_offset(self, t: float) -> float:
        """Convert a :func:`~voice_agent_next.utils.now` time to recording time (s)."""
        return t - self.origin

    def add_user(self, frame: AudioFrame, start_time: float) -> None:
        """User audio that started at ``start_time`` (``now()`` clock)."""
        self._add(self._user, frame, start_time, None)

    def add_agent(
        self, frame: AudioFrame, start_time: float, end_time: float | None = None
    ) -> None:
        """Agent audio that started playing at ``start_time``; ``end_time`` cuts it short
        (e.g. playback cleared by a barge-in)."""
        self._add(self._agent, frame, start_time, end_time)

    def _add(self, track: _Track, frame: AudioFrame, start: float, end: float | None) -> None:
        if frame.sample_rate != track.sample_rate:
            raise ValueError(f"expected {track.sample_rate} Hz audio, got {frame.sample_rate} Hz")
        samples = frame.to_mono().to_numpy()
        if end is not None:
            keep = max(0, round((end - start) * track.sample_rate))
            samples = samples[:keep]
        track.add(samples, start - self.origin)

    def user_audio(self) -> AudioFrame:
        """Left channel (mono, ``user_rate``) from t = 0 to the end of the recording."""
        n = round(self.duration * self.user_rate)
        return AudioFrame(self._user.render(n).tobytes(), self.user_rate, 1)

    def agent_audio(self) -> AudioFrame:
        """Right channel (mono, ``agent_rate``) from t = 0 to the end of the recording."""
        n = round(self.duration * self.agent_rate)
        return AudioFrame(self._agent.render(n).tobytes(), self.agent_rate, 1)

    def stereo(self, sample_rate: int | None = None) -> AudioFrame:
        """Interleaved stereo (left = user, right = agent) at ``sample_rate``
        (default: the agent rate)."""
        rate = sample_rate or self.agent_rate
        user = resample(self.user_audio(), rate).to_numpy()
        agent = resample(self.agent_audio(), rate).to_numpy()
        n = round(self.duration * rate)
        both = np.zeros((n, 2), dtype=np.int16)
        both[: min(n, len(user)), 0] = user[:n]
        both[: min(n, len(agent)), 1] = agent[:n]
        return AudioFrame.from_numpy(both, rate)

    def write_wav(self, path: str | os.PathLike[str], sample_rate: int | None = None) -> Path:
        """Write the stereo recording (left = user, right = agent)."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        stereo = self.stereo(sample_rate)
        write_wav(out, stereo if stereo else AudioFrame.silence(0.01, stereo.sample_rate, 2))
        return out


@dataclass(frozen=True, slots=True)
class Label:
    """A labelled span of a recording (seconds from t = 0)."""

    start: float
    end: float
    text: str


def write_labels(path: str | os.PathLike[str], labels: list[Label]) -> Path:
    """Write an Audacity label track (``start<TAB>end<TAB>text`` per line)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"{lb.start:.6f}\t{lb.end:.6f}\t{' '.join(lb.text.split())}"
        for lb in sorted(labels, key=lambda lb: (lb.start, lb.end))
    ]
    out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return out


def read_labels(path: str | os.PathLike[str]) -> list[Label]:
    """Read a label file written by :func:`write_labels` (or exported by Audacity)."""
    labels: list[Label] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("\\"):  # Audacity frequency lines start with '\'
            continue
        start, end, *text = line.split("\t")
        labels.append(Label(float(start), float(end), text[0] if text else ""))
    return labels
