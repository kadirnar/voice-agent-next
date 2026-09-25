"""Streaming resampler state across mid-stream drains and rate changes; PCM16 reassembly."""

from __future__ import annotations

import base64
import importlib

import numpy as np
import pytest

from tests.conftest import dominant_frequency, tone
from voice_agent_next.audio import AudioFrame, PCM16Reassembler, Resampler, StreamResampler
from voice_agent_next.stt import STT, STTCapabilities, STTStream, Transcript
from voice_agent_next.utils.deps import is_installed

BACKENDS = ["numpy"] + (["soxr"] if is_installed("soxr") else [])
RATES = [(48_000, 16_000), (44_100, 16_000), (8000, 16_000), (24_000, 16_000), (16_000, 24_000)]


def _chunks(x: AudioFrame, n: int) -> list[AudioFrame]:
    """Split into chunks of ``n`` samples (odd sizes exercise every filter phase)."""
    return [
        AudioFrame(x.data[2 * i : 2 * (i + n)], x.sample_rate)
        for i in range(0, x.samples_per_channel, n)
    ]


def _run(rs: Resampler, chunks: list[AudioFrame], drain_every: int | None) -> np.ndarray:
    out: list[bytes] = []
    for i, c in enumerate(chunks):
        out.append(rs.push(c).data)
        if drain_every and i % drain_every == drain_every - 1:
            out.append(rs.drain().data)
            assert not rs.drain(), "drain() without new input must be a no-op"
    out.append(rs.flush().data)
    return np.frombuffer(b"".join(out), dtype=np.int16).astype(np.float64)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(("src", "dst"), RATES)
def test_drain_keeps_history_and_exact_sample_count(backend: str, src: int, dst: int) -> None:
    x = tone(440, 2.0, src)
    chunks = _chunks(x, src // 50 + 3)
    continuous = _run(Resampler(src, dst, backend=backend), chunks, None)  # type: ignore[arg-type]
    drained = _run(Resampler(src, dst, backend=backend), chunks, 3)  # type: ignore[arg-type]
    # ~33 drains: not a single sample inserted or lost (no clock drift)
    assert len(drained) == len(continuous)
    # the signal is continuous: only the few samples right before each drain point
    # (computed without look-ahead) differ from the uninterrupted stream
    err = np.sqrt(np.mean((drained - continuous) ** 2)) / np.sqrt(np.mean(continuous**2))
    assert err < 0.05  # a plain flush()+reset per turn: ~1.4 (and the length drifts)
    bad = np.count_nonzero(np.abs(drained - continuous) > 0.02 * np.abs(continuous).max())
    assert bad < 12 * (len(chunks) // 3)
    y = AudioFrame(drained.astype(np.int16).tobytes(), dst)
    assert dominant_frequency(y) == pytest.approx(440, abs=3)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(("src", "dst"), RATES)
def test_output_count_never_drifts_over_many_drains(backend: str, src: int, dst: int) -> None:
    rs = Resampler(src, dst, backend=backend)  # type: ignore[arg-type]
    rng = np.random.default_rng(0)
    n_in = n_out = 0
    counts = []
    for _ in range(200):
        n = int(rng.integers(1, src // 25))
        x = (rng.standard_normal(n) * 3000).astype(np.int16)
        n_in += n
        n_out += rs.push(AudioFrame(x.tobytes(), src)).samples_per_channel
        n_out += rs.drain().samples_per_channel
        counts.append(n_out - n_in * dst / src)
    # the offset (the filter delay) is the same after every drain: no per-drain padding
    assert max(counts) - min(counts) <= 1.0


def test_numpy_drain_is_exact_after_the_drain_point() -> None:
    # outputs that come after a drain are bit-identical to an uninterrupted stream:
    # the history is real audio, not zeros
    x = tone(300, 0.5, 16_000)
    first, second = x.slice(0, 0.2), x.slice(0.2, 0.5)
    a = Resampler(16_000, 24_000, backend="numpy")
    ref = np.concatenate([a.push(first).to_numpy(), a.push(second).to_numpy()])
    b = Resampler(16_000, 24_000, backend="numpy")
    head = b.push(first).to_numpy()
    tail = b.drain().to_numpy()
    after = b.push(second).to_numpy()
    assert len(head) + len(tail) + len(after) == len(ref)
    np.testing.assert_array_equal(after, ref[len(head) + len(tail) :])


@pytest.mark.parametrize("backend", BACKENDS)
def test_flush_still_resets_between_streams(backend: str) -> None:
    rs = Resampler(48_000, 16_000, backend=backend)  # type: ignore[arg-type]
    fresh = Resampler(48_000, 16_000, backend=backend)  # type: ignore[arg-type]
    rs.push(tone(440, 0.3, 48_000))
    rs.flush()
    assert not rs.flush()  # nothing pending after a flush
    x = tone(1000, 0.3, 48_000)
    got = AudioFrame.concat([rs.push(x), rs.flush()])
    want = AudioFrame.concat([fresh.push(x), fresh.flush()])
    assert got.data == want.data


def test_flush_after_drain_emits_nothing_more() -> None:
    for backend in BACKENDS:
        rs = Resampler(44_100, 16_000, backend=backend)  # type: ignore[arg-type]
        rs.push(tone(440, 0.25, 44_100))
        assert rs.drain()
        assert not rs.flush()


@pytest.mark.parametrize("backend", BACKENDS)
def test_stream_resampler_rate_change_emits_old_tail_first(
    backend: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        importlib.import_module("voice_agent_next.audio.resample"),
        "is_installed",
        lambda name: backend == "soxr",
    )
    srs = StreamResampler(16_000)
    a = tone(440, 0.2, 48_000)
    first = srs.push(a)
    ref = Resampler(48_000, 16_000, backend=backend)  # type: ignore[arg-type]
    ref.push(a)
    old_tail = ref.flush()
    assert old_tail
    # 48 kHz -> 24 kHz: the 48 kHz tail precedes the first 24 kHz output
    b = tone(440, 0.2, 24_000)
    second = srs.push(b)
    assert second.data.startswith(old_tail.data)
    # 24 kHz -> 16 kHz (passthrough): again the 24 kHz tail comes first...
    c = tone(440, 0.1, 16_000)
    third = srs.push(c)
    assert third.data.endswith(c.data) and len(third.data) > len(c.data)
    # ...and no stale tail of an earlier rate appears after the newer audio
    assert srs.push(c) is c
    assert not srs.flush()
    total = first.samples_per_channel + second.samples_per_channel + third.samples_per_channel
    # plus the numpy filter delay once per finished rate (soxr compensates it)
    assert total == pytest.approx(3200 + 3200 + 1600, abs=32)


# ------------------------------------------------------------------ STTStream.flush()


class _RecordingSTT(STT):
    provider = "test"

    def __init__(self) -> None:
        super().__init__(
            model="rec", capabilities=STTCapabilities(streaming=True), sample_rate=16_000
        )
        self.audio: list[AudioFrame] = []
        self.flushes = 0

    async def _recognize(self, audio: AudioFrame, *, language: str | None) -> Transcript:
        return Transcript("")

    def _create_stream(self, *, language: str | None) -> STTStream:
        return _RecordingStream(self, language=language)


class _RecordingStream(STTStream):
    async def _run(self) -> None:
        stt: _RecordingSTT = self._stt  # type: ignore[assignment]
        async for item in self._input:
            if self.is_flush(item):
                stt.flushes += 1
            else:
                assert isinstance(item, AudioFrame)
                stt.audio.append(item)


async def test_stt_stream_flush_does_not_insert_audio() -> None:
    stt = _RecordingSTT()
    stream = stt.stream()
    x = tone(440, 3.0, 48_000)
    ref = Resampler(48_000, 16_000)
    expected = []
    for i, chunk in enumerate(_chunks(x, 960)):
        stream.push_audio(chunk)
        expected.append(ref.push(chunk))
        if i % 10 == 9:
            stream.flush()  # a forced finalization per "turn"
    stream.end_input()
    expected.append(ref.flush())
    async for _ in stream:
        pass
    await stream.aclose()
    assert stt.flushes >= 15
    got = AudioFrame.concat(stt.audio)
    # the STT clock does not drift: exactly the samples of one uninterrupted stream
    assert got.samples_per_channel == AudioFrame.concat(expected).samples_per_channel


# --------------------------------------------------------------------- PCM16 reassembly


def test_pcm16_reassembler_odd_splits_preserve_every_sample() -> None:
    samples = (np.arange(-500, 500) * 37).astype(np.int16)
    raw = samples.tobytes()
    rng = np.random.default_rng(1)
    cuts = sorted(set(rng.integers(1, len(raw), 60).tolist()))
    pieces = [raw[a:b] for a, b in zip([0, *cuts], [*cuts, len(raw)], strict=True)]
    assert any(len(p) % 2 for p in pieces)
    r = PCM16Reassembler()
    out = []
    for p in pieces:
        chunk = r.push(p)
        assert len(chunk) % 2 == 0
        out.append(chunk)
    assert r.pending == 0
    np.testing.assert_array_equal(np.frombuffer(b"".join(out), dtype=np.int16), samples)


def test_pcm16_reassembler_single_bytes_stereo_and_reset() -> None:
    r = PCM16Reassembler(channels=2)
    assert r.push(b"\x01") == b"" and r.pending == 1
    assert r.push(b"\x02\x03") == b"" and r.pending == 3
    assert r.push(b"\x04\x05") == b"\x01\x02\x03\x04" and r.pending == 1
    assert r.reset() == b"\x05" and r.pending == 0
    data = bytes(range(8))
    assert r.push(data) == data
    with pytest.raises(ValueError):
        PCM16Reassembler(channels=0)


def test_pcm16_reassembler_decodes_base64_deltas_split_mid_sample() -> None:
    samples = np.array([1000, -2000, 3000, -4000, 5000], dtype=np.int16)
    raw = samples.tobytes()
    deltas = [base64.b64encode(raw[:3]), base64.b64encode(raw[3:6]), base64.b64encode(raw[6:])]
    r = PCM16Reassembler()
    got = b"".join(r.push(base64.b64decode(d)) for d in deltas)
    np.testing.assert_array_equal(np.frombuffer(got, dtype=np.int16), samples)
