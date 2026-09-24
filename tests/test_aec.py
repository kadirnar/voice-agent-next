"""Echo cancellation / noise suppression processors and the half-duplex gate.

Most tests run a fake ``livekit.rtc`` that enforces the native module's contract (exact
10 ms frames; anything else aborts the real process), so framing, resampling, reference
handling and error paths are checked everywhere. Tests marked ``model`` run the real
WebRTC APM from the ``livekit`` wheel (``uv sync --extra aec``) on synthetic echo.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pytest

from voice_agent_next.audio import AudioFrame, aec
from voice_agent_next.audio.aec import HalfDuplexGate, WebRTCAudioProcessor, create_echo_canceller
from voice_agent_next.errors import MissingDependencyError
from voice_agent_next.providers.mock import synth_speech

# --------------------------------------------------------------------- fake livekit.rtc


class _FakeFrame:
    """Stand-in for ``livekit.rtc.AudioFrame``."""

    def __init__(
        self, data: Any, sample_rate: int, num_channels: int, samples_per_channel: int
    ) -> None:
        self._data = data
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self.samples_per_channel = samples_per_channel


class _FakeHandle:
    def __init__(self) -> None:
        self.disposed = False

    def dispose(self) -> None:
        self.disposed = True


class _FakeAPM:
    """Records every call; "processing" halves the capture samples in place."""

    def __init__(self, rtc: FakeRtc, **options: bool) -> None:
        self.rtc = rtc
        self.options = options
        self.log: list[tuple[Any, ...]] = []
        self._ffi_handle = _FakeHandle()
        rtc.instances.append(self)
        if rtc.fail_create:
            raise RuntimeError("native library not available")

    @staticmethod
    def _check(frame: _FakeFrame) -> None:
        # the real module aborts the process on anything but exact 10 ms frames
        assert isinstance(frame._data, bytearray)
        assert frame.samples_per_channel == frame.sample_rate // 100
        assert len(frame._data) == 2 * frame.samples_per_channel * frame.num_channels

    def process_stream(self, frame: _FakeFrame) -> None:
        self._check(frame)
        if self.rtc.fail_process:
            raise RuntimeError("an RtcError occurred")
        x = np.frombuffer(bytes(frame._data), dtype="<i2")
        frame._data[:] = (x // 2).astype("<i2").tobytes()
        self.log.append(("capture", frame.sample_rate, frame.num_channels))

    def process_reverse_stream(self, frame: _FakeFrame) -> None:
        self._check(frame)
        self.log.append(("render", frame.sample_rate, frame.num_channels, bytes(frame._data)))

    def set_stream_delay_ms(self, delay_ms: int) -> None:
        assert 0 <= delay_ms <= 500  # the real module raises outside this range
        self.log.append(("delay", delay_ms))

    # helpers for assertions
    def entries(self, kind: str) -> list[tuple[Any, ...]]:
        return [e for e in self.log if e[0] == kind]

    def real_render(self) -> list[bytes]:
        """Reference frames fed that are not silence."""
        return [e[3] for e in self.entries("render") if any(e[3])]


class FakeRtc:
    AudioFrame = _FakeFrame

    def __init__(self) -> None:
        self.instances: list[_FakeAPM] = []
        self.fail_create = False
        self.fail_process = False

    def AudioProcessingModule(self, **options: bool) -> _FakeAPM:
        return _FakeAPM(self, **options)

    @property
    def apm(self) -> _FakeAPM:
        assert self.instances, "no APM was created"
        return self.instances[-1]


@pytest.fixture
def fake_rtc(monkeypatch: pytest.MonkeyPatch) -> FakeRtc:
    rtc = FakeRtc()
    monkeypatch.setattr(aec, "_load_rtc", lambda: rtc)
    return rtc


@pytest.fixture
def no_livekit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``import livekit.rtc`` fail, whether or not the wheel is installed."""
    monkeypatch.setitem(sys.modules, "livekit.rtc", None)


# ----------------------------------------------------------------------------- helpers


def _noise(n: int, channels: int = 1, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(-12000, 12000, size=(n, channels) if channels > 1 else n).astype(np.int16)


def _split(x: np.ndarray, sample_rate: int, sizes: Sequence[int]) -> list[AudioFrame]:
    """Cut ``x`` (samples[, channels]) into frames whose lengths cycle through ``sizes``."""
    frames, i, k = [], 0, 0
    while i < len(x):
        n = sizes[k % len(sizes)]
        frames.append(
            AudioFrame.from_numpy(x[i : i + n], sample_rate, timestamp=10.0 + i / sample_rate)
        )
        i += n
        k += 1
    return frames


def _run(proc: Any, frames: Iterable[AudioFrame]) -> list[AudioFrame]:
    return [proc.process_capture(f) for f in frames]


def _samples(frames: Iterable[AudioFrame]) -> np.ndarray:
    return AudioFrame.concat(frames).to_numpy()


def _best_lag(y: np.ndarray, x: np.ndarray, max_lag: int) -> tuple[int, float]:
    """Lag (samples) maximizing the normalized correlation of ``y`` against ``x``."""
    best = (0, -1.0)
    for lag in range(max_lag + 1):
        a, b = y[lag:].astype(np.float64), x[: len(y) - lag].astype(np.float64)
        c = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
        if c > best[1]:
            best = (lag, c)
    return best


# ------------------------------------------------------------- framing and resampling


@pytest.mark.parametrize("rate", [16_000, 44_100, 48_000])
@pytest.mark.parametrize("sizes", [(1, 37, 512, 1023, 7), (333,), (2,)], ids=["mixed", "333", "2"])
def test_capture_is_framed_into_exact_10ms_frames(
    fake_rtc: FakeRtc, rate: int, sizes: tuple[int, ...]
) -> None:
    proc = WebRTCAudioProcessor()
    x = _noise(rate // 5)
    frames = _split(x, rate, sizes)
    out, latency = [], []
    for f in frames:
        out.append(proc.process_capture(f))
        latency.append(proc.latency)

    assert len(fake_rtc.instances) == 1
    captures = fake_rtc.apm.entries("capture")
    assert captures and all(e[1:] == (rate, 1) for e in captures)  # native rate, no resampling
    for f, o, lat in zip(frames, out, latency, strict=True):
        assert (o.sample_rate, o.channels, len(o.data)) == (f.sample_rate, f.channels, len(f.data))
        assert f.timestamp is not None and o.timestamp is not None
        assert o.timestamp == pytest.approx(f.timestamp - lat)  # capture time of its audio
    assert latency == sorted(latency)  # the delay only grows, once, up to its final value
    delay = round(proc.latency * rate)
    assert 0 <= delay < rate // 100  # at most the 10 ms framing remainder
    expected = np.concatenate([np.zeros(delay, np.int16), x // 2])[: len(x)]
    assert np.array_equal(_samples(out), expected)


@pytest.mark.parametrize("rate", [16_000, 44_100, 48_000])
def test_frames_that_are_multiples_of_10ms_add_no_latency(fake_rtc: FakeRtc, rate: int) -> None:
    proc = WebRTCAudioProcessor()
    x = _noise(rate // 4)
    out = _run(proc, _split(x, rate, [rate // 50, rate // 100]))  # 20 ms, 10 ms
    assert proc.latency == 0.0
    assert np.array_equal(_samples(out), x // 2)


@pytest.mark.parametrize(("rate", "apm_rate"), [(22_050, 48_000), (11_025, 16_000)])
def test_rates_without_whole_10ms_frames_are_resampled(
    fake_rtc: FakeRtc, rate: int, apm_rate: int
) -> None:
    proc = WebRTCAudioProcessor()
    x = synth_speech(0.5, rate, frequency=300.0, amplitude=0.5).to_numpy()
    frames = _split(x, rate, [331, 97, 1000])
    out = _run(proc, frames)

    captures = fake_rtc.apm.entries("capture")
    assert captures and all(e[1:] == (apm_rate, 1) for e in captures)
    assert [len(o.data) for o in out] == [len(f.data) for f in frames]
    assert proc.latency < 0.0105  # the framing remainder (rounded up to input samples)
    y = _samples(out)
    lag, corr = _best_lag(y, x // 2, max_lag=round(0.02 * rate))
    assert corr > 0.999  # a clean, delayed copy: no gaps
    assert abs(lag - round(proc.latency * rate)) <= 0.003 * rate  # ~1 ms per resampling filter


def test_forced_processing_rate(fake_rtc: FakeRtc) -> None:
    proc = WebRTCAudioProcessor(processing_rate=16_000)
    x = synth_speech(0.5, 48_000, frequency=1000.0, amplitude=0.5).to_numpy()
    frames = _split(x, 48_000, [960])
    out = _run(proc, frames)
    captures = fake_rtc.apm.entries("capture")
    assert captures and all(e[1:] == (16_000, 1) for e in captures)
    assert [len(o.data) for o in out] == [len(f.data) for f in frames]
    _, corr = _best_lag(_samples(out), x // 2, max_lag=960)
    assert corr > 0.99


def test_stereo_capture_is_processed_in_stereo(fake_rtc: FakeRtc) -> None:
    proc = WebRTCAudioProcessor()
    x = _noise(8000, channels=2)
    out = _run(proc, _split(x, 16_000, [250]))
    assert all(e[1:] == (16_000, 2) for e in fake_rtc.apm.entries("capture"))
    assert all(o.channels == 2 for o in out)
    delay = round(proc.latency * 16_000)
    expected = np.concatenate([np.zeros((delay, 2), np.int16), x // 2])[: len(x)]
    assert np.array_equal(_samples(out), expected)


def test_capture_format_change_starts_a_new_module(fake_rtc: FakeRtc) -> None:
    proc = WebRTCAudioProcessor()
    proc.process_capture(AudioFrame.from_numpy(_noise(320), 16_000))
    first = fake_rtc.apm
    out = proc.process_capture(AudioFrame.from_numpy(_noise(960), 48_000))
    assert first._ffi_handle.disposed
    assert len(fake_rtc.instances) == 2
    assert fake_rtc.apm.entries("capture")[0][1] == 48_000
    assert out.sample_rate == 48_000 and out.samples_per_channel == 960


def test_empty_frames_and_disabled_processing_pass_through(fake_rtc: FakeRtc) -> None:
    proc = WebRTCAudioProcessor()
    empty = AudioFrame.empty(16_000)
    assert proc.process_capture(empty) is empty
    proc.process_render(empty)
    off = WebRTCAudioProcessor(
        echo_cancellation=False,
        noise_suppression=False,
        high_pass_filter=False,
        auto_gain_control=False,
    )
    frame = AudioFrame.from_numpy(_noise(160), 16_000)
    assert off.process_capture(frame) is frame
    assert not fake_rtc.instances


# ------------------------------------------------------------------ echo reference


def test_reference_is_mono_10ms_at_the_capture_processing_rate(fake_rtc: FakeRtc) -> None:
    proc = WebRTCAudioProcessor()
    render = synth_speech(0.3, 24_000, amplitude=0.5).to_channels(2)
    for f in _split(render.to_numpy(), 24_000, [480, 77, 1000]):
        proc.process_render(f)
    _run(proc, _split(_noise(16_000 // 2), 16_000, [320]))

    renders = fake_rtc.apm.entries("render")
    assert all(e[1:3] == (16_000, 1) for e in renders)
    real = fake_rtc.apm.real_render()
    assert 29 <= len(real) <= 30  # 300 ms of reference (the resampler keeps a few samples)
    fed = np.frombuffer(b"".join(real), dtype="<i2").astype(np.float64)
    assert np.sqrt(np.mean(fed**2)) / 32768 == pytest.approx(render.rms(), rel=0.05)


def test_silence_is_fed_while_nothing_plays(fake_rtc: FakeRtc) -> None:
    proc = WebRTCAudioProcessor()
    _run(proc, _split(_noise(1600), 16_000, [320]))
    kinds = [e[0] for e in fake_rtc.apm.log]
    assert kinds == ["render", "capture"] * 10  # one silent reference frame per capture
    assert not fake_rtc.apm.real_render()


def test_played_reference_is_fed_as_soon_as_possible(fake_rtc: FakeRtc) -> None:
    proc = WebRTCAudioProcessor()  # reference="played"
    render = synth_speech(0.15, 16_000).to_numpy()
    for f in _split(render, 16_000, [320]):
        proc.process_render(f)
    proc.process_capture(AudioFrame.from_numpy(_noise(160), 16_000))
    assert [e[0] for e in fake_rtc.apm.log] == ["render"] * 15 + ["capture"]


def test_queued_reference_is_released_in_step_with_the_capture(fake_rtc: FakeRtc) -> None:
    proc = WebRTCAudioProcessor(reference="queued")
    render = synth_speech(0.15, 16_000).to_numpy()
    chunks = [render[i : i + 160].tobytes() for i in range(0, len(render), 160)]
    for f in _split(render, 16_000, [800, 800, 800]):  # a 150 ms burst, ahead of playback
        proc.process_render(f)
    lead = []
    for f in _split(_noise(4000), 16_000, [160]):
        proc.process_capture(f)
        log = fake_rtc.apm.log
        lead.append(sum(e[0] == "render" for e in log) - sum(e[0] == "capture" for e in log))
    assert max(lead) <= 3 - 1  # never more than 3 frames ahead of the capture (after it ran)
    assert min(lead) >= 0
    assert fake_rtc.apm.real_render() == chunks  # all of it, in order


def test_pending_reference_is_bounded(fake_rtc: FakeRtc) -> None:
    proc = WebRTCAudioProcessor()
    for _ in range(300):  # 3 s played while the microphone is not processed
        proc.process_render(synth_speech(0.01, 16_000))
    proc.process_capture(AudioFrame.from_numpy(_noise(160), 16_000))
    assert len(fake_rtc.apm.real_render()) == 100  # the last second


def test_reference_ignored_without_echo_cancellation(fake_rtc: FakeRtc) -> None:
    proc = WebRTCAudioProcessor(echo_cancellation=False)
    proc.set_device_latency(input_latency=0.01, output_latency=0.02)
    proc.process_render(synth_speech(0.1, 16_000))
    proc.process_capture(AudioFrame.from_numpy(_noise(320), 16_000))
    assert [e[0] for e in fake_rtc.apm.log] == ["capture", "capture"]
    assert fake_rtc.apm.options == {
        "echo_cancellation": False,
        "noise_suppression": True,
        "high_pass_filter": True,
        "auto_gain_control": True,
    }


def test_process_render_from_another_thread(fake_rtc: FakeRtc) -> None:
    proc = WebRTCAudioProcessor()
    render = [synth_speech(0.01, 48_000, offset=480 * i) for i in range(400)]
    ahead = threading.Semaphore(20)  # the playback thread runs at most ~200 ms ahead

    def playback() -> None:
        for f in render:
            ahead.acquire()
            proc.process_render(f)

    mic = AudioFrame.from_numpy(_noise(480), 48_000)
    t = threading.Thread(target=playback)
    t.start()
    for _ in range(100_000):  # the microphone keeps running while audio plays
        if not t.is_alive():
            break
        assert len(proc.process_capture(mic).data) == len(mic.data)
        ahead.release()
        time.sleep(0)  # let the playback thread run
    t.join(timeout=10)
    assert not t.is_alive()
    proc.process_capture(mic)  # drain the rest
    assert fake_rtc.apm.real_render() == [f.data for f in render]  # every frame, in order


# ----------------------------------------------------------------------- delay hint


def test_no_delay_hint_by_default(fake_rtc: FakeRtc) -> None:
    proc = WebRTCAudioProcessor()
    proc.process_capture(AudioFrame.from_numpy(_noise(320), 16_000))
    assert not fake_rtc.apm.entries("delay")
    assert proc.stream_delay_ms is None


def test_delay_hint_from_device_latencies(fake_rtc: FakeRtc) -> None:
    proc = WebRTCAudioProcessor()
    proc.set_device_latency(input_latency=0.012, output_latency=0.034)
    _run(proc, _split(_noise(1600), 16_000, [320]))
    assert fake_rtc.apm.entries("delay") == [("delay", 46)]  # set once, not per frame
    proc.set_device_latency(output_latency=0.9)  # clamped to the APM maximum
    proc.process_capture(AudioFrame.from_numpy(_noise(320), 16_000))
    assert proc.stream_delay_ms == 500


def test_explicit_delay_overrides_device_latencies(fake_rtc: FakeRtc) -> None:
    proc = WebRTCAudioProcessor(delay_ms=120)
    proc.set_device_latency(input_latency=0.01, output_latency=0.01)
    proc.process_capture(AudioFrame.from_numpy(_noise(320), 16_000))
    assert fake_rtc.apm.entries("delay") == [("delay", 120)]


# ---------------------------------------------------------- lifecycle and failures


def test_close_releases_the_module_and_reuse_recreates_it(fake_rtc: FakeRtc) -> None:
    proc = WebRTCAudioProcessor()
    proc.process_capture(AudioFrame.from_numpy(_noise(320), 16_000))
    first = fake_rtc.apm
    proc.close()
    assert first._ffi_handle.disposed
    assert proc.latency == 0.0
    proc.process_capture(AudioFrame.from_numpy(_noise(320), 16_000))
    assert len(fake_rtc.instances) == 2


def test_processing_errors_pass_audio_through(
    fake_rtc: FakeRtc, caplog: pytest.LogCaptureFixture
) -> None:
    fake_rtc.fail_process = True
    proc = WebRTCAudioProcessor()
    x = _noise(1600)
    with caplog.at_level(logging.ERROR, logger="voice_agent_next"):
        out = _run(proc, _split(x, 16_000, [320]))
    assert np.array_equal(_samples(out), x)
    assert len(caplog.records) == 1


def test_module_creation_failure_passes_audio_through(
    fake_rtc: FakeRtc, caplog: pytest.LogCaptureFixture
) -> None:
    fake_rtc.fail_create = True
    proc = WebRTCAudioProcessor()
    x = _noise(1600)
    with caplog.at_level(logging.ERROR, logger="voice_agent_next"):
        out = _run(proc, _split(x, 16_000, [320]))
    assert np.array_equal(_samples(out), x)
    assert len(fake_rtc.instances) == 1  # not retried for every frame
    assert len(caplog.records) == 1
    proc.reset()
    fake_rtc.fail_create = False
    proc.process_capture(AudioFrame.from_numpy(_noise(320), 16_000))
    assert len(fake_rtc.instances) == 2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"reference": "early"},
        {"delay_ms": -1},
        {"delay_ms": 501},
        {"processing_rate": 44_100},
    ],
)
def test_invalid_options(fake_rtc: FakeRtc, kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        WebRTCAudioProcessor(**kwargs)


def test_invalid_device_latency(fake_rtc: FakeRtc) -> None:
    proc = WebRTCAudioProcessor()
    with pytest.raises(ValueError):
        proc.set_device_latency(input_latency=-0.01)
    with pytest.raises(ValueError):
        proc.set_device_latency(output_latency=float("nan"))


def test_missing_livekit_raises_a_helpful_error(no_livekit: None) -> None:
    with pytest.raises(MissingDependencyError, match=r"voice-agent-next\[aec\]"):
        WebRTCAudioProcessor()


# ------------------------------------------------------------------ half-duplex gate


class _Clock:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t


def _tone(duration: float, sample_rate: int = 16_000) -> AudioFrame:
    return synth_speech(duration, sample_rate, amplitude=0.3)


def test_gate_passes_audio_when_nothing_plays() -> None:
    gate = HalfDuplexGate(clock=_Clock())
    frame = AudioFrame.from_numpy(_noise(320), 16_000)
    assert gate.process_capture(frame) is frame
    assert not gate.gating


def test_gate_mutes_while_playing_and_for_the_tail() -> None:
    clock = _Clock()
    gate = HalfDuplexGate(tail=0.3, clock=clock)
    gate.process_render(_tone(0.1))  # plays 100.0 .. 100.1
    frame = AudioFrame.from_numpy(_noise(320), 16_000, timestamp=5.0)
    for t in (100.0, 100.05, 100.39):
        clock.t = t
        muted = gate.process_capture(frame)
        assert gate.gating
        assert not muted.to_numpy().any()
        assert (muted.sample_rate, muted.channels, len(muted.data), muted.timestamp) == (
            16_000,
            1,
            640,
            5.0,
        )
    clock.t = 100.41
    assert not gate.gating
    assert gate.process_capture(frame) is frame


def test_gate_treats_a_burst_as_queued_playback() -> None:
    clock = _Clock()
    gate = HalfDuplexGate(tail=0.2, clock=clock)
    for _ in range(5):  # 500 ms handed over at once (e.g. the session's look-ahead)
        gate.process_render(_tone(0.1))
    clock.t = 100.69
    assert gate.gating
    clock.t = 100.71
    assert not gate.gating


def test_gate_follows_real_time_playback() -> None:
    clock = _Clock()
    gate = HalfDuplexGate(tail=0.3, clock=clock)
    for i in range(10):  # a playback callback every 20 ms
        clock.t = 100.0 + 0.02 * i
        gate.process_render(_tone(0.02))
    clock.t = 100.49
    assert gate.gating
    clock.t = 100.51
    assert not gate.gating


def test_gate_ignores_silent_reference_and_can_be_reset() -> None:
    clock = _Clock()
    gate = HalfDuplexGate(clock=clock)
    gate.process_render(AudioFrame.silence(0.5, 16_000))  # digital silence
    gate.process_render(AudioFrame.from_numpy(_noise(1600) // 3000, 16_000))  # -80 dBFS hiss
    assert not gate.gating
    gate.process_render(_tone(0.1))
    assert gate.gating
    gate.reset()
    assert not gate.gating
    gate.process_render(AudioFrame.empty(16_000))
    assert not gate.gating


def test_gate_with_zero_tail() -> None:
    clock = _Clock()
    gate = HalfDuplexGate(tail=0.0, clock=clock)
    gate.process_render(_tone(0.1))
    clock.t = 100.1
    assert not gate.gating


@pytest.mark.parametrize("tail", [-0.1, float("inf"), float("nan")])
def test_gate_rejects_invalid_tail(tail: float) -> None:
    with pytest.raises(ValueError):
        HalfDuplexGate(tail=tail)


# ---------------------------------------------------------------------------- factory


def test_factory_modes_without_dependencies() -> None:
    gate = create_echo_canceller("half_duplex", tail=0.5)
    assert isinstance(gate, HalfDuplexGate) and gate.tail == 0.5
    assert create_echo_canceller("headphones") is None
    assert create_echo_canceller("none") is None
    with pytest.raises(ValueError, match="unknown echo mode"):
        create_echo_canceller("magic")


@pytest.mark.parametrize("mode", ["auto", "aec", "webrtc"])
def test_factory_creates_the_webrtc_processor(fake_rtc: FakeRtc, mode: str) -> None:
    proc = create_echo_canceller(mode, noise_suppression=False, reference="queued", delay_ms=40)
    assert isinstance(proc, WebRTCAudioProcessor)
    assert proc.echo_cancellation and not proc.noise_suppression
    assert (proc.reference, proc.delay_ms) == ("queued", 40)


def test_factory_auto_without_livekit_warns(
    no_livekit: None, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="voice_agent_next"):
        assert create_echo_canceller("auto") is None
    assert "voice-agent-next[aec]" in caplog.text
    assert "half_duplex" in caplog.text


def test_factory_aec_without_livekit_raises(no_livekit: None) -> None:
    with pytest.raises(MissingDependencyError):
        create_echo_canceller("aec")


# ------------------------------------------------------ real WebRTC APM (livekit wheel)


@pytest.fixture
def livekit_rtc() -> Any:
    return pytest.importorskip("livekit.rtc", reason="needs the 'aec' extra (livekit wheel)")


def _speechlike(
    duration: float, sr: int, f0: float, seed: int, start: float, end: float
) -> np.ndarray:
    """Voiced "words": a harmonic series of ``synth_speech`` partials, gated on and off.

    (AEC3 does not adapt on pure tones, which it treats as narrow-band noise.)
    """
    n = round(duration * sr)
    x = np.zeros(n)
    for k, amp in ((1, 0.30), (2, 0.20), (3, 0.12), (4, 0.08), (5, 0.05)):
        x += synth_speech(duration, sr, frequency=f0 * k, amplitude=amp).to_float32()
    rng = np.random.default_rng(seed)
    gate = np.zeros(n)
    t, stop = round(start * sr), round(end * sr)
    while t < stop:
        on = int(rng.uniform(0.25, 0.6) * sr)
        gate[t : min(t + on, stop)] = 1.0
        t += on + int(rng.uniform(0.08, 0.3) * sr)
    w = np.hanning(int(0.01 * sr))
    return x * np.convolve(gate, w / w.sum(), mode="same") * 0.8


def _echo(render: np.ndarray, sr: int, delay_ms: float) -> np.ndarray:
    """Loudspeaker -> microphone path: delay, -6 dB, and one early reflection."""
    out = np.zeros_like(render)
    for extra_ms, gain in ((0.0, 0.5), (6.0, 0.15)):
        d = round((delay_ms + extra_ms) * sr / 1000)
        out[d:] += gain * render[: len(render) - d]
    return out


def _simulate(
    proc: WebRTCAudioProcessor,
    render: np.ndarray,
    capture: np.ndarray,
    sr: int,
    render_frame: int,
    capture_frame: int,
    lead: float = 0.02,
) -> np.ndarray:
    """Interleave playback-callback reference and microphone frames; returns processed mic."""
    r16 = AudioFrame.from_numpy(render.astype(np.float32), sr).to_numpy()
    c16 = AudioFrame.from_numpy(capture.astype(np.float32), sr).to_numpy()
    ahead = round(lead * sr)
    out, ri = [], 0
    for ci in range(0, len(c16), capture_frame):
        while ri < len(r16) and ri <= ci + capture_frame + ahead:
            proc.process_render(AudioFrame(r16[ri : ri + render_frame].tobytes(), sr))
            ri += render_frame
        frame = AudioFrame(c16[ci : ci + capture_frame].tobytes(), sr)
        processed = proc.process_capture(frame)
        assert processed.samples_per_channel == frame.samples_per_channel
        out.append(processed.to_numpy())
    y = np.concatenate(out).astype(np.float64)
    delay = round(proc.latency * sr)
    return np.concatenate([y[delay:], np.zeros(delay)])  # undo the framing delay


def _db(num: np.ndarray, den: np.ndarray) -> float:
    return float(10 * np.log10((np.mean(num**2) + 1e-9) / (np.mean(den**2) + 1e-9)))


@pytest.mark.model
@pytest.mark.parametrize(
    ("sr", "render_frame", "capture_frame", "echo_ms"),
    [
        (16_000, 320, 320, 40),  # 20 ms frames
        (48_000, 480, 336, 60),  # 7 ms capture frames
        (44_100, 882, 512, 120),  # odd sizes at 44.1 kHz (441-sample APM frames)
        (22_050, 441, 331, 60),  # resampled to 48 kHz
    ],
)
def test_webrtc_cancels_synthetic_echo(
    livekit_rtc: Any, sr: int, render_frame: int, capture_frame: int, echo_ms: float
) -> None:
    """Agent talks 0-7 s (echo only), user alone 7.5-10 s, double talk 11-14 s."""
    dur = 14.0
    render = _speechlike(dur, sr, 180.0, 1, 0.0, 7.0) + _speechlike(dur, sr, 180.0, 2, 10.0, dur)
    near = _speechlike(dur, sr, 125.0, 3, 7.5, 10.0) + _speechlike(dur, sr, 125.0, 4, 11.0, dur)
    capture = _echo(render, sr, echo_ms) + near
    proc = WebRTCAudioProcessor(noise_suppression=False, auto_gain_control=False)
    try:
        y = _simulate(proc, render, capture, sr, render_frame, capture_frame)
        assert proc.latency < 0.0105
    finally:
        proc.close()
    x = capture * 32767
    near16 = near * 32767

    def seg(a: float, b: float) -> slice:
        return slice(round(a * sr), round(b * sr))

    erle = _db(x[seg(2.0, 7.0)], y[seg(2.0, 7.0)])  # after convergence
    near_loss = _db(x[seg(7.6, 9.9)], y[seg(7.6, 9.9)])
    double_talk_loss = _db(near16[seg(11.0, dur)], y[seg(11.0, dur)])
    print(
        f"{sr} Hz: ERLE {erle:.1f} dB, near-end loss {near_loss:.1f} dB, double talk {double_talk_loss:.1f} dB"
    )
    assert erle >= 15.0
    assert abs(near_loss) <= 3.0  # the user's voice is kept when the agent is silent
    assert double_talk_loss <= 10.0  # and mostly kept while both talk


@pytest.mark.model
def test_webrtc_without_echo_cancellation_keeps_the_echo(livekit_rtc: Any) -> None:
    sr = 16_000
    render = _speechlike(6.0, sr, 180.0, 1, 0.0, 6.0)
    capture = _echo(render, sr, 40)
    proc = WebRTCAudioProcessor(
        echo_cancellation=False, noise_suppression=False, auto_gain_control=False
    )
    try:
        y = _simulate(proc, render, capture, sr, 320, 320)
    finally:
        proc.close()
    s = slice(2 * sr, 6 * sr)
    assert abs(_db(capture[s] * 32767, y[s])) < 1.0  # the baseline the ERLE is measured against


@pytest.mark.model
def test_webrtc_queued_reference_with_session_lookahead(livekit_rtc: Any) -> None:
    """Reference handed over up to 150 ms before it plays (AgentSession's playout loop)."""
    sr, frame = 16_000, 320
    turns = [(0.0, 3.0), (5.0, 8.0)]
    dur = 10.0
    render = sum(_speechlike(dur, sr, 180.0, 10 + i, a, b) for i, (a, b) in enumerate(turns))
    assert isinstance(render, np.ndarray)
    capture = _echo(render, sr, 60)
    r16 = AudioFrame.from_numpy(render.astype(np.float32), sr).to_numpy()
    c16 = AudioFrame.from_numpy(capture.astype(np.float32), sr).to_numpy()
    feeds = []  # (time the session writes it, sample index)
    for i in range(0, len(r16), frame):
        t = i / sr
        start = next((a for a, b in turns if a <= t < b), None)
        if start is not None:
            feeds.append((max(start, t - 0.15), i))
    proc = WebRTCAudioProcessor(
        reference="queued", noise_suppression=False, auto_gain_control=False
    )
    out, k = [], 0
    try:
        for ci in range(0, len(c16), frame):
            while k < len(feeds) and feeds[k][0] <= ci / sr:
                i = feeds[k][1]
                proc.process_render(AudioFrame(r16[i : i + frame].tobytes(), sr))
                k += 1
            out.append(
                proc.process_capture(AudioFrame(c16[ci : ci + frame].tobytes(), sr)).to_numpy()
            )
    finally:
        proc.close()
    y = np.concatenate(out).astype(np.float64)
    x = c16.astype(np.float64)
    second_turn = slice(5 * sr, round(8.2 * sr))  # including the echo tail after it ends
    assert _db(x[second_turn], y[second_turn]) >= 15.0
    worst = min(
        _db(
            x[slice(round(t * sr), round((t + 0.5) * sr))],
            y[slice(round(t * sr), round((t + 0.5) * sr))],
        )
        for t in np.arange(5.0, 8.0, 0.5)
    )
    assert worst >= 15.0


@pytest.mark.model
def test_webrtc_noise_suppression_reduces_stationary_noise(livekit_rtc: Any) -> None:
    sr = 16_000
    rng = np.random.default_rng(0)
    noise = rng.normal(0.0, 0.01, 6 * sr)  # -40 dBFS white noise
    proc = WebRTCAudioProcessor(
        echo_cancellation=False, high_pass_filter=False, auto_gain_control=False
    )
    try:
        frames = _split((noise * 32767).astype(np.int16), sr, [320])
        y = _samples(_run(proc, frames)).astype(np.float64)
    finally:
        proc.close()
    s = slice(3 * sr, 6 * sr)
    assert _db(noise[s] * 32767, y[s]) >= 6.0
