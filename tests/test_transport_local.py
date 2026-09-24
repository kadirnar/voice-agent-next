"""LocalAudioTransport tests with a fake ``sounddevice`` module.

No audio hardware is touched: :class:`FakeSoundDevice` replaces ``sounddevice`` in
``sys.modules`` and its streams invoke the transport's callbacks from their own threads,
like PortAudio does, either on demand (``stream.run(n)``) or in real time. Only
``test_real_devices_record_and_play`` (``@pytest.mark.audio_device``, deselected by
default) uses real devices.
"""

from __future__ import annotations

import asyncio
import functools
import importlib.machinery
import json
import logging
import queue
import sys
import threading
import time
import types
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
import pytest
from typer.testing import CliRunner

from tests.conftest import tone
from voice_agent_next import Agent, AgentSession, AgentState, AudioFrame
from voice_agent_next.audio.frame import AudioFormat
from voice_agent_next.audio.processing import AudioProcessor
from voice_agent_next.cli.main import app
from voice_agent_next.errors import ConfigurationError, MissingDependencyError, TransportError
from voice_agent_next.providers.mock import MockEngine, synth_speech
from voice_agent_next.transports import create_transport, local
from voice_agent_next.transports.local import (
    AudioDeviceInfo,
    AudioSystemInfo,
    LocalAudioTransport,
    describe_audio_system,
    find_audio_device,
    import_sounddevice,
    list_audio_devices,
    portaudio_install_hint,
    select_audio_device,
)

RATE = 16_000
BLOCK = 160  # 10 ms at 16 kHz
PORTAUDIO_19_6 = "PortAudio V19.6.0-devel, revision 396fe4b6699ae929d3a685b3ef8a7e97396139a4"


# ------------------------------------------------------------------------ fake backend
class FakeClock:
    """Replaces ``now()`` in the transport module; the fake streams advance it."""

    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class FakePortAudioError(Exception):
    pass


class FakeCallbackFlags:
    def __init__(self, *, input_overflow: bool = False, output_underflow: bool = False) -> None:
        self.input_overflow = input_overflow
        self.output_underflow = output_underflow

    def __bool__(self) -> bool:
        return self.input_overflow or self.output_underflow


class FakeMic:
    """What the microphone hears: queued audio, then a constant ``level``."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending = np.zeros(0, dtype=np.int16)
        self.level = 0

    def feed(self, frame: AudioFrame) -> None:
        with self._lock:
            self._pending = np.concatenate([self._pending, frame.to_numpy()])

    def read(self, n: int) -> np.ndarray:
        with self._lock:
            out = np.full(n, self.level, dtype=np.int16)
            take = min(n, len(self._pending))
            out[:take] = self._pending[:take]
            self._pending = self._pending[take:]
            return out


class FakeStream:
    """Stands in for ``sounddevice.InputStream``/``OutputStream``."""

    def __init__(
        self,
        backend: FakeSoundDevice,
        kind: str,
        *,
        samplerate: float,
        blocksize: int,
        device: int,
        channels: int,
        dtype: str,
        latency: Any,
        callback: Callable[..., None],
        finished_callback: Callable[[], None] | None = None,
        **extra: Any,
    ) -> None:
        if backend.fail_open == kind:
            raise FakePortAudioError("Invalid sample rate [PaErrorCode -9997]")
        assert dtype == "int16"
        assert not extra, extra
        self.backend = backend
        self.kind = kind
        self.samplerate = float(samplerate)
        self.blocksize = blocksize
        self.device = device
        self.channels = channels
        self.latency_hint = latency
        self.latency = backend.input_latency if kind == "input" else backend.output_latency
        self.frames_per_block = blocksize or round(backend.block * samplerate)
        self._callback = callback
        self._finished = finished_callback
        self.active = False
        self.closed = False
        self.callback_threads: set[int] = set()
        self._jobs: queue.Queue[tuple[int, threading.Event] | None] = queue.Queue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        backend.streams.append(self)

    def start(self) -> None:
        self.active = True
        target = self._run_realtime if self.backend.realtime else self._run_jobs
        self._thread = threading.Thread(target=target, name=f"fake-{self.kind}", daemon=True)
        self._thread.start()

    def run(self, blocks: int = 1, timeout: float = 5.0) -> None:
        """Invoke the callback ``blocks`` times on the stream thread; wait until done."""
        done = threading.Event()
        self._jobs.put((blocks, done))
        assert done.wait(timeout), "the fake audio thread is stuck"

    def fail(self) -> None:
        """The device disappears: the stream stops and calls ``finished_callback``."""
        self.active = False
        self._stop.set()
        if self._finished is not None:
            self._finished()

    def close(self, ignore_errors: bool = True) -> None:
        if self.closed:
            return
        was_active, self.active, self.closed = self.active, False, True
        self._stop.set()
        self._jobs.put(None)
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(5)
        if was_active and self._finished is not None:
            self._finished()

    def _run_jobs(self) -> None:
        while (job := self._jobs.get()) is not None:
            blocks, done = job
            for _ in range(blocks):
                if self.active:
                    self._tick()
            done.set()

    def _run_realtime(self) -> None:
        period = self.frames_per_block / self.samplerate
        deadline = time.perf_counter()
        while not self._stop.is_set():
            self._tick()
            deadline += period
            self._stop.wait(max(0.0, deadline - time.perf_counter()))

    def _tick(self) -> None:
        self.callback_threads.add(threading.get_ident())
        n = self.frames_per_block
        if self.backend.clock is not None:
            self.backend.clock.advance(n / self.samplerate)
        status = self.backend.next_status(self.kind)
        if self.kind == "input":
            block = self.backend.mic_block(n)
            self._callback(np.repeat(block[:, None], self.channels, axis=1), n, None, status)
        else:
            out = np.full((n, self.channels), 777, dtype=np.int16)  # must be overwritten
            self._callback(out, n, None, status)
            self.backend.record_played(out.copy())


def _device(name: str, hostapi: int = 0, inputs: int = 0, outputs: int = 0, rate: float = RATE):
    return {
        "name": name,
        "hostapi": hostapi,
        "max_input_channels": inputs,
        "max_output_channels": outputs,
        "default_low_input_latency": 0.008 if inputs else 0.0,
        "default_low_output_latency": 0.008 if outputs else 0.0,
        "default_high_input_latency": 0.032 if inputs else 0.0,
        "default_high_output_latency": 0.032 if outputs else 0.0,
        "default_samplerate": rate,
    }


DEVICES = [
    _device("Built-in Microphone", inputs=1),  # 0: default input
    _device("Built-in Speakers", outputs=2),  # 1: default output
    _device("USB Headset", inputs=1, outputs=2, rate=48_000),  # 2
    _device("USB Headset", hostapi=1, inputs=1, outputs=2, rate=48_000),  # 3 (JACK)
    _device("pipewire", inputs=64, outputs=64, rate=44_100),  # 4
    _device("USB Headset Monitor", inputs=2, rate=48_000),  # 5
]


class FakeSoundDevice:
    """A fake ``sounddevice`` module: a device list plus thread-driven fake streams."""

    def __init__(
        self,
        *,
        devices: list[dict[str, Any]] | None = None,
        hostapis: tuple[str, ...] = ("ALSA", "JACK Audio Connection Kit"),
        default: tuple[int, int] = (0, 1),
        input_latency: float = 0.02,
        output_latency: float = 0.03,
        block: float = 0.01,
        realtime: bool = False,
        clock: FakeClock | None = None,
        echo: bool = False,
    ) -> None:
        self.devices = [dict(d, index=i) for i, d in enumerate(devices or DEVICES)]
        self.hostapis = hostapis
        self.default = default
        self.input_latency = input_latency
        self.output_latency = output_latency
        self.block = block
        self.realtime = realtime
        self.clock = clock
        self.echo = echo  # the microphone hears the speakers
        self.fail_open: str | None = None
        self.mic = FakeMic()
        self.streams: list[FakeStream] = []
        self.played: list[np.ndarray] = []
        self.statuses: dict[str, list[FakeCallbackFlags]] = {"input": [], "output": []}
        self._lock = threading.Lock()

    def module(self) -> types.ModuleType:
        mod = types.ModuleType("sounddevice")
        mod.__spec__ = importlib.machinery.ModuleSpec("sounddevice", None)
        mod.__version__ = "0.5.6"  # type: ignore[attr-defined]
        mod.PortAudioError = FakePortAudioError  # type: ignore[attr-defined]
        mod.query_devices = self.query_devices  # type: ignore[attr-defined]
        mod.query_hostapis = self.query_hostapis  # type: ignore[attr-defined]
        mod.get_portaudio_version = lambda: (1246976, PORTAUDIO_19_6)  # type: ignore[attr-defined]
        mod.InputStream = functools.partial(FakeStream, self, "input")  # type: ignore[attr-defined]
        mod.OutputStream = functools.partial(FakeStream, self, "output")  # type: ignore[attr-defined]
        return mod

    # sounddevice API
    def query_hostapis(self, index: int | None = None) -> Any:
        apis = tuple({"name": name, "devices": []} for name in self.hostapis)
        return apis if index is None else apis[index]

    def query_devices(self, device: int | None = None, kind: str | None = None) -> Any:
        if device is None and kind is None:
            return [dict(d) for d in self.devices]
        if device is None:
            device = self.default[0 if kind == "input" else 1]
            if device < 0:
                raise FakePortAudioError(f"Error querying device {device}")
        return dict(self.devices[device])

    # test helpers
    def stream(self, kind: str) -> FakeStream:
        return next(s for s in reversed(self.streams) if s.kind == kind)

    @property
    def input_stream(self) -> FakeStream:
        return self.stream("input")

    @property
    def output_stream(self) -> FakeStream:
        return self.stream("output")

    def next_status(self, kind: str) -> FakeCallbackFlags:
        pending = self.statuses[kind]
        return pending.pop(0) if pending else FakeCallbackFlags()

    def mic_block(self, n: int) -> np.ndarray:
        block = self.mic.read(n).astype(np.int32)
        with self._lock:
            last = self.played[-1] if self.played else None
        if self.echo and last is not None and len(last) == n:
            block += last.mean(axis=1).astype(np.int32)
        return np.clip(block, -32768, 32767).astype(np.int16)

    def record_played(self, block: np.ndarray) -> None:
        with self._lock:
            self.played.append(block)

    def played_audio(self) -> np.ndarray:
        with self._lock:
            return np.concatenate(self.played) if self.played else np.zeros((0, 1), np.int16)

    def close_all(self) -> None:
        for s in self.streams:
            s.close()


class FakeEchoCanceller(AudioProcessor):
    """Records what it is fed; "cleans" the microphone by halving it."""

    def __init__(self) -> None:
        self.render: list[AudioFrame] = []
        self.capture: list[AudioFrame] = []
        self.threads: dict[str, set[int]] = {"render": set(), "capture": set()}
        self.resets = 0
        self.closed = False

    def process_capture(self, frame: AudioFrame) -> AudioFrame:
        self.capture.append(frame)
        self.threads["capture"].add(threading.get_ident())
        return AudioFrame.from_numpy(
            frame.to_numpy() // 2, frame.sample_rate, timestamp=frame.timestamp
        )

    def process_render(self, frame: AudioFrame) -> None:
        self.render.append(frame)
        self.threads["render"].add(threading.get_ident())

    def reset(self) -> None:
        self.resets += 1

    def close(self) -> None:
        self.closed = True


class BrokenEchoCanceller(FakeEchoCanceller):
    def process_capture(self, frame: AudioFrame) -> AudioFrame:
        raise RuntimeError("echo canceller exploded")


# ------------------------------------------------------------------------------ fixtures
@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    fake = FakeClock()
    monkeypatch.setattr(local, "now", fake)
    return fake


@pytest.fixture
def make_fake_sd(monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable[..., FakeSoundDevice]]:
    created: list[FakeSoundDevice] = []

    def make(**kwargs: Any) -> FakeSoundDevice:
        backend = FakeSoundDevice(**kwargs)
        monkeypatch.setitem(sys.modules, "sounddevice", backend.module())
        created.append(backend)
        return backend

    # deterministic "auto" echo mode, whether or not an echo canceller is installed
    monkeypatch.setattr(local, "_default_echo_canceller", lambda: None)
    yield make
    for backend in created:
        backend.close_all()


@pytest.fixture
def fake_sd(make_fake_sd: Callable[..., FakeSoundDevice], clock: FakeClock) -> FakeSoundDevice:
    return make_fake_sd(clock=clock)


async def take(
    transport: LocalAudioTransport, count: int, timeout: float = 5.0
) -> list[AudioFrame]:
    frames: list[AudioFrame] = []
    stream = transport.audio_input()

    async def collect() -> None:
        async for frame in stream:
            frames.append(frame)
            if len(frames) >= count:
                return

    try:
        await asyncio.wait_for(collect(), timeout)
    finally:
        await stream.aclose()  # type: ignore[attr-defined]
    return frames


async def wait_for(predicate: Callable[[], bool], timeout: float = 10.0) -> None:
    async def poll() -> None:
        while not predicate():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(poll(), timeout)


def ramp(n: int, start: int = 1, step: int = 1) -> AudioFrame:
    """A frame of distinct, non-zero samples (to check ordering)."""
    return AudioFrame.from_numpy(start + step * np.arange(n, dtype=np.int16), RATE)


# ------------------------------------------------------------------------------- devices
def test_list_audio_devices_reports_host_apis_and_defaults(fake_sd: FakeSoundDevice) -> None:
    devices = list_audio_devices()
    assert [d.index for d in devices] == list(range(len(DEVICES)))
    mic, speakers, _, jack_headset, *_ = devices
    assert (mic.name, mic.hostapi, mic.max_input_channels) == ("Built-in Microphone", "ALSA", 1)
    assert mic.is_default_input and not mic.is_default_output
    assert speakers.is_default_output and not speakers.is_default_input
    assert jack_headset.hostapi == "JACK Audio Connection Kit"
    assert str(jack_headset) == "[3] USB Headset (JACK Audio Connection Kit)"
    assert find_audio_device("pipewire", "output").default_samplerate == 44_100


@pytest.mark.parametrize(
    ("query", "kind", "index"),
    [
        (None, "input", 0),  # the default device
        (None, "output", 1),
        (2, "input", 2),
        ("2", "output", 2),  # digits from a config file or the CLI
        ("usb headset", "input", 2),  # exact name on two host APIs: the default's wins
        ("USB HEADSET jack", "input", 3),  # host API words narrow it down
        ("monitor", "input", 5),
        ("usb", "output", 2),  # only 2 and 3 have outputs; ALSA is the default's host API
        ("pipe", "input", 4),
    ],
)
def test_select_audio_device(
    fake_sd: FakeSoundDevice, query: int | str | None, kind: str, index: int
) -> None:
    assert select_audio_device(list_audio_devices(), query, kind).index == index  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("query", "kind", "message"),
    [
        (99, "input", "no audio device has index 99"),
        (1, "input", "is not an input device"),
        ("usb", "input", "matches several input devices"),  # [2] and [5], both ALSA
        ("bluetooth", "output", "no output device matches 'bluetooth'"),
        (True, "input", "invalid input_device"),
    ],
)
def test_select_audio_device_errors(
    fake_sd: FakeSoundDevice, query: Any, kind: str, message: str
) -> None:
    with pytest.raises(ConfigurationError, match=message) as info:
        select_audio_device(list_audio_devices(), query, kind)  # type: ignore[arg-type]
    if "several" in message:
        assert "[2] USB Headset (ALSA)" in str(info.value)
        assert "[5] USB Headset Monitor (ALSA)" in str(info.value)
    elif "matches" in message:  # lists what is available
        assert "[1] Built-in Speakers (ALSA)" in str(info.value)


def test_no_default_device_is_a_clear_error(make_fake_sd: Callable[..., FakeSoundDevice]) -> None:
    make_fake_sd(default=(-1, 1))
    with pytest.raises(ConfigurationError, match="no default input device"):
        LocalAudioTransport(echo_mode="headphones")
    t = LocalAudioTransport(input_device="usb headset", echo_mode="headphones")
    assert t.input_device.index == 2  # ties broken by the default output's host API


def test_stream_formats_follow_device_defaults_unless_given(fake_sd: FakeSoundDevice) -> None:
    t = LocalAudioTransport(echo_mode="headphones")
    assert (t.input_format, t.output_format) == (AudioFormat(RATE, 1), AudioFormat(RATE, 1))
    t = LocalAudioTransport(
        input_device="usb headset", output_device="pipewire", echo_mode="headphones"
    )
    assert (t.input_format.sample_rate, t.output_format.sample_rate) == (48_000, 44_100)
    t = LocalAudioTransport(sample_rate=24_000, output_sample_rate=22_050, echo_mode="headphones")
    assert (t.input_format.sample_rate, t.output_format.sample_rate) == (24_000, 22_050)
    t = LocalAudioTransport(output_channels=2, echo_mode="headphones")
    assert t.output_format == AudioFormat(RATE, 2)
    assert "Built-in Microphone" in repr(t)
    with pytest.raises(ConfigurationError, match="1 input channel"):
        LocalAudioTransport(input_channels=2, echo_mode="headphones")


def test_create_transport_local(fake_sd: FakeSoundDevice) -> None:
    t = create_transport(
        {"type": "local", "input_device": "usb headset jack", "output_device": 2,
         "echo_mode": "headphones"}
    )  # fmt: skip
    assert isinstance(t, LocalAudioTransport)
    assert (t.input_device.index, t.output_device.index) == (3, 2)
    assert t.capabilities.pause and t.capabilities.playback_position


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"echo_mode": "loud"}, "echo_mode must be one of"),
        ({"echo_mode": "headphones", "echo_canceller": FakeEchoCanceller()}, "only used with"),
        ({"echo_mode": "half_duplex", "echo_canceller": FakeEchoCanceller()}, "only used with"),
        ({"echo_mode": "aec"}, "needs an echo canceller"),  # none given, none installed
        ({"block_duration": 0}, "block_duration"),
        ({"latency": "fast"}, "latency"),
        ({"latency": -1.0}, "latency"),
        ({"half_duplex_tail": -0.1}, "half_duplex_tail"),
        ({"max_buffered": 0}, "max_buffered"),
        ({"output_channels": 0}, "channels"),
        ({"sample_rate": -8000}, "sample rate"),
    ],
)
def test_invalid_options(fake_sd: FakeSoundDevice, kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(ConfigurationError, match=message):
        LocalAudioTransport(**kwargs)


# ---------------------------------------------------------------- missing dependencies
def test_missing_sounddevice_names_the_audio_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "sounddevice", None)
    with pytest.raises(MissingDependencyError, match=r"voice-agent-next\[audio\]"):
        LocalAudioTransport()


def test_missing_portaudio_library_says_how_to_install_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    # what `import sounddevice` does on Linux without libportaudio2
    (tmp_path / "sounddevice.py").write_text("raise OSError('PortAudio library not found')\n")
    monkeypatch.delitem(sys.modules, "sounddevice", raising=False)
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(MissingDependencyError) as info:
        import_sounddevice()
    assert "PortAudio library not found" in str(info.value)
    assert portaudio_install_hint() in str(info.value)


def test_portaudio_install_hint_per_platform() -> None:
    assert "sudo apt install libportaudio2" in portaudio_install_hint("linux")
    for platform in ("darwin", "win32"):
        assert "bundle PortAudio" in portaudio_install_hint(platform)


# ----------------------------------------------------------------------------- lifecycle
async def test_start_opens_low_latency_int16_streams_and_aclose_releases_them(
    fake_sd: FakeSoundDevice,
) -> None:
    t = LocalAudioTransport(input_device="usb headset", echo_mode="headphones")
    events: list[str] = []
    t.on("connected", lambda: events.append("connected"))
    t.on("disconnected", lambda: events.append("disconnected"))
    await t.start()
    await t.start()  # idempotent
    mic, speakers = fake_sd.input_stream, fake_sd.output_stream
    assert len(fake_sd.streams) == 2
    assert (mic.device, mic.samplerate, mic.channels, mic.blocksize) == (2, 48_000, 1, 0)
    assert (speakers.device, speakers.samplerate, speakers.channels) == (1, RATE, 1)
    assert mic.latency_hint == speakers.latency_hint == "low"
    assert mic.active and speakers.active
    assert (t.input_latency, t.output_latency) == (0.02, 0.03)
    await t.aclose()
    await t.aclose()  # idempotent
    assert mic.closed and speakers.closed
    assert events == ["connected", "disconnected"]
    assert [f async for f in t.audio_input()] == []  # input ended
    with pytest.raises(TransportError, match="closed"):
        await t.start()


async def test_block_duration_sets_the_blocksize(fake_sd: FakeSoundDevice) -> None:
    t = LocalAudioTransport(echo_mode="headphones", block_duration=0.02, latency=0.05)
    await t.start()
    assert fake_sd.input_stream.blocksize == fake_sd.output_stream.blocksize == 320
    assert fake_sd.input_stream.latency_hint == 0.05
    await t.aclose()


async def test_open_failure_is_a_transport_error_and_releases_streams(
    fake_sd: FakeSoundDevice,
) -> None:
    fake_sd.fail_open = "input"
    t = LocalAudioTransport(echo_mode="headphones", sample_rate=8_000)
    with pytest.raises(TransportError, match=r"input device \[0\] Built-in Microphone") as info:
        await t.start()
    assert "default rate 16000 Hz" in str(info.value)
    assert fake_sd.output_stream.closed  # opened before the failure, then released


async def test_device_failure_ends_audio_input_with_an_error(fake_sd: FakeSoundDevice) -> None:
    t = LocalAudioTransport(echo_mode="headphones")
    await t.start()
    fake_sd.input_stream.fail()
    with pytest.raises(TransportError, match="stopped unexpectedly"):
        await take(t, 1)
    with pytest.raises(TransportError, match="stopped unexpectedly"):
        await t.write_audio(AudioFrame.silence(0.01, RATE))
    await t.aclose()


# -------------------------------------------------------------------------------- capture
async def test_capture_frames_come_from_the_audio_thread_with_capture_timestamps(
    fake_sd: FakeSoundDevice, clock: FakeClock
) -> None:
    t = LocalAudioTransport(echo_mode="headphones")
    await t.start()
    speech = synth_speech(0.03, RATE)
    fake_sd.mic.feed(speech)
    t0 = clock.t
    fake_sd.input_stream.run(3)
    frames = await take(t, 3)
    assert AudioFrame.concat(frames).data == speech.data
    assert all(f.format == AudioFormat(RATE, 1) and f.samples_per_channel == BLOCK for f in frames)
    # timestamp = capture time of the first sample = callback time - duration - input latency
    expected = [t0 + (k + 1) * 0.01 - 0.01 - 0.02 for k in range(3)]
    assert [f.timestamp for f in frames] == pytest.approx(expected)
    assert threading.get_ident() not in fake_sd.input_stream.callback_threads
    await t.aclose()


async def test_unconsumed_microphone_audio_is_bounded(
    fake_sd: FakeSoundDevice, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(LocalAudioTransport, "MAX_INPUT_BACKLOG", 0.05)
    t = LocalAudioTransport(echo_mode="headphones")
    await t.start()
    fake_sd.mic.feed(ramp(20 * BLOCK))
    fake_sd.input_stream.run(20)
    await asyncio.sleep(0.05)  # let the event loop receive the frames
    assert t.dropped_input_frames >= 14
    kept = await take(t, 20 - t.dropped_input_frames)
    assert sum(f.duration for f in kept) <= 0.06 + 1e-9
    assert kept[-1].data == ramp(20 * BLOCK).slice(0.19, 0.2).data  # the newest audio is kept
    await t.aclose()


async def test_overflows_and_underflows_are_counted(fake_sd: FakeSoundDevice) -> None:
    t = LocalAudioTransport(echo_mode="headphones")
    await t.start()
    fake_sd.statuses["input"].append(FakeCallbackFlags(input_overflow=True))
    fake_sd.statuses["output"].append(FakeCallbackFlags(output_underflow=True))
    fake_sd.input_stream.run(2)
    fake_sd.output_stream.run(2)
    assert (t.input_overflows, t.output_underflows) == (1, 1)
    await t.aclose()


# ------------------------------------------------------------------------------- playback
async def test_playback_is_in_order_and_gaps_are_silent(fake_sd: FakeSoundDevice) -> None:
    t = LocalAudioTransport(echo_mode="headphones")
    await t.start()
    a, b = ramp(240), ramp(120, start=-1, step=-1)  # 15 ms, then 7.5 ms
    await t.write_audio(a)
    await t.write_audio(b)
    fake_sd.output_stream.run(4)
    played = fake_sd.played_audio()[:, 0]
    silence = np.zeros(4 * BLOCK - 360, dtype=np.int16)
    np.testing.assert_array_equal(played, np.concatenate([a.to_numpy(), b.to_numpy(), silence]))
    assert threading.get_ident() not in fake_sd.output_stream.callback_threads
    await t.aclose()


async def test_write_audio_contract(fake_sd: FakeSoundDevice) -> None:
    t = LocalAudioTransport(echo_mode="headphones", output_channels=2)
    with pytest.raises(TransportError, match="not started"):
        await t.write_audio(AudioFrame.silence(0.01, RATE))
    await t.start()
    with pytest.raises(ValueError, match="16000 Hz"):
        await t.write_audio(AudioFrame.silence(0.01, 24_000))
    await t.write_audio(AudioFrame.empty(RATE))
    mono = ramp(BLOCK)
    await t.write_audio(mono)  # mono is sent to both channels
    fake_sd.output_stream.run(1)
    played = fake_sd.played_audio()
    assert played.shape == (BLOCK, 2)
    np.testing.assert_array_equal(played, np.repeat(mono.to_numpy()[:, None], 2, axis=1))
    await t.aclose()
    await t.write_audio(mono)  # ignored once closed


async def test_write_audio_applies_back_pressure(fake_sd: FakeSoundDevice) -> None:
    t = LocalAudioTransport(echo_mode="headphones", max_buffered=0.05)  # 800 samples
    await t.start()
    frame = ramp(2000)  # 125 ms
    writer = asyncio.create_task(t.write_audio(frame))
    await asyncio.sleep(0.05)
    assert not writer.done()  # 75 ms do not fit yet
    assert t.buffered_duration() == pytest.approx(0.05)
    for _ in range(100):
        if writer.done():
            break
        fake_sd.output_stream.run(2)
        await asyncio.sleep(0.01)
    await asyncio.wait_for(writer, 1.0)
    fake_sd.output_stream.run(15)
    played = fake_sd.played_audio()[:, 0]
    np.testing.assert_array_equal(played[played != 0], frame.to_numpy())  # order kept
    await t.aclose()


async def test_clear_audio_drops_everything_immediately(
    fake_sd: FakeSoundDevice, clock: FakeClock
) -> None:
    t = LocalAudioTransport(echo_mode="headphones", max_buffered=0.1)
    await t.start()
    writer = asyncio.create_task(t.write_audio(tone(440, 0.3, RATE)))  # 0.2 s must wait
    await asyncio.sleep(0.02)
    fake_sd.output_stream.run(1)
    # 90 ms queued + the block inside the device (30 ms latency + 10 ms block)
    assert t.buffered_duration() == pytest.approx(0.09 + 0.04)
    await t.clear_audio()
    assert t.buffered_duration() == pytest.approx(0.04)  # already in the device
    await asyncio.wait_for(writer, 1.0)  # the pending writer gives up
    fake_sd.output_stream.run(5)
    played = fake_sd.played_audio()[:, 0]
    assert played[:BLOCK].any() and not played[BLOCK:].any()
    clock.advance(0.05)
    assert t.buffered_duration() == 0.0
    await t.aclose()


async def test_pause_keeps_queued_audio_and_resume_continues(fake_sd: FakeSoundDevice) -> None:
    t = LocalAudioTransport(echo_mode="headphones")
    await t.start()
    frame = ramp(3 * BLOCK)
    await t.write_audio(frame)
    speakers = fake_sd.output_stream
    speakers.run(1)
    await t.pause_audio()
    assert t.paused
    speakers.run(2)  # silence while paused
    assert t.buffered_duration() >= 0.02  # the rest is still queued
    await t.resume_audio()
    speakers.run(2)
    played, samples = fake_sd.played_audio()[:, 0], frame.to_numpy()
    np.testing.assert_array_equal(played[:BLOCK], samples[:BLOCK])
    assert not played[BLOCK : 3 * BLOCK].any()
    np.testing.assert_array_equal(played[3 * BLOCK : 5 * BLOCK], samples[BLOCK:])
    await t.pause_audio()
    await t.clear_audio()  # nothing left to resume: clearing ends the pause
    assert not t.paused
    await t.aclose()


async def test_buffered_duration_is_queue_plus_device_latency(
    fake_sd: FakeSoundDevice, clock: FakeClock
) -> None:
    t = LocalAudioTransport(echo_mode="headphones")
    await t.start()
    assert t.buffered_duration() == 0.0
    await t.write_audio(AudioFrame.silence(0.5, RATE))
    assert t.buffered_duration() == pytest.approx(0.5)
    fake_sd.output_stream.run(10)
    assert t.buffered_duration() == pytest.approx(0.4 + 0.03 + 0.01)
    fake_sd.output_stream.run(40)
    assert t.buffered_duration() == pytest.approx(0.04)  # all handed to the device
    clock.advance(0.04)
    assert t.buffered_duration() == 0.0
    await asyncio.wait_for(t.wait_for_playout(), 1.0)
    await t.aclose()


# ----------------------------------------------------------------------------------- echo
def test_auto_echo_mode_without_canceller_is_half_duplex(
    fake_sd: FakeSoundDevice, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="voice_agent_next"):
        t = LocalAudioTransport()
    assert t.echo_mode == "auto" and t.active_echo_mode == "half_duplex"
    assert t.echo_canceller is None
    assert "half-duplex" in caplog.text and "headphones" in caplog.text
    assert LocalAudioTransport(echo_mode="headphones").active_echo_mode == "headphones"


async def test_half_duplex_mutes_the_microphone_during_playback_and_tail(
    make_fake_sd: Callable[..., FakeSoundDevice], clock: FakeClock
) -> None:
    fake_sd = make_fake_sd(clock=clock, input_latency=0.0, output_latency=0.0)
    t = LocalAudioTransport(echo_mode="half_duplex", half_duplex_tail=0.3)
    await t.start()
    fake_sd.mic.level = 1000
    mic, speakers = fake_sd.input_stream, fake_sd.output_stream

    async def mic_block() -> AudioFrame:
        mic.run(1)
        (frame,) = await take(t, 1)
        assert frame.samples_per_channel == BLOCK  # muted audio keeps the stream continuous
        return frame

    assert (await mic_block()).to_numpy().all()  # nothing playing: open
    await t.write_audio(AudioFrame.silence(0.05, RATE))
    speakers.run(5)  # the agent is heard until clock + 0.01
    assert not (await mic_block()).to_numpy().any()  # muted while playing
    clock.advance(0.2)
    assert not (await mic_block()).to_numpy().any()  # still within the 0.3 s tail
    clock.advance(0.1)
    assert (await mic_block()).to_numpy().all()  # open again
    await t.aclose()


async def test_echo_canceller_gets_the_played_audio_and_cleans_the_microphone(
    fake_sd: FakeSoundDevice, clock: FakeClock
) -> None:
    ec = FakeEchoCanceller()
    t = LocalAudioTransport(echo_canceller=ec)  # auto -> aec
    assert t.active_echo_mode == "aec" and t.echo_canceller is ec
    await t.start()
    assert ec.resets == 1
    await t.write_audio(ramp(240))
    t0 = clock.t
    fake_sd.output_stream.run(3)
    # the reference is exactly what the speakers got (silence included), timed at the DAC
    reference = np.concatenate([f.to_numpy() for f in ec.render])
    np.testing.assert_array_equal(reference, fake_sd.played_audio()[:, 0])
    assert [f.timestamp for f in ec.render] == pytest.approx(
        [t0 + (k + 1) * 0.01 + 0.03 for k in range(3)]
    )
    # the microphone goes through process_capture and is not muted while the agent speaks
    fake_sd.mic.level = 1000
    fake_sd.input_stream.run(2)
    frames = await take(t, 2)
    assert all((f.to_numpy() == 500).all() for f in frames)
    assert len(ec.capture) == 2
    assert ec.threads["render"] == fake_sd.output_stream.callback_threads
    assert ec.threads["capture"] == fake_sd.input_stream.callback_threads
    await t.aclose()
    assert not ec.closed  # owned by the caller


async def test_default_echo_canceller_is_used_in_auto_mode_and_closed(
    fake_sd: FakeSoundDevice, monkeypatch: pytest.MonkeyPatch
) -> None:
    ec = FakeEchoCanceller()
    monkeypatch.setattr(local, "_default_echo_canceller", lambda: ec)
    t = LocalAudioTransport()
    assert t.active_echo_mode == "aec" and t.echo_canceller is ec
    assert LocalAudioTransport(echo_mode="aec").echo_canceller is ec
    await t.start()
    await t.aclose()
    assert ec.closed  # created by the transport, so released by it


async def test_failing_echo_canceller_falls_back_to_half_duplex(
    fake_sd: FakeSoundDevice, caplog: pytest.LogCaptureFixture
) -> None:
    t = LocalAudioTransport(echo_canceller=BrokenEchoCanceller(), echo_mode="aec")
    await t.start()
    fake_sd.mic.level = 1000
    with caplog.at_level(logging.ERROR, logger="voice_agent_next"):
        fake_sd.input_stream.run(1)
        (frame,) = await take(t, 1)
    assert t.active_echo_mode == "half_duplex"
    assert "half-duplex" in caplog.text
    assert frame.to_numpy().all()  # nothing is playing, so the microphone stays open
    fake_sd.input_stream.run(1)  # the stream keeps running
    assert len(await take(t, 1)) == 1
    await t.aclose()


def test_default_echo_canceller_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    lookup = local.__dict__["_default_echo_canceller"]
    name = "voice_agent_next.audio.aec"
    monkeypatch.setitem(sys.modules, name, None)  # not available
    assert lookup() is None

    module = types.ModuleType(name)
    ec = FakeEchoCanceller()
    calls: list[str] = []

    def create_echo_canceller(mode: str) -> AudioProcessor | None:
        calls.append(mode)
        return ec

    module.create_echo_canceller = create_echo_canceller  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, name, module)
    assert lookup() is ec and calls == ["auto"]

    def broken(mode: str) -> AudioProcessor:
        raise RuntimeError("no APM")

    module.create_echo_canceller = broken  # type: ignore[attr-defined]
    assert lookup() is None


# ---------------------------------------------------------------------------- diagnostics
def test_linux_hint_points_to_the_alsa_pipewire_device(fake_sd: FakeSoundDevice) -> None:
    info = describe_audio_system()
    assert info.portaudio_version == PORTAUDIO_19_6
    assert info.portaudio_release == "PortAudio V19.6.0-devel"
    assert info.default_input is not None and info.default_input.index == 0
    assert info.default_output is not None and info.default_output.index == 1
    (hint,) = info.hints(platform="linux")
    assert hint.startswith("PortAudio V19.6.0-devel has no PulseAudio/PipeWire host API")
    assert "input_device='pipewire'" in hint
    assert info.hints(platform="darwin") == []


def test_hints_for_other_setups() -> None:
    mic = AudioDeviceInfo(0, "Microphone (Realtek)", "MME", 2, 0, 44_100, is_default_input=True)
    out = AudioDeviceInfo(1, "Speakers (Realtek)", "MME", 0, 2, 44_100, is_default_output=True)
    windows = AudioSystemInfo("PortAudio V19.7.0", ("MME", "Windows WASAPI"), (mic, out))
    assert any("WASAPI" in h for h in windows.hints(platform="win32"))
    pulse = AudioSystemInfo("PortAudio V19.8", ("ALSA", "PulseAudio"), (mic, out))
    assert pulse.hints(platform="linux") == []
    no_plugin = AudioSystemInfo("PortAudio V19.6.0", ("ALSA",), (mic, out))
    assert "pipewire-alsa" in no_plugin.hints(platform="linux")[0]
    empty = AudioSystemInfo("PortAudio V19.7.0", ("Core Audio",), ())
    assert [h.split(":")[0] for h in empty.hints(platform="darwin")] == [
        "no input device found",
        "no output device found",
    ]


def test_cli_devices(fake_sd: FakeSoundDevice) -> None:
    result = CliRunner().invoke(app, ["devices"])
    assert result.exit_code == 0, result.output
    for text in ("Built-in Microphone", "pipewire", "JACK", "input", "output", "V19.6.0"):
        assert text in result.output
    result = CliRunner().invoke(app, ["devices", "--json"])
    assert result.exit_code == 0, result.output
    rows = json.loads(result.stdout)
    assert rows[0]["name"] == "Built-in Microphone" and rows[0]["is_default_input"] is True
    assert rows[3]["hostapi"] == "JACK Audio Connection Kit"


def test_cli_devices_without_sounddevice(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "sounddevice", None)
    result = CliRunner().invoke(app, ["devices"])
    assert result.exit_code == 1
    assert "voice-agent-next[audio]" in result.output


def test_cli_doctor_reports_the_audio_setup(fake_sd: FakeSoundDevice) -> None:
    result = CliRunner().invoke(app, ["doctor"])
    assert result.exit_code == 0, result.output
    for text in ("portaudio", "V19.6.0", "ALSA", "Built-in Microphone", "Built-in Speakers"):
        assert text in result.output


# -------------------------------------------------------------------- session, end to end
async def test_session_with_the_mock_engine_over_fake_devices(
    make_fake_sd: Callable[..., FakeSoundDevice],
) -> None:
    """What `van run --engine mock` does, with fake real-time devices."""
    fake_sd = make_fake_sd(realtime=True)
    fake_sd.mic.feed(AudioFrame.silence(0.2, RATE))
    fake_sd.mic.feed(synth_speech(0.5, RATE))  # then silence
    session = AgentSession(MockEngine(transcripts=["hello there"], responses=["Hi, I hear you."]))
    heard: list[str] = []
    session.on("user_transcript", lambda ev: heard.append(ev.text))
    transport = LocalAudioTransport(echo_mode="headphones")
    try:
        await session.start(Agent("be brief"), transport)
        await wait_for(lambda: heard == ["hello there"])
        await wait_for(lambda: int(np.abs(fake_sd.played_audio()).max(initial=0)) > 1000)
    finally:
        await session.aclose()
    assert fake_sd.input_stream.closed and fake_sd.output_stream.closed


@pytest.mark.parametrize("echo_mode", ["headphones", "half_duplex"])
async def test_half_duplex_prevents_self_interruption_through_the_speakers(
    make_fake_sd: Callable[..., FakeSoundDevice], echo_mode: str
) -> None:
    """The microphone hears the speakers: without echo handling the agent interrupts
    itself; half-duplex lets the greeting finish and the user speak afterwards."""
    fake_sd = make_fake_sd(realtime=True, echo=True, input_latency=0.0, output_latency=0.0)
    engine = MockEngine(transcripts=["what time is it"], chars_per_second=30.0)
    session = AgentSession(engine)
    interrupted: list[Any] = []
    heard: list[str] = []
    session.on("interrupted", interrupted.append)
    session.on("user_transcript", lambda ev: heard.append(ev.text))
    transport = LocalAudioTransport(echo_mode=echo_mode, half_duplex_tail=0.2)  # type: ignore[arg-type]
    try:
        await session.start(Agent("be brief", greeting="Hello! How can I help you?"), transport)
        await wait_for(lambda: session.agent_state == AgentState.SPEAKING)
        if echo_mode == "headphones":  # the agent's own voice triggers barge-in
            await wait_for(lambda: bool(interrupted))
            return
        await wait_for(lambda: session.agent_state == AgentState.LISTENING)
        assert not interrupted and not heard
        await asyncio.sleep(0.3)  # past the tail
        fake_sd.mic.feed(synth_speech(0.5, RATE))
        await wait_for(lambda: heard == ["what time is it"])
    finally:
        await session.aclose()


# ------------------------------------------------------------------------ real hardware
@pytest.mark.audio_device
async def test_real_devices_record_and_play() -> None:
    """Smoke test on real hardware (``pytest -m audio_device``): records 1 s from the
    default microphone, then plays a short 440 Hz tone on the default speakers."""
    try:
        import_sounddevice()
    except MissingDependencyError as exc:
        pytest.skip(str(exc))
    transport = LocalAudioTransport(echo_mode="headphones")
    async with transport:
        recorded: list[AudioFrame] = []

        async def record() -> None:
            async for frame in transport.audio_input():
                recorded.append(frame)
                if sum(f.duration for f in recorded) >= 1.0:
                    return

        await asyncio.wait_for(record(), timeout=5.0)
        audio = AudioFrame.concat(recorded)
        assert audio.duration >= 1.0
        assert audio.format == transport.input_format
        fmt = transport.output_format
        beep = tone(440.0, 0.5, fmt.sample_rate, amplitude=0.2).to_channels(fmt.channels)
        await transport.write_audio(beep)
        assert transport.buffered_duration() > 0.0
        await asyncio.wait_for(transport.wait_for_playout(), timeout=5.0)
