"""``van doctor`` tests: every section on a faked machine, the signal analysis on synthetic
signals, the JSON schema and exit codes. No audio device and no network is used: the
``sounddevice`` module, the audio I/O of the interactive checks and the network probe are
fakes."""

from __future__ import annotations

import importlib.machinery
import json
import re
import socket
import sys
import threading
import types
from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from typer.testing import CliRunner

from voice_agent_next import doctor as dx
from voice_agent_next import presets
from voice_agent_next.cli import doctor as cli_doctor
from voice_agent_next.cli.main import app
from voice_agent_next.transports.local import AudioDeviceInfo

ANSI_STYLE = re.compile(r"\x1b\[[0-9;]*m")
PORTAUDIO = "PortAudio V19.7.0-devel, revision 147dd722548358763a8b649b3e4b41dfffbcfbb6"
RNG = np.random.default_rng(1234)


# ------------------------------------------------------------------------ fakes
def _device(
    name: str,
    hostapi: int = 0,
    inputs: int = 0,
    outputs: int = 0,
    rate: float = 48_000,
    latency: float = 0.01,
) -> dict[str, Any]:
    return {
        "name": name,
        "hostapi": hostapi,
        "max_input_channels": inputs,
        "max_output_channels": outputs,
        "default_low_input_latency": latency if inputs else 0.0,
        "default_low_output_latency": latency if outputs else 0.0,
        "default_samplerate": rate,
    }


LINUX_DEVICES = [
    _device("HDA Intel PCH: ALC257 Analog (hw:0,0)", inputs=2, outputs=2),  # 0
    _device("pipewire", inputs=64, outputs=64),  # 1: default in + out
    _device("default", inputs=64, outputs=64),  # 2
]


class FakeSoundDevice:
    """``sounddevice`` with a device list, format checks, and scripted recordings."""

    def __init__(
        self,
        devices: list[dict[str, Any]] | None = None,
        hostapis: tuple[str, ...] = ("ALSA",),
        default: tuple[int, int] = (1, 1),
        rates: tuple[int, ...] = (16_000, 44_100, 48_000),
        mic: Callable[[int, int], npt.NDArray[np.int16]] | None = None,
        room: Callable[[npt.NDArray[np.int16], int], npt.NDArray[np.int16]] | None = None,
    ) -> None:
        self.devices = [
            dict(d, index=i) for i, d in enumerate(LINUX_DEVICES if devices is None else devices)
        ]
        self.hostapis = hostapis
        self.default = default
        self.rates = rates
        self.mic = mic or (lambda n, rate: np.zeros(n, np.int16))
        self.room = room or (lambda played, rate: np.zeros_like(played))
        self.checked: list[tuple[str, Any, int]] = []
        self.played: list[tuple[npt.NDArray[np.int16], int, Any]] = []

    def module(self) -> types.ModuleType:
        mod = types.ModuleType("sounddevice")
        mod.__spec__ = importlib.machinery.ModuleSpec("sounddevice", None)
        mod.__version__ = "0.5.6"  # type: ignore[attr-defined]
        mod.query_devices = self.query_devices  # type: ignore[attr-defined]
        mod.query_hostapis = lambda: [{"name": n} for n in self.hostapis]  # type: ignore[attr-defined]
        mod.get_portaudio_version = lambda: (1246976, PORTAUDIO)  # type: ignore[attr-defined]
        mod.check_input_settings = self._checker("input")  # type: ignore[attr-defined]
        mod.check_output_settings = self._checker("output")  # type: ignore[attr-defined]
        mod.InputStream = self.input_stream  # type: ignore[attr-defined]
        mod.playrec = self.playrec  # type: ignore[attr-defined]
        mod.wait = lambda: None  # type: ignore[attr-defined]
        mod.get_stream = lambda: types.SimpleNamespace(latency=(0.01, 0.02))  # type: ignore[attr-defined]
        return mod

    def query_devices(self, device: int | None = None, kind: str | None = None) -> Any:
        if device is None and kind is None:
            return [dict(d) for d in self.devices]
        index = self.default[0 if kind == "input" else 1] if device is None else device
        if index < 0:
            raise RuntimeError("no default device")
        return dict(self.devices[index])

    def _checker(self, kind: str) -> Callable[..., None]:
        def check(*, device: Any, samplerate: int, channels: int, dtype: str) -> None:
            self.checked.append((kind, device, samplerate))
            if samplerate not in self.rates:
                raise RuntimeError("Invalid sample rate")

        return check

    def input_stream(self, *, device: Any, samplerate: int, channels: int, dtype: str,
                     callback: Callable[..., None]) -> Any:  # fmt: skip
        fake = self

        class Stream:
            def __enter__(self) -> Any:
                def run() -> None:
                    data = fake.mic(round(samplerate * 10), samplerate)  # plenty
                    for i in range(0, data.size, 480):
                        callback(data[i : i + 480, None], 480, None, None)

                threading.Thread(target=run, daemon=True).start()
                return self

            def __exit__(self, *exc: Any) -> None:
                pass

        return Stream()

    def playrec(self, data: npt.NDArray[np.int16], *, samplerate: int, channels: int,
                dtype: str, device: Any) -> npt.NDArray[np.int16]:  # fmt: skip
        played = data.reshape(-1).astype(np.int16)
        self.played.append((played, samplerate, device))
        return self.room(played, samplerate).reshape(-1, 1)


@pytest.fixture
def make_sd(monkeypatch: pytest.MonkeyPatch) -> Callable[..., FakeSoundDevice]:
    def make(**kwargs: Any) -> FakeSoundDevice:
        backend = FakeSoundDevice(**kwargs)
        monkeypatch.setitem(sys.modules, "sounddevice", backend.module())
        return backend

    return make


def room(
    delay: float, gain: float, noise_dbfs: float = -70.0
) -> Callable[[npt.NDArray[np.int16], int], npt.NDArray[np.int16]]:
    """A room: the microphone hears the playback ``delay`` s later, ``gain`` times as loud."""

    def hear(played: npt.NDArray[np.int16], rate: int) -> npt.NDArray[np.int16]:
        shift = round(delay * rate)
        out = np.zeros(played.size, np.float64)
        out[shift:] = played[: played.size - shift].astype(np.float64) * gain
        out += noise(played.size, noise_dbfs)
        return np.clip(np.round(out), -32768, 32767).astype(np.int16)

    return hear


def noise(n: int, dbfs: float) -> npt.NDArray[np.float64]:
    return RNG.standard_normal(n) * 32768 * 10 ** (dbfs / 20)


def speech_like(rate: int, seconds: float, speech_dbfs: float, noise_dbfs: float) -> Any:
    """Alternating 0.4 s bursts of tones (speech) and 0.4 s of background noise."""
    n = round(seconds * rate)
    t = np.arange(n) / rate
    bursts = (np.floor(t / 0.4) % 2 == 0).astype(np.float64)
    amplitude = 32768 * 10 ** (speech_dbfs / 20) * np.sqrt(2)
    signal = bursts * amplitude * np.sin(2 * np.pi * 220 * t) + noise(n, noise_dbfs)
    return np.clip(np.round(signal), -32768, 32767).astype(np.int16)


def cli(*args: str, env: dict[str, str] | None = None) -> tuple[int, str]:
    result = CliRunner().invoke(app, list(args), env={"COLUMNS": "200", **(env or {})})
    return result.exit_code, ANSI_STYLE.sub("", result.output)


def cli_json(*args: str) -> tuple[int, dict[str, Any]]:
    result = CliRunner().invoke(app, ["doctor", "--json", *args])
    return result.exit_code, json.loads(result.stdout)


def by_name(checks: list[dx.Check]) -> dict[str, list[dx.Check]]:
    out: dict[str, list[dx.Check]] = {}
    for c in checks:
        out.setdefault(c.name, []).append(c)
    return out


# ------------------------------------------------------------------ level meter
RATE = 16_000


def test_level_stats_of_a_known_signal() -> None:
    samples = speech_like(RATE, 4.0, speech_dbfs=-20.0, noise_dbfs=-60.0)
    stats = dx.level_stats(samples, RATE)
    assert stats.duration == pytest.approx(4.0)
    assert stats.speech_dbfs == pytest.approx(-20.0, abs=1.0)
    assert stats.noise_floor_dbfs == pytest.approx(-60.0, abs=2.0)
    assert stats.peak_dbfs == pytest.approx(-17.0, abs=1.0)  # sine peak = rms + 3 dB
    assert stats.clipped_ratio == 0.0
    assert dx.level_verdicts(stats)[0][0] == "ok"


def test_silence_is_a_failure() -> None:
    verdicts = dx.level_verdicts(dx.level_stats(np.zeros(RATE * 2, np.int16), RATE))
    assert [(v[0], v[1].split(":")[0]) for v in verdicts] == [("fail", "silence")]
    stats = dx.level_stats(np.zeros(0, np.int16), RATE)
    assert stats.peak_dbfs == -120.0 and dx.level_verdicts(stats)[0][0] == "fail"


def test_clipping_is_detected() -> None:
    samples = speech_like(RATE, 3.0, speech_dbfs=-20.0, noise_dbfs=-60.0)
    samples[::50] = 32767  # 2% of the samples at full scale
    verdicts = dx.level_verdicts(dx.level_stats(samples, RATE))
    assert ("warn", "clipping") in [(s, v.split(":")[0]) for s, v, _ in verdicts]


def verdict_names(samples: npt.NDArray[np.int16]) -> list[str]:
    return [v.split(":")[0] for _, v, _ in dx.level_verdicts(dx.level_stats(samples, RATE))]


@pytest.mark.parametrize(
    ("speech", "floor", "expected"),
    [(-20.0, -35.0, ["noisy"]), (-55.0, -75.0, ["quiet"])],
)
def test_noise_and_quiet_verdicts(speech: float, floor: float, expected: list[str]) -> None:
    samples = speech_like(RATE, 3.0, speech_dbfs=speech, noise_dbfs=floor)
    assert verdict_names(samples) == expected


def test_noise_without_speech() -> None:
    samples = np.round(noise(RATE * 3, -35.0)).astype(np.int16)
    assert verdict_names(samples) == ["noisy", "no speech detected"]


def test_dc_offset_is_reported() -> None:
    samples = speech_like(RATE, 2.0, speech_dbfs=-20.0, noise_dbfs=-60.0).astype(np.int32) + 1500
    stats = dx.level_stats(samples.astype(np.int16), RATE)
    assert any(v.startswith("DC offset") for _, v, _ in dx.level_verdicts(stats))


# ------------------------------------------------------------------- echo delay
@pytest.mark.parametrize("rate", [16_000, 44_100, 48_000])
@pytest.mark.parametrize("delay", [0.0, 0.0371, 0.215])
def test_estimate_delay_finds_a_delayed_attenuated_chirp(rate: int, delay: float) -> None:
    ref = dx.chirp(rate)
    rec = np.zeros(ref.size + rate, np.float64)
    shift = round(delay * rate)
    rec[shift : shift + ref.size] = ref * 0.05
    rec += noise(rec.size, -60.0)
    lag, confidence = dx.estimate_delay(ref, rec)
    assert abs(lag - shift) <= 1
    assert confidence > 0.8


def test_estimate_delay_without_the_signal_has_low_confidence() -> None:
    ref = dx.chirp(RATE)
    _, confidence = dx.estimate_delay(ref, noise(RATE * 2, -40.0))
    assert confidence < 0.2
    assert dx.estimate_delay(np.zeros(10), np.ones(10)) == (0, 0.0)


def test_chirp_is_bounded_and_faded() -> None:
    c = dx.chirp(48_000, duration=0.5, amplitude=0.5)
    assert c.dtype == np.int16 and c.size == 24_000
    assert np.abs(c).max() <= 16_384 and abs(int(c[0])) < 100 and abs(int(c[-1])) < 100


def _echo_scene(delay: float, gain: float, rate: int = 48_000) -> dx.EchoResult:
    lead, probe = 0.3, 0.5
    played = np.concatenate(
        [np.zeros(round(lead * rate), np.int16), dx.chirp(rate, probe), np.zeros(rate, np.int16)]
    )
    recorded = room(delay, gain)(played, rate)
    return dx.analyze_echo(played, recorded, rate, lead=lead, probe_duration=probe)


def test_analyze_echo_measures_delay_and_return_loss() -> None:
    result = _echo_scene(delay=0.120, gain=0.1)  # -20 dB
    assert result.detected
    assert result.delay_ms == pytest.approx(120.0, abs=0.1)
    assert result.erl_db == pytest.approx(20.0, abs=0.5)
    assert result.noise_dbfs == pytest.approx(-70.0, abs=2.0)


def test_analyze_echo_without_echo() -> None:
    result = _echo_scene(delay=0.1, gain=0.0)
    assert not result.detected and result.delay_ms is None and result.erl_db is None


def test_echo_recommendations() -> None:
    none = dx.echo_recommendation(_echo_scene(0.1, 0.0), aec_available=True)
    assert none[0] == "ok" and none[3] == {"echo_mode": "headphones"}
    normal = dx.echo_recommendation(_echo_scene(0.08, 0.1), aec_available=True)
    assert normal[0] == "ok" and normal[3] == {"echo_mode": "aec", "delay_ms": 80}
    assert "voice-agent-next[aec]" not in normal[2]
    missing = dx.echo_recommendation(_echo_scene(0.08, 0.1), aec_available=False)
    assert "voice-agent-next[aec]" in missing[2]
    loud = dx.echo_recommendation(_echo_scene(0.08, 0.8), aec_available=True)
    assert loud[0] == "warn" and "lower the speaker volume" in loud[2]
    slow = dx.echo_recommendation(_echo_scene(0.62, 0.1), aec_available=True)
    assert slow[0] == "warn" and slow[3] == {"echo_mode": "half_duplex", "half_duplex_tail": 0.9}


# ---------------------------------------------------------------- latency probe
def test_analyze_latency_with_jitter() -> None:
    rate = 48_000
    signal, onsets, pulse = dx.pulse_train(rate, count=5, interval=0.5)
    delays = [0.100, 0.104, 0.098, 0.101, 0.097]
    recorded = noise(signal.size, -70.0)
    for onset, d in zip(onsets, delays, strict=True):
        start = onset + round(d * rate)
        recorded[start : start + pulse.size] += pulse * 0.2
    result = dx.analyze_latency(recorded.astype(np.int16), rate, onsets, pulse)
    assert result.delays_ms == pytest.approx([d * 1000 for d in delays], abs=0.05)
    assert result.mean_ms == pytest.approx(100.0, abs=0.1)
    assert result.min_ms == pytest.approx(97.0, abs=0.05)
    assert result.jitter_ms == pytest.approx(float(np.std(delays)) * 1000, abs=0.05)


def test_analyze_latency_hears_nothing() -> None:
    signal, onsets, pulse = dx.pulse_train(RATE)
    result = dx.analyze_latency(np.round(noise(signal.size, -50)), RATE, onsets, pulse)
    assert result.delays_ms == [] and result.mean_ms is None and result.pulses == 5


# ------------------------------------------------------- interactive checks (fake I/O)
MIC = AudioDeviceInfo(1, "USB Mic", "ALSA", 1, 0, 16_000.0, 0.01, 0.0, True, False)
SPK = AudioDeviceInfo(2, "Speakers", "ALSA", 0, 2, 48_000.0, 0.0, 0.02, False, True)


class FakeIO:
    def __init__(
        self,
        mic: npt.NDArray[np.int16] | None = None,
        room: Callable[[npt.NDArray[np.int16], int], npt.NDArray[np.int16]] | None = None,
        rates: tuple[int, ...] | None = None,
    ) -> None:
        self.mic = mic
        self.room = room
        self.rates = rates
        self.calls: list[tuple[str, int]] = []

    def record(self, seconds: float, *, rate: int, device: int | None, on_level: Any) -> Any:
        self.calls.append(("record", rate))
        data = self.mic if self.mic is not None else np.zeros(round(seconds * rate), np.int16)
        if on_level is not None:
            on_level(seconds, -20.0, -10.0)
        return data

    def playrec(self, signal: Any, *, rate: int, input_device: Any, output_device: Any) -> Any:
        self.calls.append(("playrec", rate))
        if self.rates is not None and rate not in self.rates:
            raise RuntimeError("Invalid sample rate")
        assert self.room is not None
        return self.room(signal, rate), 0.03


def test_mic_check_verdicts() -> None:
    good = FakeIO(mic=speech_like(16_000, 3.0, -22.0, -65.0))
    levels: list[float] = []
    checks = dx.mic_check(
        good, device=MIC, seconds=3.0, on_level=lambda t, db, peak: levels.append(db)
    )
    assert [c.status for c in checks] == ["info", "ok"]
    assert checks[0].data["rate"] == 16_000 and levels == [-20.0]
    silent = dx.mic_check(FakeIO(), device=MIC, seconds=2.0)
    assert silent[-1].status == "fail" and "silence" in silent[-1].value
    short = dx.mic_check(FakeIO(mic=np.zeros(100, np.int16)), device=MIC, seconds=2.0)
    assert short[0].status == "fail" and "recorded only" in short[0].value


def test_echo_check_falls_back_to_a_shared_rate() -> None:
    io = FakeIO(room=room(0.150, 0.1), rates=(16_000,))
    checks = dx.echo_check(io, input_device=MIC, output_device=SPK, aec_available=True)
    assert io.calls == [("playrec", 48_000), ("playrec", 16_000)]
    path, advice = checks
    assert path.status == "ok" and path.data["rate"] == 16_000
    assert path.data["delay_ms"] == pytest.approx(150.0, abs=0.1)
    assert path.data["erl_db"] == pytest.approx(20.0, abs=0.5)
    assert path.data["reported_latency_ms"] == 30.0
    assert advice.data["settings"] == {"echo_mode": "aec", "delay_ms": 150}


def test_echo_check_without_a_duplex_stream() -> None:
    io = FakeIO(room=room(0.1, 0.1), rates=())
    with pytest.raises(RuntimeError, match="cannot open a duplex stream"):
        dx.echo_check(io, input_device=MIC, output_device=SPK)


def test_latency_check() -> None:
    ok = dx.latency_check(FakeIO(room=room(0.09, 0.3)), input_device=MIC, output_device=SPK)
    assert ok[0].status == "ok" and ok[0].data["mean_ms"] == pytest.approx(90.0, abs=0.1)
    assert "PortAudio reports 30 ms" in ok[0].value
    slow = dx.latency_check(FakeIO(room=room(0.3, 0.3)), input_device=MIC, output_device=SPK)
    assert slow[0].status == "warn" and "Bluetooth" in (slow[0].hint or "")
    deaf = dx.latency_check(FakeIO(room=room(0.09, 0.0)), input_device=MIC, output_device=SPK)
    assert deaf[0].status == "fail" and "heard 0 of 5 pulses" in deaf[0].value


def test_sounddevice_io(make_sd: Callable[..., FakeSoundDevice]) -> None:
    fake = make_sd(mic=lambda n, rate: np.full(n, 1000, np.int16), room=room(0.05, 0.5))
    io = dx.SoundDeviceIO()
    levels: list[tuple[float, float, float]] = []
    samples = io.record(0.5, rate=16_000, device=1, on_level=lambda *a: levels.append(a))
    assert samples.size == 8_000 and samples.dtype == np.int16 and int(samples[0]) == 1000
    assert levels and levels[0][1] == pytest.approx(20 * np.log10(1000 / 32768), abs=0.01)
    recorded, reported = io.playrec(dx.chirp(16_000), rate=16_000, input_device=1, output_device=2)
    assert recorded.shape == (8_000,) and reported == pytest.approx(0.03)
    assert fake.played[0][1:] == (16_000, (1, 2))


# ------------------------------------------------------------------ audio section
def test_audio_section_on_linux_with_pipewire(make_sd: Callable[..., FakeSoundDevice]) -> None:
    make_sd()
    checks = dx.audio_checks(
        platform="linux",
        environ={"XDG_RUNTIME_DIR": "/run/user/1000"},
        path_exists=lambda p: p.replace("\\", "/") == "/run/user/1000/pipewire-0",
    )
    rows = by_name(checks)
    assert rows["portaudio"][0].data == {"version": "PortAudio V19.7.0-devel"}
    assert rows["host API ALSA"][0].value.startswith("3 input, 3 output device(s)")
    assert rows["sound server"][0].status == "ok" and "PipeWire" in rows["sound server"][0].value
    # PortAudio 19.7 without a Pulse host API: the existing hint names the routed devices
    assert any("'pipewire'" in c.value for c in rows["audio setup"])
    default_in = rows["default input"][0]
    assert default_in.status == "ok" and default_in.data["name"] == "pipewire"
    assert default_in.data["supported_rates"] == [16_000, 44_100, 48_000]
    assert rows["input sample rates"][0].value == "16000, 44100, 48000"
    assert "rate mismatch" not in rows


def test_audio_section_warns_about_bluetooth_hfp_and_raw_hw(
    make_sd: Callable[..., FakeSoundDevice],
) -> None:
    make_sd(
        devices=[
            _device("WH-1000XM4 Hands-Free AG Audio", inputs=1, outputs=1, rate=16_000),
            _device("HDA Intel PCH: ALC257 Analog (hw:0,0)", outputs=2, rate=44_100, latency=0.2),
        ],
        default=(0, 1),
    )
    checks = dx.audio_checks(platform="linux", environ={}, path_exists=lambda p: False)
    rows = by_name(checks)
    assert rows["sound server"][0].status == "info"
    warnings = [c.value for c in checks if c.status == "warn"]
    assert any("Bluetooth headset in hands-free mode" in w for w in warnings)
    assert any("raw ALSA 'hw:' device" in w for w in warnings)
    assert any("200 ms of output latency" in w for w in warnings)
    assert "input 16000 Hz, output 44100 Hz" in rows["rate mismatch"][0].value


def test_audio_section_on_windows(make_sd: Callable[..., FakeSoundDevice]) -> None:
    make_sd(
        devices=[
            _device("Microphone (Realtek)", hostapi=0, inputs=2),
            _device("Speakers (Realtek)", hostapi=0, outputs=2),
            _device("Speakers (Realtek)", hostapi=1, outputs=2),
            _device("Speakers (Realtek)", hostapi=2, outputs=2),
        ],
        hostapis=("MME", "Windows WASAPI", "Windows WDM-KS"),
        default=(0, 3),
    )
    checks = dx.audio_checks(platform="win32")
    rows = by_name(checks)
    assert rows["host API Windows WASAPI"][0].status == "ok"
    assert "adds 50-100 ms" in rows["host API MME"][0].value
    assert "sound server" not in rows
    assert any("MME, which adds latency" in c.value for c in rows["audio setup"])
    assert any("exclusive mode" in c.value for c in rows["default output"] if c.status == "warn")


def test_audio_section_on_macos_without_core_audio(
    make_sd: Callable[..., FakeSoundDevice],
) -> None:
    make_sd(hostapis=("JACK Audio Connection Kit",), rates=())
    rows = by_name(dx.audio_checks(platform="darwin"))
    assert rows["host APIs"][0].status == "warn" and "Core Audio" in rows["host APIs"][0].value
    assert rows["input sample rates"][0].status == "warn"


def test_audio_section_without_devices_or_sounddevice(
    make_sd: Callable[..., FakeSoundDevice], monkeypatch: pytest.MonkeyPatch
) -> None:
    make_sd(devices=[], default=(-1, -1))
    rows = by_name(dx.audio_checks(platform="linux", environ={}))
    assert rows["default input"][0].status == "warn" and rows["default input"][0].value == "none"
    assert any("no input device found" in c.value for c in rows["audio setup"])
    monkeypatch.setitem(sys.modules, "sounddevice", None)
    (row,) = dx.audio_checks()
    assert row.status == "warn" and row.hint == "pip install 'voice-agent-next[audio]'"


def test_device_warnings_for_bluetooth_a2dp() -> None:
    a2dp = AudioDeviceInfo(0, "AirPods Pro", "Core Audio", 0, 2, 48_000.0, 0.0, 0.01)
    (warning,) = dx.device_warnings(a2dp, "output", "darwin")
    assert warning[0] == "info" and "Bluetooth" in warning[1]
    narrow = AudioDeviceInfo(0, "Line In", "ALSA", 1, 0, 8_000.0, 0.01, 0.0)
    assert any("below the 16 kHz" in w[1] for w in dx.device_warnings(narrow, "input", "linux"))


# ------------------------------------------------- hardware, presets, models, system
def test_hardware_section_splits_the_gpu_hint() -> None:
    checks = dx.hardware_checks(
        lambda: [
            ("NVIDIA GPU", "none"),
            ("onnxruntime device=auto", "cpu; to use the GPU: pip install onnxruntime-gpu"),
            ("ctranslate2", "error: broken"),
        ]
    )
    assert [c.status for c in checks] == ["ok", "warn", "warn"]
    assert (
        checks[1].value == "cpu" and checks[1].hint == "to use the GPU: pip install onnxruntime-gpu"
    )


def _env(**kwargs: Any) -> presets.Environment:
    defaults: dict[str, Any] = {
        "platform": "linux",
        "environ": {},
        "installed": lambda module: True,
        "nvidia_gpus": lambda: (),
        "apple_silicon": lambda: None,
        "cuda_backend": lambda: None,
        "ollama_models": lambda url: None,
    }
    return presets.Environment(**{**defaults, **kwargs})


def test_presets_section() -> None:
    keys = {"DEEPGRAM_API_KEY": "x", "GROQ_API_KEY": "x", "CARTESIA_API_KEY": "x"}
    checks = dx.preset_checks(_env(environ=keys))
    summary = checks[0]
    assert summary.status == "ok" and "cloud-fast" in summary.data["ready"]
    assert f"`van run` uses {summary.data['auto']}" in summary.value
    rows = {c.name: c for c in checks[1:]}
    assert rows["cloud-fast"].status == "ok"
    assert rows["apple"].status == "info" and rows["apple"].data["fixes"] == []
    none = dx.preset_checks(_env(installed=lambda module: False))
    assert none[0].status == "warn" and none[0].data == {"ready": [], "auto": None}


def test_models_section(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VAN_CACHE_DIR", str(tmp_path / "van"))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hf"))
    monkeypatch.setenv("VAN_OFFLINE", "1")
    rows = by_name(dx.model_checks())
    assert rows["model cache"][0].value == str(tmp_path / "van")
    catalog = rows["catalog"][0]
    assert catalog.data["cached"] == [] and catalog.data["models"] > 0
    assert catalog.value.startswith(f"0 of {catalog.data['models']} models cached")
    assert "offline mode" in rows


def test_system_section() -> None:
    rows = by_name(dx.system_checks(environ={"OPENAI_API_KEY": "sk-secret"}))
    assert rows["python"][0].status == "ok"
    assert rows["API keys set"][0].data == {"set": ["OPENAI_API_KEY"]}
    assert "sk-secret" not in json.dumps([c.to_dict() for cs in rows.values() for c in cs])


def test_guarded_turns_exceptions_into_failures() -> None:
    def boom() -> list[dx.Check]:
        raise ValueError("kaput")

    (row,) = dx.guarded("models", boom)
    assert row.status == "fail" and row.value == "error: ValueError: kaput"


# ------------------------------------------------------------------------- network
def test_cloud_endpoints_follow_the_api_keys() -> None:
    environ = {
        "ASSEMBLYAI_API_KEY": "x",
        "ELEVEN_API_KEY": "x",
        "AZURE_OPENAI_API_KEY": "x",
        "AZURE_OPENAI_ENDPOINT": "https://my-res.openai.azure.com",
    }
    eps = dx.cloud_endpoints(environ, extra=["wss://example.org:8443/v1", "10.0.0.5"])
    assert [(e.label, e.host, e.port) for e in eps] == [
        ("assemblyai", "streaming.assemblyai.com", 443),
        ("assemblyai (us)", "streaming.us.assemblyai.com", 443),
        ("assemblyai (eu)", "streaming.eu.assemblyai.com", 443),
        ("elevenlabs", "api.elevenlabs.io", 443),
        ("elevenlabs (us)", "api.us.elevenlabs.io", 443),
        ("azure_openai", "my-res.openai.azure.com", 443),
        ("wss://example.org:8443/v1", "example.org", 8443),
        ("10.0.0.5", "10.0.0.5", 443),
    ]
    assert dx.cloud_endpoints({}) == []


def test_every_endpoint_belongs_to_a_registered_cloud_provider() -> None:
    from voice_agent_next.registry import list_providers

    cloud = {s.name for s in list_providers() if not s.local}
    assert set(dx._CLOUD_HOSTS) <= cloud and set(dx._REGIONS) <= set(dx._CLOUD_HOSTS)


def test_network_checks_with_a_fake_probe() -> None:
    timings = {
        "streaming.assemblyai.com": 90.0,
        "streaming.us.assemblyai.com": 95.0,
        "streaming.eu.assemblyai.com": 12.0,
        "api.deepgram.com": 30.0,
    }

    def probe(host: str, port: int, timeout: float) -> dx.ProbeResult:
        if host == "down.example":
            return dx.ProbeResult(host, port, dns_ms=3.0, error="TCP connect failed: refused")
        ms = timings[host]
        return dx.ProbeResult(host, port, "1.2.3.4", 2.0, ms, ms * 2, "TLSv1.3")

    eps = dx.cloud_endpoints({"ASSEMBLYAI_API_KEY": "x", "DEEPGRAM_API_KEY": "x"})
    eps.append(dx.Endpoint("down.example", "down.example"))
    rows = by_name(dx.network_checks(eps, probe=probe, slow_ms=80.0))
    assert rows["deepgram"][0].status == "ok" and "TLS 60 ms (TLSv1.3)" in rows["deepgram"][0].value
    assert rows["assemblyai"][0].status == "warn"  # 90 ms > 80 ms
    assert rows["assemblyai (eu)"][0].data["connect_ms"] == 12.0
    assert rows["assemblyai region"][0].hint == 'assemblyai: region="eu"'
    assert rows["down.example"][0].status == "fail"
    (skip,) = dx.network_checks([])
    assert skip.status == "skip"


def test_probe_endpoint_reports_the_failing_stage() -> None:
    assert "DNS lookup failed" in (
        dx.probe_endpoint("does-not-exist.invalid", 443, 2.0).error or ""
    )
    with socket.socket() as server:  # accepts, then closes: TLS fails, TCP worked
        server.bind(("127.0.0.1", 0))
        server.listen()
        port = server.getsockname()[1]

        def serve() -> None:
            conn, _ = server.accept()
            conn.close()

        t = threading.Thread(target=serve, daemon=True)
        t.start()
        result = dx.probe_endpoint("127.0.0.1", port, 5.0)
        t.join(5)
    assert result.connect_ms is not None and result.dns_ms is not None
    assert (result.error or "").startswith("TLS handshake failed")


# ----------------------------------------------------------------------------- CLI
@pytest.fixture
def quiet_machine(monkeypatch: pytest.MonkeyPatch, make_sd: Callable[..., FakeSoundDevice]) -> FakeSoundDevice:  # fmt: skip
    """Fake audio, hardware and preset environment: the doctor runs offline and fast."""
    monkeypatch.setattr(presets, "current_environment", lambda: _env())
    monkeypatch.setattr(dx, "hardware_checks", lambda: [dx.Check("hardware", "GPU", "ok", "none")])
    return make_sd()


def test_cli_json_schema_and_default_sections(quiet_machine: FakeSoundDevice) -> None:
    code, report = cli_json()
    assert set(report) == {"schema", "version", "platform", "exit_code", "summary", "sections", "checks"}  # fmt: skip
    assert report["schema"] == dx.SCHEMA and report["exit_code"] == code == 0
    assert report["sections"] == list(dx.DEFAULT_SECTIONS)
    assert set(report["summary"]) == {"ok", "info", "warn", "fail", "skip"}
    for check in report["checks"]:
        assert set(check) == {"section", "name", "status", "value", "hint", "data"}
        assert check["status"] in ("ok", "info", "warn", "fail", "skip")
    assert sum(report["summary"].values()) == len(report["checks"])


def test_cli_exit_codes(quiet_machine: FakeSoundDevice, monkeypatch: pytest.MonkeyPatch) -> None:
    code, report = cli_json("--only", "audio")
    assert code == 0 and report["summary"]["warn"] > 0  # PortAudio 19.7 without Pulse
    code, report = cli_json("--only", "audio", "--strict")
    assert code == 1 and report["exit_code"] == 1
    code, output = cli("doctor", "--only", "bogus")
    assert code == 2 and "unknown section" in output
    code, output = cli("doctor", "--only", "mic", "--duration", "0")
    assert code == 2

    def failing() -> list[dx.Check]:
        raise RuntimeError("no GPU driver")

    monkeypatch.setattr(dx, "model_checks", failing)
    code, report = cli_json("--only", "models")
    assert code == 1 and report["checks"][0]["value"] == "error: RuntimeError: no GPU driver"


def test_cli_text_output(quiet_machine: FakeSoundDevice) -> None:
    code, output = cli("doctor")
    assert code == 0, output
    for text in ("System & Python", "Audio (PortAudio)", "Presets", "Models cache", "python",
                 "V19.7.0", "pipewire", "summary:", "--mic"):  # fmt: skip
        assert text in output


def test_cli_interactive_checks_with_fake_io(
    quiet_machine: FakeSoundDevice, monkeypatch: pytest.MonkeyPatch
) -> None:
    io = FakeIO(mic=speech_like(48_000, 2.0, -20.0, -65.0), room=room(0.1, 0.1))
    monkeypatch.setattr(cli_doctor, "audio_io", lambda: io)
    code, report = cli_json(
        "--only", "system", "--mic", "--echo", "--latency", "--duration", "2",
        "--input-device", "1", "--output-device", "pipewire",
    )  # fmt: skip
    assert code == 0, report
    assert report["sections"] == ["system", "mic", "echo", "latency"]
    echo = next(c for c in report["checks"] if c["section"] == "echo" and c["name"] == "echo path")
    assert echo["data"]["delay_ms"] == pytest.approx(100.0, abs=0.1)
    assert echo["data"]["recommended"]["echo_mode"] == "aec"
    code, output = cli("doctor", "--only", "system", "--mic", "--duration", "2")
    assert code == 0 and "recording 2 s from [1] pipewire" in output and "good level" in output
    io.mic = np.zeros(96_000, np.int16)
    code, output = cli("doctor", "--only", "system", "--mic", "--duration", "2")
    assert code == 1 and "silence" in output


def test_cli_interactive_checks_without_devices(
    make_sd: Callable[..., FakeSoundDevice], monkeypatch: pytest.MonkeyPatch
) -> None:
    make_sd(devices=[], default=(-1, -1))
    monkeypatch.setattr(cli_doctor, "audio_io", lambda: pytest.fail("must not open devices"))
    code, report = cli_json("--only", "system", "--mic", "--echo")
    assert code == 1
    failed = [c for c in report["checks"] if c["status"] == "fail"]
    assert [c["section"] for c in failed] == ["mic", "echo"]
    assert "no default input device" in failed[0]["value"]


def test_cli_network_flag(quiet_machine: FakeSoundDevice, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[str, int]] = []

    def probe(host: str, port: int = 443, timeout: float = 5.0) -> dx.ProbeResult:
        seen.append((host, port))
        return dx.ProbeResult(host, port, "127.0.0.1", 1.0, 2.0, 3.0, "TLSv1.3")

    monkeypatch.setattr(dx, "probe_endpoint", probe)
    code, report = cli_json("--only", "system", "--endpoint", "https://example.org:8443")
    assert code == 0 and seen == [("example.org", 8443)]
    assert report["sections"] == ["system", "network"]
