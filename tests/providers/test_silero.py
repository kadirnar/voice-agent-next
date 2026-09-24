"""Silero VAD provider.

Unit tests never import the real ``onnxruntime`` or touch the network: a fake module is
installed in ``sys.modules`` and ``model_path`` points at a dummy file. The
``@pytest.mark.model`` tests download the real model (2.3 MB) and a public-domain speech
clip (350 KB) once into the model cache:
``uv sync --extra silero && uv run pytest -m model tests/providers/test_silero.py``.
"""

from __future__ import annotations

import hashlib
import sys
import threading
import types
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from voice_agent_next import create
from voice_agent_next.audio import AudioFrame
from voice_agent_next.audio.resample import resample
from voice_agent_next.audio.wav import read_wav
from voice_agent_next.errors import ConfigurationError, MissingDependencyError, ProviderError
from voice_agent_next.metrics import VADMetrics
from voice_agent_next.providers import silero
from voice_agent_next.providers.mock import synth_speech
from voice_agent_next.providers.silero import SileroVAD
from voice_agent_next.registry import get_provider
from voice_agent_next.utils.download import DownloadError, cache_dir, download
from voice_agent_next.vad import VADEvent, VADEventType, VADOptions

CONTEXT = {16_000: 64, 8_000: 32}
MODEL_SHA256 = "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"

Array = npt.NDArray[Any]


# ---------------------------------------------------------------- fake onnxruntime


class FakeSessionOptions:
    def __init__(self) -> None:
        self.intra_op_num_threads = 0
        self.inter_op_num_threads = 0
        self.config: dict[str, str] = {}

    def add_session_config_entry(self, key: str, value: str) -> None:
        self.config[key] = value


class FakeSession:
    """Stands in for the Silero graph and records every call.

    ``output`` is 1.0 when the new samples (after the context) are louder than -40 dBFS,
    else 0.0. ``stateN`` is ``state + 1``, so the state fed to call *k* equals *k* when
    the state is threaded correctly.
    """

    def __init__(
        self,
        path: str,
        sess_options: FakeSessionOptions,
        providers: list[str],
        input_names: tuple[str, ...],
    ) -> None:
        self.path = path
        self.options = sess_options
        self.providers = providers
        self.input_names = input_names
        self.thread = threading.get_ident()
        self.feeds: list[dict[str, Array]] = []
        self.error: Exception | None = None

    def get_inputs(self) -> list[types.SimpleNamespace]:
        return [types.SimpleNamespace(name=name) for name in self.input_names]

    def run(self, output_names: list[str] | None, feeds: dict[str, Array]) -> list[Array]:
        if self.error is not None:
            raise self.error
        self.feeds.append({k: np.array(v, copy=True) for k, v in feeds.items()})
        new = feeds["input"][0, CONTEXT[int(feeds["sr"])] :]
        loud = float(np.sqrt(np.mean(np.square(new)))) > 0.01
        return [np.array([[1.0 if loud else 0.0]], dtype=np.float32), feeds["state"] + 1.0]


class FakeOrt(types.ModuleType):
    """A fake ``onnxruntime`` module that records the sessions it creates."""

    def __init__(self) -> None:
        super().__init__("onnxruntime")
        self.SessionOptions = FakeSessionOptions
        self.providers = ["CPUExecutionProvider"]
        self.input_names = ("input", "state", "sr")
        self.load_error: Exception | None = None
        self.sessions: list[FakeSession] = []

    def get_available_providers(self) -> list[str]:
        return list(self.providers)

    def InferenceSession(
        self, path: str, sess_options: FakeSessionOptions, providers: list[str]
    ) -> FakeSession:
        if self.load_error is not None:
            raise self.load_error
        session = FakeSession(path, sess_options, providers, self.input_names)
        self.sessions.append(session)
        return session


@pytest.fixture
def fake_ort(monkeypatch: pytest.MonkeyPatch) -> FakeOrt:
    fake = FakeOrt()
    monkeypatch.setitem(sys.modules, "onnxruntime", fake)
    return fake


@pytest.fixture
def model_file(tmp_path: Path) -> Path:
    path = tmp_path / "silero_vad.onnx"
    path.write_bytes(b"not a real model: the fake onnxruntime never parses it")
    return path


def chunks(frame: AudioFrame, step: float = 0.02) -> list[AudioFrame]:
    n = round(step * frame.sample_rate) * 2 * frame.channels
    return [
        AudioFrame(frame.data[i : i + n], frame.sample_rate, frame.channels)
        for i in range(0, len(frame.data), n)
    ]


def run_stream(vad: SileroVAD, audio: AudioFrame) -> list[VADEvent]:
    stream = vad.stream(emit_inference_events=True)
    events: list[VADEvent] = []
    for frame in chunks(audio):
        events += stream.push_audio(frame)
    stream.close()
    return events


def of_type(events: list[VADEvent], kind: VADEventType) -> list[VADEvent]:
    return [e for e in events if e.type is kind]


# -------------------------------------------------------------------- unit tests


@pytest.mark.parametrize(
    ("sample_rate", "window", "context"), [(16_000, 512, 64), (8_000, 256, 32)]
)
def test_window_context_and_input_format(
    fake_ort: FakeOrt, model_file: Path, sample_rate: int, window: int, context: int
) -> None:
    vad = SileroVAD(model_path=model_file, sample_rate=sample_rate)
    assert vad.window_samples == window
    assert vad.window_duration == pytest.approx(0.032)
    infer = vad._new_inference()
    windows = [((k * window + np.arange(window)) / 1e5).astype(np.float32) for k in range(3)]
    for w in windows:
        infer(w)
    feeds = fake_ort.sessions[0].feeds
    assert len(feeds) == 3
    for k, feed in enumerate(feeds):
        assert feed["input"].shape == (1, context + window)
        assert feed["input"].dtype == np.float32
        assert feed["sr"].shape == ()
        assert feed["sr"].dtype == np.int64
        assert int(feed["sr"]) == sample_rate
        # [context | window]: the context is the tail of the previous window (zeros at first)
        np.testing.assert_array_equal(feed["input"][0, context:], windows[k])
        previous = windows[k - 1][-context:] if k else np.zeros(context, np.float32)
        np.testing.assert_array_equal(feed["input"][0, :context], previous)


def test_recurrent_state_is_threaded_between_windows(fake_ort: FakeOrt, model_file: Path) -> None:
    infer = SileroVAD(model_path=model_file)._new_inference()
    for _ in range(4):
        infer(np.zeros(512, np.float32))
    for k, feed in enumerate(fake_ort.sessions[0].feeds):
        assert feed["state"].shape == (2, 1, 128)
        assert feed["state"].dtype == np.float32
        np.testing.assert_array_equal(feed["state"], np.full((2, 1, 128), k, np.float32))


def test_probability_is_the_model_output(fake_ort: FakeOrt, model_file: Path) -> None:
    infer = SileroVAD(model_path=model_file)._new_inference()
    assert infer(np.zeros(512, np.float32)) == 0.0
    loud = infer(np.full(512, 0.5, np.float32))
    assert loud == 1.0
    assert isinstance(loud, float)


def test_reset_clears_state_and_context(fake_ort: FakeOrt, model_file: Path) -> None:
    infer = SileroVAD(model_path=model_file)._new_inference()
    infer(np.full(512, 0.5, np.float32))
    infer(np.full(512, 0.5, np.float32))
    infer.reset()
    infer(np.full(512, 0.25, np.float32))
    last = fake_ort.sessions[0].feeds[-1]
    assert not last["state"].any()
    assert not last["input"][0, :64].any()


def test_stream_reset_resets_model_state(fake_ort: FakeOrt, model_file: Path) -> None:
    stream = SileroVAD(model_path=model_file).stream()
    stream.push_audio(AudioFrame.from_numpy(np.full(1024, 0.5, np.float32), 16_000))
    stream.reset()
    stream.push_audio(AudioFrame.from_numpy(np.full(512, 0.5, np.float32), 16_000))
    feeds = fake_ort.sessions[0].feeds
    assert len(feeds) == 3
    assert not feeds[-1]["state"].any()
    assert not feeds[-1]["input"][0, :64].any()


def test_streams_share_one_session_but_not_state(fake_ort: FakeOrt, model_file: Path) -> None:
    vad = SileroVAD(model_path=model_file)
    a, b = vad._new_inference(), vad._new_inference()
    a(np.full(512, 0.5, np.float32))
    a(np.full(512, 0.5, np.float32))
    b(np.full(512, 0.25, np.float32))
    vad.stream()
    vad.stream()
    assert len(fake_ort.sessions) == 1
    feeds = fake_ort.sessions[0].feeds
    assert not feeds[2]["state"].any()  # b starts from its own zero state...
    assert not feeds[2]["input"][0, :64].any()  # ...and its own zero context
    a(np.full(512, 0.5, np.float32))
    assert feeds[-1]["state"].max() == 2.0  # a continues where it left off


def test_session_options(fake_ort: FakeOrt, model_file: Path) -> None:
    SileroVAD(model_path=model_file).stream()
    session = fake_ort.sessions[0]
    assert session.path == str(model_file)
    assert session.options.intra_op_num_threads == 1
    assert session.options.inter_op_num_threads == 1
    assert session.options.config == {
        "session.intra_op.allow_spinning": "0",
        "session.inter_op.allow_spinning": "0",
    }
    assert session.providers == ["CPUExecutionProvider"]


def test_force_cpu(fake_ort: FakeOrt, model_file: Path) -> None:
    fake_ort.providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    SileroVAD(model_path=model_file).stream()
    SileroVAD(model_path=model_file, force_cpu=False).stream()
    assert [s.providers for s in fake_ort.sessions] == [
        ["CPUExecutionProvider"],
        ["CUDAExecutionProvider", "CPUExecutionProvider"],
    ]


def test_rejects_wrong_window_size(fake_ort: FakeOrt, model_file: Path) -> None:
    infer = SileroVAD(model_path=model_file)._new_inference()
    with pytest.raises(ValueError, match="512 samples"):
        infer(np.zeros(256, np.float32))


def test_stream_segments_speech(fake_ort: FakeOrt, model_file: Path) -> None:
    """End to end through ``VADStream``: 48 kHz stereo in, 512-sample windows at 16 kHz."""
    vad = SileroVAD(model_path=model_file, min_speech_duration=0.1, min_silence_duration=0.3)
    audio = AudioFrame.concat(
        [
            AudioFrame.silence(0.5, 48_000, 2),
            synth_speech(1.0, 48_000).to_channels(2),
            AudioFrame.silence(0.6, 48_000, 2),
        ]
    )
    stream = vad.stream()
    events: list[VADEvent] = []
    for frame in chunks(audio):
        events += stream.push_audio(frame)
    assert [e.type for e in events] == [VADEventType.START_OF_SPEECH, VADEventType.END_OF_SPEECH]
    start, end = events
    assert start.audio_time == pytest.approx(0.5 + 4 * 0.032, abs=0.05)  # 4 windows >= 0.1 s
    assert end.speech_duration == pytest.approx(1.0, abs=0.1)
    assert end.silence_duration == pytest.approx(0.32, abs=0.001)  # 10 windows >= 0.3 s
    feeds = fake_ort.sessions[0].feeds
    assert all(f["input"].shape == (1, 64 + 512) for f in feeds)


def test_create_from_registry(fake_ort: FakeOrt, model_file: Path) -> None:
    spec = get_provider("vad", "silero")
    assert spec.factory is SileroVAD
    assert spec.default_model == "v6.2"
    assert spec.extra == "silero"
    assert spec.requires == ("onnxruntime",)
    assert spec.local
    vad = create("vad", "silero", model_path=model_file)
    assert isinstance(vad, SileroVAD)
    assert (vad.provider, vad.model, vad.sample_rate) == ("silero", "v6.2", 16_000)
    vad8 = create(
        "vad",
        {
            "provider": "silero/v6.2",
            "sample_rate": 8000,
            "min_silence_duration": 0.3,
            "model_path": str(model_file),
        },
    )
    assert (vad8.sample_rate, vad8.window_samples) == (8000, 256)
    assert vad8.options.min_silence_duration == 0.3


def test_option_overrides_apply_on_top_of_options(fake_ort: FakeOrt) -> None:
    opts = VADOptions(activation_threshold=0.6)
    vad = SileroVAD(options=opts, min_silence_duration=0.4)
    assert vad.options.activation_threshold == 0.6
    assert vad.options.min_silence_duration == 0.4
    assert opts.min_silence_duration == 0.25  # the caller's options are not modified


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"sample_rate": 44_100}, "8000 or 16000"),
        ({"model": "v4"}, "unknown Silero VAD model"),
        ({"min_silence": 0.3}, "unknown Silero VAD option"),
    ],
)
def test_invalid_configuration(fake_ort: FakeOrt, kwargs: dict[str, Any], match: str) -> None:
    with pytest.raises(ConfigurationError, match=match):
        SileroVAD(**kwargs)


def test_model_path_skips_model_id_validation(fake_ort: FakeOrt, model_file: Path) -> None:
    assert SileroVAD(model="my-finetune", model_path=model_file).model == "my-finetune"


def test_missing_model_file(fake_ort: FakeOrt, tmp_path: Path) -> None:
    vad = SileroVAD(model_path=tmp_path / "missing.onnx")
    with pytest.raises(ConfigurationError, match="not found"):
        vad.stream()


def test_model_load_failure_is_a_provider_error(fake_ort: FakeOrt, model_file: Path) -> None:
    fake_ort.load_error = RuntimeError("[ONNXRuntimeError] : 7 : INVALID_PROTOBUF")
    vad = SileroVAD(model_path=model_file)
    with pytest.raises(ProviderError, match="INVALID_PROTOBUF") as info:
        vad.stream()
    assert info.value.provider == "silero"


def test_rejects_models_with_another_interface(fake_ort: FakeOrt, model_file: Path) -> None:
    fake_ort.input_names = ("input", "sr", "h", "c")  # Silero v4
    vad = SileroVAD(model_path=model_file)
    with pytest.raises(ConfigurationError, match="not a Silero VAD v5/v6 ONNX model"):
        vad.stream()


def test_inference_failure_is_a_provider_error(fake_ort: FakeOrt, model_file: Path) -> None:
    infer = SileroVAD(model_path=model_file)._new_inference()
    fake_ort.sessions[0].error = RuntimeError("boom")
    with pytest.raises(ProviderError, match="boom"):
        infer(np.zeros(512, np.float32))


def test_missing_onnxruntime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "onnxruntime", None)  # makes `import onnxruntime` fail
    with pytest.raises(MissingDependencyError, match=r"voice-agent-next\[silero\]"):
        SileroVAD()


def test_downloads_pinned_model_once_on_first_use(
    fake_ort: FakeOrt, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    def fake_download(url: str, **kwargs: Any) -> Path:
        calls.append((url, kwargs))
        path = tmp_path / "silero_vad_v6.2.onnx"
        path.write_bytes(b"x")
        return path

    monkeypatch.setattr(silero, "download", fake_download)
    vad = SileroVAD()
    assert calls == []  # nothing is downloaded or loaded before first use
    vad.stream()
    vad.stream()
    assert calls == [
        (
            "https://raw.githubusercontent.com/snakers4/silero-vad/v6.2/src/silero_vad/data/silero_vad.onnx",
            {"filename": "silero_vad_v6.2.onnx", "subdir": "silero", "sha256": MODEL_SHA256},
        )
    ]
    assert fake_ort.sessions[0].path == str(tmp_path / "silero_vad_v6.2.onnx")


def test_works_offline_once_cached(
    fake_ort: FakeOrt, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("VAN_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("VAN_OFFLINE", "1")
    payload = b"fake silero model"
    monkeypatch.setitem(
        silero._MODELS,
        "v6.2",
        silero._ModelFile(
            url="https://example.invalid/silero_vad.onnx",
            sha256=hashlib.sha256(payload).hexdigest(),
            filename="silero_vad_v6.2.onnx",
        ),
    )
    vad = SileroVAD()
    with pytest.raises(DownloadError, match="VAN_OFFLINE"):
        vad.stream()
    cached = cache_dir() / "silero" / "silero_vad_v6.2.onnx"  # as left by an earlier run
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_bytes(payload)
    vad.stream()
    assert fake_ort.sessions[0].path == str(cached)


async def test_warmup_loads_off_the_event_loop(fake_ort: FakeOrt, model_file: Path) -> None:
    vad = SileroVAD(model_path=model_file)
    assert fake_ort.sessions == []  # lazy
    await vad.warmup()
    assert len(fake_ort.sessions) == 1
    assert fake_ort.sessions[0].thread != threading.get_ident()
    assert len(fake_ort.sessions[0].feeds) == 1  # one warm-up window
    vad.stream()
    assert len(fake_ort.sessions) == 1  # the warmed-up session is reused
    await vad.aclose()
    vad.stream()
    assert len(fake_ort.sessions) == 2  # loaded again after aclose()


# ------------------------------------------------------------------- real model

JFK_URL = "https://raw.githubusercontent.com/ggml-org/whisper.cpp/v1.9.4/samples/jfk.wav"
JFK_SHA256 = "59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e"
# J. F. Kennedy's inaugural address, 1961 (public domain): 11 s, 16 kHz mono, four phrases:
# "And so my fellow Americans / ask not / what your country can do for you /
#  ask what you can do for your country".
JFK_PHRASE_STARTS = [0.35, 3.3, 5.4, 8.2]
"""Where each phrase starts (s); START_OF_SPEECH fires after ``min_speech_duration``."""


@pytest.fixture(scope="module")
def jfk() -> AudioFrame:
    pytest.importorskip("onnxruntime")
    return read_wav(download(JFK_URL, subdir="testdata", sha256=JFK_SHA256))


@pytest.mark.model
@pytest.mark.parametrize("input_rate", [16_000, 48_000])
@pytest.mark.parametrize("sample_rate", [16_000, 8_000])
def test_real_model_detects_recorded_speech(
    jfk: AudioFrame, sample_rate: int, input_rate: int
) -> None:
    vad = SileroVAD(sample_rate=sample_rate)
    metrics: list[VADMetrics] = []
    vad.on("metrics", metrics.append)
    audio = AudioFrame.concat([resample(jfk, input_rate), AudioFrame.silence(1.0, input_rate)])
    events = run_stream(vad, audio)

    starts = of_type(events, VADEventType.START_OF_SPEECH)
    ends = of_type(events, VADEventType.END_OF_SPEECH)
    probs = np.array([e.probability for e in of_type(events, VADEventType.INFERENCE_DONE)])
    assert len(starts) == len(ends) == 4
    expected = [t + vad.options.min_speech_duration for t in JFK_PHRASE_STARTS]
    assert [e.audio_time for e in starts] == pytest.approx(expected, abs=0.15)
    assert 0.5 < float(np.mean(probs >= 0.5)) < 0.8

    count = sum(m.inference_count for m in metrics)
    assert count == len(probs)
    per_window_ms = 1e3 * sum(m.inference_duration_total for m in metrics) / count
    print(
        f"\nSilero VAD {vad.model} @ {sample_rate} Hz (input {input_rate} Hz): "
        f"{per_window_ms:.3f} ms per {vad.window_samples}-sample window, {count} windows"
    )
    assert per_window_ms < 10.0  # real-time budget: 32 ms per window (typically ~0.1 ms)


@pytest.mark.model
@pytest.mark.parametrize("sample_rate", [16_000, 8_000])
def test_real_model_rejects_silence_and_noise(sample_rate: int) -> None:
    pytest.importorskip("onnxruntime")
    vad = SileroVAD(sample_rate=sample_rate)
    rng = np.random.default_rng(0)
    n = 3 * 16_000
    brown = np.cumsum(rng.normal(0.0, 1.0, n))
    brown -= brown.mean()
    signals = {
        "silence": np.zeros(n),
        "white noise": rng.normal(0.0, 0.1, n),
        "brown noise": 0.3 * brown / np.abs(brown).max(),
        "1 kHz tone": 0.3 * np.sin(2 * np.pi * 1000 * np.arange(n) / 16_000),
        "AM tone (synth_speech)": synth_speech(3.0, 16_000).to_float32(),
    }
    for name, x in signals.items():
        events = run_stream(vad, AudioFrame.from_numpy(x.astype(np.float32), 16_000))
        assert not of_type(events, VADEventType.START_OF_SPEECH), name
        probs = [e.probability for e in of_type(events, VADEventType.INFERENCE_DONE)]
        assert max(probs) < vad.options.effective_deactivation, name


@pytest.mark.model
@pytest.mark.parametrize("sample_rate", [16_000, 8_000])
def test_real_model_matches_reference_onnx_wrapper(jfk: AudioFrame, sample_rate: int) -> None:
    """Window by window, the same probabilities as the reference ``OnnxWrapper.__call__``
    (snakers4/silero-vad ``utils_vad.py``, with numpy in place of torch)."""
    import onnxruntime as ort

    vad = SileroVAD(sample_rate=sample_rate)
    infer = vad._new_inference()
    x = resample(jfk, sample_rate).to_float32()
    win, ctx = vad.window_samples, CONTEXT[sample_rate]
    windows = [x[i : i + win] for i in range(0, len(x) - win + 1, win)]
    ours = [infer(w) for w in windows]

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(vad._resolve_model_path()), sess_options=opts, providers=["CPUExecutionProvider"]
    )
    state = np.zeros((2, 1, 128), dtype=np.float32)
    context = np.zeros((1, ctx), dtype=np.float32)
    sr = np.array(sample_rate, dtype=np.int64)
    reference = []
    for w in windows:
        inp = np.concatenate([context, w[None, :]], axis=1)
        out, state = session.run(None, {"input": inp, "state": state, "sr": sr})
        context = inp[..., -ctx:]
        reference.append(float(out[0, 0]))

    assert max(reference) > 0.9
    np.testing.assert_allclose(ours, reference, rtol=0, atol=1e-5)
    infer.reset()
    np.testing.assert_allclose([infer(w) for w in windows], ours, rtol=0, atol=1e-5)


@pytest.mark.model
async def test_real_model_loads_offline_once_downloaded(
    jfk: AudioFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    await SileroVAD().warmup()  # downloads into the model cache on the first run only
    monkeypatch.setenv("VAN_OFFLINE", "1")
    vad = SileroVAD()
    await vad.warmup()
    assert len(of_type(run_stream(vad, jfk), VADEventType.START_OF_SPEECH)) == 4


@pytest.mark.model
async def test_real_model_without_force_cpu(jfk: AudioFrame) -> None:
    vad = SileroVAD(force_cpu=False)  # every available execution provider
    await vad.warmup()
    assert len(of_type(run_stream(vad, jfk), VADEventType.START_OF_SPEECH)) == 4
