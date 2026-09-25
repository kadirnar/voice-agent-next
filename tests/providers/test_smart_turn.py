"""Tests for the Smart Turn v3 end-of-turn detector (``providers/smart_turn.py``).

Unit tests run offline against a fake ONNX Runtime session. The ``@pytest.mark.model`` tests
download the real models (8 MB int8, 32 MB fp32) and a public-domain speech sample (352 KB,
cached, never committed)::

    uv sync --extra smart-turn && uv run pytest -m model tests/providers/test_smart_turn.py

Golden features
---------------
``data/smart_turn_log_mel.npy`` holds reference log-mel features of the signals in
:func:`golden_signals` (every 10th frame, shape ``(3, 80, 80)``, float32), computed with
``transformers.WhisperFeatureExtractor`` called exactly like Smart Turn's ``inference.py``
does (numpy code path: torch not installed). Regenerate it in a throwaway environment::

    uv run --with transformers==5.17.0 python tests/providers/test_smart_turn.py
"""

from __future__ import annotations

import asyncio
import sys
import threading
import time
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from voice_agent_next import Agent, AgentSession, AudioFrame, CascadeOptions, create
from voice_agent_next.audio import read_wav, resample
from voice_agent_next.errors import (
    ConfigurationError,
    MissingDependencyError,
    ProviderError,
    ProviderNotFoundError,
)
from voice_agent_next.metrics import EOTMetrics
from voice_agent_next.providers import smart_turn
from voice_agent_next.providers.energy import EnergyVAD
from voice_agent_next.providers.mock import MockLLM, MockSTT, MockTTS, synth_speech
from voice_agent_next.providers.smart_turn import (
    DEFAULT_MODEL,
    HF_REPO,
    HF_REVISION,
    LANGUAGES,
    MODEL_SHA256,
    SmartTurnDetector,
    log_mel_features,
    prepare_audio,
)
from voice_agent_next.registry import get_provider
from voice_agent_next.transports import LoopbackTransport
from voice_agent_next.utils.download import download

SR = 16_000
N_SAMPLES = 8 * SR
GOLDEN_PATH = Path(__file__).parent / "data" / "smart_turn_log_mel.npy"
GOLDEN_FRAMES = slice(5, None, 10)


def _pcm16(x: npt.NDArray[np.float64]) -> npt.NDArray[np.int16]:
    return np.clip(np.round(x * 32767.0), -32768, 32767).astype(np.int16)


def golden_signals() -> dict[str, npt.NDArray[np.int16]]:
    """Deterministic 16 kHz test signals, shared with the golden-file generator.

    ``RandomState`` (not ``default_rng``): its stream is frozen across numpy versions.
    """
    rng = np.random.RandomState(4)
    # 1.6 s speech-like signal (harmonics, falling pitch, 4 Hz syllables): padded at the start
    t = np.arange(round(1.6 * SR)) / SR
    phase = 2 * np.pi * np.cumsum(170.0 - 50.0 * t / t[-1]) / SR
    harmonics = sum(np.sin(k * phase) / k for k in range(1, 9))
    envelope = 0.55 + 0.45 * np.sin(2 * np.pi * 4.0 * t)
    utterance = 0.2 * envelope * harmonics + 0.01 * rng.standard_normal(t.size)
    # 10 s chirp 100 Hz -> 7.5 kHz: longer than the window, only the last 8 s count
    t10 = np.arange(10 * SR) / SR
    chirp = 0.5 * np.sin(2 * np.pi * (100.0 * t10 + 370.0 * t10**2))
    # exactly one window of white noise
    noise = 0.1 * rng.standard_normal(N_SAMPLES)
    return {"utterance": _pcm16(utterance), "chirp": _pcm16(chirp), "noise": _pcm16(noise)}


# ------------------------------------------------------------------------------- fakes
class FakeSession:
    """Stands in for ``onnxruntime.InferenceSession``: records feeds, returns scripted outputs."""

    def __init__(self, probabilities: tuple[float, ...] = (0.9,), *, fail: bool = False) -> None:
        self.probabilities = probabilities
        self.fail = fail
        self.feeds: list[dict[str, npt.NDArray[np.float32]]] = []
        self.threads: list[int] = []

    def get_inputs(self) -> list[Any]:
        return [types.SimpleNamespace(name="input_features", shape=["batch", 80, 800])]

    def run(self, output_names: Any, feeds: dict[str, npt.NDArray[np.float32]]) -> list[Any]:
        if self.fail:
            raise RuntimeError("[ONNXRuntimeError] : 2 : INVALID_ARGUMENT")
        self.feeds.append(feeds)
        self.threads.append(threading.get_ident())
        p = self.probabilities[min(len(self.feeds), len(self.probabilities)) - 1]
        return [np.full((len(feeds["input_features"]), 1), p, dtype=np.float32)]


def fake_detector(session: FakeSession | None = None, **kwargs: Any) -> SmartTurnDetector:
    detector = SmartTurnDetector(**kwargs)
    fake = session or FakeSession()
    detector._load_session = lambda: fake  # type: ignore[method-assign]
    return detector


def fake_onnxruntime(created: list[dict[str, Any]], *, fail: bool = False) -> types.ModuleType:
    """A minimal ``onnxruntime`` module recording how sessions are created."""

    class SessionOptions:
        def __init__(self) -> None:
            self.config: dict[str, str] = {}

        def add_session_config_entry(self, key: str, value: str) -> None:
            self.config[key] = value

    def inference_session(path: str, sess_options: Any = None, providers: Any = None) -> Any:
        if fail:
            raise RuntimeError("[ONNXRuntimeError] : 7 : INVALID_PROTOBUF")
        created.append({"path": path, "options": sess_options, "providers": providers})
        return FakeSession()

    module = types.ModuleType("onnxruntime")
    module.SessionOptions = SessionOptions  # type: ignore[attr-defined]
    module.InferenceSession = inference_session  # type: ignore[attr-defined]
    module.ExecutionMode = types.SimpleNamespace(ORT_SEQUENTIAL="sequential")  # type: ignore[attr-defined]
    module.GraphOptimizationLevel = types.SimpleNamespace(ORT_ENABLE_ALL="all")  # type: ignore[attr-defined]
    return module


# ---------------------------------------------------------------------------- features
def test_log_mel_features_match_transformers_golden() -> None:
    golden = np.load(GOLDEN_PATH)
    signals = golden_signals()
    assert golden.shape == (len(signals), 80, 80) and golden.dtype == np.float32
    for ref, (name, pcm) in zip(golden, signals.items(), strict=True):
        features = log_mel_features(prepare_audio(pcm.astype(np.float32) / 32768.0))
        assert features.shape == (80, 800) and features.dtype == np.float32
        # acceptance criterion is 1e-3; the port is within ~1e-7 (float32 rounding)
        np.testing.assert_allclose(features[:, GOLDEN_FRAMES], ref, rtol=0, atol=1e-4, err_msg=name)


def test_log_mel_features_of_silence_is_the_floor() -> None:
    features = log_mel_features(np.zeros(N_SAMPLES, dtype=np.float32))
    assert np.all(features == -1.5)  # log10(1e-10) = -10 -> (-10 + 4) / 4


def test_log_mel_features_requires_a_full_mono_window() -> None:
    with pytest.raises(ValueError, match="128000"):
        log_mel_features(np.zeros(SR, dtype=np.float32))
    with pytest.raises(ValueError, match="1-D"):
        log_mel_features(np.zeros((2, N_SAMPLES // 2), dtype=np.float32))
    with pytest.raises(ValueError, match="1-D"):
        prepare_audio(np.zeros((SR, 2), dtype=np.float32))


def test_prepare_audio_pads_at_the_start_and_keeps_the_end() -> None:
    short = prepare_audio(np.array([0.1, 0.2, 0.3]))
    assert short.shape == (N_SAMPLES,) and short.dtype == np.float32
    assert np.all(short[:-3] == 0) and np.allclose(short[-3:], [0.1, 0.2, 0.3])
    long = np.arange(N_SAMPLES + 500, dtype=np.float32)
    assert np.array_equal(prepare_audio(long), long[500:])
    exact = np.ones(N_SAMPLES, dtype=np.float32)
    assert np.array_equal(prepare_audio(exact), exact)


# --------------------------------------------------------------------------- inference
async def test_predict_feeds_features_of_the_last_8_seconds() -> None:
    session = FakeSession((0.83,))
    detector = fake_detector(session)
    speech = synth_speech(6.0, SR)
    turn = AudioFrame.concat([AudioFrame.from_numpy(np.full(4 * SR, 0.5), SR), speech])  # 10 s
    p = await detector.predict_end_of_turn(audio=turn)
    assert p == pytest.approx(0.83)
    (feeds,) = session.feeds
    assert list(feeds) == ["input_features"]
    features = feeds["input_features"]
    assert features.shape == (1, 80, 800) and features.dtype == np.float32
    expected = log_mel_features(prepare_audio(turn.to_float32()[-N_SAMPLES:]))
    np.testing.assert_array_equal(features[0], expected)


async def test_predict_pads_short_turns_at_the_start() -> None:
    session = FakeSession()
    detector = fake_detector(session)
    speech = synth_speech(1.2, SR)
    await detector.predict_end_of_turn(audio=speech)
    padded = np.concatenate([np.zeros(N_SAMPLES - len(speech.to_float32())), speech.to_float32()])
    np.testing.assert_array_equal(session.feeds[0]["input_features"][0], log_mel_features(padded))


async def test_predict_resamples_and_downmixes() -> None:
    session = FakeSession()
    detector = fake_detector(session)
    speech = synth_speech(2.0, SR)
    await detector.predict_end_of_turn(audio=speech)
    await detector.predict_end_of_turn(audio=resample(speech, 48_000).to_channels(2))
    await detector.predict_end_of_turn(audio=resample(synth_speech(9.0, SR), 8_000))
    reference, converted = (f["input_features"][0] for f in session.feeds[:2])
    assert np.abs(reference - converted).mean() < 0.002  # ~1e-4 with soxr or numpy
    assert all(f["input_features"].shape == (1, 80, 800) for f in session.feeds)


async def test_inference_runs_off_the_event_loop_thread() -> None:
    session = FakeSession()
    detector = fake_detector(session)
    await detector.predict_end_of_turn(audio=synth_speech(1.0, SR))
    assert session.threads and session.threads[0] != threading.get_ident()


async def test_no_audio_means_complete_without_inference() -> None:
    session = FakeSession((0.0,))
    detector = fake_detector(session)
    assert await detector.predict_end_of_turn(audio=None) == 1.0
    assert await detector.predict_end_of_turn(audio=AudioFrame.empty(SR)) == 1.0
    assert session.feeds == []


async def test_emits_eot_metrics_with_threshold_decision() -> None:
    detector = fake_detector(FakeSession((0.3, 0.7)), threshold=0.6)
    got: list[EOTMetrics] = []
    detector.on("metrics", got.append)
    await detector.predict_end_of_turn(audio=synth_speech(1.0, SR))
    await detector.predict_end_of_turn(audio=synth_speech(1.0, SR))
    assert [(m.probability, m.end_of_turn) for m in got] == [
        (pytest.approx(0.3), False),
        (pytest.approx(0.7), True),
    ]
    assert got[0].provider == "smart_turn" and got[0].model == DEFAULT_MODEL
    assert got[0].threshold == 0.6 and got[0].inference_duration >= 0


async def test_inference_errors_become_provider_errors() -> None:
    detector = fake_detector(FakeSession(fail=True))
    with pytest.raises(ProviderError, match="inference failed") as info:
        await detector.predict_end_of_turn(audio=synth_speech(1.0, SR))
    assert info.value.provider == "smart_turn"


async def test_session_is_loaded_once_under_concurrency() -> None:
    detector = SmartTurnDetector()
    loads: list[int] = []

    def slow_load() -> FakeSession:
        loads.append(1)
        time.sleep(0.05)
        return FakeSession()

    detector._load_session = slow_load  # type: ignore[method-assign]
    frame = synth_speech(1.0, SR)
    await asyncio.gather(*(detector.predict_end_of_turn(audio=frame) for _ in range(4)))
    assert len(loads) == 1
    await detector.aclose()  # drops the session; the next prediction loads it again
    await detector.predict_end_of_turn(audio=frame)
    assert len(loads) == 2


async def test_warmup_loads_and_runs_the_model_once() -> None:
    session = FakeSession()
    detector = fake_detector(session)
    await detector.warmup()
    assert len(session.feeds) == 1
    assert session.feeds[0]["input_features"].shape == (1, 80, 800)


# ------------------------------------------------------------------------ model loading
def test_session_uses_single_thread_without_spinning(monkeypatch: pytest.MonkeyPatch) -> None:
    created: list[dict[str, Any]] = []
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_onnxruntime(created))
    fetched: list[tuple[Any, ...]] = []

    def fake_hf_file(repo_id: str, filename: str, **kwargs: Any) -> Path:
        fetched.append((repo_id, filename, kwargs))
        return Path("/models") / filename

    monkeypatch.setattr(smart_turn, "hf_file", fake_hf_file)
    detector = SmartTurnDetector()
    assert isinstance(detector._get_session(), FakeSession)
    assert detector._get_session() is detector._get_session()
    assert fetched == [
        (
            HF_REPO,
            "smart-turn-v3.2-cpu.onnx",
            {"revision": HF_REVISION, "sha256": MODEL_SHA256["smart-turn-v3.2-cpu.onnx"]},
        )
    ]
    (call,) = created
    assert Path(call["path"]) == Path("/models/smart-turn-v3.2-cpu.onnx")
    assert call["providers"] == ["CPUExecutionProvider"]
    opts = call["options"]
    assert (opts.intra_op_num_threads, opts.inter_op_num_threads) == (1, 1)
    assert opts.execution_mode == "sequential" and opts.graph_optimization_level == "all"
    assert opts.config == {
        "session.intra_op.allow_spinning": "0",
        "session.inter_op.allow_spinning": "0",
    }


def test_model_variants_revision_and_threads(monkeypatch: pytest.MonkeyPatch) -> None:
    created: list[dict[str, Any]] = []
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_onnxruntime(created))
    fetched: list[tuple[str, dict[str, Any]]] = []

    def fake_hf_file(repo_id: str, filename: str, **kwargs: Any) -> Path:
        fetched.append((filename, kwargs))
        return Path(filename)

    monkeypatch.setattr(smart_turn, "hf_file", fake_hf_file)
    gpu = SmartTurnDetector(model="v3.2-gpu", num_threads=4, device=["CUDAExecutionProvider"])
    gpu._get_session()
    SmartTurnDetector(model="smart-turn-v3.1-cpu.onnx", revision="main")._get_session()
    assert fetched[0] == (
        "smart-turn-v3.2-gpu.onnx",
        {"revision": HF_REVISION, "sha256": MODEL_SHA256["smart-turn-v3.2-gpu.onnx"]},
    )
    assert fetched[1] == ("smart-turn-v3.1-cpu.onnx", {"revision": "main", "sha256": None})
    assert created[0]["options"].intra_op_num_threads == 4
    assert created[0]["providers"] == ["CUDAExecutionProvider"]


def test_local_model_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    created: list[dict[str, Any]] = []
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_onnxruntime(created))
    monkeypatch.setattr(smart_turn, "hf_file", lambda *a, **k: pytest.fail("must not download"))
    path = tmp_path / "my-finetune.onnx"
    path.write_bytes(b"onnx")
    detector = SmartTurnDetector(model_path=path)
    assert detector.model == "my-finetune"
    detector._get_session()
    assert Path(created[0]["path"]) == path
    with pytest.raises(ConfigurationError, match="not found"):
        SmartTurnDetector(model_path=tmp_path / "missing.onnx")._get_session()


def test_missing_onnxruntime_is_reported_before_downloading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    monkeypatch.setattr(smart_turn, "hf_file", lambda *a, **k: pytest.fail("must not download"))
    with pytest.raises(MissingDependencyError, match=r"voice-agent-next\[smart-turn\]"):
        SmartTurnDetector()._get_session()


def test_unloadable_model_raises_provider_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_onnxruntime([], fail=True))
    monkeypatch.setattr(smart_turn, "hf_file", lambda repo_id, filename, **k: Path(filename))
    with pytest.raises(ProviderError, match="failed to load"):
        SmartTurnDetector()._get_session()


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [({"threshold": 1.5}, "threshold"), ({"num_threads": 0}, "num_threads")],
)
def test_invalid_options(kwargs: dict[str, Any], match: str) -> None:
    with pytest.raises(ConfigurationError, match=match):
        SmartTurnDetector(**kwargs)


# ----------------------------------------------------------------- registry & languages
def test_registry_entry() -> None:
    spec = get_provider("turn", "smart_turn")
    assert spec.factory is SmartTurnDetector
    assert spec.default_model == DEFAULT_MODEL == "smart-turn-v3.2-cpu"
    assert spec.extra == "smart-turn" and spec.requires == ("onnxruntime",)
    assert spec.local and spec.env == ()
    assert "smart-turn-v3.2-gpu" in spec.models
    default = create("turn", "smart_turn")
    assert isinstance(default, SmartTurnDetector) and default.model == DEFAULT_MODEL
    assert default.modality == "audio" and default.threshold == 0.5
    assert (default.sample_rate, default.max_audio_duration) == (16_000, 8.0)
    gpu = create("turn", {"provider": "smart-turn/smart-turn-v3.2-gpu", "threshold": 0.7})
    assert gpu.model == "smart-turn-v3.2-gpu" and gpu.threshold == 0.7


def test_supported_languages() -> None:
    detector = SmartTurnDetector()
    assert len(LANGUAGES) == 23 and detector.languages == LANGUAGES
    for language in ("en", "en-US", "tr", "zh-CN", "pt_BR", "nb", "eng", "hin", "English", None):
        assert detector.supports_language(language), language
    for language in ("sw", "he", "el", "xx-YY", "klingon"):
        assert not detector.supports_language(language), language


# ------------------------------------------------------------------ AgentSession wiring
async def _wait_for(predicate: Callable[[], bool], timeout: float = 5.0) -> None:
    async def poll() -> None:
        while not predicate():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(poll(), timeout)


async def _speak(transport: LoopbackTransport, seconds: float, then_silence: float) -> None:
    await transport.play_user_audio(synth_speech(seconds, SR), realtime=False)
    await transport.play_user_audio(AudioFrame.silence(then_silence, SR), realtime=False)


async def test_agent_session_with_smart_turn_spec(monkeypatch: pytest.MonkeyPatch) -> None:
    session_fake = FakeSession((0.5, 0.05, 0.95))  # warmup, "not done yet", then "done"
    monkeypatch.setattr(SmartTurnDetector, "_load_session", lambda self: session_fake)
    durations: list[float] = []
    original_infer = SmartTurnDetector._infer

    def spy(self: SmartTurnDetector, audio: AudioFrame) -> float:
        durations.append(audio.duration)
        return original_infer(self, audio)

    monkeypatch.setattr(SmartTurnDetector, "_infer", spy)
    session = AgentSession(
        stt=MockSTT(transcripts=["I would like to", "book a table"]),
        llm=MockLLM(responses=["Sure."]),
        tts=MockTTS(),
        vad=EnergyVAD(),
        turn_detector="smart_turn",
        cascade_options=CascadeOptions(min_endpointing_delay=0.0, max_endpointing_delay=0.6),
    )
    assert isinstance(session.engine.turn_detector, SmartTurnDetector)  # type: ignore[attr-defined]
    eot: list[EOTMetrics] = []
    finals: list[str] = []
    session.on("metrics", lambda m: eot.append(m) if isinstance(m, EOTMetrics) else None)
    session.on("user_transcript", lambda e: finals.append(e.text) if e.is_final else None)
    transport = LoopbackTransport()
    await session.start(Agent("x"), transport)
    await _speak(transport, 0.6, 0.55)  # pause: Smart Turn says the user is not done
    await asyncio.sleep(0.2)
    assert finals == []  # waiting for max_endpointing_delay
    await _speak(transport, 0.6, 0.55)  # the user goes on: same turn, scored again
    await _wait_for(lambda: bool(finals), 3)
    await session.aclose()
    assert finals == ["I would like to book a table"]
    assert [m.end_of_turn for m in eot] == [False, True]
    assert all(m.provider == "smart_turn" for m in eot)
    warmup, first, second = durations  # the session pre-warms the detector at start
    assert warmup == 1.0 and second > first + 0.5  # whole turn re-scored
    assert all(f["input_features"].shape == (1, 80, 800) for f in session_fake.feeds)


# ------------------------------------------------------------------ real model (opt-in)
JFK_URL = (
    "https://raw.githubusercontent.com/ggml-org/whisper.cpp/"
    "b0a11594aec50892a02cd8d129eee2dfe93a8bb8/samples/jfk.wav"
)
JFK_SHA256 = "59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e"


@pytest.fixture(scope="module")
def jfk() -> AudioFrame:
    """JFK, 1961 (public domain): "And so my fellow Americans, ask not what your country can
    do for you, ask what you can do for your country." 11 s, 16 kHz mono."""
    pytest.importorskip("onnxruntime")
    return read_wav(download(JFK_URL, subdir="testdata", sha256=JFK_SHA256))  # shared cache


def utterances(jfk: AudioFrame) -> dict[str, AudioFrame]:
    """Complete vs truncated utterances cut from the recording, each followed by 0.2 s of the
    recording's own room tone (what a VAD pause looks like)."""
    pause = jfk.slice(2.25, 2.45)
    return {
        "complete": AudioFrame.concat([jfk.slice(0, 10.45), pause]),  # "... your country."
        "truncated: ask not": AudioFrame.concat([jfk.slice(0, 4.35), pause]),
        "truncated: ask what you can do": AudioFrame.concat([jfk.slice(0, 9.15), pause]),
    }


@pytest.mark.model
@pytest.mark.parametrize("model", ["smart-turn-v3.2-cpu", "smart-turn-v3.2-gpu"])
async def test_real_model_complete_vs_truncated(model: str, jfk: AudioFrame) -> None:
    detector = SmartTurnDetector(model=model)
    await detector.warmup()
    timings: list[float] = []
    detector.on("metrics", lambda m: timings.append(m.inference_duration))
    probabilities = {
        name: await detector.predict_end_of_turn(audio=clip)
        for name, clip in utterances(jfk).items()
    }
    # other input formats go through resampling / down-mixing and must agree
    stereo_48k = resample(utterances(jfk)["complete"], 48_000).to_channels(2)
    p_48k = await detector.predict_end_of_turn(audio=stereo_48k)
    print(f"{model}: {probabilities}, 48 kHz stereo {p_48k:.3f}, inference {timings}")
    assert probabilities.pop("complete") >= 0.5 and p_48k >= 0.5
    assert all(p < 0.5 for p in probabilities.values()), probabilities
    await detector.aclose()


@pytest.mark.model
async def test_real_model_works_offline_once_cached(
    jfk: AudioFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    await SmartTurnDetector().warmup()  # downloads the model unless it is cached already
    monkeypatch.setenv("VAN_OFFLINE", "1")
    detector = SmartTurnDetector()
    p = await detector.predict_end_of_turn(audio=utterances(jfk)["complete"])
    assert p >= 0.5


@pytest.mark.model
@pytest.mark.parametrize("vad", ["energy", "silero"])
async def test_real_model_in_agent_session(vad: str, jfk: AudioFrame) -> None:
    try:
        vad_component = create("vad", vad)
    except (ProviderNotFoundError, MissingDependencyError) as exc:
        pytest.skip(f"{vad} VAD is not available: {exc}")
    session = AgentSession(
        stt=MockSTT(default_text="words"),
        llm=MockLLM(responses=["Noted."]),
        tts=MockTTS(),
        vad=vad_component,
        turn_detector="smart_turn",
        cascade_options=CascadeOptions(max_endpointing_delay=1.5),
    )
    await session.engine.warmup()
    eot: list[EOTMetrics] = []
    finals: list[str] = []
    session.on("metrics", lambda m: eot.append(m) if isinstance(m, EOTMetrics) else None)
    session.on("user_transcript", lambda e: finals.append(e.text) if e.is_final else None)
    transport = LoopbackTransport()
    await session.start(Agent("x"), transport)
    room_tone = jfk.slice(2.25, 3.15)
    # "ask not" + pause: Smart Turn holds the turn open ...
    await transport.play_user_audio(jfk.slice(3.15, 4.35), realtime=False)
    await transport.play_user_audio(room_tone, realtime=False)
    await _wait_for(lambda: bool(eot), 10)
    assert not eot[0].end_of_turn and finals == []
    # ... the user goes on and finishes the sentence: the turn completes
    await transport.play_user_audio(jfk.slice(8.1, 10.45), realtime=False)
    await transport.play_user_audio(room_tone, realtime=False)
    await _wait_for(lambda: bool(finals), 10)
    await session.aclose()
    print(f"{vad}: end-of-turn probabilities {[round(m.probability, 3) for m in eot]}")
    assert eot[-1].end_of_turn, [m.probability for m in eot]
    assert len(finals) == 1  # one user turn despite the pause after "ask not"


# ---------------------------------------------------------------- golden file generator
def _regenerate_golden() -> None:  # pragma: no cover - run manually, see module docstring
    from transformers import WhisperFeatureExtractor
    from transformers.utils import is_torch_available

    if is_torch_available():
        raise SystemExit("uninstall torch: upstream inference.py runs the numpy code path")
    extractor = WhisperFeatureExtractor(chunk_length=8)
    rows = []
    for pcm in golden_signals().values():
        audio = pcm.astype(np.float32) / 32768.0
        # smart-turn audio_utils.truncate_audio_to_last_n_seconds(audio, n_seconds=8)
        if len(audio) > N_SAMPLES:
            audio = audio[-N_SAMPLES:]
        elif len(audio) < N_SAMPLES:
            audio = np.pad(audio, (N_SAMPLES - len(audio), 0), mode="constant", constant_values=0)
        features = extractor(
            audio,
            sampling_rate=SR,
            return_tensors="np",
            padding="max_length",
            max_length=N_SAMPLES,
            truncation=True,
            do_normalize=True,
        ).input_features[0]
        rows.append(features[:, GOLDEN_FRAMES])
    GOLDEN_PATH.parent.mkdir(exist_ok=True)
    np.save(GOLDEN_PATH, np.stack(rows).astype(np.float32))
    print(f"wrote {GOLDEN_PATH}")


if __name__ == "__main__":
    _regenerate_golden()
