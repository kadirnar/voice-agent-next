"""faster-whisper STT.

Unit tests run offline against fake ``faster_whisper`` / ``ctranslate2`` modules; the
``@pytest.mark.model`` tests load the real ``tiny.en`` model (~75 MB download) and
transcribe a short public-domain clip (``pytest -m model tests/providers``).
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, replace
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

from voice_agent_next import AudioFrame, VADOptions, create, hardware
from voice_agent_next.engines import CascadeEngine
from voice_agent_next.errors import (
    AuthenticationError,
    ConfigurationError,
    MissingDependencyError,
    ProviderConnectionError,
    ProviderError,
    RateLimitError,
)
from voice_agent_next.metrics import STTMetrics
from voice_agent_next.providers.energy import EnergyVAD
from voice_agent_next.providers.faster_whisper import FasterWhisperSTT
from voice_agent_next.providers.mock import synth_speech
from voice_agent_next.registry import get_provider
from voice_agent_next.stt import StreamAdapter, STTEventType
from voice_agent_next.utils.download import DownloadError, download

# ------------------------------------------------------------------------ fakes


@dataclass
class FakeWord:
    start: float
    end: float
    word: str
    probability: float


@dataclass
class FakeSegment:
    start: float
    end: float
    text: str
    tokens: list[int]
    avg_logprob: float
    words: list[FakeWord] | None = None


def _word(start: float, end: float, word: str, p: float) -> FakeWord:
    # faster-whisper reports word times as numpy floats
    return FakeWord(np.float64(start), np.float64(end), word, np.float64(p))  # type: ignore[arg-type]


SEGMENTS = [
    FakeSegment(
        0.0, 1.2, " Hello there.", [1, 2, 3], -0.2,
        [_word(0.0, 0.5, " Hello", 0.9), _word(0.5, 1.2, " there.", 0.8)],
    ),
    FakeSegment(
        1.2, 2.0, " How are you?", [4, 5, 6, 7, 8, 9], -0.5,
        [_word(1.2, 1.4, " How", 0.7), _word(1.4, 1.6, " are", 0.95), _word(1.6, 2.0, " you?", 0.6)],
    ),
]  # fmt: skip
TEXT = "Hello there. How are you?"


class FakeBackend:
    """Stands in for ``faster_whisper`` + ``ctranslate2`` and records every call."""

    def __init__(self) -> None:
        self.cuda_devices = 0
        self.supported = {
            "cpu": {"int8", "int8_float32", "float32"},
            "cuda": {"float16", "int8_float16", "int8", "int8_float32", "float32"},
        }
        self.segments: list[FakeSegment] = list(SEGMENTS)
        self.detected_language = "en"
        self.failures: dict[str, Exception] = {}
        """device -> error raised while decoding (like a missing libcublas)."""
        self.download_error: Exception | None = None
        self.load_delay = 0.0
        self.downloads: list[tuple[str, dict[str, Any]]] = []
        self.loads: list[dict[str, Any]] = []
        self.calls: list[dict[str, Any]] = []

    @property
    def requests(self) -> list[dict[str, Any]]:
        """``transcribe()`` calls for user audio (warm-up inferences excluded)."""
        return [c for c in self.calls if not c.get("without_timestamps")]

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        backend = self

        class WhisperModel:
            def __init__(self, path: str, **kwargs: Any) -> None:
                time.sleep(backend.load_delay)
                backend.loads.append({"path": path, "thread": threading.current_thread(), **kwargs})
                self.device = kwargs["device"]

            def transcribe(self, audio: Any, **kwargs: Any) -> tuple[Any, Any]:
                backend.calls.append({"audio": audio, "device": self.device, **kwargs})

                def decode() -> Any:  # a lazy generator, like the real one
                    if self.device in backend.failures:
                        raise backend.failures[self.device]
                    yield from backend.segments

                info = SimpleNamespace(language=kwargs.get("language") or backend.detected_language)
                return decode(), info

        def download_model(size_or_id: str, **kwargs: Any) -> str:
            if backend.download_error is not None:
                raise backend.download_error
            backend.downloads.append((size_or_id, kwargs))
            return f"/fake-models/{size_or_id}"

        fw = ModuleType("faster_whisper")
        fw.WhisperModel = WhisperModel  # type: ignore[attr-defined]
        fw.download_model = download_model  # type: ignore[attr-defined]
        ct2 = ModuleType("ctranslate2")
        ct2.get_cuda_device_count = lambda: backend.cuda_devices  # type: ignore[attr-defined]
        ct2.get_supported_compute_types = lambda device: set(backend.supported[device])  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "faster_whisper", fw)
        monkeypatch.setitem(sys.modules, "ctranslate2", ct2)


@pytest.fixture
def backend(monkeypatch: pytest.MonkeyPatch) -> FakeBackend:
    monkeypatch.delenv("VAN_OFFLINE", raising=False)
    # the machine's real GPUs and CUDA libraries must not leak into device selection
    monkeypatch.setattr(hardware, "detect_nvidia", lambda: hardware.NvidiaInfo())
    monkeypatch.setattr(hardware, "load_cuda_libraries", _fake_cuda_libraries(loaded=True))
    fake = FakeBackend()
    fake.install(monkeypatch)
    return fake


def _fake_cuda_libraries(*, loaded: bool) -> Any:
    def load(components: Any, cuda_major: int) -> tuple[hardware.CudaLibrary, ...]:
        return tuple(
            replace(
                hardware.find_cuda_libraries((c,), cuda_major, search_path=[])[0], loaded=loaded
            )
            for c in components
        )

    return load


def chunks(frame: AudioFrame, step: float = 0.02) -> list[AudioFrame]:
    n = math.ceil(frame.duration / step - 1e-9)
    return [frame.slice(i * step, (i + 1) * step) for i in range(n)]


def speech(duration: float = 0.5) -> AudioFrame:
    return synth_speech(duration, 16_000)


# --------------------------------------------------------------- transcripts


async def test_segments_become_one_transcript_with_metrics(backend: FakeBackend) -> None:
    stt = FasterWhisperSTT(model="small", device="cpu")
    metrics: list[STTMetrics] = []
    stt.on("metrics", metrics.append)
    # any rate / channel count: the base class resamples to 16 kHz mono
    result = await stt.transcribe(synth_speech(1.0, 48_000).to_channels(2))

    assert result.text == TEXT
    assert result.language == "en"  # detected: no language was requested
    assert (result.start_time, result.end_time) == (0.0, 2.0)
    assert result.confidence == pytest.approx(math.exp((3 * -0.2 + 6 * -0.5) / 9))
    assert result.words is None
    (call,) = backend.requests
    assert call["language"] is None
    assert call["beam_size"] == 1 and call["vad_filter"] is False
    assert call["word_timestamps"] is False
    audio = call["audio"]
    assert audio.dtype == np.float32 and audio.ndim == 1
    assert len(audio) == pytest.approx(16_000, abs=160)
    (m,) = metrics
    assert (m.provider, m.model, m.error) == ("faster_whisper", "small", None)
    assert m.audio_duration == pytest.approx(1.0, abs=0.02) and not m.streamed


async def test_word_timestamps_fill_transcript_words(backend: FakeBackend) -> None:
    stt = FasterWhisperSTT(model="small", device="cpu", word_timestamps=True)
    assert stt.capabilities.word_timestamps
    result = await stt.transcribe(speech())
    assert backend.requests[0]["word_timestamps"] is True
    assert result.words is not None
    assert [w.word for w in result.words] == ["Hello", "there.", "How", "are", "you?"]
    first = result.words[0]
    assert (first.start, first.end, first.confidence) == (0.0, 0.5, 0.9)
    assert all(type(v) is float for w in result.words for v in (w.start, w.end, w.confidence))


async def test_no_speech_and_empty_audio(backend: FakeBackend) -> None:
    backend.segments = []
    stt = FasterWhisperSTT(model="small", device="cpu", word_timestamps=True)
    result = await stt.transcribe(speech())
    assert result.text == "" and result.words == []
    assert result.confidence is None and result.start_time is None and result.end_time is None
    calls = len(backend.calls)
    empty = await stt.transcribe(AudioFrame.empty(16_000))
    assert empty.text == "" and len(backend.calls) == calls  # the model is not invoked


async def test_language_codes_and_per_call_override(backend: FakeBackend) -> None:
    stt = FasterWhisperSTT(model="small", language="de-DE", device="cpu")
    assert stt.language == "de"
    result = await stt.transcribe(speech())
    assert backend.requests[-1]["language"] == "de" and result.language == "de"
    await stt.transcribe(speech(), language="pt_BR")
    assert backend.requests[-1]["language"] == "pt"
    assert FasterWhisperSTT(model="small", language="auto").language is None
    assert FasterWhisperSTT(model="large-v3-turbo").capabilities.language_detection
    assert not FasterWhisperSTT(model="tiny.en").capabilities.language_detection
    assert not FasterWhisperSTT(model="distil-large-v3.5").capabilities.language_detection


async def test_decoding_options_and_passthrough(backend: FakeBackend) -> None:
    stt = FasterWhisperSTT(
        model="small",
        device="cpu",
        beam_size=5,
        vad_filter=True,
        initial_prompt="kubectl, Kubernetes",
        hotwords="Kadir",
        transcribe_options={"temperature": 0.0, "beam_size": 2},
    )
    await stt.transcribe(speech())
    call = backend.requests[-1]
    assert call["beam_size"] == 2  # transcribe_options win
    assert call["temperature"] == 0.0 and call["vad_filter"] is True
    assert call["initial_prompt"] == "kubectl, Kubernetes" and call["hotwords"] == "Kadir"


# ------------------------------------------------------- loading and devices


async def test_model_loads_once_lazily_in_a_worker_thread(backend: FakeBackend) -> None:
    backend.load_delay = 0.05
    stt = FasterWhisperSTT(
        model="small", device="cpu", download_root="model-cache", local_files_only=True
    )
    assert backend.loads == [] and backend.downloads == []  # nothing happens at construction
    results = await asyncio.gather(stt.warmup(), *(stt.transcribe(speech()) for _ in range(4)))

    assert len(backend.downloads) == 1 and len(backend.loads) == 1
    assert backend.downloads[0] == ("small", {"local_files_only": True, "cache_dir": "model-cache"})
    load = backend.loads[0]
    assert load["path"] == "/fake-models/small"
    assert load["thread"] is not threading.main_thread()
    assert (load["cpu_threads"], load["num_workers"], load["device_index"]) == (0, 1, 0)
    warm = backend.calls[0]  # a short warm-up inference follows the load
    assert warm["without_timestamps"] and warm["language"] == "en" and not warm["audio"].any()
    assert [r.text for r in results[1:]] == [TEXT] * 4
    await stt.aclose()


async def test_local_directory_and_offline_mode(
    backend: FakeBackend, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    local = FasterWhisperSTT(model=str(tmp_path), device="cpu")
    await local.warmup()
    assert backend.downloads == [] and backend.loads[-1]["path"] == str(tmp_path)
    monkeypatch.setenv("VAN_OFFLINE", "1")
    await FasterWhisperSTT(model="base", device="cpu").warmup()
    assert backend.downloads[-1][1]["local_files_only"] is True


@pytest.mark.parametrize(
    ("cuda_devices", "cuda_error", "device", "compute_type"),
    [
        (0, None, "cpu", "int8"),  # no GPU
        (1, None, "cuda", "float16"),  # GPU and CUDA libraries present
        # GPU visible to CTranslate2 but unusable (what this repo's dev machine reports)
        (
            1,
            RuntimeError("Library libcublas.so.12 is not found or cannot be loaded"),
            "cpu",
            "int8",
        ),
    ],
)
async def test_auto_device_and_compute_type(
    backend: FakeBackend,
    caplog: pytest.LogCaptureFixture,
    cuda_devices: int,
    cuda_error: Exception | None,
    device: str,
    compute_type: str,
) -> None:
    backend.cuda_devices = cuda_devices
    if cuda_error is not None:
        backend.failures["cuda"] = cuda_error
    stt = FasterWhisperSTT(model="small")  # device="auto", compute_type="auto"
    with caplog.at_level(logging.WARNING, logger="voice_agent_next"):
        await stt.warmup()

    assert (stt.resolved_device, stt.resolved_compute_type) == (device, compute_type)
    tried = ["cuda", "cpu"] if cuda_error else [device]
    assert [load["device"] for load in backend.loads] == tried
    assert backend.loads[-1]["compute_type"] == compute_type
    assert ("falling back to CPU" in caplog.text) is (cuda_error is not None)
    assert (await stt.transcribe(speech())).text == TEXT
    assert backend.requests[-1]["device"] == device


async def test_auto_stays_on_cpu_when_cublas_is_missing(
    backend: FakeBackend, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The GPU is never tried without its libraries, and the log says how to fix it."""
    backend.cuda_devices = 1
    sys.modules["ctranslate2"].__version__ = "4.8.2"  # 4.x wheels: CUDA 12
    monkeypatch.setattr(hardware, "load_cuda_libraries", _fake_cuda_libraries(loaded=False))
    stt = FasterWhisperSTT(model="small")
    with caplog.at_level(logging.INFO, logger="voice_agent_next"):
        await stt.warmup()
    assert (stt.resolved_device, stt.resolved_compute_type) == ("cpu", "int8")
    assert [load["device"] for load in backend.loads] == ["cpu"]
    messages = [r.getMessage() for r in caplog.records if "running on CPU" in r.getMessage()]
    assert len(messages) == 1
    assert "libcublas" in messages[0] or "cublas64_12.dll" in messages[0]
    assert hardware.CUDA_EXTRA_HINT in messages[0]
    assert "falling back" not in caplog.text


async def test_cuda_libraries_are_loaded_before_the_model(
    backend: FakeBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend.cuda_devices = 1
    sys.modules["ctranslate2"].__version__ = "4.8.2"
    calls: list[tuple[tuple[str, ...], int]] = []
    loader = _fake_cuda_libraries(loaded=True)

    def load(components: Any, cuda_major: int) -> Any:
        calls.append((tuple(components), cuda_major))
        return loader(components, cuda_major)

    monkeypatch.setattr(hardware, "load_cuda_libraries", load)
    for device in ("auto", "cuda"):
        stt = FasterWhisperSTT(model="small", device=device)
        await stt.warmup()
        assert (stt.resolved_device, stt.resolved_compute_type) == ("cuda", "float16")
    assert calls == [(("cublas",), 12)] * 2
    cpu = FasterWhisperSTT(model="small", device="cpu")  # explicit CPU loads nothing
    await cpu.warmup()
    assert len(calls) == 2


async def test_compute_type_follows_what_the_device_supports(backend: FakeBackend) -> None:
    backend.cuda_devices = 1
    # e.g. a Pascal GPU without fast float16, and a CPU build without int8 kernels
    backend.supported = {"cpu": {"float32"}, "cuda": {"int8", "int8_float32", "float32"}}
    gpu = FasterWhisperSTT(model="small", device="cuda")
    await gpu.warmup()
    cpu = FasterWhisperSTT(model="small", device="cpu")
    await cpu.warmup()
    assert (gpu.resolved_compute_type, cpu.resolved_compute_type) == ("int8", "float32")
    explicit = FasterWhisperSTT(model="small", device="cuda", compute_type="int8_float16")
    await explicit.warmup()
    assert backend.loads[-1]["compute_type"] == "int8_float16"  # used as given


async def test_forced_cuda_does_not_fall_back(backend: FakeBackend) -> None:
    backend.cuda_devices = 1
    backend.failures["cuda"] = RuntimeError("CUDA failed with error out of memory")
    stt = FasterWhisperSTT(model="small", device="cuda")
    with pytest.raises(ProviderError, match="out of memory") as info:
        await stt.warmup()
    assert info.value.provider == "faster_whisper"
    assert [load["device"] for load in backend.loads] == ["cuda"]
    assert stt.resolved_device is None


# --------------------------------------------------------------------- errors


class RepositoryNotFoundError(Exception):
    pass


class GatedRepoError(RepositoryNotFoundError):
    pass


class LocalEntryNotFoundError(FileNotFoundError):
    pass


class HfHubHTTPError(Exception):
    def __init__(self, message: str, status: int) -> None:
        super().__init__(message)
        self.response = SimpleNamespace(status_code=status)


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (ValueError("Invalid model size 'huge'"), ConfigurationError),
        (RepositoryNotFoundError("404 Client Error: Repository Not Found"), ConfigurationError),
        (GatedRepoError("Access to model is restricted"), AuthenticationError),
        (HfHubHTTPError("401 Unauthorized", 401), AuthenticationError),
        (HfHubHTTPError("429 Too Many Requests", 429), RateLimitError),
        (HfHubHTTPError("502 Bad Gateway", 502), ProviderConnectionError),
        (LocalEntryNotFoundError("outgoing traffic has been disabled"), ProviderConnectionError),
        (ConnectionError("network is unreachable"), ProviderConnectionError),
    ],
)
async def test_download_errors_are_mapped_and_retried(
    backend: FakeBackend, error: Exception, expected: type[Exception]
) -> None:
    backend.download_error = error
    stt = FasterWhisperSTT(model="huge", device="cpu")
    with pytest.raises(expected) as info:
        await stt.warmup()
    assert info.value.__cause__ is error
    backend.download_error = None
    await stt.warmup()  # a failed load is retried on the next call
    assert stt.resolved_device == "cpu"


async def test_inference_errors_are_mapped(backend: FakeBackend) -> None:
    stt = FasterWhisperSTT(model="small", device="cpu")
    metrics: list[STTMetrics] = []
    stt.on("metrics", metrics.append)
    await stt.warmup()
    backend.failures["cpu"] = RuntimeError("std::bad_alloc")
    with pytest.raises(ProviderError, match="bad_alloc"):
        await stt.transcribe(speech())
    assert metrics[-1].error is not None and "bad_alloc" in metrics[-1].error
    backend.failures["cpu"] = ValueError("'xx' is not a valid language code")
    with pytest.raises(ConfigurationError, match="valid language"):
        await stt.transcribe(speech(), language="xx")


def test_constructor_validation_and_defaults(backend: FakeBackend) -> None:
    with pytest.raises(ConfigurationError, match="device"):
        FasterWhisperSTT(device="tpu")
    with pytest.raises(ConfigurationError, match="beam_size"):
        FasterWhisperSTT(beam_size=0)
    stt = FasterWhisperSTT()
    assert (stt.model, stt.device, stt.compute_type, stt.beam_size) == (
        "large-v3-turbo",
        "auto",
        "auto",
        1,
    )
    assert stt.sample_rate == 16_000
    assert not stt.capabilities.streaming and not stt.capabilities.interim_results


def test_missing_dependency_has_an_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "faster_whisper", None)  # import fails
    with pytest.raises(MissingDependencyError, match=r"voice-agent-next\[faster-whisper\]"):
        FasterWhisperSTT(model="small")


# ------------------------------------------------------ registry and pipeline


def test_registry_spec_and_alias(backend: FakeBackend) -> None:
    stt = create("stt", "faster_whisper/tiny.en", device="cpu")
    assert isinstance(stt, FasterWhisperSTT) and stt.model == "tiny.en" and stt.device == "cpu"
    assert create("stt", "faster-whisper").model == "large-v3-turbo"
    aliased = create("stt", {"provider": "whisper/base", "language": "en"})
    assert isinstance(aliased, FasterWhisperSTT) and aliased.model == "base"
    spec = get_provider("stt", "whisper")
    assert spec.name == "faster_whisper" and spec.local and spec.env == ()
    assert spec.extra == "faster-whisper" and spec.default_model == "large-v3-turbo"
    assert spec.requires == ("faster_whisper", "ctranslate2")


def test_whisper_alias_resolves_in_a_fresh_interpreter() -> None:
    code = "from voice_agent_next.registry import get_provider; print(get_provider('stt', 'whisper').name)"
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60, check=True
    )
    assert out.stdout.strip() == "faster_whisper"


async def test_streams_through_stream_adapter_with_a_vad(backend: FakeBackend) -> None:
    stt = FasterWhisperSTT(model="small", device="cpu")
    adapter = StreamAdapter(stt, EnergyVAD(options=VADOptions(min_silence_duration=0.3)))
    stream = adapter.stream()
    silence = AudioFrame.silence(0.5, 16_000)
    for frame in chunks(silence) + chunks(speech(0.8)) + chunks(silence):
        stream.push_audio(frame)
    stream.end_input()
    events = [e async for e in stream]
    kinds = [e.type for e in events]
    assert [e.text for e in events if e.type == STTEventType.FINAL_TRANSCRIPT] == [TEXT]
    assert kinds.count(STTEventType.START_OF_SPEECH) == 1
    assert kinds.count(STTEventType.END_OF_SPEECH) == 1
    await adapter.aclose()


def test_cascade_wraps_it_in_a_stream_adapter(backend: FakeBackend) -> None:
    engine = CascadeEngine(stt="faster_whisper/small", vad="energy", llm="mock", tts="mock")
    assert isinstance(engine.stt, StreamAdapter)
    assert isinstance(engine.stt.wrapped, FasterWhisperSTT)


# ------------------------------------------------------ real model (opt-in)

JFK_URL = "https://github.com/openai/whisper/raw/main/tests/jfk.flac"
JFK_SHA256 = "63a4b1e4c1dc655ac70961ffbf518acd249df237e5a0152faae9a4a836949715"
# JFK's 1961 inaugural address: a public-domain, 11 s clip (Whisper's own test audio)


def _jfk_clip() -> AudioFrame:
    fw = pytest.importorskip("faster_whisper")
    try:
        path = download(JFK_URL, filename="jfk.flac", subdir="test-audio", sha256=JFK_SHA256)
    except DownloadError as exc:
        pytest.skip(f"test clip unavailable: {exc}")
    return AudioFrame.from_numpy(fw.decode_audio(str(path), sampling_rate=16_000), 16_000)


async def _loaded(stt: FasterWhisperSTT) -> FasterWhisperSTT:
    try:
        await stt.warmup()
    except ProviderConnectionError as exc:
        pytest.skip(f"model unavailable: {exc}")
    return stt


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z ]+", "", text.lower())


@pytest.mark.model
async def test_tiny_en_transcribes_a_public_domain_clip() -> None:
    clip = _jfk_clip()
    stt = await _loaded(FasterWhisperSTT(model="tiny.en", device="cpu", word_timestamps=True))
    assert (stt.resolved_device, stt.resolved_compute_type) == ("cpu", "int8")
    result = await stt.transcribe(clip)
    assert "ask not what your country can do for you" in _normalize(result.text)
    assert result.language == "en"
    assert result.confidence is not None and 0.3 < result.confidence <= 1.0
    assert result.words and _normalize(result.words[0].word) == "and"
    assert all(0.0 <= w.start <= w.end <= clip.duration + 0.5 for w in result.words)
    await stt.aclose()


@pytest.mark.model
async def test_tiny_en_streams_on_the_auto_device() -> None:
    """``device="auto"`` lands on a working device (CPU when the CUDA libraries are missing)."""
    clip = _jfk_clip()
    stt = await _loaded(FasterWhisperSTT(model="tiny.en"))
    assert stt.resolved_device in ("cpu", "cuda")
    adapter = StreamAdapter(stt, EnergyVAD())
    stream = adapter.stream()
    for frame in chunks(clip) + chunks(AudioFrame.silence(1.0, 16_000)):
        stream.push_audio(frame)
    stream.end_input()
    finals = [e.text async for e in stream if e.type == STTEventType.FINAL_TRANSCRIPT]
    assert "your country" in _normalize(" ".join(finals))
    await adapter.aclose()
