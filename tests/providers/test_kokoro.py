"""Kokoro TTS provider.

Unit tests replace ``kokoro_onnx`` and ``onnxruntime`` with in-memory fakes (no model,
no network). ``test_real_model_*`` downloads the int8 model (~142 MB with the voice pack;
fp16, ~191 MB, on ARM64) and only runs with ``-m model``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import threading
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from voice_agent_next import create, hardware
from voice_agent_next.audio import AudioFrame
from voice_agent_next.errors import ConfigurationError, MissingDependencyError, ProviderError
from voice_agent_next.metrics import TTSMetrics
from voice_agent_next.providers import kokoro as kokoro_module
from voice_agent_next.providers.kokoro import (
    KOKORO_MODELS,
    KokoroTTS,
    lang_for_voice,
    select_execution_providers,
)
from voice_agent_next.registry import get_provider
from voice_agent_next.tts import SentenceStreamAdapter
from voice_agent_next.utils.clock import now

SR = 24_000
CPU = "CPUExecutionProvider"


# ------------------------------------------------------------------------------ fakes
class FakeKokoro:
    """Stands in for ``kokoro_onnx.Kokoro``: 10 ms of tone per input character."""

    def __init__(self, backend: FakeBackend, session: Any, voices_path: str) -> None:
        self.backend = backend
        self.session = session
        self.voices_path = voices_path
        self.voices = dict.fromkeys(backend.voices)
        self.calls: list[dict[str, Any]] = []

    def get_voices(self) -> list[str]:
        return sorted(self.voices)

    def create(
        self,
        text: str,
        voice: str,
        speed: float = 1.0,
        lang: str = "en-us",
        is_phonemes: bool = False,
        trim: bool = True,
        sentence_pause: float = 0.25,
        clause_pause: float = 0.1,
        continuous: bool = False,
    ) -> tuple[np.ndarray, int]:
        self.calls.append(
            {
                "text": text,
                "voice": voice,
                "speed": speed,
                "lang": lang,
                "is_phonemes": is_phonemes,
                "trim": trim,
                "sentence_pause": sentence_pause,
                "clause_pause": clause_pause,
                "thread": threading.current_thread().name,
            }
        )
        if self.backend.delay:
            time.sleep(self.backend.delay)
        if self.backend.error is not None:
            raise self.backend.error
        if self.backend.session_output is not None:
            self.session.run(None, {"tokens": text})  # the real one runs the model
        n = round(0.01 * SR * len(text) / speed)
        tone = 0.5 * np.sin(2 * np.pi * 220.0 * np.arange(n) / SR)
        return tone.astype(np.float32), SR


@dataclass
class FakeEspeakConfig:
    lib_path: str | None = None
    data_path: str | None = None


class FakeBackend:
    """Fake ``onnxruntime``, ``kokoro_onnx`` and ``espeakng_loader`` modules and what was
    done with them."""

    def __init__(self, espeak_data: Path) -> None:
        self.available = [CPU]
        self.failing_providers: set[str] = set()
        self.voices = ("af_heart", "bf_emma", "ef_dora", "ff_siwis", "zf_001")
        self.espeak_data = espeak_data
        self.sessions: list[Any] = []
        self.engines: list[FakeKokoro] = []
        self.espeak_configs: list[FakeEspeakConfig | None] = []
        self.downloads: list[tuple[str, str, str | None]] = []
        self.error: Exception | None = None
        self.session_output: np.ndarray | None = None
        """What the fake ONNX session returns as audio (``None``: the model is not run)."""
        self.delay = 0.0

    @property
    def engine(self) -> FakeKokoro:
        return self.engines[-1]

    def modules(self) -> tuple[types.ModuleType, types.ModuleType, types.ModuleType]:
        backend = self

        class SessionOptions:
            def __init__(self) -> None:
                self.intra_op_num_threads = 0

        class InferenceSession:
            def __init__(
                self, path: str, sess_options: Any = None, providers: list[Any] | None = None
            ) -> None:
                self._model_path = path
                self.options = sess_options
                self.providers = list(providers or [])
                backend.sessions.append(self)
                if self.providers and self.providers[0] in backend.failing_providers:
                    raise RuntimeError(f"{self.providers[0]} is not usable here")

            def get_providers(self) -> list[str]:
                return [p if isinstance(p, str) else p[0] for p in self.providers]

            def run(self, output_names: Any, feed: Any, run_options: Any = None) -> list[Any]:
                return [backend.session_output, np.array([1, 2])]

        class Kokoro:
            @classmethod
            def from_session(
                cls,
                session: Any,
                voices_path: str,
                espeak_config: FakeEspeakConfig | None = None,
                vocab_config: Any = None,
            ) -> FakeKokoro:
                backend.espeak_configs.append(espeak_config)
                engine = FakeKokoro(backend, session, voices_path)
                backend.engines.append(engine)
                return engine

        ort = types.ModuleType("onnxruntime")
        ort.SessionOptions = SessionOptions  # type: ignore[attr-defined]
        ort.InferenceSession = InferenceSession  # type: ignore[attr-defined]
        ort.get_available_providers = lambda: list(backend.available)  # type: ignore[attr-defined]
        kokoro_onnx = types.ModuleType("kokoro_onnx")
        kokoro_onnx.Kokoro = Kokoro  # type: ignore[attr-defined]
        kokoro_onnx.EspeakConfig = FakeEspeakConfig  # type: ignore[attr-defined]

        def get_data_path() -> str:
            if not backend.espeak_data.is_dir():
                raise RuntimeError(f"data path not exists at {backend.espeak_data}")
            return str(backend.espeak_data)

        loader = types.ModuleType("espeakng_loader")
        loader.get_data_path = get_data_path  # type: ignore[attr-defined]
        return ort, kokoro_onnx, loader


def fake_espeak_data(path: Path) -> Path:
    path.mkdir(parents=True)
    (path / "phontab").write_bytes(b"phonemes")
    (path / "voices").mkdir()
    (path / "voices" / "en").write_bytes(b"voice")
    return path


@pytest.fixture
def backend(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> FakeBackend:
    fake = FakeBackend(fake_espeak_data(tmp_path / "espeak-ng-data"))
    ort, kokoro_onnx, loader = fake.modules()
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)
    monkeypatch.setitem(sys.modules, "kokoro_onnx", kokoro_onnx)
    monkeypatch.setitem(sys.modules, "espeakng_loader", loader)
    monkeypatch.setenv("VAN_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(hardware, "detect_nvidia", lambda: hardware.NvidiaInfo())

    def fake_download(url: str, *, subdir: str = "", sha256: str | None = None, **_: Any) -> Path:
        fake.downloads.append((url, subdir, sha256))
        path = tmp_path / url.rsplit("/", 1)[-1]
        path.touch()
        return path

    monkeypatch.setattr(kokoro_module, "download", fake_download)
    return fake


async def synthesize(tts: KokoroTTS, text: str, **kwargs: Any) -> list[AudioFrame]:
    return [chunk.frame async for chunk in tts.synthesize(text, **kwargs) if chunk.frame]


# ------------------------------------------------------------------ registration/config
def test_registered_with_metadata() -> None:
    spec = get_provider("tts", "kokoro")
    assert spec.factory is KokoroTTS
    assert spec.default_model == "v1.0"
    assert spec.extra == "kokoro"
    assert spec.local
    assert spec.env == ()
    assert set(spec.requires) == {"kokoro_onnx", "onnxruntime"}
    assert set(spec.models) == set(KOKORO_MODELS)


def test_missing_dependency_is_reported_with_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "kokoro_onnx", None)
    with pytest.raises(MissingDependencyError, match=r"voice-agent-next\[kokoro\]"):
        KokoroTTS()


def test_create_from_spec_and_model_aliases(backend: FakeBackend) -> None:
    tts = create("tts", "kokoro/kokoro-v1.0.int8.onnx", voice="bf_emma", speed=1.2)
    assert isinstance(tts, KokoroTTS)
    assert (tts.model, tts.voice, tts.speed, tts.sample_rate) == ("v1.0-int8", "bf_emma", 1.2, SR)
    default = create("tts", "kokoro")
    assert (default.model, default.voice) == ("v1.0", "af_heart")
    assert KokoroTTS(model="INT8").model == "v1.0-int8"
    assert KokoroTTS(model="v1.0-fp32").model == "v1.0"
    assert KokoroTTS(model="v1.1-zh.fp16").model == "v1.1-zh-fp16"
    assert KokoroTTS(model="v1.1-zh").voice == "zf_001"
    assert backend.sessions == []  # construction is cheap: nothing is loaded yet


@pytest.mark.parametrize(
    "kwargs",
    [
        {"model": "v9-turbo"},
        {"speed": 0.4},
        {"speed": 2.5},
        {"chunk_duration": 0},
        {"num_threads": 0},
        {"sentence_pause": -0.1},
    ],
)
def test_invalid_options_raise_configuration_error(
    backend: FakeBackend, kwargs: dict[str, Any]
) -> None:
    with pytest.raises(ConfigurationError):
        KokoroTTS(**kwargs)


@pytest.mark.parametrize(
    ("voice", "lang"),
    [
        ("af_heart", "en-us"),
        ("bm_george", "en-gb"),
        ("ef_dora", "es"),
        ("ff_siwis", "fr-fr"),
        ("hf_alpha", "hi"),
        ("im_nicola", "it"),
        ("jf_alpha", "ja"),
        ("pm_alex", "pt-br"),
        ("zf_xiaobei", "cmn"),
        ("custom", "en-us"),
    ],
)
def test_lang_for_voice(voice: str, lang: str) -> None:
    assert lang_for_voice(voice) == lang


@pytest.mark.parametrize(
    ("available", "expected"),
    [
        ([CPU], [CPU]),
        (["AzureExecutionProvider", CPU], [CPU]),
        (
            ["TensorrtExecutionProvider", "CUDAExecutionProvider", CPU],
            ["CUDAExecutionProvider", CPU],
        ),
        (
            ["CoreMLExecutionProvider", "AzureExecutionProvider", CPU],
            ["CoreMLExecutionProvider", CPU],
        ),
        (["DmlExecutionProvider", CPU], ["DmlExecutionProvider", CPU]),
    ],
)
def test_select_execution_providers(available: list[str], expected: list[str]) -> None:
    assert select_execution_providers(available) == expected


# ------------------------------------------------------------------------- synthesis
async def test_synthesize_emits_small_chunks_and_passes_options(backend: FakeBackend) -> None:
    tts = KokoroTTS(
        voice="bf_emma", speed=1.25, chunk_duration=0.04, sentence_pause=0.2, clause_pause=0.05
    )
    text = "Good morning, everyone. How are you today?"
    chunks = [chunk async for chunk in tts.synthesize(text)]
    frames = [c.frame for c in chunks if c.frame]
    assert chunks[0].text == text
    assert chunks[-1].is_final and not chunks[-1].frame
    assert all(f.sample_rate == SR and f.channels == 1 for f in frames)
    assert all(f.duration <= 0.04 + 1e-9 for f in frames)
    sentences = ["Good morning, everyone.", "How are you today?"]
    calls = backend.engine.calls
    assert [c["text"] for c in calls] == sentences  # one inference per sentence
    for call in calls:
        assert (call["voice"], call["speed"], call["lang"]) == ("bf_emma", 1.25, "en-gb")
        assert (call["sentence_pause"], call["clause_pause"], call["trim"]) == (0.2, 0.05, True)
        assert call["thread"].startswith("kokoro")  # never on the event loop thread
    # each sentence: 10 ms per character at speed 1.25, then the 0.2 s sentence pause
    expected = sum(len(s) * 0.01 / 1.25 + 0.2 for s in sentences)
    assert sum(f.duration for f in frames) == pytest.approx(expected, abs=1e-3)
    await tts.aclose()


async def test_voice_override_and_explicit_lang(backend: FakeBackend) -> None:
    tts = KokoroTTS()
    await synthesize(tts, "Hola, ¿qué tal?", voice="ef_dora")
    french = KokoroTTS(voice="ff_siwis", lang="fr-fr")
    await synthesize(french, "Bonjour à tous.")
    await synthesize(french, "Hello everyone.", voice="af_heart")  # explicit lang wins
    calls = [(c["voice"], c["lang"]) for e in backend.engines for c in e.calls]
    assert calls == [("ef_dora", "es"), ("ff_siwis", "fr-fr"), ("af_heart", "fr-fr")]


async def test_trailing_pause_follows_final_punctuation(backend: FakeBackend) -> None:
    tts = KokoroTTS(sentence_pause=0.3, clause_pause=0.1, split_sentences=False)
    for text, pause in [
        ("It works.", 0.3),
        ('He said "yes!"', 0.3),
        ("你好。", 0.3),
        ("Well, maybe;", 0.1),
        ("No punctuation", 0.0),
    ]:
        frames = await synthesize(tts, text)
        assert sum(f.duration for f in frames) == pytest.approx(len(text) * 0.01 + pause, abs=1e-3)


async def test_split_sentences_can_be_disabled(backend: FakeBackend) -> None:
    tts = KokoroTTS(split_sentences=False)
    await synthesize(tts, "One sentence here. And another one!")
    assert [c["text"] for c in backend.engine.calls] == ["One sentence here. And another one!"]


async def test_g2p_hook_sends_phonemes(backend: FakeBackend) -> None:
    tts = KokoroTTS(voice="zf_001", g2p=lambda text, lang: f"<{lang}>{text}")
    await synthesize(tts, "Hello there.")
    call = backend.engine.calls[0]
    assert (call["text"], call["is_phonemes"]) == ("<cmn>Hello there.", True)


async def test_empty_or_unpronounceable_text_loads_nothing(backend: FakeBackend) -> None:
    tts = KokoroTTS()
    assert await synthesize(tts, "   ") == []
    assert await synthesize(tts, "🙂 ...") == []
    assert backend.sessions == []


async def test_model_is_loaded_once_from_pinned_downloads(backend: FakeBackend) -> None:
    tts = KokoroTTS(model="v1.0-int8")
    await asyncio.gather(synthesize(tts, "First one."), synthesize(tts, "Second one."))
    assert len(backend.sessions) == len(backend.engines) == 1
    variant = KOKORO_MODELS["v1.0-int8"]
    subdir = "kokoro/model-files-v1.1"
    assert backend.downloads == [
        (variant.onnx.url, subdir, variant.onnx.sha256),
        (variant.voices.url, subdir, variant.voices.sha256),
    ]
    assert variant.onnx.url == (
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/"
        "model-files-v1.1/kokoro-v1.0.int8.onnx"
    )
    assert all(len(m.onnx.sha256) == len(m.voices.sha256) == 64 for m in KOKORO_MODELS.values())


async def test_local_files_skip_downloads(backend: FakeBackend, tmp_path: Path) -> None:
    model_file, voices_file = tmp_path / "custom.onnx", tmp_path / "voices.bin"
    model_file.touch()
    voices_file.touch()
    tts = KokoroTTS(model="my-finetune", model_path=model_file, voices_path=voices_file)
    await synthesize(tts, "Hi there.")
    assert backend.downloads == []
    assert backend.sessions[0]._model_path == str(model_file)
    assert backend.engine.voices_path == str(voices_file)
    assert tts.model == "my-finetune"

    missing = KokoroTTS(model_path=tmp_path / "missing.onnx", voices_path=voices_file)
    with pytest.raises(ConfigurationError, match="not found"):
        await synthesize(missing, "Hi there.")


async def test_short_espeak_data_path_is_used_as_is(backend: FakeBackend) -> None:
    await KokoroTTS().warmup()
    assert backend.espeak_configs == [None]  # kokoro-onnx's default espeak-ng setup


async def test_long_espeak_data_path_is_copied_to_the_cache(
    backend: FakeBackend, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # espeak-ng truncates longer data paths, cannot find its data and exits the process
    backend.espeak_data = fake_espeak_data(tmp_path / ("site-packages-" * 3) / "espeak-ng-data")
    cache = (tmp_path / "cache").resolve()
    limit = len(str(cache)) + 30  # the cache copy fits, the original path does not
    assert len(str(backend.espeak_data.resolve())) > limit
    monkeypatch.setattr(kokoro_module, "_ESPEAK_MAX_PATH", limit)

    await KokoroTTS().warmup()
    await KokoroTTS().warmup()  # a second load reuses the copy
    first, second = backend.espeak_configs
    assert first is not None and first == second
    copy = Path(str(first.data_path))
    assert copy.parent == cache and copy.name.startswith("espeak-ng-data-")
    assert (copy / "phontab").read_bytes() == b"phonemes"
    assert (copy / "voices" / "en").is_file()
    assert not list(cache.glob(".espeak-ng-data-*"))  # no temporary leftovers

    monkeypatch.setattr(kokoro_module, "_ESPEAK_MAX_PATH", 10)  # even the cache is too deep
    with pytest.raises(ProviderError, match="espeak-ng cannot open data paths"):
        await KokoroTTS().warmup()


async def test_custom_g2p_does_not_need_espeak(backend: FakeBackend, tmp_path: Path) -> None:
    backend.espeak_data = tmp_path / "missing"
    with pytest.raises(ProviderError, match="espeak-ng data not found"):
        await KokoroTTS().warmup()
    await KokoroTTS(g2p=lambda text, lang: text).warmup()
    assert backend.espeak_configs == [None]


async def test_session_prefers_accelerator_and_falls_back_to_cpu(backend: FakeBackend) -> None:
    backend.available = ["CUDAExecutionProvider", CPU]
    tts = KokoroTTS(num_threads=2)
    await tts.warmup()
    session = backend.sessions[0]
    assert session.providers == ["CUDAExecutionProvider", CPU]
    assert session.options.intra_op_num_threads == 2

    backend.failing_providers = {"CUDAExecutionProvider"}
    await KokoroTTS().warmup()
    assert [s.providers for s in backend.sessions[1:]] == [["CUDAExecutionProvider", CPU], [CPU]]

    explicit = KokoroTTS(providers="CUDAExecutionProvider")  # explicit choices are kept
    with pytest.raises(ProviderError, match="CUDAExecutionProvider is not usable"):
        await explicit.warmup()


async def test_cpu_only_onnxruntime_on_an_nvidia_machine_says_how_to_use_the_gpu(
    backend: FakeBackend, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    gpu = hardware.GPU(0, "NVIDIA GeForce RTX 5070 Ti", 16303, (12, 0))
    monkeypatch.setattr(hardware, "detect_nvidia", lambda: hardware.NvidiaInfo((gpu,)))
    backend.available = [CPU]
    with caplog.at_level(logging.INFO, logger="voice_agent_next"):
        await KokoroTTS().warmup()
    assert backend.sessions[0].providers == [CPU]
    (message,) = [r.getMessage() for r in caplog.records if "running on CPU" in r.getMessage()]
    assert "RTX 5070 Ti" in message and hardware.ONNXRUNTIME_GPU_HINT in message


async def test_cuda_is_skipped_when_its_libraries_are_missing(
    backend: FakeBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A GPU build of ONNX Runtime without cuDNN & co. would fail: stay on CPU instead."""
    backend.available = ["CUDAExecutionProvider", CPU]
    sys.modules["onnxruntime"].cuda_version = "12.8"  # type: ignore[attr-defined]
    requested: list[tuple[tuple[str, ...], int]] = []

    def load(components: Any, cuda_major: int) -> tuple[hardware.CudaLibrary, ...]:
        requested.append((tuple(components), cuda_major))
        return hardware.find_cuda_libraries(components, cuda_major, search_path=[])

    monkeypatch.setattr(hardware, "load_cuda_libraries", load)
    await KokoroTTS().warmup()
    assert [s.providers for s in backend.sessions] == [[CPU]]
    assert requested == [(hardware.ONNXRUNTIME_CUDA_LIBRARIES, 12)]


async def test_unknown_voice_is_a_configuration_error(backend: FakeBackend) -> None:
    tts = KokoroTTS(voice="xx_nobody")
    with pytest.raises(ConfigurationError, match="unknown Kokoro voice 'xx_nobody'"):
        await synthesize(tts, "Hello there.")


async def test_inference_failure_is_a_provider_error(backend: FakeBackend) -> None:
    metrics: list[TTSMetrics] = []
    tts = KokoroTTS()
    tts.on("metrics", metrics.append)
    backend.error = RuntimeError("onnxruntime exploded")
    with pytest.raises(ProviderError, match="onnxruntime exploded"):
        await synthesize(tts, "Hello there.")
    assert metrics[-1].error is not None


async def test_text_without_phonemes_is_skipped(backend: FakeBackend) -> None:
    backend.error = ValueError("Nothing to synthesize, 'xyz' produced no phonemes")
    tts = KokoroTTS()
    assert await synthesize(tts, "Hmm, xyz.") == []
    backend.error = ValueError("something else went wrong")
    with pytest.raises(ProviderError):
        await synthesize(tts, "Hmm, xyz.")


@pytest.mark.parametrize("arm64", [False, True])
async def test_non_finite_model_output_is_a_clear_error(
    backend: FakeBackend,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    arm64: bool,
) -> None:
    """int8 on ARM64 CPUs (Apple Silicon): a NaN in the harmonic source phase makes
    DynamicQuantizeLinear's scale NaN, so all of the audio is NaN. kokoro-onnx would trim it
    to nothing and fail in numpy; the provider names the problem and the way out."""
    monkeypatch.setattr(kokoro_module, "_ARM64", arm64)
    tts = KokoroTTS(model="v1.0-int8")
    backend.session_output = np.full(2400, 0.1, dtype=np.float32)
    assert await synthesize(tts, "Hello.")  # finite audio passes through the check
    backend.session_output = np.full(2400, np.nan, dtype=np.float32)
    with pytest.raises(ProviderError, match="non-finite audio") as info:
        await synthesize(tts, "Hello.")
    assert ("v1.0-fp16" in str(info.value)) is arm64
    assert ("may produce NaN audio" in caplog.text) is arm64
    await tts.aclose()


async def test_metrics_are_emitted(backend: FakeBackend) -> None:
    metrics: list[TTSMetrics] = []
    tts = KokoroTTS(model="v1.0-int8")
    tts.on("metrics", metrics.append)
    await synthesize(tts, "Hello world.")
    m = metrics[-1]
    assert (m.provider, m.model, m.characters, m.streamed) == ("kokoro", "v1.0-int8", 12, False)
    assert m.ttfb is not None and m.error is None
    assert m.audio_duration == pytest.approx(0.12 + 0.25, abs=1e-3)


async def test_stream_uses_the_sentence_adapter(backend: FakeBackend) -> None:
    tts = KokoroTTS()
    stream = tts.stream()
    assert isinstance(stream, SentenceStreamAdapter)
    stream.push_text("Hello there, my friend. How ")
    stream.push_text("are you today?")
    stream.end_input()
    events = [e async for e in stream]
    await stream.aclose()
    assert [e.text for e in events if e.text] == ["Hello there, my friend.", "How are you today?"]
    assert events[-1].is_final
    assert [c["text"] for c in backend.engine.calls] == [
        "Hello there, my friend.",
        "How are you today?",
    ]


async def test_closing_a_stream_stops_the_remaining_sentences(backend: FakeBackend) -> None:
    backend.delay = 0.05
    tts = KokoroTTS()
    stream = tts.synthesize("First sentence here. Second sentence here. Third sentence here.")
    first = await stream.__anext__()
    assert first.frame
    await stream.aclose()
    await asyncio.sleep(0.2)  # let the worker finish the sentence it had started
    assert len(backend.engine.calls) <= 2
    await tts.aclose()


async def test_list_voices_and_reuse_after_aclose(backend: FakeBackend) -> None:
    tts = KokoroTTS()
    assert "af_heart" in await tts.list_voices()
    await tts.aclose()
    assert await synthesize(tts, "Still works.")  # the worker is recreated on demand
    await tts.aclose()


# ------------------------------------------------------------------- real model
@pytest.mark.model
@pytest.mark.timeout(900)  # the first run downloads the model
async def test_real_model_synthesizes_a_sentence() -> None:
    """int8 by default (smallest download, but slow on x86 CPUs); pick another model with
    ``VAN_KOKORO_TEST_MODEL=v1.0 uv run pytest -m model tests/providers/test_kokoro.py -s``.
    """
    pytest.importorskip("kokoro_onnx")
    # the int8 export produces NaN audio with ONNX Runtime's ARM64 kernels (e.g. Apple
    # Silicon runners), see test_non_finite_model_output_is_a_clear_error
    default = "v1.0-fp16" if kokoro_module._ARM64 else "v1.0-int8"
    model = os.environ.get("VAN_KOKORO_TEST_MODEL", default)
    tts = KokoroTTS(model=model)
    metrics: list[TTSMetrics] = []
    tts.on("metrics", metrics.append)
    await tts.warmup()
    voices = await tts.list_voices()
    assert tts.voice in voices
    if tts.model.startswith("v1.0"):
        assert len(voices) == 54

    text = "Hello! This is Kokoro, speaking from a local ONNX model."
    t0 = now()
    ttfb: float | None = None
    frames: list[AudioFrame] = []
    async for chunk in tts.synthesize(text):
        if chunk.frame:
            if ttfb is None:
                ttfb = now() - t0
            frames.append(chunk.frame)
    elapsed = now() - t0
    await tts.aclose()

    audio = AudioFrame.concat(frames)
    assert audio.sample_rate == SR
    assert all(f.duration <= 0.05 + 1e-9 for f in frames)
    assert 1.5 < audio.duration < 10.0
    assert audio.rms() > 0.01  # speech, not silence
    assert metrics[-1].error is None and metrics[-1].ttfb is not None
    assert ttfb is not None
    print(
        f"\nkokoro {tts.model}: TTFB {ttfb * 1000:.0f} ms, RTF {elapsed / audio.duration:.3f} "
        f"({audio.duration:.2f} s of audio in {elapsed:.2f} s)"
    )


def test_normalization_language_follows_the_voice(backend: FakeBackend) -> None:
    tts = KokoroTTS()
    assert tts.normalize_by_default and tts.text_language("af_heart") == "en-us"
    assert tts.text_language("ef_dora") == "es"
    assert tts.normalizer_for("bf_emma") is not None
    assert KokoroTTS(normalize=False).normalizer_for() is None
