"""sherpa-onnx STT, TTS and VAD providers.

Unit tests run offline against a fake ``sherpa_onnx`` module that mimics the real API
(chunked streaming decoding, endpointing, callback-driven synthesis, per-window VAD).
``@pytest.mark.model`` tests download the smallest real models (streaming Zipformer
Kroko 57 MB, Moonshine tiny 30 MB, a Piper voice 23 MB, Silero VAD 0.6 MB) and run with
``pytest -m model tests/providers/test_sherpa_onnx.py``.
"""

from __future__ import annotations

import asyncio
import importlib.machinery
import math
import re
import sys
import threading
import types
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from voice_agent_next import AudioFrame, create
from voice_agent_next.audio import read_wav
from voice_agent_next.engines import CascadeEngine
from voice_agent_next.errors import ConfigurationError, MissingDependencyError, ProviderError
from voice_agent_next.metrics import STTMetrics
from voice_agent_next.providers import sherpa_onnx as so_mod
from voice_agent_next.providers.mock import synth_speech
from voice_agent_next.providers.sherpa_onnx import (
    DEFAULT_STT_MODEL,
    DEFAULT_TTS_MODEL,
    DEFAULT_VAD_MODEL,
    SHERPA_MODELS,
    SherpaOnnxSTT,
    SherpaOnnxTTS,
    SherpaOnnxVAD,
    split_long_audio,
)
from voice_agent_next.registry import get_provider
from voice_agent_next.stt import StreamAdapter, STTEvent, STTEventType
from voice_agent_next.tts import SentenceStreamAdapter
from voice_agent_next.utils.download import DownloadError, download
from voice_agent_next.vad import VADEventType

SR = 16_000
WINDOW = 5_120  # fake streaming geometry: first chunk after 0.32 s ...
SHIFT = 2_560  # ... then one chunk every 0.16 s


# ------------------------------------------------------------------------------ fakes
def _voiced(samples: np.ndarray) -> bool:
    return bool(samples.size) and float(np.abs(samples).max()) > 0.01


class FakeOnlineStream:
    def __init__(self, backend: FakeBackend) -> None:
        self.backend = backend
        self.samples = np.zeros(0, dtype=np.float32)
        self.finished = False
        self.options: dict[str, str] = {}
        self.processed = 0
        self.segment_start = 0
        self.words: list[tuple[str, int]] = []  # (word, chunk start sample)
        self.last_voice_end = 0

    def accept_waveform(self, sample_rate: float, samples: Any) -> None:
        assert sample_rate == SR
        assert not self.finished, "accept_waveform() after input_finished()"
        self.backend.record("accept_waveform")
        self.samples = np.concatenate([self.samples, np.asarray(samples, dtype=np.float32)])

    def input_finished(self) -> None:
        self.backend.record("input_finished")
        self.finished = True

    def set_option(self, key: str, value: str) -> None:
        self.options[key] = value


@dataclass
class FakeOnlineResult:
    text: str
    tokens: list[str]
    timestamps: list[float]
    start_time: float
    ys_probs: list[float]


class FakeOnlineRecognizer:
    """Chunked decoding: every chunk containing speech emits the next word of the script."""

    def __init__(self, backend: FakeBackend, factory: str, kwargs: dict[str, Any]) -> None:
        self.backend = backend
        self.factory = factory
        self.kwargs = kwargs
        self.streams: list[FakeOnlineStream] = []

    def create_stream(self, hotwords: str | None = None) -> FakeOnlineStream:
        self.backend.record("create_stream")
        stream = FakeOnlineStream(self.backend)
        self.streams.append(stream)
        return stream

    def is_ready(self, s: FakeOnlineStream) -> bool:
        return len(s.samples) >= s.processed + WINDOW

    def decode_stream(self, s: FakeOnlineStream) -> None:
        self.backend.record("decode_stream")
        if self.backend.decode_error is not None:
            raise self.backend.decode_error
        chunk = s.samples[s.processed : s.processed + SHIFT]
        if _voiced(chunk):
            s.words.append((self.backend.next_word(), s.processed - s.segment_start))
            s.last_voice_end = s.processed + SHIFT
        s.processed += SHIFT

    def get_result_all(self, s: FakeOnlineStream) -> FakeOnlineResult:
        words = [w for w, _ in s.words]
        return FakeOnlineResult(
            text=" ".join(words).upper() if self.backend.upper else " ".join(words),
            tokens=[f" {w}" for w in words],
            timestamps=[start / SR for _, start in s.words],
            start_time=s.segment_start / SR,
            ys_probs=[-0.1] * len(words),
        )

    def is_endpoint(self, s: FakeOnlineStream) -> bool:
        if not self.kwargs.get("enable_endpoint_detection"):
            return False
        silence = (s.processed - max(s.last_voice_end, s.segment_start)) / SR
        rule = "rule2_min_trailing_silence" if s.words else "rule1_min_trailing_silence"
        return silence >= self.kwargs[rule]

    def reset(self, s: FakeOnlineStream) -> bool:
        self.backend.record("reset")
        s.segment_start = s.processed
        s.words = []
        return True


@dataclass
class FakeOfflineResult:
    text: str = ""
    tokens: list[str] = field(default_factory=list)
    timestamps: list[float] = field(default_factory=list)
    lang: str = ""
    ys_log_probs: list[float] = field(default_factory=list)


class FakeOfflineStream:
    def __init__(self) -> None:
        self.samples = np.zeros(0, dtype=np.float32)
        self.result = FakeOfflineResult()

    def accept_waveform(self, sample_rate: float, samples: Any) -> None:
        assert sample_rate == SR
        self.samples = np.concatenate([self.samples, np.asarray(samples, dtype=np.float32)])


class FakeOfflineRecognizer:
    def __init__(self, backend: FakeBackend, factory: str, kwargs: dict[str, Any]) -> None:
        self.backend = backend
        self.factory = factory
        self.kwargs = kwargs
        self.pieces: list[int] = []

    def create_stream(self, hotwords: str | None = None) -> FakeOfflineStream:
        return FakeOfflineStream()

    def decode_stream(self, s: FakeOfflineStream) -> None:
        self.backend.record("offline_decode")
        self.pieces.append(len(s.samples))
        n = len(self.pieces)
        s.result = FakeOfflineResult(
            text=f"{self.backend.offline_prefix}piece {n}",
            tokens=[" piece", f" {n}"],
            timestamps=[0.1, 0.3],
            lang=self.backend.offline_lang,
            ys_log_probs=[-0.2, -0.2],
        )


class FakeConfig:
    """Keyword-argument bag standing in for sherpa's pybind config classes."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)

    def validate(self) -> bool:
        return FakeBackend.current.config_valid


class FakeGenerationConfig:
    def __init__(self) -> None:
        self.sid = 0
        self.speed = 1.0
        self.silence_scale = 0.2


@dataclass
class FakeGeneratedAudio:
    samples: list[float]
    sample_rate: int


class FakeTts:
    """Synthesizes 20 ms of tone per character, one callback per sentence."""

    def __init__(self, backend: FakeBackend, config: FakeConfig) -> None:
        backend.record("tts_load")
        self.backend = backend
        self.config = config
        self.sample_rate = backend.tts_rate
        self.num_speakers = backend.num_speakers

    def generate(
        self,
        text: str,
        config: FakeGenerationConfig,
        callback: Callable[[np.ndarray, float], int] | None = None,
    ) -> FakeGeneratedAudio:
        backend = self.backend
        backend.generate_calls.append(
            {"text": text, "sid": config.sid, "speed": config.speed,
             "thread": threading.current_thread().name}
        )  # fmt: skip
        sentences = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]
        out: list[float] = []
        for i, sentence in enumerate(sentences):
            n = round(len(sentence) * 0.02 * self.sample_rate)
            t = np.arange(n) / self.sample_rate
            samples = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
            out.extend(samples.tolist())
            if callback is not None:
                ret = callback(samples, (i + 1) / len(sentences))
                backend.callback_returns.append(ret)
                if ret == 0:
                    break
            if i == 0 and backend.gate is not None:
                backend.gate.wait(5)
        return FakeGeneratedAudio(out, self.sample_rate)


class FakeVadModel:
    created = 0

    def __init__(self, config: FakeConfig) -> None:
        self.config = config
        self.resets = 0
        FakeVadModel.created += 1

    @classmethod
    def create(cls, config: FakeConfig) -> FakeVadModel:
        return cls(config)

    def is_speech(self, samples: Any) -> bool:
        return float(np.abs(np.asarray(samples)).max()) > 0.05

    def reset(self) -> None:
        self.resets += 1


class FakeBackend:
    current: FakeBackend

    def __init__(self) -> None:
        self.script = ["hello", "world", "how", "are", "you", "today"]
        self._word = 0
        self.upper = False
        self.calls: list[tuple[str, str]] = []
        self.recognizers: list[Any] = []
        self.decode_error: Exception | None = None
        self.offline_prefix = ""
        self.offline_lang = ""
        self.tts_rate = 22_050
        self.num_speakers = 10
        self.config_valid = True
        self.generate_calls: list[dict[str, Any]] = []
        self.callback_returns: list[int] = []
        self.gate: threading.Event | None = None
        self.tts_configs: list[FakeConfig] = []

    def next_word(self) -> str:
        word = self.script[self._word % len(self.script)]
        self._word += 1
        return word

    def record(self, name: str) -> None:
        self.calls.append((name, threading.current_thread().name))

    def module(self) -> types.ModuleType:
        backend = self
        mod = types.ModuleType("sherpa_onnx")
        mod.__spec__ = importlib.machinery.ModuleSpec("sherpa_onnx", None)

        def factory(cls: type, name: str) -> Callable[..., Any]:
            def create(**kwargs: Any) -> Any:
                for key, value in kwargs.items():  # the real factories assert file existence
                    if key in ("tokens", "encoder", "decoder", "joiner", "model") and value:
                        assert Path(value).is_file(), value
                backend.record(f"load:{name}")
                recognizer = cls(backend, name, kwargs)
                backend.recognizers.append(recognizer)
                return recognizer

            return create

        online = types.SimpleNamespace(
            **{n: factory(FakeOnlineRecognizer, n) for n in
               ("from_transducer", "from_paraformer", "from_zipformer2_ctc", "from_nemo_ctc")}
        )  # fmt: skip
        offline = types.SimpleNamespace(
            **{n: factory(FakeOfflineRecognizer, n) for n in
               ("from_transducer", "from_nemo_ctc", "from_moonshine", "from_moonshine_v2",
                "from_sense_voice", "from_whisper")}
        )  # fmt: skip
        mod.OnlineRecognizer = online  # type: ignore[attr-defined]
        mod.OfflineRecognizer = offline  # type: ignore[attr-defined]
        for name in (
            "OfflineTtsVitsModelConfig",
            "OfflineTtsKokoroModelConfig",
            "OfflineTtsMatchaModelConfig",
            "OfflineTtsModelConfig",
            "SileroVadModelConfig",
            "TenVadModelConfig",
            "VadModelConfig",
        ):
            setattr(mod, name, FakeConfig)

        def tts_config(**kwargs: Any) -> FakeConfig:
            config = FakeConfig(**kwargs)
            backend.tts_configs.append(config)
            return config

        mod.OfflineTtsConfig = tts_config  # type: ignore[attr-defined]
        mod.OfflineTts = lambda config: FakeTts(backend, config)  # type: ignore[attr-defined]
        mod.GenerationConfig = FakeGenerationConfig  # type: ignore[attr-defined]
        mod.VadModel = FakeVadModel  # type: ignore[attr-defined]
        return mod

    @property
    def online(self) -> FakeOnlineRecognizer:
        (rec,) = [r for r in self.recognizers if isinstance(r, FakeOnlineRecognizer)]
        return rec

    @property
    def offline(self) -> FakeOfflineRecognizer:
        (rec,) = [r for r in self.recognizers if isinstance(r, FakeOfflineRecognizer)]
        return rec


@pytest.fixture
def backend(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> FakeBackend:
    monkeypatch.setenv("VAN_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("VAN_OFFLINE", raising=False)
    fake = FakeBackend()
    FakeBackend.current = fake
    FakeVadModel.created = 0
    monkeypatch.setitem(sys.modules, "sherpa_onnx", fake.module())
    return fake


# ---------------------------------------------------------------------- model dirs
TRANSDUCER = ("encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt")


def model_dir(root: Path, name: str, files: tuple[str, ...], dirs: tuple[str, ...] = ()) -> Path:
    path = root / name
    path.mkdir(parents=True)
    for f in files:
        (path / f).write_bytes(b"x")
    for d in dirs:
        (path / d).mkdir()
    return path


@pytest.fixture
def downloads(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[tuple[str, str | None]]:
    """Catalog downloads served from fake extracted directories (no network)."""
    calls: list[tuple[str, str | None]] = []

    def fake_archive(url: str, *, sha256: str | None = None, subdir: str = "", **_: Any) -> Path:
        calls.append((url, sha256))
        spec = next(m for m in SHERPA_MODELS.values() if m.url == url)
        root = tmp_path / "extracted" / spec.stem
        if not root.exists():
            root.mkdir(parents=True)
            for rel in spec.files.values():
                for part in rel.split(","):
                    target = root / part
                    if part == "espeak-ng-data":
                        target.mkdir()
                    else:
                        target.write_bytes(b"x")
        return root

    def fake_download(url: str, *, sha256: str | None = None, **_: Any) -> Path:
        calls.append((url, sha256))
        path = tmp_path / "files" / url.rsplit("/", 1)[-1]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
        return path

    monkeypatch.setattr(so_mod, "download_archive", fake_archive)
    monkeypatch.setattr(so_mod, "download", fake_download)
    return calls


def chunks(frame: AudioFrame, step: float = 0.02) -> list[AudioFrame]:
    n = math.ceil(frame.duration / step - 1e-9)
    return [frame.slice(i * step, (i + 1) * step) for i in range(n)]


def speech(duration: float) -> AudioFrame:
    return synth_speech(duration, SR)


def silence(duration: float) -> AudioFrame:
    return AudioFrame.silence(duration, SR)


async def collect(stream: Any) -> list[STTEvent]:
    return [ev async for ev in stream]


def kinds(events: list[STTEvent]) -> list[str]:
    return [ev.type.value for ev in events]


# ------------------------------------------------------------------------ catalog
def test_catalog_is_consistent() -> None:
    assert {DEFAULT_STT_MODEL, DEFAULT_TTS_MODEL, DEFAULT_VAD_MODEL} <= set(SHERPA_MODELS)
    stems = [m.stem for m in SHERPA_MODELS.values()]
    assert len(stems) == len(set(stems))
    for name, spec in SHERPA_MODELS.items():
        assert spec.name == name
        assert re.fullmatch(r"[0-9a-f]{64}", spec.sha256), name
        assert spec.size > 0 and spec.license and spec.languages
        assert spec.url.startswith("https://github.com/k2-fsa/sherpa-onnx/releases/download/")
        roles = set(spec.files) | {role for role, *_ in spec.extra_assets}
        assert set(so_mod._REQUIRED_FILES[spec.kind]) <= roles, name
        for _, _, _, digest in spec.extra_assets:
            assert re.fullmatch(r"[0-9a-f]{64}", digest)
        if spec.task == "tts":
            assert spec.sample_rate and spec.default_voice is not None
    assert SHERPA_MODELS["kokoro-multi-lang-v1_0"].speakers[3] == "af_heart"
    assert len(SHERPA_MODELS["kokoro-multi-lang-v1_0"].speakers) == 54


def test_registered_for_three_component_kinds() -> None:
    for kind, cls, default in (
        ("stt", SherpaOnnxSTT, DEFAULT_STT_MODEL),
        ("tts", SherpaOnnxTTS, DEFAULT_TTS_MODEL),
        ("vad", SherpaOnnxVAD, DEFAULT_VAD_MODEL),
    ):
        spec = get_provider(kind, "sherpa-onnx")  # type: ignore[arg-type]
        assert spec.factory is cls and spec.name == "sherpa_onnx"
        assert spec.default_model == default and default in spec.models
        assert spec.extra == "sherpa-onnx" and spec.requires == ("sherpa_onnx",)
        assert spec.local and not spec.env
        assert get_provider(kind, "sherpa") is spec  # type: ignore[arg-type]
    assert "moonshine-tiny-en" in get_provider("stt", "sherpa-onnx").models
    assert "piper-en_US-ljspeech-medium" in get_provider("tts", "sherpa-onnx").models


def test_missing_dependency_is_reported_with_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "sherpa_onnx", None)
    with pytest.raises(MissingDependencyError, match=r"voice-agent-next\[sherpa-onnx\]"):
        SherpaOnnxSTT()
    with pytest.raises(MissingDependencyError):
        SherpaOnnxTTS()
    with pytest.raises(MissingDependencyError):
        SherpaOnnxVAD()


def test_broken_native_install_names_sherpa_onnx_core(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(name: str) -> Any:
        raise ImportError("libonnxruntime.so: cannot open shared object file")

    monkeypatch.setattr(so_mod, "is_installed", lambda name: True)
    monkeypatch.setattr(so_mod.importlib, "import_module", broken)
    with pytest.raises(MissingDependencyError, match="sherpa-onnx-core"):
        so_mod._import_sherpa()


# ----------------------------------------------------------------- model resolution
async def test_catalog_models_download_pinned_archives(
    backend: FakeBackend, downloads: list[tuple[str, str | None]]
) -> None:
    stt = create("stt", "sherpa-onnx")
    assert isinstance(stt, SherpaOnnxSTT) and stt.model == DEFAULT_STT_MODEL
    assert downloads == []  # nothing happens before warmup / first use
    await stt.warmup()
    spec = SHERPA_MODELS[DEFAULT_STT_MODEL]
    assert downloads == [(spec.url, spec.sha256)]
    rec = backend.online
    assert rec.factory == "from_transducer"
    assert Path(rec.kwargs["encoder"]).name == "encoder.int8.onnx"
    assert (rec.kwargs["num_threads"], rec.kwargs["provider"]) == (2, "cpu")
    assert rec.kwargs["enable_endpoint_detection"] is False
    assert rec.kwargs["decoding_method"] == "greedy_search"
    await stt.warmup()  # loaded once
    assert len(backend.recognizers) == 1
    assert (stt._window, stt._shift) == (WINDOW, SHIFT)  # measured on a throwaway stream
    await stt.aclose()


async def test_catalog_options_and_archive_names(
    backend: FakeBackend, downloads: list[tuple[str, str | None]]
) -> None:
    stt = SherpaOnnxSTT(model="sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06.tar.bz2")
    assert stt.model == "zipformer-en-kroko"
    await stt.warmup()
    assert backend.online.kwargs["model_type"] == "zipformer2"
    assert Path(backend.online.kwargs["encoder"]).name == "encoder.onnx"
    await stt.aclose()


def test_wrong_task_unknown_model_and_kind(backend: FakeBackend) -> None:
    with pytest.raises(ConfigurationError, match="is a tts model"):
        SherpaOnnxSTT(model=DEFAULT_TTS_MODEL)
    with pytest.raises(ConfigurationError, match="not in the catalog"):
        SherpaOnnxSTT(model="no-such-model")
    with pytest.raises(ConfigurationError, match="unknown sherpa-onnx stt kind"):
        SherpaOnnxSTT(kind="tts-vits")
    with pytest.raises(ConfigurationError, match="pass kind="):
        SherpaOnnxSTT(model="https://example.com/model.tar.bz2")


@pytest.mark.parametrize(
    ("name", "files", "dirs", "expected"),
    [
        ("sherpa-onnx-streaming-zipformer-en-2023-06-26", TRANSDUCER, (), "online-transducer"),
        ("sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-fp16", TRANSDUCER, (), "offline-nemo-transducer"),
        ("sherpa-onnx-zipformer-en-2023-04-01", TRANSDUCER, (), "offline-transducer"),
        (
            "my-moonshine",
            ("encoder_model.ort", "decoder_model_merged.ort", "tokens.txt"),
            (),
            "offline-moonshine-v2",
        ),
        (
            "sherpa-onnx-moonshine-tiny-en-int8",
            ("preprocess.onnx", "encode.int8.onnx", "uncached_decode.int8.onnx",
             "cached_decode.int8.onnx", "tokens.txt"),
            (),
            "offline-moonshine",
        ),  # fmt: skip
        ("sherpa-onnx-sense-voice-2024", ("model.onnx", "tokens.txt"), (), "offline-sense-voice"),
        (
            "sherpa-onnx-whisper-tiny.en",
            ("tiny.en-encoder.onnx", "tiny.en-decoder.onnx", "tiny.en-tokens.txt"),
            (),
            "offline-whisper",
        ),
        (
            "sherpa-onnx-nemo-streaming-fast-conformer-ctc-en-80ms",
            ("model.onnx", "tokens.txt"),
            (),
            "online-nemo-ctc",
        ),
    ],
)
def test_local_stt_directories_are_detected(
    backend: FakeBackend,
    tmp_path: Path,
    name: str,
    files: tuple[str, ...],
    dirs: tuple[str, ...],
    expected: str,
) -> None:
    path = model_dir(tmp_path, name, files, dirs)
    stt = SherpaOnnxSTT(model=str(path))
    assert stt.kind == expected and stt.model == str(path)
    assert stt.capabilities.streaming == expected.startswith("online")


async def test_local_directory_files_are_found_int8_first(
    backend: FakeBackend, tmp_path: Path
) -> None:
    path = model_dir(
        tmp_path,
        "sherpa-onnx-streaming-zipformer-en-2023-06-26",
        (*TRANSDUCER, "encoder.onnx", "joiner.onnx"),
    )
    stt = SherpaOnnxSTT(model=str(path), num_threads=3)
    await stt.warmup()
    kwargs = backend.online.kwargs
    assert Path(kwargs["encoder"]).name == "encoder.int8.onnx"
    assert Path(kwargs["joiner"]).name == "joiner.int8.onnx"
    assert kwargs["num_threads"] == 3 and "model_type" not in kwargs
    await stt.aclose()


async def test_extracted_catalog_directory_needs_no_download(
    backend: FakeBackend, tmp_path: Path, downloads: list[tuple[str, str | None]]
) -> None:
    spec = SHERPA_MODELS["nemotron-3.5-160ms"]
    path = model_dir(tmp_path, spec.stem, TRANSDUCER)
    stt = SherpaOnnxSTT(model=str(path), language="de")
    assert stt.model == "nemotron-3.5-160ms" and stt.capabilities.language_detection
    stream = stt.stream()
    stream.push_audio(speech(0.5))
    stream.end_input()
    await collect(stream)
    assert downloads == []
    assert all(s.options == {"language": "de"} for s in backend.online.streams[1:])
    await stt.aclose()


async def test_missing_files_and_explicit_overrides(backend: FakeBackend, tmp_path: Path) -> None:
    path = model_dir(tmp_path, "sherpa-onnx-streaming-x", ("encoder.onnx", "tokens.txt"))
    stt = SherpaOnnxSTT(model=str(path))
    with pytest.raises(ConfigurationError, match="missing: decoder, joiner"):
        await stt.warmup()
    extra = model_dir(tmp_path, "extra", ("dec.onnx", "join.onnx"))
    stt = SherpaOnnxSTT(
        model=str(path), files={"decoder": extra / "dec.onnx", "joiner": extra / "join.onnx"}
    )
    await stt.warmup()
    assert Path(backend.online.kwargs["decoder"]).name == "dec.onnx"
    await stt.aclose()


def test_unknown_local_directory_asks_for_kind(backend: FakeBackend, tmp_path: Path) -> None:
    path = model_dir(tmp_path, "mystery", ("model.onnx", "tokens.txt"))
    with pytest.raises(ConfigurationError, match="pass kind="):
        SherpaOnnxSTT(model=str(path))
    assert SherpaOnnxSTT(model=str(path), kind="offline-nemo-ctc").kind == "offline-nemo-ctc"


def test_invalid_options(backend: FakeBackend) -> None:
    with pytest.raises(ConfigurationError, match="decoding_method"):
        SherpaOnnxSTT(decoding_method="beam")  # sherpa would exit the process
    with pytest.raises(ConfigurationError, match="num_threads"):
        SherpaOnnxSTT(num_threads=0)
    with pytest.raises(ConfigurationError, match="tail_padding"):
        SherpaOnnxSTT(tail_padding=-1)
    with pytest.raises(ConfigurationError, match="text_case"):
        SherpaOnnxSTT(text_case="upper")  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="needs a transducer"):
        SherpaOnnxSTT(model="moonshine-tiny-en", decoding_method="modified_beam_search")


# ------------------------------------------------------------------ streaming STT
async def test_streaming_emits_interim_and_final_transcripts(
    backend: FakeBackend, downloads: list[tuple[str, str | None]]
) -> None:
    stt = SherpaOnnxSTT()
    assert stt.capabilities.streaming and stt.capabilities.interim_results
    assert stt.capabilities.word_timestamps and not stt.capabilities.end_of_turn
    metrics: list[STTMetrics] = []
    stt.on("metrics", metrics.append)
    stream = stt.stream()
    for frame in chunks(speech(0.96)) + chunks(silence(0.3)):
        stream.push_audio(frame)
    stream.flush()
    stream.end_input()
    events = await collect(stream)
    assert kinds(events)[0] == "start_of_speech"
    interims = [e.text for e in events if e.type == STTEventType.INTERIM_TRANSCRIPT]
    assert interims and interims[-1].startswith("hello")
    finals = [e for e in events if e.type == STTEventType.FINAL_TRANSCRIPT]
    assert [e.text for e in finals] == ["hello world how are you today", ""]
    assert kinds(events)[-3:] == ["final_transcript", "end_of_speech", "final_transcript"]
    segment = {e.segment_id for e in events[: -1]}
    assert len(segment) == 1  # START ... END share the utterance's segment id
    final = finals[0].transcript
    assert final is not None and final.words is not None
    assert [w.word for w in final.words] == final.text.split()
    assert [w.start for w in final.words] == pytest.approx([0.0, 0.16, 0.32, 0.48, 0.64, 0.8])
    assert final.start_time == 0.0 and final.end_time == pytest.approx(1.05)
    assert final.confidence == pytest.approx(math.exp(-0.1))
    streamed = [m for m in metrics if m.streamed]
    assert len(streamed) == 2 and all(m.latency is not None for m in streamed)
    await stt.aclose()


async def test_flush_pads_just_enough_to_decode_the_tail(
    backend: FakeBackend, downloads: list[tuple[str, str | None]]
) -> None:
    stt = SherpaOnnxSTT()
    stream = stt.stream()
    stream.push_audio(speech(0.5))  # 8000 samples: two chunks are decoded, 2880 are pending
    stream.flush()
    stream.end_input()
    events = await collect(stream)
    finals = [e.text for e in events if e.type == STTEventType.FINAL_TRANSCRIPT]
    assert finals == ["hello world how are", ""]  # both tail chunks made it in
    first = backend.online.streams[1]  # [0] measured the chunk geometry
    assert first.finished
    # window + ceil(8000 / shift) * shift + 2 frames: one chunk beyond the last sample
    assert len(first.samples) == WINDOW + 4 * SHIFT + 320
    assert first.processed == 5 * SHIFT
    await stt.aclose()


async def test_every_flush_starts_a_new_utterance_on_the_input_clock(
    backend: FakeBackend, downloads: list[tuple[str, str | None]]
) -> None:
    stt = SherpaOnnxSTT()
    stream = stt.stream()
    for frame in chunks(speech(0.48)) + chunks(silence(0.32)):
        stream.push_audio(frame)
    stream.flush()
    stream.flush()  # nothing new: an empty final, no speech events
    for frame in chunks(silence(0.32)) + chunks(speech(0.32)):
        stream.push_audio(frame)
    stream.end_input()
    events = await collect(stream)
    finals = [e for e in events if e.type == STTEventType.FINAL_TRANSCRIPT]
    assert [f.text for f in finals] == ["hello world how", "", "are you"]
    second = finals[2].transcript
    assert second is not None and second.words is not None
    # 0.8 s before the second utterance + 0.32 s of silence inside it
    assert [w.start for w in second.words] == pytest.approx([1.12, 1.28])
    assert kinds(events).count("start_of_speech") == 2
    assert kinds(events).count("end_of_speech") == 2
    assert len(backend.online.streams) == 1 + 2  # geometry + one per utterance with audio
    await stt.aclose()


async def test_sherpa_endpoints_emit_final_and_end_of_speech(
    backend: FakeBackend, downloads: list[tuple[str, str | None]]
) -> None:
    stt = SherpaOnnxSTT(endpoint_detection=True, rule2_min_trailing_silence=0.3)
    stream = stt.stream()
    for frame in chunks(speech(0.32)) + chunks(silence(0.8)) + chunks(speech(0.32)):
        stream.push_audio(frame)
    stream.end_input()
    events = await collect(stream)
    assert [e.text for e in events if e.type == STTEventType.FINAL_TRANSCRIPT] == [
        "hello world",
        "how are",
    ]
    assert kinds(events).count("end_of_speech") == 2
    assert ("reset", "sherpa-onnx-stt_0") in backend.calls
    second = [e for e in events if e.type == STTEventType.FINAL_TRANSCRIPT][1].transcript
    assert second is not None and second.words is not None
    assert second.words[0].start == pytest.approx(1.12)  # segment start + token time
    assert backend.online.kwargs["rule2_min_trailing_silence"] == 0.3
    await stt.aclose()


async def test_all_sherpa_calls_run_on_one_worker_thread(
    backend: FakeBackend, downloads: list[tuple[str, str | None]]
) -> None:
    stt = SherpaOnnxSTT()
    stream = stt.stream()
    for frame in chunks(speech(0.5)):
        stream.push_audio(frame)
    stream.end_input()
    await collect(stream)
    await stt.transcribe(speech(0.3))
    threads = {thread for _, thread in backend.calls}
    assert threads == {"sherpa-onnx-stt_0"}
    await stt.aclose()


async def test_transcribe_with_a_streaming_model(
    backend: FakeBackend, downloads: list[tuple[str, str | None]]
) -> None:
    backend.upper = True
    stt = SherpaOnnxSTT(word_timestamps=False)
    assert not stt.capabilities.word_timestamps
    result = await stt.transcribe(synth_speech(0.5, 48_000).to_channels(2))
    assert result.text == "hello world how are"  # all-caps output is lowercased
    assert result.words is None and result.start_time == 0.0
    assert (await stt.transcribe(AudioFrame.empty(SR))).text == ""
    keep = SherpaOnnxSTT(text_case="keep")
    assert (await keep.transcribe(speech(0.16))).text == "YOU"
    await stt.aclose()
    await keep.aclose()


async def test_decoding_errors_surface_as_provider_errors(
    backend: FakeBackend, downloads: list[tuple[str, str | None]]
) -> None:
    stt = SherpaOnnxSTT()
    await stt.warmup()
    backend.decode_error = RuntimeError("onnxruntime exploded")
    stream = stt.stream()
    stream.push_audio(speech(0.5))
    stream.end_input()
    with pytest.raises(ProviderError, match="onnxruntime exploded"):
        await collect(stream)
    with pytest.raises(ProviderError, match="onnxruntime exploded"):
        await stt.transcribe(speech(0.5))
    await stt.aclose()


async def test_cascade_uses_the_streaming_recognizer_directly(
    backend: FakeBackend, downloads: list[tuple[str, str | None]]
) -> None:
    engine = CascadeEngine(stt="sherpa-onnx", vad="energy", llm="mock", tts="mock")
    assert isinstance(engine.stt, SherpaOnnxSTT)
    offline = CascadeEngine(stt="sherpa-onnx/moonshine-tiny-en", vad="energy", llm="mock", tts="mock")
    assert isinstance(offline.stt, StreamAdapter)
    assert isinstance(offline.stt.wrapped, SherpaOnnxSTT)


# -------------------------------------------------------------------- offline STT
async def test_offline_model_splits_long_audio_at_pauses(
    backend: FakeBackend, downloads: list[tuple[str, str | None]]
) -> None:
    stt = SherpaOnnxSTT(model="moonshine-tiny-en")
    assert not stt.capabilities.streaming and stt.max_segment_duration == 8.0
    audio = AudioFrame.concat([speech(6.0), silence(0.3), speech(6.2), silence(0.3), speech(4.0)])
    result = await stt.transcribe(audio)
    rec = backend.offline
    assert rec.factory == "from_moonshine_v2"
    assert set(rec.kwargs) >= {"encoder", "decoder", "tokens", "num_threads", "provider"}
    assert result.text == "piece 1 piece 2 piece 3"
    assert sum(rec.pieces) == audio.samples_per_channel
    assert all(n <= 8 * SR for n in rec.pieces)
    cut = rec.pieces[0] / SR
    assert 6.0 <= cut <= 6.3  # inside the first pause
    assert result.words is None  # moonshine gives no timestamps
    await stt.aclose()


async def test_sense_voice_language_and_tags(
    backend: FakeBackend, downloads: list[tuple[str, str | None]]
) -> None:
    backend.offline_prefix = "<|en|><|NEUTRAL|>"
    backend.offline_lang = "<|en|>"
    stt = SherpaOnnxSTT(model="sense-voice", language="en-US")
    assert stt.capabilities.language_detection and stt.capabilities.word_timestamps
    result = await stt.transcribe(speech(1.0))
    assert backend.offline.kwargs["language"] == "en"
    assert backend.offline.kwargs["use_itn"] is True
    assert result.text == "<|NEUTRAL|>piece 1" or result.text == "piece 1"
    assert result.language == "en"
    assert result.words is not None and [w.word for w in result.words] == ["piece", "1"]
    auto = SherpaOnnxSTT(model="sense-voice")
    await auto.warmup()
    assert backend.recognizers[-1].kwargs["language"] == "auto"
    await stt.aclose()
    await auto.aclose()


async def test_offline_nemo_transducer_sets_the_model_type(
    backend: FakeBackend, downloads: list[tuple[str, str | None]]
) -> None:
    stt = SherpaOnnxSTT(model="parakeet-tdt-0.6b-v3")
    await stt.warmup()
    assert backend.offline.factory == "from_transducer"
    assert backend.offline.kwargs["model_type"] == "nemo_transducer"
    await stt.aclose()


def test_split_long_audio() -> None:
    x = np.ones(10 * SR, dtype=np.float32)
    x[7 * SR : 7 * SR + 800] = 0.0  # a 50 ms pause
    pieces = split_long_audio(x, 8 * SR, SR)
    assert [len(p) for p in pieces][0] == pytest.approx(7 * SR + 400, abs=400)
    assert sum(len(p) for p in pieces) == len(x)
    assert split_long_audio(x, 0, SR)[0] is x and len(split_long_audio(x, 20 * SR, SR)) == 1


def test_words_are_grouped_from_tokens() -> None:
    marker = chr(0x2581)  # SentencePiece's word marker
    words = so_mod._words(
        [" hel", "lo", f"{marker}wor", "ld", " ", "!"], [0.1, 0.2, 0.5, 0.6, 0.7, 0.8], 1.0,
        end=2.5,
    )  # fmt: skip
    assert [(w.word, w.start, w.end) for w in words] == [
        ("hello", 1.1, 1.5),
        ("world!", 1.5, pytest.approx(1.75)),
    ]
    cjk = so_mod._words([chr(0x4F60), chr(0x597D)], [0.0, 0.2], 0.0, end=None)
    assert [w.word for w in cjk] == [chr(0x4F60), chr(0x597D)]


# ----------------------------------------------------------------------------- TTS
async def test_tts_streams_audio_while_generating(
    backend: FakeBackend, downloads: list[tuple[str, str | None]]
) -> None:
    tts = create("tts", "sherpa-onnx")
    assert isinstance(tts, SherpaOnnxTTS)
    assert (tts.model, tts.sample_rate, tts.voice) == (DEFAULT_TTS_MODEL, 22_050, "0")
    backend.gate = threading.Event()
    stream = tts.synthesize("Hello there. How are you today?")
    first: AudioFrame | None = None
    frames = []
    async for chunk in stream:
        if chunk.frame and first is None:
            first = chunk.frame
            assert not backend.gate.is_set()  # the second sentence is still pending
            backend.gate.set()
        if chunk.frame:
            frames.append(chunk.frame)
    assert first is not None and first.sample_rate == 22_050
    assert all(f.duration <= 0.05 + 1e-9 for f in frames)
    total = sum(f.duration for f in frames)
    assert total == pytest.approx(len("Hello there.How are you today?") * 0.02, abs=0.01)
    (call,) = backend.generate_calls
    assert call["sid"] == 0 and call["thread"].startswith("sherpa-onnx-tts")
    config = backend.tts_configs[-1]
    assert config.max_num_sentences == 1 and config.silence_scale == 0.2
    vits = config.model.vits
    assert Path(vits.model).name == "en_US-libritts_r-medium.onnx"
    assert Path(vits.data_dir).name == "espeak-ng-data" and vits.lexicon == ""
    await tts.aclose()


async def test_tts_resamples_to_the_declared_rate(
    backend: FakeBackend, downloads: list[tuple[str, str | None]]
) -> None:
    backend.tts_rate = 16_000
    tts = SherpaOnnxTTS(sample_rate=24_000)
    audio = await tts.synthesize("One two three.").collect()
    assert audio.sample_rate == 24_000
    assert audio.duration == pytest.approx(len("One two three.") * 0.02, abs=0.02)
    assert tts.model_sample_rate == 16_000
    await tts.aclose()


async def test_kokoro_voices_by_name(
    backend: FakeBackend, downloads: list[tuple[str, str | None]]
) -> None:
    backend.num_speakers = 54
    tts = SherpaOnnxTTS(model="kokoro-multi-lang-v1_0-int8", speed=1.2, lang="es")
    assert (tts.voice, tts.sample_rate) == ("af_heart", 24_000)
    await tts.synthesize("Hola.").collect()
    await tts.synthesize("Hi.", voice="bm_george").collect()
    await tts.synthesize("Hi.", voice="7").collect()
    assert [c["sid"] for c in backend.generate_calls] == [3, 26, 7]
    assert backend.generate_calls[0]["speed"] == 1.2
    kokoro = backend.tts_configs[-1].model.kokoro
    assert kokoro.lang == "es" and Path(kokoro.model).name == "model.int8.onnx"
    lexicons = [Path(p).name for p in kokoro.lexicon.split(",")]
    assert lexicons == ["lexicon-us-en.txt", "lexicon-zh.txt"]
    fsts = [Path(p).name for p in backend.tts_configs[-1].rule_fsts.split(",")]
    assert fsts == ["phone-zh.fst", "date-zh.fst", "number-zh.fst"]
    with pytest.raises(ConfigurationError, match="unknown sherpa-onnx voice"):
        SherpaOnnxTTS(model="kokoro-multi-lang-v1_0", voice="nobody")
    await tts.aclose()


async def test_speaker_id_out_of_range_and_invalid_config(
    backend: FakeBackend, downloads: list[tuple[str, str | None]]
) -> None:
    backend.num_speakers = 2
    tts = SherpaOnnxTTS(voice="5")
    with pytest.raises(ConfigurationError, match="has 2 speaker"):
        await tts.synthesize("Hi there.").collect()
    backend.config_valid = False
    broken = SherpaOnnxTTS()
    with pytest.raises(ConfigurationError, match="rejected the TTS configuration"):
        await broken.warmup()
    with pytest.raises(ConfigurationError, match="speed"):
        SherpaOnnxTTS(speed=10)
    with pytest.raises(ConfigurationError, match="named voice|unknown sherpa-onnx voice"):
        SherpaOnnxTTS(voice="alice")  # piper voices are numbered
    await tts.aclose()


async def test_matcha_downloads_its_vocoder(
    backend: FakeBackend, downloads: list[tuple[str, str | None]]
) -> None:
    tts = SherpaOnnxTTS(model="matcha-en_US-ljspeech")
    await tts.warmup()
    urls = [u for u, _ in downloads]
    assert any(u.endswith("/vocoder-models/vocos-22khz-univ.onnx") for u in urls)
    matcha = backend.tts_configs[-1].model.matcha
    assert Path(matcha.vocoder).name == "vocos-22khz-univ.onnx"
    assert Path(matcha.acoustic_model).name == "model-steps-3.onnx"
    await tts.aclose()


async def test_local_piper_directory_reads_its_sample_rate(
    backend: FakeBackend, tmp_path: Path
) -> None:
    path = model_dir(tmp_path, "vits-piper-de_DE-thorsten-low", ("de_DE-thorsten-low.onnx", "tokens.txt"), ("espeak-ng-data",))  # fmt: skip
    (path / "de_DE-thorsten-low.onnx.json").write_text('{"audio": {"sample_rate": 16000}}')
    tts = SherpaOnnxTTS(model=str(path))
    assert tts.kind == "tts-vits" and tts.sample_rate == 16_000
    await tts.warmup()
    assert Path(backend.tts_configs[-1].model.vits.model).name == "de_DE-thorsten-low.onnx"
    await tts.aclose()


async def test_closing_a_synthesis_stops_generation(
    backend: FakeBackend, downloads: list[tuple[str, str | None]]
) -> None:
    tts = SherpaOnnxTTS()
    await tts.warmup()
    backend.callback_returns.clear()
    backend.gate = threading.Event()
    stream = tts.synthesize("First sentence. Second sentence. Third one.")
    async for chunk in stream:
        if chunk.frame:
            break
    await stream.aclose()
    backend.gate.set()
    for _ in range(200):  # the worker finishes the current sentence, then stops
        if len(backend.callback_returns) >= 2:
            break
        await asyncio.sleep(0.01)
    assert backend.callback_returns == [1, 0]
    await tts.aclose()


async def test_unpronounceable_text_is_skipped_and_stream_uses_sentences(
    backend: FakeBackend, downloads: list[tuple[str, str | None]]
) -> None:
    tts = SherpaOnnxTTS()
    assert not (await tts.synthesize("...").collect())
    assert backend.generate_calls == []
    stream = tts.stream()
    assert isinstance(stream, SentenceStreamAdapter)
    stream.push_text("Hello there, friend. ")
    stream.push_text("Bye now.")
    stream.end_input()
    audio = [a.frame async for a in stream if a.frame]
    assert audio and len(backend.generate_calls) == 2
    await tts.aclose()


# ----------------------------------------------------------------------------- VAD
async def test_vad_windows_and_events(
    backend: FakeBackend, downloads: list[tuple[str, str | None]]
) -> None:
    vad = create("vad", "sherpa-onnx", min_silence_duration=0.2, activation_threshold=0.6)
    assert isinstance(vad, SherpaOnnxVAD)
    assert (vad.sample_rate, vad.window_samples, vad.model) == (SR, 512, "silero")
    await vad.warmup()
    stream = vad.stream()
    events = []
    for frame in chunks(silence(0.3)) + chunks(speech(0.6)) + chunks(silence(0.5)):
        events.extend(stream.push_audio(frame))
    assert [e.type for e in events] == [VADEventType.START_OF_SPEECH, VADEventType.END_OF_SPEECH]
    assert FakeVadModel.created == 2  # warm-up + one per stream
    config = stream._infer._model.config  # type: ignore[attr-defined]
    assert config.silero_vad.threshold == 0.6
    assert config.silero_vad.window_size == 512 and config.sample_rate == SR
    stream.reset()
    assert stream._infer._model.resets == 1  # type: ignore[attr-defined]
    ten = SherpaOnnxVAD(model="ten-vad")
    assert ten.window_samples == 256
    ten.stream()
    assert FakeVadModel.created == 3
    with pytest.raises(ConfigurationError, match="unknown VAD option"):
        SherpaOnnxVAD(bogus=1.0)
    assert [u for u, _ in downloads if u.endswith(".onnx")] == [
        SHERPA_MODELS["silero"].url,
        SHERPA_MODELS["ten-vad"].url,
    ]


# ------------------------------------------------------------ real models (opt-in)
JFK_URL = (
    "https://raw.githubusercontent.com/ggml-org/whisper.cpp/"
    "b0a11594aec50892a02cd8d129eee2dfe93a8bb8/samples/jfk.wav"
)
JFK_SHA256 = "59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e"


@pytest.fixture(scope="module")
def jfk() -> AudioFrame:
    """JFK, 1961 (public domain): "And so my fellow Americans, ask not what your country can
    do for you, ask what you can do for your country." 11 s, 16 kHz mono."""
    pytest.importorskip("sherpa_onnx")
    try:
        return read_wav(download(JFK_URL, subdir="testdata", sha256=JFK_SHA256))
    except DownloadError as exc:
        pytest.skip(f"test clip unavailable: {exc}")


def normalize(text: str) -> str:
    return re.sub(r"[^a-z ]+", "", text.lower())


async def loaded(component: Any) -> Any:
    try:
        await component.warmup()
    except DownloadError as exc:
        pytest.skip(f"model unavailable: {exc}")
    return component


@pytest.mark.model
async def test_real_streaming_model_finalizes_fast(jfk: AudioFrame) -> None:
    stt = await loaded(SherpaOnnxSTT(model="zipformer-en-kroko"))
    result = await stt.transcribe(jfk)
    assert "ask not what your country can do for you" in normalize(result.text)
    assert result.words and 0.0 <= result.words[0].start < 1.5
    metrics: list[STTMetrics] = []
    stt.on("metrics", metrics.append)
    stream = stt.stream()
    events: list[STTEvent] = []

    async def consume() -> None:
        async for ev in stream:
            events.append(ev)

    consumer = asyncio.create_task(consume())
    for i, frame in enumerate(chunks(jfk)):
        stream.push_audio(frame)
        if i == round(4.7 / 0.02):  # in the pause after "ask not"
            stream.flush()
        await asyncio.sleep(0)
    stream.end_input()
    await consumer
    finals = [e.text for e in events if e.type == STTEventType.FINAL_TRANSCRIPT]
    assert len(finals) == 2 and "fellow americans" in normalize(finals[0])
    assert "your country" in normalize(finals[1])
    assert any(e.type == STTEventType.INTERIM_TRANSCRIPT for e in events)
    latencies = [m.latency for m in metrics if m.latency is not None]
    print(f"flush -> final: {[round(x * 1000) for x in latencies]} ms")
    assert latencies and max(latencies) < 2.0  # generous: CI CPUs are slow
    await stt.aclose()


@pytest.mark.model
async def test_real_offline_model_and_vad_segmented_streaming(jfk: AudioFrame) -> None:
    stt = await loaded(SherpaOnnxSTT(model="moonshine-tiny-en"))
    result = await stt.transcribe(jfk)  # 11 s: split at a pause (Moonshine stops at ~9 s)
    assert "ask not what your country can do for you" in normalize(result.text)
    vad = await loaded(SherpaOnnxVAD())
    adapter = StreamAdapter(stt, vad)
    stream = adapter.stream()
    for frame in chunks(jfk) + chunks(AudioFrame.silence(1.0, SR)):
        stream.push_audio(frame)
    stream.end_input()
    finals = [e.text async for e in stream if e.type == STTEventType.FINAL_TRANSCRIPT]
    assert len(finals) >= 3 and "your country" in normalize(" ".join(finals))
    await adapter.aclose()


@pytest.mark.model
async def test_real_tts_synthesizes_intelligible_speech(jfk: AudioFrame) -> None:
    tts = await loaded(SherpaOnnxTTS(model="piper-en_US-libritts_r-medium"))
    assert tts.num_speakers and tts.num_speakers > 100
    audio = await tts.synthesize("The quick brown fox jumps over the lazy dog.").collect()
    assert audio.sample_rate == 22_050 and 1.5 < audio.duration < 6.0
    assert float(np.sqrt(np.mean(audio.to_float32() ** 2))) > 0.01
    stt = await loaded(SherpaOnnxSTT(model="zipformer-en-kroko"))
    heard = normalize((await stt.transcribe(audio)).text)
    assert "quick brown fox" in heard and "lazy dog" in heard
    await stt.aclose()
    await tts.aclose()


@pytest.mark.model
async def test_real_vad_finds_the_phrases(jfk: AudioFrame) -> None:
    vad = await loaded(SherpaOnnxVAD())
    stream = vad.stream()
    events = [ev for frame in chunks(jfk) for ev in stream.push_audio(frame)]
    starts = [e for e in events if e.type == VADEventType.START_OF_SPEECH]
    assert 3 <= len(starts) <= 5
