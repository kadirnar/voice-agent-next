"""Fake ``mlx``, ``parakeet_mlx``, ``mlx_whisper``, ``mlx_audio`` and ``huggingface_hub``
modules for the MLX provider tests (they run on every OS; the real packages need macOS on
Apple silicon). The fakes mirror the real call signatures and return types."""

from __future__ import annotations

import sys
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

from voice_agent_next.providers import _mlx

WORDS = ("Hello", "world,", "this", "is", "a", "streaming", "test.")


class FakeArray:
    """Stands in for ``mlx.core.array`` (``np.asarray`` works on it, like on the real one)."""

    def __init__(self, data: Any, dtype: str = "float32") -> None:
        self.data = np.asarray(data, dtype=np.float32).reshape(-1)
        self.dtype = dtype

    def astype(self, dtype: str) -> FakeArray:
        return FakeArray(self.data, dtype)

    def __len__(self) -> int:
        return len(self.data)

    def __array__(self, dtype: Any = None, copy: Any = None) -> np.ndarray[Any, Any]:
        return self.data if dtype is None else self.data.astype(dtype)


@dataclass
class Token:
    """``parakeet_mlx.AlignedToken``."""

    id: int
    text: str
    start: float
    duration: float
    confidence: float = 1.0


def aligned(tokens: list[Token]) -> Any:
    """``parakeet_mlx.AlignedResult`` (one sentence)."""
    text = "".join(t.text for t in tokens)
    sentences = [SimpleNamespace(text=text, tokens=tokens)] if tokens else []
    return SimpleNamespace(text=text.strip(), sentences=sentences)


def tokens_for(seconds: float) -> list[Token]:
    """One word per 0.3 s of audio; "streaming" comes as two sub-word tokens."""
    out: list[Token] = []
    for i, word in enumerate(WORDS[: int(seconds / 0.3)]):
        start = 0.3 * i
        if word == "streaming":
            out += [Token(i, " stream", start, 0.15, 0.9), Token(i, "ing", start + 0.15, 0.1, 0.8)]
        else:
            out.append(Token(i, f" {word}", start, 0.2, 0.95))
    return out


@dataclass
class MLXFakes:
    threads: set[str] = field(default_factory=set)
    """Names of the threads MLX code ran on."""
    snapshots: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    snapshot_error: Exception | None = None
    # parakeet
    parakeet_loads: list[tuple[str, Any]] = field(default_factory=list)
    attention: list[tuple[Any, ...]] = field(default_factory=list)
    generated: list[int] = field(default_factory=list)
    """Samples per batch ``generate()`` (warm-up included)."""
    streamers: list[Any] = field(default_factory=list)
    load_error: Exception | None = None
    # whisper
    whisper_loads: list[tuple[str, Any]] = field(default_factory=list)
    whisper_calls: list[dict[str, Any]] = field(default_factory=list)
    holder: Any = None
    # mlx-audio
    tts_loads: list[str] = field(default_factory=list)
    tts_calls: list[dict[str, Any]] = field(default_factory=list)
    tts_sample_rate: int = 24_000
    tts_chunks: int = 3
    tts_chunk_seconds: float = 0.2
    tts_step_delay: float = 0.0
    tts_closed: list[bool] = field(default_factory=list)

    def mark(self) -> None:
        self.threads.add(threading.current_thread().name)

    # -------------------------------------------------------------------- install
    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mx = ModuleType("mlx.core")
        mx.array = FakeArray  # type: ignore[attr-defined]
        for name in ("float32", "float16", "bfloat16"):
            setattr(mx, name, name)
        mlx = ModuleType("mlx")
        mlx.core = mx  # type: ignore[attr-defined]
        hub = ModuleType("huggingface_hub")
        hub.snapshot_download = self._snapshot_download  # type: ignore[attr-defined]
        for name, module in {
            "mlx": mlx,
            "mlx.core": mx,
            "huggingface_hub": hub,
            **self._parakeet(),
            **self._whisper(),
            **self._mlx_audio(),
        }.items():
            monkeypatch.setitem(sys.modules, name, module)

    def _snapshot_download(self, repo: str, **kwargs: Any) -> str:
        self.snapshots.append((repo, kwargs))
        if self.snapshot_error is not None:
            raise self.snapshot_error
        return f"/hf/{repo}"

    def _parakeet(self) -> dict[str, ModuleType]:
        fakes = self

        class Encoder:
            def set_attention_model(self, *args: Any) -> None:
                fakes.mark()
                fakes.attention.append(args)

        class Model:
            preprocessor_config = SimpleNamespace(sample_rate=16_000, hop_length=160)

            def __init__(self) -> None:
                self.encoder = Encoder()

            def generate(self, mel: FakeArray, *, decoding_config: Any) -> list[Any]:
                fakes.mark()
                fakes.generated.append(len(mel))
                return [aligned(tokens_for(len(mel) / 16_000))]

        class StreamingParakeet:
            def __init__(self, model: Model, context_size: Any, depth: int = 1, *,
                         keep_original_attention: bool = False, decoding_config: Any = None) -> None:  # fmt: skip
                fakes.mark()
                self.args = (context_size, depth, keep_original_attention, decoding_config)
                self.samples = 0
                self.feeds: list[int] = []
                fakes.streamers.append(self)

            def add_audio(self, audio: FakeArray) -> None:
                fakes.mark()
                self.samples += len(audio)
                self.feeds.append(len(audio))

            @property
            def result(self) -> Any:
                return aligned(tokens_for(self.samples / 16_000))

        def from_pretrained(path: str, *, dtype: Any = None, cache_dir: Any = None) -> Model:
            fakes.mark()
            if fakes.load_error is not None:
                raise fakes.load_error
            fakes.parakeet_loads.append((path, dtype))
            return Model()

        @dataclass
        class Beam:
            beam_size: int = 5

        pm = ModuleType("parakeet_mlx")
        pm.from_pretrained = from_pretrained  # type: ignore[attr-defined]
        pm.StreamingParakeet = StreamingParakeet  # type: ignore[attr-defined]
        pm.Greedy = type("Greedy", (), {})  # type: ignore[attr-defined]
        pm.Beam = Beam  # type: ignore[attr-defined]
        pm.DecodingConfig = lambda decoding=None: SimpleNamespace(decoding=decoding)  # type: ignore[attr-defined]
        audio = ModuleType("parakeet_mlx.audio")
        audio.get_logmel = lambda x, args: x  # type: ignore[attr-defined]
        pm.audio = audio  # type: ignore[attr-defined]
        return {"parakeet_mlx": pm, "parakeet_mlx.audio": audio}

    def _whisper(self) -> dict[str, ModuleType]:
        fakes = self

        class ModelHolder:
            model: Any = None
            model_path: Any = None

        self.holder = ModelHolder

        def load_model(path: str, dtype: Any = None) -> Any:
            fakes.mark()
            fakes.whisper_loads.append((path, dtype))
            return SimpleNamespace(path=path)

        def transcribe(audio: Any, **kwargs: Any) -> dict[str, Any]:
            fakes.mark()
            # the real transcribe() reloads unless the holder already has this path
            assert ModelHolder.model_path == kwargs["path_or_hf_repo"]
            assert ModelHolder.model is not None
            fakes.whisper_calls.append({"audio": audio, **kwargs})
            words = [
                {"word": " Hello", "start": 0.0, "end": 0.4, "probability": 0.9},
                {"word": " there.", "start": 0.4, "end": 0.9, "probability": 0.8},
            ]
            return {
                "text": " Hello there.",
                "language": kwargs.get("language") or "de",
                "segments": [
                    {"start": 0.0, "end": 0.9, "text": " Hello there.", "tokens": [1, 2, 3],
                     "avg_logprob": -0.1, "words": words if kwargs.get("word_timestamps") else []},
                ],
            }  # fmt: skip

        mw = ModuleType("mlx_whisper")
        mw.transcribe = transcribe  # type: ignore[attr-defined]
        lm = ModuleType("mlx_whisper.load_models")
        lm.load_model = load_model  # type: ignore[attr-defined]
        tr = ModuleType("mlx_whisper.transcribe")
        tr.ModelHolder = ModelHolder  # type: ignore[attr-defined]
        return {"mlx_whisper": mw, "mlx_whisper.load_models": lm, "mlx_whisper.transcribe": tr}

    def _mlx_audio(self) -> dict[str, ModuleType]:
        fakes = self

        class TTSModel:
            def __init__(self) -> None:
                self.sample_rate = fakes.tts_sample_rate

            def generate(self, text: str, **kwargs: Any) -> Iterator[Any]:
                fakes.mark()
                fakes.tts_calls.append({"text": text, **kwargs})
                try:
                    n = round(fakes.tts_chunk_seconds * self.sample_rate)
                    for i in range(fakes.tts_chunks):
                        if fakes.tts_step_delay:
                            time.sleep(fakes.tts_step_delay)
                        fakes.mark()
                        tone = 0.3 * np.sin(np.arange(n) * (0.05 + 0.01 * i))
                        yield SimpleNamespace(audio=FakeArray(tone), sample_rate=self.sample_rate)
                except GeneratorExit:
                    fakes.mark()
                    fakes.tts_closed.append(True)
                    raise

        def load(path: str, **kwargs: Any) -> TTSModel:
            fakes.mark()
            fakes.tts_loads.append(path)
            return TTSModel()

        pkg = ModuleType("mlx_audio")
        tts = ModuleType("mlx_audio.tts")
        tts.load = load  # type: ignore[attr-defined]
        pkg.tts = tts  # type: ignore[attr-defined]
        return {"mlx_audio": pkg, "mlx_audio.tts": tts}


@pytest.fixture
def mlx_fakes(monkeypatch: pytest.MonkeyPatch) -> Iterator[MLXFakes]:
    monkeypatch.delenv("VAN_OFFLINE", raising=False)
    fakes = MLXFakes()
    fakes.install(monkeypatch)
    # a fresh MLX thread per test, so thread-name checks see only this test's calls
    worker = _mlx.MLXWorker("mlx-test")
    monkeypatch.setattr(_mlx, "WORKER", worker)
    yield fakes
    worker.close()
