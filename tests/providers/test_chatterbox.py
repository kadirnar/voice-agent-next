"""Chatterbox provider (and the shared local-torch TTS machinery).

Unit tests replace ``chatterbox`` and ``torch`` with in-memory fakes (no model, no
network, no GPU). ``test_real_model_*`` loads Chatterbox-Turbo/Nano (~3 GB / ~2 GB) and
only runs with ``-m model`` on a machine with the ``chatterbox`` extra and a CUDA GPU.
"""

from __future__ import annotations

import asyncio
import itertools
import sys
import threading
import time
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from voice_agent_next import create
from voice_agent_next.audio import AudioFrame
from voice_agent_next.errors import ConfigurationError, MissingDependencyError, ProviderError
from voice_agent_next.hardware import Backend
from voice_agent_next.metrics import TTSMetrics
from voice_agent_next.providers import _torch_tts
from voice_agent_next.providers.chatterbox import MODELS, ChatterboxTTS
from voice_agent_next.registry import get_provider
from voice_agent_next.stt import WordTiming
from voice_agent_next.tts import SentenceStreamAdapter, SynthesizedAudio
from voice_agent_next.utils.clock import now

SR = 24_000


# ------------------------------------------------------------------------------ fakes
class FakeHandle:
    def __init__(self, hooks: list[Any], hook: Any) -> None:
        self.hooks, self.hook = hooks, hook

    def remove(self) -> None:
        self.hooks.remove(self.hook)


class FakeModule:
    """Just enough of ``torch.nn.Module``: forward pre-hooks."""

    def __init__(self) -> None:
        self.pre_hooks: list[Any] = []

    def register_forward_pre_hook(self, hook: Any) -> FakeHandle:
        self.pre_hooks.append(hook)
        return FakeHandle(self.pre_hooks, hook)

    def __call__(self) -> None:
        for hook in list(self.pre_hooks):
            hook(self, ())


class FakeChatterbox:
    """Stands in for ``ChatterboxTurboTTS`` / ``ChatterboxMultilingualTTS``: per input
    character, 20 ms of tone, with 100 ms of silence before and after; one transformer
    "step" per 20 ms."""

    def __init__(self, backend: FakeBackend, kind: str, path: Any, device: str, nano: bool):
        self.backend = backend
        self.kind, self.path, self.device, self.nano = kind, path, device, nano
        self.sr = SR
        self.conds: Any = "builtin"
        self.t3 = types.SimpleNamespace(tfmr=FakeModule())
        self.calls: list[dict[str, Any]] = []
        self.prepared: list[tuple[str, dict[str, Any]]] = []

    def prepare_conditionals(self, wav_fpath: str, **kwargs: Any) -> None:
        self.prepared.append((wav_fpath, kwargs))
        if "short" in wav_fpath:
            raise AssertionError("Audio prompt must be longer than 5 seconds!")
        self.conds = f"cond:{Path(wav_fpath).name}"

    def generate(self, text: str, **kwargs: Any) -> np.ndarray:
        self.calls.append(
            {"text": text, "conds": self.conds, "thread": threading.current_thread().name} | kwargs
        )
        if self.backend.error is not None:
            raise self.backend.error
        steps = len(text)
        for _ in range(steps):
            self.t3.tfmr()  # the stop hook runs here
            if self.backend.delay:
                time.sleep(self.backend.delay)
            self.backend.steps += 1
        n = round(0.02 * SR * steps)
        tone = 0.5 * np.sin(2 * np.pi * 220.0 * np.arange(n) / SR)
        silence = np.zeros(SR // 10)
        return np.concatenate([silence, tone, silence]).astype(np.float32)[None, :]


class FakeBackend:
    def __init__(self) -> None:
        self.models: list[FakeChatterbox] = []
        self.error: Exception | None = None
        self.load_error: Exception | None = None
        self.delay = 0.0
        self.steps = 0
        self.nano_supported = True
        self.downloads: list[dict[str, Any]] = []

    @property
    def model(self) -> FakeChatterbox:
        return self.models[-1]

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        backend = self

        def new(kind: str, path: Any, device: str, nano: bool = False) -> FakeChatterbox:
            if backend.load_error is not None:
                raise backend.load_error
            model = FakeChatterbox(backend, kind, path, device, nano)
            backend.models.append(model)
            return model

        class Turbo:
            if backend.nano_supported:

                @classmethod
                def from_local(cls, ckpt_dir: Any, device: str, nano: bool = False) -> Any:
                    return new("turbo", ckpt_dir, device, nano)

            else:

                @classmethod
                def from_local(cls, ckpt_dir: Any, device: str) -> Any:  # type: ignore[misc]
                    return new("turbo", ckpt_dir, device)

        class Multilingual:
            @classmethod
            def from_local(cls, ckpt_dir: Any, device: str) -> Any:
                return new("multilingual", ckpt_dir, device)

            @classmethod
            def from_pretrained(cls, device: str) -> Any:
                return new("multilingual", "hub", device)

        def snapshot_download(**kwargs: Any) -> str:
            backend.downloads.append(kwargs)
            return "/hub/snapshot"

        root = types.ModuleType("chatterbox")
        turbo = types.ModuleType("chatterbox.tts_turbo")
        turbo.ChatterboxTurboTTS = Turbo  # type: ignore[attr-defined]
        mtl = types.ModuleType("chatterbox.mtl_tts")
        mtl.ChatterboxMultilingualTTS = Multilingual  # type: ignore[attr-defined]
        hub = types.ModuleType("huggingface_hub")
        hub.snapshot_download = snapshot_download  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "chatterbox", root)
        monkeypatch.setitem(sys.modules, "chatterbox.tts_turbo", turbo)
        monkeypatch.setitem(sys.modules, "chatterbox.mtl_tts", mtl)
        monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
        monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
        monkeypatch.setattr(
            _torch_tts,
            "select_torch_backend",
            lambda device, accelerators=(): Backend(
                "cpu" if device == "auto" else device, reason="test"
            ),
        )


@pytest.fixture
def backend(monkeypatch: pytest.MonkeyPatch) -> FakeBackend:
    fake = FakeBackend()
    fake.install(monkeypatch)
    return fake


@pytest.fixture
def ref(tmp_path: Path) -> Path:
    path = tmp_path / "me.wav"
    path.write_bytes(b"RIFF")
    return path


async def collect(tts: Any, text: str, **kwargs: Any) -> list[SynthesizedAudio]:
    return [chunk async for chunk in tts.synthesize(text, **kwargs)]


def audio_of(items: list[SynthesizedAudio]) -> list[AudioFrame]:
    return [i.frame for i in items if i.frame]


def words_of(items: list[SynthesizedAudio]) -> list[WordTiming]:
    return [w for i in items for w in (i.words or [])]


# ------------------------------------------------------------------ registration/config
def test_registered_with_metadata() -> None:
    spec = get_provider("tts", "chatterbox")
    assert spec.factory is ChatterboxTTS
    assert spec.default_model == "turbo"
    assert spec.extra == "chatterbox"
    assert spec.local and spec.env == ()
    assert set(spec.requires) == {"chatterbox", "torch"}
    assert set(spec.models) == set(MODELS)


def test_missing_dependency_is_reported_with_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "chatterbox", None)
    with pytest.raises(MissingDependencyError, match=r"voice-agent-next\[chatterbox\]"):
        ChatterboxTTS()


def test_create_from_spec(backend: FakeBackend) -> None:
    tts = create("tts", "chatterbox")
    assert isinstance(tts, ChatterboxTTS)
    assert (tts.model, tts.voice, tts.sample_rate, tts.language) == ("turbo", None, SR, "en")
    assert create("tts", "chatterbox/nano").model == "nano"
    mtl = create("tts", "chatterbox/multilingual", language="FR")
    assert (mtl.model, mtl.language) == ("multilingual", "fr")
    assert not backend.models  # the model loads lazily
    # estimated word timings are reported (word-exact truncation on barge-in)
    assert tts.capabilities.word_timestamps
    assert not create("tts", "chatterbox", word_timings=False).capabilities.word_timestamps


@pytest.mark.parametrize(
    "kwargs",
    [
        {"model": "large"},
        {"language": "fr"},  # Turbo is English only
        {"model": "multilingual", "language": "xx"},
        {"temperature": 0.0},
        {"top_p": 1.5},
        {"top_k": 0},
    ],
)
def test_invalid_options(backend: FakeBackend, kwargs: dict[str, Any]) -> None:
    with pytest.raises(ConfigurationError):
        ChatterboxTTS(**kwargs)


async def test_turbo_downloads_only_what_it_loads(backend: FakeBackend) -> None:
    tts = ChatterboxTTS(device="cuda")
    await tts.warmup()
    download = backend.downloads[0]
    assert download["repo_id"] == "ResembleAI/chatterbox-turbo"
    assert "t3_turbo_v1.safetensors" in download["allow_patterns"]
    assert "s3gen_meanflow.safetensors" in download["allow_patterns"]
    assert "s3gen.safetensors" not in download["allow_patterns"]  # the unused 1 GB decoder
    assert (backend.model.path, backend.model.device, backend.model.nano) == (
        Path("/hub/snapshot"),
        "cuda",
        False,
    )
    assert tts.device == "cuda"
    assert backend.model.calls[0]["thread"].startswith("chatterbox")
    await tts.aclose()


async def test_nano_and_local_weights(backend: FakeBackend, tmp_path: Path) -> None:
    tts = ChatterboxTTS(model="nano", model_path=tmp_path)
    await tts.warmup()
    assert backend.model.nano and backend.model.path == tmp_path
    assert not backend.downloads
    await tts.aclose()


async def test_nano_needs_a_recent_chatterbox(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = FakeBackend()
    fake.nano_supported = False
    fake.install(monkeypatch)
    tts = ChatterboxTTS(model="nano", model_path=tmp_path)
    with pytest.raises(ConfigurationError, match=r"newer than 0\.1\.7"):
        await tts.warmup()
    await tts.aclose()


async def test_multilingual_options(backend: FakeBackend) -> None:
    tts = ChatterboxTTS(model="multilingual", language="de", exaggeration=0.7, cfg_weight=0.3)
    await collect(tts, "Guten Tag, wie geht es Ihnen?")
    call = backend.model.calls[-1]
    assert backend.model.kind == "multilingual" and backend.model.path == "hub"
    assert (call["language_id"], call["exaggeration"], call["cfg_weight"]) == ("de", 0.7, 0.3)
    assert "top_k" not in call
    await tts.aclose()


async def test_load_failure_is_a_provider_error(backend: FakeBackend) -> None:
    backend.load_error = RuntimeError("CUDA out of memory")
    with pytest.raises(ProviderError, match="out of memory"):
        await ChatterboxTTS().warmup()


# ------------------------------------------------------------------ synthesis
async def test_synthesizes_with_sampling_options_and_metrics(backend: FakeBackend) -> None:
    metrics: list[TTSMetrics] = []
    tts = ChatterboxTTS(temperature=0.6, top_p=0.9, top_k=200, repetition_penalty=1.3)
    tts.on("metrics", metrics.append)
    items = await collect(tts, "Hello world.")
    audio = AudioFrame.concat(audio_of(items))
    assert audio.sample_rate == SR
    assert audio.duration == pytest.approx(0.02 * 12 + 0.2, abs=0.002)
    call = backend.model.calls[-1]
    assert call["text"] == "Hello world." and call["conds"] == "builtin"
    assert (call["temperature"], call["top_p"], call["top_k"], call["repetition_penalty"]) == (
        0.6,
        0.9,
        200,
        1.3,
    )
    assert items[-1].is_final
    m = metrics[-1]
    assert (m.provider, m.model, m.characters, m.streamed) == ("chatterbox", "turbo", 12, False)
    assert m.ttfb is not None and m.error is None
    await tts.aclose()


async def test_sentences_are_synthesized_separately_with_word_timings(
    backend: FakeBackend,
) -> None:
    tts = ChatterboxTTS()
    text = "Hello there, my friend. How are you doing today?"
    items = await collect(tts, text)
    texts = [c["text"] for c in backend.model.calls]
    assert texts == ["Hello there, my friend.", "How are you doing today?"]
    words = words_of(items)
    assert [w.word for w in words] == text.split()
    first = 0.1 + 0.02 * len(texts[0])  # end of the first sentence's speech
    assert words[0].start == pytest.approx(0.1, abs=0.01)
    assert words[3].end == pytest.approx(first, abs=0.01)
    assert words[4].start == pytest.approx(first + 0.2, abs=0.01)  # after both silences
    total = AudioFrame.concat(audio_of(items)).duration
    assert words[-1].end <= total
    await tts.aclose()


async def test_word_timings_can_be_disabled(backend: FakeBackend) -> None:
    tts = ChatterboxTTS(word_timings=False, split_sentences=False)
    items = await collect(tts, "One sentence. Another one.")
    assert not words_of(items)
    assert len(backend.model.calls) == 1
    await tts.aclose()


async def test_stream_uses_the_sentence_adapter_and_keeps_words(backend: FakeBackend) -> None:
    tts = ChatterboxTTS()
    stream = tts.stream()
    assert isinstance(stream, SentenceStreamAdapter)
    for delta in ("Good morning", " to you. ", "The weather ", "is lovely."):
        stream.push_text(delta)
    stream.end_input()
    items = [item async for item in stream]
    words = words_of(items)
    assert [w.word for w in words] == [
        "Good",
        "morning",
        "to",
        "you.",
        "The",
        "weather",
        "is",
        "lovely.",
    ]
    assert all(a.start <= b.start for a, b in itertools.pairwise(words))
    await stream.aclose()
    await tts.aclose()


async def test_voice_cloning_is_encoded_once(backend: FakeBackend, ref: Path) -> None:
    tts = ChatterboxTTS(voice=ref, norm_loudness=False)
    await tts.load_voice(ref)
    await collect(tts, "First.")
    await collect(tts, "Second.")
    assert backend.model.prepared == [(str(ref), {"norm_loudness": False})]
    assert [c["conds"] for c in backend.model.calls] == ["cond:me.wav"] * 2
    await collect(tts, "Default voice.", voice=None)  # None: the TTS default (the clone)
    assert backend.model.calls[-1]["conds"] == "cond:me.wav"
    await tts.aclose()


async def test_per_request_voice(backend: FakeBackend, ref: Path) -> None:
    tts = ChatterboxTTS()
    await collect(tts, "Default voice.")
    await collect(tts, "Cloned voice.", voice=str(ref))
    await collect(tts, "Default again.")
    assert [c["conds"] for c in backend.model.calls] == ["builtin", "cond:me.wav", "builtin"]
    await tts.aclose()


async def test_voice_errors(backend: FakeBackend, tmp_path: Path) -> None:
    tts = ChatterboxTTS()
    with pytest.raises(ConfigurationError, match="not an audio file"):
        await collect(tts, "Hello.", voice="alba")
    short = tmp_path / "short.wav"
    short.write_bytes(b"RIFF")
    with pytest.raises(ConfigurationError, match="longer than 5 seconds"):
        await collect(tts, "Hello.", voice=str(short))
    await tts.aclose()


async def test_inference_failure_is_a_provider_error(backend: FakeBackend) -> None:
    tts = ChatterboxTTS()
    await tts.warmup()
    backend.error = RuntimeError("CUDA error: device-side assert")
    with pytest.raises(ProviderError, match="device-side assert"):
        await collect(tts, "Hello.")
    await tts.aclose()


async def test_text_without_words_is_skipped(backend: FakeBackend) -> None:
    tts = ChatterboxTTS()
    assert not audio_of(await collect(tts, " ... "))
    assert backend.models == [] or backend.model.calls == []
    await tts.aclose()


async def test_closing_a_stream_stops_the_model_mid_sentence(backend: FakeBackend) -> None:
    tts = ChatterboxTTS(split_sentences=False)
    await tts.warmup()
    backend.delay = 0.01
    steps_before = backend.steps
    stream = tts.synthesize("A long sentence that would take a good while to render completely.")
    await asyncio.sleep(0.1)
    await stream.aclose()
    await asyncio.sleep(0.2)  # let the worker notice the stop
    steps = backend.steps - steps_before
    assert steps < 40  # far fewer than the 66 steps of the whole sentence
    assert not backend.model.t3.tfmr.pre_hooks  # the stop hook is removed
    await tts.aclose()


async def test_aclose_ends_pending_requests(backend: FakeBackend) -> None:
    backend.delay = 0.01
    tts = ChatterboxTTS()
    busy = tts.synthesize("A long sentence that keeps the worker busy for a while.")
    waiting = tts.synthesize("This one waits for the worker.")
    await asyncio.sleep(0.05)
    await tts.aclose()
    with pytest.raises(ProviderError, match="closed"):
        await asyncio.wait_for(collect_stream(waiting), timeout=5)
    await busy.aclose()
    backend.delay = 0.0
    assert audio_of(await collect(tts, "Still works."))  # the worker is recreated on demand
    await tts.aclose()


async def collect_stream(stream: Any) -> list[SynthesizedAudio]:
    return [item async for item in stream]


# ------------------------------------------------------------------- real model
def _cuda() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


@pytest.mark.model
@pytest.mark.timeout(1800)  # the first run downloads the model
@pytest.mark.parametrize("model", ["turbo", "nano"])
async def test_real_model_synthesizes_and_clones(model: str, tmp_path: Path) -> None:
    pytest.importorskip("chatterbox")
    if not _cuda():
        pytest.skip("needs a CUDA GPU")
    try:
        tts = ChatterboxTTS(model=model)
        await tts.warmup()
    except ConfigurationError as exc:  # Nano on chatterbox-tts 0.1.7
        pytest.skip(str(exc))
    assert tts.device == "cuda"
    text = "Hello! This is Chatterbox, speaking from a local GPU. How can I help you today?"
    t0 = now()
    ttfb: float | None = None
    items: list[SynthesizedAudio] = []
    async for chunk in tts.synthesize(text):
        if chunk.frame and ttfb is None:
            ttfb = now() - t0
        items.append(chunk)
    elapsed = now() - t0
    audio = AudioFrame.concat(audio_of(items))
    assert 2.5 < audio.duration < 15.0
    assert audio.rms() > 0.01
    assert [w.word for w in words_of(items)] == text.split()
    assert ttfb is not None
    print(
        f"\nchatterbox/{model}: first audio {ttfb * 1000:.0f} ms, RTF "
        f"{elapsed / audio.duration:.3f} ({audio.duration:.2f} s in {elapsed:.2f} s)"
    )
    # clone the voice just rendered (> 5 s of speech needed)
    from voice_agent_next.audio.wav import write_wav

    long = await tts.synthesize(text + " " + text).collect()
    ref = tmp_path / "ref.wav"
    write_wav(ref, long)
    cloned = await tts.synthesize("A cloned voice says hello.", voice=str(ref)).collect()
    assert cloned.duration > 1.0 and cloned.rms() > 0.01
    await tts.aclose()
