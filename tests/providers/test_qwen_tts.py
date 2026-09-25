"""Qwen3-TTS provider.

Unit tests replace ``qwen_tts`` and ``torch`` with in-memory fakes: a fake model that
"generates" one 16-code frame per talker step (running the talker's forward hooks like
the real one) and a fake codec whose audio encodes the frame values, so the streamed
chunks can be checked sample for sample. ``test_real_model_*`` loads Qwen3-TTS 0.6B
(~2.5 GB) and only runs with ``-m model`` on a machine with the ``qwen-tts`` extra and a
CUDA GPU.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
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
from voice_agent_next.providers import _torch_tts
from voice_agent_next.providers.qwen_tts import MODELS, QwenTTS, resolve_model
from voice_agent_next.registry import get_provider
from voice_agent_next.stt import WordTiming
from voice_agent_next.tts import SynthesizedAudio
from voice_agent_next.utils.clock import now

SR = 24_000
UP = 1920  # samples per 80 ms frame


# ------------------------------------------------------------------------------ fakes
class T:
    """A numpy-backed stand-in for the few ``torch.Tensor`` operations the provider uses."""

    def __init__(self, array: Any) -> None:
        self.a = np.asarray(array)

    shape = property(lambda self: self.a.shape)
    dtype = property(lambda self: self.a.dtype)
    T = property(lambda self: T(self.a.T))

    def __getitem__(self, key: Any) -> T:
        return T(self.a[key])

    def to(self, *args: Any, **kwargs: Any) -> T:
        return self

    def detach(self) -> T:
        return self

    def clone(self) -> T:
        return T(self.a.copy())

    def clamp(self, min: float) -> T:
        return T(np.maximum(self.a, min))

    def unsqueeze(self, dim: int) -> T:
        return T(np.expand_dims(self.a, dim))

    def float(self) -> T:
        return T(self.a.astype(np.float32))

    def cpu(self) -> T:
        return self

    def numpy(self) -> np.ndarray:
        return self.a


def fake_torch() -> types.ModuleType:
    torch = types.ModuleType("torch")
    torch.stack = lambda items: T(np.stack([i.a for i in items]))  # type: ignore[attr-defined]
    torch.cat = lambda items: T(np.concatenate([i.a for i in items]))  # type: ignore[attr-defined]
    torch.inference_mode = contextlib.nullcontext  # type: ignore[attr-defined]
    torch.float32 = "float32"  # type: ignore[attr-defined]
    torch.bfloat16 = "bfloat16"  # type: ignore[attr-defined]
    return torch


class Hooks:
    def __init__(self) -> None:
        self.forward_hooks: list[Any] = []

    def register_forward_hook(self, hook: Any) -> Any:
        self.forward_hooks.append(hook)
        hooks = self.forward_hooks
        return types.SimpleNamespace(remove=lambda: hooks.remove(hook))

    def step(self, codes: T | None) -> None:
        output = types.SimpleNamespace(hidden_states=((), codes))
        for hook in list(self.forward_hooks):
            hook(self, (), output)


class FakeCodec:
    """``codes (1, 16, n) -> wav (1, 1, n * UP)``: every frame's samples are its first code
    / 1000, so the output shows exactly which frames were decoded."""

    def __init__(self, backend: FakeBackend) -> None:
        self.backend = backend

    def parameters(self) -> Any:
        return iter([types.SimpleNamespace(device="cpu")])

    def __call__(self, codes: T) -> T:
        self.backend.decodes.append(codes.shape[-1])
        values = codes.a[0, 0, :].astype(np.float32) / 1000
        return T(np.repeat(values, UP)[None, None, :])


class FakeQwen:
    def __init__(self, backend: FakeBackend, repo: str, kwargs: dict[str, Any]) -> None:
        self.backend, self.repo, self.kwargs = backend, repo, kwargs
        self.calls: list[dict[str, Any]] = []
        self.prompts: list[dict[str, Any]] = []
        self.full_decodes = 0
        inner = types.SimpleNamespace(
            decoder=FakeCodec(backend),
            get_model_type=lambda: "qwen3_tts_tokenizer_12hz",
            get_decode_upsample_rate=lambda: UP,
        )
        tokenizer = types.SimpleNamespace(model=inner)
        tokenizer.decode = self._full_decode  # type: ignore[attr-defined]
        self.model = types.SimpleNamespace(
            tts_model_type=backend.model_type,
            talker=Hooks(),
            speech_tokenizer=tokenizer,
        )

    def _full_decode(self, encoded: Any) -> Any:
        self.full_decodes += 1
        return [np.zeros(10, np.float32)], SR

    def _generate(self, kind: str, kwargs: dict[str, Any]) -> Any:
        self.calls.append({"kind": kind, "thread": threading.current_thread().name} | kwargs)
        if self.backend.error is not None:
            raise self.backend.error
        frames = min(len(kwargs["text"]), kwargs["max_new_tokens"])
        talker = self.model.talker
        talker.step(None)  # prefill: no codes yet
        for i in range(frames):
            if self.backend.delay:
                time.sleep(self.backend.delay)
            self.backend.steps += 1
            talker.step(T(np.full((1, 16), 100 + i, dtype=np.int64)))
        wavs, sr = self.model.speech_tokenizer.decode([{"audio_codes": None}])
        return wavs, sr

    def generate_custom_voice(self, **kwargs: Any) -> Any:
        return self._generate("custom", kwargs)

    def generate_voice_clone(self, **kwargs: Any) -> Any:
        return self._generate("clone", kwargs)

    def generate_voice_design(self, **kwargs: Any) -> Any:
        return self._generate("design", kwargs)

    def create_voice_clone_prompt(self, **kwargs: Any) -> list[Any]:
        self.prompts.append(kwargs)
        ref_code = None
        if not kwargs["x_vector_only_mode"]:
            ref_code = T(np.full((30, 16), 7, dtype=np.int64))  # 30 frames of reference
        return [types.SimpleNamespace(ref_code=ref_code)]


class FakeBackend:
    def __init__(self) -> None:
        self.models: list[FakeQwen] = []
        self.model_type = "custom_voice"
        self.error: Exception | None = None
        self.load_error: Exception | None = None
        self.delay = 0.0
        self.steps = 0
        self.decodes: list[int] = []  # frames per codec call

    @property
    def model(self) -> FakeQwen:
        return self.models[-1]

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        backend = self

        class Qwen3TTSModel:
            @classmethod
            def from_pretrained(cls, repo: str, **kwargs: Any) -> FakeQwen:
                if backend.load_error is not None:
                    raise backend.load_error
                model = FakeQwen(backend, repo, kwargs)
                backend.models.append(model)
                return model

        qwen = types.ModuleType("qwen_tts")
        qwen.Qwen3TTSModel = Qwen3TTSModel  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "qwen_tts", qwen)
        monkeypatch.setitem(sys.modules, "torch", fake_torch())
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


def frame_values(audio: AudioFrame) -> list[int]:
    """The code of every 80 ms frame (see :class:`FakeCodec`)."""
    samples = audio.to_numpy().reshape(-1)[::UP]
    return [round(float(v) / 32768 * 1000) for v in samples]


# ------------------------------------------------------------------ registration/config
def test_registered_with_metadata() -> None:
    spec = get_provider("tts", "qwen-tts")
    assert spec.factory is QwenTTS
    assert spec.default_model == "0.6b-custom"
    assert spec.extra == "qwen-tts"
    assert spec.local and spec.env == ()
    assert set(spec.requires) == {"qwen_tts", "torch"}
    assert set(spec.models) == set(MODELS)
    assert get_provider("tts", "qwen3-tts").factory is QwenTTS


def test_resolve_model(tmp_path: Path) -> None:
    assert resolve_model(None) == "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
    assert resolve_model("0.6B-Base") == "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
    assert resolve_model("1.7b-design") == "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
    assert resolve_model("me/my-finetune") == "me/my-finetune"
    assert resolve_model(str(tmp_path)) == str(tmp_path)
    with pytest.raises(ConfigurationError, match="unknown Qwen3-TTS model"):
        resolve_model("huge")


def test_missing_dependency_is_reported_with_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "qwen_tts", None)
    with pytest.raises(MissingDependencyError, match=r"voice-agent-next\[qwen-tts\]"):
        QwenTTS()


def test_create_from_spec(backend: FakeBackend) -> None:
    tts = create("tts", "qwen-tts")
    assert isinstance(tts, QwenTTS)
    assert (tts.model, tts.voice, tts.sample_rate, tts.language) == (
        "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        None,
        SR,
        "auto",
    )
    tts = create("tts", "qwen-tts/1.7b-custom", voice="vivian", language="FR")
    assert (tts.voice, tts.language) == ("vivian", "french")
    assert not backend.models  # the model loads lazily


@pytest.mark.parametrize(
    "kwargs",
    [
        {"language": "klingon"},
        {"dtype": "int4"},
        {"first_chunk_frames": 0},
        {"chunk_frames": 0},
        {"context_frames": -1},
        {"temperature": 0.0},
    ],
)
def test_invalid_options(backend: FakeBackend, kwargs: dict[str, Any]) -> None:
    with pytest.raises(ConfigurationError):
        QwenTTS(**kwargs)


async def test_load_options(backend: FakeBackend) -> None:
    tts = QwenTTS(device="cuda", attn_implementation="flash_attention_2", cuda_graphs=False)
    await tts.warmup()
    assert backend.model.repo == "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
    assert backend.model.kwargs == {
        "device_map": "cuda",
        "dtype": "bfloat16",
        "attn_implementation": "flash_attention_2",
    }
    assert tts.model_type == "custom_voice" and tts.device == "cuda"
    await tts.aclose()
    cpu = QwenTTS()
    await cpu.warmup()
    assert backend.model.kwargs["dtype"] == "float32"
    await cpu.aclose()


async def test_load_failures(backend: FakeBackend) -> None:
    backend.load_error = OSError("Qwen/nope is not a local folder or a valid repository")
    with pytest.raises(ConfigurationError, match="valid repository"):
        await QwenTTS(model="Qwen/nope").warmup()
    backend.load_error = RuntimeError("CUDA out of memory")
    with pytest.raises(ProviderError, match="out of memory"):
        await QwenTTS().warmup()


# ------------------------------------------------------------------ streaming
async def test_streams_frames_in_chunks_as_they_are_generated(backend: FakeBackend) -> None:
    tts = QwenTTS(first_chunk_frames=2, chunk_frames=4, context_frames=3, split_sentences=False)
    backend.delay = 0.005
    items = await collect(tts, "Hello world.")  # 12 characters: 12 fake frames
    frames = audio_of(items)
    audio = AudioFrame.concat(frames)
    assert audio.duration == pytest.approx(12 * 0.08)
    assert frame_values(audio) == list(range(100, 112))  # every frame once, in order
    assert len(frames) >= 3  # streamed in several chunks, not one block
    assert frames[0].duration == pytest.approx(0.16, abs=0.08)  # first chunk: ~2 frames
    # every chunk after the first is decoded with (up to) 3 frames of left context
    assert backend.decodes[0] == len(frames[0].data) // 2 // UP
    assert all(n <= 4 + 3 + 4 for n in backend.decodes[1:])
    call = backend.model.calls[0]
    assert (call["kind"], call["speaker"], call["language"]) == ("custom", "ryan", "auto")
    assert call["non_streaming_mode"] is False
    assert call["thread"] == "qwen-tts-generate"
    assert backend.model.full_decodes == 0  # the redundant whole-utterance decode is skipped
    assert not backend.model.model.talker.forward_hooks  # the hook is removed
    assert items[-1].is_final
    await tts.aclose()


async def test_sampling_options_and_instruct(backend: FakeBackend) -> None:
    tts = QwenTTS(
        voice="vivian",
        instruct="Speak cheerfully",
        language="en",
        temperature=0.7,
        top_k=20,
        max_new_tokens=5,
    )
    items = await collect(tts, "Hello world.")
    call = backend.model.calls[0]
    assert (call["speaker"], call["instruct"], call["language"]) == (
        "vivian",
        "Speak cheerfully",
        "english",
    )
    assert (call["temperature"], call["top_k"], call["max_new_tokens"]) == (0.7, 20, 5)
    assert "top_p" not in call  # unset: the model's own default
    assert frame_values(AudioFrame.concat(audio_of(items))) == list(range(100, 105))
    await tts.aclose()


async def test_sentences_get_word_timings(backend: FakeBackend) -> None:
    tts = QwenTTS()
    text = "Hello there, my friend. How are you doing today?"
    items = await collect(tts, text)
    assert [c["text"] for c in backend.model.calls] == [
        "Hello there, my friend.",
        "How are you doing today?",
    ]
    words = words_of(items)
    assert [w.word for w in words] == text.split()
    total = AudioFrame.concat(audio_of(items)).duration
    assert words[0].start == pytest.approx(0.0, abs=0.01)
    assert words[3].end == pytest.approx(23 * 0.08, abs=0.01)
    assert words[4].start == pytest.approx(23 * 0.08, abs=0.01)
    assert words[-1].end == pytest.approx(total, abs=0.01)
    await tts.aclose()


async def test_voice_clone_with_transcript_continues_the_reference(
    backend: FakeBackend, ref: Path
) -> None:
    backend.model_type = "base"
    tts = QwenTTS(model="0.6b", voice=ref, ref_text="What I say.", context_frames=4)
    await tts.load_voice(ref)
    items = await collect(tts, "Hello world.")
    await collect(tts, "Again.")
    assert backend.model.prompts == [
        {"ref_audio": str(ref), "ref_text": "What I say.", "x_vector_only_mode": False}
    ]  # encoded once
    call = backend.model.calls[0]
    assert call["kind"] == "clone" and len(call["voice_clone_prompt"]) == 1
    # the reference codes are left context of the first chunk, not audio
    assert frame_values(AudioFrame.concat(audio_of(items))) == list(range(100, 112))
    first_chunk = len(audio_of(items)[0].data) // 2 // UP
    assert backend.decodes[0] == 4 + first_chunk  # 4 reference frames + the chunk
    await tts.aclose()


async def test_voice_clone_without_transcript(backend: FakeBackend, ref: Path) -> None:
    backend.model_type = "base"
    tts = QwenTTS(model="0.6b")
    await tts.warmup()  # a Base model without a voice: nothing to warm up with
    assert backend.model.calls == []
    with pytest.raises(ConfigurationError, match="clones voices"):
        await collect(tts, "Hello.")
    await collect(tts, "Hello.", voice=str(ref))
    assert backend.model.prompts[0]["x_vector_only_mode"] is True
    await tts.aclose()


async def test_voice_errors(backend: FakeBackend, ref: Path) -> None:
    tts = QwenTTS()  # CustomVoice
    with pytest.raises(ConfigurationError, match="built-in speakers only"):
        await collect(tts, "Hello.", voice=str(ref))
    with pytest.raises(ConfigurationError, match="does not clone voices"):
        await tts.load_voice(ref)
    await tts.aclose()
    backend.model_type = "base"
    base = QwenTTS(model="0.6b")
    with pytest.raises(ConfigurationError, match="does not exist"):
        await collect(base, "Hello.", voice="missing.wav")
    await base.aclose()


async def test_voice_design(backend: FakeBackend) -> None:
    backend.model_type = "voice_design"
    tts = QwenTTS(model="1.7b-design", voice="A warm, low voice")
    await collect(tts, "Hello.")
    call = backend.model.calls[-1]
    assert (call["kind"], call["instruct"]) == ("design", "A warm, low voice")
    await tts.aclose()


async def test_inference_failure_is_a_provider_error(backend: FakeBackend) -> None:
    tts = QwenTTS()
    await tts.warmup()
    backend.error = RuntimeError("CUDA error: device-side assert")
    with pytest.raises(ProviderError, match="device-side assert"):
        await collect(tts, "Hello.")
    assert not backend.model.model.talker.forward_hooks
    await tts.aclose()


async def test_first_chunk_arrives_before_the_sentence_ends(backend: FakeBackend) -> None:
    tts = QwenTTS(split_sentences=False)
    await tts.warmup()
    backend.delay = 0.01
    t0 = now()
    stream = tts.synthesize("A sentence long enough to take a while to render here.")
    first = await stream.__anext__()
    first_at = now() - t0
    rest = [item async for item in stream]
    total = now() - t0
    assert first.frame and len(rest) > 5
    assert first_at < total / 3
    await tts.aclose()


async def test_closing_a_stream_stops_the_generation(backend: FakeBackend) -> None:
    tts = QwenTTS(split_sentences=False)
    await tts.warmup()
    backend.delay = 0.01
    steps_before = backend.steps
    stream = tts.synthesize("A long sentence that would take a good while to render completely.")
    assert (await stream.__anext__()).frame
    await stream.aclose()
    await asyncio.sleep(0.2)  # let the generation notice the stop
    assert backend.steps - steps_before < 30  # far fewer than the 66 frames of the text
    assert not backend.model.model.talker.forward_hooks
    await tts.aclose()


# ------------------------------------------------------------------- real model
def _cuda() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


@pytest.mark.model
@pytest.mark.timeout(1800)  # the first run downloads the model (~2.5 GB)
async def test_real_model_streams() -> None:
    """The default model (0.6B CustomVoice). With ``VAN_QWEN_TTS_TEST_VOICE`` set to a
    reference WAV, the 0.6B Base model clones it instead (another ~2.5 GB)."""
    pytest.importorskip("qwen_tts")
    if not _cuda():
        pytest.skip("needs a CUDA GPU")
    reference = os.environ.get("VAN_QWEN_TTS_TEST_VOICE")
    if reference:
        tts = QwenTTS(model="0.6b", voice=reference, language="en")
    else:
        tts = QwenTTS(voice="ryan", language="en")
    await tts.warmup()
    assert tts.device == "cuda"
    text = "Hello! This is Qwen three TTS, streaming speech from a local GPU."
    t0 = now()
    ttfb: float | None = None
    items: list[SynthesizedAudio] = []
    async for chunk in tts.synthesize(text):
        if chunk.frame and ttfb is None:
            ttfb = now() - t0
        items.append(chunk)
    elapsed = now() - t0
    await tts.aclose()
    frames = audio_of(items)
    audio = AudioFrame.concat(frames)
    assert len(frames) > 3  # streamed
    assert 2.0 < audio.duration < 15.0
    assert audio.rms() > 0.01
    assert [w.word for w in words_of(items)] == text.split()
    assert ttfb is not None and ttfb < elapsed / 2
    print(
        f"\nqwen-tts/{tts.model}: first audio {ttfb * 1000:.0f} ms, RTF "
        f"{elapsed / audio.duration:.3f} ({audio.duration:.2f} s in {elapsed:.2f} s)"
    )
