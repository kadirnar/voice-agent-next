"""Kyutai Pocket TTS provider.

Unit tests replace ``pocket_tts`` and ``torch`` with in-memory fakes (no model, no
network). ``test_real_model_*`` downloads the ungated English weights (~220 MB) and only
runs with ``-m model``.
"""

from __future__ import annotations

import asyncio
import itertools
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
from voice_agent_next.metrics import TTSMetrics
from voice_agent_next.providers.pocket_tts import (
    VOICES,
    PocketTTS,
    estimate_word_timings,
    parse_model,
    speech_bounds,
)
from voice_agent_next.registry import get_provider
from voice_agent_next.stt import WordTiming
from voice_agent_next.tts import SentenceStreamAdapter, SynthesizedAudio
from voice_agent_next.utils.clock import now

SR = 24_000
FRAME = 1920  # one 80 ms Mimi frame


# ------------------------------------------------------------------------------ fakes
class FakeTensor:
    """Just enough of ``torch.Tensor`` for the provider: ``.detach().cpu().numpy()``."""

    def __init__(self, array: np.ndarray) -> None:
        self.array = array

    def detach(self) -> FakeTensor:
        return self

    def cpu(self) -> FakeTensor:
        return self

    def numpy(self) -> np.ndarray:
        return self.array


class FakeModel:
    """Stands in for ``pocket_tts.TTSModel``: per input character, 20 ms of tone, emitted
    in 80 ms frames, with 80 ms of silence before and after the speech."""

    def __init__(self, backend: FakeBackend, kwargs: dict[str, Any]) -> None:
        self.backend = backend
        self.kwargs = kwargs
        self.has_voice_cloning = backend.has_voice_cloning
        self.sample_rate = SR
        self.prompts: list[tuple[Any, bool]] = []
        self.calls: list[dict[str, Any]] = []

    def get_state_for_audio_prompt(self, source: Any, truncate: bool = False) -> dict[str, Any]:
        self.prompts.append((source, truncate))
        if self.backend.prompt_error is not None:
            raise self.backend.prompt_error
        if isinstance(source, Path) and source.suffix == ".safetensors":
            return {"voice": source.read_text()}
        return {"voice": str(source)}

    def generate_audio_stream(
        self,
        model_state: dict[str, Any],
        text_to_generate: str,
        max_tokens: int = 50,
        frames_after_eos: int | None = None,
        copy_state: bool = True,
        stop: threading.Event | None = None,
    ) -> Any:
        self.calls.append(
            {
                "text": text_to_generate,
                "voice": model_state["voice"],
                "frames_after_eos": frames_after_eos,
                "copy_state": copy_state,
                "thread": threading.current_thread().name,
            }
        )
        n = round(0.02 * SR * len(text_to_generate))
        tone = 0.5 * np.sin(2 * np.pi * 220.0 * np.arange(n) / SR)
        audio = np.concatenate([np.zeros(FRAME), tone, np.zeros(FRAME)]).astype(np.float32)
        for start in range(0, audio.size, FRAME):
            if stop is not None and stop.is_set():
                return
            if self.backend.delay:
                time.sleep(self.backend.delay)
            if self.backend.error is not None and start > 0:
                raise self.backend.error
            self.backend.frames += 1
            yield FakeTensor(audio[start : start + FRAME])


class FakeBackend:
    """Fake ``pocket_tts`` and ``torch`` modules and what was done with them."""

    def __init__(self) -> None:
        self.models: list[FakeModel] = []
        self.has_voice_cloning = True
        self.load_error: Exception | None = None
        self.prompt_error: Exception | None = None
        self.error: Exception | None = None
        self.delay = 0.0
        self.frames = 0
        self.threads: list[int] = []
        self.exports: list[str] = []

    @property
    def model(self) -> FakeModel:
        return self.models[-1]

    def modules(self) -> tuple[types.ModuleType, types.ModuleType]:
        backend = self

        class TTSModel:
            @classmethod
            def load_model(cls, **kwargs: Any) -> FakeModel:
                if backend.load_error is not None:
                    raise backend.load_error
                model = FakeModel(backend, kwargs)
                backend.models.append(model)
                return model

        def export_model_state(state: dict[str, Any], dest: str) -> None:
            backend.exports.append(str(dest))
            Path(dest).write_text(state["voice"])

        pocket = types.ModuleType("pocket_tts")
        pocket.TTSModel = TTSModel  # type: ignore[attr-defined]
        pocket.export_model_state = export_model_state  # type: ignore[attr-defined]
        torch = types.ModuleType("torch")
        torch.set_num_threads = backend.threads.append  # type: ignore[attr-defined]
        return pocket, torch


@pytest.fixture
def backend(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> FakeBackend:
    fake = FakeBackend()
    pocket, torch = fake.modules()
    monkeypatch.setitem(sys.modules, "pocket_tts", pocket)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setenv("VAN_CACHE_DIR", str(tmp_path / "cache"))
    return fake


async def collect(tts: PocketTTS, text: str, **kwargs: Any) -> list[SynthesizedAudio]:
    return [chunk async for chunk in tts.synthesize(text, **kwargs)]


def audio_of(items: list[SynthesizedAudio]) -> list[AudioFrame]:
    return [i.frame for i in items if i.frame]


def words_of(items: list[SynthesizedAudio]) -> list[WordTiming]:
    return [w for i in items for w in (i.words or [])]


# ------------------------------------------------------------------ pure helpers
def test_parse_model() -> None:
    assert parse_model(None) == (None, None)
    assert parse_model("marius") == (None, "marius")
    assert parse_model("french") == ("french", None)
    assert parse_model("FR") == ("french", None)
    assert parse_model("german/juergen") == ("german", "juergen")
    assert parse_model("italian_24l/giovanni") == ("italian_24l", "giovanni")
    assert parse_model("english_2026-04") == ("english_2026-04", None)
    with pytest.raises(ConfigurationError, match="unknown Pocket TTS language 'klingon'"):
        parse_model("klingon/worf")


def test_estimate_word_timings_is_proportional_and_leaves_pauses() -> None:
    words = estimate_word_timings("Hi, wonderful world.", 1.0, 2.0)
    assert [w.word for w in words] == ["Hi,", "wonderful", "world."]
    assert words[0].start == pytest.approx(1.0)
    assert words[-1].end == pytest.approx(2.0)
    assert all(a.end <= b.start for a, b in itertools.pairwise(words))
    assert words[1].start - words[0].end > 0  # the comma pause
    assert (words[1].end - words[1].start) > (words[2].end - words[2].start)  # longer word
    assert estimate_word_timings("   ", 0.0, 1.0) == []
    assert estimate_word_timings("Hello.", 1.0, 1.0) == []


def test_speech_bounds() -> None:
    audio = np.zeros(1000, np.float32)
    assert speech_bounds(audio) is None
    assert speech_bounds(np.zeros(0, np.float32)) is None
    audio[200:300] = 0.5
    audio[250] = -0.9
    assert speech_bounds(audio) == (200, 300)


# ------------------------------------------------------------------ registration/config
def test_registered_with_metadata() -> None:
    spec = get_provider("tts", "pocket-tts")
    assert spec.factory is PocketTTS
    assert spec.default_model == "alba"
    assert spec.extra == "pocket-tts"
    assert spec.local
    assert spec.env == ()
    assert set(spec.requires) == {"pocket_tts", "torch"}
    assert set(spec.models) == set(VOICES)


def test_missing_dependency_is_reported_with_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "pocket_tts", None)
    with pytest.raises(MissingDependencyError, match=r"voice-agent-next\[pocket-tts\]"):
        PocketTTS()


def test_create_from_spec(backend: FakeBackend) -> None:
    tts = create("tts", "pocket-tts")
    assert isinstance(tts, PocketTTS)
    assert tts.capabilities.word_timestamps  # estimated word timings
    assert not create("tts", "pocket-tts", word_timings=False).capabilities.word_timestamps
    assert (tts.model, tts.voice, tts.language, tts.sample_rate) == (
        "english",
        "alba",
        "english",
        SR,
    )
    marius = create("tts", "pocket-tts/marius")
    assert (marius.model, marius.voice) == ("english", "marius")
    french = create("tts", "pocket-tts/french")
    assert (french.model, french.voice) == ("french", "estelle")
    german = create("tts", "pocket-tts/german/alba", voice="vera")
    assert (german.model, german.voice) == ("german", "vera")
    assert PocketTTS(language="es").voice == "lola"
    assert PocketTTS(language="dutch_24l").voice == "daan"
    assert not backend.models  # the model loads lazily


@pytest.mark.parametrize(
    "kwargs",
    [
        {"language": "klingon"},
        {"language": "french", "config": "my.yaml"},
        {"temperature": -1.0},
        {"sampler_decode_steps": 0},
        {"frames_after_eos": -1},
        {"num_threads": 0},
    ],
)
def test_invalid_options(backend: FakeBackend, kwargs: dict[str, Any]) -> None:
    with pytest.raises(ConfigurationError):
        PocketTTS(**kwargs)


async def test_model_options_are_passed_to_pocket_tts(backend: FakeBackend) -> None:
    tts = PocketTTS(
        model="french",
        temperature=0.5,
        sampler_decode_steps=2,
        eos_threshold=-3.0,
        quantize=True,
        num_threads=2,
        frames_after_eos=4,
    )
    await tts.warmup()
    assert backend.model.kwargs == {
        "language": "french",
        "temp": 0.5,
        "sampler_decode_steps": 2,
        "eos_threshold": -3.0,
        "quantize": True,
    }
    assert backend.threads == [2]
    assert backend.model.calls[0]["frames_after_eos"] == 4
    assert backend.model.calls[0]["copy_state"] is True
    custom = PocketTTS(config="my-model.yaml", voice="ref.safetensors")
    assert custom.model == "my-model" and custom.language is None
    await custom.aclose()
    await tts.aclose()


async def test_load_failures(backend: FakeBackend) -> None:
    backend.load_error = ValueError("Config should be a path to a YAML file")
    with pytest.raises(ConfigurationError, match="YAML"):
        await PocketTTS(config="bad.txt").warmup()
    backend.load_error = RuntimeError("HTTP 503")
    with pytest.raises(ProviderError, match="HTTP 503"):
        await PocketTTS().warmup()


# ------------------------------------------------------------------ synthesis
async def test_streams_frames_as_they_are_generated(backend: FakeBackend) -> None:
    metrics: list[TTSMetrics] = []
    tts = PocketTTS()
    tts.on("metrics", metrics.append)
    items = await collect(tts, "Hello world.")
    frames = audio_of(items)
    assert all(f.sample_rate == SR and f.duration == pytest.approx(0.08) for f in frames[:-1])
    total = AudioFrame.concat(frames)
    assert total.duration == pytest.approx(0.02 * 12 + 0.16, abs=0.001)
    assert items[0].text == "Hello world." and items[-1].is_final
    assert backend.model.calls[0]["text"] == "Hello world."
    assert backend.model.calls[0]["voice"] == "alba"
    assert backend.model.calls[0]["thread"].startswith("pocket-tts")
    m = metrics[-1]
    assert (m.provider, m.model, m.characters, m.streamed) == ("pocket-tts", "english", 12, False)
    assert m.ttfb is not None and m.error is None
    await tts.aclose()


async def test_first_frame_arrives_before_the_synthesis_ends(backend: FakeBackend) -> None:
    backend.delay = 0.03
    tts = PocketTTS()
    await tts.warmup()
    t0 = now()
    stream = tts.synthesize("A sentence long enough to take a while to render here.")
    first = await stream.__anext__()
    first_at = now() - t0
    rest = [item async for item in stream]
    total = now() - t0
    assert first.frame
    assert len(rest) > 10
    assert first_at < total / 3  # audio streams: not rendered in one piece
    await tts.aclose()


async def test_sentences_are_synthesized_separately_with_word_timings(
    backend: FakeBackend,
) -> None:
    tts = PocketTTS()
    items = await collect(tts, "Hello there, my friend. How are you doing today?")
    assert [c["text"] for c in backend.model.calls] == [
        "Hello there, my friend.",
        "How are you doing today?",
    ]
    words = words_of(items)
    assert [w.word for w in words] == [
        "Hello",
        "there,",
        "my",
        "friend.",
        "How",
        "are",
        "you",
        "doing",
        "today?",
    ]
    first_len = (0.02 * len("Hello there, my friend.") * SR + 2 * FRAME) / SR
    # speech starts after the 80 ms of leading silence of each segment
    assert words[0].start == pytest.approx(0.08, abs=0.002)
    assert words[3].end == pytest.approx(first_len - 0.08, abs=0.002)
    assert words[4].start == pytest.approx(first_len + 0.08, abs=0.002)
    assert all(a.start <= b.start for a, b in itertools.pairwise(words))
    # the words of a segment arrive right after its audio
    first_words = next(i for i, item in enumerate(items) if item.words)
    assert not items[first_words].frame
    audio_before = sum(f.duration for f in audio_of(items[:first_words]))
    assert audio_before == pytest.approx(first_len, abs=0.002)


async def test_word_timings_can_be_disabled(backend: FakeBackend) -> None:
    tts = PocketTTS(word_timings=False, split_sentences=False)
    items = await collect(tts, "Hello there. How are you?")
    assert words_of(items) == []
    assert [c["text"] for c in backend.model.calls] == ["Hello there. How are you?"]


async def test_stream_uses_the_sentence_adapter_and_keeps_words(backend: FakeBackend) -> None:
    tts = PocketTTS()
    stream = tts.stream()
    assert isinstance(stream, SentenceStreamAdapter)
    stream.push_text("Hello there, my friend. How ")
    stream.push_text("are you today?")
    stream.end_input()
    events = [e async for e in stream]
    await stream.aclose()
    assert [e.text for e in events if e.text] == ["Hello there, my friend.", "How are you today?"]
    words = words_of(events)
    assert [w.word for w in words] == [
        "Hello",
        "there,",
        "my",
        "friend.",
        "How",
        "are",
        "you",
        "today?",
    ]
    # the adapter trims the leading silence: the first word starts at the stream's start
    assert words[0].start == pytest.approx(0.0, abs=0.03)
    audio = sum(e.frame.duration for e in events)
    assert words[-1].end <= audio + 1e-6


async def test_per_request_voice_and_voice_cache(backend: FakeBackend) -> None:
    tts = PocketTTS(voice="marius")
    await collect(tts, "One.")
    await collect(tts, "Two.", voice="vera")
    await collect(tts, "Three.")
    assert [c["voice"] for c in backend.model.calls] == ["marius", "vera", "marius"]
    assert [p[0] for p in backend.model.prompts] == ["marius", "vera"]  # cached states
    assert await tts.list_voices() == list(VOICES)
    await tts.aclose()


async def test_unknown_voice_is_a_configuration_error(backend: FakeBackend) -> None:
    tts = PocketTTS(voice="nobody")
    with pytest.raises(ConfigurationError, match="unknown Pocket TTS voice 'nobody'"):
        await collect(tts, "Hello there.")


async def test_cloned_voice_is_encoded_once_and_cached_on_disk(
    backend: FakeBackend, tmp_path: Path
) -> None:
    ref = tmp_path / "me.wav"
    ref.write_bytes(b"RIFF fake wav")
    cache = tmp_path / "voices"
    tts = PocketTTS(voice=str(ref), voice_cache_dir=cache)
    await tts.load_voice(ref)
    await collect(tts, "Hello there.")
    assert backend.model.prompts == [(ref, True)]  # encoded once, truncated to 30 s
    assert backend.model.calls[0]["voice"] == str(ref)
    cached = list(cache.glob("*.safetensors"))
    assert len(cached) == 1 and not list(cache.glob("*.part"))
    await tts.aclose()

    # a new instance (or process) loads the cached state instead of the audio
    again = PocketTTS(voice=ref, voice_cache_dir=cache)
    await collect(again, "Hello again.")
    assert backend.model.prompts == [(cached[0], False)]
    # a different file (same name, new content) is encoded again
    ref.write_bytes(b"RIFF another voice")
    other = PocketTTS(voice=ref, voice_cache_dir=cache)
    await other.load_voice(ref)
    assert backend.model.prompts == [(ref, True)]
    assert len(list(cache.glob("*.safetensors"))) == 2


async def test_cloning_without_the_gated_weights(backend: FakeBackend, tmp_path: Path) -> None:
    backend.has_voice_cloning = False
    ref = tmp_path / "me.wav"
    ref.write_bytes(b"RIFF fake wav")
    tts = PocketTTS(voice=ref, voice_cache_dir=tmp_path / "voices")
    with pytest.raises(ConfigurationError, match="gated kyutai/pocket-tts"):
        await collect(tts, "Hello there.")
    # exported voice states still work
    state = tmp_path / "me.safetensors"
    state.write_text("exported-me")
    items = await collect(tts, "Hello there.", voice=str(state))
    assert audio_of(items)
    assert backend.model.calls[-1]["voice"] == "exported-me"


async def test_export_voice(backend: FakeBackend, tmp_path: Path) -> None:
    tts = PocketTTS()
    dest = await tts.export_voice("javert", tmp_path / "out" / "javert.safetensors")
    assert dest.read_text() == "javert"
    await tts.aclose()


async def test_voice_prompt_failure_is_a_provider_error(backend: FakeBackend) -> None:
    backend.prompt_error = OSError("cannot fetch hf://x/y.wav")
    tts = PocketTTS(voice="hf://x/y.wav")
    with pytest.raises(ProviderError, match="cannot fetch"):
        await collect(tts, "Hello there.")


async def test_inference_failure_is_a_provider_error(backend: FakeBackend) -> None:
    metrics: list[TTSMetrics] = []
    tts = PocketTTS()
    tts.on("metrics", metrics.append)
    backend.error = RuntimeError("torch exploded")
    with pytest.raises(ProviderError, match="torch exploded"):
        await collect(tts, "Hello there.")
    assert metrics[-1].error is not None


async def test_text_without_words_is_skipped(backend: FakeBackend) -> None:
    tts = PocketTTS()
    assert audio_of(await collect(tts, " ... !")) == []
    assert backend.model.calls == []


async def test_closing_a_stream_stops_the_generation(backend: FakeBackend) -> None:
    backend.delay = 0.02
    tts = PocketTTS()
    await tts.warmup()
    frames_before = backend.frames
    stream = tts.synthesize("First sentence is here. Second sentence is here. Third one.")
    first = await stream.__anext__()
    assert first.frame
    await stream.aclose()
    await asyncio.sleep(0.2)  # let the worker notice the stop
    generated = backend.frames - frames_before
    assert generated < 10  # far fewer than the ~60 frames of the whole text
    assert len(backend.model.calls) == 2  # warm-up + the first sentence only
    await tts.aclose()


async def test_aclose_ends_pending_requests(backend: FakeBackend) -> None:
    backend.delay = 0.02
    tts = PocketTTS()
    busy = tts.synthesize("A long sentence that keeps the worker busy for a while.")
    waiting = tts.synthesize("This one waits for the worker.")
    assert (await busy.__anext__()).frame
    await tts.aclose()
    with pytest.raises(ProviderError, match="closed"):
        await asyncio.wait_for(collect_stream(waiting), timeout=5)
    await busy.aclose()
    assert audio_of(await collect(tts, "Still works."))  # the worker is recreated on demand
    await tts.aclose()


async def collect_stream(stream: Any) -> list[SynthesizedAudio]:
    return [item async for item in stream]


# ------------------------------------------------------------------- real model
@pytest.mark.model
@pytest.mark.timeout(900)  # the first run downloads the model (~220 MB)
async def test_real_model_streams_a_sentence() -> None:
    pytest.importorskip("pocket_tts")
    tts = PocketTTS(voice=os.environ.get("VAN_POCKET_TTS_TEST_VOICE", "alba"))
    metrics: list[TTSMetrics] = []
    tts.on("metrics", metrics.append)
    await tts.warmup()

    text = "Hello! This is Pocket TTS, streaming speech from a small local model."
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
    assert audio.sample_rate == SR
    assert len(frames) > 5  # streamed in frames, not one block
    assert 2.0 < audio.duration < 12.0
    assert audio.rms() > 0.01  # speech, not silence
    words = words_of(items)
    assert [w.word for w in words] == text.split()
    assert words[-1].end <= audio.duration
    assert metrics[-1].error is None and metrics[-1].ttfb is not None
    assert ttfb is not None and ttfb < elapsed / 2
    print(
        f"\npocket-tts: TTFB {ttfb * 1000:.0f} ms, RTF {elapsed / audio.duration:.3f} "
        f"({audio.duration:.2f} s of audio in {elapsed:.2f} s)"
    )


async def test_numbers_are_spoken_and_word_timings_point_to_the_original(
    backend: FakeBackend,
) -> None:
    tts = PocketTTS()
    items = await collect(tts, "Your order 58213 costs $42.50.")
    assert backend.model.calls[0]["text"] == (
        "Your order five eight two one three costs forty two dollars and fifty cents."
    )
    assert items[0].text == "Your order 58213 costs $42.50."
    assert [w.word for w in words_of(items)] == ["Your", "order", "58213", "costs", "$42.50."]
    raw = await collect(PocketTTS(normalize=False), "It is 5.")
    assert backend.model.calls[-1]["text"] == "It is 5."
    assert [w.word for w in words_of(raw)] == ["It", "is", "5."]
    assert PocketTTS(language="french").text_language(None) == "french"
    await collect(tts, "Call 555-0142.")
    assert backend.models[0].calls[-1]["text"] == "Call five-five-five, zero-one-four-two."
    await tts.aclose()
