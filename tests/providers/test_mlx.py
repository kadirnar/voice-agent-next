"""MLX speech recognition: Parakeet (``mlx``) and Whisper (``mlx_whisper``).

These tests run on every OS against fake ``mlx`` / ``parakeet_mlx`` / ``mlx_whisper``
modules (``mlx_fakes.py``). The real models are exercised by ``test_mlx_models.py`` on
Apple silicon.
"""

from __future__ import annotations

import asyncio
import math
from typing import Any

import numpy as np
import pytest

from voice_agent_next import AudioFrame, create
from voice_agent_next import models as model_catalog
from voice_agent_next.errors import (
    ConfigurationError,
    MissingDependencyError,
    ProviderConnectionError,
    ProviderError,
)
from voice_agent_next.metrics import STTMetrics
from voice_agent_next.providers import _mlx
from voice_agent_next.providers.mlx import ParakeetMLXSTT
from voice_agent_next.providers.mlx_whisper import MLXWhisperSTT
from voice_agent_next.providers.mock import synth_speech
from voice_agent_next.registry import get_provider
from voice_agent_next.stt import StreamAdapter, STTEvent, STTEventType
from voice_agent_next.stt_guard import HallucinationGuard

from .mlx_fakes import MLXFakes, hf
from .whisper_helpers import LevelVAD, push_settled, stream_paced


def chunks(frame: AudioFrame, step: float = 0.02) -> list[AudioFrame]:
    n = math.ceil(frame.duration / step - 1e-9)
    return [frame.slice(i * step, (i + 1) * step) for i in range(n)]


def speech(seconds: float) -> AudioFrame:
    return synth_speech(seconds, 16_000)


async def drain(stream: Any) -> list[STTEvent]:
    return [ev async for ev in stream]


# ------------------------------------------------------------------- registry
def test_registry_metadata() -> None:
    stt = get_provider("stt", "mlx")
    assert stt.factory is ParakeetMLXSTT and stt.default_model == "parakeet-tdt-0.6b-v3"
    assert stt.platforms == ("darwin",) and stt.local and stt.extra == "mlx"
    assert stt.requires == ("parakeet_mlx", "mlx") and stt.env == ()
    assert get_provider("stt", "parakeet-mlx") is stt  # alias
    whisper = get_provider("stt", "mlx-whisper")
    assert whisper.factory is MLXWhisperSTT and whisper.default_model == "large-v3-turbo"
    assert whisper.platforms == ("darwin",) and whisper.extra == "mlx-whisper"
    assert not stt.supports_platform("linux") and stt.supports_platform("darwin")


def test_models_are_in_the_catalog() -> None:
    info = model_catalog.get_model("mlx/parakeet-tdt-0.6b-v3")
    [f] = info.files
    assert (f.source, f.location, f.patterns) == (
        "hf-repo",
        "mlx-community/parakeet-tdt-0.6b-v3",
        ("config.json", "model.safetensors"),
    )
    assert info.kinds == ("stt",) and info.size and info.size > 2e9
    tiny = model_catalog.get_model("mlx-whisper/tiny")
    assert tiny.files[0].location == "mlx-community/whisper-tiny" and tiny.size < 80e6
    assert model_catalog.get_model("mlx-whisper/turbo").model == "large-v3-turbo"


def test_missing_mlx_off_apple_silicon(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_mlx, "_importable", lambda module: False)
    monkeypatch.setattr(_mlx, "is_apple_silicon", lambda: False)
    with pytest.raises(MissingDependencyError, match="Apple silicon"):
        create("stt", "mlx")
    with pytest.raises(MissingDependencyError, match=r"voice-agent-next\[mlx-whisper\]"):
        create("stt", "mlx_whisper/tiny")


def test_missing_package_on_a_mac(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_mlx, "_importable", lambda module: False)
    monkeypatch.setattr(_mlx, "is_apple_silicon", lambda: True)
    monkeypatch.setitem(__import__("sys").modules, "parakeet_mlx", None)  # import fails
    with pytest.raises(MissingDependencyError, match=r"pip install 'voice-agent-next\[mlx\]'"):
        create("stt", "mlx")


@pytest.mark.usefixtures("mlx_fakes")
def test_option_validation() -> None:
    with pytest.raises(ConfigurationError, match="mlx_whisper/whisper-large-v3-turbo"):
        create("stt", "mlx/whisper-large-v3-turbo")
    with pytest.raises(ConfigurationError, match="mlx/parakeet"):
        create("stt", "mlx_whisper/parakeet-tdt-0.6b-v3")
    for bad in ({"dtype": "int8"}, {"chunk_duration": 0}, {"beam_size": 0}, {"depth": 0}):
        with pytest.raises(ConfigurationError):
            create("stt", "mlx", **bad)


# ---------------------------------------------------------------- parakeet batch
async def test_batch_transcription(mlx_fakes: MLXFakes) -> None:
    stt = create("stt", "mlx", language="en")
    assert stt.capabilities.streaming and stt.capabilities.word_timestamps
    metrics: list[STTMetrics] = []
    stt.on("metrics", metrics.append)
    await stt.warmup()
    assert mlx_fakes.snapshots == [
        (
            "mlx-community/parakeet-tdt-0.6b-v3",
            {
                "revision": None,
                "allow_patterns": ["config.json", "model.safetensors"],
                "local_files_only": False,
            },
        )
    ]
    assert mlx_fakes.parakeet_loads == [(hf("mlx-community/parakeet-tdt-0.6b-v3"), "bfloat16")]
    assert mlx_fakes.generated == [8000]  # warm-up: half a second of silence

    # 48 kHz stereo in: resampled to 16 kHz mono by STT.transcribe()
    result = await stt.transcribe(synth_speech(2.2, 48_000).to_channels(2))
    assert result.text == "Hello world, this is a streaming test."
    assert result.language == "en"
    assert [w.word for w in result.words or []] == [
        "Hello", "world,", "this", "is", "a", "streaming", "test.",
    ]  # fmt: skip
    streaming = result.words[5]  # type: ignore[index]
    assert (streaming.start, streaming.end) == pytest.approx((1.5, 1.75))
    assert streaming.confidence == pytest.approx(math.sqrt(0.9 * 0.8), rel=1e-6)
    assert (result.start_time, result.end_time) == pytest.approx((0.0, 2.0))
    assert result.confidence is not None and 0.9 < result.confidence < 1.0
    assert metrics[-1].audio_duration == pytest.approx(2.2) and not metrics[-1].streamed
    assert mlx_fakes.threads == {"mlx-test_0"}  # every MLX call on the one MLX thread
    # the model loads once
    await stt.transcribe(speech(0.5))
    assert len(mlx_fakes.parakeet_loads) == 1


async def test_batch_empty_audio_skips_the_model(mlx_fakes: MLXFakes) -> None:
    stt = create("stt", "mlx")
    result = await stt.transcribe(AudioFrame.empty(16_000))
    assert result.text == "" and mlx_fakes.parakeet_loads == []


async def test_model_names_repos_and_options(
    mlx_fakes: MLXFakes, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VAN_OFFLINE", "1")
    stt = create("stt", "mlx/nvidia/parakeet-tdt_ctc-110m", dtype="float32", beam_size=4)
    await stt.warmup()
    repo, kwargs = mlx_fakes.snapshots[-1]
    assert repo == "mlx-community/parakeet-tdt_ctc-110m" and kwargs["local_files_only"] is True
    assert mlx_fakes.parakeet_loads[-1][1] == "float32"
    assert stt._decoding_config().decoding.beam_size == 4
    # a repository id is used as given; a local directory is not downloaded
    await create("stt", "mlx/someone/parakeet-custom").warmup()
    assert mlx_fakes.snapshots[-1][0] == "someone/parakeet-custom"
    loads = len(mlx_fakes.snapshots)
    local = create("stt", "mlx", model=str(__import__("pathlib").Path(__file__).parent))
    await local.warmup()
    assert len(mlx_fakes.snapshots) == loads
    assert mlx_fakes.parakeet_loads[-1][0] == str(__import__("pathlib").Path(__file__).parent)


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (type("RepositoryNotFoundError", (Exception,), {})("404"), ConfigurationError),
        (type("LocalEntryNotFoundError", (Exception,), {})("offline"), ProviderConnectionError),
        (ConnectionError("down"), ProviderConnectionError),
        (RuntimeError("boom"), ProviderError),
    ],
)
async def test_download_errors_are_mapped(
    mlx_fakes: MLXFakes, error: Exception, expected: type[Exception]
) -> None:
    mlx_fakes.snapshot_error = error
    with pytest.raises(expected, match="downloading model"):
        await create("stt", "mlx").warmup()


async def test_load_errors_are_mapped(mlx_fakes: MLXFakes) -> None:
    mlx_fakes.load_error = ValueError("Model is not supported yet!")
    with pytest.raises(ConfigurationError, match="not supported"):
        await create("stt", "mlx").warmup()


# ------------------------------------------------------------- parakeet streaming
async def test_incremental_streaming_utterances(mlx_fakes: MLXFakes) -> None:
    stt = create("stt", "mlx", chunk_duration=0.32, incremental=True)
    metrics: list[STTMetrics] = []
    stt.on("metrics", metrics.append)
    stream = stt.stream()
    events: list[STTEvent] = []

    async def next_interim() -> str:
        async for ev in stream:
            events.append(ev)
            if ev.type == STTEventType.INTERIM_TRANSCRIPT:
                return ev.text
        raise AssertionError("stream ended")

    frames = chunks(speech(1.3))
    interims = []
    for i in range(4):  # 16 frames of 20 ms = one 0.32 s chunk each
        for frame in frames[16 * i : 16 * (i + 1)]:
            stream.push_audio(frame)
        interims.append(await next_interim())
    assert interims == ["Hello", "Hello world,", "Hello world, this", "Hello world, this is"]
    for frame in frames[64:]:  # 20 ms tail: below the 50 ms minimum, not fed
        stream.push_audio(frame)
    stream.flush()
    async for ev in stream:
        events.append(ev)
        if ev.type == STTEventType.FINAL_TRANSCRIPT:
            break
    for frame in chunks(speech(0.7)):
        stream.push_audio(frame)
    stream.end_input()
    events += await drain(stream)
    await stream.aclose()

    types = [e.type for e in events]
    finals = [e for e in events if e.type == STTEventType.FINAL_TRANSCRIPT]
    assert [f.text for f in finals] == ["Hello world, this is", "Hello world,"]
    first = [e for e in events if e.segment_id == finals[0].segment_id]
    assert first[0].type == STTEventType.START_OF_SPEECH
    assert first[-1].type == STTEventType.END_OF_SPEECH
    assert types.count(STTEventType.START_OF_SPEECH) == 2
    assert finals[0].segment_id != finals[1].segment_id
    # the second utterance's times continue after the first one's audio
    assert finals[1].transcript is not None and finals[1].transcript.words
    assert finals[1].transcript.words[0].start == pytest.approx(1.3)
    # audio is fed in chunk_duration pieces, not per 20 ms frame
    [first_streamer, second_streamer] = mlx_fakes.streamers
    assert first_streamer.feeds == [5120, 5120, 5120, 5120]
    assert sum(second_streamer.feeds) == 11200
    assert first_streamer.args[:3] == ((256, 256), 1, True)
    # local attention while a stream is open, restored when its utterance ends
    assert mlx_fakes.attention == [
        ("rel_pos_local_attn", (256, 256)),
        ("rel_pos",),
        ("rel_pos_local_attn", (256, 256)),
        ("rel_pos",),
    ]
    streamed = [m for m in metrics if m.streamed]
    assert len(streamed) == 2 and all(m.latency is not None for m in streamed)
    assert streamed[0].audio_duration == pytest.approx(1.3)
    assert mlx_fakes.threads == {"mlx-test_0"}


async def test_concurrent_streams_share_local_attention(mlx_fakes: MLXFakes) -> None:
    stt = create("stt", "mlx", incremental=True)
    a, b = stt.stream(), stt.stream()
    for frame in chunks(speech(0.4)):
        a.push_audio(frame)
        b.push_audio(frame)
    for s in (a, b):  # both have fed audio: both use local attention now
        async for ev in s:
            if ev.type == STTEventType.INTERIM_TRANSCRIPT:
                break
    a.end_input()
    await drain(a)
    assert mlx_fakes.attention == [("rel_pos_local_attn", (256, 256))]  # b still streams
    b.end_input()
    await drain(b)
    assert mlx_fakes.attention[-1] == ("rel_pos",)
    await a.aclose()
    await b.aclose()


async def test_flush_without_speech_gives_an_empty_final(mlx_fakes: MLXFakes) -> None:
    stt = create("stt", "mlx", interim_results=False)
    stream = stt.stream()
    stream.push_audio(AudioFrame.silence(0.02, 16_000))  # below the 50 ms tail threshold
    stream.flush()
    stream.end_input()
    events = await drain(stream)
    assert [(e.type, e.text) for e in events] == [
        (STTEventType.FINAL_TRANSCRIPT, ""),
        (STTEventType.FINAL_TRANSCRIPT, ""),
    ]
    assert mlx_fakes.streamers == []
    await stream.aclose()


async def test_no_interim_results_when_disabled(mlx_fakes: MLXFakes) -> None:
    stt = create("stt", "mlx", interim_results=False)
    assert not stt.capabilities.interim_results
    stream = stt.stream()
    for frame in chunks(speech(1.0)):
        stream.push_audio(frame)
    stream.end_input()
    types = [e.type for e in await drain(stream)]
    assert types == [
        STTEventType.START_OF_SPEECH,
        STTEventType.FINAL_TRANSCRIPT,
        STTEventType.END_OF_SPEECH,
    ]
    await stream.aclose()


async def test_closing_a_stream_mid_utterance_restores_attention(mlx_fakes: MLXFakes) -> None:
    stt = create("stt", "mlx", incremental=True)
    stream = stt.stream()
    for frame in chunks(speech(0.7)):
        stream.push_audio(frame)
    async for ev in stream:
        if ev.type == STTEventType.INTERIM_TRANSCRIPT:
            break
    await stream.aclose()
    await _mlx.WORKER.run(lambda: None)  # the cleanup queued on the MLX thread has run
    assert mlx_fakes.attention[-1] == ("rel_pos",)


async def test_batch_mode_is_wrapped_by_the_stream_adapter(mlx_fakes: MLXFakes) -> None:
    from voice_agent_next.providers.energy import EnergyVAD

    stt = create("stt", "mlx", streaming=False)
    assert not stt.capabilities.streaming
    adapter = StreamAdapter(stt, EnergyVAD())
    stream = adapter.stream()
    for frame in chunks(speech(1.0)) + chunks(AudioFrame.silence(1.0, 16_000)):
        stream.push_audio(frame)
    stream.end_input()
    finals = [e.text for e in await drain(stream) if e.type == STTEventType.FINAL_TRANSCRIPT]
    assert finals and finals[0].startswith("Hello")
    await adapter.aclose()


# -------------------------------------------------------------------- whisper
async def test_whisper_transcribes(mlx_fakes: MLXFakes) -> None:
    stt = create("stt", "mlx_whisper/tiny", language="en-US", word_timestamps=True)
    assert stt.language == "en" and stt.capabilities.language_detection
    await stt.warmup()
    assert mlx_fakes.snapshots[0][0] == "mlx-community/whisper-tiny"
    assert mlx_fakes.whisper_loads == [(hf("mlx-community/whisper-tiny"), "float16")]
    result = await stt.transcribe(speech(1.0))
    assert result.text == "Hello there." and result.language == "en"
    assert [w.word for w in result.words or []] == ["Hello", "there."]
    assert result.confidence == pytest.approx(math.exp(-0.1))
    call = mlx_fakes.whisper_calls[-1]
    assert call["language"] == "en" and call["temperature"] == 0.0
    assert call["condition_on_previous_text"] is False and call["fp16"] is True
    assert isinstance(call["audio"], np.ndarray) and call["audio"].dtype == np.float32
    assert mlx_fakes.holder.model_path == hf("mlx-community/whisper-tiny")
    assert mlx_fakes.threads == {"mlx-test_0"}


async def test_whisper_models_do_not_evict_each_other(mlx_fakes: MLXFakes) -> None:
    tiny = create("stt", "mlx_whisper/tiny")
    base = create("stt", "mlx_whisper/whisper-base.en", fp16=False, transcribe_options={"x": 1})
    assert not base.capabilities.language_detection
    assert (await tiny.transcribe(speech(0.5))).language == "de"  # detected
    await base.transcribe(speech(0.5))
    await tiny.transcribe(speech(0.5))
    assert len(mlx_fakes.whisper_loads) == 2  # each instance loads its model once
    assert mlx_fakes.whisper_loads[1] == (hf("mlx-community/whisper-base.en-mlx"), "float32")
    assert mlx_fakes.whisper_calls[-2]["x"] == 1
    assert mlx_fakes.whisper_calls[-1]["path_or_hf_repo"] == hf("mlx-community/whisper-tiny")


def wseg(text: str, **stats: Any) -> dict[str, Any]:
    """An mlx-whisper segment (a dict, like openai-whisper's)."""
    fields = {"start": 0.0, "end": 1.0, "tokens": [1, 2, 3], "avg_logprob": -0.2,
              "no_speech_prob": 0.01, "compression_ratio": 1.4}  # fmt: skip
    return {"text": text, **fields, **stats}


def growing(audio: Any) -> list[dict[str, Any]]:
    """One word per 0.2 s of audio: interim decodes see the transcript grow."""
    words = " ".join(f"w{i}" for i in range(max(1, int(len(audio) / 16_000 / 0.2))))
    return [wseg(f" {words}", end=len(audio) / 16_000)]


def interim_calls(fakes: MLXFakes) -> list[dict[str, Any]]:
    return [c for c in fakes.whisper_calls if c.get("without_timestamps") and c["audio"].any()]


def final_calls(fakes: MLXFakes) -> list[dict[str, Any]]:
    return [c for c in fakes.whisper_calls if not c.get("without_timestamps") and c["audio"].any()]


async def test_whisper_hallucination_guard(mlx_fakes: MLXFakes) -> None:
    mlx_fakes.whisper_segments = [
        wseg(" Hallo, wie geht's?", avg_logprob=-0.3),
        wseg(" Untertitel im Auftrag des ZDF für funk, 2017"),
        wseg(" Das ist gut.", no_speech_prob=0.9, avg_logprob=-1.5),
    ]
    stt = MLXWhisperSTT(model="tiny")
    assert stt.guard == HallucinationGuard()
    result = await stt.transcribe(speech(1.0))
    assert result.text == "Hallo, wie geht's?"
    assert result.confidence == pytest.approx(math.exp(-0.3))  # from the kept segment
    mlx_fakes.whisper_segments = [wseg(" Субтитры сделал DimaTorzok")]
    assert (await stt.transcribe(speech(1.0))).text == ""
    mlx_fakes.whisper_segments = [wseg(" no no no no no no no no thanks")]
    assert (await stt.transcribe(speech(1.0))).text == "no thanks"  # loop collapsed
    off = MLXWhisperSTT(model="tiny", hallucination_guard=False)
    mlx_fakes.whisper_segments = [wseg(" Hallo."), wseg(" Thanks for watching!")]
    assert (await off.transcribe(speech(1.0))).text == "Hallo. Thanks for watching!"
    custom = MLXWhisperSTT(model="tiny", hallucination_guard={"artifacts": ["Hallo"]})
    assert (await custom.transcribe(speech(1.0))).text == "Thanks for watching!"
    with pytest.raises(ConfigurationError, match="mlx_whisper: hallucination_guard"):
        MLXWhisperSTT(hallucination_guard={"no_such_rule": 1})
    with pytest.raises(ConfigurationError, match="interim_interval"):
        MLXWhisperSTT(interim_interval=0)


async def test_whisper_guard_uses_the_vad_confidence(mlx_fakes: MLXFakes) -> None:
    mlx_fakes.whisper_segments = [wseg(" Danke.")]
    audio = AudioFrame.concat([speech(0.6), AudioFrame.silence(0.5, 16_000)])
    for probability, expected in ((0.95, ["Danke."]), (0.45, [])):
        adapter = StreamAdapter(MLXWhisperSTT(model="tiny"), LevelVAD(probability, activation=0.4))
        stream = adapter.stream()
        events = await stream_paced(stream, audio, speed=8.0)
        finals = [ev.text for _, ev in events if ev.type == STTEventType.FINAL_TRANSCRIPT]
        assert finals == expected, probability
        assert [ev.type for _, ev in events].count(STTEventType.END_OF_SPEECH) == 1
        await adapter.aclose()
    assert mlx_fakes.threads == {"mlx-test_0"}


async def test_whisper_interim_transcripts(mlx_fakes: MLXFakes) -> None:
    mlx_fakes.whisper_segments = growing
    stt = MLXWhisperSTT(model="tiny", language="en", interim_results=True, interim_interval=0.2)
    assert stt.capabilities.interim_results and not stt.capabilities.streaming
    assert stt.resolved_interim_interval == 0.2
    assert MLXWhisperSTT(model="tiny").resolved_interim_interval == 0.25
    adapter = StreamAdapter(stt, LevelVAD())
    assert adapter.capabilities.interim_results
    assert not StreamAdapter(MLXWhisperSTT(model="tiny"), LevelVAD()).capabilities.interim_results
    await adapter.warmup()
    stream = adapter.stream()
    silence = AudioFrame.silence(0.3, 16_000)
    events = await stream_paced(stream, AudioFrame.concat([silence, speech(1.6), silence]))
    kinds = [ev.type for _, ev in events]
    interims = [ev.text for _, ev in events if ev.type == STTEventType.INTERIM_TRANSCRIPT]
    finals = [ev.text for _, ev in events if ev.type == STTEventType.FINAL_TRANSCRIPT]
    assert kinds[0] == STTEventType.START_OF_SPEECH and kinds[-1] == STTEventType.END_OF_SPEECH
    assert len(finals) == 1 and 3 <= len(interims) <= 9
    assert finals[0].startswith(interims[-1])
    calls = interim_calls(mlx_fakes)
    assert all(c["temperature"] == 0.0 and not c["word_timestamps"] for c in calls)
    assert all(c["language"] == "en" for c in calls)
    (final,) = final_calls(mlx_fakes)
    assert "without_timestamps" not in final  # the final is decoded as in batch mode
    assert mlx_fakes.threads == {"mlx-test_0"}  # every decode on the MLX thread
    await adapter.aclose()


async def test_whisper_final_waits_for_an_interim_in_flight(mlx_fakes: MLXFakes) -> None:
    mlx_fakes.whisper_segments = growing
    mlx_fakes.whisper_delay = 0.25
    stt = MLXWhisperSTT(model="tiny", interim_results=True, interim_interval=0.1)
    assert not stt.parallel_final  # one MLX thread
    adapter = StreamAdapter(stt, LevelVAD())
    await adapter.warmup()
    stream = adapter.stream()
    events = await stream_paced(stream, speech(0.5))  # the input ends mid-speech
    kinds = [ev.type for _, ev in events]
    assert kinds[-2:] == [STTEventType.FINAL_TRANSCRIPT, STTEventType.END_OF_SPEECH]
    assert STTEventType.INTERIM_TRANSCRIPT not in kinds  # the stale interim was dropped
    assert len(stream.final_waits) == 1 and stream.final_waits[0] > 0  # type: ignore[attr-defined]
    await adapter.aclose()


async def test_whisper_final_from_interim(mlx_fakes: MLXFakes) -> None:
    mlx_fakes.whisper_segments = growing
    stt = MLXWhisperSTT(
        model="tiny", interim_results=True, interim_interval=0.02, final_from_interim=True
    )
    metrics: list[STTMetrics] = []
    stt.on("metrics", metrics.append)
    adapter = StreamAdapter(stt, LevelVAD())
    await adapter.warmup()
    stream = adapter.stream()
    events = await push_settled(
        stream, AudioFrame.concat([speech(0.8), AudioFrame.silence(0.1, 16_000)])
    )
    interims = [ev.text for ev in events if ev.type == STTEventType.INTERIM_TRANSCRIPT]
    finals = [ev.text for ev in events if ev.type == STTEventType.FINAL_TRANSCRIPT]
    assert finals == interims[-1:] and final_calls(mlx_fakes) == []
    assert stream.finals_from_interim == 1  # type: ignore[attr-defined]
    assert len(metrics) == 1
    await adapter.aclose()


async def test_streaming_steps_pace_themselves(mlx_fakes: MLXFakes) -> None:
    """A step slower than chunk_duration makes the next step wait for more audio."""
    stt = create("stt", "mlx", chunk_duration=0.32)
    stream = stt.stream()
    assert stream._threshold == 5120  # type: ignore[attr-defined]
    stream._step_time = 0.4  # type: ignore[attr-defined]
    assert stream._threshold == 12_800  # type: ignore[attr-defined]  # 2 x 0.4 s of audio
    await stream.aclose()


async def _next_interim(stream: Any, events: list[STTEvent]) -> str:
    async for ev in stream:
        events.append(ev)
        if ev.type == STTEventType.INTERIM_TRANSCRIPT:
            return ev.text
    raise AssertionError("stream ended")


async def test_streaming_redecodes_the_utterance(mlx_fakes: MLXFakes) -> None:
    stt = create("stt", "mlx", chunk_duration=0.32)
    metrics: list[STTMetrics] = []
    stt.on("metrics", metrics.append)
    stream = stt.stream()
    events: list[STTEvent] = []
    frames = chunks(speech(1.3))
    interims = []
    for i in range(4):  # 16 frames of 20 ms = one 0.32 s step each
        for frame in frames[16 * i : 16 * (i + 1)]:
            stream.push_audio(frame)
        interims.append(await _next_interim(stream, events))
    assert interims == ["Hello", "Hello world,", "Hello world, this", "Hello world, this is"]
    for frame in frames[64:]:
        stream.push_audio(frame)
    stream.flush()
    for frame in chunks(speech(0.7)):
        stream.push_audio(frame)
    stream.end_input()
    events += await drain(stream)
    await stream.aclose()

    finals = [e for e in events if e.type == STTEventType.FINAL_TRANSCRIPT]
    assert [f.text for f in finals] == ["Hello world, this is", "Hello world,"]
    # every step decodes the whole utterance so far; the final one includes the tail
    assert mlx_fakes.generated[:6] == [8000, 5120, 10240, 15360, 20480, 20800]
    assert mlx_fakes.generated[-1] == 11200
    assert finals[1].transcript is not None and finals[1].transcript.words
    assert finals[1].transcript.words[0].start == pytest.approx(1.3)
    # full attention: no StreamingParakeet, no attention switch
    assert mlx_fakes.streamers == [] and mlx_fakes.attention == []
    assert len([m for m in metrics if m.streamed]) >= 1
    assert mlx_fakes.threads == {"mlx-test_0"}


async def test_silence_does_not_trigger_steps(mlx_fakes: MLXFakes) -> None:
    stt = create("stt", "mlx", chunk_duration=0.32)
    stream = stt.stream()
    events: list[STTEvent] = []
    frames = chunks(speech(0.64))
    for frame in frames[:16]:
        stream.push_audio(frame)
    await _next_interim(stream, events)
    for frame in frames[16:]:
        stream.push_audio(frame)
    await _next_interim(stream, events)
    for frame in chunks(AudioFrame.silence(0.96, 16_000)):  # the user stopped talking
        stream.push_audio(frame)
    await asyncio.sleep(0.05)
    stream.flush()
    async for ev in stream:
        if ev.type == STTEventType.FINAL_TRANSCRIPT:
            assert ev.text == "Hello world, this is a"  # 1.6 s decoded
            break
    assert mlx_fakes.generated == [8000, 5120, 10240, 25600]  # no step on the silence

    # an utterance of only silence (a VAD false alarm) is not decoded at all
    for frame in chunks(AudioFrame.silence(1.0, 16_000)):
        stream.push_audio(frame)
    stream.flush()
    async for ev in stream:
        if ev.type == STTEventType.FINAL_TRANSCRIPT:
            assert ev.text == ""
            break
    assert len(mlx_fakes.generated) == 4
    await stream.aclose()
