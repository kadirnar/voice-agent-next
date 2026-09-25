"""Cascade turn-taking under component failures and races (#136): failing turn detectors,
late final transcripts, an STT stream that ends, speech resuming mid-commit, and a
session closed in the middle of a connection rotation."""

from __future__ import annotations

import asyncio
import dataclasses
import logging
from collections.abc import AsyncIterator, Callable
from typing import Any, TypeVar

import pytest

from voice_agent_next import AudioFrame, CascadeOptions
from voice_agent_next.chat import ChatContext
from voice_agent_next.engine import EngineConnection, EngineOptions
from voice_agent_next.engines.cascade import CascadeConnection, CascadeEngine
from voice_agent_next.engines.rotation import RotatingConnection, RotatingEngine, RotationPolicy
from voice_agent_next.errors import AuthenticationError, ProviderConnectionError
from voice_agent_next.events import (
    EngineErrorEvent,
    EngineStatus,
    InputCommitted,
    InputSpeechStarted,
    InputTranscript,
    ResponseDone,
    ResponseStarted,
)
from voice_agent_next.metrics import EndpointingMetrics
from voice_agent_next.providers.energy import EnergyVAD
from voice_agent_next.providers.mock import (
    MockEngine,
    MockEngineConnection,
    MockLLM,
    MockSTT,
    MockTTS,
    MockTurnDetector,
    _MockSTTStream,
    synth_speech,
)
from voice_agent_next.stt import STT, STTCapabilities, STTStream
from voice_agent_next.turn import FusedTurnDetector
from voice_agent_next.utils import now

T = TypeVar("T")
SR = 16_000


async def wait_for(predicate: Callable[[], Any], timeout: float = 5.0) -> None:
    async def poll() -> None:
        while not predicate():
            await asyncio.sleep(0.005)

    await asyncio.wait_for(poll(), timeout)


class Collector:
    def __init__(self, conn: EngineConnection) -> None:
        self.events: list[Any] = []
        self.task = asyncio.create_task(self._run(conn))

    async def _run(self, conn: EngineConnection) -> None:
        async for ev in conn.events():
            self.events.append(ev)

    def of(self, cls: type[T]) -> list[T]:
        return [e for e in self.events if isinstance(e, cls)]

    def finals(self) -> list[str]:
        return [e.text for e in self.of(InputTranscript) if e.is_final]


async def feed(conn: EngineConnection, frame: AudioFrame, chunk: float = 0.02) -> None:
    """Send ``frame`` in 20 ms chunks, as fast as possible."""
    step = round(chunk * frame.sample_rate) * 2
    for i in range(0, len(frame.data), step):
        await conn.send_audio(AudioFrame(frame.data[i : i + step], frame.sample_rate, 1, now()))
    await asyncio.sleep(0)


async def utterance(conn: EngineConnection, speech: float = 0.6, silence: float = 0.4) -> None:
    await feed(conn, synth_speech(speech, SR))
    await feed(conn, AudioFrame.silence(silence, SR))


async def open_cascade(
    *,
    stt: Any = None,
    turn_detector: Any = None,
    responses: Any = None,
    **options: Any,
) -> tuple[CascadeEngine, CascadeConnection, Collector, list[EndpointingMetrics]]:
    options.setdefault("min_endpointing_delay", 0.0)
    engine = CascadeEngine(
        stt=stt if stt is not None else MockSTT(),
        llm=MockLLM(responses=responses or (lambda ctx: "Okay.")),
        tts=MockTTS(chars_per_second=200.0),
        vad=EnergyVAD(),
        turn_detector=turn_detector,
        options=CascadeOptions(**options),
    )
    metrics: list[EndpointingMetrics] = []
    engine.on("metrics", lambda m: metrics.append(m) if isinstance(m, EndpointingMetrics) else 0)
    conn = await engine.connect(EngineOptions())
    assert isinstance(conn, CascadeConnection)
    return engine, conn, Collector(conn), metrics


@pytest.fixture
async def closing() -> AsyncIterator[list[EngineConnection]]:
    conns: list[EngineConnection] = []
    yield conns
    for conn in conns:
        await conn.aclose()


# ------------------------------------------------------------ failing turn detectors
class BrokenDetector(MockTurnDetector):
    def __init__(self, modality: str = "text") -> None:
        super().__init__(threshold=0.5)
        self.modality = modality  # type: ignore[misc]

    async def _predict(self, *, audio: AudioFrame | None, chat_ctx: ChatContext | None) -> float:
        self.calls += 1
        raise RuntimeError("detector exploded")


class FixedAudioDetector(MockTurnDetector):
    modality = "audio"


@pytest.mark.parametrize("kind", ["text", "audio", "fused-audio", "fused-both"])
async def test_a_failing_turn_detector_still_commits_the_turn(
    kind: str, closing: list[EngineConnection], caplog: pytest.LogCaptureFixture
) -> None:
    detector: Any
    if kind == "text":
        detector = BrokenDetector("text")
    elif kind == "audio":
        detector = BrokenDetector("audio")
    elif kind == "fused-audio":  # the audio half fails, the text half still decides
        detector = FusedTurnDetector(audio=BrokenDetector("audio"), text=MockTurnDetector())
    else:
        detector = FusedTurnDetector(audio=BrokenDetector("audio"), text=BrokenDetector())
    _, conn, rec, metrics = await open_cascade(
        stt=MockSTT(transcripts=["book a table"]),
        turn_detector=detector,
        max_endpointing_delay=0.3,
        false_commit_window=0.05,
    )
    closing.append(conn)
    with caplog.at_level(logging.WARNING, logger="voice_agent_next"):
        await utterance(conn)
        await wait_for(lambda: rec.of(ResponseDone), 5)
    assert len(rec.of(InputCommitted)) == 1
    assert rec.finals() == ["book a table"]
    assert rec.of(ResponseDone)[0].status == "completed"
    assert "failed; endpointing without it" in caplog.text
    await wait_for(lambda: metrics, 3)
    [m] = metrics
    assert m.committed and m.detector_error is not None and "detector exploded" in m.detector_error
    if kind == "fused-audio":
        assert m.audio_probability is None and m.text_probability is not None
        assert m.probability is not None
    else:  # no verdict at all: endpointed as without a detector
        assert m.probability is None


class SlowAudioDetector(MockTurnDetector):
    """An audio detector whose cancellation takes a while to finish (a model call)."""

    modality = "audio"

    def __init__(self) -> None:
        super().__init__(threshold=0.5)
        self.cleaned_up = False

    async def _predict(self, *, audio: AudioFrame | None, chat_ctx: ChatContext | None) -> float:
        self.calls += 1
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            await asyncio.sleep(0.05)  # releasing the model
            self.cleaned_up = True
            raise
        return 0.9


@pytest.mark.parametrize("fused", [False, True])
async def test_cancelled_detector_calls_are_awaited(
    fused: bool, closing: list[EngineConnection]
) -> None:
    audio = SlowAudioDetector()
    detector = FusedTurnDetector(audio=audio, text=MockTurnDetector()) if fused else audio
    _, conn, _, _ = await open_cascade(stt=MockSTT(latency=0.5), turn_detector=detector)
    closing.append(conn)
    await utterance(conn)
    await wait_for(lambda: audio.calls == 1)
    task = conn._endpoint_task
    assert task is not None
    await feed(conn, synth_speech(0.3, SR))  # the user resumes: the endpointing is cancelled
    await asyncio.gather(task, return_exceptions=True)
    assert task.cancelled()
    assert audio.cleaned_up  # the detector call finished before the endpointing did


# ------------------------------------------------------------- late final transcripts
async def test_a_late_final_transcript_does_not_leak_into_the_next_turn(
    closing: list[EngineConnection],
) -> None:
    # the STT answers each flush after 0.4 s, the cascade waits for 0.1 s only: each turn
    # is committed with its interim text, and the late finals arrive after the commit
    stt = MockSTT(transcripts=["one two", "three four"], latency=0.4)
    _, conn, rec, _ = await open_cascade(stt=stt, final_transcript_timeout=0.1)
    closing.append(conn)
    await utterance(conn, speech=1.2)
    await wait_for(lambda: rec.of(InputCommitted), 5)
    await asyncio.sleep(0.6)  # the first turn's final arrives in the second turn
    await utterance(conn, speech=1.2)
    await wait_for(lambda: len(rec.of(InputCommitted)) == 2, 5)
    assert rec.finals() == ["one two", "three four"]


async def test_a_late_final_of_the_pending_turn_replaces_its_interim_text(
    closing: list[EngineConnection],
) -> None:
    # interim "one" (after 0.5 s of speech), final "one two three" 0.3 s after the flush:
    # still before the commit (0.6 s), so the turn is committed with the final text
    stt = MockSTT(transcripts=["one two three"], latency=0.3)
    _, conn, rec, _ = await open_cascade(
        stt=stt, final_transcript_timeout=0.1, min_endpointing_delay=0.6
    )
    closing.append(conn)
    await utterance(conn, speech=0.6)
    await wait_for(lambda: rec.of(InputCommitted), 5)
    assert rec.finals() == ["one two three"]


# ---------------------------------------------------------------- STT stream that ends
class EndingSTT(STT):
    """A streaming STT whose provider closes the stream after ``frames`` audio frames."""

    provider = "ending"

    def __init__(self, frames: int = 5) -> None:
        super().__init__(
            model="ending", capabilities=STTCapabilities(streaming=True), sample_rate=SR
        )
        self.frames = frames
        self.streams: list[STTStream] = []

    def _create_stream(self, *, language: str | None) -> STTStream:
        stream = _EndingStream(self, language=language)
        self.streams.append(stream)
        return stream


class _EndingStream(STTStream):
    async def _run(self) -> None:
        stt: EndingSTT = self._stt  # type: ignore[assignment]
        n = 0
        async for _ in self._input:
            n += 1
            if n >= stt.frames:
                return  # e.g. the server closed the WebSocket normally


async def test_an_ended_stt_stream_is_a_fatal_error(closing: list[EngineConnection]) -> None:
    stt = EndingSTT(frames=5)
    _, conn, rec, _ = await open_cascade(stt=stt)
    closing.append(conn)
    await feed(conn, AudioFrame.silence(0.2, SR))
    await wait_for(lambda: rec.of(EngineErrorEvent))
    [err] = rec.of(EngineErrorEvent)
    assert not err.recoverable and "STT stream ended" in str(err.error)
    [stream] = stt.streams
    assert stream._input.closed  # nothing buffers the audio sent from now on
    queued = stream._input.qsize()
    await feed(conn, AudioFrame.silence(1.0, SR))  # no error, no unbounded queue
    assert stream._input.qsize() == queued
    # the turn still ends (on the VAD), without waiting for a final that cannot come
    await utterance(conn)
    await wait_for(lambda: rec.of(InputSpeechStarted))
    await conn.aclose()
    assert len(rec.of(EngineErrorEvent)) == 1  # closing is not another failure


# ------------------------------------------------------ reopening an ended STT stream
class _ReopenStream(_MockSTTStream):
    """A mock stream that ends by itself after ``end_after`` frames (``None``: never),
    or fails with ``fail`` at its first frame."""

    end_after: int | None = None
    fail: Exception | None = None
    received = 0.0
    """Seconds of audio pushed to this stream."""

    def push_audio(self, frame: AudioFrame) -> None:
        self.received += frame.duration
        super().push_audio(frame)

    async def _run(self) -> None:
        if self.end_after is None and self.fail is None:
            await super()._run()
            return
        n = 0
        async for _ in self._input:
            n += 1
            if self.fail is not None:
                raise self.fail
            if self.end_after is not None and n >= self.end_after:
                return  # the provider closed the stream


class ReopenableSTT(MockSTT):
    """A mock STT that declares ``reconnect``; stream ``i`` ends after ``ends[i]`` frames
    (``None`` or past the list: a healthy stream)."""

    provider = "reopen"

    def __init__(
        self,
        ends: list[int | None],
        *,
        fail: Exception | None = None,
        reconnect: bool = True,
        **kw: Any,
    ) -> None:
        super().__init__(**kw)
        self.capabilities = dataclasses.replace(self.capabilities, reconnect=reconnect)
        self.ends = ends
        self.fail = fail
        self.streams: list[_ReopenStream] = []

    def _create_stream(self, *, language: str | None) -> STTStream:
        stream = _ReopenStream(self, language=language)
        i = len(self.streams)
        stream.end_after = self.ends[i] if i < len(self.ends) else None
        stream.fail = self.fail
        self.streams.append(stream)
        return stream


async def test_an_ended_stt_stream_is_reopened(closing: list[EngineConnection]) -> None:
    """An STT that can reopen gets a new stream when its stream ends (#157): no fatal
    error, the audio of the gap is replayed, and the next turn is transcribed."""
    stt = ReopenableSTT([5], transcripts=["after the reconnect"])
    _, conn, rec, _ = await open_cascade(stt=stt, stt_reconnect_backoff=0.2)
    closing.append(conn)
    await feed(conn, AudioFrame.silence(0.2, SR))  # the first stream ends after 0.1 s
    await wait_for(lambda: any(isinstance(e, EngineStatus) for e in rec.events))
    await feed(conn, AudioFrame.silence(0.1, SR))  # sent during the backoff: buffered
    # the collector sees the events one loop step after the reopen
    await wait_for(lambda: len(rec.of(EngineStatus)) == 2)
    statuses = [e.status for e in rec.of(EngineStatus)]
    assert statuses == ["reconnecting", "reconnected"]
    first, second = stt.streams
    assert first._input.closed
    # the new stream got what the old one missed (the gap), not less, not duplicated
    assert 0.1 - 1e-6 <= second.received <= 0.2 + 1e-6
    await utterance(conn)
    await wait_for(lambda: rec.of(InputCommitted), 5)
    assert rec.finals() == ["after the reconnect"]
    assert not rec.of(EngineErrorEvent)


async def test_audio_buffered_during_a_reconnect_is_bounded(
    closing: list[EngineConnection],
) -> None:
    stt = ReopenableSTT([1])
    _, conn, rec, _ = await open_cascade(
        stt=stt, stt_reconnect_backoff=0.5, stt_reconnect_buffer=0.3
    )
    closing.append(conn)
    await feed(conn, AudioFrame.silence(0.02, SR))
    await wait_for(lambda: rec.of(EngineStatus))
    await feed(conn, AudioFrame.silence(2.0, SR))  # far more than the buffer holds
    await wait_for(lambda: len(stt.streams) == 2)
    assert stt.streams[1].received == pytest.approx(0.3, abs=0.021)  # the newest 0.3 s


async def test_stt_reconnect_gives_up_after_its_attempts(
    closing: list[EngineConnection],
) -> None:
    stt = ReopenableSTT([1, 1, 1, 1, 1])  # every stream dies at once
    _, conn, rec, _ = await open_cascade(
        stt=stt, stt_reconnect_attempts=2, stt_reconnect_backoff=0.01
    )
    closing.append(conn)
    for _ in range(40):  # keep audio flowing, as a microphone does
        await feed(conn, AudioFrame.silence(0.02, SR))
        await asyncio.sleep(0.01)
        if rec.of(EngineErrorEvent):
            break
    await wait_for(lambda: rec.of(EngineErrorEvent))
    [err] = rec.of(EngineErrorEvent)
    assert not err.recoverable and "STT stream ended" in str(err.error)
    assert len(stt.streams) == 3  # the first one and two reopened
    await feed(conn, AudioFrame.silence(0.2, SR))  # deaf now, but no error
    assert len(rec.of(EngineErrorEvent)) == 1 and len(stt.streams) == 3


async def test_non_retryable_stt_errors_are_not_retried(
    closing: list[EngineConnection],
) -> None:
    stt = ReopenableSTT([], fail=AuthenticationError("bad key", provider="reopen"))
    _, conn, rec, _ = await open_cascade(stt=stt, stt_reconnect_backoff=0.01)
    closing.append(conn)
    await feed(conn, AudioFrame.silence(0.1, SR))
    await wait_for(lambda: rec.of(EngineErrorEvent))
    await asyncio.sleep(0.05)
    [err] = rec.of(EngineErrorEvent)
    assert isinstance(err.error, AuthenticationError) and not err.recoverable
    assert len(stt.streams) == 1 and not rec.of(EngineStatus)


async def test_retryable_stt_errors_reopen_the_stream(closing: list[EngineConnection]) -> None:
    stt = ReopenableSTT([], fail=ProviderConnectionError("reset", provider="reopen"))
    _, conn, rec, _ = await open_cascade(stt=stt, stt_reconnect_backoff=0.01)
    closing.append(conn)
    await feed(conn, AudioFrame.silence(0.02, SR))
    await wait_for(lambda: len(stt.streams) >= 2)
    await wait_for(lambda: rec.of(EngineStatus))
    assert rec.of(EngineStatus)[0].status == "reconnecting"


async def test_stt_reconnect_can_be_disabled(closing: list[EngineConnection]) -> None:
    stt = ReopenableSTT([5])
    _, conn, rec, _ = await open_cascade(stt=stt, stt_reconnect=False)
    closing.append(conn)
    await feed(conn, AudioFrame.silence(0.2, SR))
    await wait_for(lambda: rec.of(EngineErrorEvent))
    assert len(stt.streams) == 1 and not rec.of(EngineStatus)


@pytest.mark.parametrize(
    "spec", ["deepgram", "assemblyai", "soniox", "speechmatics", "cartesia", "elevenlabs"]
)
def test_websocket_stts_declare_that_they_can_reopen(spec: str) -> None:
    from voice_agent_next import create

    stt = create("stt", spec, api_key="test-key")
    assert stt.capabilities.streaming and stt.capabilities.reconnect
    assert not MockSTT().capabilities.reconnect  # the default: an ended stream is fatal


async def test_closing_the_cascade_is_not_an_stt_failure() -> None:
    _, conn, rec, _ = await open_cascade()
    await utterance(conn)
    await wait_for(lambda: rec.of(ResponseDone))
    await conn.aclose()
    await asyncio.sleep(0.05)
    assert not rec.of(EngineErrorEvent)


# ----------------------------------------------------------- speech resuming mid-commit
async def test_speech_resuming_during_the_commit_still_gets_a_reply(
    closing: list[EngineConnection],
) -> None:
    _, conn, rec, metrics = await open_cascade(false_commit_window=0.5)
    closing.append(conn)
    # a previous reply whose cancellation takes a while: the commit waits for it
    cancelled = asyncio.Event()

    async def slow_to_cancel() -> None:
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.set()
            await asyncio.sleep(0.3)
            raise

    conn._response_task = asyncio.create_task(slow_to_cancel())
    await asyncio.sleep(0)
    await utterance(conn)
    await asyncio.wait_for(cancelled.wait(), 5)  # the turn is being committed
    assert len(rec.of(InputCommitted)) == 1
    await feed(conn, synth_speech(0.3, SR))  # ... when the user speaks again
    await wait_for(lambda: len(rec.of(InputSpeechStarted)) == 2)
    await wait_for(lambda: rec.of(ResponseStarted), 3)
    item = rec.of(InputCommitted)[0].item_id
    assert conn.chat_ctx.get(item) is not None
    await wait_for(lambda: metrics, 3)
    assert metrics[0].committed and metrics[0].false_commit  # a false commit, not a pause


# ------------------------------------------------------------ rotation during a close
class GatedMockEngine(MockEngine):
    """Mock engine whose later connections block on audio until ``gate`` is set."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.gate = asyncio.Event()
        self.blocked = asyncio.Event()
        self.conns: list[MockEngineConnection] = []

    async def connect(self, options: EngineOptions) -> EngineConnection:
        conn = await super().connect(options)
        assert isinstance(conn, MockEngineConnection)
        if self.conns:
            send = conn._send_audio

            async def gated(frame: AudioFrame) -> None:
                self.blocked.set()
                await self.gate.wait()
                await send(frame)

            conn._send_audio = gated  # type: ignore[method-assign]
        self.conns.append(conn)
        return conn


async def test_closing_mid_rotation_closes_both_connections() -> None:
    inner = GatedMockEngine()
    engine = RotatingEngine(inner, policy=RotationPolicy(quiet_period=0.05, replay=1.0))
    conn = await engine.connect(EngineOptions())
    assert isinstance(conn, RotatingConnection)
    rec = Collector(conn)
    await feed(conn, AudioFrame.silence(0.3, SR))  # recent audio: replayed on the switch
    conn.rotate("test", deadline=now())
    await asyncio.wait_for(inner.blocked.wait(), 5)  # the switch is delivering audio
    assert rec.of(EngineStatus) and rec.of(EngineStatus)[-1].status == "reconnecting"
    await conn.aclose()  # the session closes mid-switch
    old, new = inner.conns
    assert new.closed and old.closed
