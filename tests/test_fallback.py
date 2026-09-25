"""Provider failover chains: FallbackLLM / FallbackTTS / FallbackSTT."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from tests.test_session import Recorder, mock_cascade, speak, wait_for
from voice_agent_next import Agent, AgentState, AudioFrame, ChatContext, create
from voice_agent_next.config import load_config
from voice_agent_next.errors import (
    AuthenticationError,
    ConfigurationError,
    ProviderConnectionError,
    ProviderError,
    ProviderTimeoutError,
    RateLimitError,
)
from voice_agent_next.fallback import (
    FallbackLLM,
    FallbackSTT,
    FallbackTTS,
    ProviderAvailabilityChanged,
    ProviderFailover,
    is_failover_error,
)
from voice_agent_next.metrics import LLMMetrics, TTSMetrics
from voice_agent_next.providers.mock import MockLLM, MockSTT, MockTTS, synth_speech
from voice_agent_next.stt import STTEvent, STTEventType, STTStream, Transcript
from voice_agent_next.transports import LoopbackTransport
from voice_agent_next.tts import ChunkedStream, SynthesizeStream

# ------------------------------------------------------------------ failing mocks


class FlakyLLM(MockLLM):
    """A MockLLM that fails on command.

    ``mode``: ``None`` (works), ``"connect"`` (error before any token), ``"stall"``
    (never answers), ``"midstream"`` (fails after ``fail_after`` tokens).
    """

    def __init__(self, mode: str | None = None, *, fail_after: int = 1, **kw: Any) -> None:
        super().__init__(**kw)
        self.mode = mode
        self.fail_after = fail_after
        self.calls = 0

    def _chat(self, ctx: ChatContext, **kw: Any) -> Any:
        self.calls += 1
        return _FlakyLLMStream(self, ctx, **kw)


class _FlakyLLMStream:  # built through the real stream class below
    def __new__(cls, llm: FlakyLLM, ctx: ChatContext, **kw: Any) -> Any:
        from voice_agent_next.providers.mock import _MockLLMStream

        class S(_MockLLMStream):
            def _push(self, chunk: Any) -> None:
                if llm.mode == "midstream" and chunk.delta:
                    self._n = getattr(self, "_n", 0) + 1
                    if self._n > llm.fail_after:
                        raise ProviderConnectionError("connection reset", provider="flaky")
                super()._push(chunk)

            async def _run(self) -> None:
                if llm.mode == "connect":
                    raise ProviderConnectionError("connection refused", provider="flaky")
                if llm.mode == "stall":
                    await asyncio.sleep(3600)
                await super()._run()

        return S(llm, ctx, **kw)


class FlakyTTS(MockTTS):
    """``mode``: ``None``, ``"connect"``, ``"stall"``, ``"midstream"``, ``"ends"`` (streaming
    only: the stream stops silently after the first segment)."""

    def __init__(self, mode: str | None = None, **kw: Any) -> None:
        super().__init__(**kw)
        self.mode = mode

    def _synthesize(self, text: str, *, voice: str | None) -> ChunkedStream:
        tts = self

        class S(ChunkedStream):
            async def _run(self) -> None:
                if tts.mode == "connect":
                    raise ProviderConnectionError("refused", provider="flaky")
                if tts.mode == "stall":
                    await asyncio.sleep(3600)
                n = 0

                def push(frame: AudioFrame) -> None:
                    nonlocal n
                    n += 1
                    if tts.mode == "midstream" and n > 2:
                        raise ProviderConnectionError("reset", provider="flaky")
                    self._push_audio(frame)

                await tts.generate(self.text, push)

        return S(self, text, voice=voice)

    def _create_stream(self, *, voice: str | None) -> SynthesizeStream:
        tts = self

        class S(SynthesizeStream):
            async def _run(self) -> None:
                buf: list[str] = []
                segments = 0
                async for item in self._input:
                    if not self.is_flush(item):
                        assert isinstance(item, str)
                        buf.append(item)
                        continue
                    if tts.mode == "connect":
                        raise ProviderConnectionError("refused", provider="flaky")
                    if tts.mode == "stall":
                        await asyncio.sleep(3600)
                    if tts.mode == "ends" and segments == 1:
                        return  # silent disconnect: no error, no audio
                    text = "".join(buf).strip()
                    buf = []
                    self._segment_text = text or None
                    await tts.generate(text, self._push_audio)
                    if tts.mode == "midstream":
                        raise ProviderConnectionError("reset", provider="flaky")
                    self._end_segment()
                    segments += 1

        return S(self, voice=voice)


class FlakySTT(MockSTT):
    """A streaming MockSTT whose stream dies after ``fail_after_audio`` seconds of input."""

    def __init__(self, fail_after_audio: float | None = None, **kw: Any) -> None:
        super().__init__(**kw)
        self.fail_after_audio = fail_after_audio
        self.received: list[float] = []  # audio seconds each stream received

    def _create_stream(self, *, language: str | None) -> STTStream:
        from voice_agent_next.providers.mock import _MockSTTStream

        stt = self
        idx = len(self.received)
        self.received.append(0.0)

        class S(_MockSTTStream):
            def __init__(self, *a: Any, **k: Any) -> None:
                super().__init__(*a, **k)
                inp = self._input
                orig = inp.recv

                async def recv() -> Any:
                    item = await orig()
                    if isinstance(item, AudioFrame):
                        stt.received[idx] += item.duration
                        if (
                            stt.fail_after_audio is not None
                            and stt.received[idx] > stt.fail_after_audio
                        ):
                            raise ProviderConnectionError("socket closed", provider="flaky")
                    return item

                inp.recv = recv  # type: ignore[method-assign]

        return S(self, language=language)


def ctx(text: str = "hi") -> ChatContext:
    c = ChatContext()
    c.add_message("user", text)
    return c


# ------------------------------------------------------------------ policy


def test_failover_error_policy() -> None:
    assert is_failover_error(ProviderConnectionError("x"))
    assert is_failover_error(ProviderTimeoutError("x"))
    assert is_failover_error(RateLimitError("x"))
    assert is_failover_error(AuthenticationError("x", status_code=401))
    assert is_failover_error(ProviderError("x", status_code=503))
    assert is_failover_error(TimeoutError())
    assert is_failover_error(ConnectionResetError())
    assert not is_failover_error(ProviderError("bad request", status_code=400))
    assert not is_failover_error(ConfigurationError("x"))


# ------------------------------------------------------------------ LLM


@pytest.mark.parametrize("mode", ["connect", "stall"])
async def test_llm_fails_over_before_the_first_token(mode: str) -> None:
    primary = FlakyLLM(mode, model="a", responses=["from primary"])
    backup = MockLLM(model="b", responses=["from backup"])
    llm = FallbackLLM([primary, backup], first_token_timeout=0.3)
    failovers: list[ProviderFailover] = []
    metrics: list[LLMMetrics] = []
    llm.on("provider_failover", failovers.append)
    llm.on("metrics", metrics.append)

    stream = llm.chat(ctx())
    result = await asyncio.wait_for(stream.collect(), 5)

    assert result.text == "from backup"
    assert stream.served_by == "mock/b"  # type: ignore[attr-defined]
    assert len(failovers) == 1
    ev = failovers[0]
    assert (ev.kind, ev.from_provider, ev.to_provider) == ("llm", "mock/a", "mock/b")
    assert ev.reason == ("connection" if mode == "connect" else "timeout")
    assert ev.request_id == stream.request_id
    stats = llm.stats
    assert stats.served == {"mock/b": 1} and stats.failures == {"mock/a": 1}
    assert stats.failovers == 1 and sum(stats.reasons.values()) == 1
    # every attempt reports its own metrics, attributed to the provider that made it
    await wait_for(lambda: len(metrics) == 2)
    by_model = {m.model: m for m in metrics}
    assert by_model["b"].error is None and by_model["b"].ttft is not None
    assert by_model["a"].error is not None or by_model["a"].cancelled
    await llm.aclose()


async def test_llm_does_not_switch_after_tokens_were_streamed() -> None:
    primary = FlakyLLM("midstream", fail_after=2, model="a", responses=["one two three four"])
    backup = MockLLM(model="b", responses=["backup answer"])
    llm = FallbackLLM([primary, backup])
    failovers: list[ProviderFailover] = []
    llm.on("provider_failover", failovers.append)

    stream = llm.chat(ctx())
    got: list[str] = []

    async def consume() -> None:
        async for chunk in stream:
            got.append(chunk.delta)

    with pytest.raises(ProviderConnectionError):
        await consume()
    assert "".join(got) == "one two "  # the heard prefix, never continued by another model
    assert failovers == [] and backup.requests == []
    # ...but the next request starts on the healthy provider
    assert not llm.health[0].available
    assert (await llm.chat(ctx()).collect()).text == "backup answer"
    await llm.aclose()


async def test_llm_non_failover_errors_propagate() -> None:
    class BadRequest(MockLLM):
        def _chat(self, ctx: ChatContext, **kw: Any) -> Any:
            raise ProviderError("invalid request", status_code=400)

    backup = MockLLM(model="b")
    llm = FallbackLLM([BadRequest(model="a"), backup])
    with pytest.raises(ProviderError, match="invalid request"):
        await llm.chat(ctx()).collect()
    assert backup.requests == [] and llm.health[0].available
    await llm.aclose()


async def test_llm_all_providers_failing_raises_the_last_error() -> None:
    llm = FallbackLLM([FlakyLLM("connect", model="a"), FlakyLLM("connect", model="b")])
    with pytest.raises(ProviderConnectionError):
        await llm.chat(ctx()).collect()
    assert llm.stats.failovers == 1
    await llm.aclose()


async def test_cooldown_skips_a_failed_provider_then_retries_it() -> None:
    primary = FlakyLLM("connect", model="a", responses=["primary"] * 5)
    backup = MockLLM(model="b", responses=["backup"] * 5)
    llm = FallbackLLM([primary, backup], cooldown=1.0)
    changes: list[ProviderAvailabilityChanged] = []
    llm.on("provider_availability_changed", changes.append)

    assert (await llm.chat(ctx()).collect()).text == "backup"
    assert primary.calls == 1
    assert (await llm.chat(ctx()).collect()).text == "backup"
    assert primary.calls == 1  # skipped during the cooldown

    primary.mode = None  # the provider recovers
    await asyncio.sleep(1.1)
    assert (await llm.chat(ctx()).collect()).text == "primary"
    assert primary.calls == 2
    assert [(c.provider, c.available) for c in changes] == [("mock/a", False), ("mock/a", True)]
    assert llm.health[0].available
    await llm.aclose()


async def test_background_probe_restores_a_provider_early() -> None:
    probes = 0

    class Probed(FlakyLLM):
        async def warmup(self) -> None:
            nonlocal probes
            probes += 1
            if self.mode is not None:
                raise ProviderConnectionError("still down")

    primary = Probed("connect", model="a")
    llm = FallbackLLM([primary, MockLLM(model="b")], cooldown=3600, probe_interval=0.05)
    await llm.chat(ctx()).collect()
    assert not llm.health[0].available
    await asyncio.sleep(0.2)
    assert probes >= 1 and not llm.health[0].available  # failed probes keep it down
    primary.mode = None
    await wait_for(lambda: llm.health[0].available, 3)
    await llm.chat(ctx()).collect()
    assert llm.stats.served["mock/a"] == 1
    await llm.aclose()


async def test_warmup_and_aclose_forward_to_all_providers() -> None:
    seen: list[str] = []

    class Tracked(MockLLM):
        async def warmup(self) -> None:
            seen.append(f"warm {self.model}")
            if self.model == "a":
                raise ProviderConnectionError("down")

        async def aclose(self) -> None:
            seen.append(f"close {self.model}")

    llm = FallbackLLM([Tracked(model="a"), Tracked(model="b")])
    await llm.warmup()  # one failure only marks that provider down
    assert not llm.health[0].available and llm.health[1].available
    await llm.aclose()
    assert sorted(seen) == ["close a", "close b", "warm a", "warm b"]

    down = FallbackLLM([Tracked(model="a")])
    with pytest.raises(ProviderConnectionError):
        await down.warmup()


def test_capabilities_are_the_intersection() -> None:
    from voice_agent_next import LLMCapabilities

    a, b = MockLLM(model="a"), MockLLM(model="b")
    b.capabilities = LLMCapabilities(tool_calling=False)
    assert FallbackLLM([a, b]).capabilities.tool_calling is False
    t = FallbackTTS([MockTTS(streaming=True), MockTTS(streaming=False, sample_rate=16_000)])
    assert t.capabilities.streaming is False and t.sample_rate == 24_000


# ------------------------------------------------------------------ registry / config


def test_create_from_a_list_of_specs() -> None:
    llm = create("llm", ["mock/a", {"provider": "mock", "model": "b"}])
    assert isinstance(llm, FallbackLLM)
    assert [p.model for p in llm.providers] == ["a", "b"]
    tts = create("tts", {"fallback": ["mock/x", "mock/y"], "first_audio_timeout": 1.5})
    assert isinstance(tts, FallbackTTS) and tts.first_audio_timeout == 1.5
    assert isinstance(create("stt", ["mock"]), FallbackSTT)
    with pytest.raises(ConfigurationError):
        create("vad", ["energy", "energy"])
    with pytest.raises(ConfigurationError):
        create("llm", [])

    cfg = load_config({"llm": ["mock/a", "mock/b"], "tts": "mock", "stt": ["mock"]})
    assert cfg.llm == ["mock/a", "mock/b"]


# ------------------------------------------------------------------ TTS


@pytest.mark.parametrize("mode", ["connect", "stall"])
async def test_tts_chunked_fails_over_before_first_audio_and_resamples(mode: str) -> None:
    primary = FlakyTTS(mode, model="a", sample_rate=24_000)
    backup = MockTTS(model="b", sample_rate=16_000)
    tts = FallbackTTS([primary, backup], first_audio_timeout=0.3)
    failovers: list[ProviderFailover] = []
    metrics: list[TTSMetrics] = []
    tts.on("provider_failover", failovers.append)
    tts.on("metrics", metrics.append)

    stream = tts.synthesize("Hello there, friend.")
    audio = await asyncio.wait_for(stream.collect(), 5)

    assert audio.sample_rate == 24_000
    assert audio.duration == pytest.approx(backup.audio_duration_for("Hello there, friend."), 0.05)
    assert stream.served_by == "mock/b"  # type: ignore[attr-defined]
    assert [(f.from_provider, f.to_provider) for f in failovers] == [("mock/a", "mock/b")]
    await wait_for(lambda: any(m.model == "b" and m.ttfb is not None for m in metrics))
    await tts.aclose()


async def test_tts_chunked_does_not_switch_once_audio_was_heard() -> None:
    backup = MockTTS(model="b")
    tts = FallbackTTS([FlakyTTS("midstream", model="a"), backup])
    with pytest.raises(ProviderConnectionError):
        await tts.synthesize("A long enough sentence to span several chunks.").collect()
    assert backup.requests == []
    await tts.aclose()


async def test_tts_sentence_streaming_fails_over_per_sentence() -> None:
    primary = FlakyTTS("connect", model="a")
    backup = MockTTS(model="b")
    tts = FallbackTTS([primary, backup], cooldown=3600)
    stream = tts.stream()
    stream.push_text("First sentence here. Second sentence here.")
    stream.end_input()
    texts = [a.text async for a in stream if a.text]
    assert texts == ["First sentence here.", "Second sentence here."]
    assert backup.requests == texts
    await tts.aclose()


async def test_tts_native_stream_replays_unsynthesized_text() -> None:
    # the primary dies silently after the first segment: the rest goes to the backup
    primary = FlakyTTS("ends", model="a", streaming=True, sample_rate=16_000)
    backup = MockTTS(model="b", streaming=True, sample_rate=24_000)
    tts = FallbackTTS([primary, backup], sample_rate=16_000)
    assert tts.capabilities.streaming
    failovers: list[ProviderFailover] = []
    tts.on("provider_failover", failovers.append)

    stream = tts.stream()
    stream.push_text("One. ")
    stream.flush()
    stream.push_text("Two is ")
    stream.push_text("longer. ")
    stream.flush()
    stream.push_text("Three.")
    stream.end_input()
    items = [a async for a in stream]

    assert primary.requests == ["One."]
    assert backup.requests == ["Two is longer.", "Three."]  # replayed, nothing lost
    assert [f.reason for f in failovers] == ["stream_ended"]
    finals = [a for a in items if a.is_final]
    assert len(finals) == 3
    assert {a.frame.sample_rate for a in items if a.frame} == {16_000}
    expected = sum(t.audio_duration_for(s) for t, s in
                   [(primary, "One."), (backup, "Two is longer."), (backup, "Three.")])  # fmt: skip
    total = sum(a.frame.duration for a in items)
    assert total == pytest.approx(expected, abs=0.02)
    await tts.aclose()


async def test_tts_native_stream_drains_the_resampler_per_segment() -> None:
    """Segment ends *drain* the resampler (#157): the provider's audio is resampled as one
    continuous stream (no filter restart from silence at each segment, exact sample count)
    instead of being flushed and reset per segment."""
    from voice_agent_next.audio.resample import StreamResampler

    texts = ["One.", "Two is longer.", "Three!"]

    async def run(tts: Any) -> list[Any]:
        stream = tts.stream()
        for text in texts:
            stream.push_text(text)
            stream.flush()
        stream.end_input()
        items = [a async for a in stream]
        await tts.aclose()
        return items

    source = await run(MockTTS(model="b", streaming=True, sample_rate=16_000))
    rs = StreamResampler(24_000)
    expected = b""
    for a in source:
        expected += rs.push(a.frame).data if a.frame else b""
        if a.is_final:
            expected += rs.drain().data
    got_items = await run(
        FallbackTTS([MockTTS(model="b", streaming=True, sample_rate=16_000)], sample_rate=24_000)
    )
    got = b"".join(a.frame.data for a in got_items if a.frame)
    assert len(got) == len(expected)
    # same samples up to rounding (soxr's output depends slightly on the chunking)
    import numpy as np

    diff = np.frombuffer(got, np.int16).astype(int) - np.frombuffer(expected, np.int16)
    assert np.abs(diff).max() <= 2


async def test_tts_native_stream_does_not_switch_mid_segment() -> None:
    backup = MockTTS(model="b", streaming=True)
    tts = FallbackTTS([FlakyTTS("midstream", model="a", streaming=True), backup])
    stream = tts.stream()
    stream.push_text("Hello world, how are you?")
    stream.end_input()
    with pytest.raises(ProviderConnectionError):
        async for _ in stream:
            pass
    assert backup.requests == []
    await tts.aclose()


async def test_tts_native_stream_first_audio_timeout() -> None:
    backup = MockTTS(model="b", streaming=True)
    tts = FallbackTTS(
        [FlakyTTS("stall", model="a", streaming=True), backup], first_audio_timeout=0.2
    )
    stream = tts.stream()
    stream.push_text("Anyone there?")
    stream.flush()
    first = await asyncio.wait_for(anext(aiter(stream)), 5)
    assert first.frame and backup.requests == ["Anyone there?"]
    await stream.aclose()
    await tts.aclose()


# ------------------------------------------------------------------ STT


async def test_stt_stream_failover_replays_the_current_utterance() -> None:
    primary = FlakySTT(fail_after_audio=0.5, model="a", default_text="primary")
    backup = FlakySTT(model="b", default_text="book a table for two")
    stt = FallbackSTT([primary, backup])
    failovers: list[ProviderFailover] = []
    stt.on("provider_failover", failovers.append)

    stream = stt.stream()
    speech = synth_speech(1.0, 16_000)
    for k in range(50):  # 20 ms frames
        stream.push_audio(speech.slice(k * 0.02, (k + 1) * 0.02))
        await asyncio.sleep(0)
    stream.flush()
    stream.end_input()
    events = [ev async for ev in stream]

    finals = [ev.text for ev in events if ev.type == STTEventType.FINAL_TRANSCRIPT]
    assert finals == ["book a table for two"]
    starts = [ev for ev in events if ev.type == STTEventType.START_OF_SPEECH]
    assert len(starts) == 1  # the replayed utterance is not announced twice
    assert backup.received[0] == pytest.approx(1.0, abs=0.03)  # the whole utterance
    assert [(f.from_provider, f.to_provider, f.reason) for f in failovers] == [
        ("mock/a", "mock/b", "connection")
    ]
    assert stream.served_by == "mock/b"  # type: ignore[attr-defined]
    await stt.aclose()


async def test_stt_replay_skips_audio_already_transcribed() -> None:
    primary = FlakySTT(fail_after_audio=1.3, model="a", transcripts=["first"])
    backup = FlakySTT(model="b", transcripts=["second"])
    stt = FallbackSTT([primary, backup], replay_seconds=3.0)
    stream = stt.stream()
    events: list[STTEvent] = []

    async def consume() -> None:
        async for ev in stream:
            events.append(ev)

    consumer = asyncio.create_task(consume())
    stream.push_audio(synth_speech(1.0, 16_000))
    stream.flush()
    await wait_for(lambda: any(e.type == STTEventType.FINAL_TRANSCRIPT for e in events))
    stream.push_audio(synth_speech(0.6, 16_000))  # the primary dies during this utterance
    stream.flush()
    stream.end_input()
    await asyncio.wait_for(consumer, 5)

    finals = [e.text for e in events if e.type == STTEventType.FINAL_TRANSCRIPT]
    assert finals == ["first", "second"]
    assert backup.received[0] == pytest.approx(0.6, abs=0.03)  # only the unfinished utterance
    await stt.aclose()


async def test_stt_stream_ending_silently_counts_as_failure() -> None:
    class Quitter(MockSTT):
        def _create_stream(self, *, language: str | None) -> STTStream:
            class S(STTStream):
                async def _run(self) -> None:
                    await self._input.recv()  # one frame, then a silent disconnect

            return S(self, language=language)

    backup = MockSTT(model="b", default_text="still heard")
    stt = FallbackSTT([Quitter(model="a"), backup])
    stream = stt.stream()
    stream.push_audio(synth_speech(0.5, 16_000))
    await wait_for(lambda: stt.stats.failovers == 1)  # it quit while audio kept coming
    stream.end_input()
    finals = [ev.text async for ev in stream if ev.type == STTEventType.FINAL_TRANSCRIPT]
    assert finals == ["still heard"]
    assert stt.stats.reasons == {"stream_ended": 1}
    await stt.aclose()


async def test_stt_final_timeout_catches_a_stalled_stream() -> None:
    class Deaf(MockSTT):
        def _create_stream(self, *, language: str | None) -> STTStream:
            class S(STTStream):
                async def _run(self) -> None:
                    async for _ in self._input:
                        pass  # accepts audio, never answers

            return S(self, language=language)

    stt = FallbackSTT([Deaf(model="a"), MockSTT(model="b", default_text="ok")], final_timeout=0.2)
    stream = stt.stream()
    stream.push_audio(synth_speech(0.5, 16_000))
    stream.flush()
    ev = await asyncio.wait_for(anext(aiter(stream)), 5)
    while ev.type != STTEventType.FINAL_TRANSCRIPT:
        ev = await asyncio.wait_for(anext(aiter(stream)), 5)
    assert ev.text == "ok"
    await stream.aclose()
    await stt.aclose()


async def test_stt_batch_transcribe_fails_over() -> None:
    class Down(MockSTT):
        async def _recognize(self, audio: AudioFrame, *, language: str | None) -> Transcript:
            raise ProviderConnectionError("down")

    stt = FallbackSTT(
        [Down(model="a", streaming=False), MockSTT(model="b", default_text="batch ok")]
    )
    assert not stt.capabilities.streaming
    result = await stt.transcribe(synth_speech(0.5, 48_000))
    assert result.text == "batch ok"
    assert stt.stats.served == {"mock/b": 1}
    await stt.aclose()


# ------------------------------------------------------------------ in a session


async def test_session_answers_although_the_primary_llm_is_down() -> None:
    primary = FlakyLLM("connect", model="primary")
    backup = MockLLM(model="backup", responses=["The backup answered."])
    llm = FallbackLLM([primary, backup])
    failovers: list[ProviderFailover] = []
    llm.on("provider_failover", failovers.append)
    session = mock_cascade(
        stt=FallbackSTT([FlakySTT(fail_after_audio=0.3, model="s1"), MockSTT(model="s2")]),
        llm=llm,
        tts=FallbackTTS([FlakyTTS("connect", model="t1"), MockTTS(model="t2")]),
        transcripts=None,
    )
    rec = Recorder(session)
    transport = LoopbackTransport()
    await session.start(Agent("be nice"), transport)
    await speak(transport)
    await wait_for(lambda: any("backup answered" in e.delta for e in rec.of("agent_transcript")))
    await wait_for(lambda: session.agent_state == AgentState.LISTENING)
    assert [f.to_provider for f in failovers] == ["mock/backup"]
    assert [e.text for e in rec.of("user_transcript") if e.is_final] == ["hello"]
    served = {m.model for m in rec.of("metrics") if isinstance(m, LLMMetrics) and not m.error}
    assert "backup" in served
    transport.end_user_audio()
    await asyncio.wait_for(session.wait_closed(), 5)
