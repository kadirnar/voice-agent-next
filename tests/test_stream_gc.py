"""Streams dropped without ``aclose()`` report nothing when garbage-collected (#151).

An abandoned stream's pending task is unreachable; the collector closes its coroutine
from whatever code happens to run at that moment (in CI: the middle of the next
request). Metrics or error logs emitted from there land at a random time, possibly in
another request's listeners. Each stream base must stay quiet then, while still
reporting normally from its own task (and from child tasks it spawned).
"""

from __future__ import annotations

import asyncio
import gc
import logging
from typing import Any

import pytest

from voice_agent_next.audio import AudioFrame
from voice_agent_next.chat import ChatContext
from voice_agent_next.fallback import FallbackLLM, FallbackSTT
from voice_agent_next.llm import LLMStream
from voice_agent_next.metrics import LLMMetrics, STTMetrics
from voice_agent_next.providers.mock import MockLLM, MockSTT
from voice_agent_next.stt import STTEvent, STTEventType, STTStream, Transcript
from voice_agent_next.tools import FunctionTool
from voice_agent_next.utils.aio import closed_outside

LOGGER = "voice_agent_next"


async def _park() -> None:
    """Let freshly created stream tasks run until they block on their private awaitables."""
    for _ in range(5):
        await asyncio.sleep(0)


def _failures(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.levelno >= logging.ERROR and r.name.startswith(LOGGER)  # not asyncio's own
    ]


# ------------------------------------------------------------------------------ LLM
class _HangingLLMStream(LLMStream):
    """Waits on a future only it can reach; its cleanup fails (a socket already gone)."""

    async def _run(self) -> None:
        self._push_first()
        try:
            await asyncio.get_running_loop().create_future()
        finally:
            raise ConnectionError("socket gone")

    def _push_first(self) -> None:
        from voice_agent_next.llm import ChatChunk

        self._push(ChatChunk(self.request_id, delta="partial "))


class _HangingLLM(MockLLM):
    """The first request hangs forever; later ones are ordinary mock replies."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.hang = True

    def _chat(self, ctx: ChatContext, *, tools: list[FunctionTool], **kw: Any) -> LLMStream:
        if self.hang:
            self.hang = False
            return _HangingLLMStream(self, ctx, tools=tools, **kw)
        return super()._chat(ctx, tools=tools, **kw)


def _ctx() -> ChatContext:
    ctx = ChatContext()
    ctx.add_message("user", "hello there")
    return ctx


async def test_abandoned_llm_stream_reports_nothing_when_garbage_collected(
    caplog: pytest.LogCaptureFixture,
) -> None:
    llm = _HangingLLM()
    got: list[LLMMetrics] = []
    llm.on("metrics", got.append)
    llm.chat(_ctx())  # dropped on the floor while its task is pending
    await _park()
    with caplog.at_level(logging.ERROR, logger=LOGGER):
        gc.collect()  # what the collector does at an unlucky moment
        await _park()
    assert got == []
    assert _failures(caplog) == []
    # the next request still reports exactly once, from its own task
    result = await llm.chat(_ctx()).collect()
    assert result.text.strip().endswith("hello there")
    assert len(got) == 1 and got[0].error is None and not got[0].cancelled


async def test_llm_stream_failure_is_still_reported(caplog: pytest.LogCaptureFixture) -> None:
    """The guard only silences finalization: a real failure logs and reports as before."""
    llm = _HangingLLM()
    got: list[LLMMetrics] = []
    llm.on("metrics", got.append)

    class _Boom(_HangingLLMStream):
        async def _run(self) -> None:
            raise ConnectionError("refused")

    boom = _Boom(
        llm, _ctx(), tools=[], tool_choice=None, temperature=None, max_tokens=None, extra={}
    )
    with caplog.at_level(logging.ERROR, logger=LOGGER), pytest.raises(ConnectionError):
        await boom.collect()
    assert len(got) == 1 and got[0].error is not None
    assert any("_Boom failed" in m for m in _failures(caplog))


async def test_abandoned_fallback_llm_stream_reports_nothing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    primary, backup = _HangingLLM(model="a"), MockLLM(model="b")
    fb = FallbackLLM([primary, backup])
    got: list[LLMMetrics] = []
    fb.on("metrics", got.append)
    primary.on("metrics", got.append)
    backup.on("metrics", got.append)
    fb.chat(_ctx())  # the primary's stream hangs; the wrapper stream is dropped
    await _park()
    with caplog.at_level(logging.ERROR, logger=LOGGER):
        gc.collect()
        await _park()
    assert [m for m in got if m.model == "a"] == []
    assert _failures(caplog) == []


# ------------------------------------------------------------------------------ STT
class _FinalizingSTTStream(STTStream):
    """Ignores input until it ends, then flushes a final transcript and fails to clean up."""

    async def _run(self) -> None:
        try:
            async for _ in self._input:
                pass
        finally:
            self._emit(
                STTEvent(STTEventType.FINAL_TRANSCRIPT, Transcript("late", "en", 1.0), "seg")
            )
            raise ConnectionError("socket gone")


class _FinalizingSTT(MockSTT):
    def _create_stream(self, *, language: str | None) -> STTStream:
        return _FinalizingSTTStream(self, language=language)


def _speech() -> AudioFrame:
    return AudioFrame.silence(0.2, 16_000)


async def test_abandoned_stt_stream_reports_nothing_when_garbage_collected(
    caplog: pytest.LogCaptureFixture,
) -> None:
    stt = _FinalizingSTT()
    got: list[STTMetrics] = []
    stt.on("metrics", got.append)
    stream = stt.stream()
    stream.push_audio(_speech())
    stream.flush()  # a final is now owed: emitting it would report STTMetrics
    await _park()
    del stream  # dropped without aclose() while its task waits on its own input
    with caplog.at_level(logging.ERROR, logger=LOGGER):
        gc.collect()
        await _park()
    assert got == []
    assert _failures(caplog) == []


async def test_stt_stream_still_reports_from_its_task(caplog: pytest.LogCaptureFixture) -> None:
    stt = _FinalizingSTT()
    got: list[STTMetrics] = []
    stt.on("metrics", got.append)
    stream = stt.stream()
    stream.push_audio(_speech())
    stream.end_input()
    events: list[STTEvent] = []

    async def consume() -> None:
        async for ev in stream:
            events.append(ev)

    with caplog.at_level(logging.ERROR, logger=LOGGER), pytest.raises(ConnectionError):
        await consume()
    assert [e.transcript.text for e in events if e.transcript] == ["late"]
    assert len(got) == 1
    assert any("_FinalizingSTTStream failed" in m for m in _failures(caplog))


async def test_stt_events_from_a_child_task_are_delivered() -> None:
    """Providers emit from receiver tasks their ``_run`` spawns: that is not finalization."""

    class _ChildEmitter(STTStream):
        async def _run(self) -> None:
            async def receiver() -> None:
                self._emit(
                    STTEvent(STTEventType.FINAL_TRANSCRIPT, Transcript("hi", "en", 1.0), "s")
                )

            await asyncio.create_task(receiver())

    class _STT(MockSTT):
        def _create_stream(self, *, language: str | None) -> STTStream:
            return _ChildEmitter(self, language=language)

    stt = _STT()
    got: list[STTMetrics] = []
    stt.on("metrics", got.append)
    stream = stt.stream()
    stream.push_audio(_speech())
    stream.flush()
    texts = [ev.transcript.text async for ev in stream if ev.transcript]
    assert texts == ["hi"]
    assert len(got) == 1


async def test_abandoned_fallback_stt_stream_reports_nothing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    primary = _FinalizingSTT(model="a")
    fb = FallbackSTT([primary, MockSTT(model="b")])
    got: list[STTMetrics] = []
    for emitter in (fb, primary):
        emitter.on("metrics", got.append)
    stream = fb.stream()
    stream.push_audio(_speech())
    stream.flush()
    await _park()
    del stream
    with caplog.at_level(logging.ERROR, logger=LOGGER):
        gc.collect()
        await _park()
    assert [m for m in got if m.model == "a"] == []
    assert _failures(caplog) == []


# ------------------------------------------------------------------------- helper
async def test_closed_outside() -> None:
    assert not closed_outside(None)
    seen: list[bool] = []

    async def body() -> None:
        seen.append(closed_outside(asyncio.current_task()))
        await asyncio.sleep(0)

    task = asyncio.create_task(body())
    await task
    assert seen == [False]  # inside its own task
    parked = asyncio.create_task(asyncio.sleep(10))
    await asyncio.sleep(0)
    assert not closed_outside(parked)  # suspended, observed from elsewhere
    parked.cancel()
