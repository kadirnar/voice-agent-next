"""Failure paths from the wave-4 audit not covered elsewhere (#142).

* ``_FallbackSynthesizeStream`` (native-streaming TTS failover): a provider refusing
  the connection, every provider failing, a non-failover error, and ``aclose()`` while
  a provider stalls;
* a session closed while a rotation is still opening (or preparing) the next
  connection.

The remaining items of #142 are covered by the tests listed in its PR.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from tests.test_fallback import FlakyTTS
from voice_agent_next import AudioFrame
from voice_agent_next.engine import EngineConnection, EngineOptions
from voice_agent_next.engines.rotation import RotatingConnection, RotatingEngine, RotationPolicy
from voice_agent_next.errors import (
    ProviderConnectionError,
    ProviderError,
)
from voice_agent_next.events import EngineStatus
from voice_agent_next.fallback import FallbackTTS
from voice_agent_next.providers.mock import MockEngine, MockTTS
from voice_agent_next.tts import SynthesizeStream
from voice_agent_next.utils import now

SR = 16_000


async def _drain(stream: SynthesizeStream) -> list[str]:
    return [a.text async for a in stream if a.text]


# ------------------------------------------------------------ _FallbackSynthesizeStream
async def test_tts_native_stream_fails_over_when_the_primary_refuses() -> None:
    primary = FlakyTTS("connect", model="a", streaming=True)
    backup = MockTTS(model="b", streaming=True)
    tts = FallbackTTS([primary, backup], cooldown=3600)
    stream = tts.stream()
    stream.push_text("Hello there. ")
    stream.flush()
    stream.push_text("Bye.")
    stream.end_input()
    await asyncio.wait_for(_drain(stream), 5)
    assert backup.requests == ["Hello there.", "Bye."]  # nothing was lost
    assert getattr(stream, "served_by", None) == "mock/b"
    await tts.aclose()


async def test_tts_native_stream_raises_the_last_error_when_every_provider_fails() -> None:
    tts = FallbackTTS(
        [
            FlakyTTS("connect", model="a", streaming=True),
            FlakyTTS("connect", model="b", streaming=True),
        ]
    )
    stream = tts.stream()
    stream.push_text("Anyone?")
    stream.end_input()
    with pytest.raises(ProviderConnectionError):
        await asyncio.wait_for(_drain(stream), 5)
    await tts.aclose()


async def test_tts_native_stream_does_not_fail_over_on_a_request_error() -> None:
    class BadRequestTTS(MockTTS):
        def _create_stream(self, *, voice: str | None) -> SynthesizeStream:
            class S(SynthesizeStream):
                async def _run(self) -> None:
                    async for _ in self._input:
                        raise ProviderError("bad request", status_code=400)

            return S(self, voice=voice)

    backup = MockTTS(model="b", streaming=True)
    tts = FallbackTTS([BadRequestTTS(model="a", streaming=True), backup])
    stream = tts.stream()
    stream.push_text("Hello.")
    stream.end_input()
    with pytest.raises(ProviderError, match="bad request"):
        await asyncio.wait_for(_drain(stream), 5)
    assert backup.requests == []  # the same request would fail everywhere
    await tts.aclose()


async def test_tts_native_stream_aclose_while_a_provider_stalls() -> None:
    inner: list[SynthesizeStream] = []

    class StallingTTS(FlakyTTS):
        def _create_stream(self, *, voice: str | None) -> SynthesizeStream:
            s = super()._create_stream(voice=voice)
            inner.append(s)
            return s

    tts = FallbackTTS([StallingTTS("stall", model="a", streaming=True)], first_audio_timeout=None)
    stream = tts.stream()
    stream.push_text("Hello.")
    stream.flush()
    for _ in range(10):
        await asyncio.sleep(0)
    assert inner  # the provider's stream is open and stalling
    await asyncio.wait_for(stream.aclose(), 5)
    assert stream._task.done()
    assert inner[0]._task.done()  # the provider's stream was closed with it
    await tts.aclose()


# ---------------------------------------------------------------- mid-rotation cancel
class GatedConnectEngine(MockEngine):
    """Every connection after the first blocks in ``connect()`` until released."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.connects = 0
        self.blocked = asyncio.Event()
        self.gate = asyncio.Event()
        self.conns: list[EngineConnection] = []

    async def connect(self, options: EngineOptions) -> EngineConnection:
        self.connects += 1
        if self.connects > 1:
            self.blocked.set()
            await self.gate.wait()
        conn = await super().connect(options)
        self.conns.append(conn)
        return conn


async def _collect(conn: EngineConnection, into: list[Any]) -> None:
    async for ev in conn.events():
        into.append(ev)


@pytest.mark.parametrize("forced", [True, False])
async def test_closing_while_a_rotation_opens_the_next_connection(forced: bool) -> None:
    """``forced``: the switch itself is opening the connection (``_switch``); otherwise
    the make-before-break preparation is (``_prepare``). Closing must cancel either
    promptly, close the current connection and never finish the switch."""
    inner = GatedConnectEngine()
    engine = RotatingEngine(inner, policy=RotationPolicy(quiet_period=0.05))
    conn = await engine.connect(EngineOptions())
    assert isinstance(conn, RotatingConnection)
    events: list[Any] = []
    collector = asyncio.create_task(_collect(conn, events))
    await conn.send_audio(AudioFrame.silence(0.1, SR))
    conn.rotate("test", deadline=now() if forced else None)
    await asyncio.wait_for(inner.blocked.wait(), 5)  # the next connection is being opened
    await asyncio.wait_for(conn.aclose(), 5)
    (first,) = inner.conns  # the blocked connect never produced a connection
    assert first.closed
    inner.gate.set()  # a late release changes nothing
    for _ in range(10):
        await asyncio.sleep(0)
    assert len(inner.conns) == 1
    statuses = [e.status for e in events if isinstance(e, EngineStatus)]
    assert "reconnected" not in statuses
    assert conn.rotations == 0
    collector.cancel()
    await asyncio.gather(collector, return_exceptions=True)
    await engine.aclose()
