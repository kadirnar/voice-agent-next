"""OpenAI STT: realtime transcription sessions (fake WebSocket server) and the
``/audio/transcriptions`` endpoint (``httpx.MockTransport``).

No network, no API key — except the ``integration`` tests at the end, which need
``OPENAI_API_KEY``.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
from collections.abc import Callable
from typing import Any

import httpx
import numpy as np
import pytest

from tests.fake_transcription_server import FakeTranscriptionServer
from voice_agent_next import AudioFrame
from voice_agent_next.engine import EngineOptions
from voice_agent_next.engines.cascade import CascadeEngine, CascadeOptions
from voice_agent_next.errors import (
    AuthenticationError,
    ConfigurationError,
    ProviderConnectionError,
    ProviderError,
    ProviderTimeoutError,
    RateLimitError,
)
from voice_agent_next.events import InputTranscript, ResponseDone
from voice_agent_next.metrics import STTMetrics
from voice_agent_next.providers.azure_openai import AzureOpenAISTT
from voice_agent_next.providers.energy import EnergyVAD
from voice_agent_next.providers.localai import LocalAISTT
from voice_agent_next.providers.mock import MockLLM, MockTTS, synth_speech
from voice_agent_next.providers.openai.stt import OpenAISTT
from voice_agent_next.providers.speaches import SpeachesSTT
from voice_agent_next.registry import create
from voice_agent_next.stt import StreamAdapter, STTEvent, STTEventType, STTStream

E = STTEventType
COMPLETED = "conversation.item.input_audio_transcription.completed"
RATE = 24_000
BYTES_PER_SECOND = RATE * 2


@pytest.fixture(autouse=True)
def _no_openai_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("OPENAI_BASE_URL", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_AD_TOKEN",
                 "AZURE_OPENAI_ENDPOINT", "SPEACHES_BASE_URL", "SPEACHES_API_KEY",
                 "LOCALAI_BASE_URL", "LOCALAI_API_KEY"):  # fmt: skip
        monkeypatch.delenv(name, raising=False)


def speech(seconds: float, *, offset: int = 0) -> AudioFrame:
    return synth_speech(seconds, RATE, offset=offset)


async def events_until(
    stream: STTStream, predicate: Callable[[STTEvent], bool], timeout: float = 5.0
) -> list[STTEvent]:
    """Events up to and including the first one matching ``predicate``."""
    got: list[STTEvent] = []

    async def run() -> None:
        async for ev in stream:
            got.append(ev)
            if predicate(ev):
                return
        raise AssertionError(f"stream ended before the expected event: {got}")

    await asyncio.wait_for(run(), timeout)
    return got


def is_final(ev: STTEvent) -> bool:
    return ev.type == E.FINAL_TRANSCRIPT


async def wait_until(predicate: Callable[[], bool], timeout: float = 5.0) -> None:
    async def poll() -> None:
        while not predicate():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(poll(), timeout)


def stt_for(server: FakeTranscriptionServer, **kw: Any) -> OpenAISTT:
    return OpenAISTT(api_key="test-key", base_url=server.url, **kw)


# ------------------------------------------------------------------- session setup
async def test_session_update_url_and_auth() -> None:
    async with FakeTranscriptionServer() as server:
        stt = stt_for(
            server,
            languages=["en-US", "fr"],
            prompt="A support call about billing.",
            keywords=["AC-42", "Premium Plus"],
            delay="low",
            noise_reduction="near_field",
        )
        stream = stt.stream()
        await wait_until(lambda: bool(server.events("session.update")))
        await stream.aclose()

    update = server.events("session.update")[0]
    assert update["event_id"]
    assert update["session"] == {
        "type": "transcription",
        "audio": {
            "input": {
                "format": {"type": "audio/pcm", "rate": 24_000},
                "transcription": {
                    "model": "gpt-live-transcribe",
                    "languages": ["en", "fr"],
                    "keywords": ["AC-42", "Premium Plus"],
                    "prompt": "A support call about billing.",
                    "delay": "low",
                },
                "turn_detection": None,
                "noise_reduction": {"type": "near_field"},
            }
        },
    }
    handshake = server.handshakes[0]
    assert handshake.path == "/v1/realtime"
    assert handshake.query == {"model": "gpt-live-transcribe", "intent": "transcription"}
    assert handshake.headers["authorization"] == "Bearer test-key"
    assert "openai-beta" not in handshake.headers


async def test_legacy_models_get_a_single_language_and_logprobs() -> None:
    async with FakeTranscriptionServer() as server:
        stt = stt_for(server, model="gpt-4o-mini-transcribe", language="en", logprobs=True)
        stream = stt.stream(language="de-DE")  # Agent(language=...) wins
        await wait_until(lambda: bool(server.events("session.update")))
        await stream.aclose()
    session = server.events("session.update")[0]["session"]
    assert session["audio"]["input"]["transcription"] == {
        "model": "gpt-4o-mini-transcribe",
        "language": "de",
    }
    assert session["include"] == ["item.input_audio_transcription.logprobs"]


def test_realtime_url_omits_the_model_for_openai() -> None:
    assert OpenAISTT(api_key="k").realtime_url() == (
        "wss://api.openai.com/v1/realtime?intent=transcription"
    )
    eu = OpenAISTT(api_key="k", base_url="https://eu.api.openai.com/v1")
    assert eu.realtime_url() == "wss://eu.api.openai.com/v1/realtime?intent=transcription"
    # gateways route on ?model=
    proxy = OpenAISTT(api_key="k", base_url="https://gw.example.com/v1", query={"x": "1"})
    assert proxy.realtime_url() == (
        "wss://gw.example.com/v1/realtime?model=gpt-live-transcribe&intent=transcription&x=1"
    )


def test_language_hints_are_normalized() -> None:
    live = OpenAISTT(api_key="k", languages=["zh-TW", "en_US", "yue", "en"])
    assert live.languages_for(None) == ["zh-tw", "en", "yue"]
    assert live.languages_for("pt-BR") == ["pt"]  # an explicit per-stream language wins
    legacy = OpenAISTT(api_key="k", model="whisper-1", language="zh-CN")
    assert legacy.transcription_config() == {"model": "whisper-1", "language": "zh"}
    assert OpenAISTT(api_key="k").transcription_config() == {"model": "gpt-live-transcribe"}


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"realtime": False}, "only served by realtime"),
        ({"turn_detection": "server_vad"}, "no server-side turn detection"),
        ({"model": "whisper-1", "turn_detection": "server_vad", "realtime": False}, "realtime"),
        ({"sample_rate": 16_000}, "24 kHz"),
        ({"model": "whisper-1", "keywords": ["x"]}, "keywords"),
        ({"model": "whisper-1", "languages": ["en", "fr"]}, "single language"),
        ({"keywords": ["bad <tag>"]}, "invalid keyword"),
        ({"keywords": "AC-42"}, "list"),
        ({"delay": "fast"}, "delay"),
        ({"noise_reduction": "studio"}, "noise_reduction"),
        ({"model": "gpt-4o-transcribe", "turn_detection": {"silence_duration_ms": 300}}, "type"),
        ({"chunk_ms": 0}, "chunk_ms"),
    ],
)
def test_invalid_options(kwargs: dict[str, Any], match: str) -> None:
    with pytest.raises(ConfigurationError, match=match):
        OpenAISTT(api_key="k", **kwargs)


def test_registry_and_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
        create("stt", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    stt = create("stt", "openai")
    assert isinstance(stt, OpenAISTT)
    assert stt.model == "gpt-live-transcribe" and stt.sample_rate == 24_000
    assert stt.capabilities.streaming and stt.capabilities.interim_results
    assert not stt.capabilities.end_of_turn
    assert stt.endpoint.request_headers() == {"Authorization": "Bearer sk-env"}
    whisper = create("stt", {"provider": "openai/whisper-1", "language": "en"})
    assert whisper.model == "whisper-1" and whisper.language == "en"
    # the OpenAI key is never sent to another server
    other = OpenAISTT(base_url="http://127.0.0.1:9/v1")
    assert other.endpoint.api_key is None and other.endpoint.request_headers() == {}
    # OPENAI_BASE_URL is OpenAI's own variable: the key goes with it
    monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway.example.com/v1")
    gateway = OpenAISTT()
    assert gateway.base_url == "https://gateway.example.com/v1"
    assert gateway.endpoint.api_key == "sk-env"
    assert OpenAISTT(api_key="").endpoint.api_key is None  # explicit "no key"


def test_semantic_vad_declares_end_of_turn() -> None:
    stt = OpenAISTT(api_key="k", model="gpt-4o-mini-transcribe", turn_detection="semantic_vad")
    assert stt.capabilities.end_of_turn
    assert stt.session_config()["audio"]["input"]["turn_detection"] == {"type": "semantic_vad"}


# ------------------------------------------------------------------ manual commits
async def test_flush_commits_and_the_final_follows_the_deltas() -> None:
    async with FakeTranscriptionServer(transcripts=["Book a table for two."]) as server:
        stt = stt_for(server, model="gpt-4o-transcribe")
        metrics: list[STTMetrics] = []
        stt.on("metrics", metrics.append)
        stream = stt.stream()
        audio = speech(0.5)
        for n in range(25):  # 20 ms frames, like a transport
            stream.push_audio(audio.slice(n * 0.02, (n + 1) * 0.02))
        stream.flush()
        events = await events_until(stream, is_final)
        await stream.aclose()

    interims = [ev.text for ev in events if ev.type == E.INTERIM_TRANSCRIPT]
    assert interims == ["Book", "Book a", "Book a table", "Book a table for",
                        "Book a table for two."]  # fmt: skip
    final = events[-1]
    assert final.text == "Book a table for two."
    committed = next(e for e in server.sent if e["type"] == "input_audio_buffer.committed")
    assert final.segment_id == committed["item_id"]
    assert final.transcript is not None
    assert final.transcript.start_time == 0.0
    assert final.transcript.end_time == pytest.approx(0.5)
    commits = server.events("input_audio_buffer.commit")
    assert len(commits) == 1 and commits[0]["event_id"]
    assert server.audio_bytes() == round(0.5 * BYTES_PER_SECOND)
    assert all(e["audio"] <= 0.05 * BYTES_PER_SECOND for e in server.events(
        "input_audio_buffer.append"))  # fmt: skip
    streamed = [m for m in metrics if m.streamed]
    assert streamed and streamed[0].latency is not None and streamed[0].audio_duration > 0.4


async def test_live_model_streams_deltas_before_the_commit() -> None:
    async with FakeTranscriptionServer(transcripts=["hello there my friend"]) as server:
        stream = stt_for(server).stream()
        stream.push_audio(speech(0.5))
        first = await events_until(stream, lambda ev: ev.type == E.INTERIM_TRANSCRIPT)
        assert server.events("input_audio_buffer.commit") == []  # no commit yet
        stream.flush()
        rest = await events_until(stream, is_final)
        await stream.aclose()
    assert first[-1].text == "hello"
    interims = [ev.text for ev in first + rest if ev.type == E.INTERIM_TRANSCRIPT]
    assert interims[-1] == "hello there my friend"
    assert rest[-1].text == "hello there my friend"
    assert rest[-1].segment_id == first[-1].segment_id  # same item before and after commit


async def test_a_flush_without_enough_audio_is_answered_at_once() -> None:
    async with FakeTranscriptionServer(transcripts=["Yes."]) as server:
        stream = stt_for(server, model="gpt-4o-mini-transcribe").stream()
        stream.flush()  # nothing buffered
        empty = await events_until(stream, is_final)
        stream.push_audio(speech(0.06))  # below the 100 ms the API accepts
        stream.flush()
        short = await events_until(stream, is_final)
        assert server.events("input_audio_buffer.commit") == []
        stream.push_audio(speech(0.3, offset=1440))
        stream.flush()
        final = await events_until(stream, is_final)
        await stream.aclose()
    assert [ev.text for ev in empty] == [""] and [ev.text for ev in short] == [""]
    assert final[-1].text == "Yes."
    assert len(server.events("input_audio_buffer.commit")) == 1
    assert server.audio_bytes() == round(0.36 * BYTES_PER_SECOND)  # the short tail was kept


async def test_a_flush_during_a_pending_commit_waits_for_its_final() -> None:
    async with FakeTranscriptionServer(transcripts=["Hi there."], delays=[0.3]) as server:
        stream = stt_for(server, model="gpt-4o-transcribe").stream()
        stream.push_audio(speech(0.4))
        stream.flush()
        stream.flush()  # nothing new: answered by the pending commit's final
        events = await events_until(stream, is_final)
        extra = await asyncio.wait_for(_collect_for(stream, 0.3), 2)
        await stream.aclose()
    assert events[-1].text == "Hi there."
    assert [ev for ev in extra if is_final(ev)] == []
    assert len(server.events("input_audio_buffer.commit")) == 1


async def _collect_for(stream: STTStream, seconds: float) -> list[STTEvent]:
    got: list[STTEvent] = []

    async def run() -> None:
        async for ev in stream:
            got.append(ev)

    with pytest.raises(TimeoutError):
        await asyncio.wait_for(run(), seconds)
    return got


async def test_finals_are_emitted_in_commit_order() -> None:
    async with FakeTranscriptionServer(
        transcripts=["First part.", "Second part."], delays=[0.4, 0.0]
    ) as server:
        stream = stt_for(server, model="gpt-4o-transcribe").stream()
        stream.push_audio(speech(0.3))
        stream.flush()
        stream.push_audio(speech(0.3, offset=7200))
        stream.flush()
        await wait_until(lambda: len(server.sent_events(COMPLETED)) == 2)
        events = await events_until(stream, lambda ev: is_final(ev) and ev.text.startswith("Sec"))
        await stream.aclose()
    completed = server.sent_events(COMPLETED)
    assert [e["transcript"] for e in completed] == ["Second part.", "First part."]
    assert [ev.text for ev in events if is_final(ev)] == ["First part.", "Second part."]


async def test_end_input_pads_the_tail_and_waits_for_the_transcript() -> None:
    async with FakeTranscriptionServer(transcripts=["Ok."], delays=[0.2]) as server:
        stream = stt_for(server, model="gpt-4o-transcribe").stream()
        stream.push_audio(speech(0.05))
        stream.end_input()
        events = [ev async for ev in stream]  # ends after the last final
        await stream.aclose()
    assert [ev.text for ev in events if is_final(ev)] == ["Ok."]
    assert server.audio_bytes() == round(0.1 * BYTES_PER_SECOND)  # padded to 100 ms


async def test_failed_transcription_yields_the_partial_text() -> None:
    async with FakeTranscriptionServer(transcripts=["one two three"], fail_items=[0]) as server:
        stream = stt_for(server).stream()
        stream.push_audio(speech(0.25))  # two words streamed before the commit
        stream.flush()
        events = await events_until(stream, is_final)
        await stream.aclose()
    assert events[-1].text == "one two three"  # remaining deltas arrive before the failure


async def test_logprobs_become_the_confidence() -> None:
    async with FakeTranscriptionServer(transcripts=["Fine thanks."]) as server:
        stream = stt_for(server, model="gpt-4o-transcribe", logprobs=True).stream()
        stream.push_audio(speech(0.3))
        stream.flush()
        events = await events_until(stream, is_final)
        await stream.aclose()
    assert events[-1].transcript is not None
    assert events[-1].transcript.confidence == pytest.approx(math.exp(-0.1))


async def test_gpt_transcribe_reports_the_detected_language() -> None:
    async with FakeTranscriptionServer(transcripts=["Bonjour."]) as server:
        stream = stt_for(server, model="gpt-transcribe").stream()
        stream.push_audio(speech(0.3))
        stream.flush()
        events = await events_until(stream, is_final)
        await stream.aclose()
    assert events[-1].transcript is not None and events[-1].transcript.language == "en"


async def test_transcribe_with_a_realtime_only_model_uses_a_session() -> None:
    async with FakeTranscriptionServer(transcripts=["Short clip."]) as server:
        stt = stt_for(server)
        transcript = await stt.transcribe(synth_speech(0.4, 16_000))  # resampled to 24 kHz
    assert transcript.text == "Short clip."
    assert len(server.events("input_audio_buffer.commit")) == 1


# ---------------------------------------------------------------- errors / reconnect
async def test_a_dropped_connection_resends_the_pending_commit() -> None:
    async with FakeTranscriptionServer(
        transcripts=["Lost then found."], drop_on_commits=[(0, 1)]
    ) as server:
        stt = stt_for(server, model="gpt-4o-transcribe", reconnect_backoff=0.05)
        stream = stt.stream()
        stream.push_audio(speech(0.4))
        stream.flush()
        events = await events_until(stream, is_final, timeout=10)
        await stream.aclose()
    assert events[-1].text == "Lost then found."
    assert len(server.handshakes) == 2
    assert len(server.events("session.update", connection=1)) == 1  # configured again
    assert server.audio_bytes(connection=1) == round(0.4 * BYTES_PER_SECOND)  # re-sent
    assert len(server.events("input_audio_buffer.commit", connection=1)) == 1


async def test_a_drop_during_the_replay_loses_nothing() -> None:
    def script(seconds: float) -> str:
        return "Alpha." if seconds < 0.4 else "Beta."

    async with FakeTranscriptionServer(
        transcripts=script, delays=[1.0], drop_on_commits=[(0, 2), (1, 1)]
    ) as server:
        stt = stt_for(server, model="gpt-4o-transcribe", reconnect_backoff=0.05)
        stream = stt.stream()
        stream.push_audio(speech(0.3))
        stream.flush()  # committed on the first session, transcribed slowly
        stream.push_audio(speech(0.5, offset=7200))
        stream.flush()  # the first session drops here, the second one while replaying
        events = await events_until(
            stream, lambda ev: is_final(ev) and ev.text == "Beta.", timeout=10
        )
        await stream.aclose()
    assert [ev.text for ev in events if is_final(ev)] == ["Alpha.", "Beta."]
    assert len(server.handshakes) == 3
    assert server.audio_bytes(connection=2) == round(0.8 * BYTES_PER_SECOND)
    assert len(server.events("input_audio_buffer.commit", connection=2)) == 2


async def test_a_failure_for_an_uncommitted_item_is_no_final() -> None:
    async with FakeTranscriptionServer(transcripts=["Kept."]) as server:
        stream = stt_for(server, model="gpt-4o-transcribe").stream()
        await wait_until(lambda: bool(server.events("session.update")))
        await server.push({"type": "conversation.item.input_audio_transcription.failed",
                           "event_id": "e1", "item_id": "item_discarded", "content_index": 0,
                           "error": {"type": "transcription_error", "code": "discarded",
                                     "message": "turn discarded before commit"}})  # fmt: skip
        stream.push_audio(speech(0.3))
        stream.flush()
        events = await events_until(stream, is_final)
        await stream.aclose()
    assert [ev.text for ev in events if is_final(ev)] == ["Kept."]


async def test_an_expired_session_reconnects_and_keeps_the_open_audio() -> None:
    async with FakeTranscriptionServer(transcripts=["Still here."]) as server:
        stt = stt_for(server, model="gpt-4o-transcribe", reconnect_backoff=0.05)
        stream = stt.stream()
        stream.push_audio(speech(0.3))
        await wait_until(lambda: server.audio_bytes() >= round(0.3 * BYTES_PER_SECOND))
        await server.push({"type": "error", "event_id": "event_x", "error": {
            "type": "invalid_request_error", "code": "session_expired",
            "message": "Your session hit the maximum duration of 60 minutes.",
        }})  # fmt: skip
        await wait_until(lambda: len(server.handshakes) == 2)
        stream.push_audio(speech(0.2, offset=7200))
        stream.flush()
        events = await events_until(stream, is_final, timeout=10)
        await stream.aclose()
    assert events[-1].text == "Still here."
    assert server.audio_bytes(connection=1) == round(0.5 * BYTES_PER_SECOND)


async def test_reconnects_are_bounded() -> None:
    async with FakeTranscriptionServer() as server:
        stream = stt_for(server, max_reconnect_attempts=0).stream()
        stream.push_audio(speech(0.2))
        await wait_until(lambda: server.audio_bytes() > 0)
        server.drop()
        with pytest.raises(ProviderConnectionError, match="transcription connection"):
            await events_until(stream, is_final)
        await stream.aclose()


@pytest.mark.parametrize(
    ("status", "error"),
    [
        (401, AuthenticationError),
        (403, AuthenticationError),
        (429, RateLimitError),
        (500, ProviderConnectionError),
    ],
)
async def test_handshake_errors_are_mapped(status: int, error: type[ProviderError]) -> None:
    async with FakeTranscriptionServer(reject_status=status) as server:
        stream = stt_for(server).stream()
        with pytest.raises(error) as info:
            await events_until(stream, is_final)
        await stream.aclose()
    assert info.value.status_code == status


async def test_a_wrong_api_key_is_an_authentication_error() -> None:
    async with FakeTranscriptionServer(api_key="right") as server:
        stream = stt_for(server).stream()  # sends "test-key"
        with pytest.raises(AuthenticationError):
            await events_until(stream, is_final)
        await stream.aclose()


async def test_connection_refused_is_a_connection_error() -> None:
    async with FakeTranscriptionServer() as server:
        url = server.url
    stream = OpenAISTT(api_key="k", base_url=url, connect_timeout=5).stream()  # server gone
    # refused at once on Linux/macOS; Windows retries the SYN and may hit the timeout first
    with pytest.raises((ProviderConnectionError, ProviderTimeoutError)) as info:
        await events_until(stream, is_final)
    assert info.value.retryable
    await stream.aclose()


async def test_a_rejected_session_update_fails_the_stream() -> None:
    reject = {"type": "invalid_request_error", "code": "invalid_model",
              "message": "Model 'gpt-nope' is not supported", "param": "session.audio"}  # fmt: skip
    async with FakeTranscriptionServer(reject_session_update=reject) as server:
        stream = stt_for(server).stream()
        with pytest.raises(ProviderError, match="not supported") as info:
            await events_until(stream, is_final)
        await stream.aclose()
    assert not info.value.retryable
    assert "session.update" in str(info.value)


async def test_an_auth_error_event_is_fatal() -> None:
    async with FakeTranscriptionServer() as server:
        stream = stt_for(server).stream()
        await wait_until(lambda: bool(server.events("session.update")))
        await server.push({"type": "error", "event_id": "e1", "error": {
            "type": "invalid_request_error", "code": "invalid_api_key",
            "message": "Incorrect API key provided."}})  # fmt: skip
        with pytest.raises(AuthenticationError, match="Incorrect API key"):
            await events_until(stream, is_final)
        await stream.aclose()


async def test_other_error_events_are_logged_and_the_stream_goes_on() -> None:
    async with FakeTranscriptionServer(transcripts=["Go on."]) as server:
        stream = stt_for(server, model="gpt-4o-transcribe").stream()
        await wait_until(lambda: bool(server.events("session.update")))
        await server.push(
            {
                "type": "error",
                "event_id": "e1",
                "error": {
                    "type": "server_error",
                    "code": None,
                    "message": "The server had an error.",
                },
            }
        )
        stream.push_audio(speech(0.2))
        stream.flush()
        events = await events_until(stream, is_final)
        await stream.aclose()
    assert events[-1].text == "Go on."


# --------------------------------------------------------------- server turn detection
async def test_server_vad_maps_speech_events_and_commits_by_itself() -> None:
    async with FakeTranscriptionServer(transcripts=["Server segmented."]) as server:
        stt = stt_for(
            server,
            model="gpt-4o-mini-transcribe",
            turn_detection={"type": "server_vad", "silence_duration_ms": 300},
        )
        stream = stt.stream()
        stream.push_audio(AudioFrame.silence(0.3, RATE))
        stream.push_audio(speech(0.6))
        stream.push_audio(AudioFrame.silence(0.6, RATE))
        events = await events_until(stream, is_final)
        stream.flush()  # idle: answered at once, no commit
        ack = await events_until(stream, is_final)
        await stream.aclose()
    kinds = [ev.type for ev in events]
    assert (
        kinds.index(E.START_OF_SPEECH)
        < kinds.index(E.END_OF_SPEECH)
        < kinds.index(E.FINAL_TRANSCRIPT)
    )
    start = next(ev for ev in events if ev.type == E.START_OF_SPEECH)
    stop = next(ev for ev in events if ev.type == E.END_OF_SPEECH)
    assert start.transcript is not None and start.transcript.start_time == pytest.approx(
        0.3, abs=0.1
    )
    assert stop.transcript is not None and stop.transcript.end_time is not None
    assert events[-1].text == "Server segmented."
    assert [ev.text for ev in ack] == [""]
    assert server.events("input_audio_buffer.commit") == []
    td = server.events("session.update")[0]["session"]["audio"]["input"]["turn_detection"]
    assert td == {"type": "server_vad", "silence_duration_ms": 300}


async def test_semantic_vad_ends_the_turn_after_the_final() -> None:
    async with FakeTranscriptionServer(transcripts=["All done."]) as server:
        stt = stt_for(server, model="gpt-4o-mini-transcribe", turn_detection="semantic_vad")
        stream = stt.stream()
        stream.push_audio(speech(0.5))
        stream.push_audio(AudioFrame.silence(0.7, RATE))
        events = await events_until(stream, lambda ev: ev.type == E.END_OF_TURN)
        await stream.aclose()
    finals = [n for n, ev in enumerate(events) if is_final(ev)]
    assert finals and finals[-1] == len(events) - 2
    assert events[-1].text == "All done."


# --------------------------------------------------------------------------- cascade
async def test_the_cascade_vad_commits_the_user_turn() -> None:
    async with FakeTranscriptionServer(transcripts=["What time is it?"]) as server:
        llm = MockLLM(responses=["It is noon."])
        engine = CascadeEngine(
            stt=stt_for(server, model="gpt-4o-transcribe"),
            llm=llm,
            tts=MockTTS(),
            vad=EnergyVAD(),
            options=CascadeOptions(min_endpointing_delay=0.0),
        )
        conn = await engine.connect(EngineOptions())
        events: list[object] = []

        async def drain() -> None:
            async for ev in conn.events():
                events.append(ev)
                if isinstance(ev, ResponseDone):
                    return

        drainer = asyncio.create_task(drain())
        audio = AudioFrame.concat([speech(0.6), AudioFrame.silence(0.8, RATE)])
        for n in range(70):
            await conn.send_audio(audio.slice(n * 0.02, (n + 1) * 0.02))
        await asyncio.wait_for(drainer, 10)
        await conn.aclose()
        await engine.aclose()
    finals = [e.text for e in events if isinstance(e, InputTranscript) and e.is_final]
    assert finals == ["What time is it?"]
    last_user = llm.requests[0].last_message("user")
    assert last_user is not None and last_user.text == "What time is it?"
    assert len(server.events("input_audio_buffer.commit")) >= 1


# -------------------------------------------------------------- /audio/transcriptions
class Recorder:
    """An ``httpx.MockTransport`` handler that records requests and replays responses."""

    def __init__(self, *responses: httpx.Response) -> None:
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        request.read()
        self.requests.append(request)
        return self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self))


def multipart_fields(request: httpx.Request) -> dict[str, list[str]]:
    """The non-file fields of a multipart request, by name (repeated fields as lists)."""
    boundary = request.headers["content-type"].split("boundary=")[1].encode()
    fields: dict[str, list[str]] = {}
    for part in request.content.split(b"--" + boundary):
        head, _, body = part.partition(b"\r\n\r\n")
        if b'name="' not in head or b"filename=" in head:
            continue
        name = head.split(b'name="')[1].split(b'"')[0].decode()
        fields.setdefault(name, []).append(body.rstrip(b"\r\n").decode())
    return fields


def uploaded_wav(request: httpx.Request) -> bytes:
    marker = b'filename="audio.wav"'
    part = request.content.split(marker)[1]
    return part.split(b"\r\n\r\n", 1)[1]


def sse(*events: dict[str, Any] | str) -> httpx.Response:
    lines = [f"data: {e if isinstance(e, str) else json.dumps(e)}\n\n" for e in events]
    return httpx.Response(
        200, headers={"content-type": "text/event-stream"}, content="".join(lines).encode()
    )


async def test_batch_whisper_json() -> None:
    rec = Recorder(httpx.Response(200, json={"text": " Hello there. ", "usage": {
        "type": "duration", "seconds": 1}}))  # fmt: skip
    stt = OpenAISTT(api_key="k", model="whisper-1", language="en-US", http_client=rec.client())
    transcript = await stt.transcribe(synth_speech(0.5, 16_000))
    await stt.aclose()
    assert transcript.text == "Hello there." and transcript.language == "en"
    request = rec.requests[0]
    assert str(request.url) == "https://api.openai.com/v1/audio/transcriptions"
    assert request.headers["authorization"] == "Bearer k"
    assert multipart_fields(request) == {
        "model": ["whisper-1"],
        "language": ["en"],
        "response_format": ["json"],
    }
    wav = uploaded_wav(request)
    assert wav[:4] == b"RIFF" and int.from_bytes(wav[24:28], "little") == 24_000


async def test_batch_gpt_transcribe_streams_server_sent_events() -> None:
    rec = Recorder(
        sse(
            {"type": "transcript.text.delta", "delta": "Bonjour"},
            {"type": "transcript.text.delta", "delta": ", ça va ?"},
            {"type": "transcript.text.done", "text": "Bonjour, ça va ?",
             "languages": [{"code": "fr"}], "usage": {"type": "tokens", "input_tokens": 14,
                                                      "output_tokens": 6, "total_tokens": 20}},
        )
    )  # fmt: skip
    stt = OpenAISTT(
        api_key="k",
        model="gpt-transcribe",
        languages=["fr", "en"],
        keywords=["ça va"],
        prompt="Small talk.",
        http_client=rec.client(),
    )
    assert stt.http_streaming
    transcript = await stt.transcribe(speech(0.3))
    assert transcript.text == "Bonjour, ça va ?" and transcript.language == "fr"
    assert multipart_fields(rec.requests[0]) == {
        "model": ["gpt-transcribe"],
        "languages[]": ["fr", "en"],
        "keywords[]": ["ça va"],
        "prompt": ["Small talk."],
        "response_format": ["json"],
        "stream": ["true"],
    }


async def test_batch_sse_without_a_done_event_joins_the_deltas() -> None:
    rec = Recorder(
        sse(
            {"type": "transcript.text.delta", "delta": "And so,"},
            {"type": "transcript.text.delta", "delta": " my fellow Americans"},
            "[DONE]",
        )
    )
    stt = OpenAISTT(api_key="k", model="gpt-4o-mini-transcribe", http_client=rec.client())
    transcript = await stt.transcribe(speech(0.3))
    assert transcript.text == "And so, my fellow Americans"


async def test_batch_word_timestamps_and_logprobs() -> None:
    words = [{"word": "Hi", "start": 0.1, "end": 0.3}, {"word": "you", "start": 0.4, "end": 0.6}]
    rec = Recorder(httpx.Response(200, json={
        "task": "transcribe", "language": "english", "duration": 0.7, "text": "Hi you",
        "words": words}))  # fmt: skip
    stt = OpenAISTT(api_key="k", model="whisper-1", word_timestamps=True, realtime=False,
                    http_client=rec.client())  # fmt: skip
    assert stt.capabilities.word_timestamps and not stt.capabilities.streaming
    transcript = await stt.transcribe(speech(0.7))
    assert transcript.words is not None
    assert [(w.word, w.start, w.end) for w in transcript.words] == [
        ("Hi", 0.1, 0.3),
        ("you", 0.4, 0.6),
    ]
    assert transcript.language == "english"
    fields = multipart_fields(rec.requests[0])
    assert fields["response_format"] == ["verbose_json"]
    assert fields["timestamp_granularities[]"] == ["word"]

    rec2 = Recorder(httpx.Response(200, json={"text": "Hey", "logprobs": [
        {"token": "Hey", "logprob": -0.5, "bytes": [72, 101, 121]}]}))  # fmt: skip
    stt2 = OpenAISTT(api_key="k", model="gpt-4o-transcribe", logprobs=True, http_streaming=False,
                     temperature=0.0, extra={"chunking_strategy": "auto"},
                     http_client=rec2.client())  # fmt: skip
    transcript2 = await stt2.transcribe(speech(0.3))
    assert transcript2.confidence == pytest.approx(math.exp(-0.5))
    fields2 = multipart_fields(rec2.requests[0])
    assert fields2["include[]"] == ["logprobs"] and fields2["temperature"] == ["0.0"]
    assert fields2["chunking_strategy"] == ["auto"] and "stream" not in fields2


@pytest.mark.parametrize(
    ("status", "body", "error", "retryable"),
    [
        (401, {"error": {"message": "Incorrect API key", "type": "invalid_request_error",
                         "code": "invalid_api_key"}}, AuthenticationError, False),
        (429, {"error": {"message": "Slow down", "type": "requests", "code": "rate_limit_exceeded"}},
         RateLimitError, True),
        (429, {"error": {"message": "Quota", "type": "insufficient_quota",
                         "code": "insufficient_quota"}}, RateLimitError, False),
        (404, {"error": {"message": "The model does not exist"}}, ProviderError, False),
        (500, {"error": {"message": "Internal error"}}, ProviderError, True),
        (400, {"detail": "Unsupported file"}, ProviderError, False),
        (504, "gateway timeout", ProviderTimeoutError, True),
    ],
)  # fmt: skip
async def test_batch_http_errors_are_mapped(
    status: int, body: Any, error: type[ProviderError], retryable: bool
) -> None:
    content = body.encode() if isinstance(body, str) else json.dumps(body).encode()
    rec = Recorder(httpx.Response(status, content=content))
    stt = OpenAISTT(api_key="k", model="whisper-1", http_client=rec.client())
    with pytest.raises(error) as info:
        await stt.transcribe(speech(0.2))
    assert info.value.status_code == status and info.value.retryable is retryable
    detail = body if isinstance(body, str) else json.dumps(body)
    assert any(word in str(info.value) for word in ("Incorrect", "Slow", "Quota", "model",
                                                     "Internal", "Unsupported", "gateway"))  # fmt: skip
    if status == 404:
        assert "check the model id" in str(info.value)
    assert detail


async def test_batch_network_errors_are_connection_errors() -> None:
    def fail(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    stt = OpenAISTT(api_key="k", model="whisper-1",
                    http_client=httpx.AsyncClient(transport=httpx.MockTransport(fail)))  # fmt: skip
    with pytest.raises(ProviderConnectionError, match="cannot reach"):
        await stt.transcribe(speech(0.2))


# ------------------------------------------------------------ OpenAI-compatible hosts
def test_compatible_stt_hosts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")  # never sent to these hosts
    speaches = create("stt", "speaches")
    assert isinstance(speaches, SpeachesSTT)
    assert speaches.model == "Systran/faster-distil-whisper-small.en"
    assert speaches.base_url == "http://localhost:8000/v1"
    assert not speaches.capabilities.streaming and speaches.sample_rate == 16_000
    assert speaches.endpoint.api_key is None
    monkeypatch.setenv("SPEACHES_BASE_URL", "http://gpu-box:8000/v1")
    monkeypatch.setenv("SPEACHES_API_KEY", "sp-key")
    speaches2 = create("stt", "speaches/Systran/faster-whisper-large-v3")
    assert speaches2.model == "Systran/faster-whisper-large-v3"
    assert speaches2.base_url == "http://gpu-box:8000/v1"
    assert speaches2.endpoint.request_headers() == {"Authorization": "Bearer sp-key"}

    localai = create("stt", "localai")
    assert isinstance(localai, LocalAISTT) and localai.model == "whisper-1"
    assert localai.base_url == "http://localhost:8080/v1" and not localai.capabilities.streaming


async def test_compatible_host_in_a_cascade_is_segmented_by_the_vad() -> None:
    rec = Recorder(httpx.Response(200, json={"text": "Local transcript."}))
    stt = SpeachesSTT(http_client=rec.client())
    engine = CascadeEngine(stt=stt, llm=MockLLM(responses=["Ok."]), tts=MockTTS(),
                           vad=EnergyVAD(), options=CascadeOptions(min_endpointing_delay=0.0))  # fmt: skip
    assert isinstance(engine.stt, StreamAdapter)
    conn = await engine.connect(EngineOptions())
    events: list[object] = []

    async def drain() -> None:
        async for ev in conn.events():
            events.append(ev)
            if isinstance(ev, ResponseDone):
                return

    drainer = asyncio.create_task(drain())
    audio = AudioFrame.concat([synth_speech(0.5, 16_000), AudioFrame.silence(0.8, 16_000)])
    for n in range(65):
        await conn.send_audio(audio.slice(n * 0.02, (n + 1) * 0.02))
    await asyncio.wait_for(drainer, 10)
    await conn.aclose()
    await engine.aclose()
    finals = [e.text for e in events if isinstance(e, InputTranscript) and e.is_final]
    assert finals == ["Local transcript."]
    request = rec.requests[0]
    assert str(request.url) == "http://localhost:8000/v1/audio/transcriptions"
    assert multipart_fields(request)["model"] == ["Systran/faster-distil-whisper-small.en"]
    assert "authorization" not in request.headers


async def test_speaches_warmup_loads_the_model() -> None:
    rec = Recorder(httpx.Response(200, json={"text": ""}))
    stt = SpeachesSTT(http_client=rec.client())
    await stt.warmup()
    assert [r.url.path for r in rec.requests] == ["/v1/audio/transcriptions"]


def test_azure_stt_endpoint_and_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ConfigurationError, match="AZURE_OPENAI_ENDPOINT"):
        AzureOpenAISTT(api_key="k")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://res.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "az-key")
    stt = create("stt", "azure_openai/my-transcriber")
    assert isinstance(stt, AzureOpenAISTT) and stt.model == "my-transcriber"
    assert stt.base_url == "https://res.openai.azure.com/openai/v1"
    assert stt.endpoint.request_headers() == {"api-key": "az-key"}
    assert stt.realtime_url() == (
        "wss://res.openai.azure.com/openai/v1/realtime?model=my-transcriber&intent=transcription"
    )
    token = AzureOpenAISTT(azure_ad_token="entra-token")
    assert token.endpoint.request_headers() == {"Authorization": "Bearer entra-token"}
    assert create("stt", "azure_openai").model == "gpt-4o-mini-transcribe"


# ------------------------------------------------------------------- real API (opt-in)
needs_key = pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="needs OPENAI_API_KEY")


async def _speech_sample() -> AudioFrame:
    from voice_agent_next.providers.openai.tts import OpenAITTS

    tts = OpenAITTS()
    try:
        return await tts.synthesize("The quick brown fox jumps over the lazy dog.").collect()
    finally:
        await tts.aclose()


@pytest.mark.integration
@needs_key
@pytest.mark.parametrize("model", ["gpt-live-transcribe", "gpt-transcribe"])
async def test_integration_realtime_transcription(model: str) -> None:
    audio = AudioFrame.concat([await _speech_sample(), AudioFrame.silence(0.5, 24_000)])
    stream = OpenAISTT(model=model, language="en").stream()
    for n in range(math.ceil(audio.duration / 0.1)):  # real time, 100 ms chunks
        stream.push_audio(audio.slice(n * 0.1, (n + 1) * 0.1))
        await asyncio.sleep(0.1)
    stream.end_input()
    events = [ev async for ev in stream]
    await stream.aclose()
    text = " ".join(ev.text for ev in events if is_final(ev)).lower()
    assert "fox" in text and "lazy dog" in text


@pytest.mark.integration
@needs_key
@pytest.mark.parametrize("model", ["gpt-transcribe", "whisper-1"])
async def test_integration_file_transcription(model: str) -> None:
    audio = await _speech_sample()
    stt = OpenAISTT(model=model, language="en")
    transcript = await stt.transcribe(audio)
    await stt.aclose()
    assert "fox" in transcript.text.lower()
    assert np.isfinite(audio.dbfs())
