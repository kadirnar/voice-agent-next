"""ElevenLabs TTS (multi-context WebSockets, HTTP streaming) and Scribe STT against fakes.

The fakes replay the documented message shapes (checked 2026-09-24):

* TTS ``/v1/text-to-speech/{voice_id}/multi-stream-input``: context init (``text: " "``),
  text with ``flush``, ``close_context``, keep-alive (``text: ""``); server audio chunks
  with ``alignment`` / ``normalizedAlignment`` (camelCase, times per chunk or per
  context), ``isFinal`` + ``contextId``, error payloads and 1008 policy closes;
* Text to Dialogue ``/v1/text-to-dialogue/multi-stream-input`` (Eleven v3): ``voices``
  registration, ``inputs``, ``keep_alive``, snake_case ``alignment`` / ``is_final`` /
  ``context_id``, ``is_final_audio_for_turn``;
* Scribe v2 Realtime ``/v1/speech-to-text/realtime``: ``session_started``,
  ``input_audio_chunk`` (base64, ``commit``, ``sample_rate``, ``previous_text``),
  ``partial_transcript``, ``committed_transcript`` (+ ``_with_timestamps``), error
  message types;
* HTTP: ``POST /v1/text-to-speech/{voice_id}/stream``, ``/v1/text-to-dialogue/stream``,
  ``POST /v1/speech-to-text`` (multipart) via ``httpx.MockTransport``.

Tests marked ``integration`` talk to the real API and need ``ELEVEN_API_KEY`` or
``ELEVENLABS_API_KEY``.
"""

from __future__ import annotations

import asyncio
import base64
import itertools
import json
import logging
import math
import os
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from http import HTTPStatus
from typing import Any
from urllib.parse import parse_qs, urlsplit

import httpx
import numpy as np
import pytest
from websockets.asyncio.server import ServerConnection, serve
from websockets.exceptions import ConnectionClosed
from websockets.http11 import Request, Response

from voice_agent_next import AudioFrame, create
from voice_agent_next.engine import EngineOptions
from voice_agent_next.engines.cascade import CascadeEngine
from voice_agent_next.errors import (
    AuthenticationError,
    ConfigurationError,
    ProviderConnectionError,
    ProviderError,
    ProviderTimeoutError,
    RateLimitError,
)
from voice_agent_next.events import (
    InputCommitted,
    InputTranscript,
    ResponseAudio,
    ResponseDone,
    ResponseText,
)
from voice_agent_next.metrics import STTMetrics, TTSMetrics
from voice_agent_next.providers.elevenlabs import (
    DEFAULT_VOICE,
    MAX_CONTEXTS,
    REGIONS,
    ElevenLabsSTT,
    ElevenLabsTTS,
)
from voice_agent_next.providers.energy import EnergyVAD
from voice_agent_next.providers.mock import MockLLM, MockSTT, MockTTS, synth_speech
from voice_agent_next.registry import get_provider
from voice_agent_next.stt import STTEvent, STTEventType, STTStream
from voice_agent_next.tts import SentenceStreamAdapter, SynthesizedAudio, SynthesizeStream

KEY = "sk_test_eleven"
WORD = 0.1
"""Seconds of audio the fake TTS generates per word."""
T = STTEventType

_PROXY_VARS = ("http_proxy", "https_proxy", "all_proxy", "ws_proxy", "wss_proxy")


@pytest.fixture(autouse=True)
def _no_proxy_no_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """websockets honours proxy variables (the fakes live on 127.0.0.1); keys come from
    the tests, never from the developer's environment."""
    for var in _PROXY_VARS:
        monkeypatch.delenv(var, raising=False)
        monkeypatch.delenv(var.upper(), raising=False)
    for var in ("ELEVEN_API_KEY", "ELEVENLABS_API_KEY"):
        monkeypatch.delenv(var, raising=False)


def pcm(value: int, seconds: float, rate: int) -> bytes:
    return np.full(round(seconds * rate), value, dtype="<i2").tobytes()


async def wait_until(predicate: Callable[[], object], timeout: float = 5.0) -> None:
    async def poll() -> None:
        while not predicate():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(poll(), timeout)


async def collect_tts(stream: SynthesizeStream, timeout: float = 5.0) -> list[SynthesizedAudio]:
    async def run() -> list[SynthesizedAudio]:
        return [item async for item in stream]

    return await asyncio.wait_for(run(), timeout)


def audio_of(items: list[SynthesizedAudio]) -> AudioFrame:
    return AudioFrame.concat([i.frame for i in items if i.frame])


def words_of(items: list[SynthesizedAudio]) -> list[Any]:
    return [w for i in items for w in i.words or []]


def query(request: Request) -> dict[str, list[str]]:
    return parse_qs(urlsplit(request.path).query)


def path(request: Request) -> str:
    return urlsplit(request.path).path


# ---------------------------------------------------------------------------- fake TTS
@dataclass
class _FakeContext:
    index: int
    queue: asyncio.Queue[tuple[str, str]] = field(default_factory=asyncio.Queue)
    buffer: str = ""
    elapsed: float = 0.0
    """Seconds of audio sent for this context."""
    aligned: bool = False
    task: asyncio.Task[None] | None = None


@dataclass
class FakeTTSServer:
    """Replays ElevenLabs' multi-context WebSockets (text-to-speech or text-to-dialogue).

    Every word of flushed text becomes one audio message of ``WORD`` seconds (sample
    value ``1000 * (context index + 1)``) aligned character by character; with
    ``split_words`` each word spans two messages. ``close_context`` drops unflushed
    text and ends the context with its final message once generation is done.
    """

    dialogue: bool = False
    time_base: str = "chunk"
    """``chunk``: alignment times restart in every message; ``context``: they continue."""
    split_words: bool = False
    normalized_upper: bool = False
    """``normalizedAlignment`` spells words in upper case (tells the two apart)."""
    delays: dict[int, float] = field(default_factory=dict)
    """Context index -> seconds to wait before its first generation."""
    generation_delay: float = 0.0
    expire_after_first: bool = False
    """End every context after its first generation, as an inactivity expiry would."""
    never_final: bool = False
    error: dict[str, Any] | None = None
    """Sent (with the context id when ``error_with_context``) instead of audio."""
    error_with_context: bool = True
    close_after_final: tuple[int, str] | None = None
    """Close the socket with this code and reason after a final message."""
    close_on_first_message: tuple[int, str] | None = None
    reject_status: int | None = None
    url: str = ""
    handshakes: int = 0
    requests: list[Request] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)
    inits: list[dict[str, Any]] = field(default_factory=list)
    contexts: list[str] = field(default_factory=list)
    closed: list[str] = field(default_factory=list)
    finals: list[str] = field(default_factory=list)
    keepalives: list[str] = field(default_factory=list)
    open_now: int = 0
    max_open: int = 0

    def texts(self) -> list[tuple[str, str, bool]]:
        """``(context id, text, flush)`` of every text input."""
        out = []
        for m in self.messages:
            if self.dialogue and "inputs" in m:
                text = "".join(i["text"] for i in m["inputs"])
            elif not self.dialogue and m.get("text") and m["text"] != " ":
                text = m["text"]
            else:
                continue
            out.append((m["context_id"], text, bool(m.get("flush"))))
        return out

    def process_request(self, conn: ServerConnection, request: Request) -> Response | None:
        self.handshakes += 1
        self.requests.append(request)
        if self.reject_status is not None:
            body = json.dumps(
                {"detail": {"status": "invalid_api_key", "message": "Invalid API key"}}
            )
            return conn.respond(HTTPStatus(self.reject_status), body)
        return None

    async def handler(self, ws: ServerConnection) -> None:
        assert ws.request is not None
        rate = int(query(ws.request)["output_format"][0].removeprefix("pcm_"))
        states: dict[str, _FakeContext] = {}
        try:
            async for raw in ws:
                msg = json.loads(raw)
                self.messages.append(msg)
                if self.close_on_first_message is not None:
                    code, reason = self.close_on_first_message
                    await ws.close(code, reason)
                    return
                cid = msg["context_id"]
                state = states.get(cid)
                if state is None:  # the first message of a context initializes it
                    state = states[cid] = _FakeContext(index=len(self.contexts))
                    self.contexts.append(cid)
                    self.inits.append(msg)
                    self.open_now += 1
                    self.max_open = max(self.max_open, self.open_now)
                    state.task = asyncio.create_task(self._run_context(ws, cid, state, rate))
                    continue
                if msg.get("close_context"):
                    self.closed.append(cid)
                    state.queue.put_nowait(("final", ""))
                    continue
                if self.dialogue:
                    if msg.get("keep_alive"):
                        self.keepalives.append(cid)
                        continue
                    text = "".join(i["text"] for i in msg.get("inputs") or [])
                else:
                    text = msg.get("text", "")
                    if text == "" and not msg.get("flush"):
                        self.keepalives.append(cid)
                        continue
                state.buffer += text
                if msg.get("flush"):
                    state.queue.put_nowait(("generate", state.buffer))
                    state.buffer = ""
        except ConnectionClosed:
            pass
        finally:
            tasks = [s.task for s in states.values() if s.task is not None]
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _run_context(
        self, ws: ServerConnection, cid: str, state: _FakeContext, rate: int
    ) -> None:
        first = True
        while True:
            kind, text = await state.queue.get()
            if kind == "final":
                if not self.never_final:
                    await self._final(ws, cid)
                return
            if first and self.delays.get(state.index):
                await asyncio.sleep(self.delays[state.index])
            first = False
            if self.error is not None:
                payload = dict(self.error)
                if self.error_with_context:
                    payload["contextId" if not self.dialogue else "context_id"] = cid
                await ws.send(json.dumps(payload))
                return
            await self._generate(ws, cid, state, text, rate)
            if self.expire_after_first:
                await self._final(ws, cid)
                return

    async def _final(self, ws: ServerConnection, cid: str) -> None:
        if self.dialogue:
            await ws.send(json.dumps({"is_final": True, "context_id": cid}))
        else:
            await ws.send(json.dumps({"isFinal": True, "contextId": cid}))
        self.finals.append(cid)
        self.open_now -= 1
        if self.close_after_final is not None:
            await ws.close(*self.close_after_final)

    async def _generate(
        self, ws: ServerConnection, cid: str, state: _FakeContext, text: str, rate: int
    ) -> None:
        value = 1000 * (state.index + 1)
        for word in text.split():
            if self.generation_delay:
                await asyncio.sleep(self.generation_delay)
            pieces = [word[:1], word[1:]] if self.split_words and len(word) > 1 else [word]
            for n, piece in enumerate(pieces):
                chars = list(piece) + ([" "] if n == len(pieces) - 1 else [])
                duration = WORD / len(pieces)
                step = duration * 1000 / len(chars)
                starts = [round(i * step) for i in range(len(chars))]
                if self.time_base == "context":
                    starts = [s + round(state.elapsed * 1000) for s in starts]
                durations = [round(step)] * len(chars)
                data = base64.b64encode(pcm(value, duration, rate)).decode()
                state.elapsed += duration
                if self.dialogue:
                    msg: dict[str, Any] = {
                        "audio": data,
                        "alignment": {"chars": chars, "char_start_times_ms": starts,
                                      "char_durations_ms": durations},
                        "normalized_alignment": None,
                        "context_id": cid,
                    }  # fmt: skip
                else:
                    normalized = [c.upper() for c in chars] if self.normalized_upper else chars
                    n_starts, n_durations = starts, durations
                    if not state.aligned:  # normalized text starts with a space
                        normalized = [" ", *normalized]
                        n_starts, n_durations = [starts[0], *starts], [0, *durations]
                    msg = {
                        "audio": data,
                        "alignment": {"chars": chars, "charStartTimesMs": starts,
                                      "charDurationsMs": durations},
                        "normalizedAlignment": {"chars": normalized,
                                                "charStartTimesMs": n_starts,
                                                "charDurationsMs": n_durations},
                        "contextId": cid,
                    }  # fmt: skip
                state.aligned = True
                await ws.send(json.dumps(msg))
        if self.dialogue:
            await ws.send(json.dumps({"is_final_audio_for_turn": True, "context_id": cid}))


@pytest.fixture
async def tts_server() -> AsyncIterator[FakeTTSServer]:
    fake = FakeTTSServer()
    async with serve(fake.handler, "127.0.0.1", 0, process_request=fake.process_request) as srv:
        port = next(iter(srv.sockets)).getsockname()[1]
        fake.url = f"http://127.0.0.1:{port}"
        yield fake


def make_tts(server: FakeTTSServer, **kw: Any) -> ElevenLabsTTS:
    return ElevenLabsTTS(api_key=KEY, base_url=server.url, **kw)


# --------------------------------------------------------------------------- TTS tests
async def test_stream_flushes_sentences_on_one_context(tts_server: FakeTTSServer) -> None:
    tts = make_tts(tts_server)
    metrics: list[TTSMetrics] = []
    tts.on("metrics", metrics.append)
    stream = tts.stream()
    stream.push_text("Hello there. ")
    stream.push_text("How are you? ")
    stream.end_input()
    items = await collect_tts(stream)
    await tts.aclose()

    (ctx,) = tts_server.contexts
    assert tts_server.inits == [{"text": " ", "context_id": ctx}]
    assert tts_server.texts() == [(ctx, "Hello there. ", True), (ctx, "How are you? ", True)]
    # the last text was flushed already, so the segment ends with close_context alone
    assert tts_server.messages[-1] == {"context_id": ctx, "close_context": True}
    assert tts_server.closed == [ctx]

    request = tts_server.requests[0]
    assert path(request) == f"/v1/text-to-speech/{DEFAULT_VOICE}/multi-stream-input"
    assert query(request) == {
        "model_id": ["eleven_flash_v2_5"],
        "output_format": ["pcm_24000"],
        "inactivity_timeout": ["180"],
        "auto_mode": ["true"],
        "sync_alignment": ["true"],
    }
    assert request.headers["xi-api-key"] == KEY
    assert KEY not in request.path

    audio = audio_of(items)
    assert audio.sample_rate == 24_000
    assert audio.duration == pytest.approx(5 * WORD)
    assert items[-1].is_final and not items[-1].frame
    assert items[-1].text == "Hello there. How are you?"
    assert not any(i.is_final for i in items[:-1])
    assert metrics and metrics[-1].streamed and not metrics[-1].cancelled
    assert metrics[-1].ttfb is not None and metrics[-1].audio_duration == pytest.approx(0.5)


@pytest.mark.parametrize("time_base", ["chunk", "context"])
@pytest.mark.parametrize("split_words", [False, True])
async def test_word_timings_are_on_the_stream_timeline(
    tts_server: FakeTTSServer, time_base: str, split_words: bool
) -> None:
    tts_server.time_base = time_base
    tts_server.split_words = split_words
    tts = make_tts(tts_server)
    stream = tts.stream()
    stream.push_text("One two. ")
    stream.flush()  # second segment: a new context whose times restart
    stream.push_text("Three four five six seven. ")
    stream.end_input()
    items = await collect_tts(stream)
    await tts.aclose()

    words = words_of(items)
    assert [w.word for w in words] == ["One", "two.", "Three", "four", "five", "six", "seven."]
    assert [w.start for w in words] == pytest.approx([n * WORD for n in range(7)], abs=0.002)
    assert all(w.start < w.end <= w.start + WORD + 0.002 for w in words)
    assert words[-1].end <= audio_of(items).duration
    word_items = [i for i in items if i.words]
    assert all(not i.frame and not i.is_final for i in word_items)
    # a word is reported before (or with) the audio it starts in
    for item in word_items:
        before = sum(i.frame.duration for i in items[: items.index(item)] if i.frame)
        assert all(w.start >= before - WORD - 1e-6 for w in item.words or [])


async def test_time_base_detection_is_kept_across_streams(tts_server: FakeTTSServer) -> None:
    tts_server.time_base = "context"
    tts = make_tts(tts_server)
    for _ in range(2):
        stream = tts.stream()
        stream.push_text("one two three four five six seven eight. ")
        stream.end_input()
        words = words_of(await collect_tts(stream))
        assert [w.start for w in words] == pytest.approx([n * WORD for n in range(8)], abs=0.002)
    await tts.aclose()
    assert tts._time_bases == {"tts/original": "context"}


@pytest.mark.parametrize(("alignment", "expected"), [(None, "Hello"), ("normalized", "HELLO")])
async def test_alignment_choice(
    tts_server: FakeTTSServer, alignment: str | None, expected: str
) -> None:
    tts_server.normalized_upper = True
    tts = make_tts(tts_server, alignment=alignment)
    stream = tts.stream()
    stream.push_text("Hello world. ")
    stream.end_input()
    words = words_of(await collect_tts(stream))
    await tts.aclose()
    assert words[0].word == expected and words[0].start == pytest.approx(0.0)
    assert len(words) == 2  # the leading space of the normalized text is no word


async def test_raw_tokens_are_sent_at_word_boundaries(tts_server: FakeTTSServer) -> None:
    tts = make_tts(tts_server)
    stream = tts.stream()
    for token in ["Hel", "lo", " wor", "ld.", " How", " are", " you?"]:
        stream.push_text(token)
    stream.end_input()
    items = await collect_tts(stream)
    await tts.aclose()

    texts = [(text, flush) for _, text, flush in tts_server.texts()]
    assert texts == [
        ("Hello ", False),
        ("world. ", False),  # a sentence end in the middle of a push is not flushed
        ("How ", False),
        ("are ", False),
        ("you? ", True),  # the end of the input flushes what is left
    ]
    assert tts_server.messages[-1]["close_context"] is True
    assert audio_of(items).duration == pytest.approx(5 * WORD)


async def test_first_clause_is_flushed_and_the_rest_at_the_end(
    tts_server: FakeTTSServer,
) -> None:
    tts = make_tts(tts_server)
    stream = tts.stream()
    stream.push_text("Sure, ")  # the cascade's short first chunk
    stream.push_text("I can help with that. ")
    stream.push_text("Later, maybe ")
    stream.end_input()
    items = await collect_tts(stream)
    await tts.aclose()

    (ctx,) = tts_server.contexts
    assert [(t, f) for _, t, f in tts_server.texts()] == [
        ("Sure, ", True),
        ("I can help with that. ", True),
        ("Later, maybe ", False),  # only the first clause of a segment is flushed
    ]
    assert tts_server.messages[-2:] == [
        {"context_id": ctx, "flush": True},  # generate the unflushed rest...
        {"context_id": ctx, "close_context": True},  # ...then finish the context
    ]
    assert audio_of(items).duration == pytest.approx(8 * WORD)


async def test_flush_sentences_can_be_disabled(tts_server: FakeTTSServer) -> None:
    tts = make_tts(tts_server, flush_sentences=False, chunk_length_schedule=[50, 120])
    assert not tts.auto_mode  # a schedule turns auto mode off unless asked
    stream = tts.stream()
    stream.push_text("Hello there. ")
    stream.end_input()
    await collect_tts(stream)
    await tts.aclose()
    assert [f for _, _, f in tts_server.texts()] == [False]
    assert tts_server.inits[0]["generation_config"] == {"chunk_length_schedule": [50, 120]}
    assert "auto_mode" not in query(tts_server.requests[0])


async def test_voice_settings_and_options(tts_server: FakeTTSServer) -> None:
    tts = make_tts(
        tts_server,
        voice="voice_custom",
        sample_rate=16_000,
        language="pt-BR",
        stability=0.4,
        similarity_boost=0.8,
        style=0.1,
        use_speaker_boost=False,
        speed=1.1,
        apply_text_normalization="on",
        seed=7,
        pronunciation_dictionaries=[("dict_1", "v2")],
        enable_logging=False,
        enable_ssml_parsing=True,
        inactivity_timeout=60,
        word_timestamps=False,
    )
    assert tts.alignment == "normalized"  # dictionaries make the original alignment unreliable
    stream = tts.stream()
    stream.push_text("Olá a todos. ")
    stream.end_input()
    items = await collect_tts(stream)
    await tts.aclose()

    request = tts_server.requests[0]
    assert path(request) == "/v1/text-to-speech/voice_custom/multi-stream-input"
    assert query(request) == {
        "model_id": ["eleven_flash_v2_5"],
        "output_format": ["pcm_16000"],
        "inactivity_timeout": ["60"],
        "auto_mode": ["true"],
        "language_code": ["pt"],
        "apply_text_normalization": ["on"],
        "seed": ["7"],
        "enable_logging": ["false"],
        "enable_ssml_parsing": ["true"],
    }
    assert tts_server.inits[0] == {
        "text": " ",
        "context_id": tts_server.contexts[0],
        "voice_settings": {"stability": 0.4, "similarity_boost": 0.8, "style": 0.1,
                           "use_speaker_boost": False, "speed": 1.1},
        "pronunciation_dictionary_locators": [
            {"pronunciation_dictionary_id": "dict_1", "version_id": "v2"}
        ],
    }  # fmt: skip
    assert not words_of(items)  # alignment ignored without word_timestamps
    assert audio_of(items).sample_rate == 16_000


async def test_language_is_only_sent_to_models_that_take_it(
    tts_server: FakeTTSServer, caplog: pytest.LogCaptureFixture
) -> None:
    flash = make_tts(tts_server, language="de-DE")
    assert "language_code=de" in flash.ws_url()
    with caplog.at_level(logging.WARNING, logger="voice_agent_next"):
        multilingual = make_tts(tts_server, model="eleven_multilingual_v2", language="de")
    assert "language_code" not in multilingual.ws_url()
    assert "takes no language code" in caplog.text


async def test_flush_opens_a_new_context_and_segments_play_in_order(
    tts_server: FakeTTSServer,
) -> None:
    tts_server.delays = {0: 0.3}  # the second context's audio arrives before the first one's
    tts = make_tts(tts_server)
    stream = tts.stream()
    stream.push_text("One two. ")
    stream.flush()
    stream.push_text("Three. ")
    stream.end_input()
    items = await collect_tts(stream)
    await tts.aclose()

    first, second = tts_server.contexts
    assert [(c, t) for c, t, _ in tts_server.texts()] == [(first, "One two. "), (second, "Three. ")]
    assert tts_server.closed == [first, second]
    finals = [i for i in items if i.is_final]
    assert [f.text for f in finals] == ["One two.", "Three."]
    assert finals[0].segment_id != finals[1].segment_id
    samples = np.concatenate([i.frame.to_numpy() for i in items if i.frame])
    assert samples.tolist() == [1000] * round(2 * WORD * 24_000) + [2000] * round(WORD * 24_000)
    words = words_of(items)
    assert [w.start for w in words] == pytest.approx([0.0, WORD, 2 * WORD], abs=0.002)


async def test_context_limit_waits_for_a_free_slot(tts_server: FakeTTSServer) -> None:
    tts_server.generation_delay = 0.05
    tts = make_tts(tts_server)
    stream = tts.stream()
    for n in range(MAX_CONTEXTS + 3):
        stream.push_text(f"Sentence {n}. ")
        stream.flush()
    stream.end_input()
    items = await collect_tts(stream, timeout=10)
    await tts.aclose()
    assert len(tts_server.contexts) == MAX_CONTEXTS + 3
    assert tts_server.max_open <= MAX_CONTEXTS
    finals = [i.text for i in items if i.is_final]
    assert finals[:-1] == [f"Sentence {n}." for n in range(MAX_CONTEXTS + 3)]
    assert finals[-1] is None  # end_input() after a flush: an empty last segment


async def test_concurrent_streams_share_the_socket(tts_server: FakeTTSServer) -> None:
    tts_server.delays = {0: 0.2}  # interleave the two contexts on the wire
    tts = make_tts(tts_server)
    streams = [tts.stream(), tts.stream()]
    for stream, text in zip(streams, ("One two three. ", "Four five. "), strict=True):
        stream.push_text(text)
        stream.end_input()
    results = await asyncio.gather(*(collect_tts(s) for s in streams))
    await tts.aclose()

    assert tts_server.handshakes == 1
    assert [audio_of(r).duration for r in results] == pytest.approx([3 * WORD, 2 * WORD])
    values = [set(audio_of(r).to_numpy().tolist()) for r in results]
    assert len(values[0]) == len(values[1]) == 1 and values[0] != values[1]


async def test_aclose_closes_the_open_context_and_keeps_the_socket(
    tts_server: FakeTTSServer,
) -> None:
    tts_server.generation_delay = 0.2  # still generating when the stream is closed
    tts = make_tts(tts_server)
    metrics: list[TTSMetrics] = []
    tts.on("metrics", metrics.append)
    stream = tts.stream()
    stream.push_text("Once upon a time there was a voice. ")
    first = await asyncio.wait_for(anext(stream), 5)
    assert first.frame or first.words
    await stream.aclose()
    ctx = tts_server.contexts[0]
    await wait_until(lambda: tts_server.closed)
    assert tts_server.closed == [ctx]
    # an interrupted context is closed without a final flush
    assert not any(m.get("flush") and "text" not in m for m in tts_server.messages)
    assert metrics[-1].cancelled

    # the late audio of the closed context is dropped; the socket serves the next reply
    tts_server.generation_delay = 0.0
    stream = tts.stream()
    stream.push_text("Hi again. ")
    stream.end_input()
    items = await collect_tts(stream)
    assert audio_of(items).duration == pytest.approx(2 * WORD)
    assert set(audio_of(items).to_numpy().tolist()) == {2000}
    assert tts_server.handshakes == 1
    await tts.aclose()


async def test_interrupted_cascade_reply_closes_the_context(tts_server: FakeTTSServer) -> None:
    tts_server.delays = {0: 5.0}  # generation never finishes during the test
    tts = make_tts(tts_server)
    engine = CascadeEngine(stt=MockSTT(), llm=MockLLM(), tts=tts, vad=EnergyVAD())
    conn = await engine.connect(EngineOptions())
    await conn.say("This reply will be interrupted. It keeps going.")
    await wait_until(lambda: tts_server.closed)  # say(): the whole text is in, the context closes
    await conn.cancel_response()
    await asyncio.sleep(0.2)
    # a closing context is not messaged again (a protocol error on the dialogue API)
    assert tts_server.closed == tts_server.contexts[:1]
    assert len(tts_server.contexts) == 1
    await conn.aclose()
    await engine.aclose()


async def test_cascade_speaks_and_truncates_at_the_words_heard(
    tts_server: FakeTTSServer,
) -> None:
    tts = make_tts(tts_server)
    engine = CascadeEngine(stt=MockSTT(), llm=MockLLM(), tts=tts, vad=EnergyVAD())
    await engine.warmup()
    conn = await engine.connect(EngineOptions())
    await conn.say("Hello there. How are you today?")
    events: list[object] = []

    async def drain() -> None:
        async for ev in conn.events():
            events.append(ev)
            if isinstance(ev, ResponseDone):
                return

    await asyncio.wait_for(drain(), 5)
    done = events[-1]
    assert isinstance(done, ResponseDone) and done.status == "completed"
    audio = [e for e in events if isinstance(e, ResponseAudio)]
    assert AudioFrame.concat([e.frame for e in audio]).duration == pytest.approx(6 * WORD)
    spoken = "".join(e.delta for e in events if isinstance(e, ResponseText))
    assert spoken.split() == ["Hello", "there.", "How", "are", "you", "today?"]
    assert [t for _, t, _ in tts_server.texts()] == ["Hello there. ", "How are you today? "]
    assert tts_server.handshakes == 1  # warmup() opened the socket the reply used

    # barge-in after 0.25 s of playback: the user heard the first three words
    heard = await conn.truncate(audio[0].item_id, 250)
    assert heard == "Hello there. How"
    await conn.aclose()
    await engine.aclose()


@pytest.mark.parametrize(
    ("payload", "error_type", "status", "retryable"),
    [
        ({"error": "invalid_api_key", "message": "Invalid API key", "code": 401},
         AuthenticationError, 401, False),
        ({"error": "too_many_concurrent_requests", "message": "Too many requests", "code": 429},
         RateLimitError, 429, True),
        ({"error": "quota_exceeded", "message": "This request exceeds your quota", "code": 401},
         RateLimitError, 401, False),
        ({"error": "voice_not_found", "message": "A voice with this id does not exist",
          "code": 1008}, ProviderError, 1008, False),
    ],
)  # fmt: skip
@pytest.mark.parametrize("with_context", [True, False])
async def test_error_messages_are_mapped(
    tts_server: FakeTTSServer,
    payload: dict[str, Any],
    error_type: type[ProviderError],
    status: int,
    retryable: bool,
    with_context: bool,
) -> None:
    tts_server.error = payload
    tts_server.error_with_context = with_context
    tts = make_tts(tts_server)
    stream = tts.stream()
    stream.push_text("Hi. ")
    stream.end_input()
    with pytest.raises(error_type) as info:
        await collect_tts(stream)
    assert type(info.value) is error_type
    assert info.value.status_code == status and info.value.retryable is retryable
    assert info.value.provider == "elevenlabs" and payload["message"] in str(info.value)
    await stream.aclose()

    # a context error leaves the socket usable; an error without a context retires it
    tts_server.error = None
    stream = tts.stream()
    stream.push_text("Hi. ")
    stream.end_input()
    assert audio_of(await collect_tts(stream)).duration == pytest.approx(WORD)
    assert tts_server.handshakes == (1 if with_context else 2)
    await tts.aclose()


async def test_rejected_handshake_is_an_authentication_error(tts_server: FakeTTSServer) -> None:
    tts_server.reject_status = 401
    tts = make_tts(tts_server)
    with pytest.raises(AuthenticationError, match="Invalid API key"):
        await tts.warmup()
    stream = tts.stream()
    stream.push_text("Hi. ")
    stream.end_input()
    with pytest.raises(AuthenticationError):
        await collect_tts(stream)
    await tts.aclose()


@pytest.mark.parametrize(
    ("close", "error_type"),
    [
        ((1008, "Invalid API key"), AuthenticationError),
        ((1008, "You have exceeded your quota"), RateLimitError),
        ((1008, "Unsupported model"), ProviderError),
        ((1011, "internal error"), ProviderConnectionError),
    ],
)
async def test_policy_closes_are_mapped(
    tts_server: FakeTTSServer, close: tuple[int, str], error_type: type[ProviderError]
) -> None:
    tts_server.close_on_first_message = close
    tts = make_tts(tts_server)
    stream = tts.stream()
    stream.push_text("Hi. ")
    stream.end_input()
    with pytest.raises(error_type) as info:
        await collect_tts(stream)
    assert type(info.value) is error_type
    assert info.value.status_code == close[0] and close[1] in str(info.value)
    await stream.aclose()
    await tts.aclose()


@pytest.mark.parametrize("close", [(1000, ""), (1008, "Inactivity timeout reached")])
async def test_connection_is_reopened_after_the_server_closes_it(
    tts_server: FakeTTSServer, close: tuple[int, str]
) -> None:
    tts = make_tts(tts_server)
    for text in ("First. ", "Second. "):
        stream = tts.stream()
        stream.push_text(text)
        stream.end_input()
        assert audio_of(await collect_tts(stream)).duration == pytest.approx(WORD)
    assert tts_server.handshakes == 1

    tts_server.close_after_final = close  # e.g. the inactivity timeout of an idle socket
    for text in ("Third. ", "Fourth. "):
        stream = tts.stream()
        stream.push_text(text)
        stream.end_input()
        assert audio_of(await collect_tts(stream)).duration == pytest.approx(WORD)
        await wait_until(lambda: all(c.closed for c in tts._conns.values()))
    assert tts_server.handshakes == 2
    await tts.aclose()


@pytest.mark.parametrize(
    ("close", "error_type"),
    [
        ((1008, "Invalid API key"), AuthenticationError),
        ((1008, "Inactivity timeout reached"), ProviderConnectionError),
    ],
)
async def test_sends_on_a_closed_socket_report_the_reason(
    tts_server: FakeTTSServer, close: tuple[int, str], error_type: type[ProviderError]
) -> None:
    """A rejection must not look like a dropped socket (which streams reconnect after)."""
    tts_server.close_on_first_message = close
    tts = make_tts(tts_server)
    conn = await tts._connection(DEFAULT_VOICE)
    await conn.send({"text": " ", "context_id": "ctx_1"})
    await wait_until(lambda: conn.closed and conn.error is not None)
    with pytest.raises(error_type) as info:
        await conn.send({"text": "Hello ", "context_id": "ctx_1"})
    assert type(info.value) is error_type and info.value.status_code == 1008
    await tts.aclose()


async def test_context_ended_by_the_server_continues_on_a_new_context(
    tts_server: FakeTTSServer,
) -> None:
    tts_server.expire_after_first = True
    tts = make_tts(tts_server)
    stream = tts.stream()
    stream.push_text("Hello there. ")
    await wait_until(lambda: tts_server.finals)
    await wait_until(lambda: stream._contexts and stream._contexts[0].ended)
    stream.push_text("Still here. ")
    stream.end_input()
    items = await collect_tts(stream)
    await tts.aclose()

    texts = tts_server.texts()
    assert [t for _, t, _ in texts] == ["Hello there. ", "Still here. "]
    assert texts[0][0] != texts[1][0]
    assert [f.text for f in items if f.is_final] == ["Hello there. Still here."]
    assert audio_of(items).duration == pytest.approx(4 * WORD)
    assert [w.start for w in words_of(items)] == pytest.approx([0.0, 0.1, 0.2, 0.3], abs=0.002)


async def test_watchdog_fails_when_the_final_never_arrives(tts_server: FakeTTSServer) -> None:
    tts_server.never_final = True
    tts = make_tts(tts_server, receive_timeout=0.3)
    stream = tts.stream()
    stream.push_text("Hello. ")
    stream.end_input()
    with pytest.raises(ProviderTimeoutError):
        await collect_tts(stream)
    await stream.aclose()
    await tts.aclose()


async def test_empty_flush_ends_an_empty_segment(tts_server: FakeTTSServer) -> None:
    tts = make_tts(tts_server)
    stream = tts.stream()
    stream.flush()
    stream.push_text("Hi. ")
    stream.end_input()
    items = await collect_tts(stream)
    await tts.aclose()
    assert [f.text for f in items if f.is_final] == [None, "Hi."]
    assert len(tts_server.contexts) == 1


async def test_contexts_waiting_for_text_are_kept_alive(tts_server: FakeTTSServer) -> None:
    tts = make_tts(tts_server, keepalive_interval=0.1)
    stream = tts.stream()
    stream.push_text("Let me check ")  # no sentence end: the context waits for more text
    await wait_until(lambda: len(tts_server.keepalives) >= 2)
    ctx = tts_server.contexts[0]
    assert set(tts_server.keepalives) == {ctx}
    assert {"context_id": ctx, "text": ""} in tts_server.messages
    stream.push_text("that for you. ")
    stream.end_input()
    items = await collect_tts(stream)
    await tts.aclose()
    assert audio_of(items).duration == pytest.approx(6 * WORD)


async def test_eleven_v3_streams_through_the_dialogue_websocket(
    caplog: pytest.LogCaptureFixture,
) -> None:
    fake = FakeTTSServer(dialogue=True)
    async with serve(fake.handler, "127.0.0.1", 0, process_request=fake.process_request) as srv:
        fake.url = f"http://127.0.0.1:{next(iter(srv.sockets)).getsockname()[1]}"
        with caplog.at_level(logging.WARNING, logger="voice_agent_next"):
            tts = make_tts(
                fake,
                model="eleven_v3_conversational",
                voice="voice_v3",
                stability=0.4,
                speed=1.1,
                language="en",
                keepalive_interval=0.1,
            )
        assert "ignores: speed" in caplog.text
        stream = tts.stream()
        stream.push_text("Let me see ")
        await wait_until(lambda: fake.keepalives)
        stream.push_text("what I can do. ")
        stream.end_input()
        items = await collect_tts(stream)
        await tts.aclose()

    (ctx,) = fake.contexts
    request = fake.requests[0]
    assert path(request) == "/v1/text-to-dialogue/multi-stream-input"
    assert query(request) == {
        "model_id": ["eleven_v3_conversational"],
        "output_format": ["pcm_24000"],
        "sync_alignment": ["true"],
        "language_code": ["en"],
    }
    assert fake.inits == [
        {"context_id": ctx, "voices": ["voice_v3"], "voice_settings": {"stability": 0.4}}
    ]
    inputs = [m for m in fake.messages if "inputs" in m]
    assert inputs[0] == {
        "context_id": ctx,
        "inputs": [{"text": "Let me see ", "voice_id": "voice_v3"}],
    }
    assert inputs[1]["flush"] is True
    assert {"context_id": ctx, "keep_alive": True} in fake.messages
    assert fake.messages[-1] == {"context_id": ctx, "close_context": True}
    words = words_of(items)
    assert [w.word for w in words] == ["Let", "me", "see", "what", "I", "can", "do."]
    assert [w.start for w in words] == pytest.approx([n * WORD for n in range(7)], abs=0.002)
    assert items[-1].is_final and items[-1].text == "Let me see what I can do."


# ------------------------------------------------------------------------ HTTP streaming
def mock_client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_synthesize_streams_pcm_over_http() -> None:
    seen: list[httpx.Request] = []
    body = pcm(1234, 0.25, 24_000)

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, content=body, headers={"content-type": "audio/pcm"})

    client = mock_client(handler)
    tts = ElevenLabsTTS(
        api_key=KEY,
        http_client=client,
        voice="voice_1",
        language="en",
        stability=0.5,
        speed=1.2,
        apply_text_normalization="on",
        seed=7,
    )
    items = [item async for item in tts.synthesize("Hello **world**!")]
    await tts.aclose()
    assert not client.is_closed  # injected clients belong to the caller
    await client.aclose()

    assert audio_of(items).duration == pytest.approx(0.25)
    assert items[0].text == "Hello world!" and items[-1].is_final
    request = seen[0]
    assert request.method == "POST"
    assert request.url.path == "/v1/text-to-speech/voice_1/stream"
    assert request.url.host == "api.elevenlabs.io"
    assert dict(request.url.params) == {"output_format": "pcm_24000"}
    assert request.headers["xi-api-key"] == KEY
    assert json.loads(request.content) == {
        "text": "Hello world!",
        "model_id": "eleven_flash_v2_5",
        "voice_settings": {"stability": 0.5, "speed": 1.2},
        "language_code": "en",
        "apply_text_normalization": "on",
        "seed": 7,
    }


async def test_synthesize_uses_the_dialogue_endpoint_for_v3() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, content=pcm(1, 0.1, 16_000))

    client = mock_client(handler)
    tts = ElevenLabsTTS(
        api_key=KEY, http_client=client, model="eleven_v3", sample_rate=16_000, stability=0.3,
        region="eu", enable_logging=False,
    )  # fmt: skip
    audio = await tts.synthesize("[laughs] Hello!").collect()
    await client.aclose()
    assert audio.duration == pytest.approx(0.1)
    request = seen[0]
    assert str(request.url).startswith(f"{REGIONS['eu']}/v1/text-to-dialogue/stream?")
    assert dict(request.url.params) == {"output_format": "pcm_16000", "enable_logging": "false"}
    assert json.loads(request.content) == {
        "inputs": [{"text": "[laughs] Hello!", "voice_id": DEFAULT_VOICE}],
        "model_id": "eleven_v3",
        "settings": {"stability": 0.3},
    }


@pytest.mark.parametrize(
    ("status", "body", "error_type", "retryable", "fragment"),
    [
        (401, {"detail": {"type": "authentication_error", "code": "invalid_api_key",
                          "message": "Invalid API key"}}, AuthenticationError, False,
         "invalid_api_key: Invalid API key"),
        (401, {"detail": {"status": "quota_exceeded", "message": "Quota exceeded"}},
         RateLimitError, False, "quota_exceeded"),
        (429, {"detail": {"code": "concurrent_limit_exceeded", "message": "Too many"}},
         RateLimitError, True, "concurrent_limit_exceeded"),
        (422, {"detail": [{"loc": ["body", "text"], "msg": "field required",
                           "type": "missing"}]}, ProviderError, False, "body.text: field required"),
        (500, {"detail": "Internal error"}, ProviderError, True, "Internal error"),
    ],
)  # fmt: skip
async def test_synthesize_maps_http_errors(
    status: int,
    body: dict[str, Any],
    error_type: type[ProviderError],
    retryable: bool,
    fragment: str,
) -> None:
    client = mock_client(lambda request: httpx.Response(status, json=body))
    tts = ElevenLabsTTS(api_key=KEY, http_client=client)
    with pytest.raises(error_type) as info:
        await tts.synthesize("Hello.").collect()
    assert type(info.value) is error_type
    assert info.value.status_code == status and info.value.retryable is retryable
    assert fragment in str(info.value)
    await client.aclose()


async def test_synthesize_maps_network_errors() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        raise httpx.ConnectError("connection refused", request=request)

    client = mock_client(handler)
    tts = ElevenLabsTTS(api_key=KEY, http_client=client)
    with pytest.raises(ProviderConnectionError):
        await tts.synthesize("Hello.").collect()
    assert not await tts.synthesize("   ").collect()  # nothing to say: no request at all
    assert len(requests) == 1
    await tts.aclose()
    await client.aclose()


async def test_non_streaming_mode_synthesizes_sentence_by_sentence() -> None:
    texts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        texts.append(json.loads(request.content)["text"])
        return httpx.Response(200, content=pcm(3000, 0.2, 24_000))

    client = mock_client(handler)
    tts = ElevenLabsTTS(api_key=KEY, http_client=client, streaming=False)
    stream = tts.stream()
    assert isinstance(stream, SentenceStreamAdapter)
    stream.push_text("First sentence here. Second one follows.")
    stream.end_input()
    items = await collect_tts(stream)
    await client.aclose()
    assert texts == ["First sentence here.", "Second one follows."]
    assert audio_of(items).duration > 0.2


# ------------------------------------------------------------------- registry / config
def test_registry_and_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ELEVENLABS_API_KEY", "key-b")
    tts = create("tts", "elevenlabs")
    assert isinstance(tts, ElevenLabsTTS)
    assert (tts.model, tts.sample_rate, tts.voice) == ("eleven_flash_v2_5", 24_000, DEFAULT_VOICE)
    assert tts.capabilities.streaming and tts.capabilities.word_timestamps
    assert tts._headers() == {"xi-api-key": "key-b"}
    monkeypatch.setenv("ELEVEN_API_KEY", "key-a")  # checked first
    assert create("tts", "elevenlabs")._headers() == {"xi-api-key": "key-a"}
    assert create("tts", {"provider": "elevenlabs", "api_key": "k"})._headers()["xi-api-key"] == "k"
    v3 = create("tts", "elevenlabs/eleven_v3_conversational")
    assert v3.dialogue and v3.model == "eleven_v3_conversational"

    stt = create("stt", "elevenlabs")
    assert isinstance(stt, ElevenLabsSTT) and stt.model == "scribe_v2_realtime"
    assert stt.capabilities.streaming and stt.capabilities.interim_results
    assert not stt.capabilities.end_of_turn and stt.batch_model == "scribe_v2"
    batch = create("stt", "elevenlabs/scribe_v2")
    assert not batch.capabilities.streaming and batch.batch_model == "scribe_v2"
    assert batch.capabilities.word_timestamps and batch.capabilities.language_detection

    for kind in ("tts", "stt"):
        spec = get_provider(kind, "elevenlabs")
        assert spec.env == ("ELEVEN_API_KEY", "ELEVENLABS_API_KEY")
        assert not spec.local and spec.extra is None and spec.available

    assert ElevenLabsTTS(region="us").base_url == REGIONS["us"]
    assert ElevenLabsSTT(region="SG").ws_url().startswith("wss://api.sg.residency.elevenlabs.io/")
    assert ElevenLabsTTS(base_url="http://localhost:9/").ws_url().startswith("ws://localhost:9/v1/")

    monkeypatch.delenv("ELEVEN_API_KEY")
    monkeypatch.delenv("ELEVENLABS_API_KEY")
    with pytest.raises(ConfigurationError, match="ELEVEN_API_KEY"):
        ElevenLabsTTS()
    with pytest.raises(ConfigurationError, match="ELEVENLABS_API_KEY"):
        ElevenLabsSTT()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sample_rate": 12_345},
        {"speed": 2.0},
        {"stability": 1.5},
        {"chunk_length_schedule": [20]},
        {"chunk_length_schedule": []},
        {"inactivity_timeout": 500},
        {"apply_text_normalization": "sometimes"},
        {"alignment": "phonemes"},
        {"pronunciation_dictionaries": [("a", "1")] * 4},
        {"region": "mars"},
        {"region": "us", "base_url": "https://example.com"},
        {"keepalive_interval": 0},
    ],
)
def test_invalid_tts_options(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ConfigurationError):
        ElevenLabsTTS(api_key=KEY, **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sample_rate": 11_025},
        {"commit_strategy": "sometimes"},
        {"keyterms": "ElevenLabs"},
        {"chunk_duration": 5.0},
        {"vad_silence_threshold_secs": 10.0},
    ],
)
def test_invalid_stt_options(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ConfigurationError):
        ElevenLabsSTT(api_key=KEY, **kwargs)


# --------------------------------------------------------------------------- fake Scribe
Script = Callable[["FakeScribe", ServerConnection], Awaitable[None]]


@dataclass
class FakeScribe:
    script: Script
    url: str = ""
    requests: list[Request] = field(default_factory=list)
    received: list[dict[str, Any]] = field(default_factory=list)
    reject_status: int | None = None

    def process_request(self, conn: ServerConnection, request: Request) -> Response | None:
        self.requests.append(request)
        if self.reject_status is not None:
            body = json.dumps({"detail": {"status": "unauthorized", "message": "Bad key"}})
            return conn.respond(HTTPStatus(self.reject_status), body)
        return None

    async def handler(self, ws: ServerConnection) -> None:
        try:
            await self.script(self, ws)
        except ConnectionClosed:
            pass

    def chunks(self) -> list[dict[str, Any]]:
        return [m for m in self.received if m.get("message_type") == "input_audio_chunk"]

    def commits(self) -> list[dict[str, Any]]:
        return [m for m in self.chunks() if m["commit"]]

    def audio_bytes(self) -> int:
        return sum(len(base64.b64decode(m["audio_base_64"])) for m in self.chunks())


StartScribe = Callable[[Script], Awaitable[FakeScribe]]


@pytest.fixture
async def scribe() -> AsyncIterator[StartScribe]:
    servers: list[Any] = []

    async def start(script: Script) -> FakeScribe:
        fake = FakeScribe(script)
        srv = await serve(fake.handler, "127.0.0.1", 0, process_request=fake.process_request)
        servers.append(srv)
        fake.url = f"http://127.0.0.1:{next(iter(srv.sockets)).getsockname()[1]}"
        return fake

    yield start
    for srv in servers:
        srv.close()
        await srv.wait_closed()


def scribe_words(text: str, start: float, step: float = 0.3) -> list[dict[str, Any]]:
    """Scribe's word list: words with spacing entries between them."""
    out: list[dict[str, Any]] = []
    for n, word in enumerate(text.split()):
        if n:
            out.append({"text": " ", "start": start, "end": start, "type": "spacing",
                        "logprob": 0.0})  # fmt: skip
        out.append({"text": word, "start": start, "end": start + step * 0.8, "type": "word",
                    "logprob": -0.1, "speaker_id": "speaker_0"})  # fmt: skip
        start += step
    return out


def answering(
    finals: list[str],
    *,
    partial: str | None = "Hello",
    timestamps: bool = False,
    language: str = "en",
) -> Script:
    """A Scribe session: a partial after the first audio, one committed transcript (and
    its timestamped copy) per commit; the texts come from ``finals`` (then empty)."""

    async def script(fake: FakeScribe, ws: ServerConnection) -> None:
        await ws.send(json.dumps({"message_type": "session_started", "session_id": "sess_1",
                                  "config": {"sample_rate": 16000, "model_id": "scribe_v2_realtime",
                                             "commit_strategy": "manual"}}))  # fmt: skip
        texts = iter(finals)
        sent_partial = False
        async for raw in ws:
            msg = json.loads(raw)
            fake.received.append(msg)
            if msg.get("audio_base_64") and partial and not sent_partial:
                sent_partial = True
                await ws.send(json.dumps({"message_type": "partial_transcript", "text": partial}))
            if msg.get("commit"):
                text = next(texts, "")
                await ws.send(json.dumps({"message_type": "committed_transcript", "text": text}))
                if timestamps:
                    await ws.send(json.dumps({
                        "message_type": "committed_transcript_with_timestamps",
                        "text": text,
                        "language_code": language,
                        "words": scribe_words(text, 0.2) if text else [],
                    }))  # fmt: skip

    return script


async def collect_stt(
    stream: STTStream, until: STTEventType | None = None, timeout: float = 5.0
) -> list[STTEvent]:
    events: list[STTEvent] = []

    async def run() -> None:
        async for ev in stream:
            events.append(ev)
            if ev.type == until:
                return

    await asyncio.wait_for(run(), timeout)
    return events


def push_speech(stream: STTStream, seconds: float, rate: int = 16_000) -> None:
    audio = synth_speech(seconds, rate)
    step = 0.02
    t = 0.0
    while t < seconds - 1e-9:
        stream.push_audio(audio.slice(t, t + step))
        t += step


# --------------------------------------------------------------------------- STT tests
async def test_stt_manual_commits_follow_flush(scribe: StartScribe) -> None:
    fake = await scribe(answering(["Hello world"]))
    stt = ElevenLabsSTT(api_key=KEY, base_url=fake.url, language="en-US", keyterms=["Scribe"])
    metrics: list[STTMetrics] = []
    stt.on("metrics", metrics.append)
    stream = stt.stream()
    push_speech(stream, 0.35)
    stream.flush()
    events = await collect_stt(stream, until=T.FINAL_TRANSCRIPT)
    stream.flush()  # nothing new: acknowledged at once, no second commit
    events += await collect_stt(stream, until=T.FINAL_TRANSCRIPT)
    assert len(fake.commits()) == 1
    stream.end_input()
    events += await collect_stt(stream)
    await stream.aclose()

    assert [(e.type, e.text) for e in events] == [
        (T.INTERIM_TRANSCRIPT, "Hello"),
        (T.FINAL_TRANSCRIPT, "Hello world"),
        (T.FINAL_TRANSCRIPT, ""),
        (T.FINAL_TRANSCRIPT, ""),
    ]
    assert events[0].segment_id == events[1].segment_id != events[2].segment_id
    assert events[1].transcript is not None and events[1].transcript.language == "en-US"

    chunks = fake.chunks()
    assert all(c["sample_rate"] == 16_000 for c in chunks)
    assert [len(base64.b64decode(c["audio_base_64"])) for c in chunks[:-1]] == [3200] * 3
    assert chunks[-1]["commit"] is True  # the partial last chunk carries the commit
    assert not any(c["commit"] for c in chunks[:-1])
    assert fake.audio_bytes() == round(0.35 * 16_000) * 2
    assert not any("previous_text" in c for c in chunks)

    request = fake.requests[0]
    assert path(request) == "/v1/speech-to-text/realtime"
    assert query(request) == {
        "model_id": ["scribe_v2_realtime"],
        "audio_format": ["pcm_16000"],
        "commit_strategy": ["manual"],
        "language_code": ["en"],
        "keyterms": ["Scribe"],
    }
    assert request.headers["xi-api-key"] == KEY
    assert metrics and all(m.streamed and m.latency is not None for m in metrics)


async def test_stt_timestamps_come_with_the_second_copy(scribe: StartScribe) -> None:
    fake = await scribe(answering(["Hello big world"], timestamps=True, language="eng"))
    stt = ElevenLabsSTT(api_key=KEY, base_url=fake.url, include_timestamps=True,
                        include_language_detection=True)  # fmt: skip
    assert stt.capabilities.word_timestamps and stt.capabilities.language_detection
    stream = stt.stream()
    push_speech(stream, 0.2)
    stream.flush()
    events = await collect_stt(stream, until=T.FINAL_TRANSCRIPT)
    stream.end_input()
    events += await collect_stt(stream)
    await stream.aclose()

    finals = [e for e in events if e.type == T.FINAL_TRANSCRIPT]
    assert [f.text for f in finals] == ["Hello big world", ""]
    transcript = finals[0].transcript
    assert transcript is not None and transcript.language == "eng"
    assert [(w.word, w.start) for w in transcript.words or []] == pytest.approx(
        [("Hello", 0.2), ("big", 0.5), ("world", 0.8)]
    )
    assert transcript.confidence == pytest.approx(math.exp(-0.1))
    assert (transcript.start_time, transcript.end_time) == pytest.approx((0.2, 1.04))
    assert query(fake.requests[0])["include_timestamps"] == ["true"]
    assert query(fake.requests[0])["include_language_detection"] == ["true"]


async def test_stt_flush_while_a_commit_is_in_flight(scribe: StartScribe) -> None:
    release = asyncio.Event()

    async def slow(fake: FakeScribe, ws: ServerConnection) -> None:
        async for raw in ws:
            msg = json.loads(raw)
            fake.received.append(msg)
            if msg.get("commit"):
                await release.wait()  # answer late
                await ws.send(json.dumps({"message_type": "committed_transcript", "text": "Hi"}))

    fake = await scribe(slow)
    stream = ElevenLabsSTT(api_key=KEY, base_url=fake.url).stream()
    push_speech(stream, 0.1)
    stream.flush()
    await wait_until(lambda: fake.commits())
    stream.flush()  # nothing new, a commit pending: its transcript answers both flushes
    await asyncio.sleep(0.1)
    release.set()
    events = await collect_stt(stream, until=T.FINAL_TRANSCRIPT)
    assert [(e.type, e.text) for e in events] == [(T.FINAL_TRANSCRIPT, "Hi")]
    assert len(fake.commits()) == 1
    stream.end_input()
    await collect_stt(stream)
    await stream.aclose()


async def test_stt_server_vad_emits_speech_events(scribe: StartScribe) -> None:
    async def vad(fake: FakeScribe, ws: ServerConnection) -> None:
        sent = False
        async for raw in ws:
            fake.received.append(json.loads(raw))
            if not sent and fake.audio_bytes() >= 0.2 * 32_000:
                sent = True
                for text in ("Book a", "Book a table"):
                    await ws.send(json.dumps({"message_type": "partial_transcript", "text": text}))
                await ws.send(json.dumps({"message_type": "committed_transcript",
                                          "text": "Book a table."}))  # fmt: skip

    fake = await scribe(vad)
    stt = ElevenLabsSTT(api_key=KEY, base_url=fake.url, commit_strategy="vad",
                        vad_silence_threshold_secs=0.5, vad_threshold=0.4,
                        min_speech_duration_ms=100, min_silence_duration_ms=100)  # fmt: skip
    metrics: list[STTMetrics] = []
    stt.on("metrics", metrics.append)
    stream = stt.stream()
    push_speech(stream, 0.3)
    events = await collect_stt(stream, until=T.END_OF_SPEECH)
    await stream.aclose()

    assert [(e.type, e.text) for e in events] == [
        (T.START_OF_SPEECH, ""),
        (T.INTERIM_TRANSCRIPT, "Book a"),
        (T.INTERIM_TRANSCRIPT, "Book a table"),
        (T.FINAL_TRANSCRIPT, "Book a table."),
        (T.END_OF_SPEECH, ""),
    ]
    assert query(fake.requests[0]) == {
        "model_id": ["scribe_v2_realtime"],
        "audio_format": ["pcm_16000"],
        "commit_strategy": ["vad"],
        "vad_silence_threshold_secs": ["0.5"],
        "vad_threshold": ["0.4"],
        "min_speech_duration_ms": ["100"],
        "min_silence_duration_ms": ["100"],
    }
    assert metrics and metrics[0].audio_duration > 0 and metrics[0].latency is None


async def test_stt_options_in_the_url_and_previous_text(scribe: StartScribe) -> None:
    fake = await scribe(answering(["ok"], partial=None))
    stt = ElevenLabsSTT(
        api_key=KEY,
        base_url=fake.url,
        sample_rate=24_000,
        secondary_languages=["fr-FR", "de"],
        no_verbatim=True,
        filter_background_audio=True,
        enable_logging=False,
        previous_text="How can I help?",
        chunk_duration=0.05,
    )
    stream = stt.stream()
    push_speech(stream, 0.2, rate=24_000)
    stream.end_input()
    await collect_stt(stream)
    await stream.aclose()

    assert query(fake.requests[0]) == {
        "model_id": ["scribe_v2_realtime"],
        "audio_format": ["pcm_24000"],
        "commit_strategy": ["manual"],
        "secondary_languages": ["fr", "de"],
        "no_verbatim": ["true"],
        "filter_background_audio": ["true"],
        "enable_logging": ["false"],
    }
    chunks = fake.chunks()
    assert chunks[0]["previous_text"] == "How can I help?"
    assert not any("previous_text" in c for c in chunks[1:])
    assert all(c["sample_rate"] == 24_000 for c in chunks)
    assert len(base64.b64decode(chunks[0]["audio_base_64"])) == 2400  # 50 ms at 24 kHz


@pytest.mark.parametrize(
    ("kind", "error_type", "retryable"),
    [
        ("auth_error", AuthenticationError, False),
        ("quota_exceeded", RateLimitError, False),
        ("rate_limited", RateLimitError, True),
        ("transcriber_error", ProviderError, True),
        ("input_error", ProviderError, False),
        ("insufficient_audio_activity", ProviderTimeoutError, True),
    ],
)
async def test_stt_error_messages_are_raised(
    scribe: StartScribe, kind: str, error_type: type[ProviderError], retryable: bool
) -> None:
    async def failing(fake: FakeScribe, ws: ServerConnection) -> None:
        await ws.send(json.dumps({"message_type": kind, "error": f"details about {kind}"}))
        await ws.close(1008, kind)

    fake = await scribe(failing)
    stream = ElevenLabsSTT(api_key=KEY, base_url=fake.url).stream()
    push_speech(stream, 0.1)
    with pytest.raises(error_type) as info:
        await collect_stt(stream)
    assert type(info.value) is error_type and info.value.retryable is retryable
    assert f"details about {kind}" in str(info.value)
    await stream.aclose()


async def test_stt_commit_throttled_is_not_fatal(scribe: StartScribe) -> None:
    async def throttling(fake: FakeScribe, ws: ServerConnection) -> None:
        commits = 0
        async for raw in ws:
            msg = json.loads(raw)
            fake.received.append(msg)
            if msg.get("commit"):
                commits += 1
                if commits == 1:
                    await ws.send(json.dumps({"message_type": "commit_throttled",
                                              "error": "Too many commits"}))  # fmt: skip
                else:
                    await ws.send(json.dumps({"message_type": "committed_transcript",
                                              "text": "Throttled then committed"}))  # fmt: skip

    fake = await scribe(throttling)
    stream = ElevenLabsSTT(api_key=KEY, base_url=fake.url).stream()
    push_speech(stream, 0.1)
    stream.flush()
    await wait_until(lambda: fake.commits())
    await asyncio.sleep(0.1)
    stream.flush()  # no new audio, but the throttled audio still needs a commit
    events = await collect_stt(stream, until=T.FINAL_TRANSCRIPT)
    assert [e.text for e in events] == ["Throttled then committed"]
    assert len(fake.commits()) == 2
    stream.end_input()
    await collect_stt(stream)
    await stream.aclose()


async def test_stt_unexpected_close_is_an_error(scribe: StartScribe) -> None:
    async def closing(fake: FakeScribe, ws: ServerConnection) -> None:
        await ws.close()

    fake = await scribe(closing)
    stream = ElevenLabsSTT(api_key=KEY, base_url=fake.url).stream()
    push_speech(stream, 0.1)
    with pytest.raises(ProviderConnectionError):
        await collect_stt(stream)
    await stream.aclose()


async def test_stt_rejected_handshake_is_mapped(scribe: StartScribe) -> None:
    fake = await scribe(answering([]))
    fake.reject_status = 403
    stream = ElevenLabsSTT(api_key=KEY, base_url=fake.url).stream()
    with pytest.raises(AuthenticationError, match="Bad key"):
        await collect_stt(stream)
    await stream.aclose()


async def test_stt_keepalive_sends_silence_when_no_audio_flows(scribe: StartScribe) -> None:
    fake = await scribe(answering([], partial=None))
    stream = ElevenLabsSTT(api_key=KEY, base_url=fake.url, keepalive_interval=0.1).stream()
    await wait_until(lambda: len(fake.chunks()) >= 2)
    for chunk in fake.chunks():
        assert not chunk["commit"]
        assert set(base64.b64decode(chunk["audio_base_64"])) == {0}
    stream.end_input()
    events = await collect_stt(stream)
    await stream.aclose()
    assert fake.commits()  # the keep-alive audio is committed like any other
    assert [e.type for e in events] == [T.FINAL_TRANSCRIPT]


async def test_scribe_commits_drive_cascade_endpointing(scribe: StartScribe) -> None:
    fake = await scribe(answering(["Hey can you help me?"]))
    llm = MockLLM(responses=["Sure, what do you need?"])
    stt = ElevenLabsSTT(api_key=KEY, base_url=fake.url)
    engine = CascadeEngine(stt=stt, llm=llm, tts=MockTTS(), vad=EnergyVAD())
    conn = await engine.connect(EngineOptions())
    events: list[object] = []

    async def drain() -> None:
        async for ev in conn.events():
            events.append(ev)
            if isinstance(ev, ResponseDone):
                return

    drainer = asyncio.create_task(drain())
    audio = AudioFrame.concat([synth_speech(0.6, 16_000), AudioFrame.silence(1.5, 16_000)])
    t = 0.0
    while t < audio.duration - 1e-9:
        await conn.send_audio(audio.slice(t, t + 0.02))
        t += 0.02
    await asyncio.wait_for(drainer, 10)
    await conn.aclose()
    await engine.aclose()

    assert any(isinstance(e, InputCommitted) for e in events)
    finals = [e.text for e in events if isinstance(e, InputTranscript) and e.is_final]
    assert finals == ["Hey can you help me?"]
    last_user = llm.requests[0].last_message("user")
    assert last_user is not None and last_user.text == "Hey can you help me?"
    assert fake.commits()  # the cascade's endpointing flush became a Scribe commit


# ----------------------------------------------------------------------- batch Scribe
def multipart_fields(request: httpx.Request) -> tuple[dict[str, list[str]], dict[str, Any]]:
    """Parse a multipart body into (text fields, file part headers + payload)."""
    content_type = request.headers["content-type"]
    boundary = content_type.split("boundary=")[1].encode()
    fields: dict[str, list[str]] = {}
    file: dict[str, Any] = {}
    for part in request.content.split(b"--" + boundary):
        if b"\r\n\r\n" not in part:
            continue
        head, _, payload = part.partition(b"\r\n\r\n")
        payload = payload.removesuffix(b"\r\n")
        headers = head.decode()
        name = headers.split('name="')[1].split('"')[0]
        if "filename=" in headers:
            file = {"headers": headers, "payload": payload}
        else:
            fields.setdefault(name, []).append(payload.decode())
    return fields, file


BATCH_RESPONSE = {
    "language_code": "eng",
    "language_probability": 0.98,
    "text": "Hello world! (laughs)",
    "words": [
        {"text": "Hello", "start": 0.1, "end": 0.4, "type": "word", "logprob": -0.05},
        {"text": " ", "start": 0.4, "end": 0.5, "type": "spacing", "logprob": 0.0},
        {"text": "world!", "start": 0.5, "end": 0.9, "type": "word", "logprob": -0.2},
        {"text": "(laughs)", "start": 1.0, "end": 1.3, "type": "audio_event", "logprob": 0.0},
    ],
}


async def test_transcribe_uses_the_batch_api() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=BATCH_RESPONSE)

    client = mock_client(handler)
    stt = ElevenLabsSTT(api_key=KEY, http_client=client, language="en", keyterms=["Scribe", "v2"],
                        no_verbatim=True)  # fmt: skip
    audio = synth_speech(0.5, 48_000)
    result = await stt.transcribe(audio)
    await stt.aclose()
    assert not client.is_closed
    await client.aclose()

    assert result.text == "Hello world! (laughs)" and result.language == "eng"
    assert [(w.word, w.start, w.end) for w in result.words or []] == [
        ("Hello", 0.1, 0.4),
        ("world!", 0.5, 0.9),
    ]
    assert result.confidence == pytest.approx((math.exp(-0.05) + math.exp(-0.2)) / 2)
    assert (result.start_time, result.end_time) == (0.1, 0.9)
    request = seen[0]
    assert str(request.url) == "https://api.elevenlabs.io/v1/speech-to-text"
    assert request.headers["xi-api-key"] == KEY
    fields, file = multipart_fields(request)
    assert fields == {
        "model_id": ["scribe_v2"],
        "timestamps_granularity": ["word"],
        "tag_audio_events": ["false"],
        "language_code": ["en"],
        "no_verbatim": ["true"],
        "keyterms": ["Scribe", "v2"],
        "file_format": ["pcm_s16le_16"],
    }
    # resampled to 16 kHz mono s16le and sent raw (no container to decode)
    assert abs(len(file["payload"]) - round(0.5 * 16_000) * 2) <= 64


async def test_transcribe_sends_wav_for_other_rates() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"text": "hi", "language_code": "en", "words": []})

    client = mock_client(handler)
    stt = ElevenLabsSTT(api_key=KEY, http_client=client, model="scribe_v2", sample_rate=24_000,
                        enable_logging=False, tag_audio_events=True)  # fmt: skip
    result = await stt.transcribe(synth_speech(0.2, 24_000))
    await client.aclose()
    assert result.text == "hi" and result.words is None
    fields, file = multipart_fields(seen[0])
    assert "file_format" not in fields and fields["tag_audio_events"] == ["true"]
    assert "audio/wav" in file["headers"] and file["payload"][:4] == b"RIFF"
    assert dict(seen[0].url.params) == {"enable_logging": "false"}


@pytest.mark.parametrize(
    ("status", "error_type"),
    [(401, AuthenticationError), (429, RateLimitError), (500, ProviderError)],
)
async def test_transcribe_maps_http_errors(status: int, error_type: type[ProviderError]) -> None:
    client = mock_client(
        lambda request: httpx.Response(status, json={"detail": {"message": "nope"}})
    )
    stt = ElevenLabsSTT(api_key=KEY, http_client=client)
    with pytest.raises(error_type, match="nope"):
        await stt.transcribe(synth_speech(0.2, 16_000))
    await client.aclose()


# ---------------------------------------------------------------------- integration
_KEY_SET = bool(os.environ.get("ELEVEN_API_KEY") or os.environ.get("ELEVENLABS_API_KEY"))
needs_key = pytest.mark.skipif(not _KEY_SET, reason="needs ELEVEN_API_KEY or ELEVENLABS_API_KEY")


@pytest.fixture
def real_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Undo the autouse fixture's key removal for integration tests."""
    monkeypatch.undo()


@pytest.mark.integration
@needs_key
@pytest.mark.usefixtures("real_key")
async def test_integration_stream_with_word_timings() -> None:
    tts = ElevenLabsTTS()
    try:
        await tts.warmup()
        stream = tts.stream()
        stream.push_text("Hello from voice agent next. ")
        stream.push_text("This sentence streams over the multi context socket. ")
        stream.end_input()
        items = await collect_tts(stream, timeout=30)
    finally:
        await tts.aclose()
    audio = audio_of(items)
    assert audio.sample_rate == 24_000 and audio.duration > 1.5
    words = words_of(items)
    assert len(words) >= 10
    assert all(a.start <= b.start for a, b in itertools.pairwise(words))
    assert words[-1].end <= audio.duration + 0.25
    assert items[-1].is_final


@pytest.mark.integration
@needs_key
@pytest.mark.usefixtures("real_key")
async def test_integration_cancel_keeps_the_socket_usable() -> None:
    tts = ElevenLabsTTS()
    try:
        stream = tts.stream()
        stream.push_text("This is a long answer that will be interrupted right after it starts. ")
        assert (await asyncio.wait_for(anext(stream), 30)) is not None
        await stream.aclose()
        stream = tts.stream()
        stream.push_text("Still working. ")
        stream.end_input()
        assert audio_of(await collect_tts(stream, timeout=30)).duration > 0.3
    finally:
        await tts.aclose()


@pytest.mark.integration
@needs_key
@pytest.mark.usefixtures("real_key")
async def test_integration_http_tts_and_batch_stt_round_trip() -> None:
    tts = ElevenLabsTTS(sample_rate=16_000)
    stt = ElevenLabsSTT()
    try:
        audio = await tts.synthesize("The quick brown fox jumps over the lazy dog.").collect()
        assert audio.duration > 1.0
        result = await asyncio.wait_for(stt.transcribe(audio), 60)
    finally:
        await tts.aclose()
        await stt.aclose()
    assert "fox" in result.text.lower()
    assert result.words


@pytest.mark.integration
@needs_key
@pytest.mark.usefixtures("real_key")
async def test_integration_realtime_manual_commit() -> None:
    tts = ElevenLabsTTS(sample_rate=16_000)
    try:
        audio = await tts.synthesize("Can you tell me the weather in Paris today?").collect()
    finally:
        await tts.aclose()
    stream = ElevenLabsSTT(include_timestamps=True).stream()

    async def feed() -> None:
        t = 0.0
        while t < audio.duration:
            stream.push_audio(audio.slice(t, t + 0.1))
            t += 0.1
            await asyncio.sleep(0.1)  # real time
        stream.push_audio(AudioFrame.silence(0.5, 16_000))
        stream.flush()

    feeder = asyncio.create_task(feed())
    try:
        events = await collect_stt(stream, until=T.FINAL_TRANSCRIPT, timeout=30)
    finally:
        feeder.cancel()
        await asyncio.gather(feeder, return_exceptions=True)
        await stream.aclose()
    final = events[-1]
    assert "weather" in final.text.lower()
    assert final.transcript is not None and final.transcript.words
