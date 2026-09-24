"""A scripted OpenAI realtime *transcription* server (GA protocol) for offline tests.

:class:`FakeTranscriptionServer` listens on ``127.0.0.1:0`` and replays the event sequences
of a ``type: "transcription"`` session:

* ``session.created`` (``realtime.transcription_session``); ``session.update`` is validated
  (24 kHz PCM, a known transcription model, ``language`` vs ``languages``) and answered
  with ``session.updated`` or an ``invalid_request_error``;
* ``input_audio_buffer.append`` accumulates audio. Streaming models
  (``gpt-live-transcribe``, ``gpt-realtime-whisper``) emit
  ``conversation.item.input_audio_transcription.delta`` events *while* the audio arrives,
  like the real ones; the others transcribe after the commit;
* ``input_audio_buffer.commit`` -> ``input_audio_buffer.committed`` + ``conversation.item.added``
  -> deltas -> ``conversation.item.input_audio_transcription.completed`` (with usage), or
  ``.failed``; commits of less than 100 ms are rejected with
  ``input_audio_buffer_commit_empty`` like the real API;
* with ``turn_detection`` configured, an energy VAD emits ``speech_started`` /
  ``speech_stopped`` and commits by itself.

Transcripts are scripted per committed item (``transcripts``, or a function of the item's
audio duration); ``delays`` postpones the transcription of item *n* (to reorder
completions); ``drop_on_commits`` aborts chosen connections at a chosen commit.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from http import HTTPStatus
from typing import Any
from urllib.parse import parse_qsl, urlsplit

from websockets.asyncio.server import Server, ServerConnection, serve
from websockets.exceptions import ConnectionClosed
from websockets.http11 import Request, Response

from voice_agent_next.audio import AudioFrame
from voice_agent_next.providers.energy import EnergyVAD
from voice_agent_next.utils.ids import new_id
from voice_agent_next.vad import VADEventType, VADOptions, VADStream

STREAMING_MODELS = ("gpt-live-transcribe", "gpt-realtime-whisper")
KNOWN_MODELS = (
    "gpt-live-transcribe",
    "gpt-realtime-whisper",
    "gpt-transcribe",
    "gpt-4o-transcribe",
    "gpt-4o-mini-transcribe",
    "whisper-1",
)
RATE = 24_000
WORD_AUDIO = 0.1
"""Streaming models emit one word per this many seconds of buffered audio."""


@dataclass
class Handshake:
    path: str
    query: dict[str, str]
    headers: dict[str, str]


class FakeTranscriptionServer:
    """Scripted transcription server. Use as ``async with FakeTranscriptionServer() as s``.

    Args:
        transcripts: transcript of each committed item, in commit order (then ``"hello"``),
            or a function of the item's audio duration in seconds (evaluated at the commit
            for non-streaming models).
        delays: seconds to wait before transcribing item *n* (default 0).
        fail_items: indexes of committed items whose transcription fails.
        api_key: reject handshakes without this key (401).
        reject_status: reject every handshake with this HTTP status.
        reject_session_update: error object sent instead of ``session.updated``.
        drop_on_commits: ``(connection index, commit number)`` pairs: abort that connection
            when it receives that commit (1-based), before transcribing it.
    """

    def __init__(
        self,
        *,
        transcripts: Sequence[str] | Callable[[float], str] = (),
        delays: Sequence[float] = (),
        fail_items: Sequence[int] = (),
        api_key: str | None = None,
        reject_status: int | None = None,
        reject_session_update: dict[str, Any] | None = None,
        drop_on_commits: Sequence[tuple[int, int]] = (),
    ) -> None:
        self.script = transcripts if callable(transcripts) else None
        self.transcripts = [] if callable(transcripts) else list(transcripts)
        self.delays = list(delays)
        self.fail_items = set(fail_items)
        self.api_key = api_key
        self.reject_status = reject_status
        self.reject_session_update = reject_session_update
        self.drop_on_commits = set(drop_on_commits)
        self.handshakes: list[Handshake] = []
        self.received: list[dict[str, Any]] = []
        """Client events, per connection order (append payloads replaced by byte counts)."""
        self.sent: list[dict[str, Any]] = []
        self.connections: list[_Connection] = []
        self.committed = 0
        self.port = 0
        self._server: Server | None = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    async def __aenter__(self) -> FakeTranscriptionServer:
        self._server = await serve(
            self._handle,
            "127.0.0.1",
            0,
            process_request=self._process_request,
            compression=None,
            max_size=None,
        )
        self.port = next(iter(self._server.sockets)).getsockname()[1]
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    # ------------------------------------------------------------------- helpers
    def events(self, etype: str, connection: int | None = None) -> list[dict[str, Any]]:
        """Client events of ``etype`` (on one connection when ``connection`` is given)."""
        return [
            e
            for e in self.received
            if e.get("type") == etype and (connection is None or e["_conn"] == connection)
        ]

    def sent_events(self, etype: str) -> list[dict[str, Any]]:
        """Server events of ``etype`` sent so far (all connections)."""
        return [e for e in self.sent if e.get("type") == etype]

    def audio_bytes(self, connection: int | None = None) -> int:
        return sum(e["audio"] for e in self.events("input_audio_buffer.append", connection))

    def next_transcript(self, seconds: float) -> str:
        if self.script is not None:
            return self.script(seconds)
        return self.transcripts.pop(0) if self.transcripts else "hello"

    async def push(self, event: dict[str, Any]) -> None:
        """Send a raw server event on every open connection."""
        for conn in self.connections:
            await conn.send_raw(event)

    def drop(self) -> None:
        """Abort every connection without a close frame (simulated network failure)."""
        for conn in self.connections:
            conn.ws.transport.abort()

    def _process_request(self, connection: ServerConnection, request: Request) -> Response | None:
        parts = urlsplit(request.path)
        headers = {k.lower(): v for k, v in request.headers.raw_items()}
        self.handshakes.append(Handshake(parts.path, dict(parse_qsl(parts.query)), headers))
        if self.reject_status is not None:
            return connection.respond(HTTPStatus(self.reject_status), "rejected\n")
        if self.api_key is not None:
            auth = headers.get("authorization", "")
            key = headers.get("api-key") or (auth[7:] if auth.lower().startswith("bearer ") else "")
            if key != self.api_key:
                return connection.respond(HTTPStatus.UNAUTHORIZED, "invalid api key\n")
        return None

    async def _handle(self, ws: ServerConnection) -> None:
        conn = _Connection(self, ws, len(self.connections))
        self.connections.append(conn)
        await conn.run()


class _Connection:
    def __init__(self, server: FakeTranscriptionServer, ws: ServerConnection, index: int) -> None:
        self.server = server
        self.ws = ws
        self.index = index
        self.session: dict[str, Any] = {
            "type": "transcription",
            "object": "realtime.transcription_session",
            "id": new_id("sess_"),
            "expires_at": int(time.time() + 3600),
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": RATE},
                    "noise_reduction": None,
                    "transcription": {"model": "gpt-4o-transcribe"},
                    "turn_detection": None,
                }
            },
            "include": None,
        }
        self.buffer = bytearray()
        self.item_id = new_id("item_")
        self.words_sent = 0
        self.words: list[str] | None = None
        self.previous: str | None = None
        self.commits = 0
        self.vad: VADStream | None = None
        self.vad_origin = 0.0
        self.position = 0.0
        self.tasks: set[asyncio.Task[None]] = set()

    # --------------------------------------------------------------------- config
    @property
    def model(self) -> str:
        return str(self.session["audio"]["input"]["transcription"]["model"])

    @property
    def streaming(self) -> bool:
        return self.model.startswith(STREAMING_MODELS)

    def turn_detection(self) -> dict[str, Any] | None:
        td = self.session["audio"]["input"].get("turn_detection")
        return td if isinstance(td, dict) else None

    # -------------------------------------------------------------------- sending
    async def send(self, etype: str, **fields: Any) -> None:
        await self.send_raw({"event_id": new_id("event_"), "type": etype, **fields})

    async def send_raw(self, event: dict[str, Any]) -> None:
        self.server.sent.append(event)
        with contextlib.suppress(ConnectionClosed):
            await self.ws.send(json.dumps(event))

    async def error(self, source: dict[str, Any], code: str, message: str, **extra: Any) -> None:
        err = {"type": "invalid_request_error", "code": code, "message": message, "param": None,
               "event_id": source.get("event_id"), **extra}  # fmt: skip
        await self.send("error", error=err)

    # ----------------------------------------------------------------------- loop
    async def run(self) -> None:
        await self.send("session.created", session=self.session)
        try:
            async for raw in self.ws:
                event = json.loads(raw)
                record = {**event, "_conn": self.index}
                if event.get("type") == "input_audio_buffer.append":
                    record["audio"] = len(base64.b64decode(event.get("audio", "")))
                self.server.received.append(record)
                await self.handle(event)
        except ConnectionClosed:
            pass
        finally:
            for task in list(self.tasks):
                task.cancel()

    async def handle(self, ev: dict[str, Any]) -> None:
        etype = ev.get("type")
        if etype == "session.update":
            await self.on_session_update(ev)
        elif etype == "input_audio_buffer.append":
            await self.on_append(ev)
        elif etype == "input_audio_buffer.commit":
            await self.on_commit(ev)
        elif etype == "input_audio_buffer.clear":
            self.buffer.clear()
            self.item_id, self.words, self.words_sent = new_id("item_"), None, 0
            await self.send("input_audio_buffer.cleared")
        else:
            await self.error(ev, "invalid_event", f"unknown event type {etype!r}")

    async def on_session_update(self, ev: dict[str, Any]) -> None:
        if self.server.reject_session_update is not None:
            await self.send("error", error={**self.server.reject_session_update,
                                            "event_id": ev.get("event_id")})  # fmt: skip
            return
        session = ev.get("session") or {}
        audio_in = (session.get("audio") or {}).get("input") or {}
        transcription = audio_in.get("transcription") or {}
        fmt = audio_in.get("format") or {}
        if session.get("type") != "transcription":
            await self.error(ev, "invalid_value", "session.type must be 'transcription'")
            return
        if fmt and (fmt.get("type") != "audio/pcm" or fmt.get("rate") != RATE):
            await self.error(ev, "invalid_value", "Only 24kHz PCM is supported")
            return
        model = transcription.get("model")
        if model is not None and not str(model).startswith(KNOWN_MODELS):
            await self.error(ev, "invalid_model", f"Model {model!r} is not supported",
                             param="session.audio.input.transcription.model")  # fmt: skip
            return
        if "language" in transcription and "languages" in transcription:
            await self.error(ev, "invalid_value", "Send either language or languages")
            return
        target = self.session["audio"]["input"]
        for key in ("format", "noise_reduction", "transcription", "turn_detection"):
            if key in audio_in:
                target[key] = audio_in[key]
        if "include" in session:
            self.session["include"] = session["include"]
        td = self.turn_detection()
        if td is not None:
            silence = float(td.get("silence_duration_ms", 500)) / 1000.0
            options = VADOptions(min_speech_duration=0.1, min_silence_duration=silence)
            self.vad = EnergyVAD(sample_rate=16_000, options=options).stream()
            self.vad_origin = self.position
        else:
            self.vad = None
        await self.send("session.updated", session=self.session)

    async def on_append(self, ev: dict[str, Any]) -> None:
        data = base64.b64decode(ev.get("audio", ""))
        self.buffer += data
        frame = AudioFrame(data, RATE)
        self.position += frame.duration
        if self.streaming:
            await self.stream_words()
        if self.vad is None:
            return
        for vev in self.vad.push_audio(frame):
            if vev.type == VADEventType.START_OF_SPEECH:
                start = self.vad_origin + vev.audio_time - vev.speech_duration
                await self.send("input_audio_buffer.speech_started",
                                audio_start_ms=round(max(0.0, start) * 1000),
                                item_id=self.item_id)  # fmt: skip
            elif vev.type == VADEventType.END_OF_SPEECH:
                end = self.vad_origin + vev.audio_time
                await self.send("input_audio_buffer.speech_stopped",
                                audio_end_ms=round(end * 1000), item_id=self.item_id)  # fmt: skip
                await self.commit()

    def current_words(self) -> list[str]:
        if self.words is None:
            seconds = len(self.buffer) / (2 * RATE)
            self.words = re.findall(r"\S+\s*", self.server.next_transcript(seconds))
        return self.words

    async def stream_words(self) -> None:
        """Streaming models: one delta per ``WORD_AUDIO`` of buffered audio."""
        words = self.current_words()
        due = min(len(words), int(len(self.buffer) / (2 * RATE) / WORD_AUDIO))
        while self.words_sent < due:
            await self.send("conversation.item.input_audio_transcription.delta",
                            item_id=self.item_id, content_index=0,
                            delta=words[self.words_sent])  # fmt: skip
            self.words_sent += 1

    async def on_commit(self, ev: dict[str, Any]) -> None:
        buffered = len(self.buffer) / (2 * RATE)
        if buffered < 0.1:
            await self.error(
                ev,
                "input_audio_buffer_commit_empty",
                "Error committing input audio buffer: buffer too small. Expected at least "
                f"100ms of audio, but buffer only has {buffered * 1000:.2f}ms of audio.",
            )
            return
        self.commits += 1
        server = self.server
        if (self.index, self.commits) in server.drop_on_commits:
            self.ws.transport.abort()
            return
        await self.commit()

    async def commit(self) -> None:
        item_id, words, sent = self.item_id, self.current_words(), self.words_sent
        seconds = len(self.buffer) / (2 * RATE)
        self.buffer.clear()
        self.item_id, self.words, self.words_sent = new_id("item_"), None, 0
        await self.send("input_audio_buffer.committed", previous_item_id=self.previous,
                        item_id=item_id)  # fmt: skip
        item = {"id": item_id, "type": "message", "role": "user", "status": "completed",
                "content": [{"type": "input_audio", "transcript": None}]}  # fmt: skip
        await self.send("conversation.item.added", previous_item_id=self.previous, item=item)
        self.previous = item_id
        index = self.server.committed
        self.server.committed += 1
        delay = self.server.delays[index] if index < len(self.server.delays) else 0.0
        failed = index in self.server.fail_items
        task = asyncio.create_task(self.transcribe(item_id, words, sent, seconds, delay, failed))
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    async def transcribe(
        self, item_id: str, words: list[str], sent: int, seconds: float, delay: float, failed: bool
    ) -> None:
        if delay:
            await asyncio.sleep(delay)
        for word in words[sent:]:
            await self.send("conversation.item.input_audio_transcription.delta",
                            item_id=item_id, content_index=0, delta=word)  # fmt: skip
        if failed:
            await self.send("conversation.item.input_audio_transcription.failed",
                            item_id=item_id, content_index=0,
                            error={"type": "transcription_error", "code": "audio_unintelligible",
                                   "message": "The audio could not be transcribed.",
                                   "param": None})  # fmt: skip
            return
        transcript = "".join(words).strip()
        event: dict[str, Any] = {"item_id": item_id, "content_index": 0, "transcript": transcript}
        if self.streaming:
            event["usage"] = {"type": "duration", "seconds": round(seconds, 3)}
        else:
            event["usage"] = {"type": "tokens", "total_tokens": 22, "input_tokens": 13,
                              "input_token_details": {"text_tokens": 0, "audio_tokens": 13},
                              "output_tokens": 9}  # fmt: skip
        if self.model.startswith("gpt-transcribe"):
            event["languages"] = [{"code": "en"}]
        if self.session.get("include") and not self.streaming:
            event["logprobs"] = [{"token": w, "logprob": -0.1, "bytes": list(w.encode())}
                                 for w in words]  # fmt: skip
        await self.send("conversation.item.input_audio_transcription.completed", **event)
