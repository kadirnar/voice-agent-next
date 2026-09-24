"""A fake Gemini Live API server (``BidiGenerateContent``) for offline tests and demos.

It speaks the real JSON protocol over a local WebSocket (``websockets.serve`` on
``127.0.0.1``, random port) and behaves like the service in the ways an engine cares
about:

* ``setup`` -> ``setupComplete``; API key check (``x-goog-api-key`` header or ``?key=``);
* server-side VAD (energy based) on ``realtimeInput.audio`` with ``voiceActivity``
  start/end messages, or manual ``activityStart``/``activityEnd``;
* scripted replies: ``inputTranscription`` of the user turn, synthetic 24 kHz speech in
  ``serverContent.modelTurn`` chunks, ``outputTranscription``, ``generationComplete``,
  ``turnComplete`` + ``usageMetadata``;
* function calling: ``toolCall`` (blocking or ``NON_BLOCKING`` declarations),
  ``toolResponse`` with ``scheduling``, ``toolCallCancellation`` on barge-in;
* barge-in: user speech during a reply -> ``interrupted`` (then ``turnComplete``);
* session resumption: ``sessionResumptionUpdate`` handles; resuming with a handle restores
  the session and rolls its audio back to the point the handle was issued;
* ``goAway`` (:meth:`FakeGeminiLiveServer.go_away`) and abrupt drops
  (:meth:`FakeGeminiLiveServer.drop`); HTTP rejection and error closes.

Example::

    async with FakeGeminiLiveServer(replies=["Hi there!"]) as server:
        engine = GeminiLiveEngine(api_key=server.api_key, base_url=server.url)
        session = AgentSession(engine)
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from http import HTTPStatus
from typing import TYPE_CHECKING, Any, TypeAlias
from urllib.parse import parse_qs, urlsplit

from ..audio.frame import AudioFrame
from ..providers.energy import EnergyVAD
from ..providers.mock import synth_speech
from ..utils.aio import cancel_and_wait
from ..utils.ids import new_id
from ..vad import VADEventType, VADOptions

if TYPE_CHECKING:
    from websockets.asyncio.server import Server, ServerConnection
    from websockets.http11 import Request, Response

__all__ = [
    "FakeConnection",
    "FakeGeminiLiveServer",
    "FakeReply",
    "FakeSession",
    "FakeToolCall",
]

SERVICE_PATH = "/ws/google.ai.generativelanguage.v1beta.GenerativeService."
INPUT_MIME = "audio/pcm;rate=16000"
OUTPUT_RATE = 24_000
AUDIO_TOKENS_PER_SECOND = 25
_VERBATIM = re.compile(r'Say exactly the following, verbatim, and nothing else: "(.*)"', re.S)


def _duration(seconds: float) -> str:
    """Seconds -> protobuf ``Duration`` JSON."""
    return f"{max(0.0, seconds):.3f}s"


def _words(text: str) -> list[str]:
    """Transcript deltas the way Gemini sends them (``["Hi", " there"]``)."""
    return re.findall(r"\s*\S+", text)


@dataclass(slots=True)
class FakeToolCall:
    """A scripted function call made by the fake model."""

    name: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class FakeReply:
    """One scripted model turn: tool calls first (if any), then speech (if any)."""

    text: str = ""
    tool_calls: list[FakeToolCall] = field(default_factory=list)


ReplyItem: TypeAlias = str | FakeToolCall | list[FakeToolCall] | FakeReply


def _as_reply(item: ReplyItem) -> FakeReply:
    if isinstance(item, FakeReply):
        return item
    if isinstance(item, str):
        return FakeReply(text=item)
    if isinstance(item, FakeToolCall):
        return FakeReply(tool_calls=[item])
    return FakeReply(tool_calls=list(item))


@dataclass
class FakeSession:
    """Server-side conversation state. It survives resumption, like the real service."""

    id: str = field(default_factory=lambda: new_id("session_"))
    audio: bytearray = field(default_factory=bytearray)
    """User audio consumed by the session (rolled back to the handle's point on resume)."""
    history: list[dict[str, Any]] = field(default_factory=list)
    handles: dict[str, int] = field(default_factory=dict)
    """Resumption handle -> ``len(audio)`` when it was issued."""
    pending_calls: dict[str, str] = field(default_factory=dict)
    resumptions: int = 0


class FakeConnection:
    """One client WebSocket connection to :class:`FakeGeminiLiveServer`."""

    def __init__(
        self,
        server: FakeGeminiLiveServer,
        ws: ServerConnection,
        session: FakeSession,
        setup: dict[str, Any],
    ) -> None:
        self.server = server
        self.ws = ws
        self.session = session
        self.setup = setup
        request = ws.request
        self.path = request.path if request is not None else ""
        self.headers: dict[str, str] = (
            {k.lower(): v for k, v in request.headers.raw_items()} if request is not None else {}
        )
        self.messages: list[dict[str, Any]] = []
        """Every client message after ``setup``."""
        self.audio = bytearray()
        """User audio received on this connection."""
        self.tool_responses: list[dict[str, Any]] = []
        self.client_contents: list[dict[str, Any]] = []
        self.user_turns: list[str] = []
        self.handles: list[str] = []
        self.interruptions = 0
        self.audio_stream_ends = 0
        self.close_code: int | None = None
        self.close_reason = ""
        self.closed = asyncio.Event()
        self._vad = EnergyVAD(sample_rate=16_000, options=server.vad_options).stream()
        self._reply_task: asyncio.Task[None] | None = None
        self._queued: FakeReply | None = None
        self._waiters: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._speaking = False
        history_cfg = setup.get("historyConfig") or {}
        self._initial_history = bool(history_cfg.get("initialHistoryInClientContent"))
        realtime = setup.get("realtimeInputConfig") or {}
        detection = realtime.get("automaticActivityDetection") or {}
        self.manual_activity = bool(detection.get("disabled"))
        self._interrupts = realtime.get("activityHandling") != "NO_INTERRUPTION"
        declarations = [
            d for tool in setup.get("tools") or () for d in tool.get("functionDeclarations") or ()
        ]
        self.non_blocking = {d["name"] for d in declarations if d.get("behavior") == "NON_BLOCKING"}
        self._resumption = "sessionResumption" in setup
        self._input_transcription = "inputAudioTranscription" in setup
        self._output_transcription = "outputAudioTranscription" in setup

    # ---------------------------------------------------------------- properties
    @property
    def resumed_from(self) -> str | None:
        """The resumption handle this connection was opened with."""
        handle = (self.setup.get("sessionResumption") or {}).get("handle")
        return str(handle) if handle else None

    @property
    def generating(self) -> bool:
        return self._reply_task is not None and not self._reply_task.done()

    # ------------------------------------------------------------------- output
    async def send(self, message: dict[str, Any]) -> None:
        """Send a raw server message (ignored if the connection is gone)."""
        from websockets.exceptions import ConnectionClosed

        with contextlib.suppress(ConnectionClosed):
            await self.ws.send(json.dumps(message))

    async def close(self, code: int = 1000, reason: str = "") -> None:
        await self.ws.close(code, reason)

    async def issue_handle(self) -> str | None:
        """Send a resumable ``sessionResumptionUpdate`` (if resumption was configured)."""
        if not self._resumption:
            return None
        handle = new_id("handle_")
        self.session.handles[handle] = len(self.session.audio)
        self.server._by_handle[handle] = self.session
        self.handles.append(handle)
        await self.send({"sessionResumptionUpdate": {"newHandle": handle, "resumable": True}})
        return handle

    # -------------------------------------------------------------------- input
    async def _run(self) -> None:
        from websockets.exceptions import ConnectionClosed

        try:
            async for raw in self.ws:
                try:
                    message = json.loads(raw)
                except ValueError:
                    await self._protocol_error("message is not JSON")
                    return
                self.messages.append(message)
                await self._handle(message)
        except ConnectionClosed:
            pass
        finally:
            await cancel_and_wait(self._reply_task)
            for waiter in self._waiters.values():
                waiter.cancel()
            self.close_code = self.ws.close_code
            self.close_reason = self.ws.close_reason or ""
            self.closed.set()

    async def _protocol_error(self, what: str) -> None:
        self.server.errors.append(what)
        await self.close(1007, "Request contains an invalid argument.")

    async def _handle(self, message: Any) -> None:
        kinds = ("setup", "clientContent", "realtimeInput", "toolResponse")
        if not isinstance(message, dict) or len(message) != 1 or next(iter(message)) not in kinds:
            await self._protocol_error(f"a client message needs exactly one of {kinds}")
            return
        kind, body = next(iter(message.items()))
        if not isinstance(body, dict):
            await self._protocol_error(f"{kind} must be an object")
        elif kind == "setup":
            await self._protocol_error("setup may only be sent once")
        elif kind == "realtimeInput":
            await self._on_realtime_input(body)
        elif kind == "clientContent":
            await self._on_client_content(body)
        else:
            await self._on_tool_response(body)

    async def _on_realtime_input(self, ri: dict[str, Any]) -> None:
        if "audio" in ri:
            audio = ri["audio"]
            if not isinstance(audio, dict) or audio.get("mimeType") != INPUT_MIME:
                await self._protocol_error(f"audio must be {INPUT_MIME}")
                return
            pcm = base64.b64decode(audio.get("data") or "")
            if len(pcm) % 2:
                await self._protocol_error("audio must be 16-bit PCM")
                return
            self.audio += pcm
            self.session.audio += pcm
            if self.manual_activity:
                return
            for ev in self._vad.push_audio(AudioFrame(pcm, 16_000)):
                if ev.type == VADEventType.START_OF_SPEECH:
                    await self._speech_started(ev.audio_time - ev.speech_duration)
                elif ev.type == VADEventType.END_OF_SPEECH:
                    await self._speech_ended(ev.audio_time - ev.silence_duration)
        elif ri.get("audioStreamEnd"):
            self.audio_stream_ends += 1
            if self._speaking:  # hybrid VAD: finalize the turn right away
                self._vad.reset()
                await self._speech_ended(len(self.audio) / 32_000)
        elif "activityStart" in ri or "activityEnd" in ri:
            if not self.manual_activity:
                await self._protocol_error("activity signals need automatic detection disabled")
            elif "activityStart" in ri:
                await self._barge_in()
            else:
                await self._user_turn()
        elif "text" in ri:
            await self._barge_in()
            await self._user_turn(text=str(ri["text"]))
        else:
            await self._protocol_error(f"unsupported realtimeInput {sorted(ri)}")

    async def _speech_started(self, offset: float) -> None:
        self._speaking = True
        if self.server.voice_activity:
            activity = {"type": "ACTIVITY_START", "audioOffset": _duration(offset)}
            await self.send({"voiceActivity": activity})
        await self._barge_in()

    async def _speech_ended(self, offset: float) -> None:
        self._speaking = False
        if self.server.voice_activity:
            activity = {"type": "ACTIVITY_END", "audioOffset": _duration(offset)}
            await self.send({"voiceActivity": activity})
        await self._user_turn()

    async def _barge_in(self) -> None:
        """User activity interrupts the model (``START_OF_ACTIVITY_INTERRUPTS``)."""
        if not self.generating or not self._interrupts:
            return
        await cancel_and_wait(self._reply_task)
        self._queued = None
        self.interruptions += 1
        await self.send({"serverContent": {"interrupted": True}})
        cancelled = list(self.session.pending_calls)
        if cancelled:
            self.session.pending_calls.clear()
            await self.send({"toolCallCancellation": {"ids": cancelled}})
        await self.send({"serverContent": {"turnComplete": True}})
        await self.issue_handle()

    async def _user_turn(self, text: str | None = None) -> None:
        server = self.server
        late: list[str] = []
        if text is None:
            text = server._next_transcript()
            if self._input_transcription and text:
                words = _words(text)
                if server.interim_transcription:  # low-latency hypotheses first
                    for i in range(1, len(words) + 1):
                        interim = {"text": "".join(words[:i]).strip()}
                        await self.send({"serverContent": {"interimInputTranscription": interim}})
                if server.late_transcription:
                    late = words
                else:
                    for word in words:
                        await self.send({"serverContent": {"inputTranscription": {"text": word}}})
        self.user_turns.append(text)
        self.session.history.append({"role": "user", "parts": [{"text": text}]})
        self._start_reply(server._next_reply(text), late_transcript=late)

    async def _on_client_content(self, cc: dict[str, Any]) -> None:
        self.client_contents.append(cc)
        turns = cc.get("turns") or []
        if not isinstance(turns, list) or any(
            not isinstance(t, dict) or t.get("role") not in ("user", "model") for t in turns
        ):
            await self._protocol_error("clientContent.turns must be Content with a role")
            return
        self.session.history.extend(turns)
        if self._initial_history:
            if cc.get("turnComplete"):
                self._initial_history = False
            return
        if cc.get("turnComplete"):
            await self._barge_in()  # client content interrupts the current generation
            text = ""
            for turn in turns:
                if turn["role"] == "user":
                    text = " ".join(p.get("text", "") for p in turn.get("parts") or ())
            match = _VERBATIM.search(text)
            reply = FakeReply(match.group(1)) if match else self.server._next_reply(text or None)
            self._start_reply(reply)

    async def _on_tool_response(self, tr: dict[str, Any]) -> None:
        responses = tr.get("functionResponses")
        if not isinstance(responses, list) or not responses:
            await self._protocol_error("toolResponse.functionResponses must be a non-empty list")
            return
        for fr in responses:
            self.tool_responses.append(fr)
            call_id = fr.get("id")
            name = self.session.pending_calls.pop(call_id, None)
            if name is None:
                await self._protocol_error(f"unknown function call id {call_id!r}")
                return
            if fr.get("name") != name or not isinstance(fr.get("response"), dict):
                await self._protocol_error(f"malformed FunctionResponse {fr!r}")
                return
            waiter = self._waiters.pop(call_id, None)
            if waiter is not None:  # blocking call: the paused turn continues
                if not waiter.done():
                    waiter.set_result(fr)
                continue
            scheduling = fr.get("scheduling", "WHEN_IDLE")
            if scheduling == "SILENT":
                continue
            reply = self.server._next_reply(None)
            if scheduling == "INTERRUPT":
                await self._barge_in()
            self._start_reply(reply)  # WHEN_IDLE: queued behind the current reply

    # ------------------------------------------------------------------ replies
    def _start_reply(self, reply: FakeReply, *, late_transcript: Sequence[str] = ()) -> None:
        """Start a model turn, or queue it behind the running one (never two at once)."""
        if self.generating:
            self._queued = reply
            return
        self._launch(reply, list(late_transcript))

    def _launch(self, reply: FakeReply, late_transcript: list[str]) -> None:
        self._reply_task = asyncio.create_task(self._reply(reply, late_transcript))

    async def _reply(self, reply: FakeReply, late_transcript: list[str]) -> None:
        if self._resumption:
            await self.send({"sessionResumptionUpdate": {"resumable": False}})
        text = reply.text
        if reply.tool_calls:
            calls: list[dict[str, Any]] = [
                {"id": new_id("fc_"), "name": c.name, "args": c.args} for c in reply.tool_calls
            ]
            for call in calls:
                self.session.pending_calls[call["id"]] = call["name"]
            await self.send({"toolCall": {"functionCalls": calls}})
            blocking = [c["id"] for c in calls if c["name"] not in self.non_blocking]
            if blocking:  # the model waits for the results, then answers in the same turn
                loop = asyncio.get_running_loop()
                futures = [self._waiters.setdefault(i, loop.create_future()) for i in blocking]
                await asyncio.gather(*futures)
                text = self.server._next_reply(None).text
        seconds = await self._speak(text, late_transcript) if text else 0.0
        await self.send({"serverContent": {"generationComplete": True}})
        usage = {
            "promptTokenCount": 120,
            "responseTokenCount": round(seconds * AUDIO_TOKENS_PER_SECOND),
            "promptTokensDetails": [
                {"modality": "TEXT", "tokenCount": 100},
                {"modality": "AUDIO", "tokenCount": 20},
            ],
            "responseTokensDetails": [
                {"modality": "AUDIO", "tokenCount": round(seconds * AUDIO_TOKENS_PER_SECOND)}
            ],
        }
        await self.send({"serverContent": {"turnComplete": True}, "usageMetadata": usage})
        self.session.history.append({"role": "model", "parts": [{"text": text}]})
        if not self.session.pending_calls:
            await self.issue_handle()
        queued, self._queued = self._queued, None
        if queued is not None:
            self._launch(queued, [])  # this task is finishing: the next turn starts now

    async def _speak(self, text: str, late_transcript: list[str]) -> float:
        server = self.server
        duration = max(0.2, len(text) / server.chars_per_second)
        audio = synth_speech(duration, OUTPUT_RATE, frequency=server.frequency)
        words = _words(text)
        chunks = max(1, math.ceil(duration / server.chunk_duration))
        spoken = 0
        for i in range(chunks):
            start = i * server.chunk_duration
            chunk = audio.slice(start, min(duration, start + server.chunk_duration))
            if chunk:
                inline = {"mimeType": f"audio/pcm;rate={OUTPUT_RATE}", "data": chunk.to_base64()}
                turn = {"parts": [{"inlineData": inline}]}
                await self.send({"serverContent": {"modelTurn": turn}})
            if i == 0:
                for word in late_transcript:
                    await self.send({"serverContent": {"inputTranscription": {"text": word}}})
            if self._output_transcription:
                due = math.ceil(len(words) * (i + 1) / chunks)
                while spoken < due:
                    delta = {"text": words[spoken]}
                    await self.send({"serverContent": {"outputTranscription": delta}})
                    spoken += 1
            await asyncio.sleep(server.chunk_duration * server.realtime_factor)
        return duration


class FakeGeminiLiveServer:
    """In-process fake of the Gemini Live WebSocket API (see the module docstring).

    Args:
        replies: model turns, one per user turn / tool follow-up (like ``MockLLM``);
            a callable gets the user's transcript (or ``None``). Default: echo.
        transcripts: user transcripts, one per detected user turn (then
            ``default_transcript``).
        api_key: expected key (``None`` accepts anything).
        voice_activity: send ``voiceActivity`` start/end messages.
        interim_transcription: send ``interimInputTranscription`` hypotheses first.
        late_transcription: send the user's transcript after the first reply audio.
        realtime_factor: pacing of reply audio (0 = as fast as possible, 1 = real time).
        reject_status: reject the WebSocket handshake with this HTTP status.
        close_after_setup: ``(code, reason)`` to close with instead of ``setupComplete``.
        drop_after_setup: ``(code, reason)`` to close with right after ``setupComplete``.
    """

    def __init__(
        self,
        *,
        replies: Sequence[ReplyItem] | Callable[[str | None], ReplyItem] | None = None,
        transcripts: Sequence[str] | None = None,
        default_transcript: str = "hello",
        api_key: str | None = "fake-gemini-key",
        voice_activity: bool = True,
        interim_transcription: bool = False,
        late_transcription: bool = False,
        realtime_factor: float = 0.0,
        chunk_duration: float = 0.04,
        chars_per_second: float = 15.0,
        frequency: float = 220.0,
        vad_options: VADOptions | None = None,
        reject_status: int | None = None,
        close_after_setup: tuple[int, str] | None = None,
        drop_after_setup: tuple[int, str] | None = None,
    ) -> None:
        self.replies = replies
        self.transcripts = list(transcripts or [])
        self.default_transcript = default_transcript
        self.api_key = api_key
        self.voice_activity = voice_activity
        self.interim_transcription = interim_transcription
        self.late_transcription = late_transcription
        self.realtime_factor = realtime_factor
        self.chunk_duration = chunk_duration
        self.chars_per_second = chars_per_second
        self.frequency = frequency
        self.vad_options = vad_options or VADOptions(
            min_speech_duration=0.1, min_silence_duration=0.4
        )
        self.reject_status = reject_status
        self.close_after_setup = close_after_setup
        self.drop_after_setup = drop_after_setup
        self.sessions: list[FakeSession] = []
        self.connections: list[FakeConnection] = []
        self.setups: list[dict[str, Any]] = []
        self.errors: list[str] = []
        """Protocol violations detected in client messages (tests assert it stays empty)."""
        self.port = 0
        self._by_handle: dict[str, FakeSession] = {}
        self._reply_index = 0
        self._transcript_index = 0
        self._server: Server | None = None
        self._tasks: set[asyncio.Task[None]] = set()

    # ---------------------------------------------------------------- lifecycle
    async def start(self) -> None:
        from websockets.asyncio.server import serve

        self._server = await serve(
            self._handler, "127.0.0.1", 0, process_request=self._process_request, max_size=2**24
        )
        self.port = next(iter(self._server.sockets)).getsockname()[1]

    async def aclose(self) -> None:
        await cancel_and_wait(*self._tasks)
        if self._server is not None:
            self._server.close()
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self._server.wait_closed(), 5)
            self._server = None

    async def __aenter__(self) -> FakeGeminiLiveServer:
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    @property
    def url(self) -> str:
        """Base URL for ``GeminiLiveEngine(base_url=...)`` (the engine appends the path)."""
        return f"ws://127.0.0.1:{self.port}"

    @property
    def connection(self) -> FakeConnection:
        """The most recent connection."""
        if not self.connections:
            raise RuntimeError("no client connected yet")
        return self.connections[-1]

    # ------------------------------------------------------------------ control
    async def send(self, message: dict[str, Any]) -> None:
        """Push a raw server message to the most recent connection."""
        await self.connection.send(message)

    async def go_away(self, time_left: float = 1.0, *, abort: bool = True) -> None:
        """Send ``goAway`` and (if ``abort``) terminate the connection after ``time_left``."""
        conn = self.connection
        await conn.send({"goAway": {"timeLeft": _duration(time_left)}})
        if abort:
            task = asyncio.create_task(self._abort_later(conn, time_left))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

    async def drop(self, code: int = 1011, reason: str = "Internal error encountered.") -> None:
        """Close the most recent connection abruptly (server side)."""
        await self.connection.close(code, reason)

    async def _abort_later(self, conn: FakeConnection, delay: float) -> None:
        await asyncio.sleep(delay)
        if not conn.closed.is_set():
            await conn.close(1011, "The connection was aborted (ABORTED).")

    # ------------------------------------------------------------------ scripts
    def _next_transcript(self) -> str:
        if self._transcript_index < len(self.transcripts):
            self._transcript_index += 1
            return self.transcripts[self._transcript_index - 1]
        return self.default_transcript

    def _next_reply(self, user_text: str | None) -> FakeReply:
        if callable(self.replies):
            return _as_reply(self.replies(user_text))
        if self.replies is not None and self._reply_index < len(self.replies):
            self._reply_index += 1
            return _as_reply(self.replies[self._reply_index - 1])
        return FakeReply(text=f"You said: {user_text}" if user_text else "Okay.")

    # ----------------------------------------------------------------- protocol
    def _process_request(self, connection: ServerConnection, request: Request) -> Response | None:
        if self.reject_status is not None:
            status = HTTPStatus(self.reject_status)
            return connection.respond(status, f"{status.phrase} (fake Gemini Live)\n")
        if not urlsplit(request.path).path.startswith(SERVICE_PATH):
            return connection.respond(HTTPStatus.NOT_FOUND, "unknown endpoint\n")
        return None

    async def _handler(self, ws: ServerConnection) -> None:
        from websockets.exceptions import ConnectionClosed

        try:
            raw = await asyncio.wait_for(ws.recv(), 10)
        except (ConnectionClosed, TimeoutError):
            return
        try:
            first = json.loads(raw)
        except ValueError:
            first = None
        setup = first.get("setup") if isinstance(first, dict) and len(first) == 1 else None
        if not isinstance(setup, dict):
            self.errors.append("the first message must be {'setup': {...}}")
            await ws.close(1007, "Request contains an invalid argument.")
            return
        self.setups.append(setup)
        request = ws.request
        key = request.headers.get("x-goog-api-key") if request is not None else None
        if key is None and request is not None:
            keys = parse_qs(urlsplit(request.path).query).get("key")
            key = keys[0] if keys else None
        if self.api_key is not None and key != self.api_key:
            await ws.close(1007, "API key not valid. Please pass a valid API key.")
            return
        if self.close_after_setup is not None:
            await ws.close(*self.close_after_setup)
            return
        model = str(setup.get("model") or "")
        if not model.startswith("models/"):
            await ws.close(1008, f"{model} is not found for API version v1beta")
            return
        handle = (setup.get("sessionResumption") or {}).get("handle")
        if handle:
            session = self._by_handle.get(handle)
            if session is None:
                await ws.close(1007, "Invalid session resumption handle.")
                return
            del session.audio[session.handles[handle] :]
            session.resumptions += 1
        else:
            session = FakeSession()
            self.sessions.append(session)
        conn = FakeConnection(self, ws, session, setup)
        self.connections.append(conn)
        await conn.send({"setupComplete": {}})
        await conn.issue_handle()
        if self.drop_after_setup is not None:
            await ws.close(*self.drop_after_setup)
        await conn._run()
