"""A fake GPT-Live server (``/v1/live/sessions``) for offline tests and demos.

It speaks the Live WebSocket protocol (JSON events, base64 PCM16) on ``127.0.0.1`` with a
random port, and behaves like a full-duplex model in the ways an engine cares about:

* ``session.start`` -> ``session.started`` (resolved config, ``id``, ``expires_at``); a
  startup error, an HTTP rejection or a missing/wrong API key on request;
* **clocked by input**: every 100 ms of client audio is one model step producing 100 ms
  of agent audio (``session.output_audio.delta``) while it speaks — plus quiet noise
  between utterances with ``continuous=True`` — and ``session.usage.updated`` every
  second of audio;
* scripted turns: when the user stops talking (energy based) the fake sends the turn's
  user transcript (``session.input_transcript.delta`` with ``start_ms``/``end_ms``) and
  speaks the reply (``session.output_transcript.delta``, one fragment per word);
* full duplex: user speech during a reply makes the fake yield after ``yield_after``
  seconds of overlap;
* **delegation** after a reply (``FakeTurn.delegate``): Responses delegation replays the
  backend's ``response.event`` stream (a function call, then — after
  ``response.item.create`` + ``response.create`` — a text answer) and client delegation
  sends ``session.delegation.created`` and waits for a commentary/thinking append with
  that ``delegation_id``; the result is then spoken (``FakeDelegation.answer``);
* ``session.instructions.append`` asking to say something verbatim is spoken (greetings),
  commentary without a delegation id too; every append is acknowledged;
* mute/unmute (acknowledged; muted audio is not heard), ``session.close`` ->
  ``session.closed``, expiry (``session.closed`` reason ``expired`` at ``expires_at``),
  abrupt drops and ``error`` events.

Client events that break the protocol are recorded in :attr:`FakeLiveServer.errors`
(tests assert it stays empty).
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import itertools
import json
import re
import time
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from http import HTTPStatus
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from ..audio.frame import AudioFrame
from ..providers.mock import synth_speech
from ..utils.aio import cancel_and_wait

if TYPE_CHECKING:
    from websockets.asyncio.server import Server, ServerConnection
    from websockets.http11 import Request, Response

__all__ = ["FakeDelegation", "FakeLiveServer", "FakeLiveSession", "FakeTurn", "FakeUtterance"]

FRAME = 0.1
"""One model step (seconds of client audio)."""
PATH = "/v1/live/sessions"
_SAY = re.compile(r'"(.+)"\s*$', re.S)
_CLIENT_EVENTS = frozenset(
    {
        "session.start",
        "session.update",
        "session.input_audio.append",
        "session.input_audio.mute",
        "session.input_audio.unmute",
        "session.instructions.append",
        "session.thinking.append",
        "session.commentary.append",
        "response.item.create",
        "response.create",
        "session.close",
    }
)


@dataclass
class FakeDelegation:
    """Work the fake model delegates after its reply."""

    name: str = "get_weather"
    """Responses delegation: the function the backend calls."""
    arguments: Mapping[str, Any] = field(default_factory=lambda: {"city": "Paris"})
    answer: str = "{output}"
    """Said once the result arrives (``{output}`` = the function output / appended text)."""


@dataclass
class FakeTurn:
    """One scripted exchange: what the user said and what the fake answers."""

    reply: str
    user: str = ""
    """The user's transcript (sent when the user stops talking)."""
    delegate: FakeDelegation | None = None


@dataclass
class FakeUtterance:
    """Something the fake model said (or started to say)."""

    text: str
    kind: Literal["greeting", "reply", "answer", "commentary"]
    start_step: int
    frames: int
    spoken: int = 0

    @property
    def completed(self) -> bool:
        return self.spoken >= self.frames


@dataclass
class FakeLiveSession:
    """One client connection (one Live session)."""

    ws: ServerConnection
    headers: dict[str, str]
    session_id: str
    config: dict[str, Any] = field(default_factory=dict)
    """The ``session`` object of ``session.start``."""
    events: list[dict[str, Any]] = field(default_factory=list)
    """Every client event except audio appends."""
    utterances: list[FakeUtterance] = field(default_factory=list)
    steps: int = 0
    muted: bool = False
    close_reason: str | None = None
    closed: asyncio.Event = field(default_factory=asyncio.Event)

    @property
    def delegation_type(self) -> str:
        delegation = self.config.get("delegation")
        if isinstance(delegation, Mapping) and delegation.get("type") == "responses":
            return "responses"
        return "client"

    def of(self, etype: str) -> list[dict[str, Any]]:
        return [e for e in self.events if e.get("type") == etype]


class FakeLiveServer:
    """In-process fake of the GPT-Live WebSocket endpoint (see the module docstring).

    Args:
        turns: scripted exchanges, in order (``str`` = a reply); then ``default_reply``.
        default_reply: reply once the script is exhausted.
        api_key: required ``Authorization: Bearer`` key (``None``: any).
        words_per_second: speaking rate of the synthetic speech.
        respond_after: user silence (s) after which the fake replies.
        yield_after: overlapping user speech (s) after which the fake stops talking.
        user_threshold_db: level of client audio that counts as user speech.
        continuous: send (quiet) audio every step, also while not speaking.
        session_duration: ``expires_at`` = start + this; at that time the session closes
            with reason ``expired``.
        reject_status: reject the WebSocket handshake with this HTTP status.
        start_error: answer ``session.start`` with this ``error`` object.
    """

    def __init__(
        self,
        *,
        turns: Sequence[FakeTurn | str] = (),
        default_reply: str = "Okay.",
        api_key: str | None = "sk-test",
        words_per_second: float = 5.0,
        respond_after: float = 0.4,
        yield_after: float = 0.4,
        user_threshold_db: float = -40.0,
        continuous: bool = True,
        session_duration: float = 3600.0,
        reject_status: int | None = None,
        start_error: Mapping[str, Any] | None = None,
    ) -> None:
        self.turns = [FakeTurn(t) if isinstance(t, str) else t for t in turns]
        self.default_reply = default_reply
        self.api_key = api_key
        self.words_per_second = words_per_second
        self.respond_after = respond_after
        self.yield_after = yield_after
        self.user_threshold_db = user_threshold_db
        self.continuous = continuous
        self.session_duration = session_duration
        self.reject_status = reject_status
        self.start_error = start_error
        self.sessions: list[FakeLiveSession] = []
        self.errors: list[str] = []
        """Protocol violations detected in client events."""
        self.port = 0
        self._turn_index = 0
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

    async def __aenter__(self) -> FakeLiveServer:
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    @property
    def url(self) -> str:
        """Base URL for ``OpenAILiveEngine(base_url=...)`` (``/live/sessions`` is appended)."""
        return f"ws://127.0.0.1:{self.port}/v1"

    @property
    def session(self) -> FakeLiveSession:
        """The most recent session."""
        if not self.sessions:
            raise RuntimeError("no client connected yet")
        return self.sessions[-1]

    @property
    def utterances(self) -> list[FakeUtterance]:
        return [u for s in self.sessions for u in s.utterances]

    # ------------------------------------------------------------------ control
    async def drop(self, code: int = 1011, reason: str = "internal error") -> None:
        """Close the most recent connection abruptly (no ``session.closed``)."""
        await self.session.ws.close(code, reason)

    async def close_session(self, reason: str = "expired") -> None:
        """End the most recent session with ``session.closed`` (``reason``)."""
        await _finish(self.session, reason)

    async def send_error(
        self, message: str, *, code: str | None = None, etype: str = "invalid_request_error"
    ) -> None:
        """Send an ``error`` event on the most recent session."""
        error = {"type": etype, "code": code, "message": message}
        await self.session.ws.send(
            json.dumps({"type": "error", "event_id": _eid(), "error": error})
        )

    def _next_turn(self) -> FakeTurn:
        if self._turn_index < len(self.turns):
            self._turn_index += 1
            return self.turns[self._turn_index - 1]
        return FakeTurn(self.default_reply)

    # ----------------------------------------------------------------- protocol
    def _process_request(self, connection: ServerConnection, request: Request) -> Response | None:
        if self.reject_status is not None:
            status = HTTPStatus(self.reject_status)
            return connection.respond(status, f"{status.phrase} (fake live)\n")
        if request.path.split("?")[0] != PATH:
            return connection.respond(HTTPStatus.NOT_FOUND, "unknown endpoint\n")
        if "?" in request.path:
            self.errors.append("the Live endpoint takes no query parameters")
        auth = request.headers.get("Authorization", "")
        if self.api_key is not None and auth != f"Bearer {self.api_key}":
            return connection.respond(HTTPStatus.UNAUTHORIZED, "invalid api key\n")
        return None

    async def _handler(self, ws: ServerConnection) -> None:
        from websockets.exceptions import ConnectionClosed

        headers = dict(ws.request.headers.raw_items()) if ws.request is not None else {}
        session = FakeLiveSession(ws, headers, f"live_{len(self.sessions) + 1:03d}")
        self.sessions.append(session)
        try:
            await _Model(self, session).run()
        except ConnectionClosed:
            pass
        finally:
            session.closed.set()


_ids = itertools.count(1)


def _eid() -> str:
    return f"event_{next(_ids):06d}"


async def _send(session: FakeLiveSession, event: dict[str, Any]) -> None:
    event.setdefault("event_id", _eid())
    await session.ws.send(json.dumps(event))


async def _finish(
    session: FakeLiveSession, reason: str, client_event_id: str | None = None
) -> None:
    if session.close_reason is not None:
        return
    session.close_reason = reason
    event: dict[str, Any] = {
        "type": "session.closed",
        "reason": reason,
        "session": {"id": session.session_id, "status": "active", **session.config},
        "usage": {"seconds": round(session.steps * FRAME, 3)},
    }
    if client_event_id:
        event["client_event_id"] = client_event_id
    with contextlib.suppress(Exception):
        await _send(session, event)
        await session.ws.close(1000, reason)


class _Model:
    """The event loop of one session: decode input, step, speak, delegate."""

    def __init__(self, server: FakeLiveServer, session: FakeLiveSession) -> None:
        self.s = server
        self.session = session
        self.started = False
        self.rate = 24_000
        self.pending = np.zeros(0, dtype=np.int16)
        self.expires_at = 0.0
        self.utt: FakeUtterance | None = None
        self.audio: AudioFrame | None = None
        self.words: dict[int, list[str]] = {}
        self.queue: deque[tuple[str, Literal["greeting", "reply", "answer", "commentary"]]] = (
            deque()
        )
        self.turn: FakeTurn | None = None
        self.heard = 0.0
        self.heard_from = 0
        self.silence = 0.0
        self.overlap = 0.0
        self.delegations: dict[str, FakeDelegation] = {}
        """Open delegation id -> what to say with its result."""
        self.calls: dict[str, tuple[str, FakeDelegation]] = {}
        """Responses call id -> (delegation id, delegation)."""
        self.outputs: dict[str, str] = {}
        self.rng = np.random.default_rng(0)

    async def run(self) -> None:
        async for raw in self.session.ws:
            if not isinstance(raw, str):
                self.s.errors.append("binary message (the Live protocol is JSON text)")
                continue
            try:
                event = json.loads(raw)
            except ValueError:
                self.s.errors.append(f"invalid JSON: {raw[:40]!r}")
                continue
            if not isinstance(event, dict) or event.get("type") not in _CLIENT_EVENTS:
                self.s.errors.append(f"unknown client event {event!r:.80}")
                continue
            await self.handle(event)

    # --------------------------------------------------------------- client events
    async def handle(self, event: dict[str, Any]) -> None:
        etype = event["type"]
        session = self.session
        if etype != "session.input_audio.append":
            session.events.append(event)
        if session.close_reason is not None:
            return
        if etype == "session.start":
            await self.start(event)
            return
        if not self.started:
            self.s.errors.append(f"{etype} before session.started")
            return
        ack = {"client_event_id": event.get("event_id")} if event.get("event_id") else {}
        if etype == "session.input_audio.append":
            await self.audio_in(event.get("audio"))
        elif etype in ("session.input_audio.mute", "session.input_audio.unmute"):
            session.muted = etype.endswith(".mute")
            done = "muted" if session.muted else "unmuted"
            await _send(session, {"type": f"session.input_audio.{done}", **ack})
        elif etype.endswith(".append"):
            await self.append(event, ack)
        elif etype == "session.update":
            if session.delegation_type != "responses":
                await self.reject(event, "immutable_field_update", "session.delegation.type")
                return
            update = event.get("session", {}).get("delegation", {}).get("responses", {})
            session.config["delegation"]["responses"].update(update)
            await _send(session, {"type": "session.updated", "session": session.config, **ack})
        elif etype == "response.item.create":
            await self.item_create(event)
        elif etype == "response.create":
            await self.response_create(event)
        elif etype == "session.close":
            await _finish(session, "close_requested", event.get("event_id"))

    async def reject(self, event: dict[str, Any], code: str, param: str | None = None) -> None:
        error = {
            "type": "invalid_request_error",
            "code": code,
            "message": f"rejected {event.get('type')} ({code})",
            "param": param,
            "client_event_id": event.get("event_id"),
        }
        await _send(self.session, {"type": "error", "error": error})

    async def start(self, event: dict[str, Any]) -> None:
        session = self.session
        if self.started:
            self.s.errors.append("session.start sent twice")
            return
        config = event.get("session")
        if not isinstance(config, dict) or config.get("model") != "gpt-live-1":
            self.s.errors.append(f"session.start without the gpt-live-1 model: {config!r:.80}")
            config = config if isinstance(config, dict) else {}
        if self.s.start_error is not None:
            error = {**self.s.start_error, "client_event_id": event.get("event_id")}
            await _send(session, {"type": "error", "error": error})
            await session.ws.close(1008, "invalid session")
            return
        fmt = config.get("audio", {}).get("format", {})
        self.rate = int(fmt.get("rate", 24_000))
        if fmt.get("type", "audio/pcm") != "audio/pcm" or self.rate not in (16_000, 24_000):
            self.s.errors.append(f"unsupported audio format {fmt!r}")
        for item in config.get("input", []):
            content = item.get("content", [{}])
            if item.get("role") not in ("developer", "user", "assistant") or len(content) != 1:
                self.s.errors.append(f"invalid input item {item!r:.80}")
        if len(config.get("input", [])) > 128:
            self.s.errors.append("more than 128 input messages")
        config.setdefault("delegation", {"type": "client"})
        config.setdefault("audio", {}).setdefault("output", {}).setdefault("voice", "marin")
        session.config = config
        self.expires_at = time.time() + self.s.session_duration
        self.started = True
        resolved = {
            "id": session.session_id,
            "status": "active",
            "expires_at": round(self.expires_at, 3),
            "input": [],
            **config,
        }
        started = {"type": "session.started", "session": resolved}
        if event.get("event_id"):
            started["client_event_id"] = event["event_id"]
        await _send(session, started)

    async def append(self, event: dict[str, Any], ack: dict[str, Any]) -> None:
        kind = event["type"].split(".")[1]
        content = event.get("content")
        if "delegation_id" not in event:
            self.s.errors.append(f"{event['type']} without the (nullable) delegation_id")
        if not isinstance(content, str) or not content.strip():
            self.s.errors.append(f"{event['type']} without text content")
            return
        if len(content) > 2000:
            self.s.errors.append(f"{event['type']} over 500 tokens ({len(content)} chars)")
        delegation_id = event.get("delegation_id")
        if delegation_id is not None:
            if self.session.delegation_type == "responses":
                await self.reject(event, "invalid_value", "delegation_id")
                self.s.errors.append("a delegation_id on an append with Responses delegation")
                return
            if delegation_id not in self.delegations:
                await self.reject(event, "invalid_value", "delegation_id")
                self.s.errors.append(f"append for unknown delegation {delegation_id!r}")
                return
        t = self.session.steps * FRAME * 1000
        await _send(
            self.session, {"type": f"session.{kind}.appended", "start_ms": t, "end_ms": t, **ack}
        )
        if kind == "instructions":
            match = _SAY.search(content)
            if match is not None:
                self.say(match.group(1), "greeting")
        elif kind == "commentary":
            if delegation_id is not None:
                answer = self.delegations.pop(delegation_id)
                self.say(answer.answer.format(output=content), "answer")
            else:
                self.say(content, "commentary")
        elif kind == "thinking" and delegation_id is not None:
            self.delegations.pop(delegation_id, None)  # a quiet result: nothing to say

    async def item_create(self, event: dict[str, Any]) -> None:
        if self.session.delegation_type != "responses":
            await self.reject(event, "invalid_request", "delegation")
            return
        item = event.get("item")
        if not isinstance(item, Mapping) or item.get("type") != "function_call_output":
            return  # user text for the backend: nothing to do in the fake
        call_id = str(item.get("call_id"))
        if call_id not in self.calls:
            self.s.errors.append(f"output for unknown call {call_id!r}")
            return
        output = item.get("output")
        if not isinstance(output, str):
            self.s.errors.append("function_call_output.output must be a string")
            output = str(output)
        self.outputs[call_id] = output

    async def response_create(self, event: dict[str, Any]) -> None:
        if self.session.delegation_type != "responses":
            await self.reject(event, "invalid_request", "delegation")
            return
        pending = [c for c in self.calls if c not in self.outputs]
        if pending:
            self.s.errors.append(f"response.create before the outputs of {pending}")
            return
        for call_id, output in list(self.outputs.items()):
            delegation_id, delegation = self.calls.pop(call_id)
            del self.outputs[call_id]
            rid = f"resp_{_eid()}"
            backend = [
                {"type": "response.created", "response": {"id": rid, "status": "in_progress"}},
                {
                    "type": "response.output_text.delta",
                    "item_id": "msg_1",
                    "output_index": 0,
                    "content_index": 0,
                    "delta": f"Backend: {output}",
                },
                {
                    "type": "response.completed",
                    "response": {
                        "id": rid,
                        "status": "completed",
                        "output": [],
                        "usage": {"input_tokens": 120, "output_tokens": 12},
                    },
                },
            ]
            for inner in backend:
                await _send(
                    self.session,
                    {"type": "response.event", "delegation_id": delegation_id, "event": inner},
                )
            self.say(delegation.answer.format(output=output), "answer")

    async def delegate(self, delegation: FakeDelegation) -> None:
        session = self.session
        delegation_id = f"del_{_eid()}"
        offset = session.steps * FRAME * 1000
        if session.delegation_type == "client":
            self.delegations[delegation_id] = delegation
            meta = {"id": delegation_id, "type": "delegation", "target": "client"}
            await _send(
                session,
                {"type": "session.delegation.created", "offset_ms": offset, "delegation": meta},
            )
            return
        rid = f"resp_{_eid()}"
        call_id = f"call_{_eid()}"
        self.calls[call_id] = (delegation_id, delegation)
        meta = {
            "id": delegation_id,
            "type": "delegation",
            "target": "responses",
            "response_id": rid,
        }
        await _send(
            session, {"type": "session.delegation.created", "offset_ms": offset, "delegation": meta}
        )
        call = {
            "type": "function_call",
            "id": f"fc_{call_id}",
            "call_id": call_id,
            "name": delegation.name,
            "arguments": json.dumps(dict(delegation.arguments)),
            "status": "completed",
        }
        backend = [
            {"type": "response.created", "response": {"id": rid, "status": "in_progress"}},
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {**call, "arguments": ""},
            },
            {"type": "response.output_item.done", "output_index": 0, "item": call},
            {
                "type": "response.completed",
                "response": {
                    "id": rid,
                    "status": "completed",
                    "output": [],
                    "usage": {"input_tokens": 100, "output_tokens": 20},
                },
            },
        ]
        for inner in backend:
            await _send(
                session, {"type": "response.event", "delegation_id": delegation_id, "event": inner}
            )

    # --------------------------------------------------------------------- audio
    async def audio_in(self, audio: Any) -> None:
        if not isinstance(audio, str):
            self.s.errors.append("session.input_audio.append without base64 audio")
            return
        data = base64.b64decode(audio)
        if len(data) % 2:
            self.s.errors.append("odd PCM16 chunk length")
            data = data[:-1]
        pcm = np.frombuffer(data, dtype=np.int16)
        self.pending = np.concatenate((self.pending, pcm))
        n = round(FRAME * self.rate)
        while len(self.pending) >= n and self.session.close_reason is None:
            frame = AudioFrame(self.pending[:n].tobytes(), self.rate)
            self.pending = self.pending[n:]
            await self.step(frame)

    def say(self, text: str, kind: Literal["greeting", "reply", "answer", "commentary"]) -> None:
        self.queue.append((text, kind))

    def _start_utterance(self) -> None:
        text, kind = self.queue.popleft()
        words = text.split()
        frames = max(1, round(len(words) / self.s.words_per_second / FRAME))
        self.utt = FakeUtterance(text, kind, self.session.steps, frames)
        self.session.utterances.append(self.utt)
        self.audio = synth_speech(frames * FRAME, self.rate, frequency=180.0)
        self.words = {}
        for i, word in enumerate(words):
            self.words.setdefault(i * frames // len(words), []).append(
                ("" if i == 0 else " ") + word
            )

    async def step(self, user: AudioFrame) -> None:
        s, session = self.s, self.session
        step = session.steps
        session.steps += 1
        if time.time() >= self.expires_at:
            await _finish(session, "expired")
            return
        loud = not session.muted and user.dbfs() >= s.user_threshold_db
        if self.utt is not None:
            self.overlap = self.overlap + FRAME if loud else 0.0
            if self.overlap >= s.yield_after - 1e-9:
                self.utt = None  # full duplex: yield to the user
                self.turn = None  # (and forget what came after the reply)
                self.heard, self.heard_from, self.silence = self.overlap, step, 0.0
        elif loud:
            if self.heard == 0:
                self.heard_from = step
            self.heard += FRAME
            self.silence = 0.0
        elif self.heard > 0:
            self.silence += FRAME
            if self.silence >= s.respond_after - 1e-9:
                self.heard = self.silence = 0.0
                turn = s._next_turn()
                if turn.user:
                    end = (step - round(s.respond_after / FRAME) + 1) * FRAME * 1000
                    words = turn.user.split()
                    span = (end - self.heard_from * FRAME * 1000) / len(words)
                    for i, word in enumerate(words):
                        start = self.heard_from * FRAME * 1000 + i * span
                        delta = {
                            "type": "session.input_transcript.delta",
                            "delta": ("" if i == 0 else " ") + word,
                            "start_ms": round(start),
                            "end_ms": round(start + span),
                        }
                        await _send(session, delta)
                self.turn = turn
                self.say(turn.reply, "reply")
        if self.utt is None and self.queue:
            self._start_utterance()
        utt = self.utt
        if utt is not None and self.audio is not None:
            i = utt.spoken
            n = round(FRAME * self.rate) * 2
            out: AudioFrame | None = AudioFrame(self.audio.data[i * n : (i + 1) * n], self.rate)
            words = self.words.get(i, [])
            utt.spoken += 1
            if utt.completed:
                self.utt = None
                self.overlap = 0.0
        else:
            words = []
            out = None
            if s.continuous:
                noise = self.rng.normal(0.0, 10 ** (-70 / 20), round(FRAME * self.rate))
                out = AudioFrame.from_numpy(noise.astype(np.float32), self.rate)
        if out is not None:
            await _send(session, {"type": "session.output_audio.delta", "delta": out.to_base64()})
        t = step * FRAME * 1000
        for word in words:
            delta = {
                "type": "session.output_transcript.delta",
                "delta": word,
                "start_ms": round(t),
                "end_ms": round(t + FRAME * 1000),
            }
            await _send(session, delta)
        if utt is not None and utt.completed and utt.kind == "reply":
            done, self.turn = self.turn, None
            if done is not None and done.delegate is not None:
                await self.delegate(done.delegate)
        if session.steps % round(1 / FRAME) == 0:
            usage = {"seconds": round(session.steps * FRAME, 3)}
            await _send(session, {"type": "session.usage.updated", "usage": usage})
