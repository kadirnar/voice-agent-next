"""A fake Moshi server (``/api/chat``) for offline tests and demos.

It speaks the real binary protocol over a local WebSocket (``websockets.serve`` on
``127.0.0.1``, random port), with real Ogg/Opus pages (``sphn``, the codec of
``moshi.server``), and behaves like a full-duplex model in the ways an engine cares about:

* handshake as ``moshi.server`` (``0x00``) or the Rust ``moshi-backend`` (``0x04``
  metadata JSON, then ``0x00`` + protocol/model version);
* **output clocked by input**: every 80 ms frame of decoded client audio produces one
  80 ms frame of agent audio (``0x01``), exactly like the real server's step loop;
* scripted speech: an optional greeting, then one reply each time the user stops talking
  (energy based); synthetic 24 kHz speech plus its text tokens (``0x02``, one per word,
  ``" word"`` as the server turns SentencePiece's ``▁`` into a space);
* full duplex: user speech during a reply makes the fake yield after ``yield_after``
  seconds of overlap (short backchannels do not stop it);
* the Rust server's step limit (``max_steps`` -> normal close), abrupt drops
  (:meth:`FakeMoshiServer.drop`), ``0x05`` errors and HTTP rejection;
* PersonaPlex: the ``text_prompt``/``voice_prompt`` query parameters are recorded (and
  required with ``require_prompts=True``, like the PersonaPlex server).

Example::

    async with FakeMoshiServer(greeting="Hi there!", replies=["Sure."]) as server:
        session = AgentSession(MoshiEngine(url=server.url))
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from http import HTTPStatus
from typing import TYPE_CHECKING, Literal
from urllib.parse import parse_qsl, urlsplit

import numpy as np

from ..audio.frame import AudioFrame
from ..providers.mock import synth_speech
from ..providers.moshi import FRAME_SAMPLES, SAMPLE_RATE, MsgKind, OpusDecoder, OpusEncoder
from ..utils.aio import cancel_and_wait

if TYPE_CHECKING:
    from websockets.asyncio.server import Server, ServerConnection
    from websockets.http11 import Request, Response

__all__ = ["FakeMoshiConnection", "FakeMoshiServer", "FakeUtterance"]

FRAME = FRAME_SAMPLES / SAMPLE_RATE
"""One model step (80 ms)."""


@dataclass
class FakeUtterance:
    """Something the fake model said (or started to say)."""

    text: str
    kind: Literal["greeting", "reply"]
    start_step: int
    frames: int
    spoken: int = 0
    """Frames actually produced (``< frames`` when the fake yielded to the user)."""

    @property
    def completed(self) -> bool:
        return self.spoken >= self.frames


@dataclass
class FakeMoshiConnection:
    """One client connection to :class:`FakeMoshiServer`."""

    ws: ServerConnection
    query: dict[str, str]
    utterances: list[FakeUtterance] = field(default_factory=list)
    steps: int = 0
    """Model steps run (80 ms frames of client audio received)."""
    audio_messages: int = 0
    closed: asyncio.Event = field(default_factory=asyncio.Event)


class FakeMoshiServer:
    """In-process fake of a Moshi server (see the module docstring).

    Args:
        greeting: said once, ``greeting_after`` seconds into each connection (``None``: no
            greeting).
        replies: said in turn after each user utterance (then ``default_reply``).
        flavor: ``"python"`` (``moshi.server``) or ``"rust"`` (``moshi-backend``) handshake.
        words_per_second: speaking rate of the synthetic speech.
        respond_after: user silence (s) after which the fake replies.
        yield_after: overlapping user speech (s) after which the fake stops talking.
        user_threshold_db: level of decoded client audio that counts as user speech.
        noise_db: level of the noise the fake outputs while quiet (``None``: digital silence).
        max_steps: close the connection normally after this many steps (Rust server limit).
        require_prompts: reject connections without ``text_prompt``/``voice_prompt``
            (PersonaPlex).
        handshake: send the handshake (``False`` simulates a server busy with another client).
        reject_status: reject the WebSocket handshake with this HTTP status.
    """

    def __init__(
        self,
        *,
        greeting: str | None = None,
        replies: Sequence[str] = (),
        default_reply: str = "Okay.",
        flavor: Literal["python", "rust"] = "python",
        words_per_second: float = 4.0,
        greeting_after: float = 0.16,
        respond_after: float = 0.4,
        yield_after: float = 0.4,
        user_threshold_db: float = -40.0,
        noise_db: float | None = -70.0,
        max_steps: int | None = None,
        require_prompts: bool = False,
        handshake: bool = True,
        reject_status: int | None = None,
    ) -> None:
        self.greeting = greeting
        self.replies = list(replies)
        self.default_reply = default_reply
        self.flavor = flavor
        self.words_per_second = words_per_second
        self.greeting_after = greeting_after
        self.respond_after = respond_after
        self.yield_after = yield_after
        self.user_threshold_db = user_threshold_db
        self.noise_db = noise_db
        self.max_steps = max_steps
        self.require_prompts = require_prompts
        self.handshake = handshake
        self.reject_status = reject_status
        self.connections: list[FakeMoshiConnection] = []
        self.errors: list[str] = []
        """Protocol violations detected in client messages (tests assert it stays empty)."""
        self.port = 0
        self._reply_index = 0
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

    async def __aenter__(self) -> FakeMoshiServer:
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    @property
    def url(self) -> str:
        """Origin for ``MoshiEngine(url=...)`` (the engine appends ``/api/chat``)."""
        return f"ws://127.0.0.1:{self.port}"

    @property
    def connection(self) -> FakeMoshiConnection:
        """The most recent connection."""
        if not self.connections:
            raise RuntimeError("no client connected yet")
        return self.connections[-1]

    @property
    def utterances(self) -> list[FakeUtterance]:
        return [u for c in self.connections for u in c.utterances]

    # ------------------------------------------------------------------ control
    async def drop(self, code: int = 1011, reason: str = "internal error") -> None:
        """Close the most recent connection abruptly (server side)."""
        await self.connection.ws.close(code, reason)

    async def send_error(self, text: str) -> None:
        """Send a ``0x05`` error message on the most recent connection."""
        await self.connection.ws.send(bytes([MsgKind.ERROR]) + text.encode())

    def _next_reply(self) -> str:
        if self._reply_index < len(self.replies):
            self._reply_index += 1
            return self.replies[self._reply_index - 1]
        return self.default_reply

    # ----------------------------------------------------------------- protocol
    def _process_request(self, connection: ServerConnection, request: Request) -> Response | None:
        if self.reject_status is not None:
            status = HTTPStatus(self.reject_status)
            return connection.respond(status, f"{status.phrase} (fake moshi)\n")
        parts = urlsplit(request.path)
        if parts.path != "/api/chat":
            return connection.respond(HTTPStatus.NOT_FOUND, "unknown endpoint\n")
        query = dict(parse_qsl(parts.query))
        if self.require_prompts and not ("text_prompt" in query and "voice_prompt" in query):
            return connection.respond(HTTPStatus.INTERNAL_SERVER_ERROR, "KeyError\n")
        return None

    async def _handler(self, ws: ServerConnection) -> None:
        from websockets.exceptions import ConnectionClosed

        path = ws.request.path if ws.request is not None else ""
        conn = FakeMoshiConnection(ws, dict(parse_qsl(urlsplit(path).query)))
        self.connections.append(conn)
        try:
            if self.handshake:
                if self.flavor == "rust":
                    meta = {
                        "text_temperature": 0.7,
                        "audio_temperature": 0.8,
                        "instance_name": "fake",
                    }
                    await ws.send(bytes([MsgKind.METADATA]) + json.dumps(meta).encode())
                    await ws.send(bytes([MsgKind.HANDSHAKE]) + bytes(8))
                else:
                    await ws.send(bytes([MsgKind.HANDSHAKE]))
            await _Model(self, conn).run()
        except ConnectionClosed:
            pass
        finally:
            conn.closed.set()


class _Model:
    """The step loop of one connection: decode input, decide, speak."""

    def __init__(self, server: FakeMoshiServer, conn: FakeMoshiConnection) -> None:
        self.s = server
        self.conn = conn
        self.decoder = OpusDecoder()
        self.encoder = OpusEncoder()
        self.pending = np.zeros(0, dtype=np.int16)
        self.first_audio = True
        self.utt: FakeUtterance | None = None
        self.audio: AudioFrame | None = None
        self.words: dict[int, list[str]] = {}
        self.heard = 0.0
        self.silence = 0.0
        self.overlap = 0.0
        self.rng = np.random.default_rng(0)

    async def run(self) -> None:
        ws = self.conn.ws
        async for raw in ws:
            if not isinstance(raw, bytes) or not raw:
                self.s.errors.append(f"non-binary or empty message: {raw!r:.40}")
                continue
            kind, payload = raw[0], raw[1:]
            if kind != MsgKind.AUDIO:
                if kind != MsgKind.PING:
                    self.s.errors.append(f"unexpected message kind {kind}")
                continue
            if self.first_audio and not payload.startswith(b"OggS"):
                self.s.errors.append("the first audio message must start an Ogg stream")
            self.first_audio = False
            self.conn.audio_messages += 1
            pcm = self.decoder.decode(payload).to_numpy()
            self.pending = np.concatenate((self.pending, pcm))
            while len(self.pending) >= FRAME_SAMPLES:
                frame = AudioFrame(self.pending[:FRAME_SAMPLES].tobytes(), SAMPLE_RATE)
                self.pending = self.pending[FRAME_SAMPLES:]
                await self.step(frame)
                if self.s.max_steps is not None and self.conn.steps >= self.s.max_steps:
                    await ws.close(1000, "max steps reached")
                    return

    def _say(self, text: str, kind: Literal["greeting", "reply"]) -> None:
        words = text.split()
        frames = max(1, round(len(words) / self.s.words_per_second / FRAME))
        self.utt = FakeUtterance(text, kind, self.conn.steps, frames)
        self.conn.utterances.append(self.utt)
        self.audio = synth_speech(frames * FRAME, SAMPLE_RATE, frequency=180.0)
        self.words = {}
        for i, word in enumerate(words):
            self.words.setdefault(i * frames // len(words), []).append(" " + word)

    async def step(self, user: AudioFrame) -> None:
        s, conn = self.s, self.conn
        step = conn.steps
        conn.steps += 1
        loud = user.dbfs() >= s.user_threshold_db
        if self.utt is not None:
            self.overlap = self.overlap + FRAME if loud else 0.0
            if self.overlap >= s.yield_after - 1e-9:
                self.utt = None  # full duplex: yield to the user
                self.heard, self.silence = self.overlap, 0.0
        elif loud:
            self.heard += FRAME
            self.silence = 0.0
        elif self.heard > 0:
            self.silence += FRAME
            if self.silence >= s.respond_after - 1e-9:
                self.heard = self.silence = 0.0
                self._say(s._next_reply(), "reply")
        if (
            s.greeting is not None
            and step == round(s.greeting_after / FRAME)
            and self.utt is None
            and not conn.utterances
        ):
            self._say(s.greeting, "greeting")
        tokens: list[str] = []
        utt = self.utt
        if utt is not None and self.audio is not None:
            i = utt.spoken
            nbytes = FRAME_SAMPLES * 2
            out = AudioFrame(self.audio.data[i * nbytes : (i + 1) * nbytes], SAMPLE_RATE)
            tokens = self.words.get(i, [])
            utt.spoken += 1
            if utt.completed:
                self.utt = None
                self.overlap = 0.0
        elif s.noise_db is None:
            out = AudioFrame.silence(FRAME, SAMPLE_RATE)
        else:
            level = 10 ** (s.noise_db / 20)
            noise = self.rng.normal(0.0, level, FRAME_SAMPLES).astype(np.float32)
            out = AudioFrame.from_numpy(noise, SAMPLE_RATE)
        pages = self.encoder.encode(out)
        if pages:  # moshi.server sends the audio of a step, then its text token
            await conn.ws.send(bytes([MsgKind.AUDIO]) + pages)
        for token in tokens:
            await conn.ws.send(bytes([MsgKind.TEXT]) + token.encode())
