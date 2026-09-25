"""Serve a voice agent over WebSocket (protocol van-ws/1) and talk to it from a browser or Python.

Server + browser demo (the mock engine by default: offline, no API keys)::

    python examples/websocket_agent.py serve        # then open http://127.0.0.1:8765/
    python examples/websocket_agent.py serve --host 0.0.0.0 --allowed-origin https://agent.lan:8765
    python examples/websocket_agent.py serve --engine openai/gpt-realtime
    python examples/websocket_agent.py serve --stt deepgram/nova-3 --llm openai/gpt-4.1-mini \\
        --tts cartesia/sonic-2 --vad silero

Python client (a backend talking to the agent): streams a WAV file (or synthetic speech) in
real time, plays the reply on a simulated speaker that reports its position, and records it::

    python examples/websocket_agent.py client --input question.wav --output reply.wav
    python examples/websocket_agent.py client --text "What can you do?"

The protocol is specified in docs/transports/websocket.md.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from http import HTTPStatus
from pathlib import Path
from typing import Any

from websockets.asyncio.client import ClientConnection, connect
from websockets.asyncio.server import ServerConnection
from websockets.exceptions import ConnectionClosed
from websockets.http11 import Request, Response

from voice_agent_next import Agent, AgentSession, AudioFrame, ChatContext
from voice_agent_next.audio import read_wav, write_wav
from voice_agent_next.metrics import TurnMetrics
from voice_agent_next.providers.mock import MockEngine, synth_speech
from voice_agent_next.transports.websocket import (
    PROTOCOL,
    WebSocketServerTransport,
    serve_websocket,
)
from voice_agent_next.utils import cancel_and_wait, now

WEB_PAGE = Path(__file__).parent / "web" / "index.html"
MOCK_SPEECH = "[speech]"  # what the mock engine "transcribes" (it cannot understand words)
MOCK_TAIL = (
    "I am the offline mock engine, so I cannot understand words and my voice is a synthetic "
    "tone. Interrupt me while I am talking to try barge-in, or type a message."
)


# ------------------------------------------------------------------------------ server
def mock_reply(ctx: ChatContext) -> str:
    last = ctx.last_message("user")
    if last is None or last.text == MOCK_SPEECH:
        return f"I heard you speak. {MOCK_TAIL}"
    return f"You typed: {last.text}. {MOCK_TAIL}"


def make_session(args: argparse.Namespace, transport: WebSocketServerTransport) -> AgentSession:
    """One session per connection (the transport gives access to the client's hello)."""
    if args.llm:  # cascade
        session = AgentSession(stt=args.stt, llm=args.llm, tts=args.tts, vad=args.vad)
    elif args.engine == "mock":
        engine = MockEngine(responses=mock_reply, response_delay=0.3)
        engine.stt.default_text = MOCK_SPEECH
        session = AgentSession(engine)
    else:
        session = AgentSession(args.engine)

    def log(text: str) -> None:
        print(f"[{transport.session_id}] {text}", flush=True)

    def on_user(ev: Any) -> None:
        if ev.is_final:
            log(f"user : {ev.text}")

    def on_metrics(m: Any) -> None:
        if isinstance(m, TurnMetrics) and m.voice_to_voice is not None:
            log(f"voice-to-voice latency: {m.voice_to_voice * 1000:.0f} ms")

    session.on("user_transcript", on_user)
    session.on("agent_transcript", lambda ev: log(f"agent: {ev.delta.strip()}"))
    session.on("interrupted", lambda ev: log(f"(interrupted after {ev.played:.1f} s)"))
    session.on("metrics", on_metrics)
    session.on("close", lambda ev: log(f"session closed: {ev.reason}"))
    return session


def serve_demo_page(page: str) -> Any:
    """``process_request`` hook: plain HTTP GET / returns the browser demo."""

    def process_request(connection: ServerConnection, request: Request) -> Response | None:
        if request.headers.get("Upgrade", "").lower() == "websocket":
            return None  # a WebSocket client: continue the handshake
        if request.path.split("?", 1)[0] not in ("/", "/index.html"):
            return connection.respond(HTTPStatus.NOT_FOUND, "Not found\n")
        response = connection.respond(HTTPStatus.OK, page)
        del response.headers["Content-Type"]
        response.headers["Content-Type"] = "text/html; charset=utf-8"
        return response

    return process_request


async def run_server(args: argparse.Namespace) -> None:
    server = await serve_websocket(
        lambda transport: make_session(args, transport),
        lambda: Agent(args.instructions, greeting=args.greeting),
        host=args.host,
        port=args.port,
        max_sessions=args.max_sessions,
        # browser pages from other origins than this machine's are refused (HTTP 403): the
        # demo page opened from another host name needs its origin listed here
        allowed_origins=args.allowed_origin,
        process_request=serve_demo_page(WEB_PAGE.read_text(encoding="utf-8")),
    )
    web_host = "127.0.0.1" if args.host in ("0.0.0.0", "::", "") else args.host
    print(f"voice agent listening on {server.url}")
    print(f"browser demo: http://{web_host}:{server.port}/  (Ctrl-C to stop)")
    await server.serve_forever()


# ------------------------------------------------------------------------------ client
class SimulatedSpeaker:
    """Plays agent audio in real time (like a sound card) and reports the playback position."""

    def __init__(self, sample_rate: int) -> None:
        self.sample_rate = sample_rate
        self.queue = bytearray()
        self.received = 0  # bytes of agent audio received
        self.played: list[bytes] = []

    def push(self, pcm: bytes) -> None:
        self.queue += pcm
        self.received += len(pcm)

    def clear(self) -> None:
        self.queue.clear()  # barge-in: drop everything not played yet

    def position_ms(self) -> float:
        """van-ws/1 playback cursor: audio received minus audio still queued."""
        return (self.received - len(self.queue)) / 2 / self.sample_rate * 1000

    async def run(self, ws: ClientConnection) -> None:
        last, last_report, reported = now(), now(), -1.0
        while True:
            await asyncio.sleep(0.02)
            t = now()
            n = min(len(self.queue), round((t - last) * self.sample_rate) * 2)
            last = t
            if n:
                self.played.append(bytes(self.queue[:n]))
                del self.queue[:n]
            position = self.position_ms()
            if position != reported and (t - last_report >= 0.1 or not self.queue):
                await ws.send(json.dumps({"type": "playback", "position_ms": round(position)}))
                reported, last_report = position, t


async def receive(ws: ClientConnection, speaker: SimulatedSpeaker, state: dict[str, Any]) -> None:
    async for message in ws:
        if isinstance(message, bytes):
            speaker.push(message)
            continue
        msg = json.loads(message)
        kind = msg.get("type")
        if kind == "clear":
            speaker.clear()
            print("(clear: the agent was interrupted)")
        elif kind == "transcript" and msg["final"]:
            note = " (interrupted)" if msg.get("interrupted") else ""
            print(f"{msg['role']:>9}: {msg['text']}{note}")
        elif kind == "state":
            print(f"    state: agent={msg['agent']} user={msg['user']}")
            state.update(agent=msg["agent"], since=now())
        elif kind == "metrics" and msg["kind"] == "turn":
            v2v = msg["data"]["voice_to_voice"]
            if v2v is not None:
                print(f"  latency: voice-to-voice {v2v * 1000:.0f} ms")
        elif kind == "error":
            print(f"    error: {msg['message']}")


async def wait_until_idle(state: dict[str, Any], quiet: float = 0.5, timeout: float = 15) -> None:
    """Wait until the agent has been listening for ``quiet`` seconds (at most ``timeout``)."""
    deadline = now() + timeout
    while now() < deadline:
        if state["agent"] == "listening" and now() - state["since"] > quiet:
            return
        await asyncio.sleep(0.05)


async def run_client(args: argparse.Namespace) -> None:
    speech = read_wav(args.input).to_mono() if args.input else synth_speech(1.2, 16_000)
    rate = speech.sample_rate
    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else None
    async with connect(args.url, compression=None, additional_headers=headers) as ws:
        hello: dict[str, Any] = {"type": "hello", "protocol": PROTOCOL, "codec": "pcm_s16le"}
        hello |= {"sample_rate": rate, "channels": 1, "output_sample_rate": args.output_rate}
        await ws.send(json.dumps(hello))
        ready = json.loads(await ws.recv())
        if ready.get("type") != "ready":
            raise SystemExit(f"handshake failed: {ready}")
        print(
            f"connected: session {ready['session_id']}, agent audio at {ready['output_sample_rate']} Hz"
        )
        speaker = SimulatedSpeaker(ready["output_sample_rate"])
        state: dict[str, Any] = {"agent": "initializing", "since": now()}
        tasks = [
            asyncio.create_task(receive(ws, speaker, state)),
            asyncio.create_task(speaker.run(ws)),
        ]
        try:
            await wait_until_idle(state)  # let the agent finish its greeting
            # user audio in real time (20 ms frames), then silence while the agent answers
            audio = AudioFrame.concat([speech, AudioFrame.silence(args.listen, rate)])
            step, start = round(0.02 * rate) * 2, now()
            for i in range(0, len(audio.data), step):
                await ws.send(audio.data[i : i + step])
                delay = start + (i + step) / 2 / rate - now()
                if delay > 0:
                    await asyncio.sleep(delay)
            if args.text:
                await ws.send(json.dumps({"type": "text", "text": args.text}))
                await asyncio.sleep(args.listen)
        except ConnectionClosed:
            pass
        finally:
            await cancel_and_wait(*tasks)
    if args.output and speaker.played:
        write_wav(args.output, AudioFrame(b"".join(speaker.played), speaker.sample_rate))
        print(f"agent audio written to {args.output}")


# -------------------------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    srv = sub.add_parser("serve", help="serve the agent (and the browser demo)")
    srv.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to accept remote clients")
    srv.add_argument("--port", type=int, default=8765)
    srv.add_argument("--engine", default="mock", help="speech-to-speech engine spec")
    srv.add_argument("--stt", help="cascade STT (use with --llm/--tts)")
    srv.add_argument("--llm", help="cascade LLM: switches to a cascade")
    srv.add_argument("--tts", help="cascade TTS")
    srv.add_argument("--vad", help="cascade VAD, e.g. silero or energy")
    srv.add_argument("--instructions", default="You are a friendly, concise voice assistant.")
    srv.add_argument("--greeting", default="Hi! Talk to me, and feel free to interrupt me.")
    srv.add_argument("--max-sessions", type=int, default=8)
    srv.add_argument(
        "--allowed-origin",
        action="append",
        default=[],
        help="browser origin allowed besides http://localhost:* (repeatable), e.g. the "
        "https:// address of this demo on your LAN",
    )

    cli = sub.add_parser("client", help="talk to a running agent from Python")
    cli.add_argument("--url", default="ws://127.0.0.1:8765/")
    cli.add_argument(
        "--api-key",
        default=os.environ.get("VAN_SERVER_API_KEY") or None,
        help="the server's API key, if it requires one (default: $VAN_SERVER_API_KEY)",
    )
    cli.add_argument("--input", type=Path, help="WAV file with the user's speech")
    cli.add_argument("--output", type=Path, help="write the agent's audio to this WAV file")
    cli.add_argument("--text", help="also send this typed message")
    cli.add_argument("--listen", type=float, default=8.0, help="seconds to listen after speaking")
    cli.add_argument("--output-rate", type=int, default=24_000)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("websockets").setLevel(logging.WARNING)
    try:
        asyncio.run(run_server(args) if args.command == "serve" else run_client(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
