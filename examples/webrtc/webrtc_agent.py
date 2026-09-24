"""Serve a voice agent over WebRTC (aiortc) and talk to it from a browser or from Python.

Needs the ``webrtc`` extra: ``pip install 'voice-agent-next[webrtc]'``.

Server + browser demo (the mock engine by default: offline, no API keys)::

    python examples/webrtc/webrtc_agent.py serve        # then open http://127.0.0.1:8080/
    python examples/webrtc/webrtc_agent.py serve --engine openai/gpt-realtime
    python examples/webrtc/webrtc_agent.py serve --stt deepgram/nova-3 --llm openai/gpt-4.1-mini \\
        --tts cartesia/sonic-2 --vad silero

Behind NAT or across the internet, add ICE servers (and serve HTTPS for remote browsers)::

    python examples/webrtc/webrtc_agent.py serve --host 0.0.0.0 --stun stun:stun.l.google.com:19302 \\
        --turn turn:turn.example.com:3478 --turn-user alice --turn-password secret \\
        --certfile cert.pem --keyfile key.pem

Python client (a peer that speaks a WAV file or synthetic speech and records the reply)::

    python examples/webrtc/webrtc_agent.py client --input question.wav --output reply.wav

The signalling protocol and data-channel messages are specified in docs/transports/webrtc.md.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import ssl
from pathlib import Path
from typing import Any

import httpx

from voice_agent_next import Agent, AgentSession, AudioFrame, ChatContext
from voice_agent_next.audio import read_wav, write_wav
from voice_agent_next.metrics import TurnMetrics
from voice_agent_next.providers.mock import MockEngine, synth_speech
from voice_agent_next.transports.webrtc import (
    DATA_CHANNEL_ID,
    DATA_CHANNEL_LABEL,
    AudioPlayout,
    WebRTCTransport,
    av_frame_to_audio,
    paced_audio_track,
    serve_webrtc,
)
from voice_agent_next.utils import cancel_and_wait, now, require

WEB_PAGE = Path(__file__).parent / "index.html"
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


def make_session(args: argparse.Namespace, transport: WebRTCTransport) -> AgentSession:
    """One session per peer (the transport gives access to the offer's metadata)."""
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


def ice_servers(args: argparse.Namespace) -> list[Any]:
    servers: list[Any] = list(args.stun)
    if args.turn:
        servers.append(
            {"urls": args.turn, "username": args.turn_user, "credential": args.turn_password}
        )
    return servers


async def run_server(args: argparse.Namespace) -> None:
    context = None
    if args.certfile:
        context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        context.load_cert_chain(args.certfile, args.keyfile)
    server = await serve_webrtc(
        lambda transport: make_session(args, transport),
        lambda: Agent(args.instructions, greeting=args.greeting),
        host=args.host,
        port=args.port,
        ice_servers=ice_servers(args),
        ice_transport_policy="relay" if args.relay else "all",
        max_sessions=args.max_sessions,
        index_html=WEB_PAGE.read_text(encoding="utf-8"),
        cors_origins=args.cors,
        ssl=context,
    )
    web_host = "127.0.0.1" if args.host in ("0.0.0.0", "::", "") else args.host
    scheme = "https" if context else "http"
    print(f"WebRTC signalling on {server.url}")
    print(f"browser demo: {scheme}://{web_host}:{server.port}/  (Ctrl-C to stop)")
    await server.serve_forever()


# ------------------------------------------------------------------------------ client
async def run_client(args: argparse.Namespace) -> None:
    """A Python peer: what a browser does, with a WAV file for a microphone."""
    aiortc = require("aiortc", extra="webrtc")
    from aiortc.mediastreams import MediaStreamError

    speech = read_wav(args.input).to_mono() if args.input else synth_speech(1.2, 16_000)
    base = args.url.rstrip("/") + "/"
    async with httpx.AsyncClient(timeout=15, trust_env=False) as http:
        config = (await http.get(base + "config")).json()
        servers = [aiortc.RTCIceServer(**s) for s in config["iceServers"]]
        pc = aiortc.RTCPeerConnection(aiortc.RTCConfiguration(iceServers=servers))
        mic = AudioPlayout()  # the "microphone": a real-time paced Opus track
        pc.addTrack(paced_audio_track(mic))
        channel = pc.createDataChannel(DATA_CHANNEL_LABEL, negotiated=True, id=DATA_CHANNEL_ID)
        received: list[AudioFrame] = []
        state: dict[str, Any] = {"agent": "initializing", "since": now()}
        tasks: list[asyncio.Task[None]] = []

        @channel.on("message")
        def on_message(data: str) -> None:
            msg = json.loads(data)
            kind = msg.get("type")
            if kind == "ready":
                print(f"connected: session {msg['session_id']}")
            elif kind == "transcript" and msg["final"]:
                note = " (interrupted)" if msg.get("interrupted") else ""
                print(f"{msg['role']:>9}: {msg['text']}{note}")
            elif kind == "state":
                state.update(agent=msg["agent"], since=now())
            elif kind == "metrics" and msg["kind"] == "turn":
                v2v = msg["data"]["voice_to_voice"]
                if v2v is not None:
                    print(f"  latency: voice-to-voice {v2v * 1000:.0f} ms")
            elif kind == "error":
                print(f"    error: {msg['message']}")

        async def record(track: Any) -> None:
            try:
                while True:
                    received.append(av_frame_to_audio(await track.recv()))
            except MediaStreamError:
                pass

        pc.on("track", lambda track: tasks.append(asyncio.create_task(record(track))))
        try:
            await pc.setLocalDescription(await pc.createOffer())
            offer = {"type": "offer", "sdp": pc.localDescription.sdp}
            response = await http.post(base + "offer", json=offer)
            answer = response.json()
            if response.status_code != 200:
                raise SystemExit(f"offer refused: {answer}")
            description = aiortc.RTCSessionDescription(sdp=answer["sdp"], type=answer["type"])
            await pc.setRemoteDescription(description)
            await wait_until_idle(state)  # let the agent finish its greeting
            mic.push(AudioFrame.concat([speech, AudioFrame.silence(0.6, speech.sample_rate)]))
            await asyncio.sleep(speech.duration + 0.6)
            await asyncio.sleep(0.5)
            await wait_until_idle(state, timeout=args.listen)
            channel.send(json.dumps({"type": "bye"}))
            await asyncio.sleep(0.1)
        finally:
            await pc.close()
            await cancel_and_wait(*tasks)
    if args.output and received:
        write_wav(args.output, AudioFrame.concat(received))
        print(f"agent audio written to {args.output}")


async def wait_until_idle(state: dict[str, Any], quiet: float = 0.5, timeout: float = 15) -> None:
    """Wait until the agent has been listening for ``quiet`` seconds (at most ``timeout``)."""
    deadline = now() + timeout
    while now() < deadline:
        if state["agent"] == "listening" and now() - state["since"] > quiet:
            return
        await asyncio.sleep(0.05)


# -------------------------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    srv = sub.add_parser("serve", help="serve the agent (and the browser demo)")
    srv.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to accept remote peers")
    srv.add_argument("--port", type=int, default=8080)
    srv.add_argument("--engine", default="mock", help="speech-to-speech engine spec")
    srv.add_argument("--stt", help="cascade STT (use with --llm/--tts)")
    srv.add_argument("--llm", help="cascade LLM: switches to a cascade")
    srv.add_argument("--tts", help="cascade TTS")
    srv.add_argument("--vad", help="cascade VAD, e.g. silero or energy")
    srv.add_argument("--instructions", default="You are a friendly, concise voice assistant.")
    srv.add_argument("--greeting", default="Hi! Talk to me, and feel free to interrupt me.")
    srv.add_argument("--max-sessions", type=int, default=None)
    srv.add_argument("--stun", action="append", default=[], help="STUN URL (repeatable)")
    srv.add_argument("--turn", help="TURN URL, e.g. turn:turn.example.com:3478")
    srv.add_argument("--turn-user")
    srv.add_argument("--turn-password")
    srv.add_argument("--relay", action="store_true", help="relay-only ICE (needs --turn)")
    srv.add_argument("--cors", action="append", default=[], help="allowed page origin")
    srv.add_argument("--certfile", help="TLS certificate (serve HTTPS)")
    srv.add_argument("--keyfile", help="TLS private key")

    cli = sub.add_parser("client", help="talk to a running agent from Python")
    cli.add_argument("--url", default="http://127.0.0.1:8080/")
    cli.add_argument("--input", type=Path, help="WAV file with the user's speech")
    cli.add_argument("--output", type=Path, help="write the agent's audio to this WAV file")
    cli.add_argument("--listen", type=float, default=15.0, help="max seconds to wait for a reply")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for name in ("aioice", "aiortc"):
        logging.getLogger(name).setLevel(logging.WARNING)
    try:
        asyncio.run(run_server(args) if args.command == "serve" else run_client(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
