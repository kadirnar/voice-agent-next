"""Answer phone calls with Twilio Media Streams: one agent session per call.

The server does two jobs on one port:

* ``GET /twiml`` returns the TwiML that connects a call to the media stream, with the
  call's stream token (a ``<Parameter>`` derived from ``VAN_TELEPHONY_SECRET`` and the
  ``CallSid``); with ``--auth-token`` it checks Twilio's ``X-Twilio-Signature`` first;
* WebSocket ``/stream`` carries the call audio (μ-law 8 kHz), one ``AgentSession`` per call,
  with barge-in (Twilio ``clear``), playback tracking (``mark``) and DTMF. Streams without
  a valid token are refused.

Setup (docs/transports/telephony.md)::

    export VAN_TELEPHONY_SECRET=$(python -c "import secrets; print(secrets.token_hex(32))")
    python examples/06_telephony_twilio.py --engine openai/gpt-realtime-2.1 --port 8765
    ngrok http 8765          # Twilio only connects to public https/wss URLs
    # Twilio console > phone number > "A call comes in": Webhook, HTTP GET,
    #   https://<id>.ngrok.app/twiml
    python examples/06_telephony_twilio.py --public-host <id>.ngrok.app ...   # then call it

Offline::

    python examples/06_telephony_twilio.py --mock

``--mock`` serves the mock engine on a free local port and places one fake call: a small
Twilio simulator fetches the TwiML, streams μ-law audio in Twilio's message format,
acknowledges marks, collects the agent's audio and hangs up.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import json
import os
import re
import secrets
import sys
import urllib.request
from http import HTTPStatus
from typing import Any
from urllib.parse import parse_qs, urlsplit

from _common import log_conversation
from websockets.asyncio.client import connect
from websockets.asyncio.server import ServerConnection
from websockets.http11 import Request, Response

from voice_agent_next import Agent, AgentSession, AudioFrame
from voice_agent_next.audio.codecs import mulaw_decode, mulaw_encode
from voice_agent_next.providers.mock import MockEngine, synth_speech
from voice_agent_next.transports.telephony import (
    SECRET_ENV,
    TelephonyTransport,
    serve_telephony,
    twilio_stream_twiml,
    validate_twilio_signature,
)

CALL_SID = "CA" + "0" * 32  # the simulated call (Twilio SIDs: prefix + 32 hex digits)


# --------------------------------------------------------------------------- server
def webhook(public_host: str, secret: str, auth_token: str | None) -> Any:
    """``process_request`` hook: plain HTTP ``GET /twiml`` answers with TwiML."""

    def process_request(connection: ServerConnection, request: Request) -> Response | None:
        if request.headers.get("Upgrade", "").lower() == "websocket":
            return None  # the media stream: continue the WebSocket handshake
        if request.path.split("?", 1)[0] != "/twiml":
            return connection.respond(HTTPStatus.NOT_FOUND, "Not found\n")
        signature = request.headers.get("X-Twilio-Signature")
        url = f"https://{public_host}{request.path}"  # the URL configured at Twilio
        if auth_token and not validate_twilio_signature(auth_token, url, None, signature):
            return connection.respond(HTTPStatus.FORBIDDEN, "Forbidden\n")
        call_sid = parse_qs(urlsplit(request.path).query).get("CallSid", [""])[0]
        if not call_sid:
            return connection.respond(HTTPStatus.BAD_REQUEST, "CallSid missing\n")
        twiml = twilio_stream_twiml(
            f"wss://{public_host}/stream", {"source": "example"}, secret=secret, call_id=call_sid
        )
        response = connection.respond(HTTPStatus.OK, twiml)
        del response.headers["Content-Type"]
        response.headers["Content-Type"] = "text/xml"
        return response

    return process_request


def make_session_factory(args: argparse.Namespace) -> Any:
    def session_factory(transport: TelephonyTransport) -> AgentSession:
        call = transport.call  # call id, numbers (provider-dependent), TwiML <Parameter>s
        call_id = call.call_id if call else "?"
        print(f"[call {call_id}] started, parameters: {call.custom_parameters if call else {}}")
        engine: Any = args.engine
        if args.mock:
            engine = MockEngine(transcripts=["Hi, is the shop open today?"],
                                responses=["Yes, until 6 pm."], chars_per_second=40)  # fmt: skip
        session = AgentSession(engine)
        log_conversation(session, prefix=f"[call {call_id}] ")
        transport.on("dtmf", lambda digit: print(f"(caller pressed {digit})"))
        return session

    return session_factory


def make_agent() -> Agent:
    return Agent(
        "You answer the phone for a bike shop. Keep answers to one sentence.",
        greeting="Thanks for calling the bike shop!",
    )


# ----------------------------------------------------------------- mock phone call
async def fake_twilio_call(http_url: str, ws_url: str) -> float:
    """Play Twilio's part of one call; return the seconds of agent audio received."""
    no_proxy = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # localhost
    url = f"{http_url}?CallSid={CALL_SID}"  # Twilio's GET webhook carries the call
    twiml = await asyncio.to_thread(lambda: no_proxy.open(url, timeout=5).read().decode())
    print(f"TwiML: {twiml[:120]}...")
    parameters = dict(re.findall(r'<Parameter name="(\w+)" value="(\w+)"/>', twiml))
    stream_sid, received = "MZ" + "0" * 32, bytearray()
    async with connect(ws_url) as ws:
        start = {"accountSid": "AC" + "0" * 32, "streamSid": stream_sid, "callSid": CALL_SID,
                 "tracks": ["inbound"], "customParameters": parameters,
                 "mediaFormat": {"encoding": "audio/x-mulaw", "sampleRate": 8000, "channels": 1}}  # fmt: skip
        await ws.send(json.dumps({"event": "connected", "protocol": "Call", "version": "1.0.0"}))
        await ws.send(json.dumps({"event": "start", "start": start, "streamSid": stream_sid}))

        async def listen() -> None:
            async for message in ws:
                msg = json.loads(message)
                if msg["event"] == "media":  # agent audio, μ-law base64
                    received.extend(mulaw_decode(base64.b64decode(msg["media"]["payload"])))
                elif msg["event"] == "mark":  # "tell me when this was played": say it was
                    await ws.send(json.dumps({"event": "mark", "streamSid": stream_sid,
                                              "mark": msg["mark"]}))  # fmt: skip

        listener = asyncio.create_task(listen())
        await asyncio.sleep(1.0)  # let the greeting play
        caller = AudioFrame.concat([synth_speech(0.8, 8000), AudioFrame.silence(0.8, 8000)])
        for i in range(0, len(caller.data), 320):  # 20 ms media messages
            payload = base64.b64encode(mulaw_encode(caller.data[i : i + 320])).decode()
            await ws.send(json.dumps({"event": "media", "streamSid": stream_sid,
                                      "media": {"track": "inbound", "payload": payload}}))  # fmt: skip
            await asyncio.sleep(0.005)
        await asyncio.sleep(1.0)  # the answer
        await ws.send(json.dumps({"event": "stop", "streamSid": stream_sid,
                                  "stop": {"callSid": CALL_SID}}))  # fmt: skip
        with contextlib.suppress(Exception):
            await asyncio.wait_for(listener, 5)  # the server closes the stream
    return len(received) / 2 / 8000


async def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--mock", action="store_true", help="offline: mock engine + fake call")
    parser.add_argument("--engine", default="openai/gpt-realtime-2.1", help="engine spec")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--public-host", default="localhost:8765",
                        help="host name Twilio reaches (e.g. your ngrok domain)")  # fmt: skip
    parser.add_argument("--account-sid", help="with --auth-token: hang up calls via Twilio REST")
    parser.add_argument("--auth-token", help="also check X-Twilio-Signature on the webhook")
    args = parser.parse_args(argv)
    secret = os.environ.get(SECRET_ENV)
    if args.mock:
        args.host, args.port = "127.0.0.1", 0
        secret = secret or secrets.token_hex(32)
    if not secret:
        parser.error(f"set {SECRET_ENV} (a long random string) to authenticate media streams")

    credentials = {"account_sid": args.account_sid, "auth_token": args.auth_token}
    server = await serve_telephony(
        make_session_factory(args),
        make_agent,
        provider="twilio",
        host=args.host,
        port=args.port,
        serializer_options=credentials if args.account_sid else None,
        stream_secret=secret,
        process_request=webhook(args.public_host, secret, None if args.mock else args.auth_token),
    )
    if not args.mock:
        print(
            f"media stream on {server.url}stream, TwiML on http://{args.host}:{server.port}/twiml"
        )
        await server.serve_forever()  # Ctrl-C to stop
        return 0
    try:
        http_url = f"http://127.0.0.1:{server.port}/twiml"
        seconds = await fake_twilio_call(http_url, f"{server.url}stream")
        print(f"the caller heard {seconds:.1f} s of agent audio")
    finally:
        await server.aclose()
    return 0 if seconds > 0 else 1


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        sys.exit(asyncio.run(main()))
