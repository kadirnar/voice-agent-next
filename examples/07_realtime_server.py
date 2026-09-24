"""Serve any engine behind the OpenAI Realtime API, and talk to it with the OpenAI SDK.

``van serve`` puts a cascade of local models, Gemini Live or the mock behind the
OpenAI Realtime WebSocket protocol at ``/v1/realtime``. Existing Realtime clients (the
``openai`` SDK, the Agents SDK, LiveKit/Pipecat plugins) work unchanged
(docs/deploy/realtime-server.md).

Two terminals::

    van serve --stt sherpa-onnx --llm ollama/qwen3.5:4b --tts kokoro --vad silero --name local
    python examples/07_realtime_server.py --url ws://127.0.0.1:8000/v1 --model local

Or one process (the server runs in-process, like ``van serve``)::

    python examples/07_realtime_server.py --engine mock
    python examples/07_realtime_server.py --mock          # the offline smoke test

The client sends one text message and one spoken turn and prints the transcripts. It
uses the official SDK when installed (``pip install 'voice-agent-next[openai]'``),
otherwise the same JSON events over a plain WebSocket (``--client raw``).
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import json
import sys
from collections.abc import AsyncIterator
from typing import Any, Protocol

from websockets.asyncio.client import connect

from voice_agent_next import AudioFrame
from voice_agent_next.providers.mock import MockEngine, synth_speech
from voice_agent_next.server import RealtimeServer

API_KEY = "example-key"  # `van serve --api-key ...`; any string without authentication


class Connection(Protocol):
    async def send(self, event: dict[str, Any]) -> None: ...
    def events(self) -> AsyncIterator[dict[str, Any]]: ...


# ------------------------------------------------------------------------- clients
@contextlib.asynccontextmanager
async def openai_sdk_connection(url: str, model: str) -> AsyncIterator[Connection]:
    """The official SDK: only ``websocket_base_url`` differs from talking to OpenAI."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=API_KEY, websocket_base_url=url)
    async with client.realtime.connect(model=model) as conn:

        class SDKConnection:
            async def send(self, event: dict[str, Any]) -> None:
                await conn.send(event)  # type: ignore[arg-type]

            async def events(self) -> AsyncIterator[dict[str, Any]]:
                async for event in conn:
                    yield event.model_dump()

        yield SDKConnection()
    await client.close()


@contextlib.asynccontextmanager
async def raw_connection(url: str, model: str) -> AsyncIterator[Connection]:
    """The same protocol by hand: JSON events over a WebSocket."""
    headers = {"Authorization": f"Bearer {API_KEY}"}
    async with connect(f"{url}/realtime?model={model}", additional_headers=headers) as ws:

        class RawConnection:
            async def send(self, event: dict[str, Any]) -> None:
                await ws.send(json.dumps(event))

            async def events(self) -> AsyncIterator[dict[str, Any]]:
                async for message in ws:
                    yield json.loads(message)

        yield RawConnection()


async def until_response_done(conn: Connection) -> tuple[str, float]:
    """Collect one response: its transcript and seconds of audio (24 kHz PCM16)."""
    transcript, audio = "", 0
    async for event in conn.events():
        kind = event["type"]
        if kind == "conversation.item.input_audio_transcription.completed":
            print(f"  user (transcribed by the server): {event['transcript']}")
        elif kind == "response.output_audio_transcript.delta":
            transcript += event["delta"]
        elif kind == "response.output_audio.delta":
            audio += len(base64.b64decode(event["delta"]))
        elif kind == "error":
            raise RuntimeError(event["error"]["message"])
        elif kind == "response.done":
            break
    return transcript, audio / 2 / 24_000


async def talk(conn: Connection) -> None:
    await conn.send(
        {
            "type": "session.update",
            "session": {
                "type": "realtime",
                "instructions": "You are a helpful assistant.",
                "audio": {"input": {"transcription": {"model": "whisper-1"}}},
            },
        }
    )
    # 1) a text message, then ask for a response
    await conn.send(
        {
            "type": "conversation.item.create",
            "item": {"type": "message", "role": "user",
                     "content": [{"type": "input_text", "text": "Hi there!"}]},
        }
    )  # fmt: skip
    await conn.send({"type": "response.create"})
    text, seconds = await until_response_done(conn)
    print(f"agent: {text}  ({seconds:.1f} s of audio)")

    # 2) a spoken turn: stream 24 kHz PCM16; server VAD ends the turn and answers
    speech = AudioFrame.concat([synth_speech(0.8, 24_000), AudioFrame.silence(0.8, 24_000)])
    step = 24_000 // 10 * 2  # 100 ms per append
    for i in range(0, len(speech.data), step):
        chunk = base64.b64encode(speech.data[i : i + step]).decode()
        await conn.send({"type": "input_audio_buffer.append", "audio": chunk})
    text, seconds = await until_response_done(conn)
    print(f"agent: {text}  ({seconds:.1f} s of audio)")


# ------------------------------------------------------------------------------ main
async def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--mock", action="store_true", help="offline: in-process mock server")
    parser.add_argument("--url", help="a running server, e.g. ws://127.0.0.1:8000/v1")
    parser.add_argument("--engine", default="mock", help="in-process engine spec")
    parser.add_argument("--model", default="local", help="served model name (?model=)")
    parser.add_argument("--client", choices=["auto", "openai", "raw"], default="auto")
    args = parser.parse_args(argv)

    client = args.client
    if client == "auto":
        try:
            import openai  # noqa: F401

            client = "openai"
        except ImportError:
            client = "raw"
    open_connection = openai_sdk_connection if client == "openai" else raw_connection
    print(f"client: {client}")

    async with contextlib.AsyncExitStack() as stack:
        url = args.url
        if url is None:  # serve in-process, exactly what `van serve` does
            engine: Any = args.engine
            if args.mock or engine == "mock":
                engine = MockEngine(transcripts=["What time is it?"],
                                    responses=["Hello from a local engine!", "It is noon."])  # fmt: skip
            server = RealtimeServer(engine, model=args.model, port=0, api_keys=API_KEY)
            await stack.enter_async_context(server)
            url = server.url
            print(f"serving {url}/realtime (model {args.model!r})")
        conn = await stack.enter_async_context(open_connection(url, args.model))
        await asyncio.wait_for(talk(conn), timeout=30)
    return 0


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        sys.exit(asyncio.run(main()))
