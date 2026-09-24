"""A fake Gemini API for ``httpx.MockTransport``: records requests, replays real shapes.

Streaming responses mirror ``POST /v1beta/models/{model}:streamGenerateContent?alt=sse``:
``data: <GenerateContentResponse JSON>`` events separated by ``\\r\\n\\r\\n``, each carrying
``candidates`` (with ``content.parts``, ``finishReason`` on the last one), cumulative
``usageMetadata``, ``modelVersion`` and ``responseId``, delivered in small byte chunks so
events are split across network reads. Errors use the ``google.rpc.Status`` JSON body.
"""

from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx

API_KEY = "AIzaSyTest-key-0123456789"


def sse(events: list[dict[str, Any]]) -> bytes:
    return b"".join(b"data: " + json.dumps(e).encode() + b"\r\n\r\n" for e in events)


def chunk(
    *parts: dict[str, Any],
    finish: str | None = None,
    usage: dict[str, int] | None = None,
    model: str = "gemini-3.8-flash",
    **extra: Any,
) -> dict[str, Any]:
    """One streamed ``GenerateContentResponse``."""
    candidate: dict[str, Any] = {"content": {"role": "model", "parts": list(parts)}, "index": 0}
    if finish:
        candidate["finishReason"] = finish
    event: dict[str, Any] = {
        "candidates": [candidate],
        "modelVersion": model,
        "responseId": "mZ7TaLr0KqXw1MkP-test",
    }
    if usage is not None:
        event["usageMetadata"] = usage
    event.update(extra)
    return event


def text(value: str, **extra: Any) -> dict[str, Any]:
    return {"text": value, **extra}


def call(
    name: str, args: dict[str, Any], *, signature: str | None = None, **fc: Any
) -> dict[str, Any]:
    part: dict[str, Any] = {"functionCall": {"name": name, "args": args, **fc}}
    if signature is not None:
        part["thoughtSignature"] = signature
    return part


def audio(pcm: bytes, mime: str = "audio/l16;codec=pcm;rate=24000") -> dict[str, Any]:
    return {"inlineData": {"mimeType": mime, "data": base64.b64encode(pcm).decode()}}


def usage(prompt: int, candidates: int = 0, **extra: int) -> dict[str, int]:
    out = {"promptTokenCount": prompt, "totalTokenCount": prompt + candidates}
    if candidates:
        out["candidatesTokenCount"] = candidates
    out.update(extra)
    return out


def stream_response(
    events: list[dict[str, Any]],
    *,
    chunk_size: int = 23,
    tail: Callable[[], Any] | None = None,
    closed: asyncio.Event | None = None,
) -> httpx.Response:
    body = sse(events)

    async def chunks() -> AsyncIterator[bytes]:
        try:
            for i in range(0, len(body), chunk_size):
                await asyncio.sleep(0)
                yield body[i : i + chunk_size]
            if tail is not None:
                await tail()
        finally:
            if closed is not None:
                closed.set()

    return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=chunks())


def error_response(
    status: int, rpc_status: str, message: str, *, reason: str | None = None
) -> httpx.Response:
    error: dict[str, Any] = {"code": status, "message": message, "status": rpc_status}
    if reason:
        error["details"] = [
            {
                "@type": "type.googleapis.com/google.rpc.ErrorInfo",
                "reason": reason,
                "domain": "googleapis.com",
                "metadata": {"service": "generativelanguage.googleapis.com"},
            }
        ]
    return httpx.Response(status, json={"error": error})


def model_info(model: str) -> httpx.Response:
    """``GET /v1beta/models/{model}``."""
    return httpx.Response(
        200,
        json={
            "name": f"models/{model}",
            "version": "001",
            "displayName": model,
            "inputTokenLimit": 1048576,
            "outputTokenLimit": 65536,
            "supportedGenerationMethods": ["generateContent", "countTokens"],
        },
    )


class FakeGeminiAPI:
    """MockTransport handler: records requests and replays scripted responses."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.responses: list[Any] = []

    def reply(self, *responses: Any) -> None:
        self.responses.extend(responses)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        response = self.responses.pop(0)
        return response(request) if callable(response) else response

    def body(self, index: int = -1) -> dict[str, Any]:
        return json.loads(self.requests[index].content)

    def http_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self))
