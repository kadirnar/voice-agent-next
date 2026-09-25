"""Amazon Nova 2 Sonic speech-to-speech engine (Bedrock ``InvokeModelWithBidirectionalStream``).

``AgentSession("aws/nova-2-sonic")`` runs an agent on Amazon Nova 2 Sonic
(``amazon.nova-2-sonic-v1:0``). Protocol checked on 2026-09-25 against the Nova 2 user guide:
https://docs.aws.amazon.com/nova/latest/nova2-userguide/sonic-input-events.html,
https://docs.aws.amazon.com/nova/latest/nova2-userguide/sonic-output-events.html,
https://docs.aws.amazon.com/nova/latest/nova2-userguide/sonic-tool-configuration.html,
https://docs.aws.amazon.com/nova/latest/nova2-userguide/sonic-chat-history.html and
https://docs.aws.amazon.com/nova/latest/nova2-userguide/sonic-cross-modal.html.

**Wire protocol.** One HTTP/2 bidirectional event stream per connection carries JSON
events (``{"event": {<name>: {...}}}``). The client sends ``sessionStart`` (inference
configuration, ``turnDetectionConfiguration.endpointingSensitivity``), ``promptStart``
(output audio format and voice, tools), the system prompt and the carried-over history as
``contentStart``/``textInput``/``contentEnd`` blocks (``interactive: false``), then opens
*one* audio content block (``contentStart`` AUDIO, LPCM 16 kHz) and streams
``audioInput`` into it for the life of the connection. Text typed mid-session is a
cross-modal ``TEXT`` block with ``interactive: true``; tool results are ``TOOL`` blocks
(``toolResult``). Closing: ``contentEnd`` (audio), ``promptEnd``, ``sessionEnd``.

The model answers with ``completionStart`` and content blocks: the user's transcript
(``role: USER``, ``FINAL``), a *speculative* preview of the reply text (``ASSISTANT``,
``SPECULATIVE``), the reply audio (``audioOutput``, LPCM 24 kHz), the *final* transcript of
what was actually spoken (``ASSISTANT``, ``FINAL``), ``toolUse`` blocks, ``usageEvent``
(cumulative token counts) and ``completionEnd``. ``contentEnd.stopReason`` is
``PARTIAL_TURN``, ``END_TURN``, ``TOOL_USE`` or ``INTERRUPTED`` (barge-in; Nova Sonic v1
signalled it as a ``textOutput`` of ``{"interrupted": true}``, which is also understood).

Mapping onto :mod:`voice_agent_next.events`:

* the model detects turns itself (endpointing sensitivity ``HIGH``/``MEDIUM``/``LOW``). A
  local energy VAD on the sent audio reports ``InputSpeechStarted``/``InputSpeechStopped``
  while the agent is silent (for latency metrics and user state); speech over the agent is
  left to the model, which reports it as ``INTERRUPTED`` -> ``InputSpeechStarted`` +
  ``ResponseDone(status="cancelled")`` (barge-in);
* the user transcript -> ``InputTranscript`` (partial per block), then ``InputCommitted``
  + ``InputTranscript(is_final=True)`` when the reply starts (or at the block's
  ``END_TURN``);
* the first assistant block of a turn -> ``ResponseStarted``; audio -> ``ResponseAudio``;
  the speculative text (or, with ``transcript="final"``, the final text) ->
  ``ResponseText``; the audio ``END_TURN`` -> ``ResponseDone``;
* ``toolUse`` -> ``ResponseToolCall``. Nova 2 Sonic calls tools *asynchronously* (it keeps
  talking while a tool runs), so ``tool_mode`` is ``"non_blocking"``; every result goes
  back as a ``toolResult`` block and the model speaks about it by itself;
* ``usageEvent`` -> the token usage of each ``ResponseDone`` and ``EngineMetrics``.

There is no server-side cancel: :meth:`NovaSonicSessionConnection.cancel_response` ends the
response locally and drops the rest of that turn's audio. The system prompt, voice and
tools are fixed per connection: :meth:`NovaSonicConnection.update` moves the conversation
to a fresh connection that has them.

**Rotation.** A connection lasts at most 8 minutes. :class:`NovaSonicEngine` (what
``"aws/..."`` creates) wraps the single-connection engine in
:class:`~voice_agent_next.engines.rotation.RotatingEngine`: the next connection is opened
ahead of the limit, seeded with the conversation as the user heard it (chat history
blocks, at most ~200 KB) and takes over at a quiet moment (make-before-break, see
``docs/concepts/session-rotation.md``). A dropped stream reconnects the same way.

**Client.** Bedrock's bidirectional streaming is only in the new, *experimental* Smithy
Python SDK (``aws-sdk-bedrock-runtime``, Python >= 3.12; extra ``aws``) — boto3 has no
bidirectional streams. The SDK is imported lazily (this module imports without it) and
sits behind a tiny :class:`NovaStream` interface, so tests run on a fake stream
(:mod:`voice_agent_next.testing.nova_sonic`). Credentials come from the standard AWS chain
(environment, shared config/credentials files and profiles, container and instance
metadata) unless given explicitly.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import os
import re
import uuid
from collections import deque
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Final, Literal, Protocol, TypeAlias, runtime_checkable

from ...audio.frame import AudioFrame
from ...chat import ChatContext, ChatMessage, FunctionCall, FunctionCallOutput
from ...engine import EngineCapabilities, EngineConnection, EngineOptions, S2SEngine
from ...engines.rotation import RotatingConnection, RotatingEngine, RotationPolicy, _Link
from ...errors import (
    AuthenticationError,
    ConfigurationError,
    ProviderConnectionError,
    ProviderError,
    ProviderTimeoutError,
    RateLimitError,
    VoiceAgentError,
)
from ...events import (
    EngineErrorEvent,
    EngineUsage,
    InputCommitted,
    InputSpeechStarted,
    InputSpeechStopped,
    InputTranscript,
    ResponseAudio,
    ResponseDone,
    ResponseStarted,
    ResponseStatus,
    ResponseText,
    ResponseToolCall,
)
from ...metrics import EngineMetrics
from ...registry import register_provider
from ...tools import FunctionTool, ToolScheduling
from ...utils.aio import BackgroundTasks, cancel_and_wait
from ...utils.clock import now
from ...utils.deps import require
from ...utils.ids import new_id
from ...utils.log import logger
from ...vad import VADEventType, VADOptions, VADStream
from ..energy import EnergyVAD

__all__ = [
    "DEFAULT_MODEL",
    "MODELS",
    "VOICES",
    "BedrockStream",
    "NovaSonicConnection",
    "NovaSonicEngine",
    "NovaSonicSessionConnection",
    "NovaSonicSessionEngine",
    "NovaStream",
    "StreamFactory",
    "history_messages",
    "map_aws_error",
    "resolve_model",
]

PROVIDER: Final = "aws"
DEFAULT_MODEL: Final = "amazon.nova-2-sonic-v1:0"
MODELS: Final[dict[str, str]] = {
    "nova-2-sonic": DEFAULT_MODEL,
    "nova-sonic": "amazon.nova-sonic-v1:0",
}
"""Short names accepted in specs (``"aws/nova-2-sonic"``); full model ids pass through."""
DEFAULT_REGION: Final = "us-east-1"
DEFAULT_VOICE: Final = "matthew"
VOICES: Final = (
    "matthew", "tiffany", "amy", "olivia", "lupe", "carlos", "ambre", "florian",
    "lennart", "beatrice", "lorenzo", "tina", "carolina", "leo", "kiara", "arjun",
)  # fmt: skip
"""Voice ids of ``audioOutputConfiguration.voiceId`` listed in the Nova 2 docs (2026-09)."""
SESSION_LIMIT: Final = 480.0
"""Bedrock closes a bidirectional stream after 8 minutes."""
INPUT_SAMPLE_RATE: Final = 16_000
OUTPUT_SAMPLE_RATES: Final = (8_000, 16_000, 24_000)
SAY_INSTRUCTIONS: Final = 'Say exactly the following, verbatim, and nothing else: "{text}"'
RESPOND_TEXT: Final = "Please respond now."

EndpointingSensitivity: TypeAlias = Literal["HIGH", "MEDIUM", "LOW"]
TranscriptSource: TypeAlias = Literal["speculative", "final"]
ToolChoice: TypeAlias = Literal["auto", "any"] | str

_MAX_TEXT_INPUT: Final = 12_000
"""Characters per ``textInput`` (the limit is 50 KB; UTF-8 needs at most 4 bytes/char)."""
_MAX_HISTORY_BYTES: Final = 190_000
"""The chat history may not exceed 200 KB."""
_TICK: Final = 0.04
_KEEPALIVE_IDLE: Final = 0.1
_KEEPALIVE_CHUNK: Final = 0.1
_MAX_TRACKED: Final = 256


# ----------------------------------------------------------------------------- transport
@runtime_checkable
class NovaStream(Protocol):
    """One bidirectional event stream (the only thing the engine needs from the SDK).

    ``send`` takes one input event (``{"event": {...}}``); ``receive`` returns the next
    output event, ``None`` once the stream ended, or raises a
    :class:`~voice_agent_next.errors.ProviderError`. ``close`` ends the input side and
    releases the stream.
    """

    async def send(self, event: Mapping[str, Any]) -> None: ...

    async def receive(self) -> dict[str, Any] | None: ...

    async def close(self) -> None: ...


StreamFactory: TypeAlias = Callable[["NovaSonicSessionEngine"], Awaitable[NovaStream]]
"""Opens a :class:`NovaStream` for an engine (default: :meth:`BedrockStream.open`)."""


_AUTH_ERRORS: Final = frozenset(
    {
        "AccessDeniedException",
        "UnrecognizedClientException",
        "ExpiredTokenException",
        "InvalidSignatureException",
        "IdentityChainError",
        "SmithyIdentityError",
        "MissingCredentialsError",
    }
)
_THROTTLE_ERRORS: Final = frozenset({"ThrottlingException", "ServiceQuotaExceededException"})
_RETRYABLE_ERRORS: Final = frozenset(
    {
        "InternalServerException",
        "ServiceUnavailableException",
        "ModelStreamErrorException",
        "ModelNotReadyException",
        "ModelErrorException",
    }
)


def map_aws_error(exc: BaseException, provider: str = PROVIDER) -> VoiceAgentError:
    """Map an SDK/transport exception to the library's error types (by class name, so the
    SDK does not have to be importable)."""
    if isinstance(exc, VoiceAgentError):
        return exc
    names = {cls.__name__ for cls in type(exc).__mro__}
    name = type(exc).__name__
    detail = str(getattr(exc, "message", "") or exc or "").strip()
    msg = f"{provider}: {name}" + (f": {detail}" if detail and detail != name else "")
    if names & _AUTH_ERRORS:
        return AuthenticationError(msg, provider=provider)
    if names & _THROTTLE_ERRORS:
        return RateLimitError(msg, provider=provider)
    if "ModelTimeoutException" in names or isinstance(exc, TimeoutError):
        return ProviderTimeoutError(msg, provider=provider)
    if names & _RETRYABLE_ERRORS:
        return ProviderError(msg, provider=provider, retryable=True)
    if "ValidationException" in names or "ResourceNotFoundException" in names:
        return ProviderError(msg, provider=provider)
    if isinstance(exc, (ConnectionError, OSError)) or "ClientTimeoutError" in names:
        return ProviderConnectionError(msg, provider=provider)
    return ProviderError(
        msg, provider=provider, retryable=getattr(exc, "is_retry_safe", None) is True
    )


class BedrockStream:
    """:class:`NovaStream` over ``aws-sdk-bedrock-runtime`` (the experimental Smithy SDK)."""

    def __init__(self, stream: Any, provider: str = PROVIDER) -> None:
        self._stream = stream
        self._provider = provider
        self._output: Any = None
        self._closed = False

    @classmethod
    async def open(cls, engine: NovaSonicSessionEngine) -> BedrockStream:
        """Invoke ``InvokeModelWithBidirectionalStream`` for ``engine.model``."""
        client_mod = require(
            "aws_sdk_bedrock_runtime.client", extra="aws", package="aws-sdk-bedrock-runtime"
        )
        client = await engine.bedrock_client()
        try:
            stream = await client.invoke_model_with_bidirectional_stream(
                client_mod.InvokeModelWithBidirectionalStreamOperationInput(model_id=engine.model)
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise map_aws_error(exc, engine.provider) from exc
        return cls(stream, engine.provider)

    async def send(self, event: Mapping[str, Any]) -> None:
        models = require("aws_sdk_bedrock_runtime.models", extra="aws")
        chunk = models.InvokeModelWithBidirectionalStreamInputChunk(
            value=models.BidirectionalInputPayloadPart(bytes_=json.dumps(event).encode())
        )
        try:
            await self._stream.input_stream.send(chunk)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise map_aws_error(exc, self._provider) from exc

    async def receive(self) -> dict[str, Any] | None:
        try:
            if self._output is None:
                _, self._output = await self._stream.await_output()
            while True:
                result = await self._output.receive()
                if result is None:
                    return None
                value = getattr(result, "value", None)
                if isinstance(value, BaseException):  # a modeled error in the stream
                    raise value
                payload = getattr(value, "bytes_", None)
                if not payload:
                    logger.debug("%s: ignoring %s", self._provider, type(result).__name__)
                    continue
                event = json.loads(payload.decode("utf-8"))
                if isinstance(event, dict):
                    return event
        except asyncio.CancelledError:
            raise
        except VoiceAgentError:
            raise
        except Exception as exc:
            raise map_aws_error(exc, self._provider) from exc

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with contextlib.suppress(Exception):
            await self._stream.input_stream.close()
        if self._output is not None:
            with contextlib.suppress(Exception):
                await self._output.close()


# ------------------------------------------------------------------------------- helpers
def resolve_model(model: str | None) -> str:
    """``"nova-2-sonic"`` -> ``"amazon.nova-2-sonic-v1:0"`` (full ids pass through)."""
    if not model:
        return DEFAULT_MODEL
    return MODELS.get(model.strip().lower(), model.strip())


def _chunks(text: str, limit: int = _MAX_TEXT_INPUT) -> list[str]:
    return [text[i : i + limit] for i in range(0, len(text), limit)] or [""]


def history_messages(ctx: ChatContext | None) -> tuple[str, list[tuple[str, str]]]:
    """Split a carried-over history into extra system text and chat-history messages.

    System/developer messages (e.g. a carry-over summary) join the system prompt; user and
    assistant messages become ``USER``/``ASSISTANT`` blocks (consecutive messages of one
    role are merged, as the roles must alternate); tool results are told to the model as
    assistant text (calls are dropped). The newest messages that fit ~200 KB are kept.
    """
    system: list[str] = []
    messages: list[tuple[str, str]] = []

    def add(role: str, text: str) -> None:
        if messages and messages[-1][0] == role:
            messages[-1] = (role, f"{messages[-1][1]}\n{text}")
        else:
            messages.append((role, text))

    for item in ctx.items if ctx is not None else []:
        if isinstance(item, ChatMessage):
            text = item.text.strip()
            if not text:
                continue
            if item.role in ("system", "developer"):
                system.append(text)
            else:
                add("ASSISTANT" if item.role == "assistant" else "USER", text)
        elif isinstance(item, FunctionCallOutput) and item.output.strip():
            name = item.name or "a tool"
            add("ASSISTANT", f"(Result of {name}: {item.output.strip()})")
    budget = _MAX_HISTORY_BYTES
    kept: list[tuple[str, str]] = []
    for role, text in reversed(messages):
        size = len(text.encode("utf-8")) + 64
        if size > budget:
            break
        kept.append((role, text))
        budget -= size
    kept.reverse()
    return "\n\n".join(system), kept


def _tool_spec(tool: FunctionTool) -> dict[str, Any]:
    return {
        "toolSpec": {
            "name": tool.name,
            "description": tool.description or tool.name,
            "inputSchema": {"json": json.dumps(tool.parameters)},
        }
    }


def _tool_result_content(output: FunctionCallOutput) -> str:
    """``toolResult.content`` is a stringified JSON object."""
    text = output.output
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError):
        parsed = None
    if isinstance(parsed, dict) and not output.is_error:
        return text
    return json.dumps({"error" if output.is_error else "result": text}, ensure_ascii=False)


def _stage(ev: Mapping[str, Any]) -> str | None:
    fields = ev.get("additionalModelFields")
    if isinstance(fields, str):
        with contextlib.suppress(ValueError):
            fields = json.loads(fields)
    if isinstance(fields, Mapping):
        stage = fields.get("generationStage")
        return str(stage).upper() if stage else None
    return None


def _is_interrupt_marker(text: str) -> bool:
    """Nova Sonic v1 reports barge-in as a ``textOutput`` of ``{ "interrupted" : true }``."""
    stripped = text.strip()
    if not stripped.startswith("{") or "interrupted" not in stripped:
        return False
    try:
        data = json.loads(stripped)
    except ValueError:
        return False
    return isinstance(data, dict) and data.get("interrupted") is True


def _normalize(text: str) -> str:
    return re.sub(r"[\W_]+", " ", text.lower()).strip()


def _usage(total: Mapping[str, Any]) -> EngineUsage:
    def count(side: str, kind: str) -> int:
        part = total.get(side)
        value = part.get(kind) if isinstance(part, Mapping) else None
        return int(value) if isinstance(value, (int, float)) else 0

    return EngineUsage(
        input_text_tokens=count("input", "textTokens"),
        input_audio_tokens=count("input", "speechTokens"),
        output_text_tokens=count("output", "textTokens"),
        output_audio_tokens=count("output", "speechTokens"),
    )


def _usage_delta(a: EngineUsage, b: EngineUsage) -> EngineUsage:
    return EngineUsage(
        input_text_tokens=max(0, a.input_text_tokens - b.input_text_tokens),
        input_audio_tokens=max(0, a.input_audio_tokens - b.input_audio_tokens),
        output_text_tokens=max(0, a.output_text_tokens - b.output_text_tokens),
        output_audio_tokens=max(0, a.output_audio_tokens - b.output_audio_tokens),
    )


def _usage_sum(a: EngineUsage, b: EngineUsage) -> EngineUsage:
    return EngineUsage(
        input_text_tokens=a.input_text_tokens + b.input_text_tokens,
        input_audio_tokens=a.input_audio_tokens + b.input_audio_tokens,
        output_text_tokens=a.output_text_tokens + b.output_text_tokens,
        output_audio_tokens=a.output_audio_tokens + b.output_audio_tokens,
    )


# -------------------------------------------------------------------------------- engine
class NovaSonicSessionEngine(S2SEngine):
    """Nova Sonic on one bidirectional stream per connection (no rotation; see
    :class:`NovaSonicEngine`, which wraps this engine and is what ``"aws/..."`` creates).

    Args:
        model: ``nova-2-sonic`` (default, ``amazon.nova-2-sonic-v1:0``), ``nova-sonic``
            (v1) or a full Bedrock model id / inference profile ARN.
        region: AWS region (default: ``AWS_REGION``, ``AWS_DEFAULT_REGION``, the profile's
            region, else ``us-east-1``). Nova 2 Sonic: us-east-1, us-west-2, eu-north-1,
            ap-northeast-1.
        profile: shared-config profile for credentials/region (default: ``AWS_PROFILE``).
        aws_access_key_id / aws_secret_access_key / aws_session_token: explicit
            credentials (default: the standard AWS credential chain).
        endpoint_url: Bedrock runtime endpoint override (VPC endpoints, testing).
        voice: ``voiceId`` (``matthew`` by default; see :data:`VOICES`). Fixed per
            connection: ``Agent(voice=...)`` wins.
        output_sample_rate: agent audio rate, 24000 (default), 16000 or 8000 (telephony).
        endpointing_sensitivity: how quickly the end of the user's turn is detected:
            ``"HIGH"`` (fast), ``"MEDIUM"`` or ``"LOW"`` (patient); ``None`` keeps the
            service default (Nova 2 only).
        max_tokens / top_p / temperature: ``inferenceConfiguration``
            (``EngineOptions.temperature`` overrides ``temperature``).
        tool_choice: ``"auto"`` (default), ``"any"`` or a tool name (forced first call).
        transcript: which assistant text becomes ``ResponseText``: ``"speculative"`` (the
            preview, streamed with the audio; default) or ``"final"`` (what was actually
            spoken, sentence by sentence after the audio).
        final_grace: ``transcript="final"``: how long a response waits after its audio for
            the final transcript (seconds).
        tool_grace: a response that ends with a tool call is done after this much quiet.
        report_overlap: also report user speech while the agent speaks (letting the
            session's interruption policy stop the agent; default: the model decides).
        user_vad_threshold_db / user_min_silence: local energy VAD on the user audio.
        keepalive: stream silence while the transport delivers no audio (Nova expects a
            continuous audio stream and times out without input).
        connect_timeout: timeout for opening the stream (seconds).
        close_timeout: timeout for the closing events on ``aclose()``.
        session_limit: connection limit (seconds) reported as ``max_session_duration``.
        stream_factory: opens the event stream (default: :meth:`BedrockStream.open`; tests
            pass :class:`voice_agent_next.testing.nova_sonic.FakeNovaSonic`).
    """

    provider = PROVIDER

    def __init__(
        self,
        *,
        model: str | None = None,
        region: str | None = None,
        profile: str | None = None,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        aws_session_token: str | None = None,
        endpoint_url: str | None = None,
        voice: str | None = None,
        output_sample_rate: int = 24_000,
        endpointing_sensitivity: EndpointingSensitivity | None = None,
        max_tokens: int = 1024,
        top_p: float = 0.9,
        temperature: float = 0.7,
        tool_choice: ToolChoice | None = None,
        transcript: TranscriptSource = "speculative",
        final_grace: float = 1.0,
        tool_grace: float = 0.3,
        report_overlap: bool = False,
        user_vad_threshold_db: float = -40.0,
        user_min_silence: float = 0.3,
        keepalive: bool = True,
        connect_timeout: float = 10.0,
        close_timeout: float = 2.0,
        session_limit: float = SESSION_LIMIT,
        stream_factory: StreamFactory | None = None,
    ) -> None:
        if output_sample_rate not in OUTPUT_SAMPLE_RATES:
            raise ConfigurationError(f"output_sample_rate must be one of {OUTPUT_SAMPLE_RATES}")
        if endpointing_sensitivity is not None:
            endpointing_sensitivity = endpointing_sensitivity.upper()  # type: ignore[assignment]
            if endpointing_sensitivity not in ("HIGH", "MEDIUM", "LOW"):
                raise ConfigurationError("endpointing_sensitivity must be HIGH, MEDIUM or LOW")
        if transcript not in ("speculative", "final"):
            raise ConfigurationError("transcript must be 'speculative' or 'final'")
        if session_limit <= 0:
            raise ConfigurationError("session_limit must be > 0")
        if (aws_access_key_id is None) != (aws_secret_access_key is None):
            raise ConfigurationError(
                "pass both aws_access_key_id and aws_secret_access_key (or neither)"
            )
        super().__init__(
            model=resolve_model(model),
            capabilities=EngineCapabilities(
                native_audio=True,
                server_turn_detection=True,
                tool_calling=True,
                input_transcription=True,
                output_transcription=True,
                truncation=False,
                full_duplex=False,
                text_input=True,
                tool_mode="non_blocking",
                max_session_duration=session_limit,
            ),
            input_sample_rate=INPUT_SAMPLE_RATE,
            output_sample_rate=output_sample_rate,
        )
        self.region = (
            region or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or None
        )
        self.profile = profile
        self.endpoint_url = endpoint_url
        self._credentials = (aws_access_key_id, aws_secret_access_key, aws_session_token)
        self.voice = voice
        self.endpointing_sensitivity = endpointing_sensitivity
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.temperature = temperature
        self.tool_choice = tool_choice
        self.transcript: TranscriptSource = transcript
        self.final_grace = final_grace
        self.tool_grace = tool_grace
        self.report_overlap = report_overlap
        self.user_vad_threshold_db = user_vad_threshold_db
        self.user_min_silence = user_min_silence
        self.keepalive = keepalive
        self.connect_timeout = connect_timeout
        self.close_timeout = close_timeout
        self.stream_factory: StreamFactory = stream_factory or BedrockStream.open
        self._client: Any = None
        self._client_lock = asyncio.Lock()

    async def bedrock_client(self) -> Any:
        """The shared ``AsyncBedrockRuntimeClient`` (created on first use)."""
        async with self._client_lock:
            if self._client is None:
                self._client = await self._create_client()
            return self._client

    async def _create_client(self) -> Any:
        config_mod = require(
            "aws_sdk_bedrock_runtime.config", extra="aws", package="aws-sdk-bedrock-runtime"
        )
        client_mod = require("aws_sdk_bedrock_runtime.client", extra="aws")
        overrides: dict[str, Any] = {}
        if self.region:
            overrides["region"] = self.region
        if self.endpoint_url:
            overrides["endpoint_uri"] = self.endpoint_url
        key_id, secret, token = self._credentials
        if key_id and secret:
            overrides["aws_access_key_id"] = key_id
            overrides["aws_secret_access_key"] = secret
            if token:
                overrides["aws_session_token"] = token
        try:
            config = await config_mod.AsyncBedrockRuntimeConfig.resolve(
                profile=self.profile, **overrides
            )
            if not getattr(config, "region", None):
                config.region = DEFAULT_REGION
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise map_aws_error(exc, self.provider) from exc
        return client_mod.AsyncBedrockRuntimeClient(config=config)

    async def connect(self, options: EngineOptions) -> EngineConnection:
        conn = NovaSonicSessionConnection(self, options)
        try:
            await conn.start()
        except BaseException:
            await conn.aclose()
            raise
        return conn

    async def aclose(self) -> None:
        client, self._client = self._client, None
        if client is not None:
            with contextlib.suppress(Exception):
                await client.close()


@register_provider(
    "engine",
    "aws",
    description="Amazon Nova 2 Sonic speech-to-speech (Bedrock bidirectional stream)",
    default_model="nova-2-sonic",
    models=tuple(MODELS),
    env=("AWS_ACCESS_KEY_ID", "AWS_PROFILE"),
    extra="aws",
    requires=("aws_sdk_bedrock_runtime",),
)
class NovaSonicEngine(RotatingEngine):
    """Nova Sonic with transparent rotation before the 8-minute connection limit (see the
    module docs).

    Takes every argument of :class:`NovaSonicSessionEngine`, plus ``rotation`` (a
    :class:`~voice_agent_next.engines.rotation.RotationPolicy`; default: look for a quiet
    moment from minute 5, force the switch 10 s before the limit).
    """

    provider = PROVIDER

    def __init__(
        self, *, model: str | None = None, rotation: RotationPolicy | None = None, **kwargs: Any
    ) -> None:
        super().__init__(
            NovaSonicSessionEngine(model=model, **kwargs),
            policy=rotation or RotationPolicy(lead=180.0),
        )

    @property
    def session_engine(self) -> NovaSonicSessionEngine:
        """The wrapped single-connection engine (its configuration)."""
        inner = self.inner
        assert isinstance(inner, NovaSonicSessionEngine)
        return inner

    async def connect(self, options: EngineOptions) -> EngineConnection:
        conn = NovaSonicConnection(self, options)
        try:
            await conn.start()
        except BaseException:
            await conn.aclose()
            raise
        return conn


# ---------------------------------------------------------------- rotating connection
class NovaSonicConnection(RotatingConnection):
    """A Nova Sonic conversation over a sequence of connections (rotation, reconnects)."""

    def __init__(self, engine: NovaSonicEngine, options: EngineOptions) -> None:
        super().__init__(engine, options)
        self._closed_usage = EngineUsage()

    @property
    def session(self) -> NovaSonicSessionConnection | None:
        """The connection currently carrying the conversation."""
        inner = self.inner
        return inner if isinstance(inner, NovaSonicSessionConnection) else None

    @property
    def usage(self) -> EngineUsage:
        """Token usage of every connection so far (from ``usageEvent``)."""
        session = self.session
        return _usage_sum(
            self._closed_usage, session.usage if session is not None else EngineUsage()
        )

    async def update(
        self, *, instructions: str | None = None, tools: list[FunctionTool] | None = None
    ) -> None:
        """The system prompt and tools are fixed per Nova connection: the conversation
        moves to a fresh connection that has the new ones (at the next quiet moment)."""
        await super().update(instructions=instructions, tools=tools)
        if instructions is not None or tools is not None:
            self.rotate("configuration update")

    async def _open(self, seed: ChatContext, version: int) -> _Link:
        link = await super()._open(seed, version)
        conn = link.conn
        if isinstance(conn, NovaSonicSessionConnection):
            conn.on_finished = self._on_session_finished
        return link

    async def _deliver(self, link: _Link) -> float:
        replayed = await super()._deliver(link)
        conn = link.conn
        if isinstance(conn, NovaSonicSessionConnection):
            # the 8-minute clock started when the stream opened, not at the switch: a
            # connection prepared ahead has less time left than a fresh one
            link.opened_at = conn.opened_at
        return replayed

    def _on_session_finished(self, session: NovaSonicSessionConnection) -> None:
        self._closed_usage = _usage_sum(self._closed_usage, session.usage)


# ------------------------------------------------------------------------ one connection
@dataclass
class _Content:
    role: str
    type: str
    stage: str | None
    response_id: str | None = None
    ignored: bool = False
    text: list[str] = field(default_factory=list)


class _Reply:
    """One response: the assistant's blocks between two turn boundaries."""

    __slots__ = (
        "audio_ended",
        "closing_at",
        "first_audio_at",
        "item_id",
        "response_id",
        "started_at",
        "text",
        "trigger_at",
    )

    def __init__(self, trigger_at: float | None) -> None:
        self.response_id = new_id("resp_")
        self.item_id = new_id("item_")
        self.started_at = now()
        self.trigger_at = trigger_at
        self.first_audio_at: float | None = None
        self.audio_ended = False
        self.closing_at: float | None = None
        """The response ends at this time unless more assistant content arrives."""
        self.text = ""


class NovaSonicSessionConnection(EngineConnection):
    """One Nova Sonic bidirectional stream (see the module docs)."""

    engine: NovaSonicSessionEngine

    def __init__(self, engine: NovaSonicSessionEngine, options: EngineOptions) -> None:
        super().__init__(engine, options)
        self.engine = engine
        self._e = engine
        self.prompt_name = str(uuid.uuid4())
        self.audio_content = str(uuid.uuid4())
        self.session_id: str | None = None
        self.opened_at = now()
        """When the stream was opened (the connection limit counts from here)."""
        """``sessionId`` reported by the service (from the first output event)."""
        self.usage = EngineUsage()
        """Cumulative token usage of this connection (latest ``usageEvent`` totals)."""
        self.on_finished: Callable[[NovaSonicSessionConnection], None] | None = None
        self._stream: NovaStream | None = None
        self._send_lock = asyncio.Lock()
        self._tasks = BackgroundTasks("nova-sonic")
        self._reader: asyncio.Task[None] | None = None
        self._ticker: asyncio.Task[None] | None = None
        self._started = False
        self._closing = False
        self._failed = False
        self._audio_open = False
        self._odd_byte = b""
        self._warned: set[str] = set()
        # ---- clocks
        self._t0 = now()
        self._stream_pos = 0.0
        self._last_input_at: float | None = None
        self._requested_at: float | None = None
        # ---- output
        self._contents: dict[str, _Content] = {}
        self._reply: _Reply | None = None
        self._muted = False
        """The current model turn was cancelled locally: drop its audio and text."""
        self._usage_mark = EngineUsage()
        self._tool_calls: set[str] = set()
        self._echoes: deque[str] = deque(maxlen=8)
        # ---- user side
        opts = VADOptions(min_speech_duration=0.1, min_silence_duration=engine.user_min_silence)
        self._vad: VADStream = EnergyVAD(
            sample_rate=engine.input_sample_rate,
            threshold_db=engine.user_vad_threshold_db,
            options=opts,
        ).stream()
        self._user_speaking = False
        self._user_reported = False
        self._user_pending = False
        self._user_speech_start: float | None = None
        self._user_speech_end: float | None = None
        self._user_item: str | None = None
        self._user_text: list[str] = []

    # ------------------------------------------------------------------ lifecycle
    async def start(self) -> None:
        """Open the stream and send the setup events (session, prompt, system prompt,
        history, audio block)."""
        e = self._e
        try:
            self._stream = await asyncio.wait_for(e.stream_factory(e), e.connect_timeout)
        except TimeoutError:
            raise ProviderTimeoutError(
                f"{e.provider}: the stream did not open within {e.connect_timeout:.0f}s",
                provider=e.provider,
            ) from None
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise map_aws_error(exc, e.provider) from exc
        self.opened_at = now()
        self._reader = asyncio.create_task(self._read(self._stream), name="nova-sonic-read")
        await self._send_setup()
        self._started = True
        self._t0 = now()
        self._ticker = asyncio.create_task(self._tick(), name="nova-sonic-tick")
        if not self.options.turn_detection:
            self._warn_once(
                "turns", "Nova Sonic detects turns itself; turn_detection=False is ignored"
            )

    async def _send_setup(self) -> None:
        e, opts = self._e, self.options
        inference: dict[str, Any] = {"maxTokens": e.max_tokens, "topP": e.top_p}
        temperature = opts.temperature if opts.temperature is not None else e.temperature
        inference["temperature"] = temperature
        session: dict[str, Any] = {"inferenceConfiguration": inference}
        if e.endpointing_sensitivity:
            session["turnDetectionConfiguration"] = {
                "endpointingSensitivity": e.endpointing_sensitivity
            }
        extra = opts.extra.get("session_start")
        if isinstance(extra, Mapping):
            session.update(extra)
        await self._send_required(_event("sessionStart", **session))
        await self._send_required(_event("promptStart", **self._prompt_payload()))
        extra_system, history = history_messages(opts.chat_ctx)
        system = "\n\n".join(t for t in (opts.instructions.strip(), extra_system) if t)
        if system:
            await self._send_text_block(system, "SYSTEM", interactive=False)
        for role, text in history:
            await self._send_text_block(text, role, interactive=False)
        await self._send_required(
            _event(
                "contentStart",
                promptName=self.prompt_name,
                contentName=self.audio_content,
                type="AUDIO",
                interactive=True,
                role="USER",
                audioInputConfiguration={
                    "mediaType": "audio/lpcm",
                    "sampleRateHertz": e.input_sample_rate,
                    "sampleSizeBits": 16,
                    "channelCount": 1,
                    "audioType": "SPEECH",
                    "encoding": "base64",
                },
            )
        )
        self._audio_open = True

    def _prompt_payload(self) -> dict[str, Any]:
        e, opts = self._e, self.options
        prompt: dict[str, Any] = {
            "promptName": self.prompt_name,
            "textOutputConfiguration": {"mediaType": "text/plain"},
            "audioOutputConfiguration": {
                "mediaType": "audio/lpcm",
                "sampleRateHertz": e.output_sample_rate,
                "sampleSizeBits": 16,
                "channelCount": 1,
                "voiceId": opts.voice or e.voice or DEFAULT_VOICE,
                "encoding": "base64",
                "audioType": "SPEECH",
            },
        }
        if opts.tools:
            tool_config: dict[str, Any] = {"tools": [_tool_spec(t) for t in opts.tools]}
            choice = e.tool_choice
            if choice in ("auto", "any"):
                tool_config["toolChoice"] = {choice: {}}
            elif choice:
                tool_config["toolChoice"] = {"tool": {"name": choice}}
            prompt["toolUseOutputConfiguration"] = {"mediaType": "application/json"}
            prompt["toolConfiguration"] = tool_config
        extra = opts.extra.get("prompt_start")
        if isinstance(extra, Mapping):
            prompt.update(extra)
        return prompt

    async def aclose(self) -> None:
        if self.closed:
            return
        self._closing = True
        stream = self._stream
        if stream is not None and self._started and not self._failed:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self._send_closing(), self._e.close_timeout)
        current = asyncio.current_task()
        await cancel_and_wait(*[t for t in (self._reader, self._ticker) if t and t is not current])
        await self._tasks.cancel_all()
        self._stream = None
        if stream is not None:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(stream.close(), self._e.close_timeout)
        self._vad.close()
        self._end_reply("incomplete")
        if self.on_finished is not None:
            with contextlib.suppress(Exception):
                self.on_finished(self)
        await super().aclose()

    async def _send_closing(self) -> None:
        if self._audio_open:
            self._audio_open = False
            await self._send(
                _event("contentEnd", promptName=self.prompt_name, contentName=self.audio_content)
            )
        await self._send(_event("promptEnd", promptName=self.prompt_name))
        await self._send(_event("sessionEnd"))

    # --------------------------------------------------------------------- sending
    async def _send_required(self, event: dict[str, Any]) -> None:
        """Send a setup event; failures propagate (the connection cannot start)."""
        stream = self._stream
        if stream is None:
            raise ProviderConnectionError(f"{self._e.provider}: the stream is closed")
        async with self._send_lock:
            await stream.send(event)

    async def _send(self, event: dict[str, Any]) -> bool:
        """Send an event; ``False`` if the stream is gone (the failure is reported once)."""
        stream = self._stream
        if stream is None:
            return False
        try:
            async with self._send_lock:
                await stream.send(event)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._fail(map_aws_error(exc, self._e.provider))
            return False
        return True

    async def _send_text_block(self, text: str, role: str, *, interactive: bool) -> None:
        name = str(uuid.uuid4())
        send = self._send_required if not self._started else self._send
        await send(
            _event(
                "contentStart",
                promptName=self.prompt_name,
                contentName=name,
                type="TEXT",
                interactive=interactive,
                role=role,
                textInputConfiguration={"mediaType": "text/plain"},
            )
        )
        for piece in _chunks(text):
            await send(
                _event("textInput", promptName=self.prompt_name, contentName=name, content=piece)
            )
        await send(_event("contentEnd", promptName=self.prompt_name, contentName=name))

    async def _send_audio(self, frame: AudioFrame) -> None:
        self._last_input_at = now()
        self._track_user(frame)
        await self._push_audio(frame)

    async def _push_audio(self, frame: AudioFrame) -> None:
        if not self._audio_open or self._closing:
            return
        self._stream_pos += frame.duration
        await self._send(
            _event(
                "audioInput",
                promptName=self.prompt_name,
                contentName=self.audio_content,
                content=frame.to_base64(),
            )
        )

    async def _user_text_input(self, text: str) -> None:
        """A cross-modal user message (``interactive: true``): the model answers it."""
        self._muted = False
        self._requested_at = now()
        self._echoes.append(_normalize(text))
        await self._send_text_block(text, "USER", interactive=True)

    # --------------------------------------------------------------------- control
    def _warn_once(self, what: str, message: str) -> None:
        if what not in self._warned:
            self._warned.add(what)
            logger.warning("%s: %s", self._e.provider, message)

    async def commit_input(self) -> None:
        """No-op: Nova Sonic detects the end of the user's turn itself."""

    async def clear_input(self) -> None:
        """No-op: streamed audio has already been heard by the model."""

    async def send_text(self, text: str, *, respond: bool = True) -> None:
        """A typed user message (cross-modal input). Nova always answers it."""
        if not respond:
            self._warn_once(
                "text", "Nova Sonic answers every text message; respond=False is ignored"
            )
        await self._user_text_input(text)

    async def create_response(self, *, instructions: str | None = None) -> None:
        """Nova has no explicit response trigger: sends a short text turn instead."""
        await self._user_text_input(instructions or RESPOND_TEXT)

    async def say(self, text: str) -> None:
        await self._user_text_input(SAY_INSTRUCTIONS.format(text=text))

    async def cancel_response(self) -> None:
        """End the current response locally and drop the rest of this model turn (the
        protocol has no cancel; the model stops by itself on barge-in)."""
        if self._reply is None:
            return
        self._muted = True
        self._end_reply("cancelled")

    async def send_tool_output(self, output: FunctionCallOutput, *, respond: bool = True) -> None:
        """Send a ``toolResult`` block; the model speaks about the result by itself."""
        call_id = output.call_id
        if call_id not in self._tool_calls:
            # a result this connection did not ask for (e.g. from before a rotation)
            name = output.name or "a background task"
            await self._user_text_input(f"(Result of {name}: {output.output})")
            return
        self._tool_calls.discard(call_id)
        if not respond:
            self._warn_once(
                "silent", "Nova Sonic decides itself whether to speak about a tool result"
            )
        name = str(uuid.uuid4())
        await self._send(
            _event(
                "contentStart",
                promptName=self.prompt_name,
                contentName=name,
                interactive=False,
                type="TOOL",
                role="TOOL",
                toolResultInputConfiguration={
                    "toolUseId": call_id,
                    "type": "TEXT",
                    "textInputConfiguration": {"mediaType": "text/plain"},
                },
            )
        )
        await self._send(
            _event(
                "toolResult",
                promptName=self.prompt_name,
                contentName=name,
                content=_tool_result_content(output),
            )
        )
        await self._send(_event("contentEnd", promptName=self.prompt_name, contentName=name))
        self._muted = False

    async def send_async_tool_output(
        self, output: FunctionCallOutput, *, scheduling: ToolScheduling = "when_idle"
    ) -> None:
        await self.send_tool_output(output, respond=scheduling != "silent")

    async def update(
        self, *, instructions: str | None = None, tools: list[FunctionTool] | None = None
    ) -> None:
        """Stored for the next connection (the prompt of a live stream is fixed;
        :class:`NovaSonicConnection` rotates to apply it)."""
        if instructions is not None:
            self.options.instructions = instructions
        if tools is not None:
            self.options.tools = list(tools)

    # ------------------------------------------------------------------ receiving
    async def _read(self, stream: NovaStream) -> None:
        error: VoiceAgentError | None = None
        try:
            while True:
                event = await stream.receive()
                if event is None:
                    break
                try:
                    self._dispatch(event)
                except Exception:
                    logger.exception("%s: failed to handle an output event", self._e.provider)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            error = map_aws_error(exc, self._e.provider)
        if self._closing or self.closed or self._failed:
            return
        if error is None:
            error = ProviderConnectionError(
                f"{self._e.provider}: the stream ended", provider=self._e.provider
            )
        self._fail(error)

    def _fail(self, error: Exception) -> None:
        if self.closed or self._closing or self._failed:
            return
        self._failed = True
        self._audio_open = False  # the stream is broken: no closing events
        self._emit(EngineErrorEvent(error=error, recoverable=False))
        self._tasks.spawn(self.aclose(), name="nova-sonic-close")

    def _dispatch(self, message: Mapping[str, Any]) -> None:
        event = message.get("event")
        if not isinstance(event, Mapping):
            return
        for name, body in event.items():
            if not isinstance(body, Mapping):
                continue
            if self.session_id is None and isinstance(body.get("sessionId"), str):
                self.session_id = body["sessionId"]
            handler = self._handlers.get(name)
            if handler is not None:
                handler(self, body)
            else:
                logger.debug("%s: unhandled event %s", self._e.provider, name)

    def _on_completion_start(self, ev: Mapping[str, Any]) -> None:
        pass

    def _on_completion_end(self, ev: Mapping[str, Any]) -> None:
        self._muted = False
        self._end_reply("completed")

    def _on_content_start(self, ev: Mapping[str, Any]) -> None:
        cid = str(ev.get("contentId") or new_id("content_"))
        content = _Content(str(ev.get("role") or ""), str(ev.get("type") or ""), _stage(ev))
        self._contents[cid] = content
        while len(self._contents) > _MAX_TRACKED:
            del self._contents[next(iter(self._contents))]
        if content.role == "USER":
            self._muted = False  # a new user turn: the cancelled model turn is over
            if self._user_item is None:
                self._user_item = new_id("item_")
            return
        if content.type == "TOOL":
            return  # handled by toolUse (never dropped: the model waits for a result)
        if self._muted:
            content.ignored = True
            return
        reply = self._reply
        if content.type == "TEXT" and content.stage == "FINAL":
            # what was spoken, after the audio: belongs to the current response, if any
            if reply is None:
                content.ignored = True
            else:
                content.response_id = reply.response_id
            return
        if reply is not None and reply.audio_ended:
            self._end_reply("completed")  # the model started a new turn
            reply = None
        if reply is None:
            reply = self._begin_reply()
        reply.closing_at = None
        content.response_id = reply.response_id

    def _on_text_output(self, ev: Mapping[str, Any]) -> None:
        text = ev.get("content")
        if not isinstance(text, str):
            return
        if _is_interrupt_marker(text):
            self._on_interrupted()
            return
        content = self._contents.get(str(ev.get("contentId")))
        role = content.role if content is not None else str(ev.get("role") or "")
        if role == "USER":
            if content is not None:
                content.text.append(text)
            self._on_user_text(text)
            return
        if content is None or content.ignored:
            return
        reply = self._reply
        if reply is None or content.response_id != reply.response_id:
            return
        wanted = "FINAL" if self._e.transcript == "final" else "SPECULATIVE"
        if content.type != "TEXT" or (content.stage or "SPECULATIVE") != wanted:
            return
        delta = text.strip()
        if not delta:
            return
        if reply.text:
            delta = " " + delta
        reply.text += delta
        self._emit(ResponseText(response_id=reply.response_id, item_id=reply.item_id, delta=delta))

    def _on_audio_output(self, ev: Mapping[str, Any]) -> None:
        content = self._contents.get(str(ev.get("contentId")))
        data = ev.get("content")
        if content is None or content.ignored or not isinstance(data, str) or not data:
            return
        reply = self._reply
        if reply is None or content.response_id != reply.response_id:
            return
        raw = self._odd_byte + base64.b64decode(data)
        cut = len(raw) - len(raw) % 2
        self._odd_byte = raw[cut:]
        if not cut:
            return
        frame = AudioFrame(raw[:cut], self._e.output_sample_rate)
        if reply.first_audio_at is None:
            reply.first_audio_at = now()
        self._emit(ResponseAudio(response_id=reply.response_id, item_id=reply.item_id, frame=frame))

    def _on_tool_use(self, ev: Mapping[str, Any]) -> None:
        call_id = str(ev.get("toolUseId") or new_id("call_"))
        if call_id in self._tool_calls:
            return
        arguments = ev.get("content")
        if isinstance(arguments, Mapping):
            arguments = json.dumps(arguments)
        call = FunctionCall(
            name=str(ev.get("toolName") or ""),
            arguments=str(arguments) if arguments else "{}",
            call_id=call_id,
        )
        self._tool_calls.add(call_id)
        while len(self._tool_calls) > _MAX_TRACKED:
            self._tool_calls.pop()
        reply = self._reply
        if reply is None and not self._muted:
            reply = self._begin_reply()
        response_id = reply.response_id if reply is not None else new_id("resp_")
        self._emit(ResponseToolCall(response_id=response_id, call=call))

    def _on_content_end(self, ev: Mapping[str, Any]) -> None:
        content = self._contents.pop(str(ev.get("contentId")), None)
        stop = str(ev.get("stopReason") or "").upper()
        if stop == "INTERRUPTED":
            self._on_interrupted()
            return
        ctype = str(ev.get("type") or (content.type if content is not None else "")).upper()
        role = content.role if content is not None else ""
        if role == "USER":
            if stop == "END_TURN":
                self._commit_user_turn()
            return
        if self._muted:
            if stop == "END_TURN":
                self._muted = False  # the cancelled model turn is over
            return
        reply = self._reply
        if reply is None or content is None or content.response_id != reply.response_id:
            if ctype == "TOOL" and reply is not None:
                reply.closing_at = now() + self._e.tool_grace
            return
        if ctype == "TOOL":
            reply.closing_at = now() + self._e.tool_grace
        elif ctype == "AUDIO" and stop == "END_TURN":
            reply.audio_ended = True
            if self._e.transcript == "final":
                reply.closing_at = now() + self._e.final_grace
            else:
                self._end_reply("completed")
        elif (
            ctype == "TEXT"
            and content.stage == "FINAL"
            and stop == "END_TURN"
            and (reply.audio_ended or reply.first_audio_at is None)
        ):
            self._end_reply("completed")

    def _on_usage(self, ev: Mapping[str, Any]) -> None:
        details = ev.get("details")
        total = details.get("total") if isinstance(details, Mapping) else None
        if isinstance(total, Mapping):
            self.usage = _usage(total)

    def _on_interrupted(self) -> None:
        """Barge-in reported by the model: the user talked over the agent."""
        if self._muted:
            self._muted = False  # the cancelled turn ended
            return
        reply = self._reply
        if reply is None:
            return
        if not self._user_reported:
            start = self._user_speech_start if self._user_speaking else None
            self._emit(InputSpeechStarted(audio_time=start))
            self._user_pending = True
            if self._user_speaking:
                self._user_reported = True
            else:
                self._emit(InputSpeechStopped(audio_time=self.input_audio_time))
        self._end_reply("cancelled")

    # --------------------------------------------------------------------- replies
    def _begin_reply(self) -> _Reply:
        trigger: float | None = None
        if self._user_pending or self._user_text or self._user_item is not None:
            if not self._user_speaking and self._user_speech_end is not None:
                trigger = self.audio_time_to_wall(self._user_speech_end)
            self._commit_user_turn()
        if trigger is None and self._requested_at is not None:
            trigger = self._requested_at
        self._requested_at = None
        reply = _Reply(trigger)
        self._reply = reply
        self._emit(ResponseStarted(response_id=reply.response_id))
        return reply

    def _end_reply(self, status: ResponseStatus) -> None:
        reply, self._reply = self._reply, None
        if reply is None:
            return
        usage = _usage_delta(self.usage, self._usage_mark)
        self._usage_mark = self.usage
        self._emit(ResponseDone(response_id=reply.response_id, status=status, usage=usage))
        ttfb = None
        if reply.trigger_at is not None and reply.first_audio_at is not None:
            ttfb = max(0.0, reply.first_audio_at - reply.trigger_at)
        self._e.emit(
            "metrics",
            EngineMetrics(
                provider=self._e.provider,
                model=self._e.model,
                response_id=reply.response_id,
                ttfb=ttfb,
                duration=now() - reply.started_at,
                input_text_tokens=usage.input_text_tokens,
                input_audio_tokens=usage.input_audio_tokens,
                output_text_tokens=usage.output_text_tokens,
                output_audio_tokens=usage.output_audio_tokens,
                cancelled=status == "cancelled",
            ),
        )

    # ------------------------------------------------------------------- user side
    def _on_user_text(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        if self._user_item is None:
            self._user_item = new_id("item_")
        self._user_text.append(text)
        joined = " ".join(self._user_text)
        if _normalize(joined) in self._echoes:
            return  # the model's echo of a text message we sent
        self._emit(InputTranscript(item_id=self._user_item, text=joined, is_final=False))

    def _commit_user_turn(self) -> None:
        parts, self._user_text = self._user_text, []
        item_id, self._user_item = self._user_item, None
        pending, self._user_pending = self._user_pending, False
        text = " ".join(parts).strip()
        if text and _normalize(text) in self._echoes:
            self._echoes.remove(_normalize(text))
            return  # our own text message, not user speech
        if not text and not pending:
            return
        if self._user_reported:
            self._user_reported = False
            self._emit(InputSpeechStopped(audio_time=self._user_speech_end))
        item_id = item_id or new_id("item_")
        self._emit(InputCommitted(item_id=item_id))
        if text:
            self._emit(InputTranscript(item_id=item_id, text=text, is_final=True))

    def _track_user(self, frame: AudioFrame) -> None:
        for ev in self._vad.push_audio(frame):
            if ev.type == VADEventType.START_OF_SPEECH:
                self._user_speaking = True
                self._user_pending = True
                self._user_speech_start = max(0.0, ev.audio_time - ev.speech_duration)
            elif ev.type == VADEventType.END_OF_SPEECH:
                self._user_speaking = False
                end = max(0.0, ev.audio_time - ev.silence_duration)
                self._user_speech_end = end
                if self._user_reported:
                    self._user_reported = False
                    self._emit(InputSpeechStopped(audio_time=end))
        if self._user_speaking and not self._user_reported and self._floor_free():
            self._user_reported = True
            self._emit(InputSpeechStarted(audio_time=self._user_speech_start))

    def _floor_free(self) -> bool:
        return self._e.report_overlap or self._reply is None

    # ----------------------------------------------------------------------- ticker
    async def _tick(self) -> None:
        """Response deadlines and keep-alive silence."""
        silence = AudioFrame.silence(_KEEPALIVE_CHUNK, self._e.input_sample_rate)
        while not self.closed:
            await asyncio.sleep(_TICK)
            t = now()
            reply = self._reply
            if reply is not None and reply.closing_at is not None and t >= reply.closing_at:
                self._end_reply("completed")
            if not self._e.keepalive or not self._audio_open:
                continue
            if self._last_input_at is not None and t - self._last_input_at < _KEEPALIVE_IDLE:
                continue
            behind = (t - self._t0) - self._stream_pos
            while behind >= _KEEPALIVE_CHUNK / 2 and self._audio_open:
                await self._push_audio(silence)
                behind -= _KEEPALIVE_CHUNK

    _handlers: dict[str, Any] = {
        "completionStart": _on_completion_start,
        "completionEnd": _on_completion_end,
        "contentStart": _on_content_start,
        "textOutput": _on_text_output,
        "audioOutput": _on_audio_output,
        "toolUse": _on_tool_use,
        "contentEnd": _on_content_end,
        "usageEvent": _on_usage,
    }


def _event(name: str, **body: Any) -> dict[str, Any]:
    return {"event": {name: body}}
