"""Google Gemini LLM provider (``generateContent`` streaming through ``google-genai``).

``create("llm", "google/gemini-3.8-flash")`` (alias ``"gemini/..."``) returns a
:class:`GeminiLLM` that streams text deltas, emits *complete* tool calls, keeps thinking
low for voice latency (thoughts are never spoken), accepts audio and image input, and
reports token usage including implicit/explicit cache hits and thinking tokens. Install
the extra (``pip install 'voice-agent-next[google]'``) and set ``GOOGLE_API_KEY`` (or
``GEMINI_API_KEY``), or use Vertex AI (``vertexai=True`` + project/location + ADC).

The request is ``models.generate_content_stream`` with automatic function calling
disabled: tools are executed by the session, not by the SDK. The conversation mapping is
in :mod:`voice_agent_next.providers.google._format`.
"""

from __future__ import annotations

import base64
import binascii
import json
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from ...chat import ChatContext, FunctionCall
from ...errors import ConfigurationError, VoiceAgentError
from ...llm import LLM, ChatChunk, CompletionUsage, LLMCapabilities, LLMStream, ToolChoice
from ...registry import register_provider
from ...tools import FunctionTool
from ...utils.ids import new_id
from ...utils.log import logger
from ._common import API_KEY_ENV, PROVIDER, deep_merge, make_genai_client, map_google_error
from ._format import (
    SKIP_THOUGHT_SIGNATURE,
    CallMeta,
    default_thinking_config,
    gemini_generation,
    to_gemini_contents,
    to_gemini_tool_config,
    to_gemini_tools,
)

__all__ = [
    "DEFAULT_MODEL",
    "KNOWN_MODELS",
    "GeminiLLM",
    "GeminiLLMStream",
    "GeminiUsage",
    "to_usage",
]

DEFAULT_MODEL = "gemini-3.8-flash"
KNOWN_MODELS = (
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-pro-preview",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.5-pro",
)
THINKING_LEVELS = ("minimal", "low", "medium", "high")

_CALL_MEMORY = 1024
"""Function calls whose thought signature / API id are remembered (oldest dropped first)."""
_FINISH_REASONS = {
    "STOP": "stop",
    "MAX_TOKENS": "length",
    "SAFETY": "content_filter",
    "RECITATION": "content_filter",
    "BLOCKLIST": "content_filter",
    "PROHIBITED_CONTENT": "content_filter",
    "SPII": "content_filter",
    "IMAGE_SAFETY": "content_filter",
}
_EXPECTED_FINISH = frozenset({"STOP", "MAX_TOKENS", "FINISH_REASON_UNSPECIFIED"})


# ------------------------------------------------------------------------------- usage
@dataclass(slots=True)
class GeminiUsage(CompletionUsage):
    """Token usage of one request.

    ``prompt_tokens`` is the whole prompt (cache hits and server-side tool results
    included); ``completion_tokens`` includes the thinking tokens, which are billed as
    output.
    """

    thoughts_tokens: int = 0
    """Thinking tokens (part of ``completion_tokens``)."""
    tool_use_prompt_tokens: int = 0
    """Tokens of server-side tool results, e.g. Google Search (part of ``prompt_tokens``)."""

    @property
    def uncached_prompt_tokens(self) -> int:
        """Prompt tokens billed at the full input price."""
        return self.prompt_tokens - self.cached_tokens


def _field(obj: Any, name: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, Mapping):
        return obj.get(name)
    return getattr(obj, name, None)


def _count(obj: Any, name: str) -> int:
    value = _field(obj, name)
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0


def to_usage(metadata: Any) -> GeminiUsage:
    """``usage_metadata`` (SDK object or snake_case mapping) -> :class:`GeminiUsage`."""
    thoughts = _count(metadata, "thoughts_token_count")
    tool_use = _count(metadata, "tool_use_prompt_token_count")
    return GeminiUsage(
        prompt_tokens=_count(metadata, "prompt_token_count") + tool_use,
        completion_tokens=_count(metadata, "candidates_token_count") + thoughts,
        cached_tokens=_count(metadata, "cached_content_token_count"),
        thoughts_tokens=thoughts,
        tool_use_prompt_tokens=tool_use,
    )


def _enum_name(value: Any) -> str:
    """``FinishReason.STOP`` / ``"STOP"`` -> ``"STOP"``."""
    raw = getattr(value, "value", value)
    return str(raw).rsplit(".", 1)[-1].upper()


def _signature_bytes(value: str) -> bytes:
    """Bytes the SDK serializes back to exactly ``value`` (it sends bytes as base64url)."""
    try:
        data = base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
    except (binascii.Error, ValueError) as exc:
        raise ConfigurationError(f"fallback_thought_signature must be base64url: {exc}") from exc
    if base64.urlsafe_b64encode(data).decode("ascii").rstrip("=") != value.rstrip("="):
        raise ConfigurationError(f"fallback_thought_signature {value!r} is not canonical base64")
    return data


# --------------------------------------------------------------------------------- LLM
@register_provider(
    "llm",
    PROVIDER,
    description="Google Gemini (generateContent streaming: tools, low thinking, audio/image in)",
    default_model=DEFAULT_MODEL,
    models=KNOWN_MODELS,
    env=API_KEY_ENV,
    extra="google",
    requires=("google.genai",),
    local=False,
    aliases=("gemini",),
)
class GeminiLLM(LLM):
    """Gemini chat model over streaming ``generateContent``.

    Args:
        model: model id, e.g. ``"gemini-3.8-flash"`` (default), ``"gemini-3.5-flash-lite"``.
        api_key: Gemini API key; defaults to ``GOOGLE_API_KEY``, then ``GEMINI_API_KEY``.
            With Vertex AI it selects express mode.
        vertexai: use Vertex AI. Default: when ``project``/``location``/``credentials`` is
            given or ``GOOGLE_GENAI_USE_VERTEXAI=true``.
        project, location: Vertex AI project and region (default ``GOOGLE_CLOUD_PROJECT`` /
            ``GOOGLE_CLOUD_LOCATION``; the SDK falls back to the ADC project and ``global``).
        credentials: ``google.auth`` credentials for Vertex AI (default: ADC).
        temperature: sampling temperature; only sent when set (Google recommends keeping
            Gemini 3 at its default).
        max_tokens: ``max_output_tokens``; thinking tokens count towards it.
        thinking_level: ``"auto"`` (default) picks the lowest-latency setting for the model:
            ``minimal`` or ``low`` on Gemini 3, thinking off on Gemini 2.5 Flash. ``"minimal"``,
            ``"low"``, ``"medium"``, ``"high"`` set it explicitly; ``None`` sends nothing
            (model default). Thoughts are never spoken.
        thinking_budget: legacy token budget (Gemini 2.5: ``0`` = off, ``-1`` = dynamic);
            replaces ``thinking_level``.
        tool_choice: default tool choice (``"auto"``, ``"required"``, ``"none"`` or a name).
        audio_input: send user audio (:class:`~voice_agent_next.chat.AudioContent`) as audio
            (half-cascade without STT); ``False`` sends transcripts only.
        cached_content: name of an explicit context cache (``cachedContents/...``) holding
            the instructions and tools; they are then not sent with each request.
        extra_config: extra ``GenerateContentConfig`` fields for every request (snake_case),
            e.g. ``{"top_p": 0.9, "safety_settings": [...], "media_resolution": ...}``;
            ``tools`` entries (e.g. ``[{"google_search": {}}]``) are added to the function
            declarations. The per-call ``extra`` dict overrides them.
        fallback_thought_signature: signature sent for function calls of the current turn
            that have none (history from another model, injected calls); ``None`` disables
            it. Only used for Gemini 3+ models.
        timeout: seconds per request phase (connect, each streamed read); also sent as the
            server-side deadline. ``None`` disables it.
        max_retries: retries for connection errors and 408/429/5xx before streaming starts.
        keepalive_expiry: seconds an idle HTTP connection is kept for reuse.
        base_url, api_version, headers: endpoint overrides and extra HTTP headers.
        client: a pre-built ``google.genai.Client``; the connection options above are then
            ignored and the client is not closed by :meth:`aclose`.
        http_client: an ``httpx.AsyncClient`` for the SDK (proxies, custom transports, tests);
            not closed by :meth:`aclose`.
    """

    provider = PROVIDER

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        vertexai: bool | None = None,
        project: str | None = None,
        location: str | None = None,
        credentials: Any = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        thinking_level: str | None = "auto",
        thinking_budget: int | None = None,
        tool_choice: ToolChoice | None = None,
        audio_input: bool = True,
        cached_content: str | None = None,
        extra_config: Mapping[str, Any] | None = None,
        fallback_thought_signature: str | None = SKIP_THOUGHT_SIGNATURE,
        timeout: float | None = 30.0,
        max_retries: int = 1,
        keepalive_expiry: float = 120.0,
        base_url: str | None = None,
        api_version: str | None = None,
        headers: Mapping[str, str] | None = None,
        client: Any = None,
        http_client: Any = None,
    ) -> None:
        model = model or DEFAULT_MODEL
        if thinking_level is not None:
            thinking_level = thinking_level.strip().lower()
            if thinking_level not in ("auto", *THINKING_LEVELS):
                raise ConfigurationError(
                    f"thinking_level must be 'auto', None or one of {THINKING_LEVELS}, "
                    f"got {thinking_level!r}"
                )
        if thinking_budget is not None and thinking_level not in (None, "auto"):
            raise ConfigurationError("set either thinking_level or thinking_budget, not both")
        if max_tokens is not None and max_tokens < 1:
            raise ConfigurationError(f"max_tokens must be >= 1, got {max_tokens}")
        if max_retries < 0:
            raise ConfigurationError(f"max_retries must be >= 0, got {max_retries}")
        super().__init__(
            model=model,
            capabilities=LLMCapabilities(
                tool_calling=True,
                parallel_tool_calls=True,
                audio_input=audio_input,
                image_input=True,
            ),
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self.thinking_level = thinking_level
        self.thinking_budget = thinking_budget
        self.tool_choice = tool_choice
        self.cached_content = cached_content
        self.extra_config: dict[str, Any] = dict(extra_config or {})
        self.fallback_thought_signature = fallback_thought_signature
        self._fallback_signature = (
            _signature_bytes(fallback_thought_signature) if fallback_thought_signature else None
        )
        self._calls: OrderedDict[str, CallMeta] = OrderedDict()
        self._closed = False
        self._owns_client = client is None
        if client is not None:
            self._client: Any = client
            return
        self._client = make_genai_client(
            what="Gemini LLM",
            api_key=api_key,
            vertexai=vertexai,
            project=project,
            location=location,
            credentials=credentials,
            base_url=base_url,
            api_version=api_version,
            headers=headers,
            timeout=timeout,
            attempts=max_retries + 1,
            keepalive_expiry=keepalive_expiry,
            http_client=http_client,
        )

    @property
    def client(self) -> Any:
        """The underlying ``google.genai.Client``."""
        return self._client

    @property
    def vertexai(self) -> bool:
        """Whether requests go to Vertex AI (else the Gemini Developer API)."""
        return bool(getattr(getattr(self._client, "_api_client", None), "vertexai", False))

    # ---------------------------------------------------------------------- requests
    def _chat(
        self,
        ctx: ChatContext,
        *,
        tools: list[FunctionTool],
        tool_choice: ToolChoice | None,
        temperature: float | None,
        max_tokens: int | None,
        extra: dict[str, Any],
    ) -> LLMStream:
        return GeminiLLMStream(
            self,
            ctx,
            tools=tools,
            tool_choice=tool_choice if tool_choice is not None else self.tool_choice,
            temperature=temperature,
            max_tokens=max_tokens,
            extra=extra,
        )

    def thinking_config(self) -> dict[str, Any] | None:
        """The ``thinking_config`` sent with every request (``None``: not sent)."""
        if self.thinking_budget is not None:
            return {"thinking_budget": self.thinking_budget}
        if self.thinking_level is None:
            return None
        if self.thinking_level == "auto":
            return default_thinking_config(self.model)
        return {"thinking_level": self.thinking_level}

    def build_request(
        self,
        ctx: ChatContext,
        *,
        tools: Sequence[FunctionTool] = (),
        tool_choice: ToolChoice | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Keyword arguments for ``client.aio.models.generate_content_stream``."""
        version = gemini_generation(self.model)
        fallback = self._fallback_signature if version and version[0] >= 3 else None
        prompt = to_gemini_contents(
            ctx,
            audio_input=self.capabilities.audio_input,
            calls=self._calls,
            fallback_signature=fallback,
        )
        config: dict[str, Any] = {"automatic_function_calling": {"disable": True}}
        if prompt.system_instruction:
            config["system_instruction"] = prompt.system_instruction
        declared = to_gemini_tools(tools)
        if declared:
            config["tools"] = declared
            tool_config = to_gemini_tool_config(tool_choice, tool_names=[t.name for t in tools])
            if tool_config is not None:
                config["tool_config"] = tool_config
        if temperature is not None:
            config["temperature"] = temperature
        if max_tokens is not None:
            config["max_output_tokens"] = max_tokens
        thinking = self.thinking_config()
        if thinking:
            config["thinking_config"] = thinking
        options = deep_merge(self.extra_config, extra or {})
        extra_tools = options.pop("tools", None)
        config = deep_merge(config, options)
        if extra_tools:
            config["tools"] = [*config.get("tools", []), *extra_tools]
        if self.cached_content:
            # the API rejects instructions/tools next to a cache: they live in the cache
            config["cached_content"] = self.cached_content
            for key in ("system_instruction", "tools", "tool_config"):
                config.pop(key, None)
        return {"model": self.model, "contents": prompt.contents, "config": config}

    def remember_call(self, call_id: str, *, signature: bytes | None, api_id: bool) -> None:
        """Keep a call's thought signature / API id for when its result is sent back."""
        self._calls[call_id] = CallMeta(signature=signature, api_id=api_id)
        self._calls.move_to_end(call_id)
        while len(self._calls) > _CALL_MEMORY:
            self._calls.popitem(last=False)

    def map_error(self, exc: BaseException) -> VoiceAgentError | None:
        """Map an SDK/transport exception to :mod:`voice_agent_next.errors` (or ``None``)."""
        return map_google_error(exc, what="Gemini API")

    # --------------------------------------------------------------------- lifecycle
    async def warmup(self) -> None:
        """Open the HTTPS connection ahead of the first turn.

        Sends a free ``models.get`` for the model: DNS, TCP and TLS (and, on Vertex AI, the
        access token) are done before the first turn, and a bad key or model id shows up
        as a logged warning. Failures are logged, never raised.
        """
        try:
            await self._client.aio.models.get(model=self.model)
        except Exception as exc:
            logger.warning("Gemini warmup failed: %s", self.map_error(exc) or exc)

    async def aclose(self) -> None:
        if self._owns_client and not self._closed:
            self._closed = True
            await self._client.aio.aclose()
            self._client.close()  # the SDK's sync HTTP client (unused, but open)


# ------------------------------------------------------------------------------ stream
class GeminiLLMStream(LLMStream):
    """Streams one ``generateContent`` response as :class:`ChatChunk` s."""

    _finish: str | None
    _finish_message: str | None
    _blocked: str | None
    _called: bool

    async def _run(self) -> None:
        llm = cast(GeminiLLM, self._llm)
        request = llm.build_request(
            self.ctx,
            tools=self.tools,
            tool_choice=self.tool_choice,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            extra=self.extra,
        )
        self._finish = self._finish_message = self._blocked = None
        self._called = False
        try:
            stream = await llm.client.aio.models.generate_content_stream(**request)
            async for response in stream:
                self._on_response(llm, response)
        except Exception as exc:
            mapped = llm.map_error(exc)
            if mapped is None:
                raise
            raise mapped from exc
        self._push(ChatChunk(self.request_id, usage=self._usage, finish_reason=self._reason()))

    def _on_response(self, llm: GeminiLLM, response: Any) -> None:
        metadata = getattr(response, "usage_metadata", None)
        if metadata is not None:
            # keeps the base-class usage current: a cancelled (barge-in) or failed stream
            # still reports the prompt tokens billed so far
            self._usage = to_usage(metadata)
        feedback = getattr(response, "prompt_feedback", None)
        blocked = getattr(feedback, "block_reason", None) if feedback is not None else None
        if blocked:
            self._blocked = _enum_name(blocked)
            logger.warning("gemini: the prompt was blocked (%s)", self._blocked)
        for candidate in (getattr(response, "candidates", None) or [])[:1]:
            content = getattr(candidate, "content", None)
            for part in getattr(content, "parts", None) or ():
                if part.thought:
                    continue  # reasoning (or its summary) is never spoken
                if part.function_call is not None:
                    self._emit_call(llm, part.function_call, part.thought_signature)
                elif part.text:
                    self._push(ChatChunk(self.request_id, delta=part.text))
            if candidate.finish_reason is not None:
                self._finish = _enum_name(candidate.finish_reason)
                self._finish_message = candidate.finish_message

    def _emit_call(self, llm: GeminiLLM, call: Any, signature: bytes | None) -> None:
        if not call.name:
            logger.warning("gemini: dropping a function call without a name")
            return
        call_id = call.id or new_id("call_")
        llm.remember_call(call_id, signature=signature, api_id=bool(call.id))
        self._called = True
        arguments = json.dumps(call.args or {}, ensure_ascii=False)
        self._push(
            ChatChunk(
                self.request_id,
                tool_calls=[FunctionCall(name=call.name, arguments=arguments, call_id=call_id)],
            )
        )

    def _reason(self) -> str | None:
        reason = self._finish
        if reason is None:
            if self._blocked:
                return "content_filter"
            return "tool_calls" if self._called else None
        if reason not in _EXPECTED_FINISH:
            logger.warning(
                "gemini: generation stopped with %s%s",
                reason,
                f": {self._finish_message}" if self._finish_message else "",
            )
        mapped = _FINISH_REASONS.get(reason, reason.lower())
        return "tool_calls" if mapped == "stop" and self._called else mapped
