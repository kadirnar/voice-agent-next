"""Anthropic Claude LLM provider (Messages API via the official ``anthropic`` SDK).

``create("llm", "anthropic/claude-haiku-4-5")`` (or the alias ``"claude/..."``) returns an
:class:`AnthropicLLM` that streams text deltas, emits *complete* tool calls, uses prompt
caching and reports usage including cache reads and writes. Install the extra
(``pip install 'voice-agent-next[anthropic]'``) and set ``ANTHROPIC_API_KEY``.

How a :class:`~voice_agent_next.chat.ChatContext` maps onto the Messages API
(:func:`to_anthropic_messages`):

* system/developer messages become the top-level ``system`` prompt (one text block each);
* consecutive items of the same role merge into one turn; the conversation always starts
  and ends with a user turn (a short placeholder turn is added when needed);
* :class:`~voice_agent_next.chat.FunctionCall` items become assistant ``tool_use`` blocks
  and their :class:`~voice_agent_next.chat.FunctionCallOutput` s become ``tool_result``
  blocks at the start of the next user turn (``is_error`` preserved). Calls without an
  output and outputs without a call are dropped, because the API rejects unmatched blocks;
* images become ``image`` blocks (``https://`` or ``data:`` URLs); audio is represented by
  its transcript, and audio without a transcript raises a clear :class:`ConfigurationError`;
* an interrupted assistant message contributes exactly the text the user heard.
"""

from __future__ import annotations

import base64
import json
import re
import sys
import urllib.parse
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from ..chat import (
    AudioContent,
    ChatContext,
    ChatMessage,
    FunctionCall,
    FunctionCallOutput,
    ImageContent,
)
from ..errors import (
    AuthenticationError,
    ConfigurationError,
    ProviderConnectionError,
    ProviderError,
    ProviderTimeoutError,
    RateLimitError,
)
from ..llm import LLM, ChatChunk, CompletionUsage, LLMCapabilities, LLMStream, ToolChoice
from ..registry import register_provider
from ..tools import FunctionTool
from ..utils.deps import require
from ..utils.log import logger

__all__ = [
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_MODEL",
    "AnthropicLLM",
    "AnthropicLLMStream",
    "AnthropicPrompt",
    "AnthropicUsage",
    "map_anthropic_error",
    "to_anthropic_messages",
    "to_anthropic_tool_choice",
    "to_anthropic_tools",
]

PROVIDER = "anthropic"
DEFAULT_MODEL = "claude-haiku-4-5"
DEFAULT_MAX_TOKENS = 1024

START_PLACEHOLDER = "(start of the conversation)"
"""User turn inserted when the conversation is empty or starts with the assistant."""
CONTINUE_PLACEHOLDER = "(continue)"
"""User turn appended when the conversation ends with an assistant turn (no prefill)."""

_CACHEABLE_BLOCKS = frozenset({"text", "image", "tool_use", "tool_result"})
_INVALID_TOOL_ID_CHARS = re.compile(r"[^a-zA-Z0-9_-]")
_FINISH_REASONS = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "model_context_window_exceeded": "length",
    "tool_use": "tool_calls",
    "refusal": "content_filter",
}
_STATUS_BY_ERROR_TYPE = {
    "invalid_request_error": 400,
    "authentication_error": 401,
    "billing_error": 402,
    "permission_error": 403,
    "not_found_error": 404,
    "request_too_large": 413,
    "rate_limit_error": 429,
    "api_error": 500,
    "timeout_error": 504,
    "overloaded_error": 529,
}


# ------------------------------------------------------------------------------ usage
@dataclass(slots=True)
class AnthropicUsage(CompletionUsage):
    """Token usage of one request, including the prompt-cache breakdown.

    ``prompt_tokens`` is the whole prompt: uncached input + cache writes
    (``cache_creation_tokens``) + cache reads (``cached_tokens``).
    """

    cache_creation_tokens: int = 0

    @property
    def uncached_prompt_tokens(self) -> int:
        """Prompt tokens billed at the base input price (neither read nor written)."""
        return self.prompt_tokens - self.cached_tokens - self.cache_creation_tokens


# ------------------------------------------------------------------------- conversion
@dataclass(slots=True)
class AnthropicPrompt:
    """A :class:`ChatContext` converted to Messages API ``system`` and ``messages``."""

    system: list[dict[str, Any]] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)
    stable_system_blocks: int = 0
    """Number of leading ``system`` blocks (instructions given before the conversation).

    Later system messages (e.g. per-response instructions) follow them, so the prompt-cache
    breakpoint goes after the stable ones."""
    tool_names: set[str] = field(default_factory=set)
    """Names of the tools referenced by ``tool_use`` blocks in ``messages``."""
    ends_with_placeholder: bool = False
    """``messages`` ends with :data:`CONTINUE_PLACEHOLDER` (not part of later requests)."""


def to_anthropic_messages(ctx: ChatContext) -> AnthropicPrompt:
    """Convert ``ctx`` to Messages API ``system`` blocks and ``messages`` turns.

    Raises:
        ConfigurationError: a user message holds audio without a transcript (Claude does
            not accept audio input) or an image with an unsupported URL.
    """
    items = list(ctx.items)
    call_ids = {item.call_id for item in items if isinstance(item, FunctionCall)}
    outputs: dict[str, FunctionCallOutput] = {}
    for item in items:
        if isinstance(item, FunctionCallOutput) and item.call_id in call_ids:
            outputs.setdefault(item.call_id, item)

    prompt = AnthropicPrompt()
    turns = prompt.messages
    pending: list[dict[str, Any]] = []  # tool_result blocks answering the open assistant turn
    sent_calls: set[str] = set()
    in_conversation = False

    def turn(role: str) -> list[dict[str, Any]]:
        if not turns or turns[-1]["role"] != role:
            turns.append({"role": role, "content": []})
        return cast(list[dict[str, Any]], turns[-1]["content"])

    def flush_results() -> None:
        # Results must immediately follow the assistant turn holding their tool_use blocks
        # and come first in the user turn, so they open a new user turn here.
        if pending:
            turn("user").extend(pending)
            pending.clear()

    for item in items:
        if isinstance(item, ChatMessage):
            if item.role in ("system", "developer"):
                text = _plain_text(item)
                if text:
                    prompt.system.append({"type": "text", "text": text})
                    if not in_conversation:
                        prompt.stable_system_blocks = len(prompt.system)
                continue
            in_conversation = True
            flush_results()
            if item.role == "user":
                blocks = _user_blocks(item)
            else:
                text = _plain_text(item)
                blocks = [{"type": "text", "text": text}] if text else []
            if blocks:
                turn(item.role).extend(blocks)
        elif isinstance(item, FunctionCall):
            in_conversation = True
            output = outputs.get(item.call_id)
            if output is None or item.call_id in sent_calls:
                logger.debug("anthropic: skipping tool call %s (no output/duplicate)", item.call_id)
                continue
            sent_calls.add(item.call_id)
            turn("assistant").append(_tool_use_block(item))
            pending.append(_tool_result_block(output))
            prompt.tool_names.add(item.name)
        else:
            in_conversation = True
            if item.call_id not in call_ids:
                logger.debug("anthropic: skipping tool output %s without a call", item.call_id)
            flush_results()
    flush_results()

    if not turns or turns[0]["role"] != "user":
        turns.insert(0, {"role": "user", "content": [{"type": "text", "text": START_PLACEHOLDER}]})
    if turns[-1]["role"] == "assistant":
        turns.append({"role": "user", "content": [{"type": "text", "text": CONTINUE_PLACEHOLDER}]})
        prompt.ends_with_placeholder = True
    return prompt


def to_anthropic_tools(tools: Sequence[FunctionTool]) -> list[dict[str, Any]]:
    """Messages API tool definitions (``name``/``description``/``input_schema``/``strict``)."""
    specs: list[dict[str, Any]] = []
    for tool in tools:
        schema = dict(tool.parameters or {})
        schema.setdefault("type", "object")
        spec: dict[str, Any] = {"name": tool.name}
        if tool.description:
            spec["description"] = tool.description
        spec["input_schema"] = schema
        if tool.strict:
            spec["strict"] = True
        specs.append(spec)
    return specs


def to_anthropic_tool_choice(
    choice: ToolChoice | None,
    *,
    tool_names: Collection[str] = (),
    parallel_tool_calls: bool | None = None,
) -> dict[str, Any] | None:
    """Map a :data:`~voice_agent_next.llm.ToolChoice` to the Messages API ``tool_choice``.

    ``"auto"`` (or ``None``) -> ``auto`` (omitted unless parallel calls are disabled),
    ``"required"``/``"any"`` -> ``any``, ``"none"`` -> ``none``, anything else -> that tool.
    ``parallel_tool_calls=False`` sets ``disable_parallel_tool_use``.
    """
    if choice == "none":
        return {"type": "none"}
    result: dict[str, Any]
    if choice is None or choice == "auto":
        if parallel_tool_calls is not False:
            return None
        result = {"type": "auto"}
    elif choice in ("required", "any"):
        result = {"type": "any"}
    else:
        if tool_names and choice not in tool_names:
            raise ConfigurationError(
                f"tool_choice {choice!r} is not one of the provided tools: {sorted(tool_names)}"
            )
        result = {"type": "tool", "name": choice}
    if parallel_tool_calls is False:
        result["disable_parallel_tool_use"] = True
    return result


def _plain_text(msg: ChatMessage) -> str:
    """Text of a system/assistant message (audio transcripts when it has no text)."""
    parts = [c for c in msg.content if isinstance(c, str)]
    if not any(p.strip() for p in parts):
        parts = [c.transcript for c in msg.content if isinstance(c, AudioContent) and c.transcript]
    if any(isinstance(c, ImageContent) for c in msg.content):
        logger.warning("anthropic: images are only supported in user messages; dropped one")
    text = "".join(parts)
    return text if text.strip() else ""


def _user_blocks(msg: ChatMessage) -> list[dict[str, Any]]:
    has_text = any(isinstance(c, str) and c.strip() for c in msg.content)
    blocks: list[dict[str, Any]] = []
    text: list[str] = []

    def flush_text() -> None:
        joined = "".join(text)
        text.clear()
        if joined.strip():
            blocks.append({"type": "text", "text": joined})

    for part in msg.content:
        if isinstance(part, str):
            text.append(part)
        elif isinstance(part, AudioContent):
            if has_text:
                continue  # the message text already says what the audio contains
            if not (part.transcript and part.transcript.strip()):
                raise ConfigurationError(
                    "Anthropic Claude models do not accept audio input: user message "
                    f"{msg.id} contains audio without a transcript. Add an STT component "
                    "to the cascade (or set AudioContent.transcript)."
                )
            text.append(part.transcript)
        else:
            flush_text()
            blocks.append(_image_block(part))
    flush_text()
    return blocks


def _image_block(image: ImageContent) -> dict[str, Any]:
    url = image.url.strip()
    if url.startswith("data:"):
        header, sep, payload = url[5:].partition(",")
        if not sep:
            raise ConfigurationError("malformed data: URL in ImageContent")
        media_type, *params = header.split(";")
        media_type = (media_type.strip() or image.mime_type or "").lower()
        if not media_type:
            raise ConfigurationError(
                "image data: URL has no media type; set ImageContent.mime_type"
            )
        if any(p.strip().lower() == "base64" for p in params):
            data = "".join(payload.split())
        else:
            data = base64.b64encode(urllib.parse.unquote_to_bytes(payload)).decode("ascii")
        return {
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": data},
        }
    if url.startswith(("https://", "http://")):
        return {"type": "image", "source": {"type": "url", "url": url}}
    raise ConfigurationError(
        f"unsupported image URL for Anthropic (expected https:// or data:): {url[:40]!r}"
    )


def _tool_id(call_id: str) -> str:
    """Tool-use ids must match ``[a-zA-Z0-9_-]+``; ids from other providers may not."""
    return _INVALID_TOOL_ID_CHARS.sub("_", call_id) or "call"


def _tool_use_block(call: FunctionCall) -> dict[str, Any]:
    try:
        arguments = call.parsed_arguments()
    except ValueError:
        logger.warning(
            "anthropic: tool call %s has invalid JSON arguments; sent as {}", call.call_id
        )
        arguments = {}
    return {"type": "tool_use", "id": _tool_id(call.call_id), "name": call.name, "input": arguments}


def _tool_result_block(output: FunctionCallOutput) -> dict[str, Any]:
    block: dict[str, Any] = {"type": "tool_result", "tool_use_id": _tool_id(output.call_id)}
    if output.output.strip():
        block["content"] = output.output
    elif output.is_error:
        block["content"] = "The tool call failed."
    if output.is_error:
        block["is_error"] = True
    return block


def _history_tool(name: str) -> dict[str, Any]:
    """Definition for a tool that only appears in the history (the API requires tools to
    be defined when ``tool_use`` blocks are present); sent with ``tool_choice: none``."""
    return {
        "name": name,
        "description": "Tool used earlier in the conversation; not available now.",
        "input_schema": {"type": "object"},
    }


def _apply_cache_control(
    tools: list[dict[str, Any]],
    prompt: AnthropicPrompt,
    cache_control: Mapping[str, Any],
    *,
    messages: bool,
) -> None:
    """Prompt-cache breakpoints: tools, stable system prompt, conversation so far (<= 3/4)."""
    if tools:
        tools[-1]["cache_control"] = dict(cache_control)
    if prompt.stable_system_blocks:
        prompt.system[prompt.stable_system_blocks - 1]["cache_control"] = dict(cache_control)
    if not messages:
        return
    turns = prompt.messages[:-1] if prompt.ends_with_placeholder else prompt.messages
    for turn in reversed(turns):
        for block in reversed(turn["content"]):
            if block.get("type") in _CACHEABLE_BLOCKS:
                block["cache_control"] = dict(cache_control)
                return


# ------------------------------------------------------------------------------ errors
def _error_details(body: object) -> tuple[str | None, str | None]:
    if isinstance(body, Mapping):
        error = body.get("error")
        if isinstance(error, Mapping):
            etype, message = error.get("type"), error.get("message")
            return (
                etype if isinstance(etype, str) else None,
                message if isinstance(message, str) else None,
            )
    return None, None


def _transport_error_kind(exc: BaseException) -> str | None:
    """``"timeout"``/``"connection"`` for raw httpx(2) errors raised while reading a stream."""
    for name in ("httpx2", "httpx"):
        module = sys.modules.get(name)
        if module is None:
            continue
        if isinstance(exc, module.TimeoutException):
            return "timeout"
        if isinstance(exc, module.TransportError):
            return "connection"
    return None


def map_anthropic_error(sdk: Any, exc: BaseException) -> ProviderError | None:
    """Map an ``anthropic`` SDK / transport exception to :mod:`voice_agent_next.errors`.

    Returns ``None`` for exceptions that are not provider failures.
    """
    if isinstance(exc, sdk.APITimeoutError):
        return ProviderTimeoutError("Anthropic request timed out", provider=PROVIDER)
    if isinstance(exc, sdk.APIConnectionError):
        return ProviderConnectionError(f"cannot reach the Anthropic API: {exc}", provider=PROVIDER)
    if isinstance(exc, sdk.APIStatusError):
        etype, message = _error_details(exc.body)
        status = int(exc.status_code)
        if status < 400:  # an `error` event inside a successful (200) event stream
            status = _STATUS_BY_ERROR_TYPE.get(etype or "", 500)
        text = f"Anthropic API error {status}{f' ({etype})' if etype else ''}: "
        text += message or str(getattr(exc, "message", exc))
        request_id = getattr(exc, "request_id", None)
        if request_id:
            text += f" [request-id: {request_id}]"
        if status in (401, 403):
            return AuthenticationError(text, provider=PROVIDER, status_code=status)
        if status == 429:
            return RateLimitError(text, provider=PROVIDER, status_code=status)
        if status in (408, 504):
            return ProviderTimeoutError(text, provider=PROVIDER, status_code=status)
        return ProviderError(
            text, provider=PROVIDER, status_code=status, retryable=status == 409 or status >= 500
        )
    kind = _transport_error_kind(exc)
    if kind == "timeout":
        return ProviderTimeoutError(f"Anthropic stream timed out: {exc!r}", provider=PROVIDER)
    if kind == "connection":
        return ProviderConnectionError(f"Anthropic connection failed: {exc!r}", provider=PROVIDER)
    if isinstance(exc, sdk.AnthropicError):
        return ProviderError(f"Anthropic SDK error: {exc}", provider=PROVIDER)
    return None


# --------------------------------------------------------------------------------- LLM
@register_provider(
    "llm",
    PROVIDER,
    description="Anthropic Claude (Messages API: streaming, tool use, prompt caching)",
    default_model=DEFAULT_MODEL,
    models=(
        "claude-haiku-4-5",
        "claude-sonnet-5",
        "claude-sonnet-4-6",
        "claude-opus-5",
        "claude-opus-4-8",
    ),
    env=("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"),
    extra="anthropic",
    requires=("anthropic",),
    local=False,
    aliases=("claude",),
)
class AnthropicLLM(LLM):
    """Anthropic Claude chat model over the streaming Messages API.

    Args:
        model: model id, e.g. ``"claude-haiku-4-5"`` (default), ``"claude-sonnet-5"``.
        api_key: API key; defaults to ``ANTHROPIC_API_KEY`` / ``ANTHROPIC_AUTH_TOKEN`` (or
            any other credential source the SDK resolves, such as ``ant auth login``).
        base_url: API base URL (defaults to ``ANTHROPIC_BASE_URL`` or the public API).
        temperature: sampling temperature; only sent when set (Claude Opus 4.7+ and newer
            models reject it).
        max_tokens: output token cap per response (thinking tokens count towards it).
        tool_choice: default tool choice (``"auto"``, ``"required"``, ``"none"`` or a name).
        parallel_tool_calls: ``False`` sets ``disable_parallel_tool_use``.
        prompt_caching: put ``cache_control`` breakpoints on the tool definitions, the
            leading system prompt and the conversation tail.
        cache_ttl: ``"5m"`` (default) or ``"1h"`` cache lifetime.
        timeout: per-request timeout in seconds (connect is capped at 5 s); ``None`` keeps
            the SDK default (10 minutes).
        max_retries: SDK retries for connection errors, 408/409/429/5xx before streaming.
        keepalive_expiry: seconds an idle HTTP connection is kept for reuse (saves the TLS
            handshake between turns).
        extra_params: extra request body fields for every request, e.g.
            ``{"output_config": {"effort": "low"}}``, ``{"thinking": {...}}``,
            ``{"metadata": {...}}``. Per-call ``extra`` overrides them.
        extra_headers: extra HTTP headers for every request (e.g. ``anthropic-beta``).
        client: a pre-built ``anthropic.AsyncAnthropic`` compatible client (e.g. the
            Bedrock/Vertex/Foundry variants); connection options above are then ignored.
        http_client: an ``httpx2.AsyncClient`` for the SDK (proxies, custom transports).
    """

    provider = PROVIDER

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        tool_choice: ToolChoice | None = None,
        parallel_tool_calls: bool | None = None,
        prompt_caching: bool = True,
        cache_ttl: Literal["5m", "1h"] = "5m",
        timeout: float | None = 30.0,
        max_retries: int = 1,
        keepalive_expiry: float = 120.0,
        extra_params: Mapping[str, Any] | None = None,
        extra_headers: Mapping[str, str] | None = None,
        client: Any = None,
        http_client: Any = None,
    ) -> None:
        if max_tokens < 1:
            raise ConfigurationError(f"max_tokens must be >= 1, got {max_tokens}")
        if cache_ttl not in ("5m", "1h"):
            raise ConfigurationError(f"cache_ttl must be '5m' or '1h', got {cache_ttl!r}")
        super().__init__(
            model=model,
            capabilities=LLMCapabilities(
                tool_calling=True,
                parallel_tool_calls=parallel_tool_calls is not False,
                audio_input=False,
                image_input=True,
            ),
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self.tool_choice = tool_choice
        self.parallel_tool_calls = parallel_tool_calls
        self.prompt_caching = prompt_caching
        self.cache_ttl = cache_ttl
        self.extra_params: dict[str, Any] = dict(extra_params or {})
        self.extra_headers: dict[str, str] = dict(extra_headers or {})
        self._sdk = require("anthropic", extra="anthropic")
        self._owns_client = False
        self._closed = False
        if client is not None:
            self._client: Any = client
            return
        kwargs: dict[str, Any] = {"max_retries": max_retries}
        if api_key is not None:
            kwargs["api_key"] = api_key
        if base_url is not None:
            kwargs["base_url"] = base_url
        if timeout is not None:
            kwargs["timeout"] = self._sdk.Timeout(timeout, connect=min(timeout, 5.0))
        if http_client is None:
            limits = self._sdk.DEFAULT_CONNECTION_LIMITS
            http_client = self._sdk.DefaultAsyncHttpxClient(
                limits=type(limits)(
                    max_connections=limits.max_connections,
                    max_keepalive_connections=limits.max_keepalive_connections,
                    keepalive_expiry=keepalive_expiry,
                )
            )
            self._owns_client = True
        kwargs["http_client"] = http_client
        self._client = self._sdk.AsyncAnthropic(**kwargs)
        if not (
            getattr(self._client, "api_key", None)
            or getattr(self._client, "auth_token", None)
            or getattr(self._client, "credentials", None)
        ):
            raise AuthenticationError(
                "no Anthropic credentials: pass api_key=... or set ANTHROPIC_API_KEY",
                provider=PROVIDER,
            )

    @property
    def client(self) -> Any:
        """The underlying ``anthropic.AsyncAnthropic`` client."""
        return self._client

    @property
    def cache_control(self) -> dict[str, Any]:
        """The ``cache_control`` value used for prompt-cache breakpoints."""
        control: dict[str, Any] = {"type": "ephemeral"}
        if self.cache_ttl == "1h":
            control["ttl"] = "1h"
        return control

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
        return AnthropicLLMStream(
            self,
            ctx,
            tools=tools,
            tool_choice=tool_choice if tool_choice is not None else self.tool_choice,
            temperature=temperature,
            max_tokens=max_tokens,
            extra=extra,
        )

    def build_request(
        self,
        ctx: ChatContext,
        *,
        tools: Sequence[FunctionTool] = (),
        tool_choice: ToolChoice | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        extra: Mapping[str, Any] | None = None,
        cache_conversation: bool = True,
    ) -> dict[str, Any]:
        """Keyword arguments for ``client.messages.create``/``stream`` for this request."""
        prompt = to_anthropic_messages(ctx)
        tool_specs = to_anthropic_tools(tools)
        choice = None
        if tool_specs:
            choice = to_anthropic_tool_choice(
                tool_choice,
                tool_names=[t.name for t in tools],
                parallel_tool_calls=self.parallel_tool_calls,
            )
        elif prompt.tool_names:
            tool_specs = [_history_tool(name) for name in sorted(prompt.tool_names)]
            choice = {"type": "none"}
        if self.prompt_caching:
            _apply_cache_control(
                tool_specs, prompt, self.cache_control, messages=cache_conversation
            )
        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": DEFAULT_MAX_TOKENS if max_tokens is None else max_tokens,
            "messages": prompt.messages,
        }
        if prompt.system:
            params["system"] = prompt.system
        if tool_specs:
            params["tools"] = tool_specs
        if choice is not None:
            params["tool_choice"] = choice
        body = dict(self.extra_params)
        if temperature is not None:
            body["temperature"] = temperature
        body.update(extra or {})
        if body:
            params["extra_body"] = body  # merged into the JSON body by the SDK
        if self.extra_headers:
            params["extra_headers"] = dict(self.extra_headers)
        return params

    def map_error(self, exc: BaseException) -> ProviderError | None:
        """Map an SDK/transport exception to :mod:`voice_agent_next.errors` (or ``None``)."""
        return map_anthropic_error(self._sdk, exc)

    async def warmup(
        self, ctx: ChatContext | None = None, *, tools: Sequence[FunctionTool] = ()
    ) -> None:
        """Open the HTTP connection early and optionally pre-warm the prompt cache.

        Without arguments this sends a free ``GET /v1/models/{model}``: DNS, TCP and TLS are
        done before the first turn (and credentials/model id are checked). With the agent
        prompt (``ctx``, e.g. its instructions as a system message) and/or ``tools``, it
        sends a ``max_tokens=0`` request that writes the tools + system prompt cache
        entry, so the first real turn reads it. Pass exactly what later requests send.
        Failures are logged, never raised.
        """
        try:
            thinking = self.extra_params.get("thinking")
            manual_thinking = isinstance(thinking, Mapping) and thinking.get("type") == "enabled"
            if (ctx is not None or tools) and self.prompt_caching and not manual_thinking:
                await self._prewarm_cache(ctx or ChatContext(), tools)
                return
            models = getattr(self._client, "models", None)
            if models is not None:
                await models.retrieve(self.model)
        except Exception as exc:
            logger.warning("Anthropic warmup failed: %s", self.map_error(exc) or exc)

    async def _prewarm_cache(self, ctx: ChatContext, tools: Sequence[FunctionTool]) -> None:
        conversation = any(
            not (isinstance(item, ChatMessage) and item.role in ("system", "developer"))
            for item in ctx.items
        )
        # The breakpoint must not sit on the placeholder user turn of an empty conversation.
        params = self.build_request(ctx, tools=tools, cache_conversation=conversation)
        params["max_tokens"] = 0  # prefill only: writes the cache, generates nothing
        response = await self._client.messages.create(**params)
        usage = getattr(response, "usage", None)
        logger.debug(
            "Anthropic prompt cache pre-warmed (written=%s, read=%s)",
            getattr(usage, "cache_creation_input_tokens", None),
            getattr(usage, "cache_read_input_tokens", None),
        )

    async def aclose(self) -> None:
        if self._owns_client and not self._closed:
            self._closed = True
            await self._client.close()


@dataclass(slots=True)
class _PendingToolUse:
    call_id: str
    name: str
    initial_input: dict[str, Any] | None
    parts: list[str] = field(default_factory=list)

    def finish(self) -> FunctionCall | None:
        raw = "".join(self.parts).strip()
        if not raw:
            arguments = json.dumps(self.initial_input) if self.initial_input else "{}"
            return FunctionCall(name=self.name, arguments=arguments, call_id=self.call_id)
        try:
            value = json.loads(raw)
        except ValueError:
            value = None
        if not isinstance(value, dict):
            # cut off by max_tokens (or a refusal): never hand a truncated call to a tool
            logger.warning(
                "anthropic: dropping tool call %s (%s): incomplete JSON input (%d chars)",
                self.call_id,
                self.name,
                len(raw),
            )
            return None
        return FunctionCall(name=self.name, arguments=raw, call_id=self.call_id)


class AnthropicLLMStream(LLMStream):
    """Streams one Messages API response as :class:`ChatChunk` s."""

    _tool_uses: dict[int, _PendingToolUse]
    _stop_reason: str | None
    _tokens: dict[str, int]

    async def _run(self) -> None:
        llm = cast(AnthropicLLM, self._llm)
        params = llm.build_request(
            self.ctx,
            tools=self.tools,
            tool_choice=self.tool_choice,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            extra=self.extra,
        )
        self._tool_uses: dict[int, _PendingToolUse] = {}
        self._stop_reason: str | None = None
        self._tokens = {"input": 0, "cache_creation": 0, "cache_read": 0, "output": 0}
        try:
            async with llm.client.messages.stream(**params) as stream:
                async for event in stream:
                    self._on_event(event)
        except Exception as exc:
            mapped = llm.map_error(exc)
            if mapped is None:
                raise
            raise mapped from exc
        tokens = self._tokens
        self._push(
            ChatChunk(
                self.request_id,
                usage=AnthropicUsage(
                    prompt_tokens=tokens["input"] + tokens["cache_creation"] + tokens["cache_read"],
                    completion_tokens=tokens["output"],
                    cached_tokens=tokens["cache_read"],
                    cache_creation_tokens=tokens["cache_creation"],
                ),
                finish_reason=_FINISH_REASONS.get(self._stop_reason or "", self._stop_reason),
            )
        )

    def _on_event(self, event: Any) -> None:
        kind = getattr(event, "type", None)
        if kind == "content_block_delta":
            delta = event.delta
            if delta.type == "text_delta":
                if delta.text:
                    self._push(ChatChunk(self.request_id, delta=delta.text))
            elif delta.type == "input_json_delta":
                pending = self._tool_uses.get(event.index)
                if pending is not None:
                    pending.parts.append(delta.partial_json)
            # thinking/signature/citations deltas are not part of the spoken reply
        elif kind == "content_block_start":
            block = event.content_block
            if block.type == "tool_use":
                initial = getattr(block, "input", None)
                self._tool_uses[event.index] = _PendingToolUse(
                    call_id=block.id,
                    name=block.name,
                    initial_input=initial if isinstance(initial, dict) and initial else None,
                )
            elif block.type == "text" and getattr(block, "text", ""):
                self._push(ChatChunk(self.request_id, delta=block.text))
        elif kind == "content_block_stop":
            pending = self._tool_uses.pop(event.index, None)
            call = pending.finish() if pending is not None else None
            if call is not None:
                self._push(ChatChunk(self.request_id, tool_calls=[call]))
        elif kind == "message_start":
            self._record_usage(getattr(event.message, "usage", None))
        elif kind == "message_delta":
            self._stop_reason = getattr(event.delta, "stop_reason", None) or self._stop_reason
            self._record_usage(getattr(event, "usage", None))

    def _record_usage(self, usage: Any) -> None:
        # message_delta counts are cumulative; absent/None fields keep the earlier value
        if usage is None:
            return
        for key, attr in (
            ("input", "input_tokens"),
            ("cache_creation", "cache_creation_input_tokens"),
            ("cache_read", "cache_read_input_tokens"),
            ("output", "output_tokens"),
        ):
            value = getattr(usage, attr, None)
            if isinstance(value, int):
                self._tokens[key] = value
