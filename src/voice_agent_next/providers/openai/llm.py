"""OpenAI Chat Completions LLM, also the client for every OpenAI-compatible server.

:class:`OpenAILLM` streams ``/chat/completions`` through the official ``openai`` SDK
(``pip install 'voice-agent-next[openai]'``). It works with any server that speaks the
same protocol: pass ``base_url`` (and ``api_key`` if needed), or use one of the
preconfigured hosts built on :class:`OpenAICompatibleLLM` (``ollama``, ``llamacpp``,
``vllm``, ``lmstudio``, ``groq``, ``cerebras``, ``together``, ``openrouter``,
``deepseek``, ``fireworks``, ``sambanova``).

What the stream does:

* text deltas are forwarded as they arrive. Reasoning deltas (``reasoning`` /
  ``reasoning_content``) are never spoken, and a leading ``<think>…</think>`` block in
  the text (a reasoning model served without a reasoning parser) is stripped;
* tool-call fragments are accumulated per ``index`` and emitted as complete
  :class:`~voice_agent_next.chat.FunctionCall` objects when the choice finishes;
* token usage comes from ``stream_options={"include_usage": True}``. Cached prompt
  tokens are read from ``prompt_tokens_details.cached_tokens`` or DeepSeek's
  ``prompt_cache_hit_tokens``;
* SDK errors are mapped to :mod:`voice_agent_next.errors` (401/403 ->
  ``AuthenticationError``, 429 -> ``RateLimitError``, network ->
  ``ProviderConnectionError``, timeouts -> ``ProviderTimeoutError``, other statuses ->
  ``ProviderError`` with ``status_code``).
"""

from __future__ import annotations

import asyncio
import inspect
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal, TypeAlias
from urllib.parse import urlsplit

from ...chat import ChatContext, FunctionCall
from ...errors import (
    AuthenticationError,
    ConfigurationError,
    ProviderConnectionError,
    ProviderError,
    ProviderTimeoutError,
    RateLimitError,
    VoiceAgentError,
)
from ...llm import LLM, ChatChunk, CompletionUsage, LLMCapabilities, LLMStream, ToolChoice
from ...registry import register_provider
from ...tools import FunctionTool
from ...utils.clock import now
from ...utils.deps import require
from ...utils.ids import new_id
from ...utils.log import logger
from ._format import to_chat_messages, to_chat_tools, to_tool_choice

__all__ = ["OpenAICompatibleLLM", "OpenAILLM"]

MaxTokensParam: TypeAlias = Literal["max_tokens", "max_completion_tokens"]
DeveloperRole: TypeAlias = Literal["developer", "system"]

_PLACEHOLDER_API_KEY = "no-key"  # the SDK needs a key; servers without auth ignore it

# ``chat.completions.create`` parameters, used when the SDK signature can't be inspected.
# Anything else in ``extra`` goes to the JSON body (``extra_body``).
_CREATE_PARAMS = frozenset(
    {
        "audio", "extra_body", "extra_headers", "extra_query", "frequency_penalty",
        "function_call", "functions", "logit_bias", "logprobs", "max_completion_tokens",
        "max_tokens", "messages", "metadata", "modalities", "model", "n",
        "parallel_tool_calls", "prediction", "presence_penalty", "prompt_cache_key",
        "reasoning_effort", "response_format", "safety_identifier", "seed", "service_tier",
        "stop", "store", "stream", "stream_options", "temperature", "timeout", "tool_choice",
        "tools", "top_logprobs", "top_p", "user", "verbosity", "web_search_options",
    }
)  # fmt: skip


@register_provider(
    "llm",
    "openai",
    description="OpenAI Chat Completions (streaming, tools); any compatible server via base_url",
    default_model="gpt-4.1-mini",
    models=(
        "gpt-4.1-mini",
        "gpt-4.1",
        "gpt-4.1-nano",
        "gpt-5.4-mini",
        "gpt-5.4-nano",
        "gpt-5.6-luna",
        "gpt-6-luna",
    ),
    env=("OPENAI_API_KEY",),
    extra="openai",
    requires=("openai",),
)
class OpenAILLM(LLM):
    """Streaming chat LLM for the OpenAI API and any OpenAI-compatible server.

    Args:
        model: model id. Defaults to ``DEFAULT_MODEL``; hosts without a default model use
            the first model listed by the server's ``GET /models``.
        api_key: API key. Defaults to the ``API_KEY_ENV`` environment variables. The
            ``OPENAI_API_KEY`` is only used for OpenAI itself: with a ``base_url`` that
            points elsewhere, pass ``api_key`` explicitly.
        base_url: API root including the version path (``https://host/v1``). Defaults to
            the ``BASE_URL_ENV`` environment variables, then ``DEFAULT_BASE_URL``.
        temperature: default sampling temperature (overridable per ``chat()``).
        max_tokens: default output token limit (overridable per ``chat()``).
        reasoning_effort: reasoning effort for reasoning models (``"none"``,
            ``"minimal"``, ``"low"``...). Keep it low for voice: reasoning delays the
            first spoken word.
        parallel_tool_calls: allow several tool calls per response (only sent with tools).
        extra: extra request parameters merged into every request (per-call ``extra``
            wins). Keys the SDK does not know are sent in the JSON body (``extra_body``),
            e.g. ``{"top_k": 20}``. A ``None`` value removes a default parameter.
        headers: extra HTTP headers sent with every request.
        capabilities: override the declared :class:`~voice_agent_next.llm.LLMCapabilities`
            (e.g. ``audio_input=True`` for audio-input models).
        max_tokens_param: request field for ``max_tokens``. Default:
            ``"max_completion_tokens"`` for OpenAI (required by its reasoning models),
            ``"max_tokens"`` for every other server.
        developer_role: role used for ``developer`` messages. Default: ``"developer"``
            for OpenAI, ``"system"`` for other servers.
        include_usage: request token usage at the end of the stream (``stream_options``).
        strip_thinking: drop a leading ``<think>…</think>`` block from the streamed text.
        timeout: HTTP timeout in seconds, applied to connecting and to each read (the
            wait for the next streamed chunk).
        max_retries: SDK retries for connection errors and 408/409/429/5xx responses.
            Retries happen before the first token only.
        keepalive_expiry: seconds an idle connection stays in the pool. Turns are often
            more than httpx's default 5 s apart, and every new connection costs a TLS
            handshake on the first token (and would make :meth:`warmup` pointless).
        client: a pre-built ``openai.AsyncOpenAI``-compatible client (for example
            ``openai.AsyncAzureOpenAI``). ``api_key``, ``base_url``, ``headers``,
            ``timeout``, ``max_retries``, ``keepalive_expiry`` and ``http_client`` are
            then ignored, and the client is not closed by :meth:`aclose`.
        http_client: an ``httpx.AsyncClient`` for the SDK (proxies, custom transports,
            tests). ``keepalive_expiry`` is then ignored, and it is not closed by
            :meth:`aclose`.
    """

    provider = "openai"

    DEFAULT_MODEL: ClassVar[str | None] = "gpt-4.1-mini"
    """Model used when none is given; ``None`` = ask the server (``GET /models``)."""
    DEFAULT_BASE_URL: ClassVar[str] = "https://api.openai.com/v1"
    BASE_URL_ENV: ClassVar[tuple[str, ...]] = ("OPENAI_BASE_URL",)
    """Environment variables that override ``DEFAULT_BASE_URL``."""
    API_KEY_ENV: ClassVar[tuple[str, ...]] = ("OPENAI_API_KEY",)
    """Environment variables holding the API key (the first one set wins)."""
    DEFAULT_EXTRA: ClassVar[Mapping[str, Any]] = {}
    """Request parameters this host sends by default (see ``extra``)."""
    PRELOAD_ON_WARMUP: ClassVar[bool] = False
    """:meth:`warmup` also runs a 1-token completion (servers that load models on demand)."""
    NOT_FOUND_HINT: ClassVar[str] = "check the model id and base_url"
    """Appended to HTTP 404 errors; ``{model}`` is replaced by the model id."""

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
        parallel_tool_calls: bool | None = None,
        extra: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        capabilities: LLMCapabilities | None = None,
        max_tokens_param: MaxTokensParam | None = None,
        developer_role: DeveloperRole | None = None,
        include_usage: bool = True,
        strip_thinking: bool = True,
        timeout: float = 60.0,
        max_retries: int = 1,
        keepalive_expiry: float = 120.0,
        client: Any = None,
        http_client: Any = None,
    ) -> None:
        self._openai = require("openai", extra="openai")
        if client is not None:
            resolved_url = str(client.base_url)
        else:
            resolved_url = base_url or self._base_url_from_env() or self.DEFAULT_BASE_URL
        self.base_url = resolved_url.rstrip("/")
        official = _is_openai_platform(self.base_url)
        resolved_model = model or self.DEFAULT_MODEL
        self._discover_model = resolved_model is None
        super().__init__(
            model=resolved_model or "auto",
            capabilities=capabilities or LLMCapabilities(image_input=official),
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self._developer_role: DeveloperRole = developer_role or (
            "developer" if official else "system"
        )
        self._max_tokens_param: MaxTokensParam = max_tokens_param or (
            "max_completion_tokens" if official else "max_tokens"
        )
        self._extra: dict[str, Any] = dict(self.DEFAULT_EXTRA)
        if reasoning_effort is not None:
            self._extra["reasoning_effort"] = reasoning_effort
        if parallel_tool_calls is not None:
            self._extra["parallel_tool_calls"] = parallel_tool_calls
        self._extra.update(extra or {})
        self.include_usage = include_usage
        self.strip_thinking = strip_thinking
        self._create_params: frozenset[str] | None = None
        self._model_lock = asyncio.Lock()
        if client is not None:
            self._client = client
            self._owns_client = False
            return
        key = api_key or self._default_api_key(custom_base_url=base_url is not None)
        if not key:
            if self._api_key_required():
                raise ConfigurationError(
                    f"{type(self).__name__} needs an API key: pass api_key=... or set "
                    f"{' or '.join(self.API_KEY_ENV)}"
                )
            key = _PLACEHOLDER_API_KEY
        self._owns_client = http_client is None
        if http_client is None:
            # the SDK's own defaults, with a longer keep-alive (Limits comes from the
            # HTTP library the installed SDK uses: httpx or httpx2)
            base = self._openai.DEFAULT_CONNECTION_LIMITS
            limits = type(base)(
                max_connections=base.max_connections,
                max_keepalive_connections=base.max_keepalive_connections,
                keepalive_expiry=keepalive_expiry,
            )
            http_client = self._openai.DefaultAsyncHttpxClient(limits=limits)
        self._client = self._openai.AsyncOpenAI(
            api_key=key,
            base_url=self.base_url,
            timeout=timeout,
            max_retries=max_retries,
            default_headers=dict(headers) if headers else None,
            http_client=http_client,
        )

    # ----------------------------------------------------------- configuration hooks
    def _base_url_from_env(self) -> str | None:
        return _first_env(self.BASE_URL_ENV)

    def _default_api_key(self, *, custom_base_url: bool) -> str | None:
        # never send the OpenAI key to another server
        if custom_base_url and urlsplit(self.base_url).hostname != "api.openai.com":
            return None
        return _first_env(self.API_KEY_ENV)

    def _api_key_required(self) -> bool:
        return _is_openai_platform(self.base_url)

    # --------------------------------------------------------------------- requests
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
        return _OpenAILLMStream(
            self,
            ctx,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            max_tokens=max_tokens,
            extra=extra,
        )

    def build_request(
        self,
        ctx: ChatContext,
        *,
        model: str | None = None,
        tools: list[FunctionTool] | None = None,
        tool_choice: ToolChoice | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """The keyword arguments passed to ``client.chat.completions.create()``."""
        params: dict[str, Any] = {
            "model": model or self.model,
            "messages": to_chat_messages(
                ctx,
                developer_role=self._developer_role,
                audio_input=self.capabilities.audio_input,
            ),
            "stream": True,
        }
        if self.include_usage:
            params["stream_options"] = {"include_usage": True}
        if tools:
            params["tools"] = to_chat_tools(tools)
            choice = to_tool_choice(tool_choice)
            if choice is not None:
                params["tool_choice"] = choice
        if temperature is not None:
            params["temperature"] = temperature
        if max_tokens is not None:
            params[self._max_tokens_param] = max_tokens
        known = self._sdk_params()
        body: dict[str, Any] = {}
        for key, value in {**self._extra, **(extra or {})}.items():
            if value is None:
                params.pop(key, None)
            elif key in known:
                params[key] = value
            else:
                body[key] = value
        if not tools:  # only valid alongside tools
            for key in ("tool_choice", "parallel_tool_calls"):
                params.pop(key, None)
                body.pop(key, None)
        if body:
            params["extra_body"] = {**(params.get("extra_body") or {}), **body}
        return params

    def _sdk_params(self) -> frozenset[str]:
        if self._create_params is None:
            try:
                sig = inspect.signature(self._client.chat.completions.create)
                self._create_params = frozenset(sig.parameters)
            except (AttributeError, TypeError, ValueError):
                self._create_params = _CREATE_PARAMS
        return self._create_params

    async def _ensure_model(self) -> str:
        """Resolve the model from ``GET /models`` when none was configured."""
        if not self._discover_model:
            return self.model
        async with self._model_lock:
            if self._discover_model:
                try:
                    page = await self._client.models.list()
                except Exception as exc:
                    status = getattr(exc, "status_code", None)
                    if isinstance(exc, self._openai.APIStatusError) and status in (404, 405):
                        raise ConfigurationError(
                            f"{self.provider}: cannot list the models served at "
                            f"{self.base_url} (GET /models: HTTP {status}); pass model=..."
                        ) from exc
                    raise self._map_error(exc) from exc
                ids = [str(m.id) for m in getattr(page, "data", None) or []]
                chat_ids = [i for i in ids if "embed" not in i.lower()] or ids
                if not chat_ids:
                    raise ConfigurationError(
                        f"{self.provider}: the server at {self.base_url} lists no models; "
                        "load one or pass model=..."
                    )
                self.model = chat_ids[0]
                self._discover_model = False
                logger.info("%s: using model %r from %s", self.provider, self.model, self.base_url)
        return self.model

    # -------------------------------------------------------------------- lifecycle
    async def warmup(self) -> None:
        """Open the HTTP connection (``GET /models``) ahead of the first request.

        Servers that load models on demand (``PRELOAD_ON_WARMUP``: Ollama, LM Studio)
        also get a 1-token completion so the model is in memory for the first turn.
        Failures are logged, not raised.
        """
        try:
            if self._discover_model:
                await self._ensure_model()
            else:
                try:
                    await self._client.models.list()
                except self._openai.APIStatusError as exc:
                    if exc.status_code in (401, 403):
                        raise
                    # the server doesn't list models, but the connection is open now
                    logger.debug("%s: GET /models failed: %s", self.provider, exc)
            if self.PRELOAD_ON_WARMUP:
                await self._client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": "Hi"}],
                    max_tokens=1,
                )
        except Exception as exc:
            logger.warning("%s warmup failed: %s", self.provider, self._map_error(exc))

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.close()

    # ------------------------------------------------------------------------ errors
    def _map_error(self, exc: BaseException) -> Exception:
        """Translate an SDK exception to :mod:`voice_agent_next.errors`."""
        if isinstance(exc, VoiceAgentError):
            return exc
        oa = self._openai
        who = f"{self.provider} ({self.model})"
        provider = self.provider
        if isinstance(exc, oa.APITimeoutError):
            return ProviderTimeoutError(
                f"{who}: request to {self.base_url} timed out", provider=provider
            )
        if isinstance(exc, oa.APIConnectionError):
            cause = exc.__cause__ or exc
            return ProviderConnectionError(
                f"{who}: cannot reach {self.base_url}: {cause}", provider=provider
            )
        if isinstance(exc, oa.APIStatusError):
            status = int(exc.status_code)
            message = f"{who}: HTTP {status}: {_error_detail(exc)}"
            if status in (401, 403):
                return AuthenticationError(message, provider=provider, status_code=status)
            if status == 429:
                # an exhausted quota does not recover by retrying
                retryable = getattr(exc, "code", None) != "insufficient_quota"
                return RateLimitError(
                    message, provider=provider, status_code=status, retryable=retryable
                )
            if status == 408:
                return ProviderTimeoutError(message, provider=provider, status_code=status)
            if status == 404:
                hint = self.NOT_FOUND_HINT.format(model=self.model)
                return ProviderError(f"{message} ({hint})", provider=provider, status_code=404)
            return ProviderError(
                message,
                provider=provider,
                status_code=status,
                retryable=status >= 500 or status == 409,
            )
        if isinstance(exc, oa.APIError):  # an error event inside the stream
            kind = f"{getattr(exc, 'type', '') or ''} {getattr(exc, 'code', '') or ''}".lower()
            message = f"{who}: stream error: {_error_detail(exc)}"
            if "rate" in kind:
                return RateLimitError(message, provider=provider)
            retryable = any(k in kind for k in ("server", "overload", "unavailable", "timeout"))
            return ProviderError(message, provider=provider, retryable=retryable)
        return ProviderError(f"{who}: {type(exc).__name__}: {exc}", provider=provider)


class OpenAICompatibleLLM(OpenAILLM):
    """Base class for preconfigured OpenAI-compatible hosts.

    Subclasses set ``provider`` and the class attributes below, and register themselves
    with ``@register_provider("llm", "<name>", ...)``. Unlike :class:`OpenAILLM`, the
    API key is always read from ``API_KEY_ENV`` (also with a custom ``base_url``, e.g.
    a proxy), and it is required unless ``API_KEY_REQUIRED`` is False (local servers).
    """

    provider = "openai_compatible"
    DEFAULT_MODEL: ClassVar[str | None] = None
    BASE_URL_ENV: ClassVar[tuple[str, ...]] = ()
    API_KEY_ENV: ClassVar[tuple[str, ...]] = ()
    API_KEY_REQUIRED: ClassVar[bool] = True
    """False for local servers, which accept requests without a key."""

    def _default_api_key(self, *, custom_base_url: bool) -> str | None:
        return _first_env(self.API_KEY_ENV)

    def _api_key_required(self) -> bool:
        return self.API_KEY_REQUIRED


# ------------------------------------------------------------------------------ stream
class _OpenAILLMStream(LLMStream):
    async def _run(self) -> None:
        llm: OpenAILLM = self._llm  # type: ignore[assignment]
        model = await llm._ensure_model()
        request = llm.build_request(
            self.ctx,
            model=model,
            tools=self.tools,
            tool_choice=self.tool_choice,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            extra=self.extra,
        )
        try:
            stream = await llm._client.chat.completions.create(**request)
        except Exception as exc:
            raise llm._map_error(exc) from exc
        parser = _StreamParser(strip_thinking=llm.strip_thinking)
        try:
            async for chunk in stream:
                text, calls = parser.feed(chunk)
                if self._first_token is None and parser.tool_call_started:
                    # tool calls are emitted once complete; time the first token at the
                    # first fragment so ttft / tokens_per_second reflect the model
                    self._first_token = now()
                self._forward(text, calls)
        except Exception as exc:
            raise llm._map_error(exc) from exc
        finally:
            try:
                await stream.close()  # releases the connection (also on cancellation)
            except Exception:
                logger.debug("closing the %s stream failed", llm.provider, exc_info=True)
        self._forward(*parser.finish())
        self._push(
            ChatChunk(self.request_id, usage=parser.usage, finish_reason=parser.finish_reason)
        )

    def _forward(self, text: str, calls: list[FunctionCall]) -> None:
        if text:
            self._push(ChatChunk(self.request_id, delta=text))
        if calls:
            self._push(ChatChunk(self.request_id, tool_calls=calls))


class _StreamParser:
    """Turns ``chat.completion.chunk`` objects into text, complete tool calls and usage."""

    def __init__(self, *, strip_thinking: bool = True) -> None:
        self._think = _ThinkStripper() if strip_thinking else None
        self._tools = _ToolCallBuffer()
        self.usage: CompletionUsage | None = None
        self.finish_reason: str | None = None
        self.tool_call_started = False

    def feed(self, chunk: Any) -> tuple[str, list[FunctionCall]]:
        usage = _field(chunk, "usage")
        if usage is None:
            usage = _field(_field(chunk, "x_groq"), "usage")  # Groq's legacy location
        if usage is not None:
            self.usage = _to_usage(usage)
        text: list[str] = []
        calls: list[FunctionCall] = []
        for choice in _field(chunk, "choices") or ():
            delta = _field(choice, "delta")
            for key in ("content", "refusal"):  # a refusal is what the user should hear
                piece = _field(delta, key)
                if isinstance(piece, str) and piece:
                    text.append(self._think.push(piece) if self._think else piece)
            for tc in _field(delta, "tool_calls") or ():
                fn = _field(tc, "function")
                self.tool_call_started = True
                self._tools.add(
                    _field(tc, "index"),
                    _field(tc, "id"),
                    _field(fn, "name"),
                    _field(fn, "arguments"),
                )
            reason = _field(choice, "finish_reason")
            if reason:
                self.finish_reason = str(reason)
                calls.extend(self._tools.take())
        return "".join(text), calls

    def finish(self) -> tuple[str, list[FunctionCall]]:
        """Flush what is left when the stream ends (servers that omit ``finish_reason``)."""
        tail = self._think.flush() if self._think else ""
        return tail, self._tools.take()


@dataclass
class _ToolCallDraft:
    call_id: str | None
    name: str = ""
    arguments: list[str] = field(default_factory=list)


class _ToolCallBuffer:
    """Accumulates streamed tool-call fragments (keyed by ``index``, then by ``id``)."""

    def __init__(self) -> None:
        self._drafts: list[_ToolCallDraft] = []
        self._by_index: dict[int, _ToolCallDraft] = {}

    def add(
        self, index: int | None, call_id: str | None, name: str | None, arguments: str | None
    ) -> None:
        if index is not None:
            draft = self._by_index.get(index)
        else:  # servers that omit the index continue the last call
            draft = self._drafts[-1] if self._drafts else None
        if draft is not None and call_id and draft.call_id and call_id != draft.call_id:
            draft = None  # a different call (index reused or missing)
        if draft is None:
            draft = _ToolCallDraft(call_id=call_id or None)
            self._drafts.append(draft)
            if index is not None:
                self._by_index[index] = draft
        elif call_id and not draft.call_id:
            draft.call_id = call_id
        if name and not draft.name:
            draft.name = name
        if arguments:
            draft.arguments.append(arguments)

    def take(self) -> list[FunctionCall]:
        calls: list[FunctionCall] = []
        for draft in self._drafts:
            if not draft.name:
                logger.warning("dropping a streamed tool call without a function name")
                continue
            calls.append(
                FunctionCall(
                    name=draft.name,
                    arguments="".join(draft.arguments).strip() or "{}",
                    call_id=draft.call_id or new_id("call_"),
                )
            )
        self._drafts.clear()
        self._by_index.clear()
        return calls


class _ThinkStripper:
    """Removes a leading ``<think>…</think>`` block from streamed text.

    Reasoning models served without a reasoning parser (or with thinking disabled on a
    model that always thinks) put their reasoning in the content; it must not be spoken.
    Only a block at the very start is removed, so ordinary text is passed through as soon
    as it cannot be the opening tag.
    """

    OPEN = "<think>"
    CLOSE = "</think>"

    def __init__(self) -> None:
        self._state: Literal["start", "thinking", "after", "text"] = "start"
        self._buf = ""

    def push(self, delta: str) -> str:
        if self._state == "text":
            return delta
        self._buf += delta
        if self._state == "start":
            head = self._buf.lstrip()
            if not head or (len(head) < len(self.OPEN) and self.OPEN.startswith(head)):
                return ""  # may still become "<think>"
            if not head.startswith(self.OPEN):
                self._state = "text"
                out, self._buf = self._buf, ""
                return out
            self._state = "thinking"
            self._buf = head[len(self.OPEN) :]
        if self._state == "thinking":
            end = self._buf.find(self.CLOSE)
            if end < 0:
                self._buf = self._buf[-(len(self.CLOSE) - 1) :]  # a split closing tag
                return ""
            self._buf = self._buf[end + len(self.CLOSE) :]
            self._state = "after"
        rest = self._buf.lstrip()  # "after": drop the whitespace following the block
        if not rest:
            self._buf = ""
            return ""
        self._state = "text"
        self._buf = ""
        return rest

    def flush(self) -> str:
        out = self._buf if self._state == "start" else ""
        self._buf = ""
        return out


# ----------------------------------------------------------------------------- helpers
def _field(obj: Any, name: str) -> Any:
    """Attribute or key access that tolerates ``None`` and missing fields."""
    if obj is None:
        return None
    if isinstance(obj, Mapping):
        return obj.get(name)
    return getattr(obj, name, None)


def _as_int(value: Any) -> int:
    return int(value) if isinstance(value, (int, float)) else 0


def _to_usage(usage: Any) -> CompletionUsage:
    cached = _as_int(_field(_field(usage, "prompt_tokens_details"), "cached_tokens"))
    if not cached:
        cached = _as_int(_field(usage, "prompt_cache_hit_tokens"))  # DeepSeek
    return CompletionUsage(
        prompt_tokens=_as_int(_field(usage, "prompt_tokens")),
        completion_tokens=_as_int(_field(usage, "completion_tokens")),
        cached_tokens=cached,
    )


def _error_detail(exc: Any) -> str:
    body = getattr(exc, "body", None)
    if isinstance(body, Mapping):
        detail = body.get("message") or body.get("error") or body.get("detail")
        if detail:
            return str(detail)
    elif isinstance(body, str) and body.strip():
        return body.strip()
    return str(getattr(exc, "message", None) or exc)


def _first_env(names: Iterable[str]) -> str | None:
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return None


def _is_openai_platform(url: str) -> bool:
    """True for OpenAI's own API (and Azure OpenAI), which accept the newest parameters."""
    host = (urlsplit(url).hostname or "").lower()
    return host == "api.openai.com" or host.endswith(".openai.azure.com")
