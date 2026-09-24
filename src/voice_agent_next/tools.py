"""Function tools: turn Python callables into model-callable tools.

Example:
    >>> from typing import Annotated, Literal
    >>> @function_tool
    ... async def get_weather(
    ...     city: Annotated[str, "City name, e.g. 'Paris'"],
    ...     unit: Literal["celsius", "fahrenheit"] = "celsius",
    ... ) -> str:
    ...     \"\"\"Get the current weather for a city.\"\"\"
    ...     return f"Sunny in {city}"
    >>> get_weather.schema()["name"]
    'get_weather'

JSON schemas are generated with pydantic from the signature; argument validation and
coercion use the same pydantic model. Parameter descriptions come from
``Annotated[T, "description"]``, ``Annotated[T, Field(description=...)]`` or a
Google-style ``Args:`` docstring section.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import re
import sys
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import (
    Annotated,
    Any,
    Literal,
    TypeAlias,
    get_args,
    get_origin,
    get_type_hints,
    overload,
)

import pydantic

from .chat import FunctionCall, FunctionCallOutput
from .errors import ToolError
from .utils.log import logger

__all__ = [
    "DEFAULT_TOOL_ACK",
    "FillerSpec",
    "FunctionTool",
    "ToolContext",
    "ToolScheduling",
    "execute_function_call",
    "find_tool",
    "function_tool",
]


ToolScheduling: TypeAlias = Literal["interrupt", "when_idle", "silent"]
"""When the result of a non-blocking tool reaches the conversation (Gemini Live's
``FunctionResponse.scheduling``): ``interrupt`` stops what the agent is saying and answers
right away, ``when_idle`` waits until the agent has finished speaking and then answers,
``silent`` only adds the result to the context (the model uses it later)."""

FillerSpec: TypeAlias = (
    bool | str | Sequence[str] | Callable[[FunctionCall], str | None] | None
)
"""What a slow tool says while it runs (see :func:`function_tool`): ``None``/``True`` = the
session's default fillers, ``False`` = never, a phrase, a list of phrases (picked without
repeating) or a callable ``(call) -> phrase | None``."""

DEFAULT_TOOL_ACK = (
    "The task is running in the background. Its result will be added to the conversation "
    "when it is ready; do not wait for it and do not make up a result."
)
"""Immediate output of a non-blocking tool on engines without native asynchronous tools."""


@dataclass(slots=True)
class ToolContext:
    """Injected into tools that declare a parameter annotated as ``ToolContext``.

    Attributes:
        call: the function call being executed.
        session: the running :class:`~voice_agent_next.session.AgentSession` (if any).
        userdata: application-defined state shared across tool calls.
    """

    call: FunctionCall
    session: Any = None
    userdata: Any = None

    async def report_progress(
        self, message: str, *, speak: bool = True, to_model: bool = False
    ) -> bool:
        """Report progress of a long-running tool ("Found 3 flights, comparing prices").

        The session emits a ``tool_progress`` event; with ``speak`` it also says ``message``
        (unless the user or the agent is talking), and with ``to_model`` it adds it to the
        model's context without triggering a response. Spoken progress counts as the
        round's filler. Without a session this is a no-op.

        Returns:
            ``True`` if the message is being spoken.
        """
        report = getattr(self.session, "report_tool_progress", None)
        if report is None:
            return False
        return bool(await report(self.call, message, speak=speak, to_model=to_model))


@dataclass(eq=False)
class FunctionTool:
    """A tool the model can call.

    Can wrap a Python callable (sync or async) or be declared from a raw JSON schema
    (``fn=None``) when the application executes calls itself.
    """

    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[..., Any] | None = None
    strict: bool = False
    timeout: float | None = None
    _model: type[pydantic.BaseModel] | None = field(default=None, repr=False)
    _ctx_param: str | None = field(default=None, repr=False)
    filler: FillerSpec = None
    """Spoken by the session when a (blocking) call runs longer than
    ``SessionOptions.tool_filler_delay`` (see :data:`FillerSpec`)."""
    blocking: bool = True
    """``False``: the conversation goes on while the tool runs; its result is added when
    it arrives (see :data:`ToolScheduling`)."""
    scheduling: ToolScheduling = "when_idle"
    """How the result of a non-blocking call is delivered."""
    ack: str | None = None
    """Immediate output of a non-blocking call on engines without native asynchronous
    tools (``None`` = :data:`DEFAULT_TOOL_ACK`)."""

    def __post_init__(self) -> None:
        if self.scheduling not in ("interrupt", "when_idle", "silent"):
            raise ValueError(f"unknown tool scheduling {self.scheduling!r}")

    def schema(self) -> dict[str, Any]:
        """Provider-neutral schema (``type``/``name``/``description``/``parameters``)."""
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }

    def validate_arguments(self, arguments: str | dict[str, Any]) -> dict[str, Any]:
        """Parse + validate arguments, returning keyword arguments for the callable."""
        if isinstance(arguments, str):
            arguments = json.loads(arguments) if arguments.strip() else {}
        if not isinstance(arguments, dict):
            raise ToolError(f"arguments for {self.name} must be a JSON object")
        if self._model is None:
            return dict(arguments)
        try:
            model = self._model.model_validate(arguments)
        except pydantic.ValidationError as exc:
            raise ToolError(f"invalid arguments for {self.name}: {exc}") from exc
        return {name: getattr(model, name) for name in type(model).model_fields}

    async def __call__(
        self, arguments: str | dict[str, Any], ctx: ToolContext | None = None
    ) -> Any:
        if self.fn is None:
            raise ToolError(f"tool {self.name!r} has no Python implementation")
        kwargs = self.validate_arguments(arguments)
        if self._ctx_param is not None:
            kwargs[self._ctx_param] = ctx
        if inspect.iscoroutinefunction(self.fn):
            coro: Awaitable[Any] = self.fn(**kwargs)
        else:
            coro = asyncio.to_thread(self.fn, **kwargs)
        if self.timeout is not None:
            return await asyncio.wait_for(coro, self.timeout)
        return await coro


_ARGS_SECTION = re.compile(r"^\s*(Args|Arguments|Parameters)\s*:\s*$", re.IGNORECASE)
_SECTION = re.compile(r"^\s*[A-Z][A-Za-z ]+\s*:\s*$")
_PARAM_LINE = re.compile(r"^\s*(\*{0,2}\w+)\s*(\([^)]*\))?\s*:\s*(.*)$")


def _parse_docstring(doc: str | None) -> tuple[str, dict[str, str]]:
    """Return (summary/description, {param: description}) from a Google-style docstring."""
    if not doc:
        return "", {}
    lines = inspect.cleandoc(doc).splitlines()
    desc_lines: list[str] = []
    params: dict[str, str] = {}
    i = 0
    while i < len(lines) and not _ARGS_SECTION.match(lines[i]):
        if _SECTION.match(lines[i]) and desc_lines:
            break
        desc_lines.append(lines[i])
        i += 1
    if i < len(lines) and _ARGS_SECTION.match(lines[i]):
        i += 1
        current: str | None = None
        while i < len(lines):
            line = lines[i]
            if _SECTION.match(line) and not line.startswith((" ", "\t")):
                break
            m = _PARAM_LINE.match(line)
            if m and (line.startswith((" ", "\t")) or current is None):
                current = m.group(1).lstrip("*")
                params[current] = m.group(3).strip()
            elif current and line.strip():
                params[current] = (params[current] + " " + line.strip()).strip()
            i += 1
    return "\n".join(desc_lines).strip(), params


def _strip_titles(schema: Any) -> Any:
    if isinstance(schema, dict):
        return {k: _strip_titles(v) for k, v in schema.items() if k != "title"}
    if isinstance(schema, list):
        return [_strip_titles(v) for v in schema]
    return schema


def _is_tool_context(annotation: Any) -> bool:
    if annotation is ToolContext:
        return True
    return get_origin(annotation) is Annotated and get_args(annotation)[0] is ToolContext


def _build_tool(
    fn: Callable[..., Any],
    name: str | None,
    description: str | None,
    timeout: float | None,
    localns: dict[str, Any] | None = None,
    **behavior: Any,
) -> FunctionTool:
    sig = inspect.signature(fn)
    try:
        # localns = the decorator caller's scope, so locally defined types resolve too
        hints = get_type_hints(fn, localns=localns, include_extras=True)
    except Exception:  # unresolved forward refs -> fall back to raw annotations
        hints = {k: p.annotation for k, p in sig.parameters.items()}
    doc_desc, doc_params = _parse_docstring(fn.__doc__)
    fields: dict[str, Any] = {}
    ctx_param: str | None = None
    for index, (pname, param) in enumerate(sig.parameters.items()):
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        if index == 0 and pname in ("self", "cls"):
            continue  # method tools are bound by Agent
        annotation = hints.get(pname, param.annotation)
        if annotation is inspect.Parameter.empty:
            annotation = Any
        if _is_tool_context(annotation):
            ctx_param = pname
            continue
        field_info_kwargs: dict[str, Any] = {}
        if get_origin(annotation) is Annotated:
            base, *extras = get_args(annotation)
            for extra in extras:
                if isinstance(extra, str):
                    field_info_kwargs["description"] = extra
                elif isinstance(extra, pydantic.fields.FieldInfo) and extra.description:
                    field_info_kwargs["description"] = extra.description
            kept = [e for e in extras if not isinstance(e, str)]
            annotation = Annotated[(base, *kept)] if kept else base
        if "description" not in field_info_kwargs and pname in doc_params:
            field_info_kwargs["description"] = doc_params[pname]
        default = ... if param.default is inspect.Parameter.empty else param.default
        fields[pname] = (annotation, pydantic.Field(default, **field_info_kwargs))
    model = pydantic.create_model(f"{fn.__name__}_args", **fields)
    schema = _strip_titles(model.model_json_schema())
    schema.setdefault("properties", {})
    schema.setdefault("type", "object")
    return FunctionTool(
        name=name or fn.__name__,
        description=description if description is not None else doc_desc,
        parameters=schema,
        fn=fn,
        timeout=timeout,
        _model=model,
        _ctx_param=ctx_param,
        **behavior,
    )


@overload
def function_tool(fn: Callable[..., Any], /) -> FunctionTool: ...


@overload
def function_tool(
    *,
    name: str | None = None,
    description: str | None = None,
    timeout: float | None = None,
    filler: FillerSpec = None,
    blocking: bool = True,
    scheduling: ToolScheduling = "when_idle",
    ack: str | None = None,
) -> Callable[[Callable[..., Any]], FunctionTool]: ...


def function_tool(
    fn: Callable[..., Any] | None = None,
    /,
    *,
    name: str | None = None,
    description: str | None = None,
    timeout: float | None = None,
    filler: FillerSpec = None,
    blocking: bool = True,
    scheduling: ToolScheduling = "when_idle",
    ack: str | None = None,
) -> FunctionTool | Callable[[Callable[..., Any]], FunctionTool]:
    """Decorator turning a (sync or async) function into a :class:`FunctionTool`.

    Args:
        timeout: per-call timeout in seconds (the session's ``tool_timeout`` applies too).
        filler: what the session says when a call is slow (see :data:`FillerSpec`).
        blocking: ``False`` makes the tool non-blocking: the model is not kept waiting,
            the conversation goes on and the result is delivered when it is ready.
        scheduling: delivery of a non-blocking result (see :data:`ToolScheduling`).
        ack: immediate output of a non-blocking call on engines that need one.
    """
    behavior: dict[str, Any] = {
        "filler": filler, "blocking": blocking, "scheduling": scheduling, "ack": ack
    }  # fmt: skip
    if fn is not None:
        localns = dict(sys._getframe(1).f_locals)
        return _build_tool(fn, name, description, timeout, localns, **behavior)

    def decorator(f: Callable[..., Any]) -> FunctionTool:
        localns = dict(sys._getframe(1).f_locals)
        return _build_tool(f, name, description, timeout, localns, **behavior)

    return decorator


def find_tool(tools: Sequence[FunctionTool], name: str) -> FunctionTool | None:
    for t in tools:
        if t.name == name:
            return t
    return None


def _stringify(result: Any) -> str:
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    if isinstance(result, pydantic.BaseModel):
        return result.model_dump_json()
    try:
        return json.dumps(result, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(result)


async def execute_function_call(
    call: FunctionCall,
    tools: Sequence[FunctionTool],
    *,
    ctx: ToolContext | None = None,
    timeout: float | None = None,
) -> FunctionCallOutput:
    """Execute ``call`` against ``tools``; never raises (errors become error outputs)."""
    tool = find_tool(tools, call.name)
    if tool is None:
        return FunctionCallOutput(
            call_id=call.call_id, name=call.name, output=f"Unknown tool: {call.name}", is_error=True
        )
    try:
        coro = tool(call.arguments, ctx or ToolContext(call=call))
        result = await (asyncio.wait_for(coro, timeout) if timeout is not None else coro)
        return FunctionCallOutput(call_id=call.call_id, name=call.name, output=_stringify(result))
    except TimeoutError:
        return FunctionCallOutput(
            call_id=call.call_id,
            name=call.name,
            output=f"Tool {call.name} timed out",
            is_error=True,
        )
    except ToolError as exc:
        return FunctionCallOutput(
            call_id=call.call_id, name=call.name, output=str(exc), is_error=True
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.exception("tool %s failed", call.name)
        return FunctionCallOutput(
            call_id=call.call_id,
            name=call.name,
            output=f"Tool {call.name} failed: {type(exc).__name__}: {exc}",
            is_error=True,
        )
