"""Build agents and sessions from an :class:`~voice_agent_next.config.AppConfig`."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .config import AppConfig, resolve_callable
from .engines.cascade import CascadeOptions
from .errors import ConfigurationError
from .session import DEFAULT_INSTRUCTIONS, Agent, AgentSession, SessionOptions

__all__ = ["build_agent", "build_session", "cascade_options"]


def build_agent(cfg: AppConfig) -> Agent:
    tools = [resolve_callable(t) for t in cfg.agent.tools]
    return Agent(
        cfg.agent.instructions or DEFAULT_INSTRUCTIONS,
        tools=tools,
        greeting=cfg.agent.greeting,
        voice=cfg.agent.voice,
        language=cfg.agent.language,
    )


def build_session(cfg: AppConfig) -> AgentSession:
    options = SessionOptions(**cfg.session)
    if cfg.engine is not None:
        return AgentSession(cfg.engine, options=options)
    return AgentSession(
        stt=cfg.stt,
        llm=cfg.llm,
        tts=cfg.tts,
        vad=cfg.vad,
        turn_detector=cfg.turn_detector,
        cascade_options=cascade_options(cfg),
        options=options,
    )


def cascade_options(cfg: AppConfig) -> CascadeOptions:
    """The :class:`~voice_agent_next.engines.cascade.CascadeOptions` of the ``cascade:``
    section. ``text_filter`` may be an import path (``"package.module:function"``), a list
    of them (applied left to right) or ``null`` (no filter); without it the default
    (``tts_clean``) stays."""
    fields = dict(cfg.cascade)
    if "text_filter" in fields:
        fields["text_filter"] = _text_filter(fields["text_filter"])
    return CascadeOptions(**fields)


def _text_filter(value: Any) -> Callable[[str], str] | None:
    if value is None or callable(value):
        return value
    paths = [value] if isinstance(value, str) else value
    if not isinstance(paths, list) or not all(isinstance(p, str) for p in paths):
        raise ConfigurationError(
            f"cascade.text_filter must be an import path, a list of them or null: {value!r}"
        )
    filters: list[Callable[[str], str]] = []
    for path in paths:
        fn = resolve_callable(path)
        if not callable(fn):
            raise ConfigurationError(f"cascade.text_filter: {path!r} is not callable")
        filters.append(fn)
    if len(filters) <= 1:
        return filters[0] if filters else None

    def chained(text: str) -> str:
        for fn in filters:
            text = fn(text)
        return text

    return chained
