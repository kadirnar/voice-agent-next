"""Build agents and sessions from an :class:`~voice_agent_next.config.AppConfig`."""

from __future__ import annotations

from .config import AppConfig, resolve_callable
from .engines.cascade import CascadeOptions
from .session import DEFAULT_INSTRUCTIONS, Agent, AgentSession, SessionOptions

__all__ = ["build_agent", "build_session"]


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
        cascade_options=CascadeOptions(**cfg.cascade),
        options=options,
    )
