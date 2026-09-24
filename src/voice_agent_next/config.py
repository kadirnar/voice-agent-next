"""Declarative configuration (YAML / TOML / JSON) for agents and sessions.

Example ``agent.yaml``::

    # native speech-to-speech
    engine: openai/gpt-realtime

    # ...or a cascade (omit `engine`)
    stt: {provider: deepgram/nova-3, language: en}
    llm: [groq/llama-3.3-70b-versatile, openai/gpt-4.1-mini]   # a list = failover chain
    tts: {provider: cartesia/sonic-2, voice: "<voice id>"}
    vad: silero
    turn_detector: smart_turn

    agent:
      instructions: You are a friendly assistant.
      greeting: Hi! How can I help?
      tools: ["my_package.tools:get_weather"]

    session:
      allow_interruptions: true

    transport: {type: local}

Strings of the form ``${ENV_VAR}`` or ``${ENV_VAR:-default}`` are expanded.
"""

from __future__ import annotations

import importlib
import json
import os
import re
import tomllib
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from .errors import ConfigurationError

__all__ = [
    "AgentConfig",
    "AppConfig",
    "ComponentSpec",
    "load_config",
    "resolve_callable",
]

SingleComponentSpec = str | dict[str, Any]
ComponentSpec = SingleComponentSpec | list[SingleComponentSpec]
"""A provider spec, its mapping form, or (STT/LLM/TTS) a failover list of either."""
_ENV = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


class AgentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instructions: str | None = None
    greeting: str | None = None
    tools: list[str] = Field(default_factory=list)
    """Import paths of tools: ``"package.module:function"``."""
    voice: str | None = None
    language: str | None = None


class AppConfig(BaseModel):
    """Top-level configuration file schema."""

    model_config = ConfigDict(extra="forbid")

    engine: SingleComponentSpec | None = None
    stt: ComponentSpec | None = None
    llm: ComponentSpec | None = None
    tts: ComponentSpec | None = None
    vad: SingleComponentSpec | None = None
    turn_detector: SingleComponentSpec | None = None
    cascade: dict[str, Any] = Field(default_factory=dict)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    session: dict[str, Any] = Field(default_factory=dict)
    transport: dict[str, Any] = Field(default_factory=lambda: {"type": "local"})

    def is_cascade(self) -> bool:
        return self.engine is None

    def validate_components(self) -> None:
        if self.engine is None and (self.llm is None or self.tts is None):
            raise ConfigurationError(
                "config needs `engine:` or a cascade with at least `llm:` and `tts:`"
            )
        if self.engine is not None and any(
            v is not None for v in (self.stt, self.llm, self.tts, self.turn_detector)
        ):
            raise ConfigurationError("use either `engine:` or cascade components, not both")


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):

        def repl(m: re.Match[str]) -> str:
            name, default = m.group(1), m.group(2)
            env = os.environ.get(name)
            if env is None:
                if default is None:
                    raise ConfigurationError(f"environment variable {name} is not set")
                return default
            return env

        return _ENV.sub(repl, value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def load_config(source: str | os.PathLike[str] | dict[str, Any]) -> AppConfig:
    """Load and validate a config file (``.yaml``/``.yml``/``.toml``/``.json``) or dict."""
    if isinstance(source, dict):
        data = source
    else:
        path = Path(source)
        if not path.exists():
            raise ConfigurationError(f"config file not found: {path}")
        text = path.read_text(encoding="utf-8")
        suffix = path.suffix.lower()
        if suffix in (".yaml", ".yml"):
            data = yaml.safe_load(text) or {}
        elif suffix == ".toml":
            data = tomllib.loads(text)
        elif suffix == ".json":
            data = json.loads(text)
        else:
            raise ConfigurationError(f"unsupported config format: {suffix}")
    if not isinstance(data, dict):
        raise ConfigurationError("config root must be a mapping")
    cfg = AppConfig.model_validate(_expand_env(data))
    cfg.validate_components()
    return cfg


def resolve_callable(path: str) -> Any:
    """Import ``"package.module:attr"`` (or ``"package.module.attr"``)."""
    module_name, sep, attr = path.partition(":")
    if not sep:
        module_name, _, attr = path.rpartition(".")
    if not module_name or not attr:
        raise ConfigurationError(f"invalid import path: {path!r}")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ConfigurationError(f"cannot import {module_name!r}: {exc}") from exc
    try:
        return getattr(module, attr)
    except AttributeError:
        raise ConfigurationError(f"{module_name!r} has no attribute {attr!r}") from None
