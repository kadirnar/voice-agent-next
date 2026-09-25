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

A config may start from a preset (:mod:`voice_agent_next.presets`, ``van presets``) and
change only what differs::

    extends: local-cpu
    llm: ollama/qwen3.5:4b          # replaces the preset's LLM
    stt: {language: en}             # a mapping without `provider:` tweaks the preset's STT
    agent: {instructions: You are a pirate.}

Strings of the form ``${ENV_VAR}`` or ``${ENV_VAR:-default}`` are expanded (before
``extends:`` is resolved, so the preset name may come from the environment).

On the command line the layers are ``--preset`` < ``--config`` file < flags
(:func:`layer_config`), validated once after merging.
"""

from __future__ import annotations

import copy
import importlib
import json
import os
import re
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from .errors import ConfigurationError

__all__ = [
    "AgentConfig",
    "AppConfig",
    "ComponentSpec",
    "expand_env",
    "layer_config",
    "load_config",
    "merge_config",
    "read_config_file",
    "resolve_callable",
    "resolve_extends",
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
    extends: str | None = None
    """The preset this config starts from (``van presets`` lists them)."""

    def is_cascade(self) -> bool:
        return self.engine is None

    def validate_components(self) -> None:
        if self.engine is None and self.llm is None:  # `tts:` is checked by the cascade:
            raise ConfigurationError(  # optional for audio-output LLMs
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


def read_config_file(path: str | os.PathLike[str]) -> dict[str, Any]:
    """The raw mapping of a ``.yaml``/``.yml``/``.toml``/``.json`` config file: not
    expanded (``${ENV}``), not merged with its ``extends:`` preset and not validated."""
    path = Path(path)
    if not path.is_file():
        raise ConfigurationError(f"config file not found: {path}")
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    try:
        if suffix in (".yaml", ".yml"):
            data = yaml.safe_load(text) or {}
        elif suffix == ".toml":
            data = tomllib.loads(text)
        elif suffix == ".json":
            data = json.loads(text)
        else:
            raise ConfigurationError(f"unsupported config format: {suffix}")
    except (yaml.YAMLError, tomllib.TOMLDecodeError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"{path}: invalid {suffix[1:].upper()}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigurationError(f"{path}: config root must be a mapping")
    return data


def expand_env(data: Any) -> Any:
    """``data`` with ``${ENV_VAR}`` / ``${ENV_VAR:-default}`` in its strings expanded
    (once: an expanded value is not expanded again)."""
    return _expand_env(data)


def load_config(source: str | os.PathLike[str] | Mapping[str, Any]) -> AppConfig:
    """Load and validate a config file (``.yaml``/``.yml``/``.toml``/``.json``) or mapping.

    ``${ENV}`` strings are expanded first (so ``extends: ${PRESET}`` works), then the
    ``extends:`` preset is merged underneath, then the result is validated."""
    data = dict(source) if isinstance(source, Mapping) else read_config_file(source)
    data = resolve_extends(_expand_env(data))
    cfg = AppConfig.model_validate(data)
    cfg.validate_components()
    return cfg


def layer_config(
    *,
    preset: str | None = None,
    file: str | os.PathLike[str] | Mapping[str, Any] | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The raw config of ``preset`` < ``file`` < ``overrides``, as ``van run``/``serve``/
    ``bench`` combine ``--preset``, ``--config`` and flags. Not validated: the file may
    hold only tweaks (``stt: {language: en}``) that make sense on top of the preset.

    ``${ENV}`` in the file is expanded before its own ``extends:`` is read; that preset
    is used when ``preset`` is not given (both given and different is an error). The
    result has ``extends:`` set to the preset used, if any. See :func:`merge_config`
    for how two layers combine."""
    if file is None:
        data: dict[str, Any] = {}
    elif isinstance(file, Mapping):
        data = _expand_env(dict(file))
    else:
        data = _expand_env(read_config_file(file))
    extends = data.pop("extends", None)
    if extends is not None and not isinstance(extends, str):
        raise ConfigurationError(f"`extends:` must be a preset name, got {extends!r}")
    from .presets import get_preset  # the presets module imports this one

    name: str | None = None
    if preset is not None:
        name = get_preset(preset).name
        if extends is not None and get_preset(extends).name != name:
            where = f" in {file}" if not isinstance(file, Mapping) else ""
            raise ConfigurationError(f"preset {preset} conflicts with `extends: {extends}`{where}")
    elif extends is not None:
        name = get_preset(extends).name
    raw: dict[str, Any] = dict(get_preset(name).config) if name is not None else {}
    raw = merge_config(raw, data)
    if overrides:
        raw = merge_config(raw, dict(overrides))
    if name is not None:
        raw["extends"] = name
    return raw


_COMPONENT_KEYS = ("engine", "stt", "llm", "tts", "vad", "turn_detector")
_CASCADE_KEYS = ("stt", "llm", "tts", "vad", "turn_detector")


def resolve_extends(data: dict[str, Any]) -> dict[str, Any]:
    """A raw config mapping with its ``extends:`` preset merged underneath, as
    :func:`load_config` does it. Code that fills in defaults on a raw config ("no LLM: use
    the mock engine") must look at the resolved mapping: the preset may provide them."""
    name = data.get("extends")
    if name is None:
        return data
    if not isinstance(name, str):
        raise ConfigurationError(f"`extends:` must be a preset name, got {name!r}")
    from .presets import get_preset  # the presets module imports this one

    preset = get_preset(name)
    merged = merge_config(dict(preset.config), {k: v for k, v in data.items() if k != "extends"})
    merged["extends"] = preset.name
    return merged


def _merge_component(key: str, base: Any, override: Any) -> Any:
    """A component spec over another: a mapping without ``provider:`` only changes options."""
    if not isinstance(override, dict) or {"provider", "use", "fallback"} & override.keys():
        return override
    if base is None:
        raise ConfigurationError(f"`{key}:` needs a `provider:` key (there is nothing to extend)")
    if isinstance(base, str):
        return {"provider": base, **override}
    if isinstance(base, dict) and "fallback" not in base:
        return {**base, **override}
    raise ConfigurationError(
        f"`{key}:` is a failover list in the base config: give the whole list, not only options"
    )


def _merge_mapping(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge_mapping(out[key], value)
        else:
            out[key] = value
    return out


def merge_config(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """``override`` on top of ``base`` (both raw config mappings), as ``extends:`` does it.

    * Sections (``agent``, ``session``, ``cascade``, ``transport``) merge key by key; a
      ``transport`` of another ``type`` replaces the base one.
    * A component (``engine``, ``stt``, ``llm``, ``tts``, ``vad``, ``turn_detector``) is
      replaced, except that a mapping without ``provider:`` only changes the base
      component's options (``stt: {language: fr}``). ``null`` removes a component.
    * Setting ``engine:`` drops the base's cascade components, and setting a cascade
      component drops the base's ``engine:``: a config is one or the other.
    """
    out = copy.deepcopy(base)
    override = copy.deepcopy(override)
    if override.get("engine") is not None:
        for key in _CASCADE_KEYS:
            out.pop(key, None)
    if any(override.get(key) is not None for key in _CASCADE_KEYS):
        out.pop("engine", None)
    for key, value in override.items():
        current = out.get(key)
        if key in _COMPONENT_KEYS:
            out[key] = _merge_component(key, current, value)
        elif isinstance(value, dict) and isinstance(current, dict):
            same_type = value.get("type", current.get("type")) == current.get("type")
            out[key] = _merge_mapping(current, value) if key != "transport" or same_type else value
        else:
            out[key] = value
    return out


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
