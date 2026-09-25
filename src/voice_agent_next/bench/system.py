"""The system under test: a native speech-to-speech engine or a cascade.

Built from the same pieces as ``van run``: registry specs (``"mock"``,
``"openai/gpt-realtime"``, ``{"provider": "mock", "response_delay": 0.3}``) and/or an
:class:`~voice_agent_next.config.AppConfig` file, so any engine or component that the
registry can create can be benchmarked.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import yaml

from ..app import build_agent
from ..config import AppConfig, ComponentSpec, expand_env, read_config_file, resolve_extends
from ..engine import S2SEngine
from ..engines.cascade import CascadeEngine, CascadeOptions
from ..errors import ConfigurationError
from ..registry import create
from ..session import Agent, SessionOptions

__all__ = ["BenchSystem", "parse_component_spec", "redact"]

_SECRET = re.compile(r"(api[_-]?key|token|secret|password|passwd|auth|credential)", re.I)


def parse_component_spec(text: str | None) -> ComponentSpec | None:
    """CLI helper: ``"mock"`` stays a string; ``"{provider: mock, response_delay: 0.3}"``
    (YAML/JSON flow mapping) becomes a dict."""
    if text is None:
        return None
    stripped = text.strip()
    if not stripped.startswith("{"):
        return stripped
    try:
        value = yaml.safe_load(stripped)
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"invalid inline component spec {text!r}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigurationError(f"inline component spec must be a mapping: {text!r}")
    return value


def redact(value: Any) -> Any:
    """Copy of ``value`` with credential-like mapping entries replaced by ``"***"``."""
    if isinstance(value, Mapping):
        return {
            str(k): "***" if _SECRET.search(str(k)) and v is not None else redact(v)
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact(v) for v in value]
    return value


def _spec_name(spec: ComponentSpec | None) -> str:
    if spec is None:
        return "none"
    if isinstance(spec, str):
        return spec
    if isinstance(spec, list):
        return "|".join(_spec_name(s) for s in spec)
    target = str(spec.get("provider") or spec.get("use") or "?")
    model = spec.get("model")
    return f"{target}/{model}" if model and "/" not in target else target


def _describe_object(value: Any) -> str:
    """JSON stand-in for config values that are not data (callables, instances)."""
    name = getattr(value, "__qualname__", None) or type(value).__qualname__
    return f"<{getattr(value, '__module__', '?')}.{name}>"


@dataclass
class BenchSystem:
    """What is being benchmarked, plus a short ``label`` for reports and run ids."""

    config: AppConfig
    label: str

    @classmethod
    def from_options(
        cls,
        *,
        config: str | os.PathLike[str] | Mapping[str, Any] | AppConfig | None = None,
        preset: str | None = None,
        engine: ComponentSpec | None = None,
        stt: ComponentSpec | None = None,
        llm: ComponentSpec | None = None,
        tts: ComponentSpec | None = None,
        vad: ComponentSpec | None = None,
        turn_detector: ComponentSpec | None = None,
        label: str | None = None,
        default_engine: str | None = "mock",
    ) -> BenchSystem:
        """Combine a preset, a config file/mapping and explicit specs, in that order
        (``van run``'s preset < file < flags; the file's ``extends:`` is the preset when
        ``preset`` is not given). Validated once, after merging.

        The file may hold only agent/session settings when the engine comes from the
        arguments; without any engine or LLM configured (counting the preset's
        components), ``default_engine`` is used.
        """
        resolved = isinstance(config, AppConfig)  # its preset is already merged in
        if isinstance(config, AppConfig):
            data: dict[str, Any] = config.model_dump()
        elif isinstance(config, Mapping):
            data = expand_env(dict(config))
        elif config is not None:
            data = expand_env(read_config_file(config))
        else:
            data = {}
        extends = data.pop("extends", None)
        if preset is not None:
            from ..presets import get_preset

            name = get_preset(preset).name
            if extends is not None and get_preset(str(extends)).name != name:
                raise ConfigurationError(f"preset {preset} conflicts with `extends: {extends}`")
            extends, resolved = name, False
        cascade = {"stt": stt, "llm": llm, "tts": tts, "turn_detector": turn_detector}
        explicit_cascade = any(v is not None for v in cascade.values())
        if engine is not None and explicit_cascade:
            raise ConfigurationError("pass either an engine or cascade components, not both")
        if engine is not None:  # an explicit engine replaces a configured cascade
            data.update(engine=engine, stt=None, llm=None, tts=None, turn_detector=None)
        elif explicit_cascade:  # explicit components replace a configured engine
            data["engine"] = None
            data.update({k: v for k, v in cascade.items() if v is not None})
        if vad is not None:
            data["vad"] = vad
        if extends is not None and not resolved:
            # The preset goes under the file and the explicit specs (`stt: {language: fr}`
            # tweaks its STT) and is merged before the default engine is considered: the
            # preset's own engine/LLM must win over the mock (#128).
            data = resolve_extends({**data, "extends": extends})
            extends = data.pop("extends")
        if data.get("engine") is None and data.get("llm") is None and default_engine:
            data["engine"] = default_engine
        cfg = AppConfig.model_validate(data)  # validated once, after merging
        cfg.validate_components()
        cfg.extends = extends
        return cls(cfg, label or cls.default_label(cfg))

    @staticmethod
    def default_label(cfg: AppConfig) -> str:
        if cfg.engine is not None:
            return _spec_name(cfg.engine)
        parts = [_spec_name(s) for s in (cfg.stt, cfg.llm, cfg.tts) if s is not None]
        return "cascade:" + "+".join(parts)

    @property
    def kind(self) -> Literal["native", "cascade"]:
        return "native" if self.config.engine is not None else "cascade"

    def build_engine(self) -> S2SEngine:
        """A new engine instance (sessions open connections on it)."""
        cfg = self.config
        if cfg.engine is not None:
            engine: S2SEngine = create("engine", cfg.engine)
            return engine
        return CascadeEngine(
            stt=cfg.stt, llm=cfg.llm, tts=cfg.tts, vad=cfg.vad,
            turn_detector=cfg.turn_detector, options=CascadeOptions(**cfg.cascade),
        )  # fmt: skip

    def build_agent(self) -> Agent:
        return build_agent(self.config)

    def session_options(self) -> SessionOptions:
        return SessionOptions(**self.config.session)

    def describe(self, engine: S2SEngine | None = None) -> dict[str, Any]:
        """JSON-able description for the manifest (credentials redacted)."""
        cfg = self.config
        out: dict[str, Any] = {
            "label": self.label,
            "kind": self.kind,
            "config": redact(
                cfg.model_dump(
                    mode="json",
                    exclude={"transport"},
                    exclude_none=True,
                    fallback=_describe_object,  # e.g. a callable reply script
                )
            ),
        }
        if engine is not None:
            info: dict[str, Any] = {
                "class": f"{type(engine).__module__}.{type(engine).__qualname__}",
                "provider": engine.provider,
                "model": engine.model,
                "input_sample_rate": engine.input_sample_rate,
                "output_sample_rate": engine.output_sample_rate,
                "capabilities": {
                    k: getattr(engine.capabilities, k)
                    for k in engine.capabilities.__dataclass_fields__
                },
            }
            if isinstance(engine, CascadeEngine):
                info["components"] = {
                    name: {
                        "class": type(comp).__qualname__,
                        "provider": getattr(comp, "provider", None),
                        "model": getattr(comp, "model", None),
                    }
                    for name, comp in (
                        ("vad", engine.vad), ("stt", engine.stt),
                        ("turn_detector", engine.turn_detector), ("llm", engine.llm),
                        ("tts", engine.tts),
                    )
                    if comp is not None
                }  # fmt: skip
            out["engine"] = info
        return out
