"""Provider registry: resolve ``"provider/model"`` strings to component instances.

Providers register themselves with :func:`register_provider` when their module is
imported. Resolution is lazy and convention-based — no central list to edit:

* ``create("stt", "deepgram/nova-3")`` imports ``voice_agent_next.providers.deepgram``
  (``-`` becomes ``_``), which registers ``("stt", "deepgram")``, then instantiates it
  with ``model="nova-3"``;
* third-party packages can expose providers through the ``voice_agent_next.providers``
  entry-point group (the entry point just needs to import the registering module).

Provider modules **must be importable without their optional dependencies** (import
heavy packages lazily with :func:`voice_agent_next.utils.require`), so that
:func:`list_providers` can enumerate everything.

Spec formats accepted by :func:`create`:

* ``"deepgram"`` — provider default model;
* ``"deepgram/nova-3"`` — explicit model (only the first ``/`` separates; model ids
  may contain ``/`` and ``:``, e.g. ``"together/meta-llama/Llama-3.3-70B"``);
* ``{"provider": "deepgram/nova-3", "language": "en"}`` or
  ``{"provider": "deepgram", "model": "nova-3", ...}`` — extra keys become kwargs;
* an already-built component instance (returned unchanged).
"""

from __future__ import annotations

import importlib
import importlib.metadata
import os
import pkgutil
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, TypeVar

from .errors import ConfigurationError, MissingDependencyError, ProviderNotFoundError
from .utils.deps import is_installed
from .utils.log import logger

__all__ = [
    "ComponentKind",
    "ProviderSpec",
    "create",
    "get_provider",
    "list_providers",
    "parse_spec",
    "register_alias",
    "register_provider",
]

ComponentKind = Literal["stt", "tts", "llm", "vad", "turn", "engine"]
KINDS: tuple[ComponentKind, ...] = ("stt", "tts", "llm", "vad", "turn", "engine")
ENTRY_POINT_GROUP = "voice_agent_next.providers"
PROVIDERS_PACKAGE = "voice_agent_next.providers"

T = TypeVar("T")


@dataclass(frozen=True)
class ProviderSpec:
    """Metadata describing one registered provider implementation."""

    kind: ComponentKind
    name: str
    factory: Callable[..., Any]
    description: str = ""
    default_model: str | None = None
    models: tuple[str, ...] = ()
    """Known/suggested model ids (not exhaustive)."""
    env: tuple[str, ...] = ()
    """Environment variables holding credentials (any one of them suffices)."""
    extra: str | None = None
    """``pip install 'voice-agent-next[<extra>]'`` installs the dependencies."""
    requires: tuple[str, ...] = ()
    """Importable module names needed at runtime (checked by :meth:`missing_dependencies`)."""
    local: bool = False
    """Runs on this machine (no cloud API)."""
    platforms: tuple[str, ...] = ("linux", "darwin", "win32")
    module: str = ""
    aliases: tuple[str, ...] = field(default=())

    def missing_dependencies(self) -> list[str]:
        return [m for m in self.requires if not is_installed(m)]

    def missing_env(self) -> list[str]:
        if not self.env:
            return []
        return [] if any(os.environ.get(e) for e in self.env) else list(self.env)

    def supports_platform(self, platform: str | None = None) -> bool:
        plat = platform or sys.platform
        return any(plat.startswith(p) for p in self.platforms)

    @property
    def available(self) -> bool:
        return not self.missing_dependencies() and self.supports_platform()


_REGISTRY: dict[tuple[ComponentKind, str], ProviderSpec] = {}
_ALIASES: dict[str, str] = {}
_ENTRY_POINTS_LOADED = False


def _normalize(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(".", "_")


def register_alias(alias: str, provider: str) -> None:
    """Make ``alias`` resolve to ``provider`` for every component kind."""
    _ALIASES[_normalize(alias)] = _normalize(provider)


def register_provider(
    kind: ComponentKind,
    name: str,
    *,
    description: str = "",
    default_model: str | None = None,
    models: tuple[str, ...] | list[str] = (),
    env: tuple[str, ...] | list[str] = (),
    extra: str | None = None,
    requires: tuple[str, ...] | list[str] = (),
    local: bool = False,
    platforms: tuple[str, ...] | list[str] = ("linux", "darwin", "win32"),
    aliases: tuple[str, ...] | list[str] = (),
) -> Callable[[T], T]:
    """Class/function decorator registering a component factory.

    The factory is called as ``factory(model=..., **kwargs)`` (``model`` only when the
    spec or ``default_model`` provides one).
    """
    if kind not in KINDS:
        raise ValueError(f"unknown component kind {kind!r}; expected one of {KINDS}")

    def decorator(factory: T) -> T:
        key = (kind, _normalize(name))
        spec = ProviderSpec(
            kind=kind,
            name=_normalize(name),
            factory=factory,  # type: ignore[arg-type]
            description=description
            or (getattr(factory, "__doc__", "") or "").strip().split("\n")[0],
            default_model=default_model,
            models=tuple(models),
            env=tuple(env),
            extra=extra,
            requires=tuple(requires),
            local=local,
            platforms=tuple(platforms),
            module=getattr(factory, "__module__", ""),
            aliases=tuple(_normalize(a) for a in aliases),
        )
        if key in _REGISTRY and _REGISTRY[key].factory is not factory:
            logger.debug("provider %s/%s re-registered by %s", kind, name, spec.module)
        _REGISTRY[key] = spec
        for alias in spec.aliases:
            _ALIASES.setdefault(alias, spec.name)
        return factory

    return decorator


def parse_spec(spec: str) -> tuple[str, str | None]:
    """``"deepgram/nova-3"`` -> ``("deepgram", "nova-3")``; ``"silero"`` -> ``("silero", None)``."""
    spec = spec.strip()
    if not spec:
        raise ConfigurationError("empty provider spec")
    provider, sep, model = spec.partition("/")
    return _normalize(provider), (model if sep and model else None)


def _load_entry_points() -> None:
    global _ENTRY_POINTS_LOADED
    if _ENTRY_POINTS_LOADED:
        return
    _ENTRY_POINTS_LOADED = True
    try:
        eps = importlib.metadata.entry_points(group=ENTRY_POINT_GROUP)
    except Exception:  # pragma: no cover - broken metadata
        return
    for ep in eps:
        try:
            ep.load()
        except Exception:
            logger.warning("failed to load provider entry point %s", ep.name, exc_info=True)


def _import_provider_module(name: str) -> None:
    module = f"{PROVIDERS_PACKAGE}.{name}"
    try:
        importlib.import_module(module)
    except ModuleNotFoundError as exc:
        if exc.name in (module, PROVIDERS_PACKAGE) or (
            exc.name and module.startswith(exc.name + ".")
        ):
            return  # no such built-in provider module
        raise MissingDependencyError(f"provider module {module} failed to import: {exc}") from exc


def get_provider(kind: ComponentKind, name: str) -> ProviderSpec:
    """Look up a provider, importing its module on demand."""
    key_name = _normalize(name)
    key_name = _ALIASES.get(key_name, key_name)
    if (kind, key_name) not in _REGISTRY:
        _import_provider_module(key_name)
        key_name = _ALIASES.get(key_name, key_name)
    if (kind, key_name) not in _REGISTRY:
        _load_entry_points()
        key_name = _ALIASES.get(key_name, key_name)
    try:
        return _REGISTRY[(kind, key_name)]
    except KeyError:
        known = sorted(n for (k, n) in _REGISTRY if k == kind)
        raise ProviderNotFoundError(
            f"no {kind} provider named {name!r}. Registered {kind} providers: "
            f"{', '.join(known) or '(none loaded)'}. Run `van providers` to list all."
        ) from None


def create(kind: ComponentKind, spec: Any, **kwargs: Any) -> Any:
    """Instantiate a component from a spec string/mapping (instances pass through)."""
    if spec is None:
        raise ConfigurationError(f"no {kind} configured")
    if not isinstance(spec, (str, Mapping)):
        return spec  # already a component instance
    if isinstance(spec, Mapping):
        opts = dict(spec)
        target = opts.pop("provider", None) or opts.pop("use", None)
        if not target:
            raise ConfigurationError(f"{kind} config needs a 'provider' key: {dict(spec)!r}")
        name, model = parse_spec(str(target))
        model = opts.pop("model", None) or model
        kwargs = {**opts, **kwargs}
    else:
        name, model = parse_spec(spec)
    provider = get_provider(kind, name)
    model = model or provider.default_model
    if model is not None and "model" not in kwargs:
        kwargs["model"] = model
    return provider.factory(**kwargs)


def list_providers(
    kind: ComponentKind | None = None, *, load_all: bool = True
) -> list[ProviderSpec]:
    """All registered providers (importing every built-in provider module if ``load_all``)."""
    if load_all:
        pkg = importlib.import_module(PROVIDERS_PACKAGE)
        for info in pkgutil.iter_modules(pkg.__path__):
            if info.name.startswith("_"):
                continue
            try:
                importlib.import_module(f"{PROVIDERS_PACKAGE}.{info.name}")
            except Exception:
                logger.warning("failed to import provider module %s", info.name, exc_info=True)
        _load_entry_points()
    specs = [s for (k, _), s in _REGISTRY.items() if kind is None or k == kind]
    return sorted(specs, key=lambda s: (KINDS.index(s.kind), s.name))
