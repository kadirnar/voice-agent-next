"""Provider plugins (entry points): loaded before being marked loaded, failures named and
retried (#141)."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import pytest

from voice_agent_next import registry
from voice_agent_next.errors import ProviderNotFoundError


class FakeEntryPoint:
    def __init__(self, name: str, load: Callable[[], Any]) -> None:
        self.name, self.value, self._load = name, f"fake_plugins.{name}:register", load

    def load(self) -> Any:
        return self._load()


@pytest.fixture
def plugins(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[FakeEntryPoint]]:
    eps: list[FakeEntryPoint] = []
    monkeypatch.setattr(registry.importlib.metadata, "entry_points", lambda group: list(eps))
    monkeypatch.setattr(registry, "_ENTRY_POINTS_LOADED", False)
    monkeypatch.setattr(registry, "_EP_DONE", set())
    monkeypatch.setattr(registry, "_EP_ACTIVE", set())
    monkeypatch.setattr(registry, "_EP_ERRORS", {})
    real = dict(registry._REGISTRY)
    yield eps
    for key in list(registry._REGISTRY):  # drop the fake plugins' providers
        if key not in real:
            del registry._REGISTRY[key]


def _register(name: str) -> Callable[[], Any]:
    def load() -> Any:
        @registry.register_provider("tts", name)
        def factory(**kwargs: Any) -> Any:
            return None

        return factory

    return load


def test_failing_plugin_is_named_and_retried(plugins: list[FakeEntryPoint]) -> None:
    state = {"broken": True}

    def flaky() -> Any:
        if state["broken"]:
            raise ImportError("needs the 'acme' extra")
        return _register("acme_tts")()

    plugins.append(FakeEntryPoint("good-plugin", _register("good_tts")))
    plugins.append(FakeEntryPoint("acme-plugin", flaky))
    assert registry.get_provider("tts", "good_tts").name == "good_tts"
    with pytest.raises(ProviderNotFoundError) as info:
        registry.get_provider("tts", "acme_tts")
    message = str(info.value)
    assert "acme-plugin" in message and "needs the 'acme' extra" in message
    assert not registry._ENTRY_POINTS_LOADED  # a failed plugin is not "loaded"

    state["broken"] = False  # e.g. the missing dependency was installed
    assert registry.get_provider("tts", "acme_tts").name == "acme_tts"
    assert registry._ENTRY_POINTS_LOADED and registry._EP_ERRORS == {}


def test_plugin_loading_is_reentrant(plugins: list[FakeEntryPoint]) -> None:
    """A plugin that looks up another plugin's provider while it loads finds it: the
    group is not marked loaded before its members are."""
    found: list[str] = []

    def dependent() -> Any:
        found.append(registry.get_provider("tts", "base_tts").name)
        return _register("dependent_tts")()

    plugins.append(FakeEntryPoint("dependent-plugin", dependent))
    plugins.append(FakeEntryPoint("base-plugin", _register("base_tts")))
    assert registry.get_provider("tts", "dependent_tts").name == "dependent_tts"
    assert found == ["base_tts"]
    assert registry._ENTRY_POINTS_LOADED


def test_loaded_plugins_are_not_loaded_again(plugins: list[FakeEntryPoint]) -> None:
    calls: list[str] = []

    def counted() -> Any:
        calls.append("x")
        return _register("once_tts")()

    plugins.append(FakeEntryPoint("once-plugin", counted))
    registry.get_provider("tts", "once_tts")
    with pytest.raises(ProviderNotFoundError):
        registry.get_provider("tts", "nothing_tts")
    assert calls == ["x"]
