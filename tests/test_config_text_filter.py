"""``cascade.text_filter`` in a config file: an import path, a list of them, or null (#167)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from voice_agent_next import build_session, load_config
from voice_agent_next.app import cascade_options
from voice_agent_next.bench.system import BenchSystem
from voice_agent_next.engines.cascade import CascadeEngine
from voice_agent_next.errors import ConfigurationError
from voice_agent_next.presets import get_preset
from voice_agent_next.server.realtime import engine_from_config
from voice_agent_next.server.serving import engine_from_app_config
from voice_agent_next.text.filters import strip_emoji, tts_clean

CASCADE = {"stt": "mock", "llm": "mock", "tts": "mock", "vad": "energy"}
NOT_CALLABLE = 42


def shout(text: str) -> str:
    return text.upper() + "!"


def tagged(text: str) -> str:
    return f"<{text}>"


def options(text_filter: Any) -> Any:
    return cascade_options(load_config({**CASCADE, "cascade": {"text_filter": text_filter}}))


def test_an_import_path_resolves() -> None:
    assert options("voice_agent_next.text.filters:tts_clean").text_filter is tts_clean
    assert options("voice_agent_next.text.filters.strip_emoji").text_filter is strip_emoji


def test_a_list_is_applied_left_to_right() -> None:
    fn = options([f"{__name__}:shout", f"{__name__}:tagged"]).text_filter
    assert fn is not None and fn("hi") == "<HI!>"
    fn = options([f"{__name__}:tagged", f"{__name__}:shout"]).text_filter
    assert fn is not None and fn("hi") == "<HI>!"


def test_null_disables_and_absent_keeps_the_default() -> None:
    assert options(None).text_filter is None and options([]).text_filter is None
    cfg = load_config({**CASCADE, "cascade": {"min_endpointing_delay": 0.3}})
    opts = cascade_options(cfg)
    assert opts.text_filter is tts_clean and opts.min_endpointing_delay == 0.3
    assert cfg.cascade == {"min_endpointing_delay": 0.3}  # the config is not modified


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("voice_agent_next.text.filters:nope", "has no attribute"),
        ("no_such_module_167:fn", "cannot import"),
        ("not a path", "invalid import path"),
        (f"{__name__}:NOT_CALLABLE", "is not callable"),
        (["voice_agent_next.text.filters:tts_clean", 3], "import path"),
        ({"fn": "x"}, "import path"),
    ],
)
def test_a_bad_value_is_a_configuration_error(value: Any, message: str) -> None:
    with pytest.raises(ConfigurationError, match=message):
        options(value)


def test_extends_layering_keeps_it(tmp_path: Path) -> None:
    path = tmp_path / "agent.yaml"
    path.write_text(
        f"extends: local-cpu\ncascade:\n  text_filter: [{__name__}:shout]\n", encoding="utf-8"
    )
    cfg = load_config(path)
    opts = cascade_options(cfg)
    assert opts.text_filter is shout
    for key, value in get_preset("local-cpu").config["cascade"].items():
        assert getattr(opts, key) == value  # the preset's cascade options are kept


def test_every_builder_uses_it(tmp_path: Path) -> None:
    path = tmp_path / "agent.toml"
    path.write_text(
        'stt = "mock"\nllm = "mock"\ntts = "mock"\nvad = "energy"\n'
        f'[cascade]\ntext_filter = "{__name__}:shout"\n',
        encoding="utf-8",
    )
    cfg = load_config(path)
    session = build_session(cfg)
    engines = [
        session.engine,
        engine_from_app_config(cfg),
        engine_from_config(cfg).engine,
        BenchSystem.from_options(config=path).build_engine(),
    ]
    for engine in engines:
        assert isinstance(engine, CascadeEngine)
        assert engine.options.text_filter is shout
