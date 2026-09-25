"""The config/preset entry points are importable from the package root (#141)."""

from __future__ import annotations

import voice_agent_next as van
from voice_agent_next import app, config, presets

NAMES = {
    "AppConfig": config.AppConfig,
    "load_config": config.load_config,
    "load_preset": presets.load_preset,
    "session_from_preset": presets.session_from_preset,
    "build_session": app.build_session,
    "build_agent": app.build_agent,
}


def test_top_level_exports() -> None:
    for name, obj in NAMES.items():
        assert getattr(van, name) is obj
        assert name in van.__all__


def test_quick_start_from_the_root() -> None:
    from voice_agent_next import Agent, build_agent, build_session, load_config

    cfg = load_config({"engine": "mock", "agent": {"instructions": "Be brief."}})
    session = build_session(cfg)
    agent = build_agent(cfg)
    assert isinstance(agent, Agent) and agent.instructions == "Be brief."
    assert session.engine.provider == "mock"


def test_config_exports_are_lazy() -> None:
    import subprocess
    import sys

    code = "import sys, voice_agent_next; print('voice_agent_next.config' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"  # pydantic/presets load on first use only
