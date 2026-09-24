# Guidance for AI coding agents working on voice-agent-next

Read `CONTRIBUTING.md` first — it is binding. This file adds agent-specific rules.

## Commands

```bash
uv sync                                   # env (add --extra <name> for provider deps)
uv run pytest -q                          # unit tests (must stay fast, offline)
uv run pytest -q tests/path::test_name    # one test
uv run ruff check . --fix && uv run ruff format . && uv run mypy
uv run van providers                      # registry smoke test
uv run van demo                           # offline end-to-end demo
```

## Map of the code

| Path | What |
| --- | --- |
| `src/voice_agent_next/audio/` | `AudioFrame` (s16le), buffers, resampling, G.711, WAV |
| `stt.py`, `tts.py`, `llm.py`, `vad.py`, `turn.py` | component interfaces + streaming base classes |
| `engine.py`, `events.py` | speech-to-speech engine interface and its event protocol |
| `engines/cascade.py` | `CascadeEngine` (VAD -> STT -> turn -> LLM -> TTS) |
| `session/` | `Agent`, `AgentSession` (barge-in, truncation, tools, metrics) |
| `transports/` | `Transport` interface, loopback, file (+ local/websocket/webrtc/telephony) |
| `providers/` | one module per provider; `mock.py` = deterministic fakes for tests |
| `registry.py` | `create("stt", "deepgram/nova-3")`, `@register_provider` |
| `config.py`, `app.py`, `cli/` | YAML/TOML config, builders, `van` CLI |
| `bench/`, `benchmarks/` | benchmark framework and suites |

## Rules for agents

* Work only on your issue. Do not refactor shared core files (`engine.py`, `events.py`, `session/`, `stt.py`, `tts.py`, `llm.py`, `vad.py`, `registry.py`) unless the issue says so; if you believe a core change is required, make the smallest backward-compatible change and explain it in the PR description.
* Never weaken or delete existing tests to make yours pass.
* Do not download models larger than ~300 MB or install CUDA/PyTorch unless the issue explicitly requires it. Disk space on the dev machine is limited.
* Do not commit secrets, `.env` files, generated audio, model weights, or benchmark result dumps.
* Before opening a PR: `uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest -q` must all pass.
* PR description: `Closes #<n>`, what/why, how tested, follow-ups (as a checklist).
