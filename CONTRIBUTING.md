# Contributing to voice-agent-next

Thanks for helping! This document is the contract every contributor — human or AI agent — follows.

## Setup

```bash
git clone https://github.com/kadirnar/voice-agent-next && cd voice-agent-next
uv sync                       # Python >= 3.11; creates .venv with dev tools
uv run pytest                 # fast unit tests: no network, no model downloads
uv run ruff check . && uv run ruff format --check . && uv run mypy
```

Optional provider dependencies are installed with extras, e.g. `uv sync --extra silero`.

## Workflow

1. Every change starts from an issue. Branch from `main`: `feat/<issue>-<slug>`, `fix/<issue>-<slug>`, `docs/<issue>-<slug>`.
2. Commit with [Conventional Commits](https://www.conventionalcommits.org/) (`feat(stt): add Deepgram streaming STT`).
3. Open a PR whose description contains `Closes #<issue>`, a summary, and how it was tested.
4. CI must be green (ruff, ruff format, mypy, pytest on Linux and Windows; macOS runs on `main` and on PRs labelled `ci:full` — add that label for anything platform-specific: audio devices, paths, subprocesses, native wheels).
5. Keep PRs focused: one issue per PR. Don't reformat unrelated code.

## Architecture in one paragraph

User audio flows `Transport -> AgentSession -> EngineConnection`; agent audio flows back the same way. An **engine** (`voice_agent_next.engine.S2SEngine`) is either a native speech-to-speech model (OpenAI Realtime, Gemini Live, Moshi...) or the `CascadeEngine` (VAD -> STT -> turn detector -> LLM -> TTS). All engines speak the event protocol in `voice_agent_next/events.py`. The session owns barge-in, playback pacing, truncation, tool execution, history and metrics. See `docs/ARCHITECTURE.md`.

## Adding a provider

Providers live in `src/voice_agent_next/providers/<name>.py` (or a package `providers/<name>/` for vendors with several components). The module name **is** the provider name used in specs (`"deepgram/nova-3"` imports `providers/deepgram.py`; `-` maps to `_`).

Rules:

* **Importable without optional dependencies.** Never import a third-party SDK at module top level; call `voice_agent_next.utils.require("module", extra="<extra>")` inside `__init__`/methods. `van providers` imports every provider module and CI checks it.
* Register every component with `@register_provider(kind, name, default_model=..., env=(...), extra=..., requires=(...), local=...)` and set the `provider = "<name>"` class attribute.
* Constructors take keyword arguments only and must accept `model: str | None = None`.
* Implement the base-class hooks: `STT._recognize/_create_stream`, `LLM._chat`, `TTS._synthesize/_create_stream`, `VAD._new_inference`, `TurnDetector._predict`, `S2SEngine.connect`. Base classes already handle resampling, metrics, and error propagation.
* API keys come from constructor args first, then the documented env vars. Never log secrets.
* Map provider failures to `voice_agent_next.errors` (401/403 -> `AuthenticationError`, 429 -> `RateLimitError`, network -> `ProviderConnectionError`).
* Add the dependency extra to `pyproject.toml` (alphabetical within its section).
* Add `docs/providers/<name>.md` (setup, models, options, latency notes).

## Tests

* Unit tests must not touch the network or download models. Use fake servers (`websockets.serve` on `127.0.0.1:0`, `httpx.MockTransport`) that replay the provider's real protocol messages.
* Real-API tests: `@pytest.mark.integration`, skipped unless the API key env var is set.
* Real-model tests: `@pytest.mark.model`, skipped unless the dependency is installed; keep downloads small. They run weekly on Linux, macOS and Windows (`.github/workflows/models.yml`) and on PRs labelled `ci:models` — add the label when you touch a local model provider.
* Tests that need audio hardware: `@pytest.mark.audio_device`.
* Timing assertions must use generous tolerances (CI runners are slow; Windows timers are coarse). Prefer assertions that cannot flake:
  * wait for a condition (`wait_for(...)`) instead of sleeping a fixed time; on Windows' Proactor loop a sent message may reach the peer later — wait for delivery;
  * lower bounds are exact when the delay is: mock latencies and tool delays use `utils.clock.sleep_for` / `sleep_until`, which never return early (a plain `asyncio.sleep` can be ~16 ms short on Windows);
  * upper bounds allow for how late *this run's* event loop was: `async with tests.timing.LoopLag() as lag: ...` then `assert measured <= designed + margin + lag.max`;
  * compare with timestamps measured in the same run (or count samples) rather than with wall-clock constants.
* Hunting flaky tests: the manual **flake-hunt** workflow (Actions → flake-hunt → Run workflow) runs the whole suite 5× on Linux, Windows and macOS and lists every test that failed at least once (optional extra pytest arguments, e.g. a test path). A flaky test is fixed at its root cause, never skipped or loosened until it can no longer fail on a mutation of the behaviour it checks.

## Code conventions

* `from __future__ import annotations`, full type hints, mypy clean, ruff (line length 100).
* Audio is s16le `AudioFrame` everywhere; convert at the edges. Use `StreamResampler` for rate conversion.
* Timestamps use `voice_agent_next.utils.now()` (`perf_counter`), never `time.time()` for latency.
* Never block the event loop: run CPU-heavy inference (> ~5 ms) with `asyncio.to_thread` or a dedicated executor.
* Use `Chan`, `cancel_and_wait` and `BackgroundTasks` from `voice_agent_next.utils` for task hygiene; every task you start must be cancelled/awaited on close.
* Public APIs get docstrings. Keep comments for *why*, not *what*.
* Cross-platform: `pathlib`, no shell-specific code, no Unix-only signals, `platformdirs` for caches.
