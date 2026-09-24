# API reference

Generated from the docstrings with [mkdocstrings](https://mkdocstrings.github.io/). It
covers the public API: what `voice_agent_next` and its subpackages export. Everything else
(names starting with `_`, provider internals) may change without notice.

| Page | Modules |
|---|---|
| [voice_agent_next](package.md) | the top-level package: the names most programs import |
| [Session](session.md) | `voice_agent_next.session`: `Agent`, `AgentSession`, `SessionOptions`, handoffs and flows, session events, interruptions, recording, tracing |
| [Engine and events](engine.md) | `voice_agent_next.engine`, `voice_agent_next.events`, `voice_agent_next.engines.cascade` |
| [Components](components.md) | `voice_agent_next.stt`, `tts`, `llm`, `vad`, `turn`, `tools` |
| [Transports](transports.md) | `voice_agent_next.transports` and each transport |
| [Registry](registry.md) | `voice_agent_next.registry`: provider specs and `create()` |
| [Failover](fallback.md) | `voice_agent_next.fallback` |
| [Presets](presets.md) | `voice_agent_next.presets` |
| [Models](models.md) | `voice_agent_next.models`: the model catalog and cache |
| [Benchmarks](bench.md) | `voice_agent_next.bench` |

Provider classes are documented on their [provider pages](../providers/index.md).
