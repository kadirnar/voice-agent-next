# Providers

Every component is addressed by a `provider/model` spec (`deepgram/nova-3`,
`ollama/qwen3.5:4b`, `kokoro`) and created through the registry:

```python
from voice_agent_next import AgentSession, create

stt = create("stt", "deepgram/nova-3", language="en")
session = AgentSession(stt=stt, llm="anthropic/claude-haiku-4-5", tts="cartesia", vad="silero")
```

The provider name is the module name under `voice_agent_next.providers`; everything after
the first `/` is the model. Provider modules import without their optional dependencies:
install the **extra** in the table (`pip install 'voice-agent-next[kokoro]'`) and set the
**environment variables** it needs. `van providers` shows the same list with the status on
your machine (`ready`, or what is missing). A list of specs (`llm: [anthropic, groq]`) is a
[failover chain](../concepts/failover.md).

Linked names have their own page (setup, models, options, latency notes).

<!-- providers-table -->

## Third-party providers

Packages add providers through the `voice_agent_next.providers` entry-point group; they
appear here and in `van providers` once installed. See
[Contributing: adding a provider](../contributing.md#adding-a-provider).
