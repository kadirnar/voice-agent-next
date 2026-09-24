"""Built-in providers.

Each module registers its components with
:func:`voice_agent_next.registry.register_provider` at import time and must stay
importable without its optional dependencies. Resolve providers with
``voice_agent_next.create("stt", "deepgram/nova-3")`` or list them with
``van providers``.
"""
