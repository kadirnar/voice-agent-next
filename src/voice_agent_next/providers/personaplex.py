"""NVIDIA PersonaPlex full-duplex speech-to-speech engine (Moshi protocol + persona prompts).

PersonaPlex-7B is a fine-tune of Moshi that is conditioned on a **text role prompt** and a
**voice prompt**. Its server (``python -m moshi.server`` from the ``NVIDIA/personaplex``
repository) speaks the Moshi ``/api/chat`` protocol unchanged and reads both prompts from
the query string when a conversation starts:

* ``text_prompt`` — the agent's ``instructions`` (wrapped in ``<system>`` tags by the
  server), e.g. ``"You enjoy having a good conversation."``;
* ``voice_prompt`` — a voice embedding file in the server's voice directory
  (``NATF0``-``NATF3``, ``NATM0``-``NATM3``, ``VARF0``-``VARF4``, ``VARM0``-``VARM4``);
  ``Agent(voice=...)`` wins over the engine's ``voice``.

The server processes the prompts before it sends the handshake, so connecting takes a few
seconds. Everything else — events, full-duplex behaviour, reconnects — is
:class:`~voice_agent_next.providers.moshi.MoshiEngine`.
"""

from __future__ import annotations

from typing import Any

from ..engine import EngineOptions
from ..registry import register_provider
from ..utils.log import logger
from .moshi import MoshiEngine

__all__ = ["DEFAULT_MODEL", "DEFAULT_TEXT_PROMPT", "DEFAULT_VOICE", "VOICES", "PersonaPlexEngine"]

DEFAULT_MODEL = "personaplex-7b-v1"
DEFAULT_URL = "wss://localhost:8998"
"""The server's documented start command serves HTTPS with a temporary certificate."""
DEFAULT_VOICE = "NATF2"
DEFAULT_TEXT_PROMPT = "You enjoy having a good conversation."
VOICES = (
    *(f"NATF{i}" for i in range(4)),
    *(f"NATM{i}" for i in range(4)),
    *(f"VARF{i}" for i in range(5)),
    *(f"VARM{i}" for i in range(5)),
)


@register_provider(
    "engine",
    "personaplex",
    description="NVIDIA PersonaPlex full-duplex speech-to-speech (Moshi protocol, persona prompts)",
    default_model=DEFAULT_MODEL,
    models=(DEFAULT_MODEL,),
    extra="moshi",
    requires=("sphn", "websockets"),
    local=True,
)
class PersonaPlexEngine(MoshiEngine):
    """PersonaPlex engine: :class:`MoshiEngine` plus role and voice prompts.

    Args:
        voice: voice prompt (``"NATF2"`` or a file name such as ``"NATF2.pt"``/``"x.wav"``
            in the server's voice directory); ``EngineOptions.voice`` wins.
        text_prompt: role prompt used when the agent has no instructions.
        url: server origin (default ``wss://localhost:8998``).
        **kwargs: every :class:`MoshiEngine` option.
    """

    provider = "personaplex"
    handshake_timeout_hint = "is the PersonaPlex server running? `python -m moshi.server --ssl DIR`"

    def __init__(
        self,
        *,
        model: str | None = None,
        voice: str | None = None,
        text_prompt: str | None = None,
        url: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model or DEFAULT_MODEL, url=url or DEFAULT_URL, **kwargs)
        self.voice = voice or DEFAULT_VOICE
        self.text_prompt = text_prompt or DEFAULT_TEXT_PROMPT

    def _query(self, options: EngineOptions) -> dict[str, Any]:
        voice = options.voice or self.voice
        if "." not in voice:
            voice = f"{voice}.pt"  # the packaged voices are pre-computed embeddings
        query = dict(self.query)
        query["voice_prompt"] = voice
        query["text_prompt"] = options.instructions.strip() or self.text_prompt
        return query

    def _check_options(self, options: EngineOptions) -> None:
        if options.tools:
            logger.warning(
                "personaplex: the model cannot call tools; %d tools ignored", len(options.tools)
            )
