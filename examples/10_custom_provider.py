"""Write your own provider, register it, and use it by name like a built-in one.

The provider here is a keyword FAQ "LLM": no model, deterministic answers from a table.
The same pattern wraps any in-house model or API. Implement the base-class hook
(``LLM._chat`` returning an ``LLMStream``; for other kinds: ``STT._create_stream``,
``TTS._synthesize``, ``VAD._new_inference``, ``TurnDetector._predict``,
``S2SEngine.connect``). The base class then handles metrics, errors and resampling.

``@register_provider`` puts the class in the registry, so specs, config files and the CLI
find it::

    AgentSession(stt="sherpa-onnx", llm="faq", tts="kokoro", vad="silero")
    # agent.yaml:  llm: {provider: faq, answers: {...}}

To ship it as a plugin package, declare an entry point in *your* ``pyproject.toml``. The
registry imports it the first time a name is not found (``van providers`` lists it)::

    [project.entry-points."voice_agent_next.providers"]
    faq = "my_package.faq_llm"          # the module that calls @register_provider

Run::

    python examples/10_custom_provider.py --mock          # offline: mock STT/TTS around it
    python examples/10_custom_provider.py --stt sherpa-onnx --tts kokoro --vad silero   # mic
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import re
import sys
from collections.abc import Mapping
from typing import Any

from _common import log_conversation, scratch_dir, synthetic_question

from voice_agent_next import (
    LLM,
    AgentSession,
    CascadeOptions,
    ChatChunk,
    ChatContext,
    LLMCapabilities,
    LLMStream,
    create,
    list_providers,
    register_provider,
)
from voice_agent_next.llm import ToolChoice
from voice_agent_next.session import Agent
from voice_agent_next.tools import FunctionTool
from voice_agent_next.transports import FileTransport, create_transport

DEFAULT_ANSWERS = {
    "hours|open": "We are open from nine to six, Monday to Saturday.",
    "where|address": "We are at 12 Harbour Street, next to the ferry.",
    "price|cost": "A standard service costs forty euros.",
}


@register_provider(
    "llm",
    "faq",
    description="Keyword FAQ answers (example custom provider)",
    default_model="v1",
    local=True,  # runs on this machine: no API key (otherwise env=("FAQ_API_KEY",))
)
class FaqLLM(LLM):
    """Answers with the first entry whose keyword pattern matches the user's last message."""

    provider = "faq"  # the name in specs: "faq" or "faq/v1"

    # Constructors take keyword arguments only and accept `model` (the registry passes it).
    def __init__(self, *, model: str | None = None, answers: Mapping[str, str] | None = None,
                 fallback: str = "Sorry, I can only answer questions about the shop.") -> None:  # fmt: skip
        super().__init__(model=model or "v1", capabilities=LLMCapabilities(tool_calling=False))
        self.answers = dict(answers or DEFAULT_ANSWERS)
        self.fallback = fallback

    def answer(self, question: str) -> str:
        for pattern, reply in self.answers.items():
            if re.search(pattern, question, re.IGNORECASE):
                return reply
        return self.fallback

    def _chat(
        self,
        ctx: ChatContext,
        *,
        tools: list[FunctionTool],
        tool_choice: ToolChoice | None,
        temperature: float | None,
        max_tokens: int | None,
        extra: dict[str, Any],
    ) -> LLMStream:
        return _FaqStream(self, ctx, tools=tools, tool_choice=tool_choice,
                          temperature=temperature, max_tokens=max_tokens, extra=extra)  # fmt: skip


class _FaqStream(LLMStream):
    async def _run(self) -> None:
        llm: FaqLLM = self._llm  # type: ignore[assignment]
        last = self.ctx.last_message("user")
        reply = llm.answer(last.text if last else "")
        # stream word by word, like a real model: the TTS can start on the first clause
        for word in re.findall(r"\S+\s*", reply):
            self._push(ChatChunk(self.request_id, delta=word))
        self._push(ChatChunk(self.request_id, finish_reason="stop"))


async def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--mock", action="store_true", help="offline: mock STT/TTS + WAV file")
    parser.add_argument("--stt", default="sherpa-onnx")
    parser.add_argument("--tts", default="kokoro")
    parser.add_argument("--vad", default="silero")
    args = parser.parse_args(argv)

    # the registry knows it now, next to the built-ins
    spec = next(p for p in list_providers("llm") if p.name == "faq")
    print(f"registered: llm/{spec.name} ({spec.description}), local={spec.local}")
    llm = create(
        "llm", {"provider": "faq", "answers": {**DEFAULT_ANSWERS, "dog": "Dogs are welcome!"}}
    )
    print(f"create(...) -> {type(llm).__name__}, model {llm.model!r}")

    if args.mock:
        session = AgentSession(
            stt={"provider": "mock", "transcripts": ["When are you open?"]},  # specs, not classes
            llm=llm,  # an instance works too
            tts={"provider": "mock", "chars_per_second": 40},
            vad="energy",
            cascade_options=CascadeOptions(min_endpointing_delay=0.0),
        )
        tmp = scratch_dir()
        transport: Any = FileTransport(synthetic_question(tmp / "q.wav"), realtime=False, hold=0.5)
    else:
        session = AgentSession(stt=args.stt, llm="faq", tts=args.tts, vad=args.vad)
        transport = create_transport("local")
    log_conversation(session)
    await session.run(Agent("Shop FAQ bot."), transport)
    return 0


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        sys.exit(asyncio.run(main()))
