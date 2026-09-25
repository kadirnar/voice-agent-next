"""Text end-of-turn detector that asks an LLM (any registered one) a yes/no question.

``{provider: llm_turn, model: "ollama/qwen3.5:4b"}`` (``model`` is the LLM's spec;
default: the LFM2.5 1.2B of the local presets). The detector shows the LLM the agent's
last turn and the user's transcript so far and asks, in one short system prompt, whether
the user has finished. The answer is read as a probability:

* **OpenAI-compatible LLMs** (OpenAI, Ollama, vLLM, llama.cpp, LM Studio, Groq...): one
  non-streamed completion of one token with ``logprobs``; the probability is
  ``p(yes) / (p(yes) + p(no))`` over the top alternatives, the answer text otherwise;
* **any other LLM**: the streamed answer, ``yes`` -> ``confidence``, ``no`` ->
  ``1 - confidence``.

It has a **latency budget** (``timeout``): past it the detector returns ``fallback``
(1.0 = "no objection", silence and the audio detector decide). Inside a
:class:`~voice_agent_next.turn.FusedTurnDetector`, a missed budget means audio only.

Measured on eot-bench English with LFM2.5-1.2B on Ollama (``docs/providers/lm-turn.md``):
ROC-AUC 0.68, below both Smart Turn (0.83) and ``lm_turn`` (0.74), so it is not a
default; it is for larger LLMs (a 4B+ model or a fast cloud one), whose judgement of
"the user will add details" is much better than a 1B model's.
"""

from __future__ import annotations

import asyncio
import math
from typing import Any

from ..audio.frame import AudioFrame
from ..chat import ChatContext
from ..errors import ConfigurationError
from ..llm import LLM
from ..registry import create, register_provider
from ..turn import TurnDetector, turn_text
from ..utils.log import logger

__all__ = ["DEFAULT_LLM", "DEFAULT_PROMPT", "LLMTurnDetector", "yes_probability"]

DEFAULT_LLM = "ollama/LiquidAI/lfm2.5-1.2b-instruct"
DEFAULT_PROMPT = (
    "You are the end-of-turn detector of a voice assistant. The user's words come from "
    "speech recognition and the user has just paused. Has the user finished their turn "
    "(a complete thought the assistant should answer now), or will they keep talking "
    "(an unfinished sentence, a list or number still being given)? Answer yes if "
    "finished, no if they will keep talking."
)


def _word(token: Any) -> str:
    return str(token or "").strip().strip(".,!\"'").lower()


def yes_probability(top: list[tuple[str, float]]) -> float | None:
    """``p(yes) / (p(yes) + p(no))`` from ``(token, logprob)`` alternatives (``None``: the
    model put its mass on neither)."""
    yes = sum(math.exp(lp) for tok, lp in top if _word(tok) in ("yes", "y"))
    no = sum(math.exp(lp) for tok, lp in top if _word(tok) in ("no", "n"))
    return yes / (yes + no) if yes + no > 0 else None


@register_provider(
    "turn",
    "llm_turn",
    description="Text end-of-turn: asks any LLM whether the user has finished (yes/no)",
    default_model=DEFAULT_LLM,
    env=(),
    local=False,
)
class LLMTurnDetector(TurnDetector):
    """Asks an LLM whether the user has finished their turn.

    Args:
        model: the LLM's spec (e.g. ``"ollama/qwen3.5:4b"``, ``"groq/llama-3.1-8b-instant"``).
        llm: an LLM instance or spec instead of ``model`` (an instance is not closed by
            :meth:`aclose`: share the agent's, say).
        prompt: the system prompt; it must ask for a yes (finished) / no answer.
        timeout: latency budget in seconds (``None``: none).
        fallback: probability returned past the budget or when the answer is neither yes
            nor no.
        confidence: the probability of a plain "yes" answer (LLMs without logprobs).
        threshold: probability at/above which the turn counts as complete.
    """

    provider = "llm_turn"
    modality = "text"

    def __init__(
        self,
        *,
        model: str | None = None,
        llm: Any = None,
        prompt: str = DEFAULT_PROMPT,
        timeout: float | None = 0.5,
        fallback: float = 1.0,
        confidence: float = 0.9,
        threshold: float = 0.5,
        max_agent_chars: int = 300,
    ) -> None:
        for name, value in (("fallback", fallback), ("confidence", confidence)):
            if not 0.0 <= value <= 1.0:
                raise ConfigurationError(f"{name} must be in [0, 1], got {value}")
        self.llm: LLM = create("llm", llm if llm is not None else (model or DEFAULT_LLM))
        self._owns_llm = not isinstance(llm, LLM)  # a shared instance is closed by its owner
        super().__init__(model=f"{self.llm.provider}/{self.llm.model}", threshold=threshold)
        self.prompt = prompt
        self.timeout = timeout
        self.fallback = fallback
        self.confidence = confidence
        self.max_agent_chars = max_agent_chars

    def _context(self, agent: str, user: str) -> ChatContext:
        if len(agent) > self.max_agent_chars:
            agent = agent[len(agent) - self.max_agent_chars :].lstrip()
        ctx = ChatContext()
        ctx.add_message("system", self.prompt)
        ctx.add_message("user", (f"Assistant: {agent}\n" if agent else "") + f"User: {user}")
        return ctx

    async def _ask_logprobs(self, ctx: ChatContext) -> float | None:
        """One-token completion with logprobs (OpenAI-compatible LLMs)."""
        llm: Any = self.llm
        model = await llm._ensure_model()
        request = llm.build_request(
            ctx,
            model=model,
            temperature=0.0,
            max_tokens=1,
            extra={"logprobs": True, "top_logprobs": 10},
        )
        request["stream"] = False
        request.pop("stream_options", None)
        try:
            response = await llm._client.chat.completions.create(**request)
        except Exception as exc:
            raise llm._map_error(exc) from exc
        choice = response.choices[0]
        content = getattr(getattr(choice, "logprobs", None), "content", None) or []
        if content:
            first = content[0]
            top = [(t.token, t.logprob) for t in (getattr(first, "top_logprobs", None) or [])]
            p = yes_probability(top or [(first.token, first.logprob)])
            if p is not None:
                return p
        return self._from_text(getattr(choice.message, "content", "") or "")

    def _from_text(self, text: str) -> float | None:
        word = _word(text.split()[0] if text.split() else "")
        if word.startswith("yes"):
            return self.confidence
        if word.startswith("no"):
            return 1.0 - self.confidence
        return None

    async def _ask(self, ctx: ChatContext) -> float | None:
        from .openai.llm import OpenAILLM

        if isinstance(self.llm, OpenAILLM):
            return await self._ask_logprobs(ctx)
        result = await self.llm.chat(ctx, temperature=0.0, max_tokens=3).collect()
        return self._from_text(result.text)

    async def _predict(self, *, audio: AudioFrame | None, chat_ctx: ChatContext | None) -> float:
        agent, user = turn_text(chat_ctx)
        if not user:
            return 1.0
        try:
            p = await asyncio.wait_for(self._ask(self._context(agent, user)), self.timeout)
        except TimeoutError:
            logger.debug("llm_turn: %s over its %ss budget", self.model, self.timeout)
            return self.fallback
        return self.fallback if p is None else p

    async def warmup(self) -> None:
        await self.llm.warmup()

    async def aclose(self) -> None:
        if self._owns_llm:
            await self.llm.aclose()
