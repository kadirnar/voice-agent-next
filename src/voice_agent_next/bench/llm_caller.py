"""An LLM-driven caller for the T6 tool-use track (τ-bench style user simulator).

The scripted caller of :mod:`voice_agent_next.bench.tracks.tools` says the next line of a
scenario whatever the agent answered. :class:`LLMCaller` instead *reacts*: an LLM plays
the customer, with a persona and a goal taken from the scenario, reads what the agent
said and decides what to say next (or to hang up). Each line is spoken with the suite's
caller voice (TTS, or synthetic speech) while the call streams on, so the conversation
still runs in real time over the T1 harness.

* **Persona** — ``scenario.persona``, else a generic customer of the suite's business.
* **Goal** — ``scenario.goal`` (else ``scenario.description``), plus the scripted lines as
  the details the caller knows: the LLM says them in its own words and only when the
  conversation gets there, so the expected tool calls and final state stay valid.
* **Determinism** — temperature 0 and a fixed ``seed`` (sent to OpenAI-compatible
  servers, which accept it); with a deterministic agent a call is reproducible as far as
  the caller model is.
* **End** — the LLM answers :data:`END_TOKEN` to hang up; the call also ends after
  ``max_turns`` lines (default: the script's length + 3).

The scripted caller stays the default (``--caller scripted``); ``--caller llm:<spec>``
selects this one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..chat import ChatContext
from ..llm import LLM
from ..tts import TTS
from .stimuli import Stimulus, render_stimuli
from .tool_env import ToolScenario, ToolSuite, ToolTurn

__all__ = [
    "END_TOKEN",
    "CallerLine",
    "LLMCaller",
    "LLMCallerOptions",
    "caller_prompt",
    "describe_caller",
    "parse_caller",
]

END_TOKEN = "<END>"
"""What the caller LLM answers to hang up."""

_DEFAULT_PERSONA = (
    "You are a customer phoning the assistant of the business described below. You are "
    "polite and to the point, and you talk like a person on the phone: short, plain "
    "sentences."
)


@dataclass
class LLMCallerOptions:
    """How the LLM caller generates its lines."""

    max_turns: int | None = None
    """Most lines per call (default: the scenario's scripted turns + 3)."""
    temperature: float = 0.0
    seed: int | None = 0
    """Sent as ``seed`` to OpenAI-compatible servers (others ignore it)."""
    max_tokens: int = 120

    def validate(self) -> None:
        if self.max_turns is not None and self.max_turns < 1:
            raise ValueError("caller max_turns must be >= 1")
        if self.temperature < 0:
            raise ValueError("caller temperature must be >= 0")
        if self.max_tokens < 8:
            raise ValueError("caller max_tokens must be >= 8")


def parse_caller(value: str | None) -> str | None:
    """The LLM spec of ``--caller``: ``None`` for ``scripted``, ``<spec>`` for
    ``llm:<spec>``; anything else is a ``ValueError``."""
    text = (value or "scripted").strip()
    if text == "scripted":
        return None
    if text.startswith("llm:") and text[4:].strip():
        return text[4:].strip()
    raise ValueError(f"--caller must be 'scripted' or 'llm:<llm spec>', not {value!r}")


def caller_prompt(suite: ToolSuite, scenario: ToolScenario) -> str:
    """The caller LLM's system prompt: persona, goal, the details it knows, the rules."""
    persona = (scenario.persona or _DEFAULT_PERSONA).strip()
    goal = (scenario.goal or scenario.description or "").strip()
    script = "\n".join(f"- {t.text}" for t in scenario.turns)
    business = suite.description.strip() or suite.name
    parts = [
        persona,
        f"Business: {business}. Today is {suite.long_date()}.",
    ]
    if goal:
        parts.append(f"Your goal for this call: {goal}")
    parts += [
        "The call as it was scripted. It shows what you want and every detail you know "
        "(numbers, names, dates, addresses); use exactly these details, do not invent "
        f"others:\n{script}",
        "Rules: you speak first. Say one short turn at a time (one or two sentences), in "
        "your own words, and react to what the assistant just said: answer its "
        "questions, give a detail when it is needed, confirm when it asks you to. Say "
        "numbers the way the script says them. Never play the assistant and never "
        "describe actions; write only the words you say. When your goal is done (or "
        "cannot be done) and you have said goodbye, reply with exactly "
        f"{END_TOKEN} to hang up.",
    ]
    return "\n\n".join(parts)


@dataclass(slots=True)
class CallerLine:
    """One exchange: what the agent said before and what the caller answered."""

    agent: str
    caller: str | None
    """``None``: the caller hung up."""


_STRIP = re.compile(r"^\s*(?:caller|customer|user|me)\s*:\s*", re.IGNORECASE)


def _clean(text: str) -> str:
    """One spoken line: the LLM's text without quotes, a speaker label or line breaks."""
    line = " ".join(text.split())
    line = _STRIP.sub("", line).strip().strip('"').strip()
    return line


@dataclass
class LLMCaller:
    """Plays the caller of one call (see the module docstring). ``tts`` voices the lines
    (``None``: the suite's caller voice, created per line, or synthetic speech)."""

    llm: LLM
    suite: ToolSuite
    scenario: ToolScenario
    options: LLMCallerOptions = field(default_factory=LLMCallerOptions)
    tts: TTS | None = None
    lines: list[CallerLine] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    ended: bool = False
    """The LLM hung up (as opposed to ``max_turns`` or an error)."""
    _ctx: ChatContext = field(init=False)

    def __post_init__(self) -> None:
        self.options.validate()
        self._ctx = ChatContext()
        self._ctx.add_message("system", caller_prompt(self.suite, self.scenario))

    @property
    def max_turns(self) -> int:
        return self.options.max_turns or len(self.scenario.turns) + 3

    def _extra(self) -> dict[str, Any]:
        from ..providers.openai.llm import OpenAILLM

        if self.options.seed is not None and isinstance(self.llm, OpenAILLM):
            return {"seed": self.options.seed}
        return {}

    async def next_line(self, agent_reply: str) -> str | None:
        """What the caller says after ``agent_reply`` (``""`` before the first line), or
        ``None`` to hang up."""
        if self.ended or len([ln for ln in self.lines if ln.caller]) >= self.max_turns:
            return None
        reply = " ".join(agent_reply.split())
        if self.lines or reply:
            # the assistant's words are the "user" turn of the caller model
            self._ctx.add_message("user", reply or "(silence)")
        else:
            self._ctx.add_message("user", "(The call connects. You speak first.)")
        try:
            result = await self.llm.chat(
                self._ctx,
                temperature=self.options.temperature,
                max_tokens=self.options.max_tokens,
                extra=self._extra(),
            ).collect()
        except Exception as exc:  # the call ends; the scenario is scored as it stands
            self.errors.append(f"caller LLM: {exc!r}")
            self.lines.append(CallerLine(reply, None))
            return None
        text = _clean(result.text)
        said = _clean(text.replace(END_TOKEN, " "))
        if END_TOKEN in text:
            self.ended = True
        if not said:
            self.lines.append(CallerLine(reply, None))
            self.ended = True
            return None
        self._ctx.add_message("assistant", said)
        self.lines.append(CallerLine(reply, said))
        return said

    async def render(self, index: int, text: str) -> Stimulus:
        """Speak ``text`` like a scripted turn of the scenario (same voice, loudness,
        chunking and speech annotation)."""
        turn = ToolTurn(id=f"llm{index}", text=text)
        one = self.scenario.model_copy(update={"turns": [turn]})
        (stim,) = await render_stimuli(self.suite.stimulus_scenario([one]), tts=self.tts)
        return stim

    def transcript(self) -> list[dict[str, str | None]]:
        return [{"agent": ln.agent, "caller": ln.caller} for ln in self.lines]

    def describe(self) -> dict[str, Any]:
        return {**describe_caller(self.llm, self.options), "max_turns": self.max_turns}


def describe_caller(llm: LLM, options: LLMCallerOptions) -> dict[str, Any]:
    """JSON-able description of an LLM caller for the run manifest."""
    return {
        "llm": {"provider": llm.provider, "model": llm.model},
        "temperature": options.temperature,
        "seed": options.seed,
        "max_turns": options.max_turns,
        "end_token": END_TOKEN,
    }
