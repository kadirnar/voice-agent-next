"""Scoring of the T6 tool-use track (pure functions, no audio).

* **Final state** — the database after the call equals the expected state (τ-bench),
  and every ``expect_said`` fact was said: ``passed``.
* **Tool calls** — actual calls are matched to expected ones by name, preferring the
  candidate with the most correct arguments. ``tool_precision`` = matched / made,
  ``tool_recall`` = required matched / required, ``arg_acc`` = correct scored arguments
  over matched calls, ``entity_capture_acc`` = the same for names, emails, phone numbers
  and IDs. Unmatched calls are *unnecessary*; unmatched successful state changes are
  *unexpected writes*.
* **Say-do violations** — the agent says it did something (a write tool's ``claims``
  pattern in a sentence that is neither a question nor negated) but no successful call of
  that tool happened by the end of the turn.
* **Hallucinated tool results** — numbers and order-status words in the agent's speech
  that no tool returned and that the caller or the instructions did not mention.

The hallucination and say-do checks are deterministic heuristics over the agent's text
(the engine's transcript of its own speech); every flagged sentence or token is kept in
``items.jsonl`` for inspection.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .tool_env import (
    ENTITY_KINDS,
    STATUS_WORDS,
    TOOL_LIBRARY,
    CallRecord,
    ExpectedCall,
    ToolScenario,
    ToolSuite,
    canonical,
    db_diff,
    expected_said_ok,
    state_hash,
    words_to_digits,
)

__all__ = [
    "CallMatch",
    "ScenarioScore",
    "TurnObservation",
    "grounding_tokens",
    "match_calls",
    "say_do_violations",
    "score_scenario",
    "ungrounded_facts",
]


@dataclass(slots=True)
class CallMatch:
    """An expected call and the actual call matched to it (``actual`` index or ``None``)."""

    expected: ExpectedCall
    actual: int | None
    args_correct: int = 0
    args_total: int = 0
    entities_correct: int = 0
    entities_total: int = 0
    wrong_args: list[str] = field(default_factory=list)


def _arg_checks(
    call: ExpectedCall, actual: CallRecord, year: int
) -> tuple[int, int, int, int, list[str]]:
    tool = TOOL_LIBRARY[call.name]
    ok = total = ent_ok = ent_total = 0
    wrong: list[str] = []
    for name, value in call.args.items():
        kind = tool.params[name].kind
        good = canonical(value, kind, year=year) == canonical(
            actual.arguments.get(name), kind, year=year
        )
        total += 1
        ok += good
        if kind in ENTITY_KINDS:
            ent_total += 1
            ent_ok += good
        if not good:
            wrong.append(f"{name}={actual.arguments.get(name)!r} (expected {value!r})")
    return ok, total, ent_ok, ent_total, wrong


def match_calls(
    expected: Sequence[ExpectedCall], actual: Sequence[CallRecord], *, year: int = 2026
) -> tuple[list[CallMatch], list[int]]:
    """Match actual calls to expected calls; returns the matches and the unmatched actual
    call indices. Required calls are matched first; among same-named candidates the one
    with the most correct arguments wins (then the earliest)."""
    free = set(range(len(actual)))
    matches: list[CallMatch] = []
    order = sorted(range(len(expected)), key=lambda i: (expected[i].optional, i))
    by_index: dict[int, CallMatch] = {}
    for i in order:
        call = expected[i]
        best: tuple[int, int, tuple[int, int, int, int, list[str]]] | None = None
        for j in sorted(free):
            if actual[j].name != call.name:
                continue
            checks = _arg_checks(call, actual[j], year)
            # prefer more correct args, then a successful call, then the earliest
            key = (checks[0], int(actual[j].ok))
            if best is None or key > (best[0], best[1]):
                best = (key[0], key[1], checks)
                best_j = j
        if best is None:
            by_index[i] = CallMatch(call, None)
            continue
        free.discard(best_j)
        ok, total, ent_ok, ent_total, wrong = best[2]
        by_index[i] = CallMatch(call, best_j, ok, total, ent_ok, ent_total, wrong)
    matches = [by_index[i] for i in range(len(expected))]
    return matches, sorted(free)


# ------------------------------------------------------------------ hallucination

_NUMBER = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")
_SENTENCE = re.compile(r"[^.!?]+[.!?]?")
_NEGATION = re.compile(
    r"\b(not|cannot|can't|unable|won't|couldn't|wasn't|isn't|haven't|hasn't|no longer|"
    r"didn't|don't|never|unfortunately)\b|n't\b"
)


def _norm_number(token: str) -> str:
    token = token.replace(",", "")
    try:
        value = float(token)
    except ValueError:
        return token
    return str(int(value)) if value.is_integer() else f"{value:g}"


def grounding_tokens(sources: Sequence[str]) -> set[str]:
    """Numbers a reply may mention: every number in ``sources`` (tool outputs, caller
    text, instructions), spoken numbers converted, digit runs joined ("four four one oh" ->
    4410), plus the parts of dates and times (``19:00`` also grounds ``7``)."""
    out: set[str] = set()
    for source in sources:
        text = words_to_digits(source)
        for run in re.findall(r"(?:\b\d\b[\s-]*){2,}", text):
            out.add(_norm_number(re.sub(r"\D", "", run)))
        for tok in re.findall(r"\d+(?:[.,]\d+)*", text):
            out.add(_norm_number(tok))
            for part in re.split(r"[.,]", tok):
                if part:
                    out.add(_norm_number(part))
        for h, m in re.findall(r"\b(\d{1,2}):(\d{2})\b", text):
            out.update({str(int(h)), str(int(h) % 12 or 12), str(int(m))})
        for y, mo, d in re.findall(r"\b(\d{4})-(\d{2})-(\d{2})\b", text):
            out.update({str(int(y)), str(int(mo)), str(int(d))})
    return out


def ungrounded_facts(agent_text: str, sources: Sequence[str]) -> list[str]:
    """Numbers and order-status words in ``agent_text`` that no source contains."""
    grounded = grounding_tokens(sources)
    low_sources = " ".join(sources).lower()
    out: list[str] = []
    text = agent_text.replace("$", " ")
    for m in re.finditer(r"\d+(?::\d{2})|\d+(?:,\d{3})*(?:\.\d+)?", text):
        token = m.group()
        if ":" in token:
            h, mm = token.split(":")
            parts = [str(int(h)), str(int(mm))]
            if all(p in grounded for p in parts):
                continue
        elif _norm_number(token) in grounded:
            continue
        out.append(token)
    low = agent_text.lower()
    for word in STATUS_WORDS:
        if re.search(rf"\b{word}\b", low) and word not in low_sources:
            out.append(word)
    return out


def say_do_violations(agent_text: str, tools: Sequence[str], done: set[str]) -> list[str]:
    """Sentences claiming a write tool's action although no successful call of that tool
    (``done``) happened."""
    out: list[str] = []
    for m in _SENTENCE.finditer(agent_text):
        sentence = m.group().strip()
        low = sentence.lower()
        if not sentence or low.endswith("?") or _NEGATION.search(low):
            continue
        for name in tools:
            tool = TOOL_LIBRARY[name]
            if tool.write and tool.claims and name not in done and re.search(tool.claims, low):
                out.append(f"{name}: {sentence}")
    return out


# ------------------------------------------------------------------------ scoring


@dataclass(slots=True)
class TurnObservation:
    """What happened in one turn (from the session and the call log)."""

    index: int
    user_text: str
    agent_text: str
    end: float
    """End of the turn window (next turn's speech onset; ``inf`` for the last turn)."""
    state_after: str | None = None
    """Database state hash when the caller moved on."""


@dataclass
class ScenarioScore:
    passed: bool
    state_ok: bool
    outputs_ok: bool
    missing_outputs: list[str]
    state_diff: list[str]
    required_calls: int
    matched_required: int
    matched_calls: int
    calls: int
    args_correct: int
    args_total: int
    entities_correct: int
    entities_total: int
    unnecessary_calls: int
    unexpected_writes: int
    tool_errors: int
    wrong_args: list[str]
    missing_calls: list[str]
    say_do: list[str]
    say_do_turns: list[int]
    ungrounded: dict[int, list[str]]
    turns_to_completion: int | None
    matches: list[CallMatch] = field(default_factory=list)
    unmatched: list[int] = field(default_factory=list)

    @property
    def precision(self) -> float | None:
        return self.matched_calls / self.calls if self.calls else None

    @property
    def recall(self) -> float | None:
        return self.matched_required / self.required_calls if self.required_calls else None

    @property
    def arg_acc(self) -> float | None:
        return self.args_correct / self.args_total if self.args_total else None

    def summary(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "state_ok": self.state_ok,
            "outputs_ok": self.outputs_ok,
            "missing_outputs": self.missing_outputs,
            "state_diff": self.state_diff,
            "required_calls": self.required_calls,
            "matched_required": self.matched_required,
            "matched_calls": self.matched_calls,
            "calls": self.calls,
            "tool_precision": _round(self.precision),
            "tool_recall": _round(self.recall),
            "args_correct": self.args_correct,
            "args_total": self.args_total,
            "arg_acc": _round(self.arg_acc),
            "entities_correct": self.entities_correct,
            "entities_total": self.entities_total,
            "unnecessary_calls": self.unnecessary_calls,
            "unexpected_writes": self.unexpected_writes,
            "tool_errors": self.tool_errors,
            "wrong_args": self.wrong_args,
            "missing_calls": self.missing_calls,
            "say_do_violations": self.say_do,
            "hallucinations": [t for v in self.ungrounded.values() for t in v],
            "turns_to_completion": self.turns_to_completion,
        }


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 6)


def score_scenario(
    suite: ToolSuite,
    scenario: ToolScenario,
    calls: Sequence[CallRecord],
    turns: Sequence[TurnObservation],
    final_db: dict[str, Any],
) -> ScenarioScore:
    """Score one call of ``scenario`` (see the module docstring)."""
    year = suite.year
    expected_db = suite.expected_db(scenario)
    expected_hash = state_hash(expected_db)
    state_ok = state_hash(final_db) == expected_hash
    said_all = " ".join(t.agent_text for t in turns)
    missing_outputs = expected_said_ok(said_all, scenario.expect_said)
    expected = scenario.expected_calls
    matches, unmatched = match_calls(expected, calls, year=year)
    required = [m for m in matches if not m.expected.optional]

    # hallucinations and say-do violations, turn by turn
    instructions = suite.instructions_for(scenario)
    say_do: list[str] = []
    say_do_turns: list[int] = []
    ungrounded: dict[int, list[str]] = {}
    completion: int | None = None
    for k, turn in enumerate(turns):
        before = [c for c in calls if c.started < turn.end]
        done = {c.name for c in before if c.ok}
        outputs = [c.output for c in before if c.ok or c.output]
        users = [t.user_text for t in turns[: k + 1]]
        flagged = ungrounded_facts(turn.agent_text, [*outputs, *users, instructions])
        if flagged:
            ungrounded[k] = flagged
        violations = say_do_violations(turn.agent_text, scenario.tools, done)
        if violations:
            say_do += violations
            say_do_turns.append(k)
        if completion is None and turn.state_after == expected_hash:
            said = " ".join(t.agent_text for t in turns[: k + 1])
            if not expected_said_ok(said, scenario.expect_said):
                completion = k + 1
    matched_idx = {m.actual for m in matches if m.actual is not None}
    return ScenarioScore(
        passed=state_ok and not missing_outputs,
        state_ok=state_ok,
        outputs_ok=not missing_outputs,
        missing_outputs=missing_outputs,
        state_diff=[] if state_ok else db_diff(final_db, expected_db),
        required_calls=len(required),
        matched_required=sum(m.actual is not None for m in required),
        matched_calls=len(matched_idx),
        calls=len(calls),
        args_correct=sum(m.args_correct for m in matches if m.actual is not None),
        args_total=sum(m.args_total for m in matches if m.actual is not None),
        entities_correct=sum(m.entities_correct for m in matches if m.actual is not None),
        entities_total=sum(m.entities_total for m in matches if m.actual is not None),
        unnecessary_calls=len(unmatched),
        unexpected_writes=sum(calls[j].changed_state for j in unmatched),
        tool_errors=sum(not c.ok for c in calls),
        wrong_args=[f"{m.expected.name}: {w}" for m in matches for w in m.wrong_args],
        missing_calls=[
            f"{m.expected.name}({m.expected.args})" for m in required if m.actual is None
        ],
        say_do=say_do,
        say_do_turns=say_do_turns,
        ungrounded=ungrounded,
        turns_to_completion=completion,
        matches=matches,
        unmatched=unmatched,
    )
