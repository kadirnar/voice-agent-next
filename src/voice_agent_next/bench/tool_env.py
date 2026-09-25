"""The T6 tool-use world: a deterministic mock database, mock tools and scripted scenarios.

A **suite** (YAML, see ``benchmarks/README.md`` "T6") holds a shared initial database
(orders, customers, restaurant bookings and availability), the agent's instructions and
a list of **scenarios**. A scenario offers the agent a subset of the tool library, scripts
what the caller says (3–6 turns, rendered to speech like the T1 stimuli) and states what
must happen:

* ``turns[].expect_calls`` — the tool calls the agent should make, with the arguments
  that are scored (``optional: true`` for reasonable but unnecessary look-ups);
* ``expect_said`` — facts the agent must tell the caller (any-of alternatives);
* the **expected final database state** — by default the initial database after
  replaying the expected write calls with the same tool implementations (τ-bench's
  method), or an explicit ``expected_state``.

Tools are deterministic: they read and write the per-session database, validate their
arguments, enforce simple policies (a shipped order cannot be cancelled) and wait a fixed
``tool_delay`` (equal for every system, as in Full-Duplex-Bench v3). Arguments are
compared *semantically* by parameter kind (``"October 3rd"`` = ``"2026-10-03"``,
``"7 pm"`` = ``"19:00"``, ``"jane dot doe at example dot com"`` =
``"jane.doe@example.com"``), so a voice agent is not penalized for formatting.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializationInfo,
    SerializerFunctionWrapHandler,
    model_serializer,
    model_validator,
)

from ..chat import ChatContext, FunctionCall, FunctionCallOutput
from ..errors import ConfigurationError, ToolError
from ..tools import FunctionTool
from ..utils.clock import now, sleep_until
from .stimuli import Scenario, TurnSpec

__all__ = [
    "BUILTIN_TOOL_SUITES",
    "TOOL_LIBRARY",
    "CallRecord",
    "ExpectedCall",
    "ParamKind",
    "ParamSpec",
    "ToolDef",
    "ToolScenario",
    "ToolSuite",
    "ToolTurn",
    "build_tools",
    "canonical",
    "load_tool_suite",
    "reference_engine",
    "reference_policy",
    "state_hash",
]

ParamKind = Literal["id", "int", "date", "time", "email", "phone", "name", "text", "free"]
"""How an argument is normalized before it is compared or used (``free``: never scored)."""
ENTITY_KINDS: frozenset[str] = frozenset({"id", "email", "phone", "name"})
"""Argument kinds counted by ``entity_capture_acc`` (names, emails, phone numbers, IDs)."""

_DATA_DIR = Path(__file__).parent / "data"
BUILTIN_TOOL_SUITES: dict[str, Path] = {"smoke": _DATA_DIR / "tools_smoke.yaml"}
"""Pinned suites shipped with the library (``van bench tools --scenarios smoke``)."""

# ------------------------------------------------------------------ normalization

_UNITS = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
]
_TENS = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
_NUMBER_WORDS: dict[str, int] = {w: i for i, w in enumerate(_UNITS)} | _TENS | {"oh": 0}
_ORDINALS = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
    "eleventh": 11,
    "twelfth": 12,
    "thirteenth": 13,
    "fourteenth": 14,
    "fifteenth": 15,
    "sixteenth": 16,
    "seventeenth": 17,
    "eighteenth": 18,
    "nineteenth": 19,
    "twentieth": 20,
    "thirtieth": 30,
}
_MONTHS = {
    m: i + 1
    for i, names in enumerate(
        [
            ("january", "jan"),
            ("february", "feb"),
            ("march", "mar"),
            ("april", "apr"),
            ("may",),
            ("june", "jun"),
            ("july", "jul"),
            ("august", "aug"),
            ("september", "sep", "sept"),
            ("october", "oct"),
            ("november", "nov"),
            ("december", "dec"),
        ]
    )
    for m in names
}
_ABBREVIATIONS = {
    "st": "street",
    "ave": "avenue",
    "rd": "road",
    "dr": "drive",
    "apt": "apartment",
    "blvd": "boulevard",
    "ln": "lane",
}


def words_to_digits(text: str) -> str:
    """Spoken numbers to digits: ``"table for four"`` -> ``"table for 4"``, ``"twenty one"``
    -> ``"21"``, ``"one oh four two"`` -> ``"1 0 4 2"`` (digit runs stay separate words)."""
    out: list[str] = []
    tokens = re.findall(r"[A-Za-z]+|[^A-Za-z]+", text)
    i = 0
    while i < len(tokens):
        low = tokens[i].lower()
        if low in _TENS:
            value = _TENS[low]
            # "twenty one" / "twenty-one"
            if i + 2 < len(tokens) and tokens[i + 1] in (" ", "-"):
                nxt = tokens[i + 2].lower()
                if nxt in _NUMBER_WORDS and 0 < _NUMBER_WORDS[nxt] < 10:
                    out.append(str(value + _NUMBER_WORDS[nxt]))
                    i += 3
                    continue
            out.append(str(value))
        elif low in _NUMBER_WORDS:
            out.append(str(_NUMBER_WORDS[low]))
        elif low in _ORDINALS:
            out.append(str(_ORDINALS[low]))
        else:
            out.append(tokens[i])
        i += 1
    return "".join(out)


def _canon_id(value: Any) -> str:
    text = words_to_digits(str(value)).lower()
    text = re.sub(r"\b(order|booking|reservation|number|no|id|confirmation)\b", " ", text)
    text = re.sub(r"[^a-z0-9]", "", text)
    if re.fullmatch(r"[0-9o]*[0-9][0-9o]*", text):  # "four four one o": a spoken zero
        text = text.replace("o", "0")
    return text


def _canon_int(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        return str(int(value)) if float(value).is_integer() else str(value)
    match = re.search(r"-?\d+(?:\.\d+)?", words_to_digits(str(value)))
    if match is None:
        return str(value).strip().lower()
    number = float(match.group())
    return str(int(number)) if number.is_integer() else str(number)


def _canon_date(value: Any, year: int) -> str:
    text = words_to_digits(str(value)).strip().lower()
    iso = re.search(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", text)
    if iso:
        return f"{int(iso[1]):04d}-{int(iso[2]):02d}-{int(iso[3]):02d}"
    month = day = None
    for m in re.finditer(r"[a-z]+", text):
        if m.group() in _MONTHS:
            month = _MONTHS[m.group()]
            after = re.match(r"\s*(\d{1,2})(?:st|nd|rd|th)?\b", text[m.end() :])
            before = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?\s*(?:of\s+)?$", text[: m.start()])
            if after:
                day = int(after.group(1))
            elif before:
                day = int(before.group(1))
            break
    y = re.search(r"\b(20\d\d)\b", text)
    if month is None:
        slash = re.search(r"\b(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?\b", text)
        if slash:  # US order: month/day
            month, day = int(slash[1]), int(slash[2])
    if month is None or day is None or not (1 <= month <= 12 and 1 <= day <= 31):
        return text
    return f"{int(y[1]) if y else year:04d}-{month:02d}-{day:02d}"


def _canon_time(value: Any) -> str:
    text = words_to_digits(str(value)).strip().lower().replace(".", "")
    text = text.replace("o'clock", "").replace("oclock", "")
    m = re.search(r"\b(\d{1,2})(?:[:h ](\d{2}))?\s*(am|pm|a m|p m)?\b", text)
    if m is None:
        return text
    hour, minute = int(m.group(1)), int(m.group(2) or 0)
    suffix = (m.group(3) or "").replace(" ", "")
    if "noon" in text and not suffix:
        suffix = "pm"
    if suffix == "pm" and hour < 12:
        hour += 12
    elif suffix == "am" and hour == 12:
        hour = 0
    if not (0 <= hour < 24 and 0 <= minute < 60):
        return text
    return f"{hour:02d}:{minute:02d}"


def _canon_email(value: Any) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"\s+at\s+", "@", text)
    text = re.sub(r"\s+dot\s+", ".", text)
    text = re.sub(r"\s+(underscore)\s+", "_", text)
    text = re.sub(r"\s+(dash|hyphen)\s+", "-", text)
    return re.sub(r"\s+", "", text).strip(".")


def _canon_phone(value: Any) -> str:
    return re.sub(r"\D", "", words_to_digits(str(value)))


def _canon_text(value: Any) -> str:
    words = re.findall(r"[a-z0-9]+", words_to_digits(str(value)).lower())
    return " ".join(_ABBREVIATIONS.get(w, w) for w in words)


def canonical(value: Any, kind: ParamKind, *, year: int = 2026) -> str:
    """Canonical string of an argument value (used to compare and to look records up)."""
    if value is None:
        return ""
    if kind == "id":
        return _canon_id(value)
    if kind == "int":
        return _canon_int(value)
    if kind == "date":
        return _canon_date(value, year)
    if kind == "time":
        return _canon_time(value)
    if kind == "email":
        return _canon_email(value)
    if kind == "phone":
        return _canon_phone(value)
    return _canon_text(value)  # name, text, free


# ------------------------------------------------------------------------- tools


@dataclass(frozen=True, slots=True)
class ParamSpec:
    kind: ParamKind
    description: str
    required: bool = True

    def json_type(self) -> str:
        return "integer" if self.kind == "int" else "string"


ToolImpl = Callable[[dict[str, Any], dict[str, Any]], Any]
"""``(database, canonical arguments) -> result``; raises :class:`ToolError`."""


@dataclass(frozen=True, slots=True)
class ToolDef:
    """A tool of the mock world."""

    name: str
    description: str
    params: Mapping[str, ParamSpec]
    write: bool
    run: ToolImpl
    claims: str | None = None
    """Regex: sentences in which the agent claims this (write) tool's action was done."""
    delay: float | None = None
    """Latency of this tool (s); ``None`` = the suite's ``tool_delay``."""

    def schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                name: {"type": p.json_type(), "description": p.description}
                for name, p in self.params.items()
            },
            "required": [name for name, p in self.params.items() if p.required],
        }


def _order(db: dict[str, Any], order_id: str) -> dict[str, Any]:
    orders = db.get("orders", {})
    for key, order in orders.items():
        if _canon_id(key) == order_id:
            return dict(order, order_id=key)
    raise ToolError(f"No order with number {order_id!r} was found.")


def _set_order(db: dict[str, Any], order_id: str, **changes: Any) -> None:
    for key in db.get("orders", {}):
        if _canon_id(key) == order_id:
            db["orders"][key].update(changes)
            return


def _lookup_order(db: dict[str, Any], a: dict[str, Any]) -> Any:
    order = _order(db, a["order_id"])
    return {
        k: order[k] for k in ("order_id", "status", "items", "total", "delivery_date") if k in order
    }


def _cancel_order(db: dict[str, Any], a: dict[str, Any]) -> Any:
    order = _order(db, a["order_id"])
    if order["status"] not in ("pending", "processing"):
        raise ToolError(
            f"Order {order['order_id']} is {order['status']} and can no longer be cancelled."
        )
    _set_order(db, a["order_id"], status="cancelled")
    return {"order_id": order["order_id"], "status": "cancelled", "refund": order.get("total")}


def _request_refund(db: dict[str, Any], a: dict[str, Any]) -> Any:
    order = _order(db, a["order_id"])
    if order["status"] != "delivered":
        raise ToolError(
            f"Order {order['order_id']} is {order['status']}: only delivered "
            "orders can be refunded."
        )
    if order.get("refund"):
        raise ToolError(f"A refund for order {order['order_id']} was already requested.")
    _set_order(db, a["order_id"], refund="requested")
    return {
        "order_id": order["order_id"],
        "refund": "requested",
        "amount": order.get("total"),
        "processing_days": 5,
    }


def _update_address(db: dict[str, Any], a: dict[str, Any]) -> Any:
    order = _order(db, a["order_id"])
    if order["status"] not in ("pending", "processing"):
        raise ToolError(
            f"Order {order['order_id']} is {order['status']}: the address can no longer be changed."
        )
    if not a["address"]:
        raise ToolError("The new address is empty.")
    _set_order(db, a["order_id"], address=a["address"])
    return {"order_id": order["order_id"], "address": a["address"]}


def _lookup_customer(db: dict[str, Any], a: dict[str, Any]) -> Any:
    for customer in db.get("customers", {}).values():
        if _canon_email(customer.get("email", "")) == a["email"]:
            return customer
    raise ToolError(f"No customer with the email {a['email']!r} was found.")


def _slot(db: dict[str, Any], date: str, time: str) -> str:
    return f"{date} {time}"


def _check_availability(db: dict[str, Any], a: dict[str, Any]) -> Any:
    party = int(a["party_size"]) if a["party_size"].isdigit() else 0
    if party < 1:
        raise ToolError("party_size must be a positive number.")
    slots: dict[str, int] = db.get("availability", {})
    key = _slot(db, a["date"], a["time"])
    free = slots.get(key, 0)
    alternatives = sorted(
        s.split(" ")[1] for s, n in slots.items() if s.startswith(a["date"]) and n > 0 and s != key
    )
    return {
        "date": a["date"],
        "time": a["time"],
        "party_size": party,
        "available": free > 0,
        "alternative_times": alternatives,
    }


def _book_table(db: dict[str, Any], a: dict[str, Any]) -> Any:
    party = int(a["party_size"]) if a["party_size"].isdigit() else 0
    if party < 1:
        raise ToolError("party_size must be a positive number.")
    if not a["name"]:
        raise ToolError("A name is needed for the booking.")
    slots: dict[str, int] = db.setdefault("availability", {})
    key = _slot(db, a["date"], a["time"])
    if slots.get(key, 0) <= 0:
        raise ToolError(f"No table is free on {a['date']} at {a['time']}.")
    slots[key] -= 1
    digest = hashlib.sha256(f"{a['name']}|{key}|{party}".encode()).hexdigest()
    booking_id = str(1000 + int(digest[:6], 16) % 9000)  # deterministic: same booking, same id
    db.setdefault("bookings", {})[booking_id] = {
        "name": a["name"],
        "date": a["date"],
        "time": a["time"],
        "party_size": party,
        "status": "confirmed",
    }
    return {
        "booking_id": booking_id,
        "status": "confirmed",
        "date": a["date"],
        "time": a["time"],
        "party_size": party,
    }


def _cancel_booking(db: dict[str, Any], a: dict[str, Any]) -> Any:
    bookings: dict[str, Any] = db.get("bookings", {})
    for key, booking in bookings.items():
        if _canon_id(key) == a["booking_id"]:
            if booking["status"] == "cancelled":
                raise ToolError(f"Booking {key} is already cancelled.")
            booking["status"] = "cancelled"
            slot = _slot(db, booking["date"], booking["time"])
            slots = db.setdefault("availability", {})
            slots[slot] = slots.get(slot, 0) + 1
            return {"booking_id": key, "status": "cancelled"}
    raise ToolError(f"No booking with number {a['booking_id']!r} was found.")


def _transfer_to_human(db: dict[str, Any], a: dict[str, Any]) -> Any:
    db["transfers"] = int(db.get("transfers", 0)) + 1
    return {"transferred": True, "queue_position": 1}


_ORDER_ID = ParamSpec("id", "The order number, digits only, e.g. '1042'.")
TOOL_LIBRARY: dict[str, ToolDef] = {
    t.name: t
    for t in (
        ToolDef(
            "lookup_order",
            "Look up an order: status, items, total and delivery date.",
            {"order_id": _ORDER_ID},
            write=False,
            run=_lookup_order,
        ),
        ToolDef(
            "cancel_order",
            "Cancel an order that has not shipped yet. Only call it after the customer "
            "confirmed the cancellation.",
            {
                "order_id": _ORDER_ID,
                "reason": ParamSpec("free", "Why the customer cancels.", required=False),
            },
            write=True,
            run=_cancel_order,
            claims=r"\b(has|have|was|were|is|are|been|now|successfully)\b[\w ]{0,20}\bcancel+ed\b"
            r"|\bi(?: have|'ve)? cancel+ed\b",
        ),
        ToolDef(
            "request_refund",
            "Request a refund for a delivered order.",
            {
                "order_id": _ORDER_ID,
                "reason": ParamSpec("free", "Why the customer wants a refund.", required=False),
            },
            write=True,
            run=_request_refund,
            claims=r"\brefund\b[\w ]{0,25}\b(requested|issued|processed|submitted|initiated|"
            r"started|on its way)\b|\b(requested|issued|processed|submitted|initiated|started)"
            r"\b[\w ]{0,20}\brefund\b",
            delay=2.0,
        ),
        ToolDef(
            "update_address",
            "Change the delivery address of an order that has not shipped yet.",
            {
                "order_id": _ORDER_ID,
                "address": ParamSpec("text", "The complete new delivery address."),
            },
            write=True,
            run=_update_address,
            claims=r"\baddress\b[\w ]{0,25}\b(updated|changed)\b|\b(updated|changed)\b[\w ]{0,25}"
            r"\baddress\b",
        ),
        ToolDef(
            "lookup_customer",
            "Find a customer account and its orders by email address.",
            {"email": ParamSpec("email", "The customer's email address, e.g. 'jo@example.com'.")},
            write=False,
            run=_lookup_customer,
        ),
        ToolDef(
            "check_availability",
            "Check whether a restaurant table is free at a date and time.",
            {
                "date": ParamSpec("date", "Date as YYYY-MM-DD."),
                "time": ParamSpec("time", "Time as HH:MM (24 h)."),
                "party_size": ParamSpec("int", "Number of guests."),
            },
            write=False,
            run=_check_availability,
        ),
        ToolDef(
            "book_table",
            "Book a restaurant table. Only call it when the time is available and you know "
            "the guest's name.",
            {
                "name": ParamSpec("name", "Full name of the guest."),
                "date": ParamSpec("date", "Date as YYYY-MM-DD."),
                "time": ParamSpec("time", "Time as HH:MM (24 h)."),
                "party_size": ParamSpec("int", "Number of guests."),
            },
            write=True,
            run=_book_table,
            claims=r"\b(booked|reserved)\b|\b(booking|reservation|table)\b[\w ]{0,20}\b"
            r"(is|has been) (confirmed|booked|reserved)\b",
        ),
        ToolDef(
            "cancel_booking",
            "Cancel a restaurant booking by its booking number.",
            {"booking_id": ParamSpec("id", "The booking number, digits only.")},
            write=True,
            run=_cancel_booking,
            claims=r"\b(booking|reservation)\b[\w ]{0,20}\b(has|have|is|was)\b[\w ]{0,10}"
            r"\bcancel+ed\b|\bi(?: have|'ve)? cancel+ed\b",
        ),
        ToolDef(
            "transfer_to_human",
            "Transfer the caller to a human agent. Use it when the customer asks for a human "
            "or the request is outside your tools.",
            {"reason": ParamSpec("free", "Short summary of the issue.", required=False)},
            write=True,
            run=_transfer_to_human,
            claims=r"\b(transferr(ed|ing) you|connect(ed|ing) you)\b",
        ),
    )
}
STATUS_WORDS = ("shipped", "delivered", "processing", "pending", "in transit")
"""Order facts only a tool can know: saying one that no tool returned is a hallucination."""


# ------------------------------------------------------------------------- schema


class ExpectedCall(BaseModel):
    """A tool call the agent should make (``args``: the scored arguments)."""

    model_config = ConfigDict(extra="forbid")

    name: str
    args: dict[str, Any] = Field(default_factory=dict)
    optional: bool = False
    """A reasonable call that is not required (e.g. a look-up before a cancellation): it
    is not counted as unnecessary, and missing it does not lower the recall."""
    turn: int | None = None
    """Turn index it belongs to (set from ``turns[].expect_calls``)."""


class ToolTurn(BaseModel):
    """What the caller says in one turn."""

    model_config = ConfigDict(extra="forbid")

    id: str | None = None
    text: str
    expect_calls: list[ExpectedCall] = Field(default_factory=list)
    duration: float | None = Field(default=None, gt=0)
    """Synthetic stimuli only: speech duration (s)."""


class ToolScenario(BaseModel):
    """One scripted conversation with its expected tool calls and final state."""

    model_config = ConfigDict(extra="forbid")

    id: str
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    tools: list[str] = Field(min_length=1)
    instructions: str | None = None
    """Appended to the suite's instructions."""
    database: dict[str, Any] = Field(default_factory=dict)
    """Tables merged over the suite's database (records replace records)."""
    turns: list[ToolTurn] = Field(min_length=1)
    expect_said: list[str | list[str]] = Field(default_factory=list)
    """Facts the agent must say (case-insensitive substrings; a list = any of them)."""
    expected_state: dict[str, Any] | None = None
    """Explicit expected final database (default: replay the expected write calls)."""
    tool_delays: dict[str, float] = Field(default_factory=dict)
    persona: str | None = None
    """LLM-driven caller only: who the caller is (default: a polite customer)."""
    goal: str | None = None
    """LLM-driven caller only: what the caller wants (default: ``description``; the
    scripted turns are always given as the details the caller knows)."""

    @model_serializer(mode="wrap")
    def _omit_unset_caller(
        self, handler: SerializerFunctionWrapHandler, info: SerializationInfo
    ) -> dict[str, Any]:
        # unset caller fields are left out, so suites keep the hashes they had before
        data: dict[str, Any] = handler(self)
        for key in ("persona", "goal"):
            if data.get(key) is None:
                data.pop(key, None)
        return data

    @property
    def expected_calls(self) -> list[ExpectedCall]:
        return [
            c.model_copy(update={"turn": i})
            for i, t in enumerate(self.turns)
            for c in t.expect_calls
        ]

    def turn_id(self, index: int) -> str:
        return self.turns[index].id or f"t{index}"


class ToolSuite(BaseModel):
    """A pinned set of tool-use scenarios (see the module docstring)."""

    model_config = ConfigDict(extra="forbid")

    name: str
    version: int = 1
    description: str = ""
    today: str = "2026-09-29"
    """The conversation's date (``YYYY-MM-DD``): told to the agent, used for dates
    without a year."""
    instructions: str = ""
    """Agent instructions; ``{today}`` is replaced with the long date."""
    sample_rate: int = Field(default=16_000, gt=0)
    chunk: float = Field(default=0.02, gt=0, le=0.2)
    loudness_dbfs: float | None = -20.0
    lead_in: float = Field(default=0.5, ge=0)
    stimuli: Literal["synthetic", "tts"] = "synthetic"
    tts: str | dict[str, Any] | None = None
    reply_timeout: float = Field(default=12.0, gt=0)
    gap_after_reply: float = Field(default=1.0, ge=0)
    max_reply: float = Field(default=60.0, gt=0)
    tool_delay: float = Field(default=0.3, ge=0)
    """Latency of every mock tool (s) unless the tool or scenario sets its own."""
    database: dict[str, Any] = Field(default_factory=dict)
    scenarios: list[ToolScenario] = Field(min_length=1)

    @model_validator(mode="after")
    def _check(self) -> ToolSuite:
        ids = [s.id for s in self.scenarios]
        if len(set(ids)) != len(ids):
            raise ValueError(f"scenario ids must be unique: {ids}")
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", self.today):
            raise ValueError("today must be YYYY-MM-DD")
        for s in self.scenarios:
            for name in s.tools:
                if name not in TOOL_LIBRARY:
                    raise ValueError(
                        f"{s.id}: unknown tool {name!r} (known: {', '.join(TOOL_LIBRARY)})"
                    )
            for call in s.expected_calls:
                if call.name not in s.tools:
                    raise ValueError(f"{s.id}: expected call {call.name!r} is not offered")
                unknown = set(call.args) - set(TOOL_LIBRARY[call.name].params)
                if unknown:
                    raise ValueError(f"{s.id}: {call.name} has no argument {sorted(unknown)}")
            for name in s.tool_delays:
                if name not in s.tools:
                    raise ValueError(f"{s.id}: tool_delays names an unoffered tool {name!r}")
        return self

    @property
    def year(self) -> int:
        return int(self.today[:4])

    def select(self, ids: Sequence[str] | None) -> ToolSuite:
        """A copy with only these scenarios (in suite order)."""
        if not ids:
            return self
        unknown = set(ids) - {s.id for s in self.scenarios}
        if unknown:
            raise ConfigurationError(f"unknown scenario(s): {', '.join(sorted(unknown))}")
        return self.model_copy(update={"scenarios": [s for s in self.scenarios if s.id in ids]})

    def definition_sha256(self) -> str:
        return _sha256_json(self.model_dump(mode="json"))

    def scenario_sha256(self, scenario: ToolScenario) -> str:
        return _sha256_json(
            {
                "scenario": scenario.model_dump(mode="json"),
                "database": self.initial_db(scenario),
                "instructions": self.instructions_for(scenario),
                "tool_delay": self.tool_delay,
            }
        )

    def long_date(self) -> str:
        from datetime import date

        d = date.fromisoformat(self.today)
        return f"{d:%A}, {d:%B} {d.day}, {d.year}"

    def instructions_for(self, scenario: ToolScenario) -> str:
        base = self.instructions.replace("{today}", self.long_date()).strip()
        extra = (scenario.instructions or "").strip()
        return f"{base}\n\n{extra}".strip() if extra else base

    def initial_db(self, scenario: ToolScenario) -> dict[str, Any]:
        db = copy.deepcopy(self.database)
        for table, value in scenario.database.items():
            if isinstance(value, dict) and isinstance(db.get(table), dict):
                db[table] = {**db[table], **copy.deepcopy(value)}
            else:
                db[table] = copy.deepcopy(value)
        return db

    def delay_for(self, scenario: ToolScenario, tool: str) -> float:
        if tool in scenario.tool_delays:
            return scenario.tool_delays[tool]
        own = TOOL_LIBRARY[tool].delay
        return self.tool_delay if own is None else own

    def expected_db(self, scenario: ToolScenario) -> dict[str, Any]:
        """The expected final database: ``expected_state`` or the replayed write calls."""
        if scenario.expected_state is not None:
            return copy.deepcopy(scenario.expected_state)
        db = self.initial_db(scenario)
        for call in scenario.expected_calls:
            tool = TOOL_LIBRARY[call.name]
            if not tool.write or call.optional:
                continue
            args = canonical_args(tool, call.args, year=self.year)
            try:
                tool.run(db, args)
            except ToolError as exc:
                raise ConfigurationError(
                    f"{scenario.id}: expected call {call.name}({call.args}) fails: {exc}"
                ) from exc
        return db

    def stimulus_scenario(self, scenarios: Sequence[ToolScenario] | None = None) -> Scenario:
        """The caller turns of ``scenarios`` (default: all) as one T1
        :class:`~voice_agent_next.bench.stimuli.Scenario`; turn ids are
        ``<scenario>/<turn>``."""
        turns = []
        for scenario in self.scenarios if scenarios is None else scenarios:
            for i, t in enumerate(scenario.turns):
                duration = t.duration
                if self.stimuli == "synthetic" and duration is None:
                    duration = round(min(1.2, max(0.5, len(t.text) / 25.0)), 2)
                turns.append(
                    TurnSpec(
                        id=f"{scenario.id}/{scenario.turn_id(i)}", text=t.text, duration=duration
                    )
                )
        return Scenario(
            name=self.name,
            version=self.version,
            sample_rate=self.sample_rate,
            chunk=self.chunk,
            loudness_dbfs=self.loudness_dbfs,
            lead_in=self.lead_in,
            stimuli=self.stimuli,
            tts=self.tts,
            reply_timeout=self.reply_timeout,
            gap_after_reply=self.gap_after_reply,
            max_reply=self.max_reply,
            turns=turns,
        )

    def with_caller(self, tts: str | dict[str, Any] | None) -> ToolSuite:
        """A copy whose caller speaks with ``tts`` (a TTS spec), or synthetic speech for
        ``"synthetic"``; ``None`` keeps the suite's setting."""
        if tts is None:
            return self
        if tts == "synthetic":
            return self.model_copy(update={"stimuli": "synthetic", "tts": None})
        return self.model_copy(update={"stimuli": "tts", "tts": tts})


def _sha256_json(data: Any) -> str:
    text = json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def state_hash(db: Mapping[str, Any]) -> str:
    """Hash of a database state (canonical JSON)."""
    return _sha256_json(db)


def load_tool_suite(source: str | os.PathLike[str] | Mapping[str, Any] | ToolSuite) -> ToolSuite:
    """Load a suite: a built-in name (``smoke``), a YAML/JSON file or a mapping."""
    if isinstance(source, ToolSuite):
        return source
    if isinstance(source, Mapping):
        return ToolSuite.model_validate(dict(source))
    path = Path(source)
    if not path.exists():
        if isinstance(source, str) and source in BUILTIN_TOOL_SUITES:
            path = BUILTIN_TOOL_SUITES[source]
        else:
            raise ConfigurationError(
                f"tool scenarios not found: {source} (built-in: {', '.join(BUILTIN_TOOL_SUITES)})"
            )
    text = path.read_text(encoding="utf-8")
    data = json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ConfigurationError(f"{path}: the suite root must be a mapping")
    try:
        return ToolSuite.model_validate(data)
    except ValueError as exc:
        raise ConfigurationError(f"{path}: invalid tool suite: {exc}") from exc


# ----------------------------------------------------------------------- runtime


def canonical_args(tool: ToolDef, args: Mapping[str, Any], *, year: int) -> dict[str, Any]:
    """Validate and normalize arguments for ``tool`` (raises :class:`ToolError`)."""
    out: dict[str, Any] = {}
    for name, spec in tool.params.items():
        value = args.get(name)
        if value is None or (isinstance(value, str) and not value.strip()):
            if spec.required:
                raise ToolError(f"missing required argument {name!r}")
            out[name] = None
            continue
        out[name] = (
            str(value).strip() if spec.kind == "free" else canonical(value, spec.kind, year=year)
        )
    return out


@dataclass(slots=True)
class CallRecord:
    """One tool call the agent made (``now()`` clock)."""

    name: str
    arguments: dict[str, Any]
    raw_arguments: str
    started: float
    ended: float | None = None
    ok: bool = False
    output: str = ""
    write: bool = False
    changed_state: bool = False

    def describe(self, origin: float) -> dict[str, Any]:
        return {
            "name": self.name,
            "arguments": self.arguments,
            "ok": self.ok,
            "write": self.write,
            "changed_state": self.changed_state,
            "output": self.output[:300],
            "t_s": round(self.started - origin, 3),
            "duration_ms": None
            if self.ended is None
            else round((self.ended - self.started) * 1000.0, 1),
        }


def build_tools(
    suite: ToolSuite,
    scenario: ToolScenario,
    db: dict[str, Any],
    log: list[CallRecord],
    *,
    delay_scale: float = 1.0,
) -> list[FunctionTool]:
    """The scenario's tools as :class:`FunctionTool` objects bound to ``db``; every call
    is appended to ``log``."""
    tools: list[FunctionTool] = []
    for name in scenario.tools:
        tool = TOOL_LIBRARY[name]
        tools.append(
            FunctionTool(
                name=tool.name,
                description=tool.description,
                parameters=tool.schema(),
                fn=_bind(tool, suite.delay_for(scenario, name) * delay_scale, suite.year, db, log),
            )
        )
    return tools


def _bind(
    tool: ToolDef, delay: float, year: int, db: dict[str, Any], log: list[CallRecord]
) -> Callable[..., Any]:
    async def fn(**kwargs: Any) -> Any:
        raw = json.dumps(kwargs, sort_keys=True, default=str)
        rec = CallRecord(tool.name, dict(kwargs), raw, now(), write=tool.write)
        log.append(rec)
        try:
            if delay > 0:
                await sleep_until(rec.started + delay)  # never short (Windows timers)
            before = state_hash(db)
            result = tool.run(db, canonical_args(tool, kwargs, year=year))
            rec.ok = True
            rec.changed_state = state_hash(db) != before
            rec.output = json.dumps(result, default=str)
            return result
        except ToolError as exc:
            rec.output = f"error: {exc}"
            raise
        finally:
            rec.ended = now()

    return fn


# --------------------------------------------------------------------- reference


def _reference_reply(scenario: ToolScenario, turn: int) -> str:
    """What the reference agent says after its calls of ``turn``."""
    phrases = [p if isinstance(p, str) else p[0] for p in scenario.expect_said]
    last_call_turn = max((c.turn or 0 for c in scenario.expected_calls), default=0)
    if turn == last_call_turn and phrases:
        return "Okay: " + ", ".join(phrases) + "."
    return "Okay." if turn else "Sure, I can help with that."


def reference_policy(scenario: ToolScenario) -> Callable[[ChatContext], Any]:
    """A :class:`~voice_agent_next.providers.mock.MockLLM` response script that makes
    exactly the expected calls (one per round) and then says the expected facts: the
    harness's smoke test (a perfect agent), not a capability score."""
    from ..providers.mock import MockToolCall

    calls = scenario.expected_calls

    def respond(ctx: ChatContext) -> Any:
        items = list(ctx.items)
        users = [i for i, it in enumerate(items) if getattr(it, "role", None) == "user"]
        turn = max(0, len(users) - 1)
        since = items[users[-1] + 1 :] if users else items
        made = [it.name for it in since if isinstance(it, FunctionCall)]
        for call in (c for c in calls if c.turn == turn):
            if call.name in made:
                made.remove(call.name)
                continue
            args = dict(call.args)
            for pname, spec in TOOL_LIBRARY[call.name].params.items():
                if pname not in args and spec.required:
                    args[pname] = "not given"
            return MockToolCall(call.name, args)
        if since and isinstance(since[-1], FunctionCallOutput) and since[-1].is_error:
            return "Sorry, that did not work."
        return _reference_reply(scenario, turn)

    return respond


def reference_engine(suite: ToolSuite, scenario: ToolScenario) -> Any:
    """A :class:`~voice_agent_next.providers.mock.MockEngine` that hears the scripted
    turns (scripted transcripts) and follows :func:`reference_policy`."""
    from ..providers.mock import MockEngine

    return MockEngine(
        transcripts=[t.text for t in scenario.turns],
        responses=reference_policy(scenario),
        chars_per_second=40.0,
        vad_options={"min_silence_duration": 0.3},
    )


def expected_said_ok(said: str, expect: Sequence[str | Sequence[str]]) -> list[str]:
    """The ``expect_said`` entries missing from ``said`` (their first alternative)."""
    low = said.lower()
    missing: list[str] = []
    for entry in expect:
        options = [entry] if isinstance(entry, str) else list(entry)
        if not any(o.lower() in low for o in options):
            missing.append(options[0])
    return missing


def db_diff(actual: Any, expected: Any, path: str = "", limit: int = 10) -> list[str]:
    """Paths at which two database states differ (at most ``limit``)."""
    out: list[str] = []
    if isinstance(actual, dict) and isinstance(expected, dict):
        for key in sorted(set(actual) | set(expected), key=str):
            sub = f"{path}.{key}" if path else str(key)
            if key not in actual:
                out.append(f"{sub}: missing")
            elif key not in expected:
                out.append(f"{sub}: unexpected")
            else:
                out += db_diff(actual[key], expected[key], sub, limit)
            if len(out) >= limit:
                break
    elif actual != expected:
        out.append(f"{path}: {actual!r} != {expected!r}")
    return out[:limit]
