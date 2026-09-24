"""Griffe extension: render the reST-isms of our docstrings as Markdown in the API reference.

The docstrings use Google-style sections with a few Sphinx habits:
``:class:`~voice_agent_next.engine.S2SEngine``` roles and ``::`` before literal blocks.
This rewrites them (in the docs build only) to inline code and a plain colon; the
indented block that follows already renders as code in Markdown.

Registered in ``mkdocs.yml`` (``extensions:`` of the mkdocstrings Python handler).
"""

from __future__ import annotations

import re
from typing import Any

import griffe

_ROLE = re.compile(r":(?:py:)?(?:class|meth|func|mod|attr|data|exc|obj|const|ref):`(~?)([^`]+)`")
_LITERAL = re.compile(r"(\S)::$", re.MULTILINE)
_BARE_LITERAL = re.compile(r"^(\s*)::$", re.MULTILINE)
_DOUBLE_TICKS = re.compile(r"(?<!`)``(?!`)")


def _role(match: re.Match[str]) -> str:
    short, target = match.groups()
    target = target.strip()
    if "<" in target and target.endswith(">"):  # :ref:`text <target>`
        target = target.split("<", 1)[0].strip()
    elif short:
        target = target.rsplit(".", 1)[-1]
    return f"`{target}`"


def to_markdown(text: str) -> str:
    text = _ROLE.sub(_role, text)
    text = _BARE_LITERAL.sub("", text)
    text = _LITERAL.sub(r"\1:", text)
    return _DOUBLE_TICKS.sub("`", text)


class RstToMarkdown(griffe.Extension):
    def on_instance(self, *, obj: griffe.Object, **kwargs: Any) -> None:
        if obj.docstring is not None:
            obj.docstring.value = to_markdown(obj.docstring.value)
