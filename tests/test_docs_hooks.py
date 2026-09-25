"""The MkDocs build hooks (docs/hooks/generated.py): generated provider table, link rewriting.

The full site build (`mkdocs build --strict`) runs in .github/workflows/docs.yml; these
tests only need the hook module, not MkDocs.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


@pytest.fixture(scope="module")
def hooks() -> ModuleType:
    spec = importlib.util.spec_from_file_location("docs_hooks", DOCS / "hooks" / "generated.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_provider_table_lists_every_registered_provider(hooks: ModuleType) -> None:
    from voice_agent_next.registry import list_providers

    table = hooks.render_providers_table(DOCS / "providers")
    for s in list_providers():
        if s.name != "mock":
            assert f"`{s.name}`" in table, s.name
    assert "### Speech-to-speech engines" in table
    assert "[`deepgram`](deepgram.md)" in table
    assert "[`google` (`gemini`)](gemini-live.md)" in table  # engine page override
    assert "`[faster-whisper]`" in table


def test_provider_page_overrides_exist(hooks: ModuleType) -> None:
    for page in set(hooks.PAGE_OVERRIDES.values()):
        path, _, _anchor = page.partition("#")
        assert (DOCS / "providers" / path).is_file(), page


def test_links_out_of_docs_point_to_github_or_included_pages(hooks: ModuleType) -> None:
    md = (
        "[a](../../examples/web/index.html) [b](../../benchmarks/README.md#t1-metrics) "
        "[c](websocket.md) [d](https://example.com) [e](#local)"
    )
    out = hooks.rewrite_links(md, "docs/transports/websocket.md", "transports/websocket.md")
    assert f"[a]({hooks.REPO_URL}/blob/main/examples/web/index.html)" in out
    assert "[b](../benchmarks/methodology.md#t1-metrics)" in out
    assert "[c](websocket.md)" in out
    assert "[d](https://example.com)" in out
    assert "[e](#local)" in out


def test_included_file_links_are_rewritten(hooks: ModuleType) -> None:
    md = "[note](../docs/research/REPORT.md) [dir](scenarios) [c](../CONTRIBUTING.md)"
    out = hooks.rewrite_links(md, "benchmarks/README.md", "benchmarks/methodology.md")
    assert "[note](../research/REPORT.md)" in out
    assert f"[dir]({hooks.REPO_URL}/tree/main/benchmarks/scenarios)" in out
    assert "[c](../contributing.md)" in out


def test_back_to_back_citations_are_escaped(hooks: ModuleType) -> None:
    md = "text [7][8][9] and [link](x.md)\n```\na[0][1]\n```"
    out = hooks.escape_citations(md)
    assert "text [7]\\[8]\\[9] and [link](x.md)" in out
    assert "a[0][1]" in out  # fenced code is untouched


def test_rst_roles_become_markdown() -> None:
    pytest.importorskip("griffe")  # only in the `docs` dependency group
    spec = importlib.util.spec_from_file_location("griffe_rst", DOCS / "hooks" / "griffe_rst.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    text = "See :class:`~voice_agent_next.engine.S2SEngine` and :meth:`Agent.say`, ``x``::"
    assert module.to_markdown(text) == "See `S2SEngine` and `Agent.say`, `x`:"
    assert module.to_markdown("```python\ncode\n```") == "```python\ncode\n```"
