"""MkDocs hooks: content generated at build time, so it cannot go stale.

* ``<!-- providers-table -->`` renders the provider registry (the data behind
  ``van providers --json``) as Markdown tables, one per component kind. The machine-specific
  ``status`` column of the CLI is replaced by what a reader needs to make a provider ready:
  its extra and its environment variables.
* ``<!-- include: path/in/repo.md -->`` inlines a Markdown file from outside ``docs/``
  (``CONTRIBUTING.md``, ``benchmarks/README.md``, ``examples/README.md``) with its relative
  links rewritten: links into ``docs/`` become links between pages, anything else points
  at the file on GitHub.
  Links from pages to repository files outside ``docs/`` are rewritten the same way, so
  they work both on GitHub and on the site.
* Numeric citations written back to back (``[7][8]``, the research notes) would parse as
  reference-style links; the second bracket is escaped so they render as written.

Registered in ``mkdocs.yml`` (``hooks:``).
"""

from __future__ import annotations

import posixpath
import re
from pathlib import Path
from typing import Any

REPO_URL = "https://github.com/kadirnar/voice-agent-next"
ROOT = Path(__file__).resolve().parents[2]

KIND_TITLES = {
    "engine": "Speech-to-speech engines",
    "stt": "Speech-to-text (STT)",
    "llm": "Language models (LLM)",
    "tts": "Text-to-speech (TTS)",
    "vad": "Voice activity detection (VAD)",
    "turn": "Turn detection",
}
KIND_ORDER = ("engine", "stt", "llm", "tts", "vad", "turn")

# Provider pages that are not simply docs/providers/<name>.md (or <name-with-dashes>.md).
_REALTIME = "openai-realtime.md"
_OPENAI_COMPAT = "openai-compatible.md"
_SPEECH_SERVERS = "openai.md#compatible-speech-servers"
PAGE_OVERRIDES: dict[tuple[str, str], str] = {
    ("engine", "openai"): _REALTIME,
    ("engine", "azure_openai"): _REALTIME,
    ("engine", "xai"): _REALTIME,
    ("engine", "qwen_omni"): _REALTIME,
    ("engine", "vllm_realtime"): _REALTIME,
    ("engine", "speaches"): _REALTIME,
    ("engine", "localai"): _REALTIME,
    ("engine", "google"): "gemini-live.md",
    ("engine", "personaplex"): "moshi.md",
    ("llm", "azure_openai"): "openai.md#azure-openai-and-custom-clients",
    ("stt", "azure_openai"): "openai.md",
    ("tts", "azure_openai"): "openai.md",
    ("stt", "speaches"): _SPEECH_SERVERS,
    ("tts", "speaches"): _SPEECH_SERVERS,
    ("stt", "localai"): _SPEECH_SERVERS,
    ("tts", "localai"): _SPEECH_SERVERS,
    ("tts", "kokoro_fastapi"): _SPEECH_SERVERS,
    ("stt", "mlx_whisper"): "mlx.md#stt-whisper-mlx_whisper",
    ("tts", "mlx_audio"): "mlx.md#tts-mlx-audio-mlx_audio",
    ("llm", "mlx_lm"): "mlx.md#llm-mlx-lm-server-mlx_lm",
    ("llm", "vllm_omni"): f"{_OPENAI_COMPAT}#vllm-omni",
    ("llm", "dashscope"): f"{_OPENAI_COMPAT}#dashscope-qwen-omni",
    **{
        ("llm", name): _OPENAI_COMPAT
        for name in (
            "ollama",
            "llamacpp",
            "vllm",
            "lmstudio",
            "groq",
            "cerebras",
            "together",
            "openrouter",
            "deepseek",
            "fireworks",
            "sambanova",
        )
    },
}


def _provider_page(kind: str, name: str, providers_dir: Path) -> str | None:
    """Page (relative to ``docs/providers/``) documenting a provider, if there is one."""
    if (kind, name) in PAGE_OVERRIDES:
        return PAGE_OVERRIDES[(kind, name)]
    for candidate in (name, name.replace("_", "-")):
        if (providers_dir / f"{candidate}.md").is_file():
            return f"{candidate}.md"
    return None


def _cell(text: str) -> str:
    return text.replace("|", r"\|").replace("\n", " ")


def provider_rows() -> list[dict[str, Any]]:
    """The registry as ``van providers --json`` sees it, without the machine-local status."""
    from voice_agent_next.registry import list_providers

    rows = []
    for s in list_providers():
        rows.append(
            {
                "kind": s.kind,
                "name": s.name,
                "where": "local" if s.local else "cloud",
                "default_model": s.default_model or "",
                "description": s.description,
                "extra": s.extra,
                "env": list(s.env),
                "aliases": list(s.aliases),
            }
        )
    return rows


def render_providers_table(providers_dir: Path, link_prefix: str = "") -> str:
    rows = [r for r in provider_rows() if r["name"] != "mock"]  # test doubles
    names = {r["name"] for r in rows}
    out = [
        f"**{len(names)} providers, {len(rows)} components** "
        "(generated from the registry at build time).",
        "",
    ]
    for kind in KIND_ORDER:
        kind_rows = [r for r in rows if r["kind"] == kind]
        if not kind_rows:
            continue
        out += [
            f"### {KIND_TITLES[kind]}",
            "",
            "| Spec | Where | Default model | Install | Needs | Description |",
            "|---|---|---|---|---|---|",
        ]
        for r in kind_rows:
            page = _provider_page(kind, r["name"], providers_dir)
            spec = f"`{r['name']}`"
            if r["aliases"]:
                spec += " (" + ", ".join(f"`{a}`" for a in r["aliases"]) + ")"
            if page:
                spec = f"[{spec}]({link_prefix}{page})"
            model = f"`{r['default_model']}`" if r["default_model"] else "—"
            extra = f"`[{r['extra']}]`" if r["extra"] else "core"
            env = " or ".join(f"`{e}`" for e in r["env"]) if r["env"] else "—"
            out.append(
                f"| {spec} | {r['where']} | {_cell(model)} | {extra} | {env} "
                f"| {_cell(r['description'])} |"
            )
        out.append("")
    return "\n".join(out)


_LINK = re.compile(r"(!?\[[^\]]*\])\(([^)\s]+)\)")

# Repository files that are published as pages (via ``<!-- include: -->``).
INCLUDED_AS = {
    "CHANGELOG.md": "docs/changelog.md",
    "CONTRIBUTING.md": "docs/contributing.md",
    "benchmarks/README.md": "docs/benchmarks/methodology.md",
    "examples/README.md": "docs/examples.md",
}


def rewrite_links(markdown: str, source: str, page: str) -> str:
    """Make the relative links of repo file ``source`` work from docs page ``page``.

    Links that stay inside ``docs/`` point at pages; links to repository files outside it
    (``../../examples/web/index.html``) point at GitHub, or at the page that includes them.
    """
    source_dir = posixpath.dirname(source)
    page_dir = posixpath.dirname(posixpath.join("docs", page))

    def fix(match: re.Match[str]) -> str:
        label, target = match.groups()
        if re.match(r"^[a-z][a-z0-9+.-]*:", target) or target.startswith("#"):
            return match.group(0)
        path, _, anchor = target.partition("#")
        resolved = posixpath.normpath(posixpath.join(source_dir, path))
        resolved = INCLUDED_AS.get(resolved, resolved)
        suffix = f"#{anchor}" if anchor else ""
        if resolved.startswith("docs/"):
            return f"{label}({posixpath.relpath(resolved, page_dir)}{suffix})"
        kind = "tree" if (ROOT / resolved).is_dir() else "blob"
        return f"{label}({REPO_URL}/{kind}/main/{resolved}{suffix})"

    return _LINK.sub(fix, markdown)


_CITATION = re.compile(r"(?<=\])\[(\d+)\]")


def escape_citations(markdown: str) -> str:
    """``[7][8]`` -> ``[7]\\[8]`` outside fenced code blocks."""
    out, fenced = [], False
    for line in markdown.split("\n"):
        if line.lstrip().startswith(("```", "~~~")):
            fenced = not fenced
        out.append(line if fenced else _CITATION.sub(r"\\[\1]", line))
    return "\n".join(out)


_INCLUDE = re.compile(r"<!--\s*include:\s*(\S+)\s*-->")


def on_page_markdown(markdown: str, page: Any, config: Any, files: Any) -> str:
    docs_dir = Path(config["docs_dir"])
    src_uri: str = page.file.src_uri
    if "<!-- providers-table -->" in markdown:
        prefix = posixpath.relpath("providers", posixpath.dirname(src_uri) or ".") + "/"
        if prefix == "./":
            prefix = ""
        table = render_providers_table(docs_dir / "providers", prefix)
        markdown = markdown.replace("<!-- providers-table -->", table)

    def include(match: re.Match[str]) -> str:
        source = match.group(1)
        text = (ROOT / source).read_text(encoding="utf-8")
        return rewrite_links(text, source, src_uri)

    markdown = rewrite_links(markdown, f"docs/{src_uri}", src_uri)
    return escape_citations(_INCLUDE.sub(include, markdown))
