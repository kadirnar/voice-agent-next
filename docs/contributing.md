<!-- include: CONTRIBUTING.md -->

## Building these docs

```bash
uv sync --group docs
uv run mkdocs serve            # live preview at http://127.0.0.1:8000
uv run mkdocs build --strict   # what CI runs: broken links and API reference warnings fail it
```

Pages live in `docs/` and the navigation in `mkdocs.yml`. The [provider index](providers/index.md)
and the API reference are generated at build time (`docs/hooks/generated.py`, mkdocstrings):
add a provider with `@register_provider` and a `docs/providers/<name>.md` page, and it
shows up in the table with a link.
