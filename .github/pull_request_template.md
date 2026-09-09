**What changed, and why.**

**Which gates you ran.**

```
uv run ruff check src tests
uv run python -m mypy --strict src tests
uv run python -m pytest -q
```

**If a published figure moved.**

```
uv run forecast-lab reproduce
```

A number quoted in a document and a number the code produces are two different things, and
this repository exists because they drifted apart once. If `reproduce` reports drift, say which
kind and why it is expected - or regenerate the payloads and update every document that quotes
them **in the same commit**. Never the payload alone.

**If you added a decision rather than a change.**

Architecture, method, a trade-off that is not obvious: those get an ADR under `docs/adr/`, with
the reasoning and the cost, written in the same step rather than afterwards. A measurement gets
a STATUS log and its `--json` sidecar. See [docs/INDEX.md](../docs/INDEX.md) for what exists.
