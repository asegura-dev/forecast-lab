---
name: Something does not work
about: A command fails, produces nothing, or does something other than what it says
title: ''
labels: ''
---

**What you ran, and what happened.**
The full command line and the output. If a command names a model it could not run, include that
line - it is deliberate and it is diagnostic.

**Environment.**

```
uv run forecast-lab --help    # confirms the install works at all
uv run python -V
```

Operating system too. Several known failures are Windows-specific: Smart App Control blocks
unsigned console scripts and the native DLLs behind LightGBM and XGBoost, and OneDrive Files
On-Demand leaves placeholder stubs where CSVs should be. Section 12 of the
[RUNBOOK](../../docs/guides/RUNBOOK-getting-started.md) covers both, with the versions measured
to work.

**Do the gates pass?**

```
uv run ruff check src tests
uv run python -m mypy --strict src tests
uv run python -m pytest -q
```
