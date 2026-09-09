---
name: A published figure looks wrong
about: A number in a document does not match what the code produces, or does not make sense
title: ''
labels: ''
---

**Which figure, and where it is quoted.**
The document and the sentence - README, FINDINGS, an ADR, a STATUS log, a chapter of the book.

**What `reproduce` says.**

```
uv run forecast-lab reproduce --only <command>
```

Paste the output. `IDENTICAL` means the figure still follows from the committed data and the
disagreement is in the prose. `DRIFTED` names which kind, and the three mean different things:
more rows means the data extended, added fields means the code records more than it did, and a
moved figure is the one worth reporting.

**What `verify` says.**

```
uv run forecast-lab verify
```

`EXTENDED` is expected after any `fetch` and is not a fault. `REVISED` or `MISSING` means the
bytes a published number came from are no longer on disk.

**Why it looks wrong.**
Including "this number cannot exist" - that is how the largest defect in this project's history
was found. A loss of 196.7% was a loss larger than the capital available to lose it, and it sat
in the README for a fortnight because it was internally consistent everywhere it appeared.
