# forecast-lab

**Quantitative research on short-horizon market direction.** A symbol-agnostic pipeline that measures what an experiment can actually resolve *before* it claims an edge.

> Re-engineering of a postgraduate project (MSc in Data Science and Engineering, CUCEI, University of Guadalajara, 2025), originally carried out as a team notebook. See [Origin and scope](#origin-and-scope-of-the-re-analysis).

---

## The finding that started this

The original analysis reported a model that beat chance: **51.53% accuracy**, presented as evidence that "it is possible to build ML models that beat random" on hourly gold direction.

Its own classification report printed the answer two lines below:

```
              precision    recall  f1-score   support
    DOWN (0)     0.4955    0.3795    0.4298      1436
      UP (1)     0.5268    0.6412    0.5784      1547
    accuracy                         0.5153      2983    <- the model
                                     0.5186            <- predicting UP every time
```

With 1,547 of 2,983 test bars going up, **always predicting UP scores 51.86%**. The celebrated model loses to the most trivial baseline there is, by 0.33 points. The number that refutes it was on the same screen as the number that was believed.

That is not a story about one careless project. It is what happens when a pipeline has no baseline, no power analysis, and no separation between choosing a model and testing it. This repository rebuilds the experiment so those three things are impossible to skip.

## What this is, and is not

**It is** a reproducible measurement instrument: fetch public data, build features that survive a temporal split, and evaluate a prediction against baselines, transaction costs, and the honest question of whether the sample can detect the effect at all.

**It is not** a trading system, and it does not claim an edge. The intended outcome is a falsifiable verdict - including the verdict "this experiment cannot resolve the question", which is itself a result and is stated up front rather than discovered late.

## Why the sample size decides everything

Before modelling, two numbers are computed and pre-registered:

- **Break-even accuracy** - where a directional edge starts paying for its own costs. With a measured spread of ~1.6 bps and a mean hourly move of 13.0 bps, that is **51.92%** at best.
- **Minimum detectable effect** - the smallest edge the design can tell apart from luck.

| Validation design | Out-of-sample bars | MDE |
|---|---:|---:|
| Single 70/15/15 split | 3,478 | **52.11%** |
| Walk-forward | 11,590 | 51.16% |

The single split used by the original analysis **cannot resolve a barely-profitable edge**: its detection floor sits above the profitability threshold. No amount of model tuning fixes that; only a different validation design does. This is the kind of thing that is obvious once measured and invisible otherwise.

## Architecture

Four layers, dependencies pointing inward:

```
interfaces --+--> ingest ----+
             |               +--> contracts
             +--> research --+
```

The load-bearing rule: **`research` never reaches for data.** It receives frames and cannot open a file or call an API. A layer that can quietly re-read the world produces results that cannot be reproduced from stored inputs, and a result that cannot be reproduced cannot be falsified. This is enforced by [`tests/test_layering.py`](tests/test_layering.py), not by convention.

Details and the reasoning: [ADR-001](docs/adr/ADR-001-hexagonal-architecture.md).

## Quickstart

```bash
uv sync                    # install runtime and dev dependencies
uv run forecast-lab --help
```

Data is **not** committed - it is regenerable. Nothing else works until you fetch it:

```bash
uv run forecast-lab fetch     # public CDN, no API key, no registration
uv run forecast-lab symbols   # what is now available on disk
uv run forecast-lab verify    # does it still match the committed manifest?
```

Then put several symbols on one timeline, anchored to the one being predicted:

```bash
uv run forecast-lab align --target XAUUSD --timeframe 1H
```

It reports what had to be carried forward and how old it was, per symbol - which is the
difference between "the S&P is at 4,500" and "the S&P was at 4,500, sixteen hours ago".

To reproduce the baseline this project corrects, point `ingest` at the original
project's exports; they are read once and never feed the engine:

```bash
uv run forecast-lab ingest --from <path to the exports>
```

The three quality gates, which every change must leave green:

```bash
uv run ruff check src tests
uv run python -m mypy --strict src tests
uv run python -m pytest -q
```

## Documentation

Written as a research book: each document exists because a decision was made, and records the reasoning and the trade-off rather than just the outcome.

| | |
|---|---|
| [Engineering conventions](docs/guides/engineering-conventions.md) | The rules, and why each exists |
| [ADR-001](docs/adr/ADR-001-hexagonal-architecture.md) | The architecture, and the abstractions deliberately not built |
| [ADR-002](docs/adr/ADR-002-data-source-and-symbol-set.md) | The data source and the symbol set |
| [STATUS 2026-08-18](docs/status/STATUS-2026-08-dukascopy-probe.md) | The data probe: instrument identity confirmed, and why VIX was dropped |
| [CHANGELOG](CHANGELOG.md) | Notable changes, newest first |

## Status

Early, and honest about it. Data gets in, gets verified, and gets onto one timeline without a fabricated row; nothing is modelled yet. The order is deliberate - the original analysis went wrong before any model was fitted.

- [x] Architecture, data source and symbol set decided and recorded
- [x] Contracts, the layering guard, and the `symbols` command
- [x] Reading boundary, provenance manifest, `ingest` and `verify`
- [x] `fetch`: both offer sides, with the per-bar spread
- [x] Target-anchored alignment (the correction at the heart of the re-analysis)
- [ ] Features with an enforced stationarity policy
- [ ] Walk-forward validation, baselines, and the power analysis
- [ ] Models, the evaluation battery, and the verdict
- [ ] Dashboard

## Origin and scope of the re-analysis

The underlying question, the asset universe and the first pipeline come from a postgraduate team project completed in December 2025. That work is the **baseline** this repository reproduces and corrects; it is published with the consent of its co-author.

Everything else is later, independent work: the architecture, the nine methodological corrections, the change of data source, the power analysis, the evaluation battery, and the verdict. Where a finding contradicts the original conclusions, that is stated plainly - the point of the exercise is the correction, not the embarrassment.

## License

[MIT](LICENSE).
