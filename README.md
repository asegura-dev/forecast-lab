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

**And the corrected pipeline now reaches the same verdict independently.** Rebuilt end to end - labels on an explicit horizon, a purged split, features that carry no price level, every transform fitted on train alone, selection on validation - it selects **LightGBM**, the same model the original selected, and on test it scores 50.67% against a 51.31% baseline: **-0.64%**. The original reported -0.33%. Two pipelines, different data handling, the same choice and the same sign ([STATUS 2026-08-24](docs/status/STATUS-2026-08-models.md)).

![Every model against the rules it has to beat](docs/status/figures/edge-test.png)

Thirteen of eighteen configurations fall below the constant predictor; exactly one clears break-even, and it is the best of eighteen coin flips. The measured cost of the shortcut is the sharpest number this project has produced. Selecting on validation gives -0.64% on test; selecting by *looking at* test - which is what `results_df['Test_AUC'].idxmax()` does - gives **+2.00%**. Same data, same eighteen configurations, **2.64 points that hang entirely on when the test set is consulted**.

**And it computes its own version of the number rather than only quoting that one.** `forecast-lab baseline` runs the corrected pipeline end to end and reports what a model would have to beat: on the test block, always-UP scores **51.46%** with 100% recall and **0.00% specificity** - the signature of a constant wearing a model's clothes - while a seeded coin flip scores **51.70%**, above it. Break-even against the friendliest cost assumption is 51.92%; nothing available without a model reaches it. The full run is in [STATUS 2026-08-23](docs/status/STATUS-2026-08-notebook-baseline.md), which also says which figures this repository computes and which it merely quotes.

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

Then label the target, cut the timeline, and score the rules a model has to beat:

```bash
uv run forecast-lab baseline --target XAUUSD --timeframe 1H
uv run forecast-lab features --target XAUUSD --timeframe 1H --mode whole
uv run forecast-lab train    --target XAUUSD --timeframe 1H --figures docs/status/figures
```

Every command that produces a result offers `--json`, because the dashboard is going to
run these commands rather than reimplement them ([ADR-005](docs/adr/ADR-005-the-dashboard-runs-the-cli.md)).

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
| [RUNBOOK](docs/guides/RUNBOOK-getting-started.md) | **Start here to run it**: fresh clone to feature matrix, and what to do when a step fails |
| [Engineering conventions](docs/guides/engineering-conventions.md) | The rules, and why each exists |
| [ADR-001](docs/adr/ADR-001-hexagonal-architecture.md) | The architecture, and the abstractions deliberately not built |
| [ADR-002](docs/adr/ADR-002-data-source-and-symbol-set.md) | The data source and the symbol set |
| [ADR-003](docs/adr/ADR-003-target-anchored-alignment.md) | The target's bars are the timeline, and nothing may extend it |
| [ADR-004](docs/adr/ADR-004-labels-splits-and-baselines.md) | What a bar's answer is, where the boundaries fall, and what a model must beat |
| [ADR-005](docs/adr/ADR-005-the-dashboard-runs-the-cli.md) | The dashboard runs the CLI rather than reimplementing it (**Plan**) |
| [ADR-006](docs/adr/ADR-006-features-and-stationarity.md) | Indicators before alignment, and no column that carries a price |
| [ADR-007](docs/adr/ADR-007-fitting-models-without-leaking.md) | Fitting on train, selecting on validation, scoring test once |
| [ADR-008](docs/adr/ADR-008-figures-are-built-in-memory.md) | Figures are built in memory and committed as PNG |
| [STATUS 2026-08-18](docs/status/STATUS-2026-08-dukascopy-probe.md) | The data probe: instrument identity confirmed, and why VIX was dropped |
| [STATUS 2026-08-23](docs/status/STATUS-2026-08-notebook-baseline.md) | The baseline recomputed from data, and the 59% of gaps nobody had measured |
| [STATUS 2026-08-24](docs/status/STATUS-2026-08-features.md) | The feature matrix, and a guard that real data corrected twice |
| [STATUS 2026-08-24](docs/status/STATUS-2026-08-models.md) | **The result**: the same model the original chose, losing to a constant |
| [CHANGELOG](CHANGELOG.md) | Notable changes, newest first |

## Status

The pipeline runs end to end: data in, verified, onto one timeline without a fabricated row, labelled, split, turned into features that carry no price level, and fitted. What is missing is not the modelling but the **evaluation**: transaction costs, walk-forward validation, and the significance battery that turns "-0.64%" into a verdict rather than a number. The order was deliberate - the original went wrong before any model was fitted, so the corrections came first.

- [x] Architecture, data source and symbol set decided and recorded
- [x] Contracts, the layering guard, and the `symbols` command
- [x] Reading boundary, provenance manifest, `ingest` and `verify`
- [x] `fetch`: both offer sides, with the per-bar spread
- [x] Target-anchored alignment (the correction at the heart of the re-analysis)
- [x] Labels on an explicit horizon, a purged chronological split, and the three baselines
- [x] Features with an enforced stationarity policy, checked by rescaling rather than by name
- [ ] Walk-forward validation, the cost model, and the power analysis
- [x] Models: six estimators, PCA, selection on validation, scored against the baselines
- [ ] The evaluation battery, the cost model, and the verdict
- [ ] Dashboard (decided in [ADR-005](docs/adr/ADR-005-the-dashboard-runs-the-cli.md), built last)

## Origin and scope of the re-analysis

The underlying question, the asset universe and the first pipeline come from a postgraduate team project completed in December 2025. That work is the **baseline** this repository reproduces and corrects; it is published with the consent of its co-author.

Everything else is later, independent work: the architecture, the nine methodological corrections, the change of data source, the power analysis, the evaluation battery, and the verdict. Where a finding contradicts the original conclusions, that is stated plainly - the point of the exercise is the correction, not the embarrassment.

## License

[MIT](LICENSE).
