# STATUS 2026-08-18 - The Dukascopy probe: a clean data source, and the VIX costs half the sample

- **Date:** 2026-08-18
- **Question:** Can Dukascopy replace the Capital.com CSV exports as the canonical source - and
  what does the symbol set cost us in statistical resolution?
- **Method:** A read-only probe run outside the repository, in an ephemeral `uv` environment
  (`uv run --no-project --with dukascopy-python`). Nothing was installed into any project and no
  file outside the scratchpad was written. The Capital.com CSVs were read, never modified.
- **Verdict:** **GO.** Dukascopy is the same instrument (hourly-return correlation **0.9996** against
  Capital.com), on a clean UTC grid, with volume the CSVs never had. But **VIX history starts only
  in 2022-10**, and keeping it costs **28,136 hourly bars - 55% of the available sample** - to buy
  a feature whose measured correlation with the target is **-0.011**. VIX is dropped and replaced by
  causally computed realised volatility.

---

## 1. What the probe had to answer

The plan fixed five questions *before* adopting Dukascopy, so that the decision could be made on
measurement rather than on the package's README:

1. Does each of the 12 symbols exist?
2. What is the coverage and density over the modelling window?
3. How far back does hourly history reach?
4. What anchor do the bars use?
5. **The go/no-go:** does the XAU/USD hourly close match Capital.com in the overlapping window?

## 2. Instrument mapping

All twelve symbols have a counterpart. **VIX included** - it had been flagged as the doubtful one,
and it does exist; its problem turned out to be history depth, not existence.

| Our symbol | Dukascopy instrument | | Our symbol | Dukascopy instrument |
|---|---|---|---|---|
| XAUUSD | `XAU/USD` | | SPX | `E_SandP-500` |
| XAGUSD | `XAG/USD` | | NDX | `E_NQ-100` |
| EURUSD | `EUR/USD` | | VIX | `VOL.IDX/USD` |
| GBPUSD | `GBP/USD` | | DXY | `DOLLAR.IDX/USD` |
| USDJPY | `USD/JPY` | | WTIUSD | `E_Light` |
| BTCUSD | `BTC/USD` | | ETHUSD | `ETH/USD` |

## 3. The grid is clean, and it carries volume

Every bar of every symbol falls exactly on the top of the UTC hour (`timestamp % 3600 == 0` for all
twelve). That matters more than it looks: the whole anti-look-ahead argument for aligning auxiliary
symbols by forward-fill rests on the timestamp being the bar **open**, so that a bar opening at
`s <= t` closes at `s + 1h <= t + 1h` - the instant the target's own bar closes. A shifted grid would
smuggle look-ahead into every row.

The bar-count pattern also reproduces what the CSVs show: metals, indices and commodities have 23
distinct hours per day (the daily maintenance break), FX and crypto have 24.

Dukascopy additionally returns a **volume** column, which the Capital.com exports do not. That
re-opens a whole family of indicators (OBV, MFI, CMF, VWAP) that were unavailable before. Not used
yet; recorded because it changes what is possible later.

## 4. The go/no-go: is it the same instrument?

XAU/USD hourly, 2025-11-01 -> 2025-12-01, against `CAPITALCOM_XAUUSD_1H.csv`:

| Side | Common timestamps | Mean difference | Median | p95 abs | Max abs | Correlation of hourly returns |
|---|---:|---:|---:|---:|---:|---:|
| BID | **457 / 457** | -0.194 USD (-0.47 bps) | -0.155 | 0.505 | 2.911 | **0.999589** |
| ASK | **457 / 457** | +0.501 USD (+1.23 bps) | +0.465 | 0.795 | 2.915 | **0.999681** |

Three readings:

- **Perfect grid overlap** - 457 of 457 timestamps matched, so no translation layer is needed.
- **Correlation 0.9996 on hourly returns.** This is the same instrument, not a lookalike.
- Capital.com's close sits *between* Dukascopy's bid and ask, nearer the bid. The implied spread is
  **~ 0.70 USD ~ 1.6 bps** - and that is a **measured** number that replaces an assumption. The
  cost model had been sketched with 1 bp as the optimistic case and 3 bps as the realistic one;
  reality sits between them, and the break-even accuracy can now be computed from a spread we
  observed rather than one we guessed.

## 5. History depth, and the constraint nobody expected

Hourly history does not start at the same place for every symbol. Probing one week in June of each
candidate year:

| Symbol | 1999 | 2007 | 2012 | 2017 | 2020 | 2022 | 2023 |
|---|---|---|---|---|---|---|---|
| XAUUSD, EURUSD, GBPUSD, USDJPY | - | yes | yes | yes | yes | yes | yes |
| WTIUSD | - | - | yes | yes | yes | yes | yes |
| SPX, NDX | - | - | yes | yes | yes | yes | yes |
| XAGUSD | - | - | - | yes | yes | yes | yes |
| BTCUSD | - | - | - | yes | yes | yes | yes |
| ETHUSD, DXY | - | - | - | - | yes | yes | yes |
| **VIX** | - | - | - | - | - | - | **yes (from 2022-10)** |

Refined by monthly probing: **VIX starts 2022-10**, DXY and ETHUSD 2018-01, XAGUSD 2017-01.

So the symbol set silently decides the window, and the window decides the statistical resolution:

| Configuration | Bars (1H) | Test at 15% | MDE, single split | MDE, walk-forward |
|---|---:|---:|---:|---:|
| All 12 (VIX binds, from 2022-10) | 22,937 | 3,440 | **52.12%** | 51.16% |
| Without VIX (DXY binds, from 2018-01) | **51,073** | 7,660 | **51.42%** | 50.78% |

The minimum detectable effect is the accuracy a design can distinguish from chance at 80% power,
alpha = 0.05 one-sided. Against a break-even accuracy of **51.92%** (the optimistic 1 bp cost), the
all-twelve configuration **cannot resolve a barely-profitable edge on a single split**; dropping
VIX resolves it even there, and comfortably under walk-forward.

## 6. So is VIX worth 28,000 bars? Measured, not argued

Two questions had to be separated.

**Can realised volatility stand in for VIX?** Over 21,000+ common hourly bars of the Capital.com
data, comparing VIX against the annualised rolling realised volatility of the S&P 500:

| Window | Correlation, levels | Correlation, changes | Mean RV | Mean VIX |
|---|---:|---:|---:|---:|
| 24 h | +0.645 | -0.005 | 15.9 | 19.8 |
| 5 d | +0.744 | +0.005 | 16.9 | 19.8 |
| 7 d | **+0.754** | -0.000 | 17.1 | 19.8 |
| 21 d | +0.755 | +0.000 | 17.4 | 19.7 |

Partly. Realised volatility tracks the **level** of the fear regime well (+0.75) but its
**changes** are uncorrelated with VIX's (~ 0.00) - which is what you would expect, since VIX is a
forward-looking implied measure and realised volatility is its backward-looking twin. As a regime
indicator it substitutes; as a shock detector it does not.

**But the question that actually decides it is different.** What does either contribute to the
target? Over 21,156 hourly bars:

| Feature | Correlation with gold's next-hour direction |
|---|---:|
| VIX (level) | **-0.011** |
| Realised volatility (7 d) of the S&P 500 | **+0.001** |

Both are zero. Keeping VIX costs **more than half the available sample** to buy a feature whose
measured linear association with the target is 0.011.

## 7. Decision and consequences

- **Dukascopy becomes the canonical source of the engine.** The Capital.com CSVs stay as the
  historical reference, used once to reproduce the notebook's baseline.
- **VIX is dropped from the feature set.** The modelling window becomes **2018-01 -> today,
  51,073 hourly bars** - more than double what the original project had.
- **Realised volatility is computed instead**, causally, from the target and from the S&P - free,
  available across the whole window, and dependent on no external symbol.
- **The measured spread (~ 1.6 bps) replaces the assumed cost** in the break-even calculation.
- A side effect worth naming: because the data is now fetchable with one command and no API key,
  **anyone who clones the repository can reproduce every number**. The previous plan had the raw
  data gitignored with no public source, which would have made the repository unrunnable by a
  reviewer - the single worst property a portfolio piece can have.

## 8. Caveats

- **A linear correlation against a binary target is a weak test.** A tree model could exploit a
  non-linear interaction that a correlation of -0.011 cannot see. What is certain is the cost
  (28,136 bars); what is uncertain is the benefit. The decision is made on that asymmetry, not on
  proof that VIX is useless.
- **The go/no-go was measured on one month** (457 bars) of one symbol. It is decisive for
  instrument identity - a 0.9996 return correlation is not a coincidence - but coverage over the
  full window still has to be validated when the data is fetched in bulk.
- **The realised-volatility comparison used Capital.com data**, because that is where VIX history
  exists at all. It transfers to Dukascopy only insofar as both track the same underlying index.
- **Dukascopy's licensing terms for the data are not published.** The repository redistributes no
  raw data and publishes only derived statistics, which is standard practice in quantitative
  research, but the terms remain unread because they are not available to read.
- **Instrument caveat:** Dukascopy's indices and commodities are CFDs from its own liquidity pool
  (bid/ask), not the official index prints. That is the same class of instrument the Capital.com
  exports were, which is why they agree - but neither is the exchange's official series.
