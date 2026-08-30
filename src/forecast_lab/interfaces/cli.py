"""The command line, and the composition root.

The only module allowed to import concrete implementations, and the only one that
**decides** what to read and where to write. `ingest` performs the I/O - it is the layer
that knows bytes exist - but it never chooses a path on its own; this module hands it
one. The load-bearing rule, and the one a test enforces, is that **`research` never
reaches for data** (ADR-001 sec. 1): it receives frames, which is what makes every
research function callable from a test with synthetic input and no disk.

Invoke as ``forecast-lab <command>`` (in development, ``uv run forecast-lab <command>``).

Console output is deliberately plain ASCII. A scheduled task on Windows redirecting
stdout to a file under cp1252 will crash on a stray typographic character, and a crash
in the logging is indistinguishable from a crash in the work.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import pandas as pd
import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from forecast_lab.contracts import (
    ForecastLabError,
    SymbolSpec,
    Timeframe,
)
from forecast_lab.ingest import (
    DEFAULT_START,
    DEFAULT_SYMBOLS,
    NOTEWORTHY_ANCHORS,
    fetch_series,
    import_directory,
    manifest,
    read_series,
    scan,
    write_series,
)
from forecast_lab.ingest.importer import ImportReport
from forecast_lab.research import (
    DEFAULT_FLIP_RATE,
    DEFAULT_FOLDS,
    LONGEST_WINDOW,
    PCA_VARIANCE,
    Dependence,
    Design,
    Direction,
    FeatureMatrix,
    Mode,
    PolicyReport,
    PooledScore,
    ScaleVerdict,
    WalkForward,
    align_to_target,
    availability,
    bars_held,
    block_chart,
    break_even,
    build_features,
    by_year,
    calibration_chart,
    confusion,
    confusion_grid,
    correlation_comparison,
    correlation_pairs,
    count_moves,
    describe,
    design,
    edge_chart,
    evaluate_baselines,
    evaluate_stationarity,
    feature_correlation_chart,
    feature_correlations,
    fit_and_predict,
    indicators,
    label_direction,
    lag_one_autocorrelation,
    measure_dependence,
    multicollinear_pairs,
    normality,
    positive_rate,
    price_overview,
    probe_scale,
    qq_points,
    redundancy_chart,
    required_sample,
    returns_overview,
    roc_chart,
    roc_points,
    score_model,
    score_walk_forward,
    summarise_spread,
    temporal_split,
    variance_chart,
    walk_forward,
    yearly_overview,
)

app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()

#: Everything the manifest describes lives under here, and every recorded path is
#: relative to it - so `reference/XAUUSD_1H.csv` and `raw/XAUUSD_1H.csv` stay distinct.
DATA_ROOT = Path("data")
DEFAULT_DATA_DIR = DATA_ROOT / "raw"
DEFAULT_REFERENCE_DIR = DATA_ROOT / "reference"
#: Committed, unlike the data it describes: it is what ties a published number to the
#: bytes behind it (ADR-002 sec. 7).
DEFAULT_MANIFEST = Path("docs/status/data-manifest.json")
#: Figures are committed - they are derived statistics rather than vendor data, and a
#: research log whose charts only exist on the author's machine is not a research log.
DEFAULT_FIGURES = Path("docs/status/figures")
#: Break-even used when the series carries no spread column - the reference exports do
#: not, since their venue never published one. It assumes a 1 bp round trip, which is
#: optimistic: measured on the canonical data the median spread is 1.86 bps and the
#: break-even is **53.49%**. Whenever a spread is available the figure is computed from
#: it rather than taken from here (ADR-010).
FALLBACK_BREAK_EVEN = 0.5192


@app.callback()
def main() -> None:
    """Reproducible research on short-horizon market direction.

    Data is not committed to the repository - it is regenerable. Run `fetch` before
    anything else.
    """
    # This callback exists so Typer always expects a subcommand. Without it, an app
    # holding a single command collapses into a bare command, and `forecast-lab
    # symbols` fails with "unexpected extra argument". Found by running it.


@app.command("symbols")
def symbols_command(
    directory: Annotated[
        Path, typer.Option("--dir", "-d", help="Directory to scan for series files.")
    ] = DEFAULT_DATA_DIR,
    show_skipped: Annotated[
        bool, typer.Option("--show-skipped", help="Also list files that were not recognised.")
    ] = False,
    manifest_path: Annotated[
        Path, typer.Option("--manifest", help="Manifest to read row counts from.")
    ] = DEFAULT_MANIFEST,
) -> None:
    """List the series available on disk, without reading them."""
    # File names only. "What is here?" is the cheap question and is kept apart from
    # "what is inside?", so this stays instant on a directory of any size. Row counts
    # come from the manifest when one exists - already computed, nothing re-read.
    try:
        report = scan(directory)
    except NotADirectoryError:
        console.print(f"[red]No such directory:[/red] {directory}")
        console.print("Run [bold]forecast-lab fetch[/bold] first, or pass --dir.")
        raise typer.Exit(code=2) from None
    except ForecastLabError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None

    if not report.series:
        console.print(f"No series found in {directory}.")
        console.print("Run [bold]forecast-lab fetch[/bold] to download them.")
        raise typer.Exit(code=1)

    rows = _manifest_rows(manifest_path, directory)

    console.print(f"Series in [bold]{directory}[/bold]")
    table = Table()
    table.add_column("Symbol")
    for timeframe in ("1H", "4H", "1D"):
        table.add_column(timeframe, justify="right")

    available = {s.key for s in report.series}
    for symbol in report.symbols:
        cells = []
        for tf in ("1H", "4H", "1D"):
            if (symbol.name, tf) not in available:
                cells.append("-")
            else:
                count = rows.get(f"{symbol.name}_{tf}.csv")
                cells.append(f"{count:,}" if count is not None else "yes")
        table.add_row(symbol.name, *cells)

    console.print(table)
    console.print(f"{len(report.symbols)} symbols, {len(report.series)} series files.")
    if not rows:
        console.print("[dim]Row counts appear once a manifest exists (forecast-lab ingest).[/dim]")

    # Skipped files are surfaced, never swallowed: a scan that quietly ignores half a
    # directory is how somebody models a subset they did not choose.
    if report.skipped:
        console.print(f"[yellow]{len(report.skipped)} file(s) ignored.[/yellow]", end=" ")
        if show_skipped:
            console.print("")
            for path, reason in report.skipped:
                console.print(f"  [dim]{path.name}: {reason}[/dim]")
        else:
            console.print("[dim]Use --show-skipped to list them.[/dim]")


@app.command("fetch")
def fetch_command(
    symbols: Annotated[
        str,
        typer.Option("--symbols", "-s", help="Comma-separated, or 'all' for every mapped symbol."),
    ] = ",".join(DEFAULT_SYMBOLS),
    timeframe: Annotated[
        str, typer.Option("--timeframe", "-t", help="Bar interval: 1H, 4H or 1D.")
    ] = "1H",
    start: Annotated[
        str, typer.Option("--from", help="First bar, as YYYY-MM-DD.")
    ] = DEFAULT_START.strftime("%Y-%m-%d"),
    end: Annotated[
        str, typer.Option("--to", help="Last bar, as YYYY-MM-DD. Defaults to now.")
    ] = "",
    destination: Annotated[
        Path, typer.Option("--dir", "-d", help="Where to write the series.")
    ] = DEFAULT_DATA_DIR,
    manifest_path: Annotated[
        Path, typer.Option("--manifest", help="Manifest to update.")
    ] = DEFAULT_MANIFEST,
) -> None:
    """Download bars from the public venue: both sides, mid prices, and the spread."""
    # Both sides are downloaded because the spread IS the transaction cost, and the
    # break-even accuracy that decides whether any edge is worth having is computed from
    # it. A constant would hide the variation that matters (ADR-002 sec. 2).
    try:
        interval = Timeframe.parse(timeframe)
        wanted = _requested_symbols(symbols)
        first = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=UTC)
        last = datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=UTC) if end else datetime.now(UTC)
    except (ForecastLabError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from None

    console.print(
        f"Fetching {len(wanted)} symbol(s) at {interval.value}, "
        f"{first:%Y-%m-%d} to {last:%Y-%m-%d}. Both sides, so this takes a while."
    )

    fresh: list[manifest.Entry] = []
    failed: list[tuple[str, str]] = []
    for symbol in wanted:
        try:
            fetched = fetch_series(symbol, interval, first, last)
            path = write_series(fetched, destination)
            # Read back what was just written: a series that cannot survive its own
            # reader has no business in the manifest.
            _, meta = read_series(path, symbol, interval)
        except ForecastLabError as exc:
            # One bad response must not cost the whole download (ADR-002 sec. 1).
            failed.append((symbol.name, str(exc)))
            console.print(f"  [red]{symbol.name}[/red]: {exc}")
            continue
        fresh.append(manifest.entry_for(path, DATA_ROOT, meta, source="dukascopy"))
        span = f"{meta.first:%Y-%m-%d} to {meta.last:%Y-%m-%d}"
        console.print(f"  {symbol.name:8s} {meta.rows:>7,} bars  {span}")

    if not fresh:
        console.print("[red]Nothing fetched.[/red]")
        raise typer.Exit(code=1)

    merged = _merge_entries(manifest_path, fresh)
    manifest.write(manifest_path, merged)
    total = sum(e.rows for e in fresh)
    console.print(f"Wrote [bold]{len(fresh)}[/bold] series ({total:,} bars) to {destination}.")
    console.print(f"Manifest updated ({len(merged)} entries).")

    if failed:
        console.print(f"[yellow]{len(failed)} symbol(s) failed - see above.[/yellow]")
        raise typer.Exit(code=1)


def _requested_symbols(raw: str) -> list[SymbolSpec]:
    """Parse the --symbols option into validated symbols."""
    names = DEFAULT_SYMBOLS if raw.strip().lower() == "all" else raw.split(",")
    return [SymbolSpec(name=n.strip().upper()) for n in names if n.strip()]


@app.command("align")
def align_command(
    target: Annotated[
        str, typer.Option("--target", "-T", help="Symbol whose bars define the timeline.")
    ],
    timeframe: Annotated[
        str, typer.Option("--timeframe", "-t", help="Bar interval: 1H, 4H or 1D.")
    ] = "1H",
    directory: Annotated[
        Path, typer.Option("--dir", "-d", help="Directory holding the series.")
    ] = DEFAULT_DATA_DIR,
    max_staleness: Annotated[
        int,
        typer.Option("--max-staleness", help="Drop a carried value older than this, in seconds."),
    ] = 0,
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit machine-readable output instead of a table.")
    ] = False,
) -> None:
    """Put every symbol on the target's timeline, and report what had to be carried."""
    # The correction at the centre of the re-analysis: the target's own bars are the
    # timeline, so no row exists that the target did not trade. The original pipeline
    # outer-joined and forward-filled, which invented 1,251 gold bars and flipped which
    # class was the majority.
    try:
        interval = Timeframe.parse(timeframe)
        spec = SymbolSpec(name=target.upper())
        catalogue = scan(directory)
    except (ForecastLabError, NotADirectoryError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from None

    wanted = [s for s in catalogue.series if s.timeframe is interval]
    if not any(s.symbol == spec for s in wanted):
        console.print(f"[red]No {interval.value} series for {spec} in {directory}.[/red]")
        raise typer.Exit(code=1)

    # The CLI reads the disk; research receives frames and never learns where they came
    # from (ADR-001 sec. 1).
    series = {}
    for item in wanted:
        frame, _ = read_series(item.path, item.symbol, item.timeframe)
        series[item.symbol.name] = frame

    try:
        panel = align_to_target(
            spec.name,
            series,
            interval,
            max_staleness_seconds=max_staleness or None,
        )
    except ForecastLabError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None

    if as_json:
        console.print_json(data=_align_payload(panel, interval, max_staleness))
        return

    console.print(
        f"Timeline: [bold]{panel.rows:,}[/bold] bars of {panel.target} at {interval.value}, "
        f"{panel.frame.index[0]:%Y-%m-%d} to {panel.frame.index[-1]:%Y-%m-%d}"
    )

    table = Table()
    table.add_column("Symbol")
    table.add_column("Missing", justify="right")
    table.add_column("Carried", justify="right")
    table.add_column("Oldest", justify="right")
    for cover in panel.coverage:
        oldest = f"{cover.max_stale_seconds / 3600:.0f}h" if cover.max_stale_seconds else "-"
        table.add_row(
            cover.symbol,
            f"{cover.missing:,} ({cover.missing_fraction:.1%})" if cover.missing else "-",
            f"{cover.stale:,} ({cover.stale_fraction:.1%})" if cover.stale else "-",
            oldest,
        )
    console.print(table)
    console.print(
        f"{len(panel.frame.columns)} columns. "
        f"No row exists that {panel.target} did not trade."
    )


@app.command("features")
def features_command(
    target: Annotated[
        str, typer.Option("--target", "-T", help="Symbol whose bars define the timeline.")
    ],
    timeframe: Annotated[
        str, typer.Option("--timeframe", "-t", help="Bar interval: 1H, 4H or 1D.")
    ] = "1H",
    mode: Annotated[
        str, typer.Option("--mode", "-m", help="focus (target only) or whole (every symbol).")
    ] = "focus",
    directory: Annotated[
        Path, typer.Option("--dir", "-d", help="Directory holding the series.")
    ] = DEFAULT_REFERENCE_DIR,
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit machine-readable output instead of a table.")
    ] = False,
) -> None:
    """Build the feature matrix and report what each column is made of."""
    # Two things are being demonstrated, not just computed. That indicators are built on
    # each symbol's native grid before any reindexing - the other order manufactures zero
    # returns on stale rows, and staleness tracks the hour of day, so a tree learns a
    # session clock. And that no column is a price level, checked by rescaling the input
    # prices and observing which columns move rather than by trusting their names.
    try:
        interval = Timeframe.parse(timeframe)
        spec = SymbolSpec(name=target.upper())
        selected = Mode(mode.lower())
        catalogue = scan(directory)
    except ValueError:
        console.print(f"[red]Unknown mode {mode!r}: expected 'focus' or 'whole'.[/red]")
        raise typer.Exit(code=2) from None
    except (ForecastLabError, NotADirectoryError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from None

    wanted = [s for s in catalogue.series if s.timeframe is interval]
    if not any(s.symbol == spec for s in wanted):
        console.print(f"[red]No {interval.value} series for {spec} in {directory}.[/red]")
        raise typer.Exit(code=1)

    series = {}
    for item in wanted:
        frame, _ = read_series(item.path, item.symbol, item.timeframe)
        series[item.symbol.name] = frame

    try:
        matrix = build_features(spec.name, series, interval, mode=selected)
        # The probe rebuilds the target's own indicators on prices multiplied by ten.
        # Only the target's are probed: the verdict is a property of the indicator, not
        # of the symbol, and rebuilding the whole panel would cost a second run of the
        # alignment to learn nothing extra.
        verdicts = probe_scale(indicators, series[spec.name])
        policy = evaluate_stationarity(matrix.frame, _prefixed(verdicts, matrix))
    except ForecastLabError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None

    if as_json:
        console.print_json(data=_features_payload(matrix, policy))
        return

    console.print(
        f"[bold]{matrix.target}[/bold] at {interval.value}, mode {selected.value}: "
        f"{matrix.rows:,} rows x {matrix.columns} columns "
        f"from {len(matrix.symbols)} symbol(s)"
    )
    console.print(
        f"[dim]{matrix.warmup_dropped:,} warm-up rows dropped: the longest window is "
        f"{LONGEST_WINDOW} bars, and a row without a full window behind it would report "
        "a value computed from less history than it claims.[/dim]"
    )

    table = Table()
    table.add_column("Column")
    table.add_column("Scale")
    table.add_column("Moved by (p99)", justify="right")
    table.add_column("Worst row", justify="right")
    table.add_column("ADF p", justify="right")
    table.add_column("KPSS p", justify="right")
    for column in policy.columns:
        adf = f"{column.adf_pvalue:.3f}" if column.adf_pvalue is not None else "-"
        kp = f"{column.kpss_pvalue:.3f}" if column.kpss_pvalue is not None else "-"
        style = "red" if not column.allowed else ""
        table.add_row(
            column.name,
            column.scale.value,
            f"{column.relative_change:.1e}",
            f"{column.worst_change:.1e}",
            adf,
            kp,
            style=style,
        )
    console.print(table)

    if policy.passes:
        console.print(
            "[green]No column is a price level.[/green] Every one was rebuilt on prices "
            "multiplied by ten and came back unchanged."
        )
    else:
        names = ", ".join(c.name for c in policy.violations)
        console.print(
            f"[red]{len(policy.violations)} column(s) scale with the price: {names}.[/red] "
            "The test block would sit outside the training data's support."
        )
        raise typer.Exit(code=1)

    console.print(
        f"[dim]ADF and KPSS are reported, never decisive. At n={policy.columns[0].n_finite:,} "
        "the ADF rejects a unit root on almost anything, and both are invalid under the "
        "heteroskedasticity and regime change that characterise this data. What settles "
        "the question is distribution shift between blocks, which needs the splits.[/dim]"
    )
    if policy.disagreements:
        console.print(
            f"[yellow]{len(policy.disagreements)} column(s) where ADF and KPSS point opposite "
            "ways.[/yellow] That is the honest outcome for a series that is neither clearly "
            "stationary nor clearly a random walk."
        )


def _prefixed(
    verdicts: dict[str, ScaleVerdict], matrix: FeatureMatrix
) -> dict[str, ScaleVerdict]:
    """Map the probe's bare column names onto the panel's namespaced ones.

    The probe runs on one symbol's indicators, so it yields `rsi_14`; the panel calls the
    same column `XAUUSD_rsi_14` and repeats it per symbol. The verdict is a property of
    the indicator, so it applies to every symbol's copy of it.
    """
    out: dict[str, ScaleVerdict] = {}
    for column in matrix.frame.columns:
        name = str(column)
        for bare, verdict in verdicts.items():
            if name.endswith(f"_{bare}"):
                out[name] = verdict
                break
    return out


def _features_payload(matrix: FeatureMatrix, policy: PolicyReport) -> dict[str, Any]:
    """The same result, shaped for a machine (ADR-005)."""
    return {
        "target": matrix.target,
        "timeframe": matrix.timeframe.value,
        "mode": matrix.mode.value,
        "symbols": list(matrix.symbols),
        "rows": matrix.rows,
        "columns": matrix.columns,
        "warmup_dropped": matrix.warmup_dropped,
        "first": matrix.frame.index[0].isoformat() if matrix.rows else None,
        "last": matrix.frame.index[-1].isoformat() if matrix.rows else None,
        "policy_passes": policy.passes,
        "violations": [c.name for c in policy.violations],
        "features": [
            {
                "name": c.name,
                "scale": c.scale.value,
                "relative_change": c.relative_change,
                "worst_change": c.worst_change,
                "adf_pvalue": c.adf_pvalue,
                "kpss_pvalue": c.kpss_pvalue,
                "n_finite": c.n_finite,
            }
            for c in policy.columns
        ],
    }


@app.command("explore")
def explore_command(
    target: Annotated[
        str, typer.Option("--target", "-T", help="Symbol to describe.")
    ],
    timeframe: Annotated[
        str, typer.Option("--timeframe", "-t", help="Bar interval: 1H, 4H or 1D.")
    ] = "1H",
    directory: Annotated[
        Path, typer.Option("--dir", "-d", help="Directory holding the series.")
    ] = DEFAULT_REFERENCE_DIR,
    figures: Annotated[
        Path | None, typer.Option("--figures", help="Write charts to this directory as PNG.")
    ] = None,
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit machine-readable output instead of a table.")
    ] = False,
) -> None:
    """Describe the data before modelling it, and test what is worth testing."""
    # The original project's EDA ran three correct procedures on questions that did not
    # need asking: a normality test on the price rather than on returns, a t-test between
    # groups defined by the very variable being compared, and correlations on price
    # levels. Each produced a confident number that licensed nothing (ADR-009).
    try:
        interval = Timeframe.parse(timeframe)
        spec = SymbolSpec(name=target.upper())
        catalogue = scan(directory)
    except (ForecastLabError, NotADirectoryError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from None

    wanted = [s for s in catalogue.series if s.timeframe is interval]
    if not any(s.symbol == spec for s in wanted):
        console.print(f"[red]No {interval.value} series for {spec} in {directory}.[/red]")
        raise typer.Exit(code=1)

    series = {}
    for item in wanted:
        frame, _ = read_series(item.path, item.symbol, item.timeframe)
        series[item.symbol.name] = frame

    close = series[spec.name]["close"]
    returns = close.pct_change()

    try:
        price = describe(close, name=f"{spec.name} close")
        moves = count_moves(returns)
        price_normality = normality(close, name="price")
        return_normality = normality(returns, name="returns")
        yearly = by_year(close)
        closes = pd.DataFrame({name: frame["close"] for name, frame in series.items()})
        correlations = correlation_pairs(closes, target=spec.name)
    except ForecastLabError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None

    if as_json:
        console.print_json(
            data=_explore_payload(
                spec.name, interval, price, moves, price_normality,
                return_normality, yearly, correlations,
            )
        )
        return

    console.print(
        f"[bold]{spec.name}[/bold] at {interval.value}: {price.count:,} bars, "
        f"{close.index[0]:%Y-%m-%d} to {close.index[-1]:%Y-%m-%d}"
    )

    summary = Table(title="Price", title_justify="left")
    summary.add_column("Statistic")
    summary.add_column("Value", justify="right")
    for label, value in (
        ("mean", f"{price.mean:,.2f}"),
        ("median", f"{price.median:,.2f}"),
        ("std", f"{price.std:,.2f}"),
        ("range", f"{price.range:,.2f}"),
        ("IQR", f"{price.iqr:,.2f}"),
        ("skewness", f"{price.skewness:.4f}"),
        ("kurtosis", f"{price.kurtosis:.4f}"),
        ("minimum", f"{price.minimum:,.2f}"),
        ("maximum", f"{price.maximum:,.2f}"),
        ("total change", f"{price.total_change:+,.2f} ({price.total_change_pct:+.2f}%)"),
    ):
        summary.add_row(label, value)
    console.print(summary)
    if price.minimum_at is not None and price.maximum_at is not None:
        console.print(
            f"[dim]Low {price.minimum:,.2f} on {price.minimum_at:%Y-%m-%d}, "
            f"high {price.maximum:,.2f} on {price.maximum_at:%Y-%m-%d}.[/dim]"
        )

    console.print(
        f"\nHourly moves: [bold]{moves.up:,}[/bold] up, [bold]{moves.down:,}[/bold] down, "
        f"{moves.flat:,} flat - [bold]{moves.up_share:.2%}[/bold] of decided bars rose."
    )

    # The whole point of running the test twice.
    console.print("\n[bold]Normality (D'Agostino-Pearson)[/bold]")
    for test in (price_normality, return_normality):
        verdict = "rejects" if test.rejects_normality else "does not reject"
        console.print(
            f"  {test.name:9s} statistic {test.statistic:>10,.1f}  p = {test.p_value:.3e}  "
            f"-> {verdict} normality"
        )
    console.print(
        "[dim]The original ran this on the price alone and reported \"not normal\". Any "
        "trending series fails it, and nothing follows: a price is not what a model here "
        "consumes. On returns the rejection has a consequence - fat tails are why a "
        "Sharpe ratio's textbook confidence interval is wrong on this data.[/dim]"
    )

    years = Table(title="By year", title_justify="left")
    years.add_column("Year")
    years.add_column("Bars", justify="right")
    years.add_column("Mean price", justify="right")
    years.add_column("Volatility", justify="right")
    years.add_column("Range", justify="right")
    years.add_column("Rose", justify="right")
    for year, row in yearly.iterrows():
        years.add_row(
            str(year), f"{int(row['bars']):,}", f"{row['mean_price']:,.0f}",
            f"{row['volatility']:.1%}", f"{row['range']:,.0f}", f"{row['up_share']:.1%}",
        )
    console.print(years)
    spread = float(yearly["up_share"].max() - yearly["up_share"].min())
    console.print(
        f"[yellow]The share of rising bars moves {spread:.2%} across years.[/yellow] "
        "Which years land in the test block therefore decides part of any accuracy "
        "measured on it - the oracle gap of ADR-004 sec. 5, seen from the data's side."
    )

    if len(correlations):
        table = Table(title=f"Correlation with {spec.name}", title_justify="left")
        table.add_column("Symbol")
        table.add_column("On levels", justify="right")
        table.add_column("On returns", justify="right")
        table.add_column("Inflated by", justify="right")
        for symbol, row in correlations.iterrows():
            table.add_row(
                str(symbol), f"{row['on_levels']:+.3f}", f"{row['on_returns']:+.3f}",
                f"{row['inflation']:+.3f}",
            )
        console.print(table)
        console.print(
            "[dim]Two series that both trend upward correlate on levels whatever their "
            "relationship. The returns column is what a model at this horizon sees, and "
            "the gap between the two is how much of the original's EDA was trend.[/dim]"
        )

    if figures is not None:
        try:
            written = _write_eda_figures(figures, spec.name, close, returns, yearly, closes)
        except (ForecastLabError, OSError) as exc:
            console.print(f"[red]Could not write figures: {exc}[/red]")
            raise typer.Exit(code=1) from None
        console.print(f"\n{len(written)} figure(s) written to [bold]{figures}[/bold]")


def _write_eda_figures(
    destination: Path, symbol: str, close: pd.Series, returns: pd.Series,
    yearly: pd.DataFrame, closes: pd.DataFrame,
) -> list[Path]:
    """Build and write the exploratory figures. The only place they touch disk."""
    import matplotlib.pyplot as plt

    destination.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    figures = {
        "eda-price": price_overview(close, qq_points(close), symbol=symbol),
        "eda-returns": returns_overview(returns, qq_points(returns), symbol=symbol),
        "eda-yearly": yearly_overview(yearly, symbol=symbol),
    }
    if len(closes.columns) > 1:
        figures["eda-correlations"] = correlation_comparison(closes, target=symbol)

    for name, figure in figures.items():
        path = destination / f"{name}.png"
        figure.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(figure)
        written.append(path)
    return written


def _explore_payload(
    symbol: str, interval: Timeframe, price: Any, moves: Any, price_normality: Any,
    return_normality: Any, yearly: pd.DataFrame, correlations: pd.DataFrame,
) -> dict[str, Any]:
    """The same result, shaped for a machine (ADR-005)."""
    return {
        "symbol": symbol,
        "timeframe": interval.value,
        "price": {
            "count": price.count,
            "mean": price.mean,
            "median": price.median,
            "mode": price.mode,
            "std": price.std,
            "variance": price.variance,
            "minimum": price.minimum,
            "maximum": price.maximum,
            "range": price.range,
            "iqr": price.iqr,
            "skewness": price.skewness,
            "kurtosis": price.kurtosis,
            "total_change": price.total_change,
            "total_change_pct": price.total_change_pct,
            "minimum_at": price.minimum_at.isoformat() if price.minimum_at else None,
            "maximum_at": price.maximum_at.isoformat() if price.maximum_at else None,
        },
        "moves": {
            "up": moves.up, "down": moves.down, "flat": moves.flat,
            "up_share": moves.up_share,
        },
        "normality": {
            test.name: {
                "statistic": test.statistic,
                "p_value": test.p_value,
                "rejects": test.rejects_normality,
                "n": test.n,
            }
            for test in (price_normality, return_normality)
        },
        "by_year": {
            str(year): {k: float(v) for k, v in row.items()}
            for year, row in yearly.iterrows()
        },
        "correlations": {
            str(symbol_name): {k: float(v) for k, v in row.items()}
            for symbol_name, row in correlations.iterrows()
        },
    }


@app.command("train")
def train_command(
    target: Annotated[
        str, typer.Option("--target", "-T", help="Symbol whose direction is predicted.")
    ],
    timeframe: Annotated[
        str, typer.Option("--timeframe", "-t", help="Bar interval: 1H, 4H or 1D.")
    ] = "1H",
    horizon: Annotated[int, typer.Option("--horizon", help="Bars ahead to predict.")] = 1,
    mode: Annotated[
        str, typer.Option("--mode", "-m", help="focus (target only) or whole (every symbol).")
    ] = "focus",
    directory: Annotated[
        Path, typer.Option("--dir", "-d", help="Directory holding the series.")
    ] = DEFAULT_REFERENCE_DIR,
    pca: Annotated[
        bool, typer.Option("--pca/--no-pca", help="Also fit on PCA at 95% and 90% variance.")
    ] = True,
    figures: Annotated[
        Path | None,
        typer.Option("--figures", help="Write charts to this directory as PNG."),
    ] = None,
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit machine-readable output instead of a table.")
    ] = False,
) -> None:
    """Fit every model on train, select on validation, and score once on test."""
    # Three refusals, one per defect. Transforms are fitted on train alone. Selection
    # happens on validation, never on test - the original ordered by Test_Accuracy in one
    # notebook and picked with Test_AUC.idxmax() in the other, which turns the test set
    # into a hyperparameter. And every score is printed beside the baseline it has to
    # clear, because 51.53% is not a result until you know that always-UP scores 51.46%.
    try:
        interval = Timeframe.parse(timeframe)
        spec = SymbolSpec(name=target.upper())
        selected = Mode(mode.lower())
        catalogue = scan(directory)
    except ValueError:
        console.print(f"[red]Unknown mode {mode!r}: expected 'focus' or 'whole'.[/red]")
        raise typer.Exit(code=2) from None
    except (ForecastLabError, NotADirectoryError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from None

    wanted = [s for s in catalogue.series if s.timeframe is interval]
    if not any(s.symbol == spec for s in wanted):
        console.print(f"[red]No {interval.value} series for {spec} in {directory}.[/red]")
        raise typer.Exit(code=1)

    series = {}
    for item in wanted:
        frame, _ = read_series(item.path, item.symbol, item.timeframe)
        series[item.symbol.name] = frame

    try:
        matrix = build_features(spec.name, series, interval, mode=selected)
        close = series[spec.name]["close"]
        labels, label_report = label_direction(close, interval, horizon=horizon)
        # Only UP and DOWN are modelled. A FLAT bar has no direction to be right about,
        # and a model rewarded for guessing on ties is being scored on coin flips.
        direction = labels["label"].where(labels["label"] != float(Direction.FLAT.value))
        direction = direction.reindex(matrix.frame.index)

        split = temporal_split(pd.DatetimeIndex(matrix.frame.index), horizon=horizon)
        blocks = {b.name: b.index for b in split.blocks}
        baselines = {
            name: positive_rate(direction.reindex(blocks["train"]))
            for name in blocks
        }
    except ForecastLabError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None

    # The accuracy that would pay for its own costs. Computed from the venue's own
    # quoted spread when the series carries one; the reference exports do not, so those
    # runs fall back to the optimistic assumption and say so.
    threshold, threshold_source = _break_even_for(series[spec.name])

    # The baseline a constant predictor achieves on each block: it predicts train's
    # majority, so its accuracy is that class's share OF THAT BLOCK.
    train_up = baselines["train"]
    majority = 1.0 if train_up > 0.5 else 0.0
    block_baseline = {
        name: _constant_accuracy(direction.reindex(index), majority)
        for name, index in blocks.items()
    }

    models = availability()
    unavailable = [m for m in models if not m.available]

    representations: list[float | None] = [None]
    if pca:
        representations.extend(PCA_VARIANCE)

    scores: list[Any] = []
    fitted_probabilities: dict[str, dict[str, pd.Series]] = {}
    variances: dict[str, tuple[float, ...]] = {}
    failures: list[tuple[str, str]] = []
    for entry in models:
        if not entry.available:
            continue
        for variance in representations:
            try:
                fitted = fit_and_predict(
                    entry.spec, matrix.frame, direction, blocks, variance=variance
                )
            except ForecastLabError as exc:
                failures.append((entry.name, str(exc)))
                continue
            fitted_probabilities[fitted.key] = fitted.probabilities
            if fitted.explained_variance and fitted.representation not in variances:
                variances[fitted.representation] = fitted.explained_variance
            for block_name, proba in fitted.probabilities.items():
                if block_name == "train":
                    continue  # train accuracy measures memorisation, not skill
                scores.append(
                    score_model(
                        model=fitted.model,
                        representation=fitted.representation,
                        block=block_name,
                        probabilities=proba,
                        labels=direction,
                        baseline_accuracy=block_baseline[block_name],
                        components=fitted.components,
                    )
                )

    if not scores:
        console.print("[red]No model could be fitted.[/red]")
        raise typer.Exit(code=1)

    # THE selection: the best on validation, chosen before test is looked at.
    validation = [s for s in scores if s.block == "validation"]
    chosen = max(validation, key=lambda s: s.auc)
    on_test = next(
        (s for s in scores if s.block == "test" and s.key == chosen.key), None
    )

    if as_json:
        console.print_json(
            data=_train_payload(
                spec.name, interval, horizon, selected, matrix, label_report,
                split, block_baseline, scores, chosen, on_test, models,
            )
        )
        return

    console.print(
        f"[bold]{spec.name}[/bold] at {interval.value}, horizon {horizon}, mode "
        f"{selected.value}: {matrix.rows:,} rows x {matrix.columns} columns"
    )
    console.print(f"[dim]Break-even accuracy {threshold:.2%} - {threshold_source}.[/dim]")
    if unavailable:
        # Never silent. A comparison that quietly omits a model is a comparison of a
        # different experiment than the one the table claims to describe.
        for entry in unavailable:
            console.print(f"[yellow]{entry.name} not run:[/yellow] [dim]{entry.reason}[/dim]")
    for name, why in failures:
        console.print(f"[yellow]{name} failed to fit:[/yellow] [dim]{why}[/dim]")

    console.print(
        "\n[bold]Selection on validation[/bold] [dim](test is scored once, afterwards)[/dim]"
    )
    console.print(_score_table(validation, block_baseline["validation"]))

    console.print("\n[bold]Test[/bold]")
    console.print(_score_table([s for s in scores if s.block == "test"], block_baseline["test"]))

    console.print(
        f"\nSelected on validation AUC: [bold]{escape(chosen.key)}[/bold] "
        f"(AUC {chosen.auc:.4f}, accuracy {chosen.accuracy:.2%})"
    )
    if on_test is not None:
        verdict = "beats" if on_test.beats_baseline else "[red]loses to[/red]"
        console.print(
            f"On test it scores [bold]{on_test.accuracy:.2%}[/bold] against a "
            f"{on_test.baseline_accuracy:.2%} baseline - it {verdict} the constant "
            f"predictor by [bold]{on_test.edge:+.2%}[/bold]."
        )
        if on_test.accuracy < threshold:
            console.print(
                f"It does not reach the {threshold:.2%} needed to cover costs either - "
                f"short by [bold]{threshold - on_test.accuracy:.2%}[/bold]."
            )
        if on_test.specificity < 0.05:
            console.print(
                f"[yellow]Specificity {on_test.specificity:.2%} with recall "
                f"{on_test.recall:.2%}.[/yellow] That is a constant, not a model - the "
                "exact signature the original analysis reported as a result."
            )
        # The minimum detectable effect, computed from the block actually scored rather
        # than quoted from a planning document. One standard error is 0.5/sqrt(n); the
        # MDE at 80% power and a one-sided 5% test is 2.49 of them. Printing the SE
        # instead - which an earlier version of this line did - understates the design's
        # blind spot by a factor of two and a half.
        standard_error = 0.5 / math.sqrt(on_test.n)
        mde = 2.49 * standard_error
        console.print(
            f"[dim]Over {on_test.n:,} scored bars one standard error is "
            f"{standard_error:.2%}, so this design resolves {mde:.2%} at 80% power. A gap "
            f"of {abs(on_test.edge):.2%} is inside the noise either way - a fact about "
            "the design, not about the model.[/dim]"
        )

    # Two facts about the matrix that explain the results, measured rather than asserted.
    correlations = feature_correlations(matrix.frame, direction)
    redundant = multicollinear_pairs(matrix.frame)
    possible = matrix.columns * (matrix.columns - 1) // 2
    console.print(
        f"\n[dim]Strongest single feature correlates with the direction at "
        f"{correlations.abs().max():.4f}, and {len(redundant)} of {possible:,} feature "
        f"pairs exceed |0.9| - which is why PCA collapses {matrix.columns} columns into "
        "a handful.[/dim]"
    )

    if figures is not None:
        try:
            written = _write_figures(
                figures, scores, fitted_probabilities, direction, block_baseline, chosen,
                break_even_accuracy=threshold,
                matrix=matrix, correlations=correlations, redundant=redundant,
                variances=variances, blocks=blocks,
            )
        except (ForecastLabError, OSError) as exc:
            console.print(f"[red]Could not write figures: {exc}[/red]")
            raise typer.Exit(code=1) from None
        console.print(f"\n{len(written)} figure(s) written to [bold]{figures}[/bold]")


@app.command("validate")
def validate_command(
    target: Annotated[
        str, typer.Option("--target", "-T", help="Symbol whose direction is predicted.")
    ],
    timeframe: Annotated[
        str, typer.Option("--timeframe", "-t", help="Bar interval: 1H, 4H or 1D.")
    ] = "1H",
    horizon: Annotated[int, typer.Option("--horizon", help="Bars ahead to predict.")] = 1,
    mode: Annotated[
        str, typer.Option("--mode", "-m", help="focus (target only) or whole (every symbol).")
    ] = "focus",
    folds: Annotated[
        int, typer.Option("--folds", help="Walk-forward folds; each tests the block after it.")
    ] = DEFAULT_FOLDS,
    rolling: Annotated[
        bool,
        typer.Option(
            "--rolling/--expanding",
            help="Rolling trains on a fixed window; expanding on all history to date.",
        ),
    ] = False,
    pca: Annotated[
        bool,
        typer.Option("--pca/--no-pca", help="Also score PCA representations, at triple the cost."),
    ] = False,
    directory: Annotated[
        Path, typer.Option("--dir", "-d", help="Directory holding the series.")
    ] = DEFAULT_DATA_DIR,
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit machine-readable output instead of a table.")
    ] = False,
) -> None:
    """Score every model across walk-forward folds, with the power the design carries.

    `train` answers "what does this model score on one held-out block". This answers "what
    does it score across the whole history, and could the design have seen an edge worth
    having" - the second question being the one the original project never asked, and the
    one that decides whether a negative result means anything at all.
    """
    try:
        interval = Timeframe.parse(timeframe)
        spec = SymbolSpec(name=target.upper())
        selected = Mode(mode.lower())
        catalogue = scan(directory)
    except ValueError:
        console.print(f"[red]Unknown mode {mode!r}: expected 'focus' or 'whole'.[/red]")
        raise typer.Exit(code=2) from None
    except (ForecastLabError, NotADirectoryError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from None

    wanted = [s for s in catalogue.series if s.timeframe is interval]
    if not any(s.symbol == spec for s in wanted):
        console.print(f"[red]No {interval.value} series for {spec} in {directory}.[/red]")
        raise typer.Exit(code=1)

    series = {}
    for item in wanted:
        frame, _ = read_series(item.path, item.symbol, item.timeframe)
        series[item.symbol.name] = frame

    try:
        matrix = build_features(spec.name, series, interval, mode=selected)
        labels, _ = label_direction(series[spec.name]["close"], interval, horizon=horizon)
        direction = labels["label"].where(labels["label"] != float(Direction.FLAT.value))
        direction = direction.reindex(matrix.frame.index)
        index = pd.DatetimeIndex(matrix.frame.index)
        scheme = walk_forward(index, folds=folds, horizon=horizon, expanding=not rolling)
        # The comparison the whole design exists to justify, counted the same way on both
        # sides: usable rows, after the unlabelled ones are dropped.
        split = temporal_split(index, horizon=horizon)
        test_block = next(b.index for b in split.blocks if b.name == "test")
        single_rows = int(direction.reindex(test_block).notna().sum())
    except ForecastLabError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None

    # Only the provenance phrase is wanted here: the threshold itself is now per model,
    # computed from each one's measured turnover rather than from a shared assumption.
    _, threshold_source = _break_even_for(series[spec.name])
    entries = availability()

    # Off by default, and that is a measured choice rather than laziness: PCA lost every
    # single-split run, and fitting it per fold per model triples a command that already
    # takes forty seconds. ADR-011 recorded the omission as a debt; exposing it as a flag
    # discharges the debt without making every run pay for a representation that loses.
    representations: list[float | None] = [None]
    if pca:
        representations.extend(PCA_VARIANCE)

    pooled: list[PooledScore] = []
    for entry in entries:
        if not entry.available:
            continue
        for variance in representations:
            try:
                pooled.append(
                    score_walk_forward(
                        entry.spec, matrix.frame, direction, scheme, variance=variance
                    )
                )
            except ForecastLabError as exc:
                console.print(f"[yellow]{entry.name} not scored:[/yellow] [dim]{exc}[/dim]")

    if not pooled:
        console.print("[red]No model could be scored on any fold.[/red]")
        raise typer.Exit(code=1)

    # Each model gets the threshold ITS OWN turnover implies. One global break-even
    # assumes every model flips position half the time, which is what this project did
    # for a week and what none of these models actually does (ADR-012).
    thresholds = {entry.key: _break_even_at(series[spec.name], entry.flip_rate) for entry in pooled}

    # Sorted by how close each comes to paying for itself - the question - rather than by
    # edge over the baseline, which is a different one. Descending, so `pooled[0]` is the
    # model a reader has to argue with, not the one easiest to dismiss.
    pooled.sort(key=lambda p: p.accuracy - thresholds[p.key], reverse=True)
    best = pooled[0]
    plan = design(best.n, label="walk-forward")
    single = design(max(2, single_rows), label="single split")
    #: What the closest model actually has to clear, in points above a coin flip.
    gap = thresholds[best.key] - 0.5

    # The caveat every power figure in this project carried, now measured rather than
    # asserted. Computed on the model that comes closest, because that is the one whose
    # margin a reader will want an interval around.
    try:
        dependence: Dependence | None = measure_dependence(best.correct_segments)
    except ForecastLabError:
        dependence = None

    # The contrast that explains the result above: the inputs are strongly dependent and
    # the thing being averaged is not. Quoted in ADR-012 sec. 2, so it is computed here
    # rather than in a one-off script that nobody can rerun.
    scored_index = matrix.frame.index
    contrast = {
        column: lag_one_autocorrelation([matrix.frame[column].dropna().to_numpy(dtype=float)])
        for column in matrix.frame.columns
    }
    label_persistence = lag_one_autocorrelation(
        [direction.reindex(scored_index).dropna().to_numpy(dtype=float)]
    )

    if as_json:
        console.print_json(
            data=_validate_payload(
                spec.name, interval, horizon, selected, scheme, pooled, plan,
                single, thresholds, threshold_source, dependence, contrast,
                label_persistence,
            )
        )
        return

    console.print(
        f"[bold]{spec.name}[/bold] at {interval.value}, horizon {horizon}, mode "
        f"{selected.value}: {len(scheme)} "
        f"{'rolling' if rolling else 'expanding'} folds over {matrix.rows:,} rows, "
        f"{len(pooled)} configuration(s)"
    )
    console.print(
        f"[dim]{scheme.purged} bars purged ({horizon} per boundary), "
        f"{scheme.test_rows:,} scored. Costs {threshold_source}.[/dim]"
    )
    for entry in entries:
        if not entry.available:
            console.print(f"[yellow]{entry.name} not run:[/yellow] [dim]{entry.reason}[/dim]")

    console.print("\n[bold]Folds[/bold]")
    console.print(_fold_table(scheme))

    console.print("\n[bold]Pooled across folds[/bold]")
    console.print(_pooled_table(pooled, thresholds))

    console.print(
        "\n[dim]`edge` is the gap to a constant predictor refitted inside every fold; "
        "`break-even` is the accuracy each model needs to cover ITS OWN turnover. A "
        "persistent model trades less and faces a lower bar - which is why these differ "
        "by more than a point across the table, and why one global threshold (this "
        "project published 53.49% for a week) flatters the verdict.[/dim]"
    )
    console.print(
        f"\nOver {best.n:,} scored bars this design resolves "
        f"[bold]{plan.minimum_detectable_effect:.2%}[/bold] at 80% power, against "
        f"{single.minimum_detectable_effect:.2%} for a single split of the same series "
        f"({single_rows:,} bars)."
    )
    console.print(
        f"The closest model needs an edge of [bold]{gap:.2%}[/bold] over a coin flip to "
        f"pay for itself; this design would see one [bold]{plan.power_for(gap):.1%}[/bold] "
        f"of the time, from only {required_sample(gap):,} bars."
    )
    clearing = [p for p in pooled if p.accuracy >= thresholds[p.key]]
    if clearing:
        for entry_score in clearing:
            console.print(
                f"[green]{escape(entry_score.key)} clears its own break-even of "
                f"{thresholds[entry_score.key]:.2%} at {entry_score.accuracy:.2%}.[/green]"
            )
    else:
        short = thresholds[best.key] - best.accuracy
        console.print(
            f"[bold]None of the {len(pooled)} models reaches its own break-even.[/bold] The "
            f"closest is {escape(best.key)} at {best.accuracy:.2%} against "
            f"{thresholds[best.key]:.2%}, short by [bold]{short:.2%}[/bold] "
            f"({short / plan.standard_error:.2f} standard errors) while holding a position "
            f"{bars_held(best.flip_rate):.1f} bars."
        )
    if dependence is not None:
        verdict = (
            "so the standard errors above are already honest"
            if not dependence.material
            else "so every power figure above is optimistic by that factor"
        )
        console.print(
            f"[dim]Serial dependence, measured rather than assumed: a stationary bootstrap "
            f"over {dependence.n:,} bars puts the standard error at "
            f"{dependence.bootstrap_standard_error:.4%} against "
            f"{dependence.naive_standard_error:.4%} for independent draws - an inflation of "
            f"{dependence.inflation:.2f}x (lag-one autocorrelation {dependence.lag_one:+.4f}, "
            f"block length {max(dependence.blocks):.1f}), {verdict}.[/dim]"
        )


def _fold_table(scheme: WalkForward) -> Table:
    """The blocks themselves, so a reader can see that none trains on its own future."""
    table = Table(box=None, pad_edge=False)
    table.add_column("fold", justify="right")
    table.add_column("train", justify="right")
    table.add_column("test", justify="right")
    table.add_column("tests from", justify="left")
    table.add_column("to", justify="left")
    for fold in scheme:
        table.add_row(
            str(fold.number),
            f"{fold.train_rows:,}",
            f"{fold.test_rows:,}",
            fold.test[0].strftime("%Y-%m"),
            fold.test[-1].strftime("%Y-%m"),
        )
    return table


def _pooled_table(pooled: list[PooledScore], thresholds: dict[str, float]) -> Table:
    """Accuracy beside both things it has to beat: the baseline, and its own costs.

    `flip` and `held` are on the same row deliberately. A model that changes position
    every 5.7 bars faces a threshold more than a point below one that changes every 2.6,
    and putting the turnover next to the threshold is what makes that legible instead of
    looking like an unexplained difference between rows.
    """
    table = Table(box=None, pad_edge=False)
    table.add_column("model", justify="left", no_wrap=True)
    table.add_column("accuracy", justify="right")
    table.add_column("baseline", justify="right")
    table.add_column("edge", justify="right")
    table.add_column("flip", justify="right")
    table.add_column("held", justify="right")
    table.add_column("break-even", justify="right")
    table.add_column("short by", justify="right")
    for entry in pooled:
        threshold = thresholds[entry.key]
        short = entry.accuracy - threshold
        table.add_row(
            # Parentheses rather than `entry.key`: rich reads `[pca-95]` as markup and
            # eats it, which made two rows of the same model indistinguishable.
            entry.model
            if entry.representation == "raw"
            else f"{entry.model} ({entry.representation})",
            f"{entry.accuracy:.2%}",
            f"{entry.baseline_accuracy:.2%}",
            f"[{'green' if entry.edge > 0 else 'red'}]{entry.edge:+.2%}[/]",
            f"{entry.flip_rate:.1%}",
            f"{bars_held(entry.flip_rate):.1f}",
            f"{threshold:.2%}",
            f"[{'green' if short >= 0 else 'red'}]{short:+.2%}[/]",
        )
    return table


def _validate_payload(
    symbol: str,
    interval: Timeframe,
    horizon: int,
    mode: Mode,
    scheme: WalkForward,
    pooled: list[PooledScore],
    plan: Design,
    single: Design,
    thresholds: dict[str, float],
    threshold_source: str,
    dependence: Dependence | None,
    contrast: dict[str, float],
    label_persistence: float,
) -> dict[str, Any]:
    best = pooled[0]
    gap = thresholds[best.key] - 0.5
    return {
        "symbol": symbol,
        "timeframe": interval.value,
        "horizon": horizon,
        "mode": mode.value,
        "generated_at": datetime.now(UTC).isoformat(),
        "scheme": {
            "folds": len(scheme),
            "expanding": scheme.expanding,
            "purged": scheme.purged,
            "scored": scheme.test_rows,
            "blocks": [
                {
                    "fold": fold.number,
                    "train_rows": fold.train_rows,
                    "test_rows": fold.test_rows,
                    "tests_from": fold.test[0].isoformat(),
                    "tests_to": fold.test[-1].isoformat(),
                }
                for fold in scheme
            ],
        },
        "break_even": {
            "closest_model": best.key,
            "accuracy": thresholds[best.key],
            "edge_required": gap,
            "source": threshold_source,
            # Kept so a reader can see what the shared-assumption figure would have been,
            # which is what this project published before turnover was measured.
            "at_assumed_flip_rate": _break_even_at_assumed(thresholds, pooled),
        },
        # What ADR-012 sec. 2 argues from: the features carry the dependence, the
        # outcome does not, and the gap between them is why the naive standard error
        # survived.
        "serial_dependence_contrast": {
            "label": label_persistence,
            "most_autocorrelated_features": dict(
                sorted(contrast.items(), key=lambda kv: -abs(kv[1]))[:5]
            ),
            "least_autocorrelated_features": dict(
                sorted(contrast.items(), key=lambda kv: abs(kv[1]))[:3]
            ),
        },
        "dependence": None
        if dependence is None
        else {
            "n": dependence.n,
            "lag_one_autocorrelation": dependence.lag_one,
            "block_lengths": list(dependence.blocks),
            "naive_standard_error": dependence.naive_standard_error,
            "bootstrap_standard_error": dependence.bootstrap_standard_error,
            "inflation": dependence.inflation,
            "effective_sample": dependence.effective_sample,
            "material": dependence.material,
        },
        "power": {
            "walk_forward_mde": plan.minimum_detectable_effect,
            "single_split_mde": single.minimum_detectable_effect,
            "single_split_rows": single.n,
            "power_for_break_even": plan.power_for(gap),
            "bars_required": required_sample(gap),
        },
        "models": [
            {
                "model": entry.model,
                "representation": entry.representation,
                "n": entry.n,
                "accuracy": entry.accuracy,
                "baseline_accuracy": entry.baseline_accuracy,
                "edge": entry.edge,
                "flip_rate": entry.flip_rate,
                "prediction_persistence": entry.persistence,
                "bars_held": bars_held(entry.flip_rate),
                "break_even": thresholds[entry.key],
                "clears_break_even": entry.accuracy >= thresholds[entry.key],
                "folds": [
                    {
                        "fold": fold.number,
                        "n": fold.n,
                        "accuracy": fold.accuracy,
                        "baseline_accuracy": fold.baseline_accuracy,
                    }
                    for fold in entry.folds
                ],
                "skipped": [{"fold": n, "reason": why} for n, why in entry.skipped],
            }
            for entry in pooled
        ],
    }


def _align_payload(
    panel: Any, interval: Timeframe, max_staleness: int
) -> dict[str, Any]:
    """The staleness report, machine-readable.

    `align` was the last analysis command without `--json`, which ADR-005 makes a hard
    rule precisely so the dashboard can run the CLI rather than reimplement it. Carried
    here rather than deferred to Phase 4, because a rule with one standing exception is
    a convention.

    `max_staleness` is echoed back because a coverage report means something different
    under a limit than without one: with no limit nothing is dropped for being old, and
    the `missing` counts are purely history that has not started yet.
    """
    return {
        "target": panel.target,
        "timeframe": interval.value,
        "generated_at": datetime.now(UTC).isoformat(),
        "max_staleness_seconds": max_staleness or None,
        "rows": panel.rows,
        "columns": len(panel.frame.columns),
        "first": panel.frame.index[0].isoformat(),
        "last": panel.frame.index[-1].isoformat(),
        "coverage": [
            {
                "symbol": cover.symbol,
                "rows": cover.rows,
                "missing": cover.missing,
                "missing_fraction": cover.missing_fraction,
                "stale": cover.stale,
                "stale_fraction": cover.stale_fraction,
                "max_stale_seconds": cover.max_stale_seconds,
            }
            for cover in panel.coverage
        ],
    }


def _break_even_at_assumed(thresholds: dict[str, float], pooled: list[PooledScore]) -> float:
    """The single shared threshold this project published before turnover was measured.

    Kept in the payload so the correction is visible rather than silent: a reader
    comparing this run against an earlier one needs to see 53.49% and the per-model
    figures side by side. `p = 0.5 + f*c/(2*E|r|)` is linear in f, so it rescales from
    any model's own threshold instead of recomputing the spread summary.
    """
    best = pooled[0]
    return 0.5 + (thresholds[best.key] - 0.5) * (DEFAULT_FLIP_RATE / best.flip_rate)


def _break_even_at(bars: pd.DataFrame, rate: float) -> float:
    """The accuracy that pays for a strategy turning over at ``rate``.

    Separate from `_break_even_for`, which answers the question `train` asks - what does
    a generic strategy need - because that one has no prediction series to measure
    turnover from. Where predictions exist, assuming the rate is exactly the defect
    ADR-012 records.
    """
    if "spread" not in bars.columns:
        # `p = 0.5 + f*c/(2*E|r|)` is linear in f, so the assumed threshold - which is
        # quoted at f = 0.5 - rescales exactly rather than needing to be recomputed.
        return 0.5 + (FALLBACK_BREAK_EVEN - 0.5) * (rate / DEFAULT_FLIP_RATE)
    try:
        summary = summarise_spread(bars)
        return break_even(
            summary.median_bps, summary.mean_absolute_return_bps, flip_rate=rate
        ).accuracy
    except ForecastLabError:
        return FALLBACK_BREAK_EVEN


def _break_even_for(bars: pd.DataFrame) -> tuple[float, str]:
    """The break-even accuracy, measured from the venue's spread where one exists.

    Returns the figure and a phrase describing where it came from, because a threshold
    quoted without its cost assumption is the thing this project spent a week correcting:
    every accuracy here was compared against 51.92% while the measured spread sat unused
    in a downloaded column.
    """
    if "spread" not in bars.columns:
        return FALLBACK_BREAK_EVEN, "no spread in this series, so an optimistic 1 bp is assumed"
    try:
        summary = summarise_spread(bars)
        computed = break_even(summary.median_bps, summary.mean_absolute_return_bps)
    except ForecastLabError:
        return FALLBACK_BREAK_EVEN, "the spread could not be summarised; assuming 1 bp"
    return (
        computed.accuracy,
        f"from a measured median spread of {summary.median_bps:.2f} bps against a "
        f"{summary.mean_absolute_return_bps:.2f} bps average move",
    )


def _write_figures(
    destination: Path,
    scores: list[Any],
    fitted: dict[str, Any],
    direction: pd.Series,
    baselines: dict[str, float],
    chosen: Any,
    *,
    break_even_accuracy: float,
    matrix: Any,
    correlations: pd.Series,
    redundant: pd.DataFrame,
    variances: dict[str, tuple[float, ...]],
    blocks: dict[str, pd.DatetimeIndex],
) -> list[Path]:
    """Build every figure and write it. The only place a figure touches disk.

    `research/plots.py` returns `Figure` objects and never saves one - the layering guard
    treats `savefig` as I/O for exactly that reason. Deciding where output goes is the
    composition root's job.
    """
    import matplotlib.pyplot as plt

    destination.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for block in ("validation", "test"):
        block_scores = [s for s in scores if s.block == block]
        if not block_scores:
            continue

        figures = {
            f"edge-{block}": edge_chart(
                block_scores, baseline=baselines[block], break_even=break_even_accuracy,
                block=block, selected=chosen.key,
            ),
            f"roc-{block}": roc_chart(
                {
                    key: roc_points(probs[block], direction)
                    for key, probs in fitted.items()
                    if block in probs
                },
                block=block, selected=chosen.key,
            ),
            f"calibration-{block}": calibration_chart(
                {
                    key: (probs[block], direction)
                    for key, probs in fitted.items()
                    if block in probs
                },
                block=block,
            ),
            f"confusion-{block}": confusion_grid(
                {
                    key: confusion(probs[block], direction)
                    for key, probs in fitted.items()
                    if block in probs
                },
                block=block,
            ),
        }
        for name, figure in figures.items():
            path = destination / f"{name}.png"
            figure.savefig(path, dpi=140, bbox_inches="tight")
            plt.close(figure)
            written.append(path)

    # Panels about the matrix rather than about a block, so they are drawn once.
    counts = {name: len(direction.reindex(index).dropna()) for name, index in blocks.items()}
    balance = {
        name: float((direction.reindex(index).dropna() == 1.0).mean())
        for name, index in blocks.items()
    }
    possible = matrix.columns * (matrix.columns - 1) // 2
    once = {
        "feature-correlations": feature_correlation_chart(correlations),
        "feature-redundancy": redundancy_chart(redundant, total=possible),
        "blocks": block_chart(counts, balance),
    }
    if variances:
        once["pca-variance"] = variance_chart(variances, columns=matrix.columns)

    for name, figure in once.items():
        path = destination / f"{name}.png"
        figure.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(figure)
        written.append(path)

    return written


def _constant_accuracy(labels: pd.Series, majority: float) -> float:
    """What predicting `majority` every time scores on this block."""
    decided = labels.dropna()
    return float((decided == majority).mean()) if len(decided) else 0.0


def _score_table(scores: list[Any], baseline: float) -> Table:
    table = Table()
    # Truncating a long model name is legible; wrapping it onto a second row turns a
    # comparison table into a wall and hides which figure belongs to which model.
    table.add_column("Model", no_wrap=True, max_width=20)
    table.add_column("Repr.", no_wrap=True)
    table.add_column("Acc", justify="right")
    table.add_column("Edge", justify="right")
    table.add_column("AUC", justify="right")
    table.add_column("Brier", justify="right")
    table.add_column("Recall", justify="right")
    table.add_column("Spec.", justify="right")
    for s in sorted(scores, key=lambda x: x.auc, reverse=True):
        table.add_row(
            s.model,
            s.representation + (f" ({s.components})" if s.components else ""),
            f"{s.accuracy:.2%}",
            f"[{'green' if s.beats_baseline else 'red'}]{s.edge:+.2%}[/]",
            f"{s.auc:.4f}",
            f"{s.brier:.4f}",
            f"{s.recall:.2%}",
            f"{s.specificity:.2%}",
        )
    table.add_row(
        "[dim]always-UP (from train)[/dim]", "[dim]-[/dim]", f"[dim]{baseline:.2%}[/dim]",
        "[dim]0.00%[/dim]", "[dim]0.5000[/dim]", "[dim]-[/dim]",
        "[dim]100.00%[/dim]", "[dim]0.00%[/dim]",
    )
    return table


def _train_payload(
    symbol: str, interval: Timeframe, horizon: int, mode: Mode, matrix: Any,
    label_report: Any, split: Any, baselines: dict[str, float], scores: list[Any],
    chosen: Any, on_test: Any, models: tuple[Any, ...],
) -> dict[str, Any]:
    """The same result, shaped for a machine (ADR-005)."""
    return {
        "target": symbol,
        "timeframe": interval.value,
        "horizon": horizon,
        "mode": mode.value,
        "rows": matrix.rows,
        "columns": matrix.columns,
        "labels": {
            "labelled": label_report.labelled,
            "gapped": label_report.gapped,
            "flat": label_report.flat,
        },
        "blocks": {
            b.name: {"rows": b.rows, "baseline_accuracy": baselines[b.name]}
            for b in split.blocks
        },
        "models_unavailable": [
            {"name": m.name, "reason": m.reason} for m in models if not m.available
        ],
        "scores": [
            {
                "model": s.model,
                "representation": s.representation,
                "components": s.components,
                "block": s.block,
                "n": s.n,
                "accuracy": s.accuracy,
                "baseline_accuracy": s.baseline_accuracy,
                "edge": s.edge,
                "auc": s.auc,
                "brier": s.brier,
                "precision": s.precision,
                "recall": s.recall,
                "specificity": s.specificity,
                "predicted_up_rate": s.predicted_up_rate,
            }
            for s in scores
        ],
        "selected_on_validation": chosen.key,
        "test_edge": on_test.edge if on_test is not None else None,
    }


@app.command("baseline")
def baseline_command(
    target: Annotated[
        str, typer.Option("--target", "-T", help="Symbol to label and score.")
    ],
    timeframe: Annotated[
        str, typer.Option("--timeframe", "-t", help="Bar interval: 1H, 4H or 1D.")
    ] = "1H",
    horizon: Annotated[int, typer.Option("--horizon", help="Bars ahead to predict.")] = 1,
    directory: Annotated[
        Path, typer.Option("--dir", "-d", help="Directory holding the series.")
    ] = DEFAULT_REFERENCE_DIR,
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit machine-readable output instead of a table.")
    ] = False,
) -> None:
    """Score the rules a model must beat: majority class, persistence and chance."""
    # This exists because of one measurement. The original analysis reported 51.53%
    # accuracy as evidence that ML beats chance on hourly gold - while its own class
    # support showed that predicting UP every time scores 51.86%. The celebrated model
    # lost to the most trivial rule there is, and nothing was arranged to notice.
    try:
        interval = Timeframe.parse(timeframe)
        spec = SymbolSpec(name=target.upper())
        catalogue = scan(directory)
    except (ForecastLabError, NotADirectoryError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from None

    match = [s for s in catalogue.series if s.symbol == spec and s.timeframe is interval]
    if not match:
        console.print(f"[red]No {interval.value} series for {spec} in {directory}.[/red]")
        raise typer.Exit(code=1)

    frame, _ = read_series(match[0].path, spec, interval)

    try:
        labels, report = label_direction(frame["close"], interval, horizon=horizon)
        split = temporal_split(pd.DatetimeIndex(labels.index), horizon=horizon)
        blocks = {
            block.name: evaluate_baselines(
                labels.loc[split.train.index, "label"],
                labels.loc[block.index, "label"],
                block=block.name,
            )
            for block in split.blocks
        }
    except ForecastLabError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None

    if as_json:
        payload = _baseline_payload(spec.name, interval, horizon, report, split, blocks)
        console.print_json(data=payload)
        return

    console.print(
        f"[bold]{spec}[/bold] at {interval.value}, horizon {horizon}: "
        f"{report.labelled:,} of {report.rows:,} bars labelled"
    )
    console.print(
        f"[dim]{report.gapped:,} ({report.gapped_fraction:.2%}) skipped: the horizon spans a gap, "
        f"so the answer would describe a longer stretch than it claims.[/dim]"
    )
    console.print(f"[dim]{report.flat:,} flat (exact ties), kept out of the direction.[/dim]")
    if report.gapped:
        # Printed rather than discarded. Those bars are what a positional shift folds
        # silently into the headline number, and if they behave differently then any
        # metric computed over the mixture is averaging two prediction problems.
        console.print(
            f"[dim]Across those gaps price rose {report.gapped_up_rate:.2%} of the time, "
            f"against {report.up_rate:.2%} across the stated horizon.[/dim]"
        )

    table = Table()
    table.add_column("Block")
    table.add_column("Bars", justify="right")
    table.add_column("Period")
    table.add_column("UP rate", justify="right")
    for block in split.blocks:
        rep = blocks[block.name]
        span = f"{block.first:%Y-%m-%d} to {block.last:%Y-%m-%d}" if block.first else "-"
        table.add_row(block.name, f"{rep.n:,}", span, f"{rep.up_rate:.2%}")
    console.print(table)

    for name in ("validation", "test"):
        rep = blocks[name]
        scores = Table(title=f"Baselines on {name}", title_justify="left")
        scores.add_column("Rule")
        scores.add_column("Accuracy", justify="right")
        scores.add_column("Precision", justify="right")
        scores.add_column("Recall", justify="right")
        scores.add_column("Specificity", justify="right")
        for item in rep.scores:
            scores.add_row(
                item.name,
                f"{item.accuracy:.2%}",
                f"{item.precision:.2%}",
                f"{item.recall:.2%}",
                f"{item.specificity:.2%}",
            )
        console.print(scores)
        if rep.oracle_gap > 0.005:
            console.print(
                f"[yellow]The {name} block's own UP rate is {rep.up_rate:.2%} against train's "
                f"{rep.train_up_rate:.2%}.[/yellow] A rule fitted on this block instead of on "
                f"train would gain {rep.oracle_gap:.2%} for free - which is the trap the "
                "original analysis fell into."
            )


def _baseline_payload(
    symbol: str,
    interval: Timeframe,
    horizon: int,
    report: Any,
    split: Any,
    blocks: Any,
) -> dict[str, Any]:
    """The same result, shaped for a machine.

    The dashboard runs these commands rather than reimplementing them (ADR-005), so
    every command that produces a result offers it as JSON. Parsing a rendered table
    would couple the dashboard to a presentation detail, and the first column-width
    change would break it silently.
    """
    return {
        "symbol": symbol,
        "timeframe": interval.value,
        "horizon": horizon,
        "labels": {
            "rows": report.rows,
            "labelled": report.labelled,
            "up": report.up,
            "down": report.down,
            "flat": report.flat,
            "gapped": report.gapped,
            "gapped_up": report.gapped_up,
            "gapped_down": report.gapped_down,
            "up_rate": report.up_rate,
            "gapped_up_rate": report.gapped_up_rate,
        },
        "blocks": [
            {
                "name": block.name,
                "rows": block.rows,
                "first": block.first.isoformat() if block.first else None,
                "last": block.last.isoformat() if block.last else None,
                "up_rate": blocks[block.name].up_rate,
                "train_up_rate": blocks[block.name].train_up_rate,
                "oracle_gap": blocks[block.name].oracle_gap,
                "baselines": [
                    {
                        "rule": s.name,
                        "n": s.n,
                        "accuracy": s.accuracy,
                        "precision": s.precision,
                        "recall": s.recall,
                        "specificity": s.specificity,
                    }
                    for s in blocks[block.name].scores
                ],
            }
            for block in split.blocks
        ],
        "purged": split.purged,
    }


@app.command("ingest")
def ingest_command(
    source: Annotated[
        Path, typer.Option("--from", "-f", help="Directory holding the series files to import.")
    ],
    destination: Annotated[
        Path, typer.Option("--to", help="Where to copy them.")
    ] = DEFAULT_REFERENCE_DIR,
    label: Annotated[
        str, typer.Option("--label", help="Recorded in the manifest as the origin.")
    ] = "reference",
    manifest_path: Annotated[
        Path, typer.Option("--manifest", help="Manifest to write.")
    ] = DEFAULT_MANIFEST,
) -> None:
    """Validate an external directory of series files and copy them in."""
    # Every file is read and validated BEFORE it is copied, so `data/` only ever holds
    # series the rest of the pipeline may trust.
    try:
        report = import_directory(source, destination, label=label, data_root=DATA_ROOT)
    except ForecastLabError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from None

    for name, reason in report.rejected:
        console.print(f"[red]rejected[/red] {name}: {reason}")
    if report.skipped:
        console.print(f"[dim]{len(report.skipped)} file(s) ignored (not series files).[/dim]")

    if not report.imported:
        console.print(f"[red]Nothing imported from {source}.[/red]")
        raise typer.Exit(code=1)

    merged = _merge_entries(manifest_path, report.entries)
    manifest.write(manifest_path, merged)

    total = sum(item.meta.rows for item in report.imported)
    console.print(
        f"Imported [bold]{len(report.imported)}[/bold] series "
        f"({total:,} bars) into {destination}."
    )
    _report_grid_shape(report)
    console.print(f"Manifest written to [bold]{manifest_path}[/bold] ({len(merged)} entries).")
    if report.rejected:
        console.print(f"[yellow]{len(report.rejected)} file(s) rejected - see above.[/yellow]")
        raise typer.Exit(code=1)


@app.command("verify")
def verify_command(
    root: Annotated[
        Path, typer.Option("--dir", "-d", help="Data root the manifest paths are relative to.")
    ] = DATA_ROOT,
    manifest_path: Annotated[
        Path, typer.Option("--manifest", help="Manifest to check against.")
    ] = DEFAULT_MANIFEST,
) -> None:
    """Re-hash the data on disk and compare it against the committed manifest."""
    try:
        entries = manifest.read(manifest_path)
    except ForecastLabError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from None

    report = manifest.verify(root, entries)

    for path, reason in report.changed:
        console.print(f"[red]CHANGED[/red] {path}: {reason}")
    for path in report.missing:
        console.print(f"[red]MISSING[/red] {path}")
    if report.untracked:
        console.print(f"[yellow]{len(report.untracked)} untracked file(s) under {root}.[/yellow]")

    if report.is_clean:
        console.print(f"[green]OK[/green] {len(report.ok)} file(s) match the manifest.")
        return

    console.print(
        f"[red]{len(report.changed)} changed, {len(report.missing)} missing.[/red] "
        "Any result computed from this data is no longer reproducible from it."
    )
    raise typer.Exit(code=1)


def _report_grid_shape(report: ImportReport) -> None:
    """Surface series whose grid is worth a second look.

    These are not errors - a US index quoted on a European calendar genuinely sits on
    three offsets, and a daylight-saving Monday genuinely is a 23-hour day. But an
    unusual grid is exactly the kind of thing that explains a strange result three weeks
    later, so it is said out loud once, here, rather than discovered then.
    """
    odd = [
        item
        for item in report.imported
        if len(item.meta.anchors) > NOTEWORTHY_ANCHORS or item.meta.short_gaps
    ]
    if not odd:
        return
    console.print(f"[dim]{len(odd)} series with an unusual grid (recorded, not an error):[/dim]")
    for item in odd:
        console.print(
            f"  [dim]{item.series.symbol.name} {item.series.timeframe.value}: "
            f"{len(item.meta.anchors)} grid offsets, "
            f"{item.meta.short_gaps:,} short session(s)[/dim]"
        )


def _manifest_rows(manifest_path: Path, directory: Path) -> dict[str, int]:
    """Row counts for one data directory, keyed by file name.

    Entries are filtered to ``directory`` before the file name becomes the key.
    Keying on the bare name across the whole manifest would collide
    ``reference/XAUUSD_1H.csv`` with ``raw/XAUUSD_1H.csv`` and report one series'
    row count under the other - the same ambiguity the manifest paths exist to remove.

    Best effort by design: `symbols` answers "what is here?" and must keep working
    before anything has been ingested.
    """
    try:
        entries = manifest.read(manifest_path)
    except ForecastLabError:
        return {}
    try:
        prefix = directory.resolve().relative_to(DATA_ROOT.resolve()).as_posix()
    except ValueError:
        return {}
    return {
        Path(e.path).name: e.rows
        for e in entries
        if Path(e.path).parent.as_posix() == prefix
    }


def _merge_entries(manifest_path: Path, fresh: list[manifest.Entry]) -> list[manifest.Entry]:
    """Keep entries this run did not touch, so two data roots share one manifest.

    Importing the reference exports must not erase the record of the fetched series,
    and vice versa.
    """
    try:
        existing = manifest.read(manifest_path)
    except ForecastLabError:
        existing = []
    replaced = {e.path for e in fresh}
    return [e for e in existing if e.path not in replaced] + fresh


if __name__ == "__main__":  # pragma: no cover - convenience for `python -m`
    app()
