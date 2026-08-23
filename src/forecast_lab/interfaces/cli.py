"""The command line, and the composition root.

The ONLY module allowed to import concrete implementations and the ONLY one that
touches the filesystem or the network (ADR-001 sec. 1). Everything else receives what
it needs as an argument, which is what makes every research function callable from a
test with synthetic data and no disk.

Invoke as ``forecast-lab <command>`` (in development, ``uv run forecast-lab <command>``).

Console output is deliberately plain ASCII. A scheduled task on Windows redirecting
stdout to a file under cp1252 will crash on a stray typographic character, and a crash
in the logging is indistinguishable from a crash in the work.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
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
from forecast_lab.research import align_to_target

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
