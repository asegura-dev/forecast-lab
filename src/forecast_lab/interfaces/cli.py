"""The command line, and the composition root.

The ONLY module allowed to import concrete implementations and the ONLY one that
touches the filesystem or the network (ADR-001 sec. 1). Everything else receives what
it needs as an argument, which is what makes every research function callable from a
test with synthetic data and no disk.

Invoke as ``forecast-lab <command>`` (in development, ``uv run forecast-lab <command>``).

Console output is deliberately plain ASCII. A scheduled task on Windows redirecting
stdout to a file under cp1252 will crash on a stray typographic character, and a
crash in the logging is indistinguishable from a crash in the work.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from forecast_lab.contracts import ForecastLabError
from forecast_lab.ingest import scan

app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()

DEFAULT_DATA_DIR = Path("data/raw")


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
        Path,
        typer.Option("--dir", "-d", help="Directory to scan for series files."),
    ] = DEFAULT_DATA_DIR,
    show_skipped: Annotated[
        bool,
        typer.Option("--show-skipped", help="Also list files that were not recognised."),
    ] = False,
) -> None:
    """List the series available on disk, without reading them."""
    # File names only. "What is here?" is the cheap question and is kept apart from
    # "what is inside?", so this stays instant on a directory of any size. Row counts
    # and time spans arrive with the reader.
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

    # The path goes on its own line rather than in the table title: Rich wraps a title
    # to the table's width, and a long absolute path comes out shredded.
    console.print(f"Series in [bold]{directory}[/bold]")
    table = Table()
    table.add_column("Symbol")
    for timeframe in ("1H", "4H", "1D"):
        table.add_column(timeframe, justify="center")

    available = {s.key for s in report.series}
    for symbol in report.symbols:
        marks = ["yes" if (symbol.name, tf) in available else "-" for tf in ("1H", "4H", "1D")]
        table.add_row(symbol.name, *marks)

    console.print(table)
    console.print(f"{len(report.symbols)} symbols, {len(report.series)} series files.")

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


if __name__ == "__main__":  # pragma: no cover - convenience for `python -m`
    app()
