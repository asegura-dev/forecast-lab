"""Bringing an external directory of series files into the project, validated.

Two things happen here, and the order is the point: every file is **read and validated
before it is copied**. A file that does not sit on a regular grid never reaches
`data/`, so the data directory only ever contains series the rest of the pipeline is
allowed to trust.

Copying rather than referencing is deliberate. A manifest that points at files
somewhere else on the machine describes something that can move, be edited, or vanish -
which is precisely the failure the manifest exists to detect.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from forecast_lab.contracts import ForecastLabError, SeriesMeta
from forecast_lab.ingest.catalog import SeriesFile, scan
from forecast_lab.ingest.csv_reader import read_series
from forecast_lab.ingest.manifest import Entry, entry_for


@dataclass(frozen=True)
class Imported:
    """One file that made it in."""

    series: SeriesFile
    meta: SeriesMeta
    entry: Entry


@dataclass(frozen=True)
class ImportReport:
    """Everything the import did, including what it refused and why.

    Rejections are carried rather than raised. Importing thirty files and stopping at
    the fourth means four round trips to learn about four problems; the operator wants
    the whole list at once.
    """

    imported: tuple[Imported, ...]
    rejected: tuple[tuple[str, str], ...]  # (file name, reason)
    skipped: tuple[tuple[str, str], ...]  # (file name, why it was not a series)

    @property
    def entries(self) -> list[Entry]:
        return [item.entry for item in self.imported]


def import_directory(
    source: Path, destination: Path, *, label: str, data_root: Path | None = None
) -> ImportReport:
    """Validate and copy every series file in ``source`` into ``destination``.

    ``label`` records where the data came from, so a manifest holding files from two
    venues can still say which is which.

    ``data_root`` anchors the paths written into the manifest. It defaults to the
    destination's parent, so a series copied into ``data/reference`` is recorded as
    ``reference/XAUUSD_1H.csv``. Recording it relative to the destination instead would
    make it ``XAUUSD_1H.csv`` - the same string a fetch into ``data/raw`` would produce,
    and two different files sharing one manifest key is exactly the ambiguity a manifest
    exists to remove.
    """
    if not source.is_dir():
        raise ForecastLabError(f"not a directory: {source}")

    root = data_root or destination.parent

    found = scan(source)
    destination.mkdir(parents=True, exist_ok=True)

    imported: list[Imported] = []
    rejected: list[tuple[str, str]] = []

    for item in found.series:
        try:
            _, meta = read_series(item.path, item.symbol, item.timeframe)
        except ForecastLabError as exc:
            rejected.append((item.path.name, str(exc)))
            continue

        target = destination / f"{item.symbol.name}_{item.timeframe.value}.csv"
        shutil.copyfile(item.path, target)
        imported.append(
            Imported(
                series=item,
                meta=meta,
                entry=entry_for(target, root, meta, source=label),
            )
        )

    return ImportReport(
        imported=tuple(imported),
        rejected=tuple(rejected),
        skipped=tuple((p.name, reason) for p, reason in found.skipped),
    )
