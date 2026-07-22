"""
Filename conventions and path resolution for FlyLearning data.

All code that needs to translate between a `day_fly_cell` string (e.g.
``"241203_F2_C1"``), a protocol name (e.g. ``"LEDFlashTriggerPiezoControl"``),
and on-disk parquet/.mat paths should route through this module. Keeps the
FlySoundAcquisition naming convention in one place:

    {protocol}_{day}_F{fly}_C{cell}_Table.parquet
    {protocol}_Raw_{day}_F{fly}_C{cell}_{trial}.mat
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from os import path as _ospath
from pathlib import Path


# ---------------------------------------------------------------------
# Data root + small filename helpers (formerly mapd.helpers)
# ---------------------------------------------------------------------
def default_data_directory(verbose: bool = True) -> str:
    """Return the first existing data root from the known candidate list."""
    possible_paths = [
        r"D:\Data",
        r"C:\Users\Tony\Data",
    ]
    for p in possible_paths:
        if _ospath.isdir(p):
            if verbose:
                print(f"Found data directory: {p}")
            return p
    raise FileNotFoundError("No valid data directory found in expected locations.")


def get_path(filename: str) -> str:
    """Directory portion of `filename`, or the data root if empty."""
    directory = _ospath.dirname(filename)
    return directory if directory else default_data_directory()


def get_file(filename: str) -> str:
    """Basename of `filename` (thin wrapper around os.path.basename)."""
    return _ospath.basename(filename)


def get_day_fly_cell(file_path: str) -> tuple[str, str, str]:
    """Extract (day, fly, cell) strings from a filename containing ``_YYMMDD_Fn_Cn_``.

    Returns strings (not ints) for backward compatibility with existing
    callers that then do ``str(fly)``/``int(fly)`` themselves. For new code,
    prefer ``CellId.parse``.
    """
    match = re.search(r"_(\d{6})_F(\d)_C(\d)_", file_path, re.IGNORECASE)
    if match:
        yymmdd, fly, cell = match.groups()
        return yymmdd, fly, cell
    raise ValueError(f"No day/fly/cell identifiers found in {file_path!r}")


_TABLE_RE = re.compile(
    r"^(?P<protocol>.+?)_"
    r"(?P<day>\d{6})_F(?P<fly>\d+)_C(?P<cell>\d+)_Table$"
)

_TRIAL_RE = re.compile(
    r"^(?P<protocol>.+?)_Raw_"
    r"(?P<day>\d{6})_F(?P<fly>\d+)_C(?P<cell>\d+)_"
    r"(?P<trial>\d+)$"
)

_CELL_ID_RE = re.compile(r"^(?P<day>\d{6})_F(?P<fly>\d+)_C(?P<cell>\d+)$")


@dataclass(frozen=True)
class CellId:
    day: str
    fly: int
    cell: int

    @classmethod
    def parse(cls, s: str) -> "CellId":
        m = _CELL_ID_RE.match(s)
        if not m:
            raise ValueError(
                f"Not a valid cell id: {s!r} (expected e.g. '241203_F2_C1')"
            )
        return cls(day=m["day"], fly=int(m["fly"]), cell=int(m["cell"]))

    def __str__(self) -> str:
        return f"{self.day}_F{self.fly}_C{self.cell}"

    def cell_dir(self, data_root: Path | str | None = None) -> Path:
        root = Path(data_root) if data_root else Path(default_data_directory(verbose=False))
        return root / self.day / str(self)


def _coerce_cell(cell_id: str | CellId) -> CellId:
    return cell_id if isinstance(cell_id, CellId) else CellId.parse(cell_id)


def parse_protocol_from_path(path: str | Path) -> str | None:
    """Extract the protocol from a parquet or .mat filename. None if unparseable."""
    stem = Path(path).stem
    m = _TABLE_RE.match(stem) or _TRIAL_RE.match(stem)
    return m["protocol"] if m else None


def list_protocols(cell_id: str | CellId, data_root: Path | str | None = None) -> list[str]:
    """Protocols with a Table parquet in the cell's directory, sorted alphabetically."""
    cid = _coerce_cell(cell_id)
    cdir = cid.cell_dir(data_root)
    if not cdir.is_dir():
        raise FileNotFoundError(f"Cell directory not found: {cdir}")
    found: set[str] = set()
    for p in cdir.glob("*_Table.parquet"):
        proto = parse_protocol_from_path(p)
        if proto is not None:
            found.add(proto)
    return sorted(found)


def _pick_protocol(cell_id: CellId, protocol: str | None,
                   data_root: Path | str | None) -> str:
    if protocol is not None:
        return protocol
    protocols = list_protocols(cell_id, data_root)
    if len(protocols) == 1:
        return protocols[0]
    if not protocols:
        raise FileNotFoundError(
            f"No Table parquet found for {cell_id} in {cell_id.cell_dir(data_root)}"
        )
    raise ValueError(
        f"Multiple protocols for {cell_id}: {protocols}. "
        "Pass protocol=... to disambiguate."
    )


def resolve_table_path(cell_id: str | CellId,
                       protocol: str | None = None,
                       data_root: Path | str | None = None) -> Path:
    """Return the parquet path for a cell's table. Disambiguate with `protocol` if needed."""
    cid = _coerce_cell(cell_id)
    proto = _pick_protocol(cid, protocol, data_root)
    p = cid.cell_dir(data_root) / f"{proto}_{cid}_Table.parquet"
    if not p.is_file():
        raise FileNotFoundError(p)
    return p


def resolve_trial_path(cell_id: str | CellId,
                       trial: int,
                       protocol: str | None = None,
                       data_root: Path | str | None = None) -> Path:
    """Return the .mat path for one trial. Disambiguate with `protocol` if needed."""
    cid = _coerce_cell(cell_id)
    proto = _pick_protocol(cid, protocol, data_root)
    p = cid.cell_dir(data_root) / f"{proto}_Raw_{cid}_{trial}.mat"
    if not p.is_file():
        raise FileNotFoundError(p)
    return p


def trial_filename_template(parquet_name: str) -> str:
    """Given a Table parquet basename, return the trial .mat basename template.

    Template has a single ``{x}`` placeholder for the trial number::

        LEDFlashTriggerPiezoControl_241203_F2_C1_Table.parquet
        -> LEDFlashTriggerPiezoControl_Raw_241203_F2_C1_{x}.mat
    """
    stem = Path(parquet_name).stem
    m = _TABLE_RE.match(stem)
    if not m:
        raise ValueError(f"Not a recognizable Table parquet name: {parquet_name!r}")
    return (
        f"{m['protocol']}_Raw_{m['day']}_F{m['fly']}_C{m['cell']}_{{x}}.mat"
    )
