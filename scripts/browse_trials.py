"""
Terminal-launched Trial Browser.

Loads a ``mapd.Table`` for one fly/cell and opens a PySide6 desktop window
that walks through its trials. Probe_position sits in the top axes, ephys
channels overlay in the bottom axes, metadata lives in the side panel.

Usage
-----
    conda activate flop_py312
    python scripts/browse_trials.py 241203_F2_C1

    # multi-protocol cell: disambiguate
    python scripts/browse_trials.py 210319_F2_C1 --protocol LEDFlashWithPiezoCueControl
    python scripts/browse_trials.py 210319_F2_C1 --list-protocols

    # drop into a REPL after the window closes:
    python -i scripts/browse_trials.py 241203_F2_C1
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Make the FlyLearning project root importable so `import mapd` works when
# running as `python scripts/browse_trials.py ...` from any cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("cell_id", help="e.g. 241203_F2_C1")
    ap.add_argument("--protocol", default=None,
                    help="protocol name (required only if the cell has more "
                         "than one Table parquet)")
    ap.add_argument("--list-protocols", action="store_true",
                    help="print protocols available for this cell and exit")
    ap.add_argument("--trial", type=int, default=None,
                    help="jump to this trial number on open")
    ap.add_argument("--include-excluded", action="store_true",
                    help="include excluded trials in the trial list "
                         "(default: skip)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # `import mapd` triggers matplotlib.use("Agg") inside mapd's headless
    # rendering helpers. Force the interactive Qt backend AFTER those imports
    # so we win the race.
    import mapd
    from mapd import paths as _paths
    os.environ.setdefault("QT_API", "pyside6")
    import matplotlib
    matplotlib.use("QtAgg", force=True)

    if args.list_protocols:
        protocols = _paths.list_protocols(args.cell_id)
        if not protocols:
            print(f"No Table parquets found for {args.cell_id}")
            return 1
        print("Protocols for {}:".format(args.cell_id))
        for p in protocols:
            print(f"  - {p}")
        return 0

    parquet_path = _paths.resolve_table_path(args.cell_id, protocol=args.protocol)
    print(f"Loading {parquet_path}")
    table = mapd.Table.for_path(parquet_path)

    from PySide6.QtWidgets import QApplication
    from mapd.trial_browser import TrialBrowser

    app = QApplication.instance() or QApplication(sys.argv)
    win = TrialBrowser(table,
                       start_trial=args.trial,
                       include_excluded=args.include_excluded)
    win.show()
    print("\nBrowser open — close the window to return to the REPL.")
    exit_code = app.exec()

    globals().update(locals())
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
