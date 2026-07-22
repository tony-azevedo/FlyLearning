"""
Plot the spikes saved on a mapd.Trial.

Shows raw voltage, filtered signal, spike-triggered waveform overlay,
and an ISI histogram in a single native matplotlib window. Reads
/spikeDetectionParams and /spikes from the trial's .mat file.

Usage
-----
    conda activate flop_py312
    python scripts/plot_trial.py 241203_F2_C1 56
    python scripts/plot_trial.py 241203_F2_C1 56 --channel voltage_2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the FlyLearning project root importable so `import mapd` works when
# running as `python scripts/plot_trial.py ...` from any cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("cell_id", help="e.g. 241203_F2_C1")
    ap.add_argument("trial", type=int, help="trial number, e.g. 56")
    ap.add_argument("--channel", default="voltage_1",
                    help="ephys channel to read (default voltage_1)")
    args = ap.parse_args()

    # `import mapd` triggers matplotlib.use("Agg") inside mapd's headless
    # rendering helpers (table_movie_maker, table_export_methods). Force the
    # interactive backend AFTER those imports so we win the race.
    import mapd
    from mapd import spike_detection as sds
    import matplotlib
    matplotlib.use("TkAgg", force=True)

    trial_path = sds.find_trial_path(args.cell_id, args.trial)
    print(f"Loading {trial_path}")
    trial = mapd.Trial(str(trial_path))
    sds.plot_trial_spikes(trial, channel=args.channel)
    return 0


if __name__ == "__main__":
    sys.exit(main())
