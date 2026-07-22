"""
Interactive spike-detection parameter tuning on a single trial (Qt backend).

All four GUIs run in Qt windows. FilterGUIQt is a native Qt dialog with
real QSlider/QPushButton; TemplateSelectionGUI / ThresholdGUI /
SpotCheckGUI are matplotlib widgets embedded in Qt-hosted figure windows.

Routes through ``mapd.spike_detection`` so params are loaded with the right
precedence (existing trial /spikeDetectionParams → per-cell JSON → project
defaults → stock) and saved back to the trial's .mat file plus the per-cell
JSON sidecar.

Usage
-----
    conda activate flop_py312
    python -i scripts/tune_trial.py 241203_F2_C1 56
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Belt: pin the matplotlib backend before any indirect import that might
# touch pyplot. The braces+force=True call below is the suspenders.
os.environ.setdefault("MPLBACKEND", "QtAgg")

# Make the FlyLearning project root importable so `import mapd` works when
# running as `python scripts/tune_trial.py ...` from any cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("cell_id", help="e.g. 241203_F2_C1")
    ap.add_argument("trial", type=int, help="trial number, e.g. 56")
    ap.add_argument("--channel", default="voltage_1",
                    help="ephys channel on the Trial (default voltage_1)")
    ap.add_argument("--retune", action="store_true",
                    help="force the tuning pass even if the trial already "
                         "has complete /spikeDetectionParams")
    ap.add_argument("--spotcheck", action="store_true",
                    help="run SpotCheckGUI after detection (only meaningful "
                         "when re-tuning)")
    ap.add_argument("--no-save", action="store_true",
                    help="skip persistence (no .mat write, no JSON write)")
    ap.add_argument("--save-json-only", action="store_true",
                    help="write the per-cell JSON but don't write params "
                         "into the trial's .mat")
    ap.add_argument("--no-plot", action="store_true",
                    help="skip the final plot")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # mapd's headless helpers (table_movie_maker, table_export_methods)
    # call matplotlib.use("Agg") at import. Override with force=True
    # afterwards so all interactive figures land in Qt windows.
    import mapd
    from mapd import spike_detection as sds
    import matplotlib
    matplotlib.use("QtAgg", force=True)

    import spikedetect as sd
    from spikedetect import SignalFilter
    # from spikedetect.gui import (
    #     FilterGUIQt,            # Qt-native dialog
    #     TemplateSelectionGUI,   # matplotlib widgets in a Qt window
    #     ThresholdGUI,
    #     SpotCheckGUI,
    # )

    from spikedetect.gui import (
        FilterGUIQt,
        TemplateSelectionGUIQt,
        ThresholdGUIQt,
        SpotCheckGUIQt,
    )
    # ...


    # FilterGUIQt and matplotlib's QtAgg backend both need a QApplication.
    # Create one once; both will share it. Adjust the import if you're on
    # a different binding (PyQt6 / PyQt5 / PySide2 work the same).
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv)

    trial_path = sds.find_trial_path(args.cell_id, args.trial)
    print(f"Loading {trial_path}")
    trial = mapd.Trial(str(trial_path))

    rec = sds.trial_to_recording(trial, channel=args.channel)
    print(f"  {rec.name}: {rec.sample_rate:g} Hz, "
          f"{rec.n_samples} samples ({rec.duration:.2f} s)")

    existing = sds.load_spikes_from_trial(trial)
    has_complete_params = (
        existing is not None and existing.params.spike_template is not None
    )
    if has_complete_params and not args.retune:
        print("\nTrial already has /spikeDetectionParams:")
        sds.print_trial_params(trial)
        ans = input("\nRe-tune? [y/N]: ").strip().lower()
        retune = ans.startswith("y")
    else:
        retune = True

    params = sds.initial_params(trial, fs=rec.sample_rate,
                                cell_id=args.cell_id)
    print(f"\n  seeded params: hp={params.hp_cutoff:g}, lp={params.lp_cutoff:g}, "
          f"polarity={params.polarity:+d}, "
          f"template={'yes' if params.spike_template is not None else 'no'}")

    result = None
    if retune:
        print("\n[1/3] FilterGUIQt — Enter to accept, Esc to cancel")
        params = FilterGUIQt(rec.voltage, params).run()

        filtered = SignalFilter.filter_data(
            rec.voltage, fs=params.fs,
            hp_cutoff=params.hp_cutoff, lp_cutoff=params.lp_cutoff,
            diff_order=params.diff_order, polarity=params.polarity,
        )

        print("\n[2/3] TemplateSelectionGUI — click a clean spike; close to continue")
        # params.spike_template = TemplateSelectionGUI(filtered, params).run()
        params.spike_template = TemplateSelectionGUIQt(filtered, params).run()
        print(f"  template length: {len(params.spike_template)}")

        print("\n[3/3] ThresholdGUI — drag thresholds; close to continue")
        match = sds.match_candidates(rec, params)
        # params = ThresholdGUI(match_result=match, params=params).run()
        params = ThresholdGUIQt(match_result=match, params=params).run()
    
        print("\nRunning detection with tuned params...")
        result = sd.detect_spikes(rec, params)
        print(f"  {result.n_spikes} spikes")

        if args.spotcheck and result.n_spikes > 0:
            print("\nSpotCheckGUI — y/n per spike; close when done")
            # result = SpotCheckGUI(rec, result).run()
            result = SpotCheckGUIQt(rec, result).run()
    
            print(f"  spot-checked: {result.n_spikes} spikes remain")
    elif has_complete_params and existing.n_spikes > 0:
        result = existing

    if retune and not args.no_save:
        json_path = sds.save_cell_params(args.cell_id, params)
        print(f"\nSaved -> {json_path}")
        if not args.save_json_only:
            sds.save_params_to_trial(trial, params, invalidate_spikes=True)
            print(f"Saved -> /spikeDetectionParams/ in {trial_path.name} "
                  "(stale /spikes cleared)")
            if result is not None:
                sds.save_spikes_to_trial(trial, result)
                print(f"Saved -> /spikes in {trial_path.name}")

    if result is not None and not args.no_plot:
        print("\nPlotting...")
        sds.plot_trial_spikes(trial, result=result, channel=args.channel)

    print("\nDone. `python -i` leaves `trial`, `params`, `rec`, `result`, "
          "and `app` in scope for inspection.")
    globals().update(locals())
    return 0


if __name__ == "__main__":
    sys.exit(main())

