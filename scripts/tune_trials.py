"""
Interactive spike-detection tuning on N concatenated trials.

Tuning on a single trial can overfit — a filter that looks clean on
trial 56 may behave differently across 20 trials. This script
concatenates N consecutive trials (starting from a seed) into one long
Recording and runs FilterGUI → TemplateSelectionGUI → ThresholdGUI on
that combined signal, so the params you land on are the params that
work across the subset.

Routes through ``mapd.spike_detection`` so params are seeded with the
right precedence (seed trial's /spikeDetectionParams → per-cell JSON →
project defaults → stock) and saved back to both the per-cell JSON
sidecar and the seed trial's .mat file.

Usage
-----
    conda activate flop_py312
    python -i scripts/tune_trials.py 241203_F2_C1 56

A small dialog asks how many trials to concatenate (default 20). Skip
the dialog with ``--n``:

    python -i scripts/tune_trials.py 241203_F2_C1 56 --n 50
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Tuple

# Make the FlyLearning project root importable so `import mapd` works when
# running as `python scripts/tune_trials.py ...` from any cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np


def prompt_num_trials(default: int, max_available: int) -> int:
    """Tiny modal dialog: TextBox + Go button. Returns clamped int."""
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, TextBox

    result = {"n": default}

    fig = plt.figure(figsize=(4.5, 2.2))
    fig.canvas.manager.set_window_title("Concatenate trials")
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    ax.text(0.5, 0.85, "How many trials to concatenate?",
            ha="center", va="center", transform=ax.transAxes, fontsize=12)
    ax.text(0.5, 0.65, f"(1 – {max_available} available from seed)",
            ha="center", va="center", transform=ax.transAxes,
            fontsize=9, color="gray")

    tb_ax = fig.add_axes([0.20, 0.30, 0.45, 0.20])
    tb = TextBox(tb_ax, "N = ", initial=str(default))
    btn_ax = fig.add_axes([0.70, 0.30, 0.18, 0.20])
    btn = Button(btn_ax, "Go")

    def submit(_=None):
        try:
            n = int(tb.text)
        except ValueError:
            return
        n = max(1, min(n, max_available))
        result["n"] = n
        plt.close(fig)

    btn.on_clicked(submit)
    tb.on_submit(lambda _: submit())
    plt.show()
    return result["n"]


def concat_voltages(trial_files: List[Tuple[int, Path]],
                    sds, channel: str = "voltage_1"):
    """Load `channel` from each trial and concatenate.

    Returns (voltage: np.ndarray, fs: float, boundaries: list[int]) where
    boundaries[i] is the sample index where trial i ends (exclusive).
    """
    import mapd

    voltages = []
    fs = None
    boundaries: List[int] = []
    total = 0
    for tn, p in trial_files:
        trial = mapd.Trial(str(p))
        rec = sds.trial_to_recording(trial, channel=channel)
        if fs is None:
            fs = rec.sample_rate
        elif rec.sample_rate != fs:
            raise ValueError(
                f"Sample-rate mismatch: trial {tn} is {rec.sample_rate} Hz, "
                f"expected {fs} Hz"
            )
        voltages.append(rec.voltage)
        total += len(rec.voltage)
        boundaries.append(total)
    return np.concatenate(voltages), float(fs), boundaries


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("cell_id", help="e.g. 241203_F2_C1")
    ap.add_argument("seed_trial", type=int, help="starting trial, e.g. 56")
    ap.add_argument("--n", type=int, default=None,
                    help="number of trials to concatenate; prompts via GUI if omitted")
    ap.add_argument("--default-n", type=int, default=20,
                    help="default value in the N-prompt dialog (default 20)")
    ap.add_argument("--channel", default="voltage_1",
                    help="ephys channel on the Trial (default voltage_1)")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--save-json-only", action="store_true",
                    help="write the per-cell JSON but don't write params "
                         "into the seed trial's .mat")
    args = ap.parse_args()

    # Surface spikedetect's logger.info messages
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # `import mapd` triggers matplotlib.use("Agg") inside mapd's headless
    # rendering helpers (table_movie_maker, table_export_methods). Force the
    # interactive backend AFTER those imports so we win the race.
    import mapd
    from mapd import spike_detection as sds
    import matplotlib
    matplotlib.use("QtAgg", force=True)
    import spikedetect as sd
    from spikedetect import SignalFilter
    # from spikedetect.gui import FilterGUI, TemplateSelectionGUI, ThresholdGUI
    from spikedetect.gui import (
        FilterGUIQt,
        TemplateSelectionGUIQt,
        ThresholdGUIQt,
        SpotCheckGUIQt,
    )


    all_trials = sds.list_cell_trials(args.cell_id)
    all_nums = [tn for tn, _ in all_trials]
    try:
        seed_idx = all_nums.index(args.seed_trial)
    except ValueError:
        raise SystemExit(
            f"Seed trial {args.seed_trial} not found. "
            f"Available: {all_nums[0]}..{all_nums[-1]} "
            f"({len(all_nums)} trials)"
        )
    max_available = len(all_trials) - seed_idx

    n = args.n if args.n is not None else prompt_num_trials(
        min(args.default_n, max_available), max_available,
    )
    n = max(1, min(n, max_available))
    subset = all_trials[seed_idx:seed_idx + n]
    first_tn, last_tn = subset[0][0], subset[-1][0]
    print(f"Concatenating trials {first_tn}..{last_tn} ({len(subset)} files)")

    voltage, fs, boundaries = concat_voltages(subset, sds, channel=args.channel)
    print(f"  {len(voltage)} samples @ {fs:g} Hz "
          f"({len(voltage)/fs:.1f} s total)")

    rec = sd.Recording(
        name=f"{args.cell_id}_tr{first_tn}-{last_tn}",
        voltage=voltage, sample_rate=fs,
    )

    # Seed params via the mapd precedence chain, keyed on the seed trial
    seed_trial = mapd.Trial(str(subset[0][1]))
    params = sds.initial_params(seed_trial, fs=fs, cell_id=args.cell_id)
    print(f"  seeded params: hp={params.hp_cutoff:g}, lp={params.lp_cutoff:g}, "
          f"polarity={params.polarity:+d}, "
          f"template={'yes' if params.spike_template is not None else 'no'}")

    print("\n[1/3] FilterGUIQt — close window to continue")
    params = FilterGUIQt(rec.voltage, params).run()

    filtered = SignalFilter.filter_data(
        rec.voltage, fs=fs,
        hp_cutoff=params.hp_cutoff, lp_cutoff=params.lp_cutoff,
        diff_order=params.diff_order, polarity=params.polarity,
    )

    print("\n[2/3] TemplateSelectionGUI — click a clean spike; close to continue")
    params.spike_template = TemplateSelectionGUIQt(filtered, params).run()
    print(f"  template length: {len(params.spike_template)}")

    print("\n[3/3] ThresholdGUI — drag thresholds; close to continue")
    match = sds.match_candidates(rec, params)
    params = ThresholdGUIQt(match_result=match, params=params).run()
    print(f"  distance_threshold={params.distance_threshold:g}, "
          f"amplitude_threshold={params.amplitude_threshold:g}")

    # Quick sanity check on the full concatenated recording
    print("\nRunning detection on concatenated signal (sanity check)...")
    result = sd.detect_spikes(rec, params)
    print(f"  {result.n_spikes} spikes across {len(subset)} trials "
          f"(~{result.n_spikes / len(subset):.1f} per trial)")

    if not args.no_save:
        json_path = sds.save_cell_params(args.cell_id, params)
        print(f"\nSaved -> {json_path}")
        if not args.save_json_only:
            sds.save_params_to_trial(seed_trial, params)
            print(f"Saved -> /spikeDetectionParams/ in seed trial "
                  f"{subset[0][1].name}")

    globals().update({
        "rec": rec, "params": params, "filtered": filtered,
        "subset": subset, "boundaries": boundaries,
        "seed_trial": seed_trial, "result": result, "match": match,
    })
    return 0


if __name__ == "__main__":
    sys.exit(main())
