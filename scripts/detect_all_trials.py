"""
Batch spike detection across every trial of a cell.

Assumes you've already tuned params with ``tune_trial.py`` /
``tune_trials.py`` so the per-cell JSON sidecar exists at
``<data_root>/<day>/<cell_id>/spike_detection_<cell_id>.json``.
Loads those params, runs detection on every Raw .mat in the cell dir,
and writes /spikes + /spikeDetectionParams back into each trial file.

Usage
-----
    conda activate flop_py312
    python scripts/detect_all_trials.py 241203_F2_C1

Seed params from a specific trial's /spikeDetectionParams instead of
the per-cell JSON:

    python scripts/detect_all_trials.py 241203_F2_C1 --seed-trial 56

Restrict to a subset:

    python scripts/detect_all_trials.py 241203_F2_C1 --range 50-200
    python scripts/detect_all_trials.py 241203_F2_C1 --trials 56,58,62

Skip trials that already have /spikes written:

    python scripts/detect_all_trials.py 241203_F2_C1 --skip-existing

Dry-run (detect + summary, no .mat writes):

    python scripts/detect_all_trials.py 241203_F2_C1 --no-save
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Make the FlyLearning project root importable so `import mapd` works when
# running as `python scripts/detect_all_trials.py ...` from any cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _parse_trial_selection(args, all_nums: list[int]) -> list[int] | None:
    """Resolve --trials / --range into a set of trial numbers, or None for 'all'."""
    if args.trials:
        wanted = {int(x) for x in args.trials.split(",") if x.strip()}
        missing = wanted - set(all_nums)
        if missing:
            raise SystemExit(f"Trials not found in cell dir: {sorted(missing)}")
        return sorted(wanted)
    if args.range:
        lo, hi = args.range.split("-")
        lo, hi = int(lo), int(hi)
        return [t for t in all_nums if lo <= t <= hi]
    return None  # all


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("cell_id", help="e.g. 241203_F2_C1")
    ap.add_argument("--seed-trial", type=int, default=None,
                    help="seed params from this trial's /spikeDetectionParams "
                         "instead of the per-cell JSON")
    ap.add_argument("--trials", default=None,
                    help="comma-separated trial numbers, e.g. 56,58,62")
    ap.add_argument("--range", default=None,
                    help="inclusive range, e.g. 50-200")
    ap.add_argument("--channel", default="voltage_1",
                    help="ephys channel on the Trial (default voltage_1)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="don't redetect trials that already have /spikes")
    ap.add_argument("--no-save", action="store_true",
                    help="run detection but don't write /spikes back to .mat")
    ap.add_argument("--plot-summary", action="store_true",
                    help="after detection, show spikes-per-trial bar plot")
    args = ap.parse_args()

    if args.trials and args.range:
        ap.error("--trials and --range are mutually exclusive")

    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    # `import mapd` triggers matplotlib.use("Agg") inside mapd's headless
    # rendering helpers. Force an interactive backend AFTER those imports.
    import mapd
    from mapd import spike_detection as sds
    import matplotlib
    matplotlib.use("TkAgg", force=True)

    # Enumerate trials
    all_trials = sds.list_cell_trials(args.cell_id)
    all_nums = [tn for tn, _ in all_trials]
    print(f"Cell {args.cell_id}: {len(all_trials)} Raw .mat files "
          f"(trials {all_nums[0]}..{all_nums[-1]})")

    selected_nums = _parse_trial_selection(args, all_nums)
    if selected_nums is not None:
        subset = [(tn, p) for tn, p in all_trials if tn in selected_nums]
        print(f"  restricted to {len(subset)} trials")
    else:
        subset = all_trials

    # Pre-flight: every .mat must be HDF5 (MATLAB -v7.3). Legacy -v5/-v6/-v7
    # files lack the HDF5 magic and h5py can't open them, so Trial() would
    # crash partway through the run. Fail loud up front with the offending
    # filenames so the user can re-save them before writing any /spikes.
    import h5py
    bad = [(tn, p.name) for tn, p in subset if not h5py.is_hdf5(str(p))]
    if bad:
        print(f"\n{len(bad)} trial file(s) are not HDF5 (need MATLAB -v7.3):")
        for tn, name in bad:
            print(f"  trial {tn}: {name}")
        raise SystemExit(
            "Re-save these files in MATLAB with save('...', '-v7.3'), then re-run."
        )

    # Seed params
    first_trial = mapd.Trial(str(subset[0][1]))
    fs = float(first_trial.params["sampratein"])

    if args.seed_trial is not None:
        seed_path = sds.find_trial_path(args.cell_id, args.seed_trial)
        print(f"\nSeeding params from trial {args.seed_trial} ({seed_path.name})")
        seed = mapd.Trial(str(seed_path))
        params = sds.initial_params(seed, fs=fs, cell_id=None)
        if sds.load_spikes_from_trial(seed) is None:
            raise SystemExit(
                f"Trial {args.seed_trial} has no /spikeDetectionParams. "
                f"Run tune_trial.py on it first."
            )
    else:
        # Bypass the trial-first step of initial_params; go straight to JSON
        p = sds.load_cell_params(args.cell_id)
        if p is None:
            raise SystemExit(
                f"No per-cell JSON at <data_root>/{args.cell_id.split('_')[0]}"
                f"/{args.cell_id}/spike_detection_{args.cell_id}.json. "
                f"Run tune_trial.py or tune_trials.py first, "
                f"or pass --seed-trial N."
            )
        p.fs = fs
        params = p
        print(f"\nSeeding params from per-cell JSON")

    if params.spike_template is None:
        raise SystemExit(
            "Seeded params have no spike_template. "
            "Run TemplateSelectionGUI via tune_trial.py first."
        )

    print(f"  hp={params.hp_cutoff:g}, lp={params.lp_cutoff:g}, "
          f"polarity={params.polarity:+d}, "
          f"distance_threshold={params.distance_threshold:g}, "
          f"amplitude_threshold={params.amplitude_threshold:g}")

    # Load trial objects, optionally skipping ones that already have /spikes
    trials = []
    skipped = 0
    for tn, p in subset:
        tr = mapd.Trial(str(p))
        if args.skip_existing and sds.load_spikes_from_trial(tr) is not None:
            skipped += 1
            continue
        trials.append(tr)
    if skipped:
        print(f"\nSkipping {skipped} trials that already have /spikes "
              f"(--skip-existing)")
    print(f"\nDetecting on {len(trials)} trials...")

    results, summary = sds.batch_detect_spikes(
        trials, params, channel=args.channel, save=not args.no_save,
        verbose=True,
    )

    print("\nSummary:")
    print(summary.describe()[["n_spikes", "rate_hz"]].to_string())
    print(f"\n  total spikes: {int(summary.n_spikes.sum())}")
    print(f"  trials with zero spikes: "
          f"{int((summary.n_spikes == 0).sum())} / {len(summary)}")

    if args.no_save:
        print("\n(dry run — no /spikes written)")

    if args.plot_summary and len(summary):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(summary.trial, summary.n_spikes, width=0.8)
        ax.set_xlabel("trial")
        ax.set_ylabel("n_spikes")
        ax.set_title(f"{args.cell_id} — spikes per trial")
        fig.tight_layout()
        plt.show()

    globals().update(locals())
    return 0


if __name__ == "__main__":
    sys.exit(main())
