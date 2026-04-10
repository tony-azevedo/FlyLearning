"""
tests/test_junction_spike.py
============================
Regression test / visual diagnostic for velocity spikes at trial junctions.

Loads trial 144 from 210302_F1_C2, concatenates with trial 145, runs bout
detection, and checks that no sample in the rolling-RMS speed exceeds a
sanity threshold near the junction.  Also saves an HTML figure for visual
inspection.

Run directly:
    python tests/test_junction_spike.py

Or import and call:
    from tests.test_junction_spike import run
    fig, passed = run()
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import mapd
import mapd.kinematics as kin

PARQUET = r"D:\Data\210302\210302_F1_C2\LEDFlashWithPiezoCueControl_210302_F1_C2_Table.parquet"
TRIAL_NUM = 144
V_TH = 20.0
V_REST = 13.4
OUT_HTML = os.path.join(os.path.dirname(__file__), f'trial_{TRIAL_NUM}_junction_spike.html')

# Speed above this near the junction is considered a spike artefact
SPIKE_THRESHOLD = 100.0   # um/s — well above v_th but below real movement


def run(save_html=True):
    T = mapd.Table(PARQUET)
    T.extract_trial_properties()
    T.find_successful_trials()

    hard_rows = T.df.loc[T.df['hard_success']].copy()
    hard_rows['next_trial'] = T.df['Trial'].shift(-1).loc[hard_rows.index]

    row = hard_rows.loc[TRIAL_NUM]
    trial1 = row['Trial']
    print(trial1.total_duration)
    print(trial1.params['stimDurInSec']+trial1.params['postDurInSec'])
    trial2 = row['next_trial']
    print(trial2.total_duration)

    bouts, states, t, x = kin.detect_bouts_across_trials(
        [trial1, trial2],
        classifier=kin._classify_states_schmitt,
    )

    # ── Junction spike check ──────────────────────────────────────────────────
    # The junction is at the boundary between trial1 and trial2 samples.
    junction_idx = int(np.searchsorted(t, trial1.total_duration))

    # Check a window of 1s around the junction
    fs = 1.0 / np.median(np.diff(t))
    window = int(fs)
    lo = max(0, junction_idx - window)
    hi = min(len(t), junction_idx + window)

    fs_local = 1.0 / np.median(np.diff(t))
    v_raw  = kin.velocity(t, x)
    v_zeroed = kin._zero_short_dt_velocity(v_raw.copy(), t, smooth_window=0.05, fs=fs_local)
    speed_near_junction = np.abs(v_zeroed[lo:hi])
    max_speed = float(np.max(speed_near_junction))

    passed = max_speed < SPIKE_THRESHOLD
    status = 'PASS' if passed else 'FAIL'
    print(f'[{status}] Max |v| near junction: {max_speed:.1f} um/s  '
          f'(threshold {SPIKE_THRESHOLD} um/s)')

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = kin.plot_bouts(
        bouts, states, t, x,
        trials=[trial1, trial2],
        v_th=V_TH, v_rest=V_REST,
    )

    if save_html:
        fig.write_html(OUT_HTML)
        print(f'Figure saved: {OUT_HTML}')

    return fig, passed


if __name__ == '__main__':
    _, passed = run()
    sys.exit(0 if passed else 1)
