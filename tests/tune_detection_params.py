"""
tests/tune_detection_params.py
==============================
Interactive script for tuning bout detection thresholds.

Uses the module-level V_TH / V_REST defaults from kinematics.py.
Loads trial 144 from 210302_F1_C2, concatenates with trial 145, runs bout
detection, and saves an HTML figure for visual inspection.

Run directly:
    python tests/tune_detection_params.py

Or import and call:
    from tests.tune_detection_params import run
    fig = run()
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import mapd
import mapd.kinematics as kin
from mapd.kinematics import (
    _rolling_rms, _classify_states_schmitt, velocity,
    STATE_REST, STATE_DRIFT, STATE_MOVE,
    _build_trace, detect_movement_bouts, detect_bouts_across_trials, plot_bouts,
)

PARQUET = r"D:\Data\210302\210302_F1_C2\LEDFlashWithPiezoCueControl_210302_F1_C2_Table.parquet"
PARQUET2 = r"D:\Data\250304\250304_F3_C1\LEDFlashTriggerPiezoControl_250304_F3_C1_Table.parquet"
TRIAL_NUM = 144
V_TH = 60
V_REST = 30
OUT_HTML = os.path.join(os.path.dirname(__file__), f'trial_{TRIAL_NUM}_tune_params.html')


# ── Helpers ──────────────────────────────────────────────────────────────────

def concatenate_trials(T, n_trials=200):
    """Concatenate downsampled traces from the first *n_trials* non-excluded trials."""
    trials = [tr for tr in T.df.Trial[:n_trials] if not tr.excluded]
    ts, xs = [], []
    t_offset = 0.0
    for tr in trials:
        idx = tr.downsample_probe
        t_raw = tr.time[idx].squeeze()
        x_raw = -(tr.probe_position[idx].squeeze() - tr.probeZero)
        if ts:
            dt = np.median(np.diff(t_raw))
            t_raw = t_raw - t_raw[0] + t_offset + dt
        t_offset = t_raw[-1]
        ts.append(t_raw)
        xs.append(x_raw)

    t = np.concatenate(ts)
    x = np.concatenate(xs)
    fs = 1.0 / np.median(np.diff(t))
    print(f"{len(trials)} trials, {len(t)} samples, fs={fs:.1f} Hz")
    return trials, t, x, fs


def speed_histogram(t, x, fs, v_th=V_TH, v_rest=V_REST, smooth_window=0.05,
                    Dt=0.2, x_excursion_min=8.0, title_prefix=''):
    """Plot smoothed-speed histogram, overall and coloured by state."""
    v = velocity(t, x)
    n_smooth = max(1, int(round(smooth_window * fs)))
    speed = _rolling_rms(v, n_smooth)

    states = _classify_states_schmitt(v, x=x, v_th=v_th, v_rest=v_rest, Dt=Dt,
                                       fs=fs, smooth_window=smooth_window,
                                       x_excursion_min=x_excursion_min)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.hist(speed, bins=200, range=(0, 200), color='gray', alpha=0.7)
    ax.axvline(v_rest, color='steelblue', lw=1.5, label=f'v_rest={v_rest}')
    ax.axvline(v_th,   color='tomato',    lw=1.5, label=f'v_th={v_th}')
    ax.set_xlabel('Smoothed speed (um/s)')
    ax.set_ylabel('Sample count')
    ax.set_title(f'{title_prefix}All samples')
    ax.legend()
    ax.set_yscale('log')

    ax = axes[1]
    colors = {STATE_REST: 'steelblue', STATE_DRIFT: 'gray', STATE_MOVE: 'tomato'}
    labels = {STATE_REST: 'REST', STATE_DRIFT: 'DRIFT', STATE_MOVE: 'MOVE'}
    bins = np.linspace(0, 200, 201)
    for state, color in colors.items():
        mask = states == state
        ax.hist(speed[mask], bins=bins, color=color, alpha=0.5,
                label=f'{labels[state]} (n={mask.sum()})')
    ax.axvline(v_rest, color='steelblue', lw=1.5, ls='--')
    ax.axvline(v_th,   color='tomato',    lw=1.5, ls='--')
    ax.set_xlabel('Smoothed speed (um/s)')
    ax.set_title(f'{title_prefix}By state')
    ax.legend()
    ax.set_yscale('log')

    plt.tight_layout()
    plt.show()

    print(f"REST: {(states==STATE_REST).mean():.1%}  "
          f"DRIFT: {(states==STATE_DRIFT).mean():.1%}  "
          f"MOVE: {(states==STATE_MOVE).mean():.1%}")


def plot_trial_states(trials, v_th=V_TH, v_rest=V_REST, smooth_window=0.05,
                      Dt=0.2, x_excursion_min=8.0, tr_start=0, n_plot=10):
    """Plot position traces coloured by state for a batch of trials."""
    colors = {STATE_REST: 'steelblue', STATE_DRIFT: 'gray', STATE_MOVE: 'tomato'}

    fig, axes = plt.subplots(n_plot, 1, figsize=(12, 1.8 * n_plot), sharex=False)
    for ax, tr in zip(axes, trials[tr_start:tr_start + n_plot]):
        idx = tr.downsample_probe
        t_tr = tr.time[idx].squeeze()
        x_tr = -(tr.probe_position[idx].squeeze() - tr.probeZero)

        v_tr = velocity(t_tr, x_tr)
        fs_tr = 1.0 / np.median(np.diff(t_tr))
        st_tr = _classify_states_schmitt(v_tr, x=x_tr, v_th=v_th, v_rest=v_rest,
                                          Dt=Dt, fs=fs_tr, smooth_window=smooth_window,
                                          x_excursion_min=x_excursion_min)
        for state, color in colors.items():
            mask = st_tr == state
            ax.scatter(t_tr[mask], x_tr[mask], s=1, c=color, rasterized=True)
        ax.set_ylabel('pos (um)', fontsize=8)
        ax.set_title(f'Trial {tr.params["trial"]}', fontsize=8, loc='left')
        ax.axhline(0, color='k', lw=0.5, ls='--')

    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.show()


def plot_trial_detail(trial, v_th=V_TH, v_rest=V_REST, smooth_window=0.05,
                      Dt=0.2, x_excursion_min=8.0):
    """Position, velocity, and rolling RMS for a single trial."""
    idx = trial.downsample_probe
    t6 = trial.time[idx].squeeze()
    x6 = -(trial.probe_position[idx].squeeze() - trial.probeZero)

    v6 = velocity(t6, x6)
    fs6 = 1.0 / np.median(np.diff(t6))
    n_sm = max(1, int(round(smooth_window * fs6)))
    spd6 = _rolling_rms(v6, n_sm)
    st6 = _classify_states_schmitt(v6, x=x6, v_th=v_th, v_rest=v_rest, Dt=Dt,
                                    fs=fs6, smooth_window=smooth_window,
                                    x_excursion_min=x_excursion_min)

    colors = {STATE_REST: 'steelblue', STATE_DRIFT: 'gray', STATE_MOVE: 'tomato'}
    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)

    ax = axes[0]
    for state, color in colors.items():
        mask = st6 == state
        ax.scatter(t6[mask], x6[mask], s=1, c=color, rasterized=True)
    ax.axhline(0, color='k', lw=0.5, ls='--')
    ax.set_ylabel('Position (um)')
    ax.set_title(f'Trial {trial.params.get("trial", "?")}')

    ax = axes[1]
    ax.plot(t6, v6, color='k', lw=0.5)
    ax.axhline(0, color='k', lw=0.5, ls='--')
    ax.set_ylabel('Velocity (um/s)')

    ax = axes[2]
    ax.plot(t6, spd6, color='k', lw=0.8)
    ax.axhline(v_rest, color='steelblue', lw=1.5, ls='--', label=f'v_rest={v_rest}')
    ax.axhline(v_th,   color='tomato',    lw=1.5, ls='--', label=f'v_th={v_th}')
    ax.set_ylabel('Rolling RMS speed (um/s)')
    ax.set_xlabel('Time (s)')
    ax.set_ylim([0, 5 * v_th])
    ax.legend()

    plt.tight_layout()
    plt.show()


def sweep_detection_params(T, v_th_values=None, v_rest_factor=0.67,
                           x_excursion_min=8.0, smooth_window=0.05, Dt=0.2):
    """
    Sweep v_th over hard-success trials and report fraction with a bout detected.
    v_rest = v_th * v_rest_factor (keeps the ratio constant).
    """
    if v_th_values is None:
        v_th_values = list(range(20, 200, 10))

    T.find_successful_trials()
    hard_rows = T.df.loc[T.df['hard_success']].copy()
    hard_rows['next_trial'] = T.df['Trial'].shift(-1).loc[hard_rows.index]
    n_total = len(hard_rows)

    results = []
    for v_th in v_th_values:
        v_rest = round(v_th * v_rest_factor, 1)
        n_found = 0
        for t_num, row in hard_rows.iterrows():
            t_arr, x_arr = _build_trace(row['Trial'], row['next_trial'])
            bouts, _, _, _ = detect_movement_bouts(
                t_arr, x_arr,
                start_time=0,
                v_th=v_th, v_rest=v_rest, Dt=Dt,
                x_excursion_min=x_excursion_min,
                smooth_window=smooth_window,
            )
            if bouts:
                n_found += 1
        results.append({'v_th': v_th, 'v_rest': v_rest,
                        'n_found': n_found, 'n_total': n_total,
                        'fraction': round(n_found / n_total, 3)})

    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    return df


def browse_hard_successes(T, v_th=V_TH, v_rest=V_REST, n_show=20,
                          out_dir=None, show_first=3):
    """Save plot_bouts HTML/PNG for each hard-success trial."""
    T.find_successful_trials()
    hard_rows = T.df.loc[T.df['hard_success']].copy()
    hard_rows['next_trial'] = T.df['Trial'].shift(-1).loc[hard_rows.index]

    if out_dir is None:
        out_dir = os.path.join('Notes', f'successes_{T.dfc}')
    os.makedirs(out_dir, exist_ok=True)

    for idx in range(min(n_show, len(hard_rows))):
        row = hard_rows.iloc[idx]
        t_num = row.name
        trial1, trial2 = row['Trial'], row['next_trial']

        bouts, states, t_b, x_b = detect_bouts_across_trials(
            [trial1, trial2], v_th=v_th, v_rest=v_rest)

        fig = plot_bouts(bouts, states, t_b, x_b,
                         trials=[trial1, trial2],
                         v_th=v_th, v_rest=v_rest,
                         title=f'Trial {t_num} (hard success)',
                         show_grid=False, show_drift_patches=True,
                         cum_vrms_all_bouts=True)

        fig.write_image(os.path.join(out_dir, f'trial_{t_num}_bouts.png'))
        if idx < show_first:
            fig.show()

    print(f"Saved {min(n_show, len(hard_rows))} figures to {out_dir}/")


# ── Single-trial inspection (original run function) ─────────────────────────

def run(save_html=True):
    T = mapd.Table(PARQUET)
    T.extract_trial_properties()
    T.find_successful_trials()

    hard_rows = T.df.loc[T.df['hard_success']].copy()
    hard_rows['next_trial'] = T.df['Trial'].shift(-1).loc[hard_rows.index]

    row = hard_rows.loc[TRIAL_NUM]
    trial1 = row['Trial']
    trial2 = row['next_trial']

    print(f'v_th={V_TH}  v_rest={V_REST}')
    print(f'Trial {TRIAL_NUM}: duration={trial1.total_duration:.3f}s')

    bouts, states, t, x = kin.detect_bouts_across_trials(
        [trial1, trial2],
        v_th=V_TH, v_rest=V_REST,
    )

    print(f'{len(bouts)} bout(s) detected')
    for i, b in enumerate(bouts):
        print(f'  bout {i}: t={b["start_time"]:.2f}–{b["end_time"]:.2f}s  '
              f'dur={b["duration"]:.2f}s  from_rest={b["started_from_rest"]}')

    # Cumulative metrics
    metrics = kin.bout_cumulative_metrics(bouts, t, x, trials=[trial1, trial2])
    print(f'v_rms={metrics["v_rms"]:.1f} um/s  effort={metrics["effort"]:.1f} uN*um')

    # Figure
    fig = kin.plot_bouts(
        bouts, states, t, x,
        trials=[trial1, trial2],
        v_th=V_TH, v_rest=V_REST,
        show_grid=True,
        show_drift_patches=True,
        cum_vrms_all_bouts=True,
    )

    if save_html:
        fig.write_html(OUT_HTML)
        print(f'Figure saved: {OUT_HTML}')

    return fig


if __name__ == '__main__':
    run()
