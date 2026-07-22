"""Reusable primitives for movement-bout / firing-rate analyses.

Per-frame and per-bout feature extraction, cross-correlation between rate and
velocity, target-zone force binning, and bout/stim-aligned segment stacking.
Figure-bound plotting and saving live alongside the notebooks that use them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import ephys
from . import kinematics as kin

SIGMA_S = 0.025  # 25 ms - full Gaussian ~ 100 ms


# ---------------------------------------------------------------------------
# Per-frame records (one row per camera frame, with rate + bout state)
# ---------------------------------------------------------------------------

def per_frame_records(trial, sigma_s=SIGMA_S, trim_edges_s=0.075, **bout_kwargs):
    """Return DataFrame of {t, x, state, rate} sampled at camera frames.
    Returns None if the trial has no detected spikes or no probe data.

    Drops the first/last ``trim_edges_s`` of the trial from the returned
    frame so the Gaussian-smoothed rate is only kept where the kernel has
    full coverage on both sides (avoids the rate dip at trial edges).
    Bout detection runs on the full trace before trimming, so bouts that
    straddle the boundary are still classified correctly.
    Pass ``trim_edges_s=0`` or ``None`` to disable trimming.
    """
    ds = trial.downsample_probe
    t = np.asarray(trial.time)[ds].squeeze()
    x = -(np.asarray(trial.probe_position).squeeze()[ds] - trial.probeZero)
    if t.size < 4:
        return None
    _, states, _, _ = kin.detect_movement_bouts(t, x, **bout_kwargs)
    rate_result = ephys.spike_rate_from_trial(trial, t_axis=t, sigma_s=sigma_s)
    if rate_result is None:
        return None
    _, rate = rate_result
    if trim_edges_s:
        dt = float(np.median(np.diff(t)))
        n_trim = int(np.ceil(trim_edges_s / dt))
        if 2 * n_trim >= len(t):
            return None
        sl = slice(n_trim, len(t) - n_trim)
        t, x, states, rate = t[sl], x[sl], states[sl], rate[sl]
    return pd.DataFrame({'t': t, 'x': x, 'state': states, 'rate': rate})


def trial_state_positions(trial, percentiles=(25, 50, 75),
                          t_min=None, t_max=None, **bout_kwargs):
    """Per-state probe-position percentiles for a single trial.

    Classifies the trial's downsampled trace via ``kin.detect_movement_bouts``
    and returns position percentiles within each of REST / DRIFT / MOVE.
    Positions are reported in the flipped ``-(probe_position - probeZero)``
    frame (positive = toward target), matching the convention used elsewhere
    in ``mapd``.

    Parameters
    ----------
    trial         : Trial object
    percentiles   : iterable of percentiles in [0, 100]
    t_min, t_max  : optional time-window restriction (s, trial time) applied
                    before computing percentiles. State classification still
                    runs on the full trace.
    bout_kwargs   : forwarded to ``kin.detect_movement_bouts``.
                    Note ``start_time`` defaults to 0.0; pass ``start_time=None``
                    if you want bout detection over pre-stim too.

    Returns
    -------
    dict with keys ``rest_p{p}``, ``drift_p{p}``, ``move_p{p}`` for each p
    in ``percentiles``, plus ``rest_n``, ``drift_n``, ``move_n`` sample counts.
    """
    percentiles = np.atleast_1d(percentiles).tolist()
    if trial is None or getattr(trial, 'excluded', False):
        return _empty_state_positions(percentiles)
    ds = trial.downsample_probe
    t = np.asarray(trial.time)[ds].squeeze()
    x = -(np.asarray(trial.probe_position).squeeze()[ds] - trial.probeZero)
    if t.size < 4:
        return _empty_state_positions(percentiles)
    _, states, _, _ = kin.detect_movement_bouts(t, x, **bout_kwargs)
    sel = np.ones_like(t, dtype=bool)
    if t_min is not None:
        sel &= (t >= t_min)
    if t_max is not None:
        sel &= (t <= t_max)
    out = {}
    for name, sv in (('rest', kin.STATE_REST),
                     ('drift', kin.STATE_DRIFT),
                     ('move', kin.STATE_MOVE)):
        mask = sel & (states == sv)
        out[f'{name}_n'] = int(mask.sum())
        if mask.any():
            xs = x[mask]
            for p in percentiles:
                out[f'{name}_p{int(p)}'] = float(np.percentile(xs, p))
        else:
            for p in percentiles:
                out[f'{name}_p{int(p)}'] = np.nan
    return out


def _empty_state_positions(percentiles):
    percentiles = np.atleast_1d(percentiles).tolist()
    out = {}
    for name in ('rest', 'drift', 'move'):
        out[f'{name}_n'] = 0
        for p in percentiles:
            out[f'{name}_p{int(p)}'] = np.nan
    return out


def collect_cell_records(T, sigma_s=SIGMA_S, trim_edges_s=0.075, **bout_kwargs):
    """Concatenate per-frame records across every non-excluded trial in T."""
    frames = []
    for tn, trial in T.df['Trial'].items():
        if trial is None or getattr(trial, 'excluded', False):
            continue
        rec = per_frame_records(trial, sigma_s=sigma_s,
                                trim_edges_s=trim_edges_s, **bout_kwargs)
        if rec is None:
            continue
        rec['trial'] = tn
        frames.append(rec)
    if not frames:
        return pd.DataFrame(columns=['t', 'x', 'state', 'rate', 'trial'])
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Cross-correlation: rate(t) vs. v(t + tau)
# ---------------------------------------------------------------------------

def _zscore(a):
    a = np.asarray(a, dtype=float)
    s = a.std()
    return (a - a.mean()) / (s if s > 0 else 1.0)


def trial_ccf(rate, v, dt, lags_samp):
    """Pearson cross-correlation of (z-scored) rate and v at each integer
    sample lag. Positive lag = rate at t correlates with v at t + lag
    (rate leads movement).
    """
    rz = _zscore(rate)
    vz = _zscore(v)
    n = len(rate)
    out = np.full(len(lags_samp), np.nan)
    for i, k in enumerate(lags_samp):
        if abs(k) >= n:
            continue
        if k >= 0:
            r1, r2 = rz[: n - k], vz[k:]
        else:
            r1, r2 = rz[-k:], vz[: n + k]
        out[i] = np.mean(r1 * r2)
    return out


def _state_runs(g, state):
    """Yield contiguous DataFrame slices of g where g['state'] == state."""
    yield from _mask_runs(g, g['state'].to_numpy() == state)


def _mask_runs(g, mask):
    """Yield contiguous DataFrame slices of g where bool ``mask`` is True."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return
    diff = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    for s, e in zip(starts, ends):
        yield g.iloc[s:e]


def per_trial_ccfs(records, lag_window_s=0.5, signal_fn=None, state=None):
    """Return (lags_s, ccfs_by_trial dict). signal_fn maps a trial group
    -> 1D movement signal aligned to its rate. Defaults to signed v.

    When *state* is set (e.g. ``kin.STATE_MOVE``), each trial is split into
    contiguous runs of that state and the per-run CCFs are averaged into
    one CCF per trial. Trials with no run long enough to span the lag
    window are dropped.
    """
    if signal_fn is None:
        signal_fn = lambda g: np.gradient(g['x'].to_numpy(), g['t'].to_numpy())
    grp = records.groupby('trial', sort=True)
    dt = float(np.median([np.median(np.diff(g['t'])) for _, g in grp]))
    half = int(round(lag_window_s / dt))
    lags_samp = np.arange(-half, half + 1)
    lags_s = lags_samp * dt
    min_n = 4 * half + 1
    ccfs = {}
    for tn, g in grp:
        if state is None:
            if len(g) < min_n:
                continue
            ccfs[tn] = trial_ccf(g['rate'].to_numpy(), signal_fn(g),
                                 dt, lags_samp)
            continue
        run_ccfs = []
        for run in _state_runs(g, state):
            if len(run) < min_n:
                continue
            run_ccfs.append(trial_ccf(run['rate'].to_numpy(), signal_fn(run),
                                      dt, lags_samp))
        if run_ccfs:
            ccfs[tn] = np.nanmean(np.stack(run_ccfs), axis=0)
    return lags_s, ccfs


def per_trial_ccfs_masked(records, lag_window_s=0.25, signal_fn=None,
                          mask_fn=None):
    """Per-trial CCFs over contiguous slices selected by ``mask_fn(g)``.

    ``mask_fn(g) -> bool array`` of length ``len(g)``. Within each trial,
    contiguous True-runs of the mask are CCFed individually; per-run CCFs
    are averaged into a single per-trial CCF. Trials with no qualifying run
    are dropped.
    """
    if signal_fn is None:
        signal_fn = lambda g: np.gradient(g['x'].to_numpy(), g['t'].to_numpy())
    if mask_fn is None:
        mask_fn = lambda g: np.ones(len(g), dtype=bool)
    grp = records.groupby('trial', sort=True)
    dt = float(np.median([np.median(np.diff(g['t'])) for _, g in grp]))
    half = int(round(lag_window_s / dt))
    lags_samp = np.arange(-half, half + 1)
    lags_s = lags_samp * dt
    min_n = 4 * half + 1
    ccfs = {}
    for tn, g in grp:
        mask = np.asarray(mask_fn(g), dtype=bool)
        run_ccfs = []
        for run in _mask_runs(g, mask):
            if len(run) < min_n:
                continue
            r = trial_ccf(run['rate'].to_numpy(), signal_fn(run),
                          dt, lags_samp)
            if np.any(np.isfinite(r)):
                run_ccfs.append(r)
        if run_ccfs:
            ccfs[tn] = np.nanmean(np.stack(run_ccfs), axis=0)
    return lags_s, ccfs


def per_bout_ccfs(records, lag_window_s=0.25, signal_fn=None,
                  state=None, peak_fn=None):
    """One CCF per contiguous state-run (bout).

    Returns ``(lags_s, items)`` where ``items`` is a list of
    ``(peak, ccf, trial_id)`` tuples — one per bout long enough to span the
    lag window. ``peak_fn(run)`` summarises bout strength for downstream
    binning; defaults to ``max(|smoothed signed v|)`` over the run.
    """
    if signal_fn is None:
        signal_fn = lambda g: np.gradient(g['x'].to_numpy(), g['t'].to_numpy())
    if peak_fn is None:
        peak_fn = lambda g: (float(np.max(np.abs(signed_v(g))))
                             if len(g) > 1 else 0.0)
    grp = records.groupby('trial', sort=True)
    dt = float(np.median([np.median(np.diff(g['t'])) for _, g in grp]))
    half = int(round(lag_window_s / dt))
    lags_samp = np.arange(-half, half + 1)
    lags_s = lags_samp * dt
    min_n = 4 * half + 1
    out = []
    for tn, g in grp:
        runs = [g] if state is None else list(_state_runs(g, state))
        for run in runs:
            if len(run) < min_n:
                continue
            ccf = trial_ccf(run['rate'].to_numpy(), signal_fn(run),
                            dt, lags_samp)
            if np.any(np.isfinite(ccf)):
                out.append((peak_fn(run), ccf, tn))
    return lags_s, out


def permutation_null(records, lag_window_s=0.5, signal_fn=None,
                     n_iter=200, rng=None, state=None):
    """Shuffle the rate-vs-velocity trial pairing and recompute the
    averaged CCF. Returns array shape (n_iter, n_lags).

    When *state* is set, the per-trial signals used in the shuffle are the
    longest contiguous run of that state within each trial. Trials whose
    longest run is too short to span the lag window are skipped.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    if signal_fn is None:
        signal_fn = lambda g: np.gradient(g['x'].to_numpy(), g['t'].to_numpy())
    groups = list(records.groupby('trial', sort=True))
    dt = float(np.median([np.median(np.diff(g['t'])) for _, g in groups]))
    half = int(round(lag_window_s / dt))
    lags_samp = np.arange(-half, half + 1)
    null = np.full((n_iter, len(lags_samp)), np.nan)

    rates, vs = [], []
    for _, g in groups:
        if state is None:
            rates.append(g['rate'].to_numpy())
            vs.append(signal_fn(g))
        else:
            longest = max(_state_runs(g, state),
                          key=lambda r: len(r), default=None)
            if longest is None:
                rates.append(np.array([]))
                vs.append(np.array([]))
            else:
                rates.append(longest['rate'].to_numpy())
                vs.append(signal_fn(longest))

    min_n = 4 * half + 1
    for it in range(n_iter):
        perm = rng.permutation(len(groups))
        per = []
        for i, j in enumerate(perm):
            if i == j:
                continue
            n = min(len(rates[i]), len(vs[j]))
            if n < min_n:
                continue
            per.append(trial_ccf(rates[i][:n], vs[j][:n], dt, lags_samp))
        if per:
            null[it] = np.nanmean(np.stack(per), axis=0)
    return null


def cv_lag(ccfs, lags_s, k=5, rng=None):
    """K-fold CV: pick tau* = argmax(mean CCF) on training trials, then
    return the held-out trials' average CCF value at exactly that tau*.

    Returns (taus_train, r_holdout, mean_train_ccf).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    trial_ids = np.array(list(ccfs.keys()))
    mat = np.stack([ccfs[tn] for tn in trial_ids])
    n = len(trial_ids)
    if n < k:
        k = max(2, n)
    order = rng.permutation(n)
    folds = np.array_split(order, k)
    taus_train, r_holdout = [], []
    for fold in folds:
        train_mask = np.ones(n, bool)
        train_mask[fold] = False
        train_mean = np.nanmean(mat[train_mask], axis=0)
        i_star = int(np.nanargmax(train_mean))
        taus_train.append(lags_s[i_star])
        r_holdout.append(np.nanmean(mat[fold][:, i_star]))
    return np.array(taus_train), np.array(r_holdout), np.nanmean(mat, axis=0)


def signed_v(g, smooth_window=0.05):
    return kin.smoothed_velocity(g['t'].to_numpy(), g['x'].to_numpy(),
                                 smooth_window=smooth_window)


def positive_v(g, smooth_window=0.05):
    return np.maximum(signed_v(g, smooth_window), 0.0)


# ---------------------------------------------------------------------------
# Target-zone force regimes
# ---------------------------------------------------------------------------

def force_bin_edges(T):
    """Edges (in flipped-x coords) for 5 regimes:
    below_lo / in_lo / between / in_hi / above_hi.
    """
    def zone(t):
        a = -t['pyasXPosition']
        b = -(t['pyasXPosition'] + t['pyasWidth'])
        return sorted([a, b])
    lo = zone(T.targets['lo'])
    hi = zone(T.targets['hi'])
    if np.mean(lo) > np.mean(hi):
        lo, hi = hi, lo
    edges = [-np.inf, lo[0], lo[1], hi[0], hi[1], np.inf]
    labels = ['below_lo', 'in_lo', 'between', 'in_hi', 'above_hi']
    return edges, labels


def force_bin_edges_3(T):
    """3 force regimes: below_lo / between / above_hi (in-target initial
    positions never produce as_off trials, so the in_* bins are dropped).
    """
    def zone(t):
        a = -t['pyasXPosition']
        b = -(t['pyasXPosition'] + t['pyasWidth'])
        return sorted([a, b])
    lo = zone(T.targets['lo'])
    hi = zone(T.targets['hi'])
    if np.mean(lo) > np.mean(hi):
        lo, hi = hi, lo
    edges = [-np.inf, lo[0], hi[1], np.inf]
    labels = ['below_lo', 'between', 'above_hi']
    return edges, labels


# ---------------------------------------------------------------------------
# Bout-level metadata: initiation position + initial speed
# ---------------------------------------------------------------------------

def collect_bouts_with_meta(T, init_window_s=1.0, **bout_kwargs):
    """One row per movement bout: trial, as_outcome, start_idx, start_time,
    init_pos, init_abs_speed (mean |smoothed v| over first init_window_s).
    """
    rows = []
    for tn, trial in T.df['Trial'].items():
        if trial is None or getattr(trial, 'excluded', False):
            continue
        ds = trial.downsample_probe
        t = np.asarray(trial.time)[ds].squeeze()
        x = -(np.asarray(trial.probe_position).squeeze()[ds] - trial.probeZero)
        if t.size < 4:
            continue
        bouts, _, _, _ = kin.detect_movement_bouts(t, x, **bout_kwargs)
        if not bouts:
            continue
        v_sm = kin.smoothed_velocity(t, x)
        dt = float(np.median(np.diff(t)))
        n_win = max(1, int(round(init_window_s / dt)))
        outcome = T.df.at[tn, 'as_outcome'] if 'as_outcome' in T.df.columns else None
        for b in bouts:
            i0 = int(b['start_idx'])
            i1 = min(i0 + n_win, len(t))
            rows.append({
                'trial': tn,
                'as_outcome': outcome,
                'start_idx': i0,
                'start_time': float(b['start_time']),
                'init_pos': float(x[i0]),
                'init_abs_speed': float(np.mean(np.abs(v_sm[i0:i1]))),
            })
    return pd.DataFrame(rows)


def collect_trial_meta_as_off(T, pre_window_s=0.5):
    """One row per as_off trial: init_pos = mean flipped-x over the
    [-pre_window_s, 0] pre-stim window.
    """
    rows = []
    for tn, trial in T.df['Trial'].items():
        if trial is None or getattr(trial, 'excluded', False):
            continue
        if T.df.at[tn, 'as_outcome'] != 'as_off':
            continue
        ds = trial.downsample_probe
        t = np.asarray(trial.time)[ds].squeeze()
        x = -(np.asarray(trial.probe_position).squeeze()[ds] - trial.probeZero)
        if t.size < 4:
            continue
        mask = (t >= -pre_window_s) & (t <= 0.0)
        if not mask.any():
            continue
        rows.append({'trial': tn, 'init_pos': float(np.mean(x[mask]))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Bout-aligned and stim-aligned segment extraction
# ---------------------------------------------------------------------------

def _cell_dt(T, trial_ids):
    dts = []
    for tn in trial_ids:
        trial = T.df.at[tn, 'Trial']
        if trial is None or getattr(trial, 'excluded', False):
            continue
        ds = trial.downsample_probe
        t = np.asarray(trial.time)[ds].squeeze()
        if t.size > 1:
            dts.append(np.median(np.diff(t)))
    if not dts:
        return None
    return float(np.median(dts))


def bout_aligned_rate_segments(T, bouts_df_subset, sigma_s=SIGMA_S,
                                pre_s=1.0, post_s=0.5):
    """(t_rel, segments) where each segment is the smoothed rate over a
    fixed-length window aligned to bout onset. Bouts whose window would
    run off the trial are dropped.
    """
    dt = _cell_dt(T, bouts_df_subset['trial'].unique())
    if dt is None:
        return None, []
    n_pre = int(round(pre_s / dt))
    n_post = int(round(post_s / dt))
    t_rel = (np.arange(n_pre + n_post + 1) - n_pre) * dt

    out = []
    for tn, sub in bouts_df_subset.groupby('trial', sort=True):
        trial = T.df.at[tn, 'Trial']
        if trial is None or getattr(trial, 'excluded', False):
            continue
        ds = trial.downsample_probe
        t = np.asarray(trial.time)[ds].squeeze()
        rr = ephys.spike_rate_from_trial(trial, t_axis=t, sigma_s=sigma_s)
        if rr is None:
            continue
        _, rate = rr
        for _, row in sub.iterrows():
            i0 = int(row['start_idx'])
            a, b = i0 - n_pre, i0 + n_post + 1
            if a < 0 or b > len(rate):
                continue
            seg = rate[a:b]
            if len(seg) != len(t_rel):
                continue
            out.append((row['force_bin'], row['vel_q'], seg))
    return t_rel, out


def bout_aligned_position_segments(T, bouts_df_subset, pre_s=1.0, post_s=0.5):
    """Same shape contract as bout_aligned_rate_segments but extracts the
    flipped probe position (positive = toward target).
    """
    dt = _cell_dt(T, bouts_df_subset['trial'].unique())
    if dt is None:
        return None, []
    n_pre = int(round(pre_s / dt))
    n_post = int(round(post_s / dt))
    t_rel = (np.arange(n_pre + n_post + 1) - n_pre) * dt

    out = []
    for tn, sub in bouts_df_subset.groupby('trial', sort=True):
        trial = T.df.at[tn, 'Trial']
        if trial is None or getattr(trial, 'excluded', False):
            continue
        ds = trial.downsample_probe
        x = -(np.asarray(trial.probe_position).squeeze()[ds] - trial.probeZero)
        for _, row in sub.iterrows():
            i0 = int(row['start_idx'])
            a, b = i0 - n_pre, i0 + n_post + 1
            if a < 0 or b > len(x):
                continue
            seg = x[a:b]
            if len(seg) != len(t_rel):
                continue
            out.append((row['force_bin'], row['vel_q'], seg))
    return t_rel, out


def stim_aligned_segments(T, trial_meta, signal, sigma_s=SIGMA_S,
                          pre_s=1.0, post_s=3.0):
    """(t_rel, segments) for traces aligned to stim onset (t = 0 in trial time).

    signal : {'rate', 'position'}
    trial_meta : DataFrame with columns ['trial', 'force_bin'].
    """
    dt = _cell_dt(T, trial_meta['trial'].unique())
    if dt is None:
        return None, []
    n_pre = int(round(pre_s / dt))
    n_post = int(round(post_s / dt))
    t_rel = (np.arange(n_pre + n_post + 1) - n_pre) * dt

    out = []
    for _, row in trial_meta.iterrows():
        tn = row['trial']
        trial = T.df.at[tn, 'Trial']
        if trial is None or getattr(trial, 'excluded', False):
            continue
        ds = trial.downsample_probe
        t = np.asarray(trial.time)[ds].squeeze()
        if signal == 'rate':
            rr = ephys.spike_rate_from_trial(trial, t_axis=t, sigma_s=sigma_s)
            if rr is None:
                continue
            _, y = rr
        elif signal == 'position':
            y = -(np.asarray(trial.probe_position).squeeze()[ds] - trial.probeZero)
        else:
            raise ValueError(f'unknown signal: {signal!r}')
        i0 = int(np.argmin(np.abs(t)))
        a, b = i0 - n_pre, i0 + n_post + 1
        if a < 0 or b > len(y):
            continue
        seg = y[a:b]
        if len(seg) != len(t_rel):
            continue
        out.append((row['force_bin'], seg))
    return t_rel, out


def mean_sem(t_rel, segments):
    """Stack segments and return (t_rel, mean, sem). Segments can be either
    (state, seg) 2-tuples or (state, state2, seg) 3-tuples — the last
    element is always the array.
    """
    arr = np.stack([s[-1] for s in segments])
    return t_rel, arr.mean(axis=0), arr.std(axis=0) / np.sqrt(arr.shape[0])
