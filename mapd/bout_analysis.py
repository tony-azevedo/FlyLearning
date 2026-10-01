"""Reusable primitives for movement-bout / firing-rate analyses.

Per-frame and per-bout feature extraction, cross-correlation between rate and
velocity, target-zone force binning, and bout/stim-aligned segment stacking.
Figure-bound plotting and saving live alongside the notebooks that use them.
"""
from __future__ import annotations

from dataclasses import dataclass

import warnings
import numpy as np
import pandas as pd
import gc

from scipy.stats import chi2 as _chi2

from . import ephys
from . import kinematics as kin

SIGMA_S = 0.025  # 25 ms - full Gaussian ~ 100 ms

# Averaging windows for the "does slower rate measurement tighten the cloud?"
# sweep. The first is close to the Gaussian default, the last is longer than
# most rest epochs — so the sweep runs from "faster than the behaviour" to
# "slower than it".
RATE_WINDOWS_S = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0)


def boxcar_rate_specs(windows_s=RATE_WINDOWS_S, prefix='rate_box'):
    """``{column_name: ('box', window_s)}`` for :func:`per_frame_records`."""
    return {f'{prefix}_{int(round(w * 1000))}ms': ('box', float(w))
            for w in windows_s}


def _spec_edge_s(kind, scale):
    """Half-width of trace a rate estimator needs on each side to be unbiased."""
    return 3.0 * float(scale) if kind == 'gauss' else 0.5 * float(scale)


# ---------------------------------------------------------------------------
# Per-frame records (one row per camera frame, with rate + bout state)
# ---------------------------------------------------------------------------

def cue_window(trial, plateau_only=False, fallback=(-0.8, -0.5)):
    """``(t_start, t_end)`` of the piezo cue, in trial time, from ``trial.params``.

    Every trial delivers a brief mechanical cue during the pre-stimulus period.
    It is a known, imposed perturbation of probe position, Vm and firing rate —
    which makes it the natural reference for the *spontaneous* fluctuations
    during a hold, and something never to be mistaken for them.

    The timing is exact from params: the cue **ends** ``cueDelayDurInSec`` before
    stimulus onset and lasts ``cueStimDurInSec``, so

        ``t_end   = -cueDelayDurInSec``
        ``t_start = -cueDelayDurInSec - cueStimDurInSec``

    With the usual 0.5 / 0.3 s that is -0.8 .. -0.5 s. ``cueRampDurInSec`` (70 ms)
    is the rise and fall *inside* that window, which is why a threshold-crossing
    measurement on the ``sgsmonitor`` trace reads slightly narrower: 10% of the
    way up a 70 ms ramp is 7 ms, giving -0.793 / -0.507 against the -0.790 /
    -0.505 actually measured. ``plateau_only=True`` returns the fully displaced
    part, ``t_start + ramp .. t_end - ramp``.

    Note the cue lands ~200 ms into the trial, well inside the 750 ms a 250 ms
    kernel needs for full support — so measuring the cue response at that width
    requires ``pad_spikes=True`` / ``pad_trials=``, or the response sits on the
    kernel's own ramp out of nothing.
    """
    try:
        p = trial.params
        delay = float(p['cueDelayDurInSec'])
        stim = float(p['cueStimDurInSec'])
    except (AttributeError, KeyError, TypeError, ValueError):
        return fallback
    if not (np.isfinite(delay) and np.isfinite(stim)):
        return fallback
    t_start, t_end = -delay - stim, -delay
    if plateau_only:
        try:
            ramp = float(trial.params['cueRampDurInSec'])
        except (KeyError, TypeError, ValueError):
            ramp = 0.0
        if np.isfinite(ramp) and 2 * ramp < stim:
            t_start, t_end = t_start + ramp, t_end - ramp
    return float(t_start), float(t_end)


def fit_position_rate(records, rate_col='rate', x_col='x',
                      state=kin.STATE_REST, mask=None):
    """Linear fit between probe position and firing rate, both directions.

    Both are reported because they answer different questions and, with a noisy
    x, they are not reciprocals:

    ``slope_pos_on_rate``   um per Hz — the line to draw on a panel with rate on
                            x and position on y (the panel's own y-on-x fit).
    ``slope_rate_on_pos``   Hz per um — rate treated as the response.

    Expect the two kernels to disagree in a specific way. Regression dilution
    attenuates the slope of y on a noisy x, and a 25 ms rate is much noisier than
    a 250 ms one, so ``slope_pos_on_rate`` should be *shallower* at 25 ms while
    ``slope_rate_on_pos`` (position is the clean variable) barely moves. If both
    slopes shift together instead, the difference is not measurement noise.

    ``mask`` is an optional boolean over ``records`` applied on top of ``state``
    — pass ``~cue_mask(...) & ~premove`` to exclude the imposed perturbations.
    """
    g = records
    if state is not None:
        g = g[np.isin(g['state'], _state_tuple(state))]
        if mask is not None:
            mask = np.asarray(mask)[np.isin(records['state'], _state_tuple(state))]
    if mask is not None:
        g = g[np.asarray(mask, dtype=bool)]
    g = g.dropna(subset=[rate_col, x_col])
    if len(g) < 10:
        return {}
    r = g[rate_col].to_numpy(dtype=float)
    x = g[x_col].to_numpy(dtype=float)
    if np.std(r) == 0 or np.std(x) == 0:
        return {}
    rho = float(np.corrcoef(r, x)[0, 1])
    a1 = np.polyfit(r, x, 1)          # position on rate  -> um per Hz
    a2 = np.polyfit(x, r, 1)          # rate on position  -> Hz per um
    return {'n': int(len(g)), 'rho': rho, 'r2': rho ** 2,
            'slope_pos_on_rate': float(a1[0]), 'intercept_pos': float(a1[1]),
            'slope_rate_on_pos': float(a2[0]), 'intercept_rate': float(a2[1]),
            'rate_mean': float(np.mean(r)), 'x_mean': float(np.mean(x)),
            'rate_sd': float(np.std(r, ddof=1)), 'x_sd': float(np.std(x, ddof=1))}


def cue_mask(records, pre_s=0.0, post_s=0.2):
    """Boolean mask of the cue window, widened by ``pre_s`` / ``post_s``.

    ``in_cue`` marks only the 300 ms cue itself, but the response outlasts it —
    there is an off-response on the piezo's return ramp, of opposite sign, that
    decays over ~300-400 ms. So excluding "the cue" from a position-vs-rate plot
    means excluding the cue *plus* its aftermath, which is what this returns.
    """
    if {'cue_t0', 'cue_t1'} <= set(records.columns):
        t0 = records['cue_t0'].to_numpy(dtype=float) - pre_s
        t1 = records['cue_t1'].to_numpy(dtype=float) + post_s
        t = records['t'].to_numpy(dtype=float)
        return (t >= t0) & (t <= t1)
    if 'in_cue' in records.columns:
        return records['in_cue'].to_numpy().astype(bool)
    return np.zeros(len(records), dtype=bool)


def cue_displacement(trial):
    """The piezo cue's signed displacement for this trial (``params.displacement``).

    It varies trial to trial over ``params.displacements`` (e.g. -5, -2.5, 2.5,
    5), which is why the cue response nearly vanishes in a trial average even
    though single trials show tens of Hz: opposite displacements cancel. Group by
    this before averaging anything cue-aligned.
    """
    try:
        return float(trial.params['displacement'])
    except (AttributeError, KeyError, TypeError, ValueError):
        return np.nan



#: Measured window of the pre-cue current step, in trial time (seconds).
#:
#: A short hyperpolarizing step is commanded at -0.990 s and released at
#: -0.940 s. Measured trial-averaged on all six A2 cells with
#: :func:`measure_current_step`: the command timing is identical to the
#: millisecond across every cell (-0.990 to -0.940, except 210903_F3_C1 at
#: -0.989/-0.948) and only the amplitude varies, -3.4 to -7.5 pA displacing Vm
#: -1.3 to -4.5 mV. Vm is back within 10% of rest by -0.913 at the latest, so
#: the window closes at -0.905 to leave a few ms of margin -- do not tighten it
#: to the nominal -0.915, which clips the recovery on 210908 and 210915.
#: It is *not* described in ``trial.params``
#: (mode/gain are zero there) -- only ``current_1`` shows it -- so this is a
#: measured constant rather than a read one, and :func:`measure_current_step`
#: re-derives it from a Table when that needs checking.
#:
#: The step ends well before the cue at -0.800, so a 25 ms rate kernel keeps
#: the two cleanly separated. A 250 ms kernel does not: it draws on +-2 sigma, so
#: every g250 sample inside the cue window has support reaching back past
#: -1.000 s and is contaminated by the step. Cue analysis therefore has to stay
#: at sigma <= 50 ms.
CURRENT_STEP_WINDOW = (-0.990, -0.905)



#: Cue ramp duration, ``params.cueRampDurInSec``. Verified at 70 ms on
#: 210602_F1_C1, 210915_F1_C1 and 240430_F1_C1; pass ``ramp_s`` explicitly for a
#: protocol that differs.
CUE_RAMP_S = 0.07


def add_cue_command(records, ramp_s=CUE_RAMP_S, out_col='cue_cmd'):
    """The *commanded* cue waveform as a records column, from params alone.

    Straight from the acquisition code: the cue is a plateau ``cueStimDurInSec``
    long starting ``cueStimDurInSec + cueDelayDurInSec`` before the stimulus,
    ramped linearly over ``cueRampDurInSec`` at each end (a symmetric triangular
    window) and scaled by the trial's signed ``displacement``.

    Prefer this over any measured channel when timing the cue. The probe trace
    mixes the piezo with the fly's own movement, and ``sgsmonitor`` — though it
    tracks the command at r = 0.999 and lags it by only 2-3 ms — is still a
    measurement with a noise floor and an onset that has to be detected. The
    command has neither: its onset is an exact instant, known from params, and
    its pre-onset baseline is identically zero, so a baseline-departure threshold
    resolves the onset to a single frame.

    Requires ``cue_t0``, ``cue_t1`` and ``cue_displacement`` in ``records`` (all
    written by :func:`per_frame_records`). The sign follows ``displacement``,
    which varies trial to trial over ``params.displacements``, so a cue-aligned
    average of this column is near zero by construction — group by the sign
    before averaging.
    """
    need = {'t', 'cue_t0', 'cue_t1'}
    missing = need - set(records.columns)
    if missing:
        raise KeyError(f'add_cue_command needs {sorted(missing)} in records; '
                       'build them with per_frame_records(mark_cue=True)')
    t = records['t'].to_numpy(dtype=float)
    t0 = records['cue_t0'].to_numpy(dtype=float)
    t1 = records['cue_t1'].to_numpy(dtype=float)
    disp = (records['cue_displacement'].to_numpy(dtype=float)
            if 'cue_displacement' in records.columns
            else np.ones(len(records), dtype=float))
    stim = t1 - t0
    u = t - t0
    y = np.zeros(len(records), dtype=float)
    inside = np.isfinite(u) & np.isfinite(stim) & (u >= 0) & (u <= stim)
    y[inside] = 1.0
    if ramp_s and ramp_s > 0:
        rise = inside & (u < ramp_s)
        y[rise] = u[rise] / ramp_s
        fall = inside & (u > stim - ramp_s)
        y[fall] = (stim[fall] - u[fall]) / ramp_s
    return records.assign(**{out_col: y * disp})

def current_step_mask(records, pre_s=0.0, post_s=0.0, window=None):
    """Boolean mask of the pre-cue current step, widened by ``pre_s``/``post_s``.

    Mirrors :func:`cue_mask`. The default window runs from the command onset to
    the point where Vm has recovered, so it covers the imposed transient and its
    decay rather than only the current command itself.
    """
    t = records['t'].to_numpy(dtype=float)
    lo, hi = CURRENT_STEP_WINDOW if window is None else window
    return (t >= lo - pre_s) & (t <= hi + post_s)


def current_step_events(records, window=None):
    """One row per trial: ``trial`` and ``t0`` = current-step onset.

    Feeds :func:`event_latency` the way :func:`cue_events` does. This is the
    control the cue cannot supply: the step displaces Vm and the firing rate with
    the probe held still, so it tests the *other* direction of the estimator. The
    cue establishes that an imposed probe movement is timed as probe-first; the
    step establishes that an imposed rate change with no movement yields no probe
    response. A latency measured here is one the method manufactured.
    """
    lo = (CURRENT_STEP_WINDOW if window is None else window)[0]
    rows = [{'trial': tn, 't0': lo}
            for tn, g in records.groupby('trial', sort=True)
            if float(g['t'].min()) <= lo]
    return pd.DataFrame(rows, columns=['trial', 't0'])


def measure_current_step(trials, n_trials=60, thresh_frac=0.3, min_pa=0.4):
    """Measure the pre-cue current step from ``current_1``, trial-averaged.

    Use this to verify :data:`CURRENT_STEP_WINDOW` on a cell rather than assuming
    it. ``trials`` is a :class:`~mapd.table.Table` or any iterable of Trials.

    Two things matter for getting this right, both learned by getting them wrong.
    The holding current is not zero -- it runs about -27 to -35 pA -- so the step
    must be measured against a robust estimate of the holding level, the median of
    the whole trace, and not against a window just before the cue, which sits on
    the step's own settling tail and reports a spuriously long, shallow step. And
    the search must not be capped at the cue onset, or a sub-pA tail crossing
    threshold is reported as a step running exactly up to the cap.

    Returns a dict with ``on_s``, ``off_s``, ``dur_ms``, ``I_pA``, ``dVm_mV``,
    ``vm_recover_s`` and ``n_trials``; the measurements are NaN if no step is
    found.
    """
    if hasattr(trials, 'df'):
        trs = [t for _, t in trials.df['Trial'].items() if t is not None]
    else:
        trs = [t for t in trials if t is not None]
    cur, vm, tt = [], [], None
    for tr in trs[:n_trials]:
        try:
            i_ = np.asarray(tr.current_1, dtype=float).ravel()
            v_ = np.asarray(tr.voltage_1, dtype=float).ravel()
            t_ = np.asarray(tr.trialtime, dtype=float).ravel()
        except (AttributeError, KeyError, TypeError, ValueError):
            continue
        tt = t_ if tt is None else tt
        n = min(len(t_), len(tt), len(i_), len(v_))
        cur.append(i_[:n])
        vm.append(v_[:n])
        tt = tt[:n]
    out = {'on_s': np.nan, 'off_s': np.nan, 'dur_ms': np.nan, 'I_pA': np.nan,
           'dVm_mV': np.nan, 'vm_recover_s': np.nan, 'n_trials': len(cur)}
    if not cur:
        return out
    n = min(len(c) for c in cur)
    cur_m = np.mean([c[:n] for c in cur], axis=0)
    vm_m = np.mean([v[:n] for v in vm], axis=0)
    t = np.asarray(tt[:n], dtype=float)
    d = cur_m - float(np.median(cur_m))
    thr = max(thresh_frac * float(np.nanmax(np.abs(d))), min_pa)
    best = None
    for a, b, v in _rle_states(np.abs(d) > thr):
        if not v or t[a] > 0.0 or (b - a) < 5:
            continue
        if best is None or (b - a) > (best[1] - best[0]):
            best = (a, b)
    if best is None:
        return out
    a, b = best
    on_s, off_s = float(t[a]), float(t[b - 1])
    # Reference Vm from the quiet stretch AFTER the step: before it there are only
    # ~10 ms of recording, too little to average.
    quiet = (t > off_s + 0.05) & (t < off_s + 0.14)
    v_ref = float(np.median(vm_m[quiet])) if quiet.sum() > 3 else np.nan
    t_rec = np.nan
    if np.isfinite(v_ref):
        pk = float(np.nanmax(np.abs(vm_m[a:b] - v_ref)))
        back = [i for i in np.where(t > off_s)[0]
                if abs(vm_m[i] - v_ref) < 0.1 * pk]
        if back:
            t_rec = float(t[back[0]])
    out.update({'on_s': on_s, 'off_s': off_s, 'dur_ms': (off_s - on_s) * 1e3,
                'I_pA': float(np.mean(d[a:b])),
                'dVm_mV': (float(np.mean(vm_m[a:b]) - v_ref)
                           if np.isfinite(v_ref) else np.nan),
                'vm_recover_s': t_rec})
    return out

def cue_aligned_segments(records, pre_s=0.3, post_s=0.9, align='onset',
                         dt=0.005, cols=('rate', 'vm'), baseline_s=0.2):
    """Cue-aligned segments of ``cols``, one row per trial, on a common grid.

    ``align='onset'`` puts t=0 at the cue's start, ``'offset'`` at its end — the
    latter is the one for an off-response, since the piezo's return ramp is what
    triggers it and the cue's *duration* is what would otherwise smear it.

    ``post_s`` deliberately extends well past the cue: the response can outlast
    the 300 ms stimulus, and there is often a separate off-response on the return
    ramp, so a window clipped to the cue itself would report neither.

    ``baseline_s`` subtracts each trial's mean over the ``baseline_s`` immediately
    **before the cue onset**, whichever alignment is used — never before the
    alignment point. Aligning to the offset and baselining before *that* would put
    the baseline window inside the cue plateau, so every later sample would be
    measured against the response instead of against rest, and the off-response
    would come out as a spurious return-to-baseline of the opposite sign.

    Note the pre-cue period is short: the cue ends ``cueDelayDurInSec`` (0.5 s)
    before stimulus onset and the trial starts at ``-preDurInSec`` (-1.0 s), so
    there are only ~200 ms between the trial's first sample and the cue. Ask for
    more ``pre_s`` than that with ``align='onset'`` and no trial can cover the
    window; the returned ``meta`` is then empty and ``n_dropped`` says why.

    Returns ``(t_rel, stacks, meta)``: ``t_rel`` the common time base, ``stacks``
    a dict ``col -> (n_trials, n_time)`` array of baseline-subtracted values, and
    ``meta`` a DataFrame with ``trial`` and ``cue_displacement`` per row, carrying
    ``meta.attrs['n_dropped']``.

    Group by ``cue_displacement`` before averaging. The commanded displacement
    alternates in sign trial to trial, so responses of opposite sign cancel and a
    pooled average understates the effect by an order of magnitude.
    """
    need = {'cue_t0', 'cue_t1', 'cue_displacement'}
    if not need <= set(records.columns):
        raise KeyError(f'records need {sorted(need)} — rebuild with '
                       f'per_frame_records(..., mark_cue=True) (the default)')
    cols = [c for c in np.atleast_1d(cols).tolist() if c in records.columns]
    t_rel = np.arange(-pre_s, post_s + 0.5 * dt, dt)
    stacks = {c: [] for c in cols}
    rows = []
    n_dropped = 0
    for tn, g in records.groupby('trial', sort=True):
        c0, c1 = float(g['cue_t0'].iloc[0]), float(g['cue_t1'].iloc[0])
        if not (np.isfinite(c0) and np.isfinite(c1)):
            n_dropped += 1
            continue
        origin = c0 if align == 'onset' else c1
        t_trial = g['t'].to_numpy()
        t = t_trial - origin
        if t.min() > -pre_s or t.max() < post_s:
            n_dropped += 1
            continue                      # window not fully covered by this trial
        # Baseline is always the pre-*cue* period, in trial time.
        base_m = (t_trial >= c0 - baseline_s) & (t_trial < c0)
        ok = True
        seg = {}
        for c in cols:
            y = g[c].to_numpy(dtype=float)
            if not np.isfinite(y).any():
                ok = False
                break
            b = float(np.nanmean(y[base_m])) if base_m.any() else 0.0
            seg[c] = np.interp(t_rel, t, y - b, left=np.nan, right=np.nan)
        if not ok:
            n_dropped += 1
            continue
        for c in cols:
            stacks[c].append(seg[c])
        rows.append({'trial': tn,
                     'cue_displacement': float(g['cue_displacement'].iloc[0]),
                     'cue_dur': c1 - c0})
    if not rows:
        empty = pd.DataFrame(columns=['trial', 'cue_displacement', 'cue_dur'])
        empty.attrs['n_dropped'] = n_dropped
        return t_rel, {c: np.zeros((0, len(t_rel))) for c in cols}, empty
    meta = pd.DataFrame(rows)
    meta.attrs['n_dropped'] = n_dropped
    return t_rel, {c: np.stack(v) for c, v in stacks.items()}, meta


def cue_response_windows(records, windows=None, align='onset', cols=('rate', 'vm'),
                         baseline_s=0.2, by='cue_displacement'):
    """Cue response measured over several windows, to find where it actually lives.

    ``windows`` is ``{label: (t_start, t_end)}`` relative to the alignment point.
    The default set walks from the onset ramp through the sustained part and past
    the offset, because the response is not confined to the stimulus: it can
    outlast it, and the piezo's return ramp can produce a separate off-response of
    the opposite sign.

    Returns a long DataFrame with one row per (window x group x column):
    ``window``, ``t_start``, ``t_end``, the ``by`` value, ``n``, ``mean``, ``sem``,
    plus ``slope_vs_disp`` — the regression of the response on displacement within
    that window, which is the single number saying whether the window contains a
    displacement-dependent response at all (and its sign).
    """
    if windows is None:
        # Defaults follow the alignment: with t=0 at the cue's end, "on ramp" and
        # "sustained" are at negative times and the interesting part is after 0.
        windows = {
            'pre':            (-0.20, 0.00),
            'on ramp':        (0.00, 0.07),
            'early':          (0.00, 0.10),
            'sustained':      (0.10, 0.30),
            'off ramp':       (0.30, 0.37),
            'post 0-0.2':     (0.30, 0.50),
            'post 0.2-0.5':   (0.50, 0.80),
            'post 0.5-1.0':   (0.80, 1.30),
        } if align == 'onset' else {
            'cue early':      (-0.30, -0.20),
            'cue late':       (-0.20, 0.00),
            'off ramp':       (0.00, 0.07),
            'off 0-0.1':      (0.00, 0.10),
            'off 0.1-0.3':    (0.10, 0.30),
            'off 0.3-0.6':    (0.30, 0.60),
            'off 0.6-1.0':    (0.60, 1.00),
        }
    span = max(v for w in windows.values() for v in w)
    lead = max(-min(v for w in windows.values() for v in w), 0.0)
    # Only the pre-cue baseline needs headroom before the alignment point; with
    # align='onset' the trial itself offers only ~200 ms there.
    pre_needed = (baseline_s + 0.02 if align == 'onset'
                  else max(lead, baseline_s + 0.32))
    t_rel, stacks, meta = cue_aligned_segments(
        records, pre_s=pre_needed, post_s=span + 0.05,
        align=align, cols=cols, baseline_s=baseline_s)
    rows = []
    if not len(meta):
        return pd.DataFrame(columns=['window', 't_start', 't_end', 'col', by,
                                     'n', 'mean', 'sem', 'slope_vs_disp',
                                     'rho_vs_disp'])
    for label, (a, b) in windows.items():
        m = (t_rel >= a) & (t_rel < b)
        if not m.any():
            continue
        for c, arr in stacks.items():
            vals = np.nanmean(arr[:, m], axis=1)
            for key, idx in meta.groupby(by).groups.items():
                v = vals[meta.index.get_indexer(idx)]
                v = v[np.isfinite(v)]
                if not len(v):
                    continue
                rows.append({'window': label, 't_start': a, 't_end': b,
                             'col': c, by: key, 'n': len(v),
                             'mean': float(np.mean(v)),
                             'sem': float(np.std(v, ddof=1) / np.sqrt(len(v)))
                             if len(v) > 1 else np.nan,
                             # Always present, even when it cannot be computed:
                             # a single trial (or one displacement) gives no slope,
                             # and a column that appears only sometimes breaks
                             # every caller that selects it.
                             'slope_vs_disp': np.nan, 'rho_vs_disp': np.nan})
            d = meta[by].to_numpy(dtype=float)
            good = np.isfinite(vals) & np.isfinite(d)
            if good.sum() > 2 and np.std(d[good]) > 0:
                slope = float(np.polyfit(d[good], vals[good], 1)[0])
                rho = float(np.corrcoef(d[good], vals[good])[0, 1])
                for r in rows:
                    if r['window'] == label and r['col'] == c:
                        r['slope_vs_disp'] = slope
                        r['rho_vs_disp'] = rho
    return pd.DataFrame(rows)


def neighbour_trials(T, tn, require_ephys_ok=True):
    """``(prev, next)`` Trial objects for spike padding, or ``None`` where unusable.

    A neighbour qualifies only if it is the *consecutive* trial number and is
    usable — present, not excluded, and (by default) not flagged bad ephys. Any
    other trial would splice spikes from a different stretch of recording across
    the boundary, which is worse than the edge falloff padding is meant to fix.
    A gap in the numbering means a trial was dropped, so the recording is not
    continuous there either.
    """
    def _ok(k):
        if k not in T.df.index:
            return None
        trial = T.df.at[k, 'Trial']
        if trial is None or getattr(trial, 'excluded', False):
            return None
        if require_ephys_ok and not getattr(trial, 'ephys_ok', True):
            return None
        return trial
    return _ok(tn - 1), _ok(tn + 1)


def per_frame_records(trial, sigma_s=SIGMA_S, trim_edges_s=0.075,
                      rate_specs=None, with_vm=False, vm_kwargs=None,
                      vm_control=False, vm_specs=None, pad_trials=None,
                      mark_cue=True, cue_margin_s=0.0, raw_channels=None,
                      keep_silent=False, **bout_kwargs):
    """Return DataFrame of {t, x, state, rate, n_sp} sampled at camera frames.
    Returns None if the trial has no detected spikes or no probe data.

    Drops the first/last ``trim_edges_s`` of the trial from the returned
    frame so the Gaussian-smoothed rate is only kept where the kernel has
    full coverage on both sides (avoids the rate dip at trial edges).
    Bout detection runs on the full trace before trimming, so bouts that
    straddle the boundary are still classified correctly.
    Pass ``trim_edges_s=0`` or ``None`` to disable trimming.

    ``n_sp`` is the exact spike count attributed to each frame
    (:func:`ephys.spikes_per_sample`). Summing it over a set of frames gives the
    true count there, which is what lets the windowed analyses below compute a
    Poisson noise floor exactly instead of inferring one from a smoothed trace.

    Parameters
    ----------
    rate_specs : dict, optional
        Extra rate columns as ``{column: (kind, scale)}`` with ``kind`` in
        ``{'gauss', 'box'}`` and ``scale`` the Gaussian sd or the boxcar width
        in seconds — see :func:`boxcar_rate_specs`. The default ``rate`` column
        (Gaussian, ``sigma_s``) is always present. Edge trimming grows to
        whatever the widest estimator needs, so a 2 s boxcar costs 1 s off each
        end of the trial rather than silently reporting a rate that ramps up
        from zero.
    with_vm : bool
        Add a spike-blanked, smoothed ``vm`` column
        (:func:`ephys.subthreshold_vm`) plus ``vm_frac_blanked`` — the
        per-trial fraction of voltage samples replaced by interpolation, kept
        per frame so trials can be filtered on it. NaN columns if the channel
        or the detection is missing, so a partly-loaded cell still concatenates.
    vm_kwargs : dict, optional
        Forwarded to :func:`ephys.subthreshold_vm` (``channel``,
        ``blank_pre_s``, ``blank_post_s``, ``sigma_s``). Defaults to
        ``sigma_s=sigma_s`` so Vm and rate carry the same bandwidth — required
        for their correlation to mean anything.
    vm_control : bool
        Also compute ``vm_noblank`` — the same trace with no spike blanking at
        all. This is the control for the one artifact that can fabricate a
        Vm-vs-rate relation out of nothing: blanking replaces each spike with an
        interpolation across it, so a higher rate means more of the depolarized
        samples are removed and the estimated Vm is pushed *down* in proportion
        to the rate. That bias alone produces a negative Vm-rate correlation.
        Without blanking the bias runs the other way — spike peaks are left in,
        pushing Vm *up* with rate. A relation that holds with the same sign and
        similar magnitude in both is not an artifact of either.
    vm_specs : dict, optional
        Extra Vm columns at other kernel widths, ``{column: sigma_s}``. The
        voltage trace is cached on the Trial after the first read, so each extra
        sigma costs only the smoothing. Use this rather than re-smoothing an
        existing ``vm`` column when exactness matters; see
        :func:`add_smoothed_column` for the cheap approximate route.
    """
    ds = trial.downsample_probe
    t = np.asarray(trial.time)[ds].squeeze()
    # A NaN probeZero silently NaNs every position in the trial, and the failure
    # then surfaces far downstream -- as an all-NaN window inside a latency or a
    # correlation, which reads like a null result rather than missing data. It has
    # happened: 209 trials of 210602_F1_C1 carry /meta/probeZero = NaN, and the
    # symptom was a crash three analyses later. Say so here instead.
    try:
        pz = float(np.asarray(trial.probeZero).squeeze())
    except (TypeError, ValueError):
        pz = np.nan
    if not np.isfinite(pz):
        warnings.warn(
            f'{getattr(trial, "fn", "trial")}: probeZero is not finite '
            f'({pz!r}), so every probe position in this trial will be NaN. '
            'Check /meta/probeZero.', RuntimeWarning, stacklevel=2)
    x = -(np.asarray(trial.probe_position).squeeze()[ds] - pz)
    if t.size < 4:
        return None
    _, states, _, _ = kin.detect_movement_bouts(t, x, **bout_kwargs)

    # keep_silent keeps trials where detection ran and found nothing -- their
    # rate is a true zero, not missing data. Trials with no detection result at
    # all still return None and are still dropped. See
    # ephys.trial_spike_times for why this matters on a quiet cell.
    spike_s = ephys.trial_spike_times(trial, allow_empty=keep_silent)
    if spike_s is None:
        return None
    # Spikes from the adjacent trials, in this trial's clock. Recording is
    # continuous across trial boundaries, so these give the kernel full support at
    # the edges and the trim below can shrink to whatever is still unsupported.
    padded = spike_s
    rate_supported = False
    if pad_trials is not None:
        prev_trial, next_trial = pad_trials
        padded = ephys.padded_spike_times(trial, prev_trial, next_trial,
                                          allow_empty=keep_silent)
        if padded is None:
            padded = spike_s
        # Only a pad on *both* sides makes the rate fully supported; with one
        # neighbour the other end still ramps, and the trim has to stay.
        rate_supported = (prev_trial is not None) and (next_trial is not None)

    if mark_cue:
        # A boolean rather than a drop. The cue is the one moment in a trial with
        # a *known* imposed perturbation, so it is the natural reference for the
        # spontaneous fluctuations during a hold — and its sign is informative in
        # its own right (imposed force up -> rate down, opposite to the voluntary
        # relation). Highlight or split on it with ``rec['in_cue']``; never drop it
        # silently.
        c0, c1 = cue_window(trial)
        in_cue = (t >= c0 - cue_margin_s) & (t <= c1 + cue_margin_s)
        # Carry the cue's timing and commanded size per row, so anything
        # downstream can align to it without needing the Trial back. The
        # *commanded* displacement is the imposed variable — probe position during
        # the cue is moved by the piezo and the fly together, so it is not a
        # measure of the perturbation.
        cue_meta = {'cue_t0': c0, 'cue_t1': c1,
                    'cue_displacement': cue_displacement(trial)}
    cols = {'t': t, 'x': x, 'state': states,
            'rate': ephys.gaussian_rate(padded, t, sigma_s=sigma_s),
            # n_sp counts only *this* trial's spikes, by frame: it must stay an
            # exact within-trial count for the windowed analyses' noise floor.
            'n_sp': ephys.spikes_per_sample(spike_s, t)}
    # Raw acquisition channels sampled at frame times. ``sgsmonitor`` is the
    # reason this exists: during the cue the camera-tracked probe position is
    # moved by the piezo AND the fly together, so it is not a measure of the
    # imposed perturbation, and a lead/lag calibration built on it is invalid.
    # The monitor trace is the imposed signal. Averaged over each frame's bin
    # rather than point-sampled, so a 50 kHz channel is not aliased onto a
    # ~200 Hz frame grid.
    for name, chan in (raw_channels or {}).items():
        raw = getattr(trial, chan, None)
        if raw is None:
            cols[name] = np.full(len(t), np.nan)
            continue
        raw = np.asarray(raw, dtype=float).ravel()
        fs = float(trial.params['sampratein'])
        t_raw = float(np.asarray(trial.time).ravel()[0]) + np.arange(raw.size) / fs
        mid = 0.5 * (t[:-1] + t[1:])
        edges = np.concatenate([[t[0] - (mid[0] - t[0])], mid,
                                [t[-1] + (t[-1] - mid[-1])]])
        idx = np.clip(np.searchsorted(edges, t_raw, side='right') - 1,
                      0, len(t) - 1)
        inside = (t_raw >= edges[0]) & (t_raw < edges[-1])
        ssum = np.bincount(idx[inside], weights=raw[inside], minlength=len(t))
        scnt = np.bincount(idx[inside], minlength=len(t)).astype(float)
        with np.errstate(invalid='ignore'):
            vals = np.where(scnt > 0, ssum / np.maximum(scnt, 1), np.nan)
        cols[name] = vals
    if mark_cue:
        cols['in_cue'] = in_cue
        for k, v in cue_meta.items():
            cols[k] = np.full(len(t), v, dtype=float)

    # Trim is driven by whichever estimator still lacks support. Padded rate
    # estimators need none; Vm always does, because it is smoothed from *this*
    # trial's voltage and the neighbours' voltage is not spliced in.
    edge_needed = 0.0 if rate_supported else _spec_edge_s('gauss', sigma_s)
    for name, (kind, scale) in (rate_specs or {}).items():
        if kind == 'gauss':
            cols[name] = ephys.gaussian_rate(padded, t, sigma_s=scale)
        elif kind == 'box':
            cols[name] = ephys.boxcar_rate(padded, t, window_s=scale)
        else:
            raise ValueError(f"rate_specs kind must be 'gauss' or 'box', got {kind!r}")
        if not rate_supported:
            edge_needed = max(edge_needed, _spec_edge_s(kind, scale))

    if with_vm:
        vm_kw = {'sigma_s': sigma_s, **(vm_kwargs or {})}
        # The neighbours' *voltage* splices as cleanly as their spikes (one
        # continuous acquisition, one sample between trials), so Vm gets the same
        # padding — otherwise it keeps its own edge trim and the records still lose
        # 3 sigma per end even when the rate no longer needs it.
        vm_pad = {'pad_trials': pad_trials} if pad_trials is not None else {}
        vm_result = ephys.subthreshold_vm(trial, t_axis=t, **vm_kw, **vm_pad)
        if vm_result is None:
            cols['vm'] = np.full(len(t), np.nan)
            cols['vm_frac_blanked'] = np.full(len(t), np.nan)
        else:
            _, vm, info = vm_result
            cols['vm'] = vm
            cols['vm_frac_blanked'] = np.full(len(t), info['frac_blanked'])
        if vm_control:
            ctl_kw = {**vm_kw, 'blank_pre_s': 0.0, 'blank_post_s': 0.0}
            ctl = ephys.subthreshold_vm(trial, t_axis=t, **ctl_kw, **vm_pad)
            cols['vm_noblank'] = (np.full(len(t), np.nan) if ctl is None
                                  else ctl[1])
        if not rate_supported:   # same condition: both neighbours present
            edge_needed = max(edge_needed, _spec_edge_s('gauss', vm_kw['sigma_s']))
        # Extra Vm columns at other kernel widths. The voltage trace is cached on
        # the Trial after the first read, so each additional sigma costs only the
        # smoothing, not another pass over the file.
        for name, sigma in (vm_specs or {}).items():
            extra = ephys.subthreshold_vm(trial, t_axis=t,
                                          **{**vm_kw, 'sigma_s': sigma}, **vm_pad)
            cols[name] = (np.full(len(t), np.nan) if extra is None else extra[1])
            if not rate_supported:
                edge_needed = max(edge_needed, _spec_edge_s('gauss', sigma))

    # ``trim_edges_s`` is a floor for the unpadded case (its default, 75 ms, is
    # 3 sigma of the default kernel). Once the rate is padded that floor no longer
    # means anything, so only what an estimator actually still needs is trimmed —
    # and with no Vm requested that is nothing, so the whole trial survives.
    trim = edge_needed if rate_supported else max(trim_edges_s or 0.0, edge_needed)
    if trim > 0:
        dt = float(np.median(np.diff(t)))
        n_trim = int(np.ceil(trim / dt))
        if 2 * n_trim >= len(t):
            return None
        sl = slice(n_trim, len(t) - n_trim)
        cols = {k: v[sl] for k, v in cols.items()}
    return pd.DataFrame(cols)


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


def collect_cell_records(T, sigma_s=SIGMA_S, trim_edges_s=0.075,
                         rate_specs=None, with_vm=False, vm_kwargs=None,
                         vm_control=False, vm_specs=None, pad_spikes=False,
                         mark_cue=True, raw_channels=None, keep_silent=False,
                         **bout_kwargs):
    """Concatenate per-frame records across every non-excluded trial in T.

    ``rate_specs`` / ``with_vm`` / ``vm_kwargs`` / ``vm_control`` / ``vm_specs``
    are forwarded to :func:`per_frame_records`.
    """
    frames = []
    for tn, trial in T.df['Trial'].items():
        if trial is None or getattr(trial, 'excluded', False):
            continue
        rec = per_frame_records(trial, sigma_s=sigma_s,
                                trim_edges_s=trim_edges_s,
                                rate_specs=rate_specs, with_vm=with_vm,
                                vm_kwargs=vm_kwargs, vm_control=vm_control,
                                vm_specs=vm_specs,
                                pad_trials=(neighbour_trials(T, tn)
                                            if pad_spikes else None),
                                mark_cue=mark_cue, raw_channels=raw_channels,
                                keep_silent=keep_silent, **bout_kwargs)
        if rec is None:
            continue
        rec['trial'] = tn
        frames.append(rec)
    if not frames:
        return pd.DataFrame(columns=['t', 'x', 'state', 'rate', 'n_sp', 'trial'])
    return pd.concat(frames, ignore_index=True)


def add_movement_timing(records, min_move_ptp=8.0, init_s=0.15,
                        out_to='t_to_move', out_since='t_since_move',
                        out_dx='next_dx_init', inplace=False):
    """Per frame, the time until the next movement and since the previous one.

    The problem this addresses: the firing rate often dips in the few hundred ms
    *before* the probe moves, while the probe is still classified REST. Those
    frames therefore contribute a rate change at an unchanged position, which
    widens the position-vs-rate cloud for a reason that has nothing to do with
    position — motor preparation, not force.

    They are identified by *timing*, not by the rate itself. Selecting frames on
    "the rate dropped" would be circular: rate is the variable being measured, so
    thresholding it would guarantee finding an effect. Distance to the next
    movement onset is independent of rate, which makes "does excluding them
    tighten the cloud?" an honest question.

    ``t_to_move`` is ``+inf`` where no later movement exists in that trial, and
    ``t_since_move`` likewise for none earlier, so a plain ``< lead_s`` test picks
    out only genuine pre-movement frames.

    ``next_dx_init`` is the displacement over the first ``init_s`` of that upcoming
    movement, so pre-movement frames can be split by the *direction* the probe is
    about to go (positive = toward target / more force). That split matters: a rate
    *decrease* precedes most relaxations while a rate *increase* precedes about
    half of the force-increasing movements, so pooling the two reports the net of
    two opposing populations.

    A "movement" is a maximal non-REST stretch with a peak-to-peak excursion of at
    least ``min_move_ptp`` um — not a MOVE run. The classifier normally leaves a
    DRIFT between the rest and the movement proper (a slow approach, or a MOVE
    demoted for a small excursion), so keying on MOVE runs alone both mis-times the
    onset and discards most transitions.
    """
    out = records if inplace else records.copy()
    to_move = np.full(len(out), np.inf)
    since_move = np.full(len(out), np.inf)
    next_dx = np.full(len(out), np.nan)
    positions = np.arange(len(out))

    for _, g in out.groupby('trial', sort=False):
        idx = positions[out.index.get_indexer(g.index)]
        t = g['t'].to_numpy(dtype=float)
        x = g['x'].to_numpy(dtype=float)
        st = g['state'].to_numpy()
        if len(t) < 2:
            continue
        dt = float(np.median(np.diff(t)))
        n_init = max(int(round(init_s / dt)), 1)
        # maximal non-REST stretches that move far enough to count
        stretches = [(a, b) for a, b, v in _rle_states(st != kin.STATE_REST)
                     if v and (not min_move_ptp or np.ptp(x[a:b]) >= min_move_ptp)]
        if not stretches:
            continue
        onsets = np.array([t[a] for a, _ in stretches])
        offsets = np.array([t[b - 1] for _, b in stretches])
        dx_init = np.array([x[min(a + n_init, b - 1)] - x[a] for a, b in stretches])

        j = np.searchsorted(onsets, t, side='right')
        has_next = j < len(onsets)
        vals = np.full(len(t), np.inf)
        vals[has_next] = onsets[j[has_next]] - t[has_next]
        to_move[idx] = vals
        dxv = np.full(len(t), np.nan)
        dxv[has_next] = dx_init[j[has_next]]
        next_dx[idx] = dxv

        k = np.searchsorted(offsets, t, side='left') - 1
        has_prev = k >= 0
        vals = np.full(len(t), np.inf)
        vals[has_prev] = t[has_prev] - offsets[k[has_prev]]
        since_move[idx] = vals

    out[out_to] = to_move
    out[out_since] = since_move
    out[out_dx] = next_dx
    return out


def premovement_transitions(records, lead_s=0.2, base_to_s=0.5, base_from_s=1.5,
                            min_rest_s=1.0, min_move_ptp=8.0, init_s=0.15,
                            cols=('rate', 'vm')):
    """One row per REST bout that ends in a movement: what the rate did first.

    The pre-movement change is measured per transition, not as a population
    average, because the average can only report the net of two opposing
    populations. If the muscle lags the firing rate, then a movement that
    *increases* force should be preceded by a rate **increase** and a relaxation by
    a decrease — so pooling both directions cancels part of the effect and could
    report a mean of either sign, or none.

    For each qualifying REST run, the change is the mean over the last ``lead_s``
    before the movement minus the mean over ``base_to_s``..``base_from_s`` before
    it (a baseline inside the same bout, so a between-bout offset cannot leak in).

    ``dx_init`` is the movement's displacement over its first ``init_s``, and is
    the variable to split on: it is the direction the pre-movement rate change
    could plausibly have caused. ``net_dx`` (start to end) is also returned but is
    a poor test — a movement that goes out and comes back has ``net_dx`` near zero
    while its onset was firmly signed.

    Returns a DataFrame with ``trial``, ``rest_dur``, ``x_rest``, ``net_dx``,
    ``x_ptp_move``, ``move_dur`` and, per column, ``d_<col>``, ``base_<col>``,
    ``lead_<col>``.
    """
    cols = [c for c in np.atleast_1d(cols).tolist() if c in records.columns]
    rows = []
    for tn, g in records.groupby('trial', sort=True):
        g = g.reset_index(drop=True)
        st = g['state'].to_numpy()
        t = g['t'].to_numpy(dtype=float)
        x = g['x'].to_numpy(dtype=float)
        if len(t) < 4:
            continue
        segs = _rle_states(st)
        for i, (s0, e0, sv) in enumerate(segs):
            if sv != kin.STATE_REST or i + 1 >= len(segs):
                continue
            # The movement is the whole non-REST stretch that follows, not just a
            # MOVE run. The Schmitt classifier leaves a DRIFT between the rest and
            # the movement proper (a slow approach, or a MOVE demoted for a small
            # excursion), so a REST run is almost never followed *immediately* by
            # MOVE — requiring that found 3 transitions in this cell instead of
            # several hundred.
            m0 = e0
            j = i + 1
            while j < len(segs) and segs[j][2] != kin.STATE_REST:
                j += 1
            m1 = segs[j - 1][1]
            if m1 <= m0:
                continue
            if min_move_ptp and np.ptp(x[m0:m1]) < min_move_ptp:
                continue          # a wobble, not a movement
            t_rest = t[s0:e0]
            if t_rest[-1] - t_rest[0] < min_rest_s:
                continue
            t_end = t_rest[-1]
            lead_m = (t_rest > t_end - lead_s)
            base_m = ((t_rest <= t_end - base_to_s)
                      & (t_rest > t_end - base_from_s))
            if lead_m.sum() < 2 or base_m.sum() < 2:
                continue
            row = {'trial': tn, 'rest_dur': float(t_rest[-1] - t_rest[0]),
                   'x_rest': float(np.mean(x[s0:e0][base_m])),
                   'net_dx': float(x[m1 - 1] - x[m0]),
                   # The *initial* direction is the one the pre-movement rate
                   # change could plausibly cause: a movement that goes out and
                   # returns has net_dx ~ 0 while its onset was firmly signed.
                   'dx_init': float(
                       x[min(m0 + max(int(round(init_s / max(np.median(np.diff(t)),
                                                             1e-9))), 1),
                             m1 - 1)] - x[m0]),
                   'x_ptp_move': float(np.ptp(x[m0:m1])),
                   'move_dur': float(t[m1 - 1] - t[m0])}
            for c in cols:
                v = g[c].to_numpy(dtype=float)[s0:e0]
                b = float(np.nanmean(v[base_m]))
                l = float(np.nanmean(v[lead_m]))
                row[f'base_{c}'] = b
                row[f'lead_{c}'] = l
                row[f'd_{c}'] = l - b
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _rle_states(states):
    """``[(start, stop, value), ...]`` run-length encoding of a state array."""
    states = np.asarray(states)
    if states.size == 0:
        return []
    change = np.where(np.diff(states) != 0)[0] + 1
    starts = np.concatenate([[0], change])
    stops = np.concatenate([change, [states.size]])
    return [(int(a), int(b), states[a]) for a, b in zip(starts, stops)]


def add_smoothed_rate(records, sigma_s, out_col=None, count_col='n_sp',
                      inplace=False):
    """Add a Gaussian-smoothed rate column computed from the per-frame counts.

    Re-smoothing at a new kernel width normally means another pass over every
    trial file. It doesn't have to: ``n_sp`` holds the *exact* spike count per
    frame, so convolving it with a Gaussian on the frame grid gives the smoothed
    rate directly — which means a records frame already in hand can be
    re-smoothed at any width, instantly.

    The only cost is that spike times are discretised to frame bins (~10-20 ms).
    That is negligible once ``sigma_s`` is a few frames or more; for a kernel as
    narrow as the frames themselves, prefer the exact column
    ``per_frame_records`` builds from the spike times.

    Smoothing runs per trial, so a kernel never reaches across a trial boundary,
    and it is edge-corrected — the kernel is renormalised by its own coverage
    (``convolve(counts) / convolve(ones)``) rather than treating the outside of
    the trace as silence. Without that, records already edge-trimmed by
    ``per_frame_records`` would show a spurious rate droop over the first and
    last few sigma of every trial.

    Parameters
    ----------
    records   : per-frame frame carrying ``t``, ``trial`` and ``count_col``.
    sigma_s   : Gaussian sd in seconds.
    out_col   : column name; defaults to ``rate_g{sigma_ms}ms``.
    inplace   : mutate ``records`` instead of returning a copy.

    Returns the frame with the new column.
    """
    if count_col not in records.columns:
        raise KeyError(f'{count_col!r} not in records — rebuild them with a '
                       f'current per_frame_records, which always adds it.')
    if out_col is None:
        out_col = f'rate_g{int(round(sigma_s * 1000))}ms'
    out = records if inplace else records.copy()
    values = np.full(len(out), np.nan)
    positions = np.arange(len(out))

    for _, g in out.groupby('trial', sort=False):
        idx = positions[out.index.get_indexer(g.index)]
        t = g['t'].to_numpy()
        counts = g[count_col].to_numpy(dtype=float)
        if len(t) < 2:
            continue
        dt = float(np.median(np.diff(t)))
        sigma_samples = sigma_s / dt
        if sigma_samples < 0.5:            # kernel narrower than a frame
            values[idx] = counts / dt
            continue
        kernel = ephys._gaussian_kernel(sigma_samples)
        num = np.convolve(counts, kernel, mode='same')
        den = np.convolve(np.ones_like(counts), kernel, mode='same') * dt
        values[idx] = num / den

    out[out_col] = values
    return out


def add_smoothed_column(records, col, sigma_s, from_sigma_s=None, out_col=None,
                        inplace=False):
    """Re-smooth an existing frame-level column to a wider Gaussian kernel.

    For a continuous signal like Vm there is no count column to re-smooth from,
    but Gaussians compose: smoothing a σ₁-smoothed trace by σ_extra yields
    σ_total = sqrt(σ₁² + σ_extra²). So passing ``from_sigma_s`` (the width the
    column already carries) and ``sigma_s`` (the width wanted) applies the
    difference, ``σ_extra = sqrt(sigma_s² - from_sigma_s²)``, and the result
    matches a direct computation at ``sigma_s``.

    This is the cheap route — it needs no access to the raw voltage, so a records
    frame already in hand can be re-smoothed instantly. For an exact answer, pass
    ``vm_specs`` to :func:`per_frame_records` instead and let it smooth the
    voltage itself. The approximation costs only the frame-rate discretisation
    and the edge treatment.

    Smoothing runs per trial so a kernel never crosses a trial boundary, and it
    is edge-corrected: the kernel is renormalised by its own coverage rather than
    treating the outside of the trace as zero, which for a signal with a nonzero
    mean would otherwise pull the first and last few sigma toward 0.

    ``from_sigma_s=None`` treats the column as unsmoothed and applies
    ``sigma_s`` directly.
    """
    if col not in records.columns:
        raise KeyError(f'{col!r} not in records')
    if from_sigma_s:
        if sigma_s <= from_sigma_s:
            raise ValueError(f'sigma_s ({sigma_s}) must exceed from_sigma_s '
                             f'({from_sigma_s}) — a Gaussian cannot be un-smoothed')
        sigma_extra = float(np.sqrt(sigma_s ** 2 - from_sigma_s ** 2))
    else:
        sigma_extra = float(sigma_s)
    if out_col is None:
        out_col = f'{col}_g{int(round(sigma_s * 1000))}ms'
    out = records if inplace else records.copy()
    values = np.full(len(out), np.nan)
    positions = np.arange(len(out))

    for _, g in out.groupby('trial', sort=False):
        idx = positions[out.index.get_indexer(g.index)]
        t = g['t'].to_numpy()
        y = g[col].to_numpy(dtype=float)
        if len(t) < 2:
            continue
        dt = float(np.median(np.diff(t)))
        sigma_samples = sigma_extra / dt
        if sigma_samples < 0.5:
            values[idx] = y
            continue
        finite = np.isfinite(y)
        if not finite.any():
            continue
        kernel = ephys._gaussian_kernel(sigma_samples)
        num = np.convolve(np.where(finite, y, 0.0), kernel, mode='same')
        den = np.convolve(finite.astype(float), kernel, mode='same')
        with np.errstate(invalid='ignore', divide='ignore'):
            values[idx] = np.where(den > 0, num / den, np.nan)

    out[out_col] = values
    return out


def add_velocity(records, sigma_s=None, x_col='x', out_col=None, inplace=False):
    """Signed probe velocity (um/s) at frame times, differentiated per trial.

    ``sigma_s`` Gaussian-smooths the position first, so the velocity carries the
    *same* bandwidth as a rate column smoothed at the same sigma. That matters
    for any velocity-vs-rate correlation: a raw frame-to-frame derivative is
    dominated by tracking noise, and correlating it against a 25 ms rate reports
    the wider of the two kernels rather than the cell. ``None`` differentiates
    the unsmoothed column.

    Differentiation runs per trial, so no derivative crosses a trial boundary
    (where the clock resets and the gap is one sample, not one frame). NaN
    positions stay NaN in the output — they are interpolated across only so a
    single missing frame does not blank its two neighbours as well.
    """
    out = records if inplace else records.copy()
    if out_col is None:
        out_col = ('v' if not sigma_s
                   else f'v_g{int(round(sigma_s * 1000))}ms')
    src, tmp = x_col, None
    if sigma_s:
        tmp = f'_{x_col}_for_{out_col}'
        out = add_smoothed_column(out, x_col, sigma_s, out_col=tmp,
                                  inplace=True)
        src = tmp
    values = np.full(len(out), np.nan)
    positions = np.arange(len(out))
    for _, g in out.groupby('trial', sort=False):
        idx = positions[out.index.get_indexer(g.index)]
        t = g['t'].to_numpy(dtype=float)
        y = g[src].to_numpy(dtype=float)
        if len(t) < 2:
            continue
        fin = np.isfinite(y) & np.isfinite(t)
        if fin.sum() < 2:
            continue
        v = np.gradient(np.interp(t, t[fin], y[fin]), t)
        v[~fin] = np.nan
        values[idx] = v
    out[out_col] = values
    if tmp is not None:
        del out[tmp]      # in place, so inplace=True still writes to the caller
    return out


def bout_signal_correlation(records, pairs, state=kin.STATE_REST,
                            min_duration_s=1.0, max_x_ptp=None):
    """Within-bout correlation between two frame-level signals, bout by bout.

    Answers "do these two fluctuate *together* while the probe sits still?" —
    which pooling every frame of a session cannot, because between-bout and
    between-trial offsets dominate a pooled correlation (the same trap that
    inverted the Vm-rate correlation for a drifting cell).

    ``pairs`` is a list of ``(col_a, col_b)`` or ``(col_a, col_b, label)``. Both
    columns of a pair should carry the *same* kernel width, or the correlation is
    limited by the wider one and means something different for every pairing.

    Returns one row per bout per pair: ``trial``, ``epoch``, ``duration``,
    ``n_frames``, ``label``, ``rho``, ``slope`` (b on a), ``sd_a``, ``sd_b``,
    ``x_ptp``.
    """
    norm, missing = [], []
    for p in pairs:
        a, b = p[0], p[1]
        label = p[2] if len(p) > 2 else f'{a} vs {b}'
        absent = [c for c in (a, b) if c not in records.columns]
        if absent:
            missing.append((label, absent))
        else:
            norm.append((a, b, label))
    if missing:
        # Loudly, not silently: a dropped pair leaves a result that looks complete
        # but is missing a whole condition, and the omission only shows up much
        # later as a column that has quietly gone from a summary table.
        raise KeyError(
            'bout_signal_correlation: no such column(s) for '
            + '; '.join(f'{lab!r} needs {cols}' for lab, cols in missing)
            + f'. Available: {sorted(records.columns)}. Add the smoothed columns '
              'first, e.g. add_smoothed_rate(rec, 0.25, out_col="rate_g250ms") '
              'and add_smoothed_column(rec, "vm", 0.25, from_sigma_s=0.025, '
              'out_col="vm_g250ms").')
    rows = []
    for tn, g in records.groupby('trial', sort=True):
        g = g.reset_index(drop=True)
        runs = ([g] if state is None
                else list(_mask_runs(g, g['state'].to_numpy() == state)))
        for i_epoch, run in enumerate(runs):
            t = run['t'].to_numpy()
            if len(t) < 5:
                continue
            dt = float(np.median(np.diff(t)))
            duration = len(t) * dt
            if duration < min_duration_s:
                continue
            x_ptp = float(np.ptp(run['x'].to_numpy()))
            if max_x_ptp is not None and x_ptp > max_x_ptp:
                continue
            for a, b, label in norm:
                va = run[a].to_numpy(dtype=float)
                vb = run[b].to_numpy(dtype=float)
                m = np.isfinite(va) & np.isfinite(vb)
                if m.sum() < 5:
                    continue
                va, vb = va[m], vb[m]
                sa, sb = float(np.std(va, ddof=1)), float(np.std(vb, ddof=1))
                if sa == 0 or sb == 0:
                    continue
                rows.append({
                    'trial': tn, 'epoch': i_epoch, 'duration': duration,
                    'n_frames': int(m.sum()), 'label': label,
                    'rho': float(np.corrcoef(va, vb)[0, 1]),
                    'slope': float(np.polyfit(va, vb, 1)[0]),
                    'sd_a': sa, 'sd_b': sb, 'x_ptp': x_ptp,
                })
    cols = ['trial', 'epoch', 'duration', 'n_frames', 'label', 'rho', 'slope',
            'sd_a', 'sd_b', 'x_ptp']
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows)[cols]


def bout_rate_spread(records, rate_cols, state=kin.STATE_REST,
                     min_duration_s=1.0, count_col='n_sp'):
    """Per-bout summary of how much a rate estimate wanders *within* one bout.

    During REST the probe is nearly still by construction, so on a
    position-vs-rate plot each bout is close to a horizontal line and the rate
    estimate's fluctuation is the line's *length*. This measures that length for
    each column in ``rate_cols``, which is what a longer kernel is supposed to
    shorten.

    Returns one row per bout per rate column: ``trial``, ``epoch``,
    ``duration``, ``n_frames``, ``rate_col``, ``rate_mean``, ``rate_sd``,
    ``rate_ptp``, ``rate_iqr``, plus ``x_mean`` / ``x_ptp`` so a bout whose
    position was *not* steady can be filtered out.

    ``rate_sd`` here is within-bout scatter at frame resolution — a different
    quantity from ``conditional_spread``'s ``sd_within_bin``, which compares
    *different* bouts at the same position. This one says "how long is the
    horizontal streak"; that one says "how far apart are the streaks".
    """
    rate_cols = [c for c in np.atleast_1d(rate_cols).tolist()
                 if c in records.columns]
    rows = []
    for tn, g in records.groupby('trial', sort=True):
        g = g.reset_index(drop=True)
        runs = ([g] if state is None
                else list(_mask_runs(g, g['state'].to_numpy() == state)))
        for i_epoch, run in enumerate(runs):
            t = run['t'].to_numpy()
            if len(t) < 3:
                continue
            dt = float(np.median(np.diff(t)))
            duration = len(t) * dt
            if duration < min_duration_s:
                continue
            x = run['x'].to_numpy()
            base = {'trial': tn, 'epoch': i_epoch, 'duration': duration,
                    'n_frames': len(t), 'x_mean': float(np.mean(x)),
                    'x_ptp': float(np.ptp(x)),
                    'n_spikes': float(run[count_col].sum())
                    if count_col in run.columns else np.nan}
            for c in rate_cols:
                v = run[c].to_numpy(dtype=float)
                v = v[np.isfinite(v)]
                if len(v) < 3:
                    continue
                rows.append({**base, 'rate_col': c,
                             'rate_mean': float(np.mean(v)),
                             'rate_sd': float(np.std(v, ddof=1)),
                             'rate_ptp': float(np.ptp(v)),
                             'rate_iqr': float(np.subtract(*np.percentile(v, [75, 25])))})
    cols = ['trial', 'epoch', 'duration', 'n_frames', 'n_spikes', 'x_mean',
            'x_ptp', 'rate_col', 'rate_mean', 'rate_sd', 'rate_ptp', 'rate_iqr']
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows)[cols]


# ---------------------------------------------------------------------------
# Windowed samples: rate and position averaged over the *same* window
# ---------------------------------------------------------------------------
# The per-frame scatter of rate against position asks a slightly unfair
# question: rate is measured over ~100 ms while position is a single frame. To
# ask whether slower measurement tightens the relation, both sides have to be
# averaged over the same window — which is what these produce. Windows are cut
# inside contiguous runs of one state, so a window never straddles a state
# change, and each carries the exact spike count it contains so its own Poisson
# noise floor is computable rather than assumed.

def windowed_state_samples(records, window_s, state=kin.STATE_REST,
                           rate_col='rate', mean_cols=(), min_coverage=0.8,
                           max_x_ptp=None, count_col='n_sp', min_epoch_s=None):
    """Non-overlapping ``window_s`` windows tiled inside runs of ``state``.

    Each window yields one row: the exact spike count it contains (turned into
    a rate by its own covered duration), the mean of ``rate_col`` over the same
    frames, and the mean/spread of position over the same frames.

    Parameters
    ----------
    records      : per-frame frame from :func:`collect_cell_records`.
    window_s     : window duration (s). Windows are tiled from each run's start
                   and only kept if they fall entirely inside the run.
    state        : state to tile inside (``None`` for the whole trial, ignoring
                   state). Runs are contiguous in the records' own ordering.
    rate_col     : which smoothed-rate column to average per window.
    mean_cols    : extra numeric columns to average per window (e.g. ``('vm',)``).
    min_coverage : reject a window whose frames cover less than this fraction of
                   ``window_s`` — dropped frames would otherwise inflate the rate
                   (count over a full window, divided by a partial duration).
    max_x_ptp    : reject a window whose position range exceeds this (um). During
                   REST the probe is nearly still by construction, so this is a
                   guard against a mis-classified window rather than the main
                   filter; leaving it None keeps everything and lets ``x_ptp``
                   be filtered downstream.
    min_epoch_s  : skip runs shorter than this. Set it to the *longest* window in
                   a sweep (:func:`window_sweep` does this by default) so every
                   window length is measured on the same rest epochs. Without it
                   a long window silently keeps only the long epochs, and the
                   cloud narrows partly because the data changed rather than
                   because the measurement got quieter.

    Returns
    -------
    DataFrame with ``trial``, ``epoch`` (index of the state run within the
    trial), ``i_win``, ``window_s``, ``t_start``, ``t_mid``, ``duration``,
    ``n_frames``, ``n_spikes``, ``rate`` (count / duration), ``rate_smoothed``
    (mean of ``rate_col``), ``x_mean``, ``x_sd``, ``x_ptp``, and one column per
    entry of ``mean_cols``.

    ``rate`` and ``rate_smoothed`` differ only by the smoothing kernel's leakage
    across the window edges; ``rate`` is the one with an exact noise floor, so
    prefer it for the spread analyses and keep the other as a cross-check.
    """
    mean_cols = tuple(mean_cols)
    rows = []
    for tn, g in records.groupby('trial', sort=True):
        g = g.reset_index(drop=True)
        if state is None:
            runs = [g]
        else:
            runs = list(_mask_runs(g, g['state'].to_numpy() == state))
        for i_epoch, run in enumerate(runs):
            t = run['t'].to_numpy()
            if t.size < 2:
                continue
            dt = float(np.median(np.diff(t)))
            if min_epoch_s is not None and (t[-1] - t[0] + dt) < min_epoch_s:
                continue
            n_win = int(np.floor((t[-1] - t[0] + dt) / window_s))
            if n_win < 1:
                continue
            for k in range(n_win):
                lo = t[0] - 0.5 * dt + k * window_s
                hi = lo + window_s
                m = (t >= lo) & (t < hi)
                n_frames = int(m.sum())
                if n_frames == 0:
                    continue
                duration = n_frames * dt
                if duration < min_coverage * window_s:
                    continue
                xs = run['x'].to_numpy()[m]
                x_ptp = float(np.ptp(xs))
                if max_x_ptp is not None and x_ptp > max_x_ptp:
                    continue
                row = {
                    'trial': tn, 'epoch': i_epoch, 'i_win': k,
                    'window_s': float(window_s),
                    't_start': float(t[m][0]),
                    't_mid': float(np.mean(t[m])),
                    'duration': duration,
                    'n_frames': n_frames,
                    'n_spikes': float(run[count_col].to_numpy()[m].sum()),
                    'rate_smoothed': float(np.mean(run[rate_col].to_numpy()[m])),
                    'x_mean': float(np.mean(xs)),
                    'x_sd': float(np.std(xs, ddof=1)) if n_frames > 1 else 0.0,
                    'x_ptp': x_ptp,
                }
                row['rate'] = row['n_spikes'] / duration
                for c in mean_cols:
                    if c in run.columns:
                        row[c] = float(np.nanmean(run[c].to_numpy()[m]))
                rows.append(row)
    cols = ['trial', 'epoch', 'i_win', 'window_s', 't_start', 't_mid',
            'duration', 'n_frames', 'n_spikes', 'rate', 'rate_smoothed',
            'x_mean', 'x_sd', 'x_ptp', *mean_cols]
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows)[cols]


def window_sweep(records, windows_s=RATE_WINDOWS_S, common_epochs=True, **kwargs):
    """:func:`windowed_state_samples` at each window length, concatenated.

    Tell the windows apart by the ``window_s`` column. Windows of different
    lengths overlap in time by construction — they are alternative
    measurements of the same data, not independent samples, so compare
    summaries across them rather than pooling them.

    ``common_epochs`` (default) restricts every window length to the state runs
    long enough to hold the *longest* window, so the sweep compares
    measurements of the same data. Without it the long windows quietly analyse
    only the long rest epochs — a different, longer-resting subset of trials —
    and any narrowing conflates the measurement with the sample. Pass False to
    use each window's full complement of epochs (more data per window, not
    comparable across them).
    """
    kwargs = dict(kwargs)
    if common_epochs:
        kwargs.setdefault('min_epoch_s', max(windows_s))
    parts = [windowed_state_samples(records, w, **kwargs) for w in windows_s]
    parts = [p for p in parts if len(p)]
    if not parts:
        return windowed_state_samples(records, windows_s[0], **kwargs)
    return pd.concat(parts, ignore_index=True)


def allan_sd(windows, value_col='rate', levels=('trial', 'epoch'),
             order_col='i_win'):
    """Window-to-window sd of ``value_col`` between *adjacent* windows.

    ``sqrt(mean((r[i+1] - r[i])**2) / 2)`` over pairs of neighbouring windows in
    the same epoch — the Allan deviation at that window length. Differencing
    neighbours cancels any slow drift, so this measures only the variability at
    the window's own timescale, whatever the spike train's statistics happen to
    be.

    That makes it the noise floor to compare against, in preference to a Poisson
    prediction. A regularly firing motor neuron is far *less* variable than
    Poisson, so a Poisson floor sits above the observed scatter and reports an
    impossible "zero excess" — see :func:`conditional_spread`.
    """
    num = den = 0.0
    keys = [c for c in levels if c in windows.columns]
    groups = windows.groupby(keys, sort=False, observed=True) if keys else [((), windows)]
    for _, g in groups:
        if order_col in g.columns:
            g = g.sort_values(order_col)
            adj = np.diff(g[order_col].to_numpy()) == 1
        else:
            adj = np.ones(max(len(g) - 1, 0), dtype=bool)
        v = g[value_col].to_numpy(dtype=float)
        if len(v) < 2:
            continue
        d = np.diff(v)[adj]
        d = d[np.isfinite(d)]
        num += float(np.sum(d ** 2))
        den += 2.0 * len(d)
    return (float(np.sqrt(num / den)) if den else np.nan), int(den / 2)


def conditional_spread(windows, value_col='rate', x_col='x_mean',
                       x_bin_width=10.0, min_per_bin=5, by='window_s',
                       detrend=True, floor='allan'):
    """How much rate varies *at a given position*, and how that compares to
    counting noise.

    For each group of ``by`` (window length, normally), position is binned at
    ``x_bin_width``, and the within-bin sd of ``value_col`` is pooled across
    bins. That pooled sd is the width of the cloud with the position dependence
    taken out — the quantity that should shrink if the scatter is fast rate
    fluctuation being averaged away.

    What makes it interpretable is the floor it is compared against. Two are
    reported:

    ``sd_allan``    the measured window-to-window variability inside epochs
                    (:func:`allan_sd`) — variability at the window's own
                    timescale, with slow drift differenced away. Assumption-free,
                    and the default ``floor``.
    ``sd_poisson``  ``sqrt(rate / duration)``, what a Poisson train of the same
                    mean rate would give. A *theoretical reference*, not a floor:
                    these motor neurons fire regularly, so their counts are much
                    less variable than Poisson and this sits well above the
                    observed scatter. Taken as the floor it yields a nonsensical
                    zero excess for a cell whose scatter is entirely real.
                    ``fano_emp`` = ``sd_allan**2 * duration / rate`` quantifies
                    that directly: 1 is Poisson, and these cells run ~0.1-0.3.

    So, against ``sd_allan``:

        ``sd_within ~ sd_allan``   the scatter at fixed position is all
                                   window-timescale fluctuation; averaging
                                   longer will keep shrinking it.
        ``sd_within >> sd_allan``  scatter beyond what fluctuates at this
                                   timescale — a slower offset between epochs or
                                   trials, which no window length removes.
                                   ``sd_excess`` estimates its size.

    Also returns the position dependence itself — ``rho`` (Pearson), ``r2`` and
    ``slope`` of ``value_col`` on ``x_col`` — which should *rise* with window
    length if a real relation is being uncovered by averaging noise away.

    ``detrend`` removes each bin's residual linear dependence on position before
    taking its sd. Without it a bin of finite width carries some of the very
    position dependence being conditioned out (a 0.4 Hz/um slope across a 10 um
    bin contributes ~1.2 Hz of sd on its own), which shows up as a small fake
    ``sd_excess``. Bins with too few points to fit are then skipped.

    Returns one row per group.
    """
    by = [by] if isinstance(by, str) else list(by)
    groups = windows.groupby(by, sort=True, observed=True) if by else [((), windows)]
    rows = []
    for key, g in groups:
        key = key if isinstance(key, tuple) else (key,)
        g = g.dropna(subset=[value_col, x_col])
        if len(g) < 2:
            continue
        v = g[value_col].to_numpy(dtype=float)
        x = g[x_col].to_numpy(dtype=float)
        dur = g['duration'].to_numpy(dtype=float)

        bin_idx = np.floor(x / x_bin_width).astype(int)
        min_n = max(min_per_bin, 3 if detrend else 2)
        num_w = num_p = den = 0.0
        n_bins = 0
        for b in np.unique(bin_idx):
            m = bin_idx == b
            n_b = int(m.sum())
            if n_b < min_n:
                continue
            vb, xb = v[m], x[m]
            if detrend and np.std(xb) > 0:
                resid = vb - np.polyval(np.polyfit(xb, vb, 1), xb)
                var_b, w = float(np.sum(resid ** 2) / (n_b - 2)), n_b - 2
            else:
                var_b, w = float(np.var(vb, ddof=1)), n_b - 1
            n_bins += 1
            num_w += w * var_b
            num_p += w * float(np.mean(vb) / np.mean(dur[m]))
            den += w
        sd_within = float(np.sqrt(num_w / den)) if den else np.nan
        sd_poisson = float(np.sqrt(num_p / den)) if den else np.nan
        sd_allan, n_pairs = allan_sd(g, value_col=value_col)
        mean_dur = float(np.mean(dur))
        mean_rate = float(np.mean(v))
        fano_emp = (sd_allan ** 2 * mean_dur / mean_rate
                    if mean_rate > 0 and sd_allan == sd_allan else np.nan)

        floor_sd = sd_allan if floor == 'allan' else sd_poisson
        if not (floor_sd == floor_sd):          # NaN floor (e.g. no adjacent pairs)
            floor_sd = sd_poisson
        sd_excess = (float(np.sqrt(max(sd_within ** 2 - floor_sd ** 2, 0.0)))
                     if den else np.nan)
        # A one-sided 95% lower bound on the within-bin variance, from the chi2
        # sampling distribution of a pooled variance with ``den`` df. Long
        # windows leave few windows per bin, and sqrt() of a difference of two
        # noisy variances is upward-biased there — a train with no slow structure
        # at all can show sd_excess of a couple of Hz on noise alone.
        # sd_excess_lo95 is the honest version: > 0 means the scatter really
        # exceeds the floor.
        if den >= 1 and floor_sd:
            var_lo = (num_w / den) * den / _chi2.ppf(0.95, den)
            sd_excess_lo95 = float(np.sqrt(max(var_lo - floor_sd ** 2, 0.0)))
            ratio_lo95 = float(np.sqrt(var_lo) / floor_sd)
        else:
            sd_excess_lo95 = ratio_lo95 = np.nan

        rho = float(np.corrcoef(x, v)[0, 1]) if np.std(x) > 0 and np.std(v) > 0 else np.nan
        slope = float(np.polyfit(x, v, 1)[0]) if np.std(x) > 0 else np.nan
        rows.append({
            **dict(zip(by, key)),
            'n_windows': len(g),
            'n_bins': n_bins,
            'n_used': int(den + n_bins) if den else 0,
            'rate_mean': float(np.mean(v)),
            'duration_mean': float(np.mean(dur)),
            'sd_total': float(np.std(v, ddof=1)),
            'sd_within_bin': sd_within,
            'sd_allan': sd_allan,
            'sd_poisson': sd_poisson,
            'fano_emp': fano_emp,
            'floor': floor,
            'sd_excess': sd_excess,
            'sd_excess_lo95': sd_excess_lo95,
            'ratio_lo95': ratio_lo95,
            'df': int(den),
            'n_pairs': n_pairs,
            'poisson_frac_var': (sd_poisson ** 2 / sd_within ** 2
                                 if den and sd_within else np.nan),
            'rho': rho,
            'r2': rho ** 2 if rho == rho else np.nan,
            'slope': slope,
        })
    return pd.DataFrame(rows)


def variance_decomposition(windows, value_col='rate', by='window_s',
                           levels=('trial', 'epoch')):
    """Nested variance decomposition of ``value_col`` — where the scatter lives.

    Splits the total sum of squares into a between-trial part, a
    between-``epoch``-within-trial part, and a within-epoch part, and reports
    each as a fraction of the total. This answers the same question as
    :func:`conditional_spread` without having to pick a window at all:

        within-epoch dominates   fast fluctuation; longer averaging will help.
        between-epoch dominates  each rest epoch sits at its own rate; averaging
                                 within an epoch will not touch it.
        between-trial dominates  a per-trial offset (state, drift, adaptation);
                                 no window length helps, and pooling trials is
                                 what widened the cloud in the first place.

    ``poisson_frac_var`` is the share of the *total* variance expected from
    spike counting alone, so the within-epoch fraction can be read against it.

    Returns one row per group of ``by``.
    """
    by = [by] if isinstance(by, str) else list(by)
    l_outer, l_inner = levels
    groups = windows.groupby(by, sort=True, observed=True) if by else [((), windows)]
    rows = []
    for key, g in groups:
        key = key if isinstance(key, tuple) else (key,)
        g = g.dropna(subset=[value_col])
        if len(g) < 3:
            continue
        v = g[value_col].to_numpy(dtype=float)
        grand = float(np.mean(v))
        ss_total = float(np.sum((v - grand) ** 2))
        if ss_total <= 0:
            continue
        ss_trial = ss_epoch = ss_within = 0.0
        for _, gt in g.groupby(l_outer, sort=False, observed=True):
            vt = gt[value_col].to_numpy(dtype=float)
            mt = float(np.mean(vt))
            ss_trial += len(vt) * (mt - grand) ** 2
            for _, ge in gt.groupby(l_inner, sort=False, observed=True):
                ve = ge[value_col].to_numpy(dtype=float)
                me = float(np.mean(ve))
                ss_epoch += len(ve) * (me - mt) ** 2
                ss_within += float(np.sum((ve - me) ** 2))
        poisson_var = float(np.mean(v / g['duration'].to_numpy(dtype=float)))
        rows.append({
            **dict(zip(by, key)),
            'n_windows': len(g),
            'n_trials': int(g[l_outer].nunique()),
            'n_epochs': int(g.groupby([l_outer, l_inner], observed=True).ngroups),
            'var_total': ss_total / (len(v) - 1),
            'frac_between_trial': ss_trial / ss_total,
            'frac_between_epoch': ss_epoch / ss_total,
            'frac_within_epoch': ss_within / ss_total,
            'poisson_frac_var': poisson_var / (ss_total / (len(v) - 1)),
        })
    return pd.DataFrame(rows)


def vm_rate_correlations(records, state=None, vm_col='vm', rate_col='rate',
                         min_frames=500):
    """Vm-vs-rate correlation split into its within-trial and between-trial parts.

    Pooling every frame of a session into one correlation is not safe for Vm.
    The electrode's access resistance and the cell's health drift over an hour of
    recording, so per-trial mean Vm can wander by tens of mV independently of
    anything the neuron is doing. If the firing rate also drifts — through
    adaptation, or because the behaviour changes as the fly learns — the pooled
    correlation measures the two drifts against each other and can come out
    *opposite in sign* to the relation inside every single trial.

    So three numbers are returned instead of one:

    ``rho_within_trial``   each trial centred on its own median first. The
                           interpretable one: does Vm covary with rate moment to
                           moment, within a stretch of recording where the
                           electrode has not moved.
    ``rho_between_trial``  per-trial medians against each other. Real biology and
                           drift are indistinguishable here.
    ``rho_pooled``         no centring. Reported only so it can be compared with
                           the other two; a large gap between pooled and
                           within-trial is the warning.

    ``rho_vm_vs_trial`` / ``rho_rate_vs_trial`` and ``vm_range`` say directly
    whether drift is present: a Vm that marches with trial number across a
    20 mV range is an electrode, not a neuron.
    """
    g = records.dropna(subset=[vm_col, rate_col])
    if state is not None:
        g = g[np.isin(g['state'], _state_tuple(state))]
    if len(g) < min_frames:
        return {}
    vm_c = g[vm_col] - g.groupby('trial')[vm_col].transform('median')
    r_c = g[rate_col] - g.groupby('trial')[rate_col].transform('median')
    pt = (g.groupby('trial')
           .agg(vm=(vm_col, 'median'), rate=(rate_col, 'median')).dropna())

    def _rho(a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
            return np.nan
        return float(np.corrcoef(a, b)[0, 1])

    tn = pt.index.to_numpy(dtype=float)
    return {
        'n_frames': len(g), 'n_trials': len(pt),
        'rate_mean': float(g[rate_col].mean()),
        'vm_range': float(pt['vm'].max() - pt['vm'].min()) if len(pt) else np.nan,
        'rho_pooled': _rho(g[vm_col], g[rate_col]),
        'rho_within_trial': _rho(vm_c, r_c),
        'rho_between_trial': _rho(pt['vm'], pt['rate']),
        'rho_vm_vs_trial': _rho(tn, pt['vm']),
        'rho_rate_vs_trial': _rho(tn, pt['rate']),
        'slope_within_trial': (float(np.polyfit(vm_c, r_c, 1)[0])
                               if np.std(vm_c) > 0 else np.nan),
    }


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
# In-target movement: how long does the fly keep moving once inside the target
# ---------------------------------------------------------------------------
# ``state`` throughout this section is a single kinematic state or an iterable of
# them. ``STATES_ACTIVE`` (drift + move) is 'not resting': a slide that never
# reaches v_th counts as engagement in its own right, and a move is extended back
# through the slow approach that led into it. Note the threshold this depends on:
# DRIFT is v_rest <= speed < v_th, so with the defaults (60/30) a 6 um/s creep is
# REST, not DRIFT, and no state set will pick it up until v_rest comes down.
# Whichever set is used, each run and each trial also carries its MOVE and DRIFT
# components separately, so the with-/without-drift comparison needs one pass.

STATES_ACTIVE = (kin.STATE_DRIFT, kin.STATE_MOVE)

_STATE_NAMES = {kin.STATE_REST: 'rest', kin.STATE_DRIFT: 'drift',
                kin.STATE_MOVE: 'move'}


def _state_tuple(state):
    """``state`` as a sorted tuple of ints, scalar or iterable alike."""
    return tuple(sorted(int(s) for s in np.atleast_1d(state)))


def state_set_label(state):
    """'move', 'drift', 'drift+move' — a column-safe name for a state set."""
    return '+'.join(_STATE_NAMES.get(s, str(s)) for s in _state_tuple(state))


def _run_composition(states_seg, dt):
    """MOVE/DRIFT breakdown of one run's samples.

    ``frac_move`` is by sample count, and ``run_kind`` is 'move' / 'drift' /
    'mixed' ('other' if the state set reaches outside MOVE and DRIFT).

    With the Schmitt classifier a mixed run is always drift-*then*-move: MOVE is
    left only on a run of >= Dt below v_rest, and that stretch then classifies as
    REST, which ends the run. So merging DRIFT never glues two moves together.
    What it does do is (a) admit slides that never reach v_th as runs in their
    own right, and (b) extend a move backwards through the slow approach that
    led into it. Filtering ``run_kind == 'move'`` therefore keeps only the moves
    that had no drift approach — it does not reproduce a ``state=MOVE`` pass.
    """
    n = len(states_seg)
    n_move = int(np.count_nonzero(states_seg == kin.STATE_MOVE))
    n_drift = int(np.count_nonzero(states_seg == kin.STATE_DRIFT))
    if n_move and n_drift:
        kind = 'mixed'
    elif n_move and n_move == n:
        kind = 'move'
    elif n_drift and n_drift == n:
        kind = 'drift'
    else:
        kind = 'other'
    return {
        'move_time':   n_move * dt,
        'drift_time':  n_drift * dt,
        'n_move':      n_move,
        'n_drift':     n_drift,
        'frac_move':   (n_move / n) if n else np.nan,
        'run_kind':    kind,
    }


def target_bounds(trial):
    """(lo, hi) bounds of *this trial's* target zone in the flipped frame.

    Flipped frame is ``-(probe_position - probeZero)``, i.e. positive = toward
    target / more force, matching ``kinematics`` and ``per_frame_records``.
    In the raw frame the zone spans ``[pyasXPosition, pyasXPosition+pyasWidth]``
    (see ``Trial.on_target``), so flipping puts ``hi = -(pyasXPosition -
    probeZero)`` at the top and ``lo = hi - pyasWidth`` at the bottom.

    Returns (nan, nan) if the trial carries no usable target metadata.
    """
    try:
        hi = -(float(trial.pyasXPosition) - float(trial.probeZero))
        width = float(trial.pyasWidth)
    except (KeyError, TypeError, ValueError):
        return np.nan, np.nan
    if not np.isfinite(hi) or not np.isfinite(width):
        return np.nan, np.nan
    return hi - width, hi


def _true_runs(mask):
    """Yield (start, stop) index pairs (stop exclusive) of contiguous True."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return
    diff = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    for s, e in zip(np.where(diff == 1)[0], np.where(diff == -1)[0]):
        yield int(s), int(e)


def trial_in_target_moves(trial, t_min=0.0, t_max=None, state=kin.STATE_MOVE,
                          **bout_kwargs):
    """Contiguous runs of ``state`` that occur while the probe is in the target.

    The state classification runs on the whole downsampled trace (so runs are
    classified with full context); the ``t_min`` / ``t_max`` window and the
    in-target test are applied afterwards as sample masks. A run is therefore
    a maximal stretch of samples that are simultaneously MOVE **and** inside
    the target band **and** inside the time window.

    Parameters
    ----------
    trial       : Trial object
    t_min,t_max : time window (s, trial time). Default ``t_min=0.0`` keeps the
                  post-stim-onset window, matching the ``*_time_fraction``
                  scalars. Pass ``t_min=None`` for the whole trace.
    state       : state, or iterable of states, to measure. Defaults to MOVE;
                  pass ``STATES_ACTIVE`` to merge drift into the runs.
    bout_kwargs : forwarded to ``kin.detect_movement_bouts`` (v_th, v_rest, …).

    Returns
    -------
    (runs, summary)

    runs : list of dicts, one per run —
        ``duration``      run length in s (n_samples * dt)
        ``n_samples``     samples in the run
        ``t_start``/``t_end``   trial time at the first/last sample
        ``x_start``/``x_end``   flipped position at the first/last sample
        ``x_ptp``         peak-to-peak excursion within the run (um)
        ``net_dx``        x_end - x_start (positive = toward more force)
        ``mean_abs_v``    mean |smoothed velocity| over the run (um/s)
        ``move_time``/``drift_time``   the run's MOVE / DRIFT components (s);
                          they sum to ``duration`` unless the state set reaches
                          beyond MOVE and DRIFT
        ``frac_move``     fraction of the run's samples in MOVE
        ``run_kind``      'move' | 'drift' | 'mixed'
        ``ends_by``       'exit_target' | 'state_change' | 'window_edge' —
                          why the run stopped. ``exit_target`` means the fly
                          was still moving when it left the band (moved
                          *through* the target); ``state_change`` means the
                          movement itself stopped while still inside it.
        ``starts_by``     same idea for the run's onset ('enter_target',
                          'state_change', 'window_edge')

    summary : dict of per-trial totals —
        ``window_time``, ``in_target_time``, ``move_time``,
        ``in_target_move_time``, ``n_runs``, ``dt``, ``target_lo``,
        ``target_hi``. All times in s. ``move_time`` and ``in_target_move_time``
        follow whatever ``state`` was asked for, so
        ``in_target_move_time / in_target_time`` is the fraction of in-target
        time spent in that state set. ``in_target_move_only_time`` and
        ``in_target_drift_only_time`` split it by state regardless of the set
        requested, and ``state_set`` records which set was used.
    """
    empty = ([], {'window_time': 0.0, 'in_target_time': 0.0, 'move_time': 0.0,
                  'in_target_move_time': 0.0,
                  'in_target_move_only_time': 0.0,
                  'in_target_drift_only_time': 0.0,
                  'state_set': state_set_label(state),
                  'n_runs': 0, 'dt': np.nan,
                  'target_lo': np.nan, 'target_hi': np.nan})
    if trial is None or getattr(trial, 'excluded', False):
        return empty

    lo, hi = target_bounds(trial)
    if not np.isfinite(lo):
        return empty

    ds = trial.downsample_probe
    t = np.asarray(trial.time)[ds].squeeze()
    x = -(np.asarray(trial.probe_position).squeeze()[ds] - trial.probeZero)
    if t.size < 4:
        return empty

    _, states, _, _ = kin.detect_movement_bouts(t, x, **bout_kwargs)
    dt = float(np.median(np.diff(t)))
    v_sm = kin.smoothed_velocity(t, x)

    window = np.ones(len(t), dtype=bool)
    if t_min is not None:
        window &= (t >= t_min)
    if t_max is not None:
        window &= (t <= t_max)

    in_tgt = (x >= lo) & (x <= hi)
    is_state = np.isin(states, _state_tuple(state))
    sel = window & in_tgt & is_state

    runs = []
    n = len(t)
    for s, e in _true_runs(sel):
        # why did the run end? window edge < target exit < state change
        if e >= n or not window[e]:
            ends_by = 'window_edge'
        elif not in_tgt[e]:
            ends_by = 'exit_target'
        else:
            ends_by = 'state_change'
        if s == 0 or not window[s - 1]:
            starts_by = 'window_edge'
        elif not in_tgt[s - 1]:
            starts_by = 'enter_target'
        else:
            starts_by = 'state_change'
        seg_x = x[s:e]
        runs.append({
            'duration':    (e - s) * dt,
            'n_samples':   e - s,
            't_start':     float(t[s]),
            't_end':       float(t[e - 1]),
            'x_start':     float(seg_x[0]),
            'x_end':       float(seg_x[-1]),
            'x_ptp':       float(np.ptp(seg_x)),
            'net_dx':      float(seg_x[-1] - seg_x[0]),
            'mean_abs_v':  float(np.mean(np.abs(v_sm[s:e]))),
            **_run_composition(states[s:e], dt),
            'ends_by':     ends_by,
            'starts_by':   starts_by,
        })

    in_tgt_win = window & in_tgt
    summary = {
        'window_time':               float(window.sum()) * dt,
        'in_target_time':            float(in_tgt_win.sum()) * dt,
        'move_time':                 float((window & is_state).sum()) * dt,
        'in_target_move_time':       float(sel.sum()) * dt,
        'in_target_move_only_time':  float((in_tgt_win & (states == kin.STATE_MOVE)).sum()) * dt,
        'in_target_drift_only_time': float((in_tgt_win & (states == kin.STATE_DRIFT)).sum()) * dt,
        'state_set':                 state_set_label(state),
        'n_runs':                    len(runs),
        'dt':                        dt,
        'target_lo':                 float(lo),
        'target_hi':                 float(hi),
    }
    return runs, summary


_TRIAL_META_COLS = ('as_outcome', 'pyasState', 'is_rest', 'is_probe',
                    'trial_in_block', 'block')


def _trial_meta(T, tn):
    """Whatever of ``_TRIAL_META_COLS`` this Table happens to carry."""
    return {c: T.df.at[tn, c] for c in _TRIAL_META_COLS if c in T.df.columns}


def collect_in_target_moves(T, dfc=None, t_min=0.0, t_max=None,
                            state=kin.STATE_MOVE, **bout_kwargs):
    """Run ``trial_in_target_moves`` over every non-excluded trial in a Table.

    Returns ``(runs_df, trials_df)``: one row per in-target movement run, and
    one row per trial with the totals. Both carry ``dfc`` and ``trial`` plus
    any of ``as_outcome`` / ``pyasState`` / ``is_rest`` / ``is_probe`` the
    Table has, so downstream filtering (e.g. as_off trials only) is a
    ``.query``.
    """
    if dfc is None and getattr(T, 'day', None) is not None:
        dfc = f'{T.day}_F{T.fly}_C{T.cell}'
    run_rows, trial_rows = [], []
    for tn, trial in T.df['Trial'].items():
        if trial is None or getattr(trial, 'excluded', False):
            continue
        runs, summary = trial_in_target_moves(
            trial, t_min=t_min, t_max=t_max, state=state, **bout_kwargs)
        meta = {'dfc': dfc, 'trial': tn, 'state_set': state_set_label(state),
                **_trial_meta(T, tn)}
        trial_rows.append({**meta, **summary})
        for r in runs:
            run_rows.append({**meta, **r})

    run_cols = ['dfc', 'trial', 'state_set', 'duration', 'n_samples', 't_start', 't_end',
                'x_start', 'x_end', 'x_ptp', 'net_dx', 'mean_abs_v',
                'move_time', 'drift_time', 'n_move', 'n_drift', 'frac_move',
                'run_kind', 'ends_by', 'starts_by']
    trial_cols = ['dfc', 'trial', 'window_time', 'in_target_time', 'move_time',
                  'in_target_move_time', 'in_target_move_only_time',
                  'in_target_drift_only_time', 'state_set', 'n_runs', 'dt',
                  'target_lo', 'target_hi']
    runs_df = (pd.DataFrame(run_rows) if run_rows
               else pd.DataFrame(columns=run_cols))
    trials_df = (pd.DataFrame(trial_rows) if trial_rows
                 else pd.DataFrame(columns=trial_cols))
    return runs_df, trials_df


def collect_in_target_moves_sinq(sinq, dfcs=None, drop_tables=True,
                                 verbose=True, across_trials=False,
                                 state_sets=None, **kwargs):
    """``collect_in_target_moves`` across the flies of a Sinq.

    Restores each Table in turn (and drops it again unless ``drop_tables`` is
    False — the traces are large). ``kwargs`` are forwarded, so v_th / v_rest /
    t_min all apply. Returns the concatenated ``(runs_df, trials_df)``.

    ``across_trials=True`` switches to ``collect_in_target_moves_across_trials``,
    so runs may span trial boundaries.

    ``state_sets`` runs the collector once per state set on each restored Table
    — e.g. ``state_sets=(kin.STATE_MOVE, STATES_ACTIVE)`` for the with- and
    without-drift versions in one pass over the data. The results are stacked
    and told apart by the ``state_set`` column, so plot with a
    ``groupby('state_set')``. Restoring is the expensive part, so this costs
    far less than calling the function twice; it is mutually exclusive with
    passing ``state`` directly.
    """
    if state_sets is not None and 'state' in kwargs:
        raise TypeError('pass either state= or state_sets=, not both')
    sets = list(state_sets) if state_sets is not None else [kwargs.pop('state', kin.STATE_MOVE)]
    collector = (collect_in_target_moves_across_trials if across_trials
                 else collect_in_target_moves)
    if dfcs is None:
        dfcs = list(sinq.df.index)
    runs_all, trials_all = [], []
    for i, dfc in enumerate(dfcs, 1):
        if verbose:
            print(f'[{i}/{len(dfcs)}] {dfc}', flush=True)
        T = sinq.restore_table(dfc)
        if T is None:
            continue
        try:
            for st in sets:
                runs_df, trials_df = collector(T, dfc=dfc, state=st, **kwargs)
                runs_all.append(runs_df)
                trials_all.append(trials_df)
        finally:
            if drop_tables:
                sinq.drop_tables(index=[dfc])
                del(T)
                gc.collect()
    if not runs_all:
        return pd.DataFrame(), pd.DataFrame()
    return (pd.concat(runs_all, ignore_index=True),
            pd.concat(trials_all, ignore_index=True))


# ---------------------------------------------------------------------------
# The same measurement, classified across trial boundaries
# ---------------------------------------------------------------------------
# ``trial_in_target_moves`` classifies each trial on its own, so a slow slide
# that carries on past the end of a trial is cut in two, and its second half is
# re-classified without the history that made it a movement. The functions below
# concatenate consecutive trials first, classify once, and split the result up
# afterwards — so a run may span trials, and each sample is tested against the
# target band of the trial it actually came from.

def trace_across_trials(trials):
    """Concatenate the downsampled traces of consecutive trials, keeping track
    of where each sample came from.

    The concatenation is the one ``kin._build_trace`` /
    ``kin.detect_bouts_across_trials`` perform — first-junction dedup included,
    and trials after the second butted against the previous last sample — but
    this also returns the per-sample trial index, the junction sample indices and
    the per-trial time offsets, all of which the library discards.

    Returns ``(t, x, trial_idx, junctions, offsets)``. ``x`` is
    ``-(probe_position - probeZero)``, i.e. positive = more force.
    """
    tr0 = trials[0]
    offsets = [0.0]
    xs = [-(np.asarray(tr0.probe_position) - tr0.probeZero).squeeze()[tr0.downsample_probe]]
    ts = [np.asarray(tr0.time)[tr0.downsample_probe].squeeze()]

    if len(trials) > 1:
        tr1 = trials[1]
        off = float(tr0.total_duration)
        offsets.append(off)
        xs.append(-(np.asarray(tr1.probe_position) - tr1.probeZero).squeeze()[tr1.downsample_probe])
        ts.append(np.asarray(tr1.time)[tr1.downsample_probe].squeeze() + off)

    t, x = np.concatenate(ts), np.concatenate(xs)

    if len(trials) > 1:                     # first-junction dedup, as in _build_trace
        nominal_dt = tr0.frame_length_mode / tr0.params['sampratein']
        keep = np.ones(len(t), dtype=bool)
        keep[1:] = np.diff(t) >= 0.5 * nominal_dt
        t, x = t[keep], x[keep]

    for tr in trials[2:]:                   # butt against the previous last sample
        dt_step = t[-1] - t[-2]
        t_extra = np.asarray(tr.time)[tr.downsample_probe].squeeze()
        off = float(t[-1] + dt_step - t_extra[0])
        offsets.append(off)
        t = np.concatenate([t, t_extra + off])
        x = np.concatenate(
            [x, -(np.asarray(tr.probe_position) - tr.probeZero).squeeze()[tr.downsample_probe]])

    # Assign samples to trials by time, not by counting: the dedup above can drop
    # a sample, and a count would then be off by one from the first junction on.
    starts = np.array([o + float(np.asarray(tr.time)[tr.downsample_probe].squeeze()[0])
                       for tr, o in zip(trials, offsets)])
    trial_idx = np.clip(np.searchsorted(starts, t, side='right') - 1, 0, len(trials) - 1)
    junctions = (np.where(np.diff(trial_idx) != 0)[0] + 1).tolist()
    return t, x, trial_idx, junctions, offsets


def contiguous_trial_chunks(T, trial_numbers=None):
    """Split ``trial_numbers`` into runs of consecutive, usable trials.

    A gap in the numbering, a missing Trial object or an excluded trial ends the
    chunk: the concatenated trace must not pretend to run continuously through a
    trial that was never recorded or was thrown out.
    """
    if trial_numbers is None:
        trial_numbers = list(T.df.index)
    chunks, cur, prev = [], [], None
    for tn in sorted(trial_numbers):
        trial = T.df.at[tn, 'Trial'] if tn in T.df.index else None
        usable = trial is not None and not getattr(trial, 'excluded', False)
        if not usable or (prev is not None and tn != prev + 1):
            if cur:
                chunks.append(cur)
            cur = []
        if usable:
            cur.append(tn)
            prev = tn
        else:
            prev = None
    if cur:
        chunks.append(cur)
    return chunks


NON_TASK_OUTCOMES = ('rest', 'info')


def task_trial_numbers(T, exclude_outcomes=NON_TASK_OUTCOMES, verbose=True):
    """Trial numbers to analyse — everything except the non-task outcomes.

    ``rest`` trials carry a nominal target but deliver no stimulus, so their
    samples answer neither 'where does the fly hold the probe during the task'
    nor 'how much of the task time is spent resting'. ``info`` trials are a
    plotting placeholder rather than a recording; finding one means the Table
    wants looking at, so they are counted and reported separately instead of
    being dropped in silence.

    Dropping trials also breaks the trial numbering, and
    ``contiguous_trial_chunks`` treats a gap as the end of a chunk — so the
    concatenated trace never runs through a trial that was excluded here.
    """
    if 'as_outcome' not in T.df.columns:
        return list(T.df.index)
    outcomes = T.df['as_outcome']
    dropped = {}
    keep = pd.Series(True, index=T.df.index)
    for outcome in exclude_outcomes:
        hit = (outcomes == outcome)
        dropped[outcome] = int(hit.sum())
        keep &= ~hit
    if verbose:
        parts = [f'{n} {name}' for name, n in dropped.items() if n]
        if parts:
            print(f'  excluding {", ".join(parts)} trials')
        if dropped.get('info'):
            print(f'  WARNING: {dropped["info"]} info trials in this Table — '
                  f'these are a plotting placeholder, not a recording')
    return list(T.df.index[keep])


@dataclass
class ClassifiedChunk:
    """One run of consecutive usable trials, concatenated and classified.

    Everything the across-trials measurements need from a chunk, computed once:
    the concatenated trace, the per-sample state, and the per-sample target band
    taken from the trial each sample actually came from. Array fields are all
    the same length as ``time`` unless noted.

    index             position of this chunk in the Table (chunks whose trace
                      is too short to classify are skipped, so indices can skip)
    trial_numbers     the chunk's trial numbers, in order
    pyas_states       each trial's ``pyasState`` ('lo' / 'hi' / 'no_state')
    trial_index       per sample — index into ``trial_numbers``
    time              concatenated clock (s)
    trial_time        each sample back on its own trial's clock (s)
    position          -(probe_position - probeZero), um; positive = more force
    states            per-sample kinematic state
    velocity_smoothed per-sample smoothed velocity (um/s)
    dt                median sample interval of the whole chunk (s)
    bounds_per_trial  (n_trials, 2) array of each trial's (lo, hi) band
    target_lo/hi      per-sample band bounds, from the sample's own trial
    in_target         per sample — inside that band
    window            per sample — inside the t_min / t_max window
    junctions         sample indices where a trial boundary falls
    """
    index: int
    trial_numbers: list
    pyas_states: list
    trial_index: np.ndarray
    time: np.ndarray
    trial_time: np.ndarray
    position: np.ndarray
    states: np.ndarray
    velocity_smoothed: np.ndarray
    dt: float
    bounds_per_trial: np.ndarray
    target_lo: np.ndarray
    target_hi: np.ndarray
    in_target: np.ndarray
    window: np.ndarray
    junctions: list


def _pyas_state_per_trial(T, trial_numbers):
    """Each trial's ``pyasState`` as a plain string, missing values as
    'no_state' — the category the Table itself uses for them."""
    if 'pyasState' not in T.df.columns:
        return ['no_state'] * len(trial_numbers)
    out = []
    for tn in trial_numbers:
        value = T.df.at[tn, 'pyasState']
        out.append('no_state' if pd.isna(value) else str(value))
    return out


def _classified_chunks(T, trial_numbers=None, t_min=None, t_max=None,
                       zero_junctions=True, **bout_kwargs):
    """Yield a ``ClassifiedChunk`` per run of consecutive usable trials.

    The concatenation, the single ``detect_movement_bouts`` pass over the whole
    chunk and the per-sample target band are shared by every across-trials
    measurement below, so they live here rather than in each of them.
    """
    for index, chunk in enumerate(contiguous_trial_chunks(T, trial_numbers)):
        trials = [T.df.at[tn, 'Trial'] for tn in chunk]
        time, position, trial_index, junctions, offsets = trace_across_trials(trials)
        if time.size < 4:
            continue

        _, states, _, _ = kin.detect_movement_bouts(
            time, position, start_time=None,
            junction_indices=junctions if zero_junctions else None,
            **bout_kwargs)
        velocity_smoothed = kin.smoothed_velocity(time, position)
        dt = float(np.median(np.diff(time)))

        # Per-sample target band, taken from the trial the sample belongs to.
        bounds = np.array([target_bounds(tr) for tr in trials], dtype=float)
        target_lo, target_hi = bounds[trial_index, 0], bounds[trial_index, 1]
        in_target = (np.isfinite(target_lo)
                     & (position >= target_lo) & (position <= target_hi))

        trial_time = time - np.asarray(offsets)[trial_index]  # onto each trial's clock
        window = np.ones(len(time), dtype=bool)
        if t_min is not None:
            window &= (trial_time >= t_min)
        if t_max is not None:
            window &= (trial_time <= t_max)

        yield ClassifiedChunk(
            index=index,
            trial_numbers=list(chunk),
            pyas_states=_pyas_state_per_trial(T, chunk),
            trial_index=trial_index,
            time=time,
            trial_time=trial_time,
            position=position,
            states=states,
            velocity_smoothed=velocity_smoothed,
            dt=dt,
            bounds_per_trial=bounds,
            target_lo=target_lo,
            target_hi=target_hi,
            in_target=in_target,
            window=window,
            junctions=junctions,
        )


def in_target_runs_across_trials(T, trial_numbers=None, state=kin.STATE_MOVE,
                                 t_min=None, t_max=None, zero_junctions=True,
                                 **bout_kwargs):
    """In-target runs of ``state``, classified on the concatenated trace.

    Consecutive trials are joined, ``kin.detect_movement_bouts`` runs once over
    the whole chunk, and the in-target test is applied per sample against *that
    sample's own trial's* target band. A run is a maximal stretch of samples that
    are simultaneously in ``state`` and inside the band, and it is free to cross
    trial boundaries.

    Parameters
    ----------
    trial_numbers  : trials to include; default every trial in the Table.
    state          : state, or iterable of states, to measure. ``kin.STATE_MOVE``
                     by default; pass ``kin.STATE_DRIFT`` for slow slides only,
                     or ``STATES_ACTIVE`` for 'not resting'. Every run carries
                     its ``move_time`` / ``drift_time`` split either way.
    t_min, t_max   : optional window in *trial* time (s), applied per sample
                     against its own trial's clock. Default ``None`` keeps the
                     whole trace, so both classification and measurement start
                     at the beginning of each trial rather than at stim onset.
    zero_junctions : zero the velocity for ±``smooth_window`` around each trial
                     junction, so a position step between trials cannot read as
                     a fast movement. That window is shorter than ``Dt``, so it
                     cannot manufacture a REST and split a run on its own.
    bout_kwargs    : forwarded to ``kin.detect_movement_bouts`` (v_th, v_rest,
                     Dt, smooth_window, x_excursion_min, classifier, …).

    Returns
    -------
    ``(runs, trial_summaries)`` — lists of dicts. Run dicts carry the fields
    ``trial_in_target_moves`` returns, plus

        ``trial_start`` / ``trial_end``       trial the run starts / ends in
        ``n_trials``                          how many trials it touches
        ``crosses_boundary``                  n_trials > 1
        ``trial_t_start`` / ``trial_t_end``   trial-clock time of each end
        ``t_start`` / ``t_end``               time in the concatenated frame
        ``chunk``                             index of the concatenated chunk
        ``move_time`` / ``drift_time``        the run's state components (s)
        ``frac_move`` / ``run_kind``          composition of the run

    and ``ends_by`` / ``starts_by`` gain a ``'chunk_edge'`` value for runs cut
    off by the end of a concatenated block rather than by leaving the target or
    by the movement stopping.
    """
    states_wanted = _state_tuple(state)
    label = state_set_label(state)
    runs_out, trials_out = [], []

    for classified in _classified_chunks(T, trial_numbers=trial_numbers,
                                         t_min=t_min, t_max=t_max,
                                         zero_junctions=zero_junctions,
                                         **bout_kwargs):
        i_chunk = classified.index
        chunk = classified.trial_numbers
        t, x = classified.time, classified.position
        trial_idx, states = classified.trial_index, classified.states
        v_sm, dt = classified.velocity_smoothed, classified.dt
        bounds, in_tgt = classified.bounds_per_trial, classified.in_target
        t_rel, window = classified.trial_time, classified.window

        is_state = np.isin(states, states_wanted)
        sel = window & in_tgt & is_state
        n = len(t)
        tn_arr = np.asarray(chunk)
        n_runs_by_trial = {}

        for s, e in _true_runs(sel):
            if e >= n:
                ends_by = 'chunk_edge'
            elif not window[e]:
                ends_by = 'window_edge'
            elif not in_tgt[e]:
                ends_by = 'exit_target'
            else:
                ends_by = 'state_change'
            if s == 0:
                starts_by = 'chunk_edge'
            elif not window[s - 1]:
                starts_by = 'window_edge'
            elif not in_tgt[s - 1]:
                starts_by = 'enter_target'
            else:
                starts_by = 'state_change'
            seg_x = x[s:e]
            touched = np.unique(trial_idx[s:e])
            tn_start = int(tn_arr[trial_idx[s]])
            n_runs_by_trial[tn_start] = n_runs_by_trial.get(tn_start, 0) + 1
            runs_out.append({
                'trial':            tn_start,      # the run's home trial
                'trial_start':      tn_start,
                'trial_end':        int(tn_arr[trial_idx[e - 1]]),
                'n_trials':         int(len(touched)),
                'crosses_boundary': bool(len(touched) > 1),
                'chunk':            i_chunk,
                'duration':         (e - s) * dt,
                'n_samples':        e - s,
                't_start':          float(t[s]),
                't_end':            float(t[e - 1]),
                'trial_t_start':    float(t_rel[s]),
                'trial_t_end':      float(t_rel[e - 1]),
                'x_start':          float(seg_x[0]),
                'x_end':            float(seg_x[-1]),
                'x_ptp':            float(np.ptp(seg_x)),
                'net_dx':           float(seg_x[-1] - seg_x[0]),
                'mean_abs_v':       float(np.mean(np.abs(v_sm[s:e]))),
                **_run_composition(states[s:e], dt),
                'ends_by':          ends_by,
                'starts_by':        starts_by,
            })

        # Per-trial totals: the sample masks split by the trial each sample came
        # from, so in-target time stays attributable even when runs do not.
        for k, tn in enumerate(chunk):
            m = trial_idx == k
            if not m.any():
                continue
            dt_k = float(np.median(np.diff(t[m]))) if m.sum() > 1 else dt
            in_tgt_k = window & m & in_tgt
            trials_out.append({
                'trial':               int(tn),
                'chunk':               i_chunk,
                'window_time':         float((window & m).sum()) * dt_k,
                'in_target_time':      float(in_tgt_k.sum()) * dt_k,
                'move_time':           float((window & m & is_state).sum()) * dt_k,
                'in_target_move_time': float((sel & m).sum()) * dt_k,
                'in_target_move_only_time':
                    float((in_tgt_k & (states == kin.STATE_MOVE)).sum()) * dt_k,
                'in_target_drift_only_time':
                    float((in_tgt_k & (states == kin.STATE_DRIFT)).sum()) * dt_k,
                'state_set':           label,
                'n_runs':              int(n_runs_by_trial.get(int(tn), 0)),
                'dt':                  dt_k,
                'target_lo':           float(bounds[k, 0]),
                'target_hi':           float(bounds[k, 1]),
            })
    return runs_out, trials_out


def collect_in_target_moves_across_trials(T, dfc=None, trial_numbers=None,
                                          state=kin.STATE_MOVE, **kwargs):
    """``collect_in_target_moves``'s cross-trial twin.

    Same ``(runs_df, trials_df)`` contract — ``dfc``, ``trial`` and whatever of
    ``_TRIAL_META_COLS`` the Table carries are attached to both — but the state
    classification runs on concatenated trials, so runs may span them. See
    ``in_target_runs_across_trials`` for the extra run columns. Trial-level
    metadata is that of the run's *starting* trial.
    """
    if dfc is None and getattr(T, 'day', None) is not None:
        dfc = f'{T.day}_F{T.fly}_C{T.cell}'
    runs, trial_rows = in_target_runs_across_trials(
        T, trial_numbers=trial_numbers, state=state, **kwargs)
    label = state_set_label(state)

    def _with_meta(rows):
        return [{'dfc': dfc, 'state_set': label, **r,
                 **_trial_meta(T, r['trial'])} for r in rows]

    run_cols = ['dfc', 'state_set', 'trial', 'trial_start', 'trial_end', 'n_trials',
                'crosses_boundary', 'chunk', 'duration', 'n_samples',
                't_start', 't_end', 'trial_t_start', 'trial_t_end',
                'x_start', 'x_end', 'x_ptp', 'net_dx', 'mean_abs_v',
                'move_time', 'drift_time', 'n_move', 'n_drift', 'frac_move',
                'run_kind', 'ends_by', 'starts_by']
    trial_cols = ['dfc', 'trial', 'chunk', 'window_time', 'in_target_time',
                  'move_time', 'in_target_move_time',
                  'in_target_move_only_time', 'in_target_drift_only_time',
                  'state_set', 'n_runs', 'dt', 'target_lo', 'target_hi']
    runs_df = (pd.DataFrame(_with_meta(runs)) if runs
               else pd.DataFrame(columns=run_cols))
    trials_df = (pd.DataFrame(_with_meta(trial_rows)) if trial_rows
                 else pd.DataFrame(columns=trial_cols))
    return runs_df, trials_df


def run_time_weight(runs, thresholds=None, by='dfc', weight='duration',
                    duration_col='duration'):
    """How much of the in-target time sits in *long* runs.

    For each group and each threshold, the share of ``weight`` contributed by
    runs lasting at least ``threshold`` seconds. Time-weighted, not run-counted:
    a fly with many brief re-entries and a fly with one long hold can have the
    same total in-target time and very different curves here.

    Parameters
    ----------
    runs        : a runs frame from ``collect_in_target_moves*``.
    thresholds  : durations (s) to evaluate. ``None`` uses every observed run
                  duration, which gives the full time-weighted survival curve
                  on a common x for every group.
    by          : grouping column(s); ``'dfc'`` for per-fly curves, e.g.
                  ``['genotype', 'dfc']`` or ``['state_set', 'dfc']`` to keep
                  the with-/without-drift versions apart.
    weight      : which column to weight by. ``'duration'`` weights by total run
                  time; ``'move_time'`` answers 'how much of the *move* time is
                  in long runs' when the runs themselves were cut with drift
                  merged in — the two differ exactly by the drift that glued
                  the long runs together.

    Returns
    -------
    Long-form DataFrame: the ``by`` columns plus ``threshold``, ``weight``,
    ``time_total``, ``time_long``, ``frac_time_long``, ``n_runs``,
    ``n_runs_long``. ``frac_time_long`` is NaN for a group with no run time.
    """
    by = [by] if isinstance(by, str) else list(by)
    if thresholds is None:
        thresholds = np.unique(np.asarray(runs[duration_col].dropna(), dtype=float))
    thresholds = np.atleast_1d(np.asarray(thresholds, dtype=float))

    groups = (runs.groupby(by, observed=True, dropna=False) if by
              else [((), runs)])
    rows = []
    for key, g in groups:
        key = key if isinstance(key, tuple) else (key,)
        total = float(g[weight].sum())
        dur = g[duration_col].to_numpy(dtype=float)
        w = g[weight].to_numpy(dtype=float)
        for th in thresholds:
            m = dur >= th
            long_w = float(w[m].sum())
            rows.append({
                **dict(zip(by, key)),
                'threshold':      float(th),
                'weight':         weight,
                'time_total':     total,
                'time_long':      long_w,
                'frac_time_long': (long_w / total) if total > 0 else np.nan,
                'n_runs':         int(len(g)),
                'n_runs_long':    int(m.sum()),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Where the fly holds the probe: state occupancy on a standardised axis
# ---------------------------------------------------------------------------
# The in-target section above asks how long movement continues once the probe is
# inside the band. This section asks the complementary question about REST: where
# is the probe while the fly is *not* moving, how long does each rest last, and
# how much of that time sits inside the target band rather than past it. Two
# things differ from ``in_target_runs_across_trials``:
#
#   * bouts are NOT gated on being in the target. A rest held 100 um past the hi
#     band is one long rest, not a gap in the record. Each bout instead carries
#     the fraction of its own samples that were in / above / below the band.
#   * a time-weighted position histogram is accumulated per fly while the trace
#     is in memory, on shared bin edges. Restoring a Table costs ~1.5 min, so
#     anything needing per-sample data has to come out of that one pass; the
#     histogram is what makes a pooled position PDF possible without keeping
#     every sample of every fly.
#
# Everything is on one standardised position axis — ``-(probe_position -
# probeZero)`` in um, positive = more force — and ``lo`` and ``hi`` target
# trials are kept apart by the ``pyas_state`` column rather than aligned to each
# other. The band sits at a different absolute position in each fly, so the
# pooled histogram is broad by construction; the quantities that answer 'how
# much rest is past the hi target' are the per-trial times, which are measured
# per sample against that sample's own band and so do not depend on the axis.

DEFAULT_POSITION_BINS = np.arange(-10.0, 502.0, 2.0)

# Subsets of samples each occupancy histogram is accumulated over.
OCCUPANCY_SUBSETS = ('all', 'state', 'state_in_target')


# --- the fly's own noise floor, and v_rest scaled to it ---------------------
# kin thresholds a rolling-RMS speed, and RMS carries jitter power: a probe held
# still but noisily still reads tens of um/s. That is why a noisy recording has
# its rests classified as DRIFT and comes out with a rest fraction near zero.
# The floor is measurable, so v_rest can be set relative to it instead of by
# hand -- which matters because the floor is graded across flies, not bimodal, so
# there is no gap at which to justify a hand-picked per-fly value.
#
# Note only v_rest is worth tuning for this: REST is carved out of the non-MOVE
# remainder by the v_rest test alone, so v_th changes nothing about rest.

def _boxcar_mean(values, n):
    """Centred boxcar mean with edge padding (no zero-pull at the ends)."""
    if n <= 1:
        return np.asarray(values, dtype=float)
    padded = np.pad(np.asarray(values, dtype=float), (n // 2, n - n // 2),
                    mode='edge')
    return np.convolve(padded, np.ones(n) / n, mode='valid')[:len(values)]


def quiet_speed_floor(T, trial_numbers=None, n_trials=200,
                      exclude_outcomes=NON_TASK_OUTCOMES, **bout_kwargs):
    """How fast the probe 'moves' while the fly is not moving it, in um/s.

    Measures the rolling-RMS speed — the quantity ``v_rest`` is compared against
    — over the samples the classifier does **not** call MOVE. Non-MOVE rather
    than REST on purpose: REST is what ``v_rest`` decides, so using it as the
    baseline would presuppose the answer.

    Only the first ``n_trials`` task trials are used; a median needs no more, and
    this runs an extra classification pass over the traces. Pass
    ``n_trials=None`` for all of them.

    Returns a dict —
        ``quiet_p50``               the floor: median RMS speed when not moving
        ``quiet_p10`` / ``quiet_p90``   its spread
        ``rms_p50_all``             median RMS speed over every sample
        ``mean_speed_p50_quiet``    median |boxcar-mean| speed when not moving.
                                    Near zero for every fly measured so far,
                                    which is what shows the difference between
                                    flies is jitter power and not travel.
        ``frac_quiet_above_v_rest`` share of non-MOVE samples already above the
                                    ``v_rest`` in force — the direct read on how
                                    much rest the current threshold is losing
        ``n_samples`` / ``n_trials_used``
    """
    if trial_numbers is None:
        trial_numbers = task_trial_numbers(T, exclude_outcomes=exclude_outcomes,
                                           verbose=False)
    if n_trials is not None:
        trial_numbers = list(trial_numbers)[:n_trials]
    v_rest = bout_kwargs.get('v_rest', kin.V_REST)
    smooth_window = bout_kwargs.get('smooth_window', 0.05)

    rms_all, rms_quiet, mean_quiet = [], [], []
    for chunk in _classified_chunks(T, trial_numbers=trial_numbers, **bout_kwargs):
        n_smooth = max(1, int(round(smooth_window / chunk.dt)))
        velocity = kin.velocity(chunk.time, chunk.position)
        rms_speed = kin._rolling_rms(velocity, n_smooth)
        not_moving = chunk.states != kin.STATE_MOVE
        rms_all.append(rms_speed)
        rms_quiet.append(rms_speed[not_moving])
        mean_quiet.append(np.abs(_boxcar_mean(velocity, n_smooth))[not_moving])

    if not rms_quiet or not sum(len(a) for a in rms_quiet):
        return {'quiet_p50': np.nan, 'quiet_p10': np.nan, 'quiet_p90': np.nan,
                'rms_p50_all': np.nan, 'mean_speed_p50_quiet': np.nan,
                'frac_quiet_above_v_rest': np.nan, 'n_samples': 0,
                'n_trials_used': len(trial_numbers)}

    rms_all = np.concatenate(rms_all)
    rms_quiet = np.concatenate(rms_quiet)
    mean_quiet = np.concatenate(mean_quiet)
    return {
        'quiet_p50':               float(np.median(rms_quiet)),
        'quiet_p10':               float(np.percentile(rms_quiet, 10)),
        'quiet_p90':               float(np.percentile(rms_quiet, 90)),
        'rms_p50_all':             float(np.median(rms_all)),
        'mean_speed_p50_quiet':    float(np.median(mean_quiet)),
        'frac_quiet_above_v_rest': float(np.mean(rms_quiet > v_rest)),
        'n_samples':               int(rms_quiet.size),
        'n_trials_used':           len(trial_numbers),
    }


def noise_scaled_v_rest(floor, multiple=1.0, statistic='quiet_p90',
                        minimum=None, maximum=90.0, round_to=1.0):
    """``v_rest`` set from a fly's own non-MOVE speed distribution.

    ``floor`` is either a number or the dict ``quiet_speed_floor`` returns;
    ``statistic`` picks which of its entries to use when it is a dict.

    The default is the **p90 rule**: ``v_rest`` = the 90th percentile of that
    fly's non-MOVE RMS speed, i.e. every fly is allowed the same ~10% of quiet
    samples above threshold. That is a quantile normalisation rather than a
    scale one, and it is what the measurements support — kin's default 30 um/s
    already sits near p90-p95 for a clean recording (5-22% of quiet samples
    above it), while on a noisy one it sits near the median (36-64% above), which
    is exactly why those rests are classified DRIFT.

    A multiple of the *median* (``statistic='quiet_p50'``, ``multiple=1.5-2``)
    was the first idea and it does not work: 1.5x drops the cleanest fly's
    ``v_rest`` below the default and manufactures drift (rest 0.77 -> 0.47),
    while 2x inflates a mid-range control by +0.36 rest. The p90 rule leaves
    clean flies where they already were.

    ``minimum`` defaults to ``kin.V_REST``, so no fly ends up with a *lower*
    threshold than the current global default — the rule may recover rest that
    noise was hiding, but it should not take rest away from a clean recording.
    ``maximum`` matters too: too high and genuinely slow movement is absorbed
    into REST. Pass ``minimum=None`` to disable the floor.
    """
    if minimum is None:
        minimum = kin.V_REST
    if isinstance(floor, dict):
        floor = floor[statistic]
    if not np.isfinite(floor):
        return float(kin.V_REST)
    value = float(floor) * multiple
    if minimum is not None:
        value = max(value, minimum)
    if maximum is not None:
        value = min(value, maximum)
    if round_to:
        value = round(value / round_to) * round_to
    return float(value)


def _percentile_fields(values, prefix='position'):
    """p25 / p50 / p75 of ``values``, NaN throughout if it is empty."""
    if len(values):
        p25, p50, p75 = np.percentile(values, [25, 50, 75])
    else:
        p25 = p50 = p75 = np.nan
    return {f'{prefix}_p25': float(p25),
            f'{prefix}_p50': float(p50),
            f'{prefix}_p75': float(p75)}


class _OccupancyAccumulator:
    """Time-weighted position histograms, one per (pyas_state, subset).

    Time rather than samples, so a fly recorded at a different frame rate is
    still comparable. Samples outside the bin range are counted into ``below`` /
    ``above`` rather than dropped, so the total is always conserved and a badly
    placed axis shows up as mass in the overflow rather than as a quietly
    truncated curve.
    """

    def __init__(self, bins):
        self.bins = np.asarray(bins, dtype=float)
        self.histograms = {}

    def add(self, pyas_state, subset, positions, seconds_per_sample):
        key = (pyas_state, subset)
        if key not in self.histograms:
            self.histograms[key] = {'seconds': np.zeros(len(self.bins) - 1),
                                    'below': 0.0, 'above': 0.0}
        entry = self.histograms[key]
        if not len(positions):
            return
        counts, _ = np.histogram(positions, bins=self.bins)
        entry['seconds'] += counts * seconds_per_sample
        entry['below'] += float(np.count_nonzero(positions < self.bins[0])) * seconds_per_sample
        entry['above'] += float(np.count_nonzero(positions > self.bins[-1])) * seconds_per_sample

    def to_rows(self, state_label):
        """Long-form rows: one per (pyas_state, subset, bin), plus one
        underflow and one overflow row per (pyas_state, subset)."""
        rows = []
        left_edges, right_edges = self.bins[:-1], self.bins[1:]
        centres = 0.5 * (left_edges + right_edges)
        for (pyas_state, subset), entry in self.histograms.items():
            common = {'state': state_label, 'pyas_state': pyas_state,
                      'subset': subset}
            for left, right, centre, seconds in zip(left_edges, right_edges,
                                                    centres, entry['seconds']):
                rows.append({**common, 'bin_left': float(left),
                             'bin_right': float(right),
                             'bin_centre': float(centre),
                             'bin_width': float(right - left),
                             'seconds': float(seconds)})
            rows.append({**common, 'bin_left': -np.inf,
                         'bin_right': float(self.bins[0]), 'bin_centre': np.nan,
                         'bin_width': np.nan, 'seconds': entry['below']})
            rows.append({**common, 'bin_left': float(self.bins[-1]),
                         'bin_right': np.inf, 'bin_centre': np.nan,
                         'bin_width': np.nan, 'seconds': entry['above']})
        return rows


def state_occupancy_across_trials(T, trial_numbers=None, state=kin.STATE_REST,
                                  t_min=None, t_max=None, zero_junctions=True,
                                  position_bins=None, **bout_kwargs):
    """Bouts of ``state``, per-trial time budget, and position occupancy.

    States are classified once per chunk of consecutive trials
    (``_classified_chunks``), so a rest that outlasts a trial stays one bout
    instead of being cut in two and reclassified without its history — which
    matters more for REST than for MOVE, since rests routinely run past the end
    of a trial.

    Parameters
    ----------
    trial_numbers  : trials to include; default every trial in the Table. Pass
                     ``task_trial_numbers(T)`` to drop the non-task outcomes.
    state          : state, or iterable of states, to measure. REST by default.
    t_min, t_max   : optional window in *trial* time (s), applied per sample
                     against its own trial's clock. Default ``None`` keeps the
                     whole trace: the question is where the fly holds the probe
                     generally, not only after stimulus onset.
    position_bins  : edges for the occupancy histogram, in um on the flipped
                     axis. Defaults to ``DEFAULT_POSITION_BINS`` (-10 to 500 in
                     2 um steps) so every fly and group share one axis.
    bout_kwargs    : forwarded to ``kin.detect_movement_bouts`` — ``v_rest`` is
                     the one that matters here, since it is the DRIFT/REST
                     boundary: a fly whose baseline noise sits above ``v_rest``
                     reads as drifting while it is in fact holding still.

    Returns
    -------
    ``(bouts, trials, occupancy)`` — three lists of dicts.

    bouts : one row per contiguous run of ``state``, ungated by target —
        ``duration`` / ``n_samples``       bout length
        ``trial`` / ``trial_start`` / ``trial_end`` / ``n_trials`` /
        ``crosses_boundary``               which trials it touches
        ``pyas_state``                     of the trial the bout starts in
        ``position_mean`` / ``_p25`` / ``_p50`` / ``_p75`` / ``_start`` /
        ``_end`` / ``_ptp``                where the probe sat during it
        ``target_lo`` / ``target_hi``      band of the starting trial
        ``position_p50_rel_hi``            median position minus ``target_hi``:
                                           positive = held past the hi edge
        ``frac_in_target`` / ``frac_above_target`` / ``frac_below_target``
                                           fractions of the bout's own samples
        ``time_in_target`` / ``time_above_target``   the same as seconds
        ``mean_abs_velocity``              mean |smoothed v| over the bout
        ``starts_by`` / ``ends_by``        'state_change' | 'window_edge' |
                                           'chunk_edge'

    trials : one row per trial, with the time budget the fractions come from —
        ``total_time``                     analysed time in the trial
        ``state_time``                     time in ``state``
        ``rest_time`` / ``drift_time`` / ``move_time``
                                           the full budget regardless of which
                                           ``state`` was asked for, so the
                                           rest/drift/move fractions come out of
                                           this pass rather than a second one
        ``in_target_time``                 time in the band, any state
        ``state_in_target_time``           time in ``state`` and in the band
        ``state_above_target_time`` / ``state_below_target_time``
        ``n_bouts``                        bouts *starting* in this trial
        ``position_p25`` / ``_p50`` / ``_p75``   position percentiles within
                                           ``state`` on this trial — the
                                           ``plot_state_positions`` quantity
        ``target_lo`` / ``target_hi`` / ``dt`` / ``pyas_state``

    occupancy : long-form position histogram for the whole Table, one row per
        (``pyas_state``, ``subset``, bin), where ``subset`` is
        ``all`` (every analysed sample), ``state`` (samples in ``state``) or
        ``state_in_target`` (in ``state`` and inside the band). ``seconds`` is
        time, not samples. Underflow / overflow rows carry a non-finite
        ``bin_centre``.
    """
    if position_bins is None:
        position_bins = DEFAULT_POSITION_BINS
    states_wanted = _state_tuple(state)
    state_label = state_set_label(state)

    bout_rows, trial_rows = [], []
    occupancy = _OccupancyAccumulator(position_bins)

    for classified in _classified_chunks(T, trial_numbers=trial_numbers,
                                         t_min=t_min, t_max=t_max,
                                         zero_junctions=zero_junctions,
                                         **bout_kwargs):
        position = classified.position
        is_state = np.isin(classified.states, states_wanted)
        has_band = np.isfinite(classified.target_lo)
        above_target = has_band & (position > classified.target_hi)
        below_target = has_band & (position < classified.target_lo)
        selected = classified.window & is_state
        n_samples = len(classified.time)
        bouts_by_trial = {}

        # --- one row per bout of the state ---------------------------------
        for start, stop in _true_runs(selected):
            if stop >= n_samples:
                ends_by = 'chunk_edge'
            elif not classified.window[stop]:
                ends_by = 'window_edge'
            else:
                ends_by = 'state_change'
            if start == 0:
                starts_by = 'chunk_edge'
            elif not classified.window[start - 1]:
                starts_by = 'window_edge'
            else:
                starts_by = 'state_change'

            bout_position = position[start:stop]
            trials_touched = np.unique(classified.trial_index[start:stop])
            home = classified.trial_index[start]
            trial_home = int(classified.trial_numbers[home])
            bouts_by_trial[trial_home] = bouts_by_trial.get(trial_home, 0) + 1
            band_lo, band_hi = classified.bounds_per_trial[home]
            in_target_here = int(np.count_nonzero(classified.in_target[start:stop]))
            above_here = int(np.count_nonzero(above_target[start:stop]))
            below_here = int(np.count_nonzero(below_target[start:stop]))
            length = stop - start
            median_position = float(np.median(bout_position))

            bout_rows.append({
                'trial':              trial_home,      # the bout's home trial
                'trial_start':        trial_home,
                'trial_end':          int(classified.trial_numbers[
                                          classified.trial_index[stop - 1]]),
                'n_trials':           int(len(trials_touched)),
                'crosses_boundary':   bool(len(trials_touched) > 1),
                'chunk':              classified.index,
                'pyas_state':         classified.pyas_states[home],
                'duration':           length * classified.dt,
                'n_samples':          int(length),
                't_start':            float(classified.time[start]),
                't_end':              float(classified.time[stop - 1]),
                'trial_t_start':      float(classified.trial_time[start]),
                'trial_t_end':        float(classified.trial_time[stop - 1]),
                'position_start':     float(bout_position[0]),
                'position_end':       float(bout_position[-1]),
                'position_mean':      float(np.mean(bout_position)),
                **_percentile_fields(bout_position),
                'position_ptp':       float(np.ptp(bout_position)),
                'target_lo':          float(band_lo),
                'target_hi':          float(band_hi),
                'position_p50_rel_hi': median_position - float(band_hi),
                'frac_in_target':     in_target_here / length,
                'frac_above_target':  above_here / length,
                'frac_below_target':  below_here / length,
                'time_in_target':     in_target_here * classified.dt,
                'time_above_target':  above_here * classified.dt,
                'mean_abs_velocity':  float(np.mean(np.abs(
                                          classified.velocity_smoothed[start:stop]))),
                'starts_by':          starts_by,
                'ends_by':            ends_by,
            })

        # --- one row per trial, plus that trial's contribution to occupancy -
        # Split by the trial each sample came from, so time stays attributable
        # even when a bout does not, and so the lo / hi histograms never mix.
        for position_in_chunk, trial_number in enumerate(classified.trial_numbers):
            from_this_trial = classified.trial_index == position_in_chunk
            if not from_this_trial.any():
                continue
            analysed = classified.window & from_this_trial
            in_state = analysed & is_state
            in_state_in_target = in_state & classified.in_target
            dt_trial = (float(np.median(np.diff(classified.time[from_this_trial])))
                        if from_this_trial.sum() > 1 else classified.dt)
            pyas_state = classified.pyas_states[position_in_chunk]
            band_lo, band_hi = classified.bounds_per_trial[position_in_chunk]

            trial_rows.append({
                'trial':                    int(trial_number),
                'chunk':                    classified.index,
                'pyas_state':               pyas_state,
                'dt':                       dt_trial,
                'total_time':               float(analysed.sum()) * dt_trial,
                'state_time':               float(in_state.sum()) * dt_trial,
                # The whole budget, regardless of which state was asked for, so
                # the rest/drift/move fractions need no second pass.
                'rest_time':                float((analysed & (classified.states == kin.STATE_REST)).sum()) * dt_trial,
                'drift_time':               float((analysed & (classified.states == kin.STATE_DRIFT)).sum()) * dt_trial,
                'move_time':                float((analysed & (classified.states == kin.STATE_MOVE)).sum()) * dt_trial,
                'in_target_time':           float((analysed & classified.in_target).sum()) * dt_trial,
                'state_in_target_time':     float(in_state_in_target.sum()) * dt_trial,
                'state_above_target_time':  float((in_state & above_target).sum()) * dt_trial,
                'state_below_target_time':  float((in_state & below_target).sum()) * dt_trial,
                'n_samples':                int(analysed.sum()),
                'n_state_samples':          int(in_state.sum()),
                'n_bouts':                  int(bouts_by_trial.get(int(trial_number), 0)),
                **_percentile_fields(position[in_state]),
                'target_lo':                float(band_lo),
                'target_hi':                float(band_hi),
            })

            occupancy.add(pyas_state, 'all', position[analysed], dt_trial)
            occupancy.add(pyas_state, 'state', position[in_state], dt_trial)
            occupancy.add(pyas_state, 'state_in_target',
                          position[in_state_in_target], dt_trial)

    return bout_rows, trial_rows, occupancy.to_rows(state_label)


_OCCUPANCY_BOUT_COLS = [
    'dfc', 'state', 'trial', 'trial_start', 'trial_end', 'n_trials',
    'crosses_boundary', 'chunk', 'pyas_state', 'duration', 'n_samples',
    't_start', 't_end', 'trial_t_start', 'trial_t_end',
    'position_start', 'position_end', 'position_mean',
    'position_p25', 'position_p50', 'position_p75', 'position_ptp',
    'target_lo', 'target_hi', 'position_p50_rel_hi',
    'frac_in_target', 'frac_above_target', 'frac_below_target',
    'time_in_target', 'time_above_target', 'mean_abs_velocity',
    'starts_by', 'ends_by',
]

_OCCUPANCY_TRIAL_COLS = [
    'dfc', 'state', 'trial', 'chunk', 'pyas_state', 'dt',
    'total_time', 'state_time', 'rest_time', 'drift_time', 'move_time',
    'in_target_time', 'state_in_target_time',
    'state_above_target_time', 'state_below_target_time',
    'n_samples', 'n_state_samples', 'n_bouts',
    'position_p25', 'position_p50', 'position_p75', 'target_lo', 'target_hi',
]

_OCCUPANCY_HIST_COLS = [
    'dfc', 'state', 'pyas_state', 'subset',
    'bin_left', 'bin_right', 'bin_centre', 'bin_width', 'seconds',
]


def collect_state_occupancy(T, dfc=None, state=kin.STATE_REST,
                            trial_numbers=None,
                            exclude_outcomes=NON_TASK_OUTCOMES,
                            verbose=True, **kwargs):
    """``state_occupancy_across_trials`` over one Table, as three DataFrames.

    Non-task trials (``exclude_outcomes``) are dropped before the traces are
    concatenated, so they neither contribute samples nor let a chunk run through
    them. Pass ``trial_numbers`` explicitly to override that choice, or
    ``exclude_outcomes=()`` to keep everything.

    Returns ``(bouts, trials, occupancy)``. ``bouts`` and ``trials`` carry
    ``dfc`` plus whatever of ``_TRIAL_META_COLS`` the Table has, so downstream
    filtering is a ``.query``.
    """
    if dfc is None and getattr(T, 'day', None) is not None:
        dfc = f'{T.day}_F{T.fly}_C{T.cell}'
    if trial_numbers is None:
        trial_numbers = task_trial_numbers(T, exclude_outcomes=exclude_outcomes,
                                           verbose=verbose)
    bouts, trials, occupancy = state_occupancy_across_trials(
        T, trial_numbers=trial_numbers, state=state, **kwargs)
    state_label = state_set_label(state)

    def with_meta(rows):
        return [{'dfc': dfc, 'state': state_label, **row,
                 **_trial_meta(T, row['trial'])} for row in rows]

    def frame(rows, columns):
        return pd.DataFrame(rows) if rows else pd.DataFrame(columns=columns)

    return (frame(with_meta(bouts), _OCCUPANCY_BOUT_COLS),
            frame(with_meta(trials), _OCCUPANCY_TRIAL_COLS),
            frame([{'dfc': dfc, **row} for row in occupancy],
                  _OCCUPANCY_HIST_COLS))


def collect_state_occupancy_sinq(sinq, dfcs=None, state=kin.STATE_REST,
                                 bout_kwargs_by_fly=None, v_rest_from_noise=None,
                                 noise_kwargs=None, drop_tables=True,
                                 verbose=True, **kwargs):
    """``collect_state_occupancy`` across the flies of a Sinq.

    Restores each Table in turn and drops it again (the traces are large), so
    this is the expensive call: ~1.5 min per fly.

    ``v_rest_from_noise`` sets each fly's ``v_rest`` to that multiple of its own
    quiet-RMS floor (``quiet_speed_floor`` → ``noise_scaled_v_rest``), measured
    while the Table is already in memory. ``2.0`` is the calibrated value: it
    leaves a clean recording near kin's default of 30 um/s and only moves the
    boundary where the noise actually is. Prefer this to hand-set overrides —
    the floor is graded across flies rather than bimodal, so there is no gap at
    which a hand-picked per-fly value can be justified. ``noise_kwargs`` is
    passed to ``noise_scaled_v_rest`` (``minimum``, ``maximum``, ``round_to``).

    ``bout_kwargs_by_fly`` overrides the settings for individual flies and
    **wins over** ``v_rest_from_noise``, so a fly can still be pinned by hand.

    The settings actually used are written into the ``v_th`` / ``v_rest`` columns
    of both the bouts and trials frames, along with ``quiet_floor`` when it was
    measured, so a cached parquet always says what produced it. ``genotype`` is
    copied across from the Sinq where it has one.
    """
    if dfcs is None:
        dfcs = list(sinq.df.index)
    bout_kwargs_by_fly = bout_kwargs_by_fly or {}
    noise_kwargs = noise_kwargs or {}
    all_bouts, all_trials, all_occupancy = [], [], []

    for i, dfc in enumerate(dfcs, 1):
        per_fly = dict(kwargs, **bout_kwargs_by_fly.get(dfc, {}))
        if verbose:
            note = ''
            if dfc in bout_kwargs_by_fly:
                note = f'  [pinned: {bout_kwargs_by_fly[dfc]}]'
            print(f'[{i}/{len(dfcs)}] {dfc}{note}', flush=True)
        T = sinq.restore_table(dfc)
        if T is None:
            continue
        try:
            floor = None
            if v_rest_from_noise and 'v_rest' not in per_fly:
                floor = quiet_speed_floor(
                    T, **{k: v for k, v in per_fly.items()
                          if k not in ('t_min', 't_max', 'position_bins',
                                       'trial_numbers', 'exclude_outcomes')})
                per_fly['v_rest'] = noise_scaled_v_rest(
                    floor, multiple=v_rest_from_noise, **noise_kwargs)
                if verbose:
                    statistic = noise_kwargs.get('statistic', 'quiet_p90')
                    print(f'      quiet RMS p50={floor["quiet_p50"]:.1f} '
                          f'p90={floor["quiet_p90"]:.1f} um/s '
                          f'({floor["frac_quiet_above_v_rest"]:.0%} of quiet samples above '
                          f'the kin default {kin.V_REST:.0f}) -> v_rest='
                          f'{per_fly["v_rest"]:.0f} from {statistic}', flush=True)
            bouts, trials, occupancy = collect_state_occupancy(
                T, dfc=dfc, state=state, verbose=verbose, **per_fly)
            settings = {'v_th': per_fly.get('v_th', kin.V_TH),
                        'v_rest': per_fly.get('v_rest', kin.V_REST),
                        'quiet_floor': floor['quiet_p50'] if floor else np.nan,
                        'quiet_p90': floor['quiet_p90'] if floor else np.nan}
            for frame in (bouts, trials):
                for name, value in settings.items():
                    frame[name] = value
            all_bouts.append(bouts)
            all_trials.append(trials)
            all_occupancy.append(occupancy)
        finally:
            if drop_tables:
                sinq.drop_tables(index=[dfc])
                del T
                gc.collect()

    if not all_trials:
        return (pd.DataFrame(columns=_OCCUPANCY_BOUT_COLS),
                pd.DataFrame(columns=_OCCUPANCY_TRIAL_COLS),
                pd.DataFrame(columns=_OCCUPANCY_HIST_COLS))

    bouts = pd.concat(all_bouts, ignore_index=True)
    trials = pd.concat(all_trials, ignore_index=True)
    occupancy = pd.concat(all_occupancy, ignore_index=True)
    if 'genotype' in sinq.df.columns:
        for frame in (bouts, trials, occupancy):
            frame['genotype'] = frame['dfc'].map(sinq.df['genotype'])
    return bouts, trials, occupancy


# ---------------------------------------------------------------------------
# Turning the occupancy frames into per-fly summaries and PDFs
# ---------------------------------------------------------------------------

def occupancy_summary_by_fly(bouts, trials, pyas_states=None):
    """One row per fly: the time budget and the bout-duration summary.

    ``pyas_states`` restricts to e.g. ``('hi',)``; the default pools lo and hi,
    which is fine for the time fractions (each is measured against its own
    trial's band) but note it mixes two band positions in the raw
    ``position_p50`` columns.

    Fractions, all per fly —
        ``rest_fraction``            state time / analysed time
        ``frac_state_in_target``     of the time in the state, how much in band
        ``frac_state_above_target``  ... and how much past the hi edge
        ``frac_time_state_in_target`` state-and-in-band time / analysed time —
                                     'what fraction of total time is spent
                                     resting in the target'
    Bout durations are summarised twice, because the two disagree strongly: by
    count (``median_bout_s``, most bouts are brief) and by time
    (``time_weighted_median_bout_s``, where the time actually sits).
    """
    if pyas_states is not None:
        trials = trials[trials['pyas_state'].isin(pyas_states)]
        bouts = bouts[bouts['pyas_state'].isin(pyas_states)]

    by_trial = trials.groupby('dfc', observed=True)
    summary = pd.DataFrame({
        'n_trials':                by_trial.size(),
        'total_time':              by_trial['total_time'].sum(),
        'state_time':              by_trial['state_time'].sum(),
        'in_target_time':          by_trial['in_target_time'].sum(),
        'state_in_target_time':    by_trial['state_in_target_time'].sum(),
        'state_above_target_time': by_trial['state_above_target_time'].sum(),
        'state_below_target_time': by_trial['state_below_target_time'].sum(),
        'median_trial_position_p50': by_trial['position_p50'].median(),
    })
    for column in ('genotype', 'v_th', 'v_rest', 'quiet_floor', 'quiet_p90'):
        if column in trials.columns:
            summary[column] = by_trial[column].first()

    # The rest/drift/move budget, named to match the Sinq scalar columns
    # (rest_time_fraction / drift_time_fraction / move_time_fraction) so the
    # existing fraction_lines() plotting works on this frame unchanged. Note
    # these are NOT the same numbers as the cached Sinq scalars: those classify
    # each trial on its own and count post-stim samples only, while these
    # classify across concatenated trials over the whole trace.
    for state_name in ('rest', 'drift', 'move'):
        column = f'{state_name}_time'
        if column in trials.columns:
            summary[column] = by_trial[column].sum()
            summary[f'{state_name}_time_fraction'] = (summary[column]
                                                      / summary['total_time'])

    summary['rest_fraction'] = summary['state_time'] / summary['total_time']
    summary['frac_state_in_target'] = (summary['state_in_target_time']
                                       / summary['state_time'])
    summary['frac_state_above_target'] = (summary['state_above_target_time']
                                          / summary['state_time'])
    summary['frac_time_state_in_target'] = (summary['state_in_target_time']
                                            / summary['total_time'])
    summary['frac_time_in_target'] = summary['in_target_time'] / summary['total_time']

    if len(bouts):
        by_bout = bouts.groupby('dfc', observed=True)
        summary['n_bouts'] = by_bout.size()
        summary['median_bout_s'] = by_bout['duration'].median()
        summary['p90_bout_s'] = by_bout['duration'].quantile(0.9)
        summary['time_weighted_median_bout_s'] = by_bout['duration'].apply(
            _time_weighted_median)
        summary['median_bout_position_p50'] = by_bout['position_p50'].median()
        summary['median_bout_position_rel_hi'] = by_bout['position_p50_rel_hi'].median()
    return summary


def _time_weighted_median(durations):
    """The duration at which half the *time* sits in shorter bouts.

    Counted per bout the distribution is dominated by brief events; weighted by
    time it is dominated by the few long ones. This is the second number.
    """
    values = np.sort(np.asarray(durations, dtype=float))
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan
    cumulative = np.cumsum(values)
    return float(values[np.searchsorted(cumulative, cumulative[-1] / 2.0)])


def occupancy_density(occupancy, trials, subset='state', normalise='total_time',
                      pyas_states=('lo', 'hi')):
    """Per-fly position PDFs from the occupancy histogram.

    Parameters
    ----------
    subset      : ``'all'`` | ``'state'`` | ``'state_in_target'``.
    normalise   : ``'total_time'`` — divide by the fly's whole analysed time, so
                  the **area under the curve is that subset's share of the fly's
                  time**. Two flies' curves are then directly comparable, and
                  the area under the ``state_in_target`` curve is the answer to
                  'what fraction of total time is spent resting in target'.
                  ``'subset_time'`` — area 1 within the subset, i.e. shape only.
    pyas_states : which target states to keep, as separate rows. Never pooled:
                  the lo and hi bands sit at different positions on the axis.

    Returns long-form ``dfc, pyas_state, bin_centre, bin_width, seconds,
    density`` (density in fraction of time per um). Underflow / overflow bins
    are dropped here — they have no width to spread over — and the seconds they
    held are reported if they are not negligible.
    """
    if subset not in OCCUPANCY_SUBSETS:
        raise ValueError(f'unknown subset {subset!r}; expected one of {OCCUPANCY_SUBSETS}')
    selected = occupancy[occupancy['subset'] == subset]
    if selected.empty:
        raise ValueError(f'no occupancy rows with subset={subset!r}')
    if pyas_states is not None:
        selected = selected[selected['pyas_state'].isin(pyas_states)]

    finite = np.isfinite(selected['bin_centre'])
    off_axis = selected.loc[~finite, 'seconds'].sum()
    on_axis = selected.loc[finite, 'seconds'].sum()
    if off_axis > 0.01 * max(on_axis, 1e-9):
        print(f'  note: {off_axis:.1f} s of {subset} time ({off_axis / (on_axis + off_axis):.1%}) '
              f'falls outside the position bins and is not plotted')
    selected = selected.loc[finite].copy()

    if normalise == 'total_time':
        denominator = (trials[trials['pyas_state'].isin(pyas_states)]
                       if pyas_states is not None else trials)
        denominator = denominator.groupby(['dfc', 'pyas_state'],
                                          observed=True)['total_time'].sum()
        keys = pd.MultiIndex.from_frame(selected[['dfc', 'pyas_state']])
        seconds_total = denominator.reindex(keys).to_numpy()
    elif normalise == 'subset_time':
        seconds_total = selected.groupby(['dfc', 'pyas_state'],
                                         observed=True)['seconds'].transform('sum').to_numpy()
    else:
        raise ValueError(f'unknown normalise: {normalise!r}')

    with np.errstate(divide='ignore', invalid='ignore'):
        selected['density'] = (selected['seconds'].to_numpy()
                               / seconds_total / selected['bin_width'].to_numpy())
    selected['subset'] = subset
    selected['normalise'] = normalise
    return selected


def pool_density(density, group_columns=('group', 'pyas_state')):
    """Average the per-fly PDFs within each group — mean, sem and n flies.

    Averaged across flies rather than pooled across samples, so a fly with three
    times as many trials does not count three times.
    """
    group_columns = list(group_columns) + ['bin_centre', 'bin_width']
    grouped = density.groupby(group_columns, observed=True)['density']
    pooled = grouped.agg(['mean', 'sem', 'count']).reset_index()
    return pooled.rename(columns={'mean': 'density_mean',
                                  'sem': 'density_sem',
                                  'count': 'n_flies'})


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


# ---------------------------------------------------------------------------
# Lead / lag between firing rate and probe position, by condition
# ---------------------------------------------------------------------------
# The first attempt at this was uninterpretable for two reasons, both fixed here.
#
# 1. It pooled conditions whose lags run in *opposite* directions, which cancel.
#    The piezo cue imposes movement on the animal (probe leads rate); the run-up
#    to a voluntary movement has the rate change first (rate leads probe).
#
# 2. It correlated rate against *velocity* only. Both pairings are defensible and
#    they correspond to different mechanical regimes, so ``signal_col`` is a
#    hypothesis under test, not a preprocessing decision:
#
#      position  - while the fly HOLDS, the probe is a linear spring
#                  (``kinematics.k_spring_constant`` = 0.0829 uN/um), so force
#                  balance gives x = F/k and rate pairs with position.
#      velocity  - while the fly MOVES, the muscle is shortening, and under a
#                  Hill-type force-velocity relation activation sets shortening
#                  velocity, so rate pairs with velocity.
#
#    Velocity leads position by a quarter cycle (differentiating a sinusoid
#    advances its phase by 90 degrees), so a lag measured against velocity is
#    SHORTER than the same lag measured against position. The shift equals T/4
#    only for a narrowband signal: measured on 210915_F1_C1 it is 200 ms for
#    'rest' against a predicted 205 ms, but 125 ms for 'move' against a predicted
#    345 ms - transients are not sinusoids. The two pairings are therefore NOT
#    inter-convertible, and neither should be reported as if it were the other.
#
# What a lag between A2's rate and position does NOT license: expecting A2's rate
# to predict position. A2 is one of a population of motor neurons, so position
# reflects the summed drive. Reading a lag as a muscle activation latency requires
# a condition where A2 is known to be the drive - which is what the calibration
# conditions establish and what the 'move' condition does not.

#: What each condition should show if the method works. Three of the four have a
#: known answer, so they calibrate the estimator before it is used on the fourth.
CCF_EXPECTATION = {
    'cue':     {'lag': 'negative (probe leads rate)',
                'corr': 'negative - flexion hyperpolarizes a flexor '
                        '(resistance reflex)',
                'why': 'the piezo imposes the movement'},
    'premove': {'lag': 'positive (rate leads probe)',
                'corr': 'positive',
                'why': 'the rate changes before the probe moves'},
    'rest':    {'lag': 'no structure at short lags',
                'corr': 'positive but slow only',
                'why': 'position is not changing, so fast rate fluctuations '
                       'cannot move it - a null control'},
    'current_step': {'lag': 'none - no probe response at any lag',
                'corr': 'none',
                'why': 'the injected step displaces Vm and rate with the probe '
                       'held still, so it is the mirror of the cue: the cue '
                       'shows an imposed movement timed as probe-first, this '
                       'shows an imposed rate change producing no movement. Any '
                       'lag found here is manufactured by the method.'},
    'move':    {'lag': 'UNKNOWN - do not presume a direction',
                'corr': 'UNKNOWN - do not presume a sign',
                'why': 'A2 is one motor neuron in a population. Other MNs recruit '
                       'during movement, so probe position reflects the summed '
                       'motor drive and A2 is only one component of it. The '
                       'pre-movement data make the point directly: A2 rate '
                       'changes there that do NOT immediately move the probe. So '
                       'neither the sign nor the existence of a lag can be '
                       'assumed for this condition - it is the open question, not '
                       'a prediction to confirm.'},
}


def condition_masks(records, cue_post_s=0.2, premove_s=0.5, split_premove=False,
                    include_current_step=True):
    """The lead/lag conditions as boolean masks over ``records``.

    Four carry a known answer and so calibrate the estimator (see
    ``CCF_EXPECTATION``); ``move`` is the one being asked about. ``rest`` is the
    null control: the probe is not moving, so a fast rate fluctuation cannot
    displace it, and any short-lag structure there is an artifact of the method.

    ``split_premove`` separates the run-up by the direction of the upcoming
    movement, since a rate *increase* precedes force-up movements while a
    decrease precedes relaxations.

    ``include_current_step`` adds the injected pre-cue step as its own condition
    and removes its frames from the others. It belongs with the calibration
    conditions because it is the only imposed perturbation that moves the rate
    without moving the probe. Note what this is and is not for: pulling those
    frames out of ``rest`` barely changes it -- about 0.6% of rest frames on
    210915_F1_C1, shifting pooled rho(Vm,rate) from 0.113 to 0.111 -- so the
    point is the extra control, not repairing ``rest``. Pass ``False`` for the
    previous four-condition behaviour.
    """
    cue = cue_mask(records, post_s=cue_post_s)
    istep = (current_step_mask(records) if include_current_step
             else np.zeros(len(records), dtype=bool))
    cue = cue & ~istep
    pm = (records['t_to_move'].to_numpy() <= premove_s
          if 't_to_move' in records.columns
          else np.zeros(len(records), dtype=bool))
    is_rest = np.isin(records['state'].to_numpy(), _state_tuple(kin.STATE_REST))
    is_move = np.isin(records['state'].to_numpy(), _state_tuple(kin.STATE_MOVE))
    out = {'cue': cue}
    if include_current_step:
        out['current_step'] = istep
    keep = ~cue & ~istep
    if split_premove and 'next_dx_init' in records.columns:
        dxi = records['next_dx_init'].to_numpy()
        out['premove_toward'] = is_rest & pm & keep & (dxi > 0)
        out['premove_away'] = is_rest & pm & keep & (dxi < 0)
    else:
        out['premove'] = is_rest & pm & keep
    out['rest'] = is_rest & ~pm & keep
    out['move'] = is_move & keep
    return out


def cv_lag_signed(ccfs, lags_s, k=5, sign=1.0, rng=None):
    """:func:`cv_lag` with an explicit correlation sign.

    ``cv_lag`` peak-finds with ``argmax``, which is wrong for a condition whose
    correlation is negative (the cue: flexion hyperpolarizes a flexor). Passing
    ``sign=-1`` finds the trough instead, so the held-out score is evaluated at
    the extremum that actually exists rather than at a meaningless maximum.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    trial_ids = np.array(list(ccfs.keys()))
    mat = np.stack([ccfs[tn] for tn in trial_ids])
    n = len(trial_ids)
    if n < k:
        k = max(2, n)
    folds = np.array_split(rng.permutation(n), k)
    taus, held = [], []
    for fold in folds:
        train = np.ones(n, bool)
        train[fold] = False
        if not train.any():
            continue
        train_mean = np.nanmean(mat[train], axis=0)
        i_star = int(np.nanargmax(sign * train_mean))
        taus.append(lags_s[i_star])
        held.append(np.nanmean(mat[fold][:, i_star]))
    return np.array(taus), np.array(held), np.nanmean(mat, axis=0)


def ccf_by_condition(records, conditions=None, rate_col='rate', signal_col='x',
                     lag_window_s=0.2, min_run_factor=2.0, n_null=200,
                     k_folds=5, rng=None, cue_post_s=0.2, premove_s=0.5,
                     split_premove=False):
    """Cross-correlation of rate against position, computed per condition.

    Positive lag means **rate leads** position (the convention of
    :func:`trial_ccf`). The peak is located on ``sign * ccf`` where ``sign`` comes
    from the condition's own mean correlation — necessary because the cue's
    correlation is negative while the voluntary conditions are positive, so a
    plain ``argmax`` would report a meaningless lag for the cue.

    The null is a **circular rotation** of the rate within each run by at least
    the lag window. That destroys the temporal relation while preserving both
    signals' autocorrelation, which a trial-shuffle does not — and autocorrelation
    is what sets the width of the null here, not sample count.

    Run accounting is returned (``n_runs``, ``n_too_short``) rather than left
    implicit: a condition whose epochs are shorter than ``min_run_factor`` lag
    windows contributes nothing, and that must be visible instead of appearing as
    a quietly missing row.

    Returns ``{condition: dict}`` with ``lags_s``, ``ccf`` (run-averaged),
    ``null_lo`` / ``null_hi`` (5-95%), ``lag_peak``, ``r_peak``, ``sign``,
    ``lag_cv`` / ``r_cv`` (K-fold: peak picked on training trials, scored on
    held-out ones), ``n_runs``, ``n_too_short``, ``n_trials``.
    """
    if conditions is None:
        conditions = condition_masks(records, cue_post_s, premove_s,
                                     split_premove=split_premove)
    rng = np.random.default_rng(0) if rng is None else rng
    groups = list(records.groupby('trial', sort=True))
    if not groups:
        return {}
    dt = float(np.median([np.median(np.diff(g['t'])) for _, g in groups]))
    positions = np.arange(len(records))

    out = {}
    for name, mask in conditions.items():
        # Per-condition lag window. The cue and pre-movement epochs are ~0.5 s by
        # construction, so a window sized for the long rest/move epochs rejects
        # every one of them - which is how the calibration conditions vanished
        # from the first run of this analysis without saying so.
        lw = (lag_window_s.get(name, 0.2) if isinstance(lag_window_s, dict)
              else lag_window_s)
        half = max(int(round(lw / dt)), 2)
        lags_samp = np.arange(-half, half + 1)
        lags_s = lags_samp * dt
        min_n = int(round(min_run_factor * (2 * half + 1)))
        m = np.asarray(mask, dtype=bool)
        runs, per_trial, n_short = [], {}, 0
        for tn, g in groups:
            idx = positions[records.index.get_indexer(g.index)]
            gm = m[idx]
            if not gm.any():
                continue
            gg = g.reset_index(drop=True)
            run_ccfs = []
            for run in _mask_runs(gg, gm):
                if len(run) < min_n:
                    n_short += 1
                    continue
                r = run[rate_col].to_numpy(dtype=float)
                x = run[signal_col].to_numpy(dtype=float)
                if not (np.std(r) > 0 and np.std(x) > 0):
                    continue
                runs.append((r, x))
                run_ccfs.append(trial_ccf(r, x, dt, lags_samp))
            if run_ccfs:
                per_trial[tn] = np.nanmean(np.stack(run_ccfs), axis=0)
        res = {'lags_s': lags_s, 'n_runs': len(runs), 'n_too_short': n_short,
               'n_trials': len(per_trial), 'ccf': None, 'dt': dt,
               'lag_window_s': lw, 'min_run_frames': min_n}
        if not runs:
            out[name] = res
            continue
        obs = np.nanmean(np.stack([trial_ccf(r, x, dt, lags_samp)
                                   for r, x in runs]), axis=0)
        res['ccf'] = obs
        # How much the probe actually moved, and how much the rate did. Needed to
        # read the correlation honestly: during rest and the run-up to a movement
        # the probe is nearly still, so r is small however well-timed the peak is.
        # A late peak with a low r on a stationary probe is not evidence of a lag.
        res['x_ptp_median'] = float(np.median([np.ptp(x) for _, x in runs]))
        res['x_sd_median'] = float(np.median([np.std(x, ddof=1) for _, x in runs]))
        res['rate_sd_median'] = float(np.median([np.std(r, ddof=1) for r, _ in runs]))
        sign = 1.0 if np.nansum(obs) >= 0 else -1.0   # from the data, not assumed
        i_pk = int(np.nanargmax(sign * obs))
        res.update(sign=sign, lag_peak=float(lags_s[i_pk]),
                   r_peak=float(obs[i_pk]))
        if n_null:
            nulls = np.full((n_null, len(lags_samp)), np.nan)
            for it in range(n_null):
                acc = []
                for r, x in runs:
                    lo = half + 1
                    hi = max(len(r) - half - 1, lo + 1)
                    acc.append(trial_ccf(np.roll(r, int(rng.integers(lo, hi))),
                                         x, dt, lags_samp))
                nulls[it] = np.nanmean(np.stack(acc), axis=0)
            res['null_lo'] = np.nanpercentile(nulls, 5, axis=0)
            res['null_hi'] = np.nanpercentile(nulls, 95, axis=0)
        if len(per_trial) >= 3:
            taus, rs, _ = cv_lag_signed(per_trial, lags_s, k=k_folds, sign=sign,
                                        rng=rng)
            res.update(lag_cv=float(np.nanmedian(taus)),
                       r_cv=float(np.nanmean(rs)))
        out[name] = res
    return out


def event_latency(records, events, pre_s=0.3, post_s=0.5, frac=0.5,
                  rate_col='rate', x_col='x', min_d_rate=2.0, min_d_x=2.0,
                  mode='frac', n_sd=3.0, base_from_s=None, base_to_s=0.0):
    """Per-event latency between a rate change and a position change, in ms.

    Independent of any cross-correlation: within a window around each event, take
    each signal's excursion from its pre-event baseline and find when it first
    reaches ``frac`` of that excursion. The latency is ``t_position - t_rate``, so
    **positive means the rate moved first**.

    Worth having alongside the CCF because it needs no lag grid, gives one number
    per event rather than a curve, and fails *visibly* — an event with no
    excursion to time is dropped, whereas a CCF always returns a peak somewhere.

    ``events`` is a DataFrame with ``trial`` and ``t0``. ``min_d_rate`` /
    ``min_d_x`` reject events too small to time reliably.
    """
    rows = []
    by_trial = {tn: g for tn, g in records.groupby('trial', sort=False)}
    for _, ev in events.iterrows():
        g = by_trial.get(ev['trial'])
        if g is None:
            continue
        t = g['t'].to_numpy(dtype=float)
        t0 = float(ev['t0'])
        w = (t >= t0 - pre_s) & (t <= t0 + post_s)
        # Baseline window, separate from the display window. It must not overlap
        # the change being timed: for a movement onset the preceding ~0.5 s is the
        # pre-movement rate ramp itself, so a baseline of [t0-pre_s, t0) is
        # measured on the very signal change whose onset is being sought, and the
        # rate's departure is mis-timed. Pass base_from_s/base_to_s to place the
        # baseline before the ramp.
        bf = pre_s if base_from_s is None else base_from_s
        base = (t >= t0 - bf) & (t < t0 - base_to_s)
        if w.sum() < 5 or base.sum() < 3:
            continue
        rec = {'trial': int(ev['trial']), 't0': t0}
        ok = True
        for col, key, min_d in ((rate_col, 'rate', min_d_rate),
                                (x_col, 'x', min_d_x)):
            y = g[col].to_numpy(dtype=float)
            yb, yw = y[base], y[w]
            # A window with no finite sample cannot be timed, so drop the event
            # the same way an under-amplitude one is dropped. Without this,
            # nanargmax raises "All-NaN slice encountered" and takes down the
            # whole batch -- and the cause is usually mundane: a raw channel that
            # exists but is unfilled on some trials.
            if not np.isfinite(yb).any() or not np.isfinite(yw).any():
                ok = False
                break
            b = float(np.nanmean(yb))
            seg, tseg = yw - b, t[w]
            i_ext = int(np.nanargmax(np.abs(seg)))
            amp = seg[i_ext]
            if not np.isfinite(amp) or abs(amp) < min_d:
                ok = False
                break
            after = tseg >= t0
            if mode == 'onset':
                # Departure from baseline, not a fraction of the peak. With
                # mismatched shapes the two differ badly: an adapting rate
                # transient reaches half of ITS peak long before a ramped step
                # reaches half of ITS plateau, so a frac-based latency reports the
                # rate as first even when it demonstrably started later. Onset
                # order needs a baseline-departure threshold.
                nse = float(np.nanstd(y[base], ddof=1)) if base.sum() > 2 else 0.0
                thr = max(n_sd * nse, 1e-12)
                cross = np.where(after & (np.sign(amp) * seg >= thr))[0]
            else:
                cross = np.where(after & (np.sign(amp) * seg >= frac * abs(amp)))[0]
            if not len(cross):
                ok = False
                break
            rec[f't_{key}'] = float(tseg[cross[0]])
            rec[f'amp_{key}'] = float(amp)
        if not ok:
            continue
        rec['latency_ms'] = (rec['t_x'] - rec['t_rate']) * 1e3
        rows.append(rec)
    cols = ['trial', 't0', 't_rate', 't_x', 'amp_rate', 'amp_x', 'latency_ms']
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


def cue_events(records):
    """One row per trial: ``trial`` and ``t0`` = cue onset, for ``event_latency``."""
    rows = [{'trial': tn, 't0': float(g['cue_t0'].iloc[0])}
            for tn, g in records.groupby('trial', sort=True)
            if 'cue_t0' in g.columns and np.isfinite(g['cue_t0'].iloc[0])]
    return pd.DataFrame(rows, columns=['trial', 't0'])


def movement_onset_events(records, min_move_ptp=8.0, init_s=0.15):
    """One row per movement onset: ``trial``, ``t0``, ``dx_init``.

    ``t0`` is the first frame of the movement stretch, using the same definition
    as :func:`add_movement_timing` (a maximal non-REST run clearing
    ``min_move_ptp``), so these latencies line up with the pre-movement masks.
    """
    rows = []
    for tn, g in records.groupby('trial', sort=True):
        g = g.reset_index(drop=True)
        t = g['t'].to_numpy(dtype=float)
        x = g['x'].to_numpy(dtype=float)
        st = g['state'].to_numpy()
        if len(t) < 3:
            continue
        dt = float(np.median(np.diff(t)))
        n_init = max(int(round(init_s / dt)), 1)
        for a, b, v in _rle_states(st != kin.STATE_REST):
            if not v:
                continue
            if min_move_ptp and np.ptp(x[a:b]) < min_move_ptp:
                continue
            rows.append({'trial': tn, 't0': float(t[a]),
                         'dx_init': float(x[min(a + n_init, b - 1)] - x[a])})
    return pd.DataFrame(rows, columns=['trial', 't0', 'dx_init'])
