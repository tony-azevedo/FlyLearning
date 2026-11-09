"""Figure 7, MN spike-raster version: example traces per cell type and condition.

One row per motor neuron (A2, A4, F1), one column per condition -- steady hold in
the low-force target, steady hold in the high-force target, movement into the
high-force target on an as_off trial -- each panel a probe-position trace over a
spike-rate trace. Below them, one stable-period (REST) position-vs-rate panel
per neuron, the same crunch as the A2 panels D/E.

Gathering is separated from drawing, as in ``figure7_analysis``: ``load_cell``
builds (or reads back from the parquet cache) the per-frame records and a
per-trial metadata table, and everything else runs off those two frames.
"""
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mapd import bout_analysis as ba
from figure7_analysis import build_cell_records

kin = ba.kin

CACHE_DIR = Path('./data_cache/figure7_raster')
FIG_DIR = Path('./Figure7_raster')

RATE_YLIM = (0, 120)        # Hz, every rate axis in this figure
MIN_STABLE_S = 1.0          # stable-period (REST) bout floor, as in panels D/E
PREMOVE_S = 0.5             # run-up to a movement: excluded from the crunch
CUE_POST_S = 0.2            # cue off-response: excluded from the crunch

# High-force target = larger x, because x = -(probe_position - probeZero) is
# positive toward more force; the Table calls it pyasState 'hi'.
CONDITIONS = ('lo_hold', 'hi_hold', 'hi_move')
CONDITION_TITLES = {'lo_hold': 'steady hold, low-force target',
                    'hi_hold': 'steady hold, high-force target',
                    'hi_move': 'movement into high-force target'}


# ---------------------------------------------------------------------------
# Gathering
# ---------------------------------------------------------------------------

def trial_meta(T):
    """One row per trial: outcome, target state and the target band in the x frame."""
    rows = []
    for tn, tr in T.df['Trial'].items():
        if tr is None:
            continue
        try:
            lo, hi = ba.target_bounds(tr)
        except AttributeError:          # trial predates the pyas target metadata
            lo, hi = np.nan, np.nan
        state = tr.pyasState
        rows.append({'trial': tn,
                     'excluded': bool(getattr(tr, 'excluded', False)),
                     'as_outcome': str(tr.as_outcome),
                     'pyasState': 'no_state' if pd.isna(state) else str(state),
                     'tgt_lo': float(lo), 'tgt_hi': float(hi),
                     'probeZero': float(tr.probeZero)})
    return pd.DataFrame(rows).set_index('trial')


def _cache_paths(cell_id, cache_dir):
    d = Path(cache_dir)
    return d / f'{cell_id}_records.parquet', d / f'{cell_id}_trials.parquet'


def load_cell(sinq, cell_id, cache_dir=CACHE_DIR, recompute=False, verbose=True):
    """``(records, meta)`` for one cell, from the parquet cache when it exists.

    Builds with ``keep_silent=True`` (silent trials are real trials -- see the
    zero-spike bug) and no Vm, which this figure does not draw. The Table is
    kept just long enough to read the per-trial metadata, then dropped.
    """
    rec_p, meta_p = _cache_paths(cell_id, cache_dir)
    if not recompute and rec_p.exists() and meta_p.exists():
        if verbose:
            print(f'{cell_id}: reading cached records ({rec_p})')
        return pd.read_parquet(rec_p), pd.read_parquet(meta_p)

    if verbose:
        print(f'{cell_id}: building records (slow)', flush=True)
    built = build_cell_records(sinq, cell_id, with_vm=False, pad_spikes=True,
                               trim_edges_s=0.0, keep_silent=True,
                               keep_table=True, drop_table=False,
                               verbose=verbose)
    try:
        meta = trial_meta(built['table'])
    finally:
        sinq.drop_tables(index=[cell_id])
    rec = built['records']
    if rec is None:
        raise RuntimeError(f'{cell_id}: build_cell_records returned no records')
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    rec.to_parquet(rec_p)
    meta.to_parquet(meta_p)
    if verbose:
        print(f'   cached to {rec_p.resolve()}')
    return rec, meta


def cell_protocol(sinq, cell_id):
    """The protocol of the cell's Table, from its filename in the Sinq."""
    return Path(str(sinq.df.at[cell_id, 'parquet'])).name.split(f'_{cell_id}')[0]


def _trial_file(cell_id, protocol, trial):
    from mapd.paths import resolve_trial_path
    from mapd.trial import Trial
    return Trial(str(resolve_trial_path(cell_id, int(trial), protocol=protocol)))


def load_spike_times(sinq, cell_id, trials, cache_dir=CACHE_DIR, recompute=False,
                     verbose=True):
    """Exact spike times, one row per spike: columns ``trial``, ``t`` (s, the
    trial's own clock -- the same clock as the records' ``t``).

    Read straight from the trial files (no Table needed) and cached as parquet.
    The protocol comes from the Sinq's Table filename, since a cell directory
    can hold several (210915_F1_C1 also has CurrentStepControl trials).
    Trials whose detection never ran contribute no rows, like silent ones; the
    records already dropped them, so the raster cannot show a gap they don't.
    """
    from mapd import ephys

    p = Path(cache_dir) / f'{cell_id}_spikes.parquet'
    if not recompute and p.exists():
        return pd.read_parquet(p)
    protocol = cell_protocol(sinq, cell_id)
    if verbose:
        print(f'{cell_id}: reading spike times from {len(trials)} {protocol} trial files',
              flush=True)
    parts = []
    for tn in trials:
        st = ephys.trial_spike_times(_trial_file(cell_id, protocol, tn),
                                     allow_empty=True)
        if st is not None and len(st):
            parts.append(pd.DataFrame({'trial': int(tn), 't': st}))
    spikes = (pd.concat(parts, ignore_index=True) if parts
              else pd.DataFrame({'trial': pd.Series(dtype=int),
                                 't': pd.Series(dtype=float)}))
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    spikes.to_parquet(p)
    if verbose:
        print(f'   {len(spikes):,} spikes cached to {p.resolve()}')
    return spikes


VM_FS = 5000.0              # Hz, membrane potential as drawn


@lru_cache(maxsize=32)
def _trial_vm(cell_id, protocol, trial, channel, fs_out):
    from mapd import ephys
    out = ephys._trial_voltage(_trial_file(cell_id, protocol, trial), channel)
    if out is None:
        return None
    from scipy.signal import resample_poly
    v, fs, t0 = out
    ratio = Fraction(fs_out / fs).limit_denominator(1000)
    v = resample_poly(v, ratio.numerator, ratio.denominator)
    return t0 + np.arange(v.size) / fs_out, v


def trial_vm(cell_id, protocol, trial, t0, t1, channel='voltage_1', fs_out=VM_FS):
    """``(t, v)`` of the raw membrane potential (mV, spikes in) over [t0, t1],
    resampled to ``fs_out`` with an anti-aliasing filter (``resample_poly``).

    Read from the trial file on demand -- the full-rate voltage is far too big
    to cache for every trial -- with the last few trials kept in memory.
    """
    full = _trial_vm(cell_id, protocol, int(trial), channel, float(fs_out))
    if full is None:
        return None
    t, v = full
    m = (t >= t0) & (t <= t1)
    return t[m], v[m]


def restrict_trials(rec, meta, trial_max=None):
    """Keep trials up to ``trial_max`` (e.g. a recording that degrades later)."""
    if trial_max is None:
        return rec, meta
    return (rec[rec['trial'] <= trial_max].reset_index(drop=True),
            meta[meta.index <= trial_max])


# ---------------------------------------------------------------------------
# Stable periods: the p90 v_rest rule, applied per cell
# ---------------------------------------------------------------------------
# The A4 notebook re-classifies its cells at v_rest = p90 of the cell's own
# non-MOVE RMS speed (A4-VREST-1/2), because a noisy position channel otherwise
# turns rest into DRIFT. The rule never lowers v_rest below kin's default, so a
# clean cell is untouched; applying it to every row here keeps the bottom-row
# crunch comparable across neurons.

def speed_floor(rec, smooth_window=0.05, state_col='state'):
    """p50 / p90 of the rolling-RMS speed over non-MOVE frames, px/s."""
    rms_quiet = []
    for _, g in rec.groupby('trial', sort=True):
        t = g['t'].to_numpy(dtype=float)
        x = g['x'].to_numpy(dtype=float)
        if len(t) < 4 or not np.isfinite(x).all():
            continue
        n_sm = max(1, int(round(smooth_window / float(np.median(np.diff(t))))))
        rms = kin._rolling_rms(kin.velocity(t, x), n_sm)
        rms_quiet.append(rms[g[state_col].to_numpy() != kin.STATE_MOVE])
    rms_quiet = np.concatenate(rms_quiet)
    return {'quiet_p50': float(np.median(rms_quiet)),
            'quiet_p90': float(np.percentile(rms_quiet, 90))}


def apply_p90_v_rest(rec):
    """Re-classify at the cell's p90 v_rest. Returns ``(rec, v_rest)``.

    Measured off ``state_default`` when present, so re-running does not ratchet
    the threshold. Movement timing is recomputed, since v_rest is also the
    Schmitt exit threshold and so moves the MOVE edges.
    """
    rec = rec.reset_index(drop=True)
    if 'state_default' not in rec.columns:
        rec = rec.assign(state_default=rec['state'].to_numpy())
    v_rest = float(ba.noise_scaled_v_rest(speed_floor(rec, state_col='state_default')))
    out = rec['state_default'].to_numpy().copy()
    if v_rest > kin.V_REST:
        for _, g in rec.groupby('trial', sort=False):
            t = g['t'].to_numpy(dtype=float)
            if len(t) < 4:
                continue
            _, st, _, _ = kin.detect_movement_bouts(t, g['x'].to_numpy(dtype=float),
                                                    v_rest=v_rest)
            out[g.index.to_numpy()] = st
    rec = rec.assign(state=out)
    rec = rec.drop(columns=[c for c in ('t_to_move', 't_since_move', 'next_dx_init')
                            if c in rec.columns])
    return ba.add_movement_timing(rec), v_rest


def imposed_mask(rec, cue_post_s=CUE_POST_S):
    """Frames whose rate is imposed on the cell: the cue (+ off-response) and
    the pre-cue current step. Neither reports a position/rate relation."""
    return ba.cue_mask(rec, post_s=cue_post_s) | ba.current_step_mask(rec)


# ---------------------------------------------------------------------------
# Candidate windows, one finder per condition
# ---------------------------------------------------------------------------

def _dt(g):
    return float(np.median(np.diff(g['t'].to_numpy(dtype=float))))


def find_holds(rec, meta, side, min_s=3.0, margin_px=5.0, min_in_target=0.95,
               exclude_outcomes=('rest', 'info')):
    """Stable periods inside the target, on trials whose target is ``side``.

    A candidate is a REST run of at least ``min_s`` (imposed frames removed)
    during which the probe sits inside the trial's band, widened by
    ``margin_px``, for at least ``min_in_target`` of the frames. Ranked longest
    and stillest first. Columns t0/t1 are the run's own edges.
    """
    imposed = imposed_mask(rec)
    trials = meta.index[(meta['pyasState'] == side)
                        & ~meta['as_outcome'].isin(exclude_outcomes)]
    rows = []
    sub = rec[rec['trial'].isin(trials)]
    for tn, g in sub.groupby('trial', sort=True):
        lo, hi = meta.at[tn, 'tgt_lo'], meta.at[tn, 'tgt_hi']
        if not (np.isfinite(lo) and np.isfinite(hi)):
            continue
        dt = _dt(g)
        ok = ((g['state'].to_numpy() == kin.STATE_REST)
              & ~imposed[g.index.to_numpy()])
        for run in ba._mask_runs(g, ok):
            dur = len(run) * dt
            if dur < min_s:
                continue
            x = run['x'].to_numpy(dtype=float)
            inside = float(np.mean((x >= lo - margin_px) & (x <= hi + margin_px)))
            if inside < min_in_target:
                continue
            rows.append({'trial': int(tn), 't0': float(run['t'].iloc[0]),
                         't1': float(run['t'].iloc[-1]), 'dur_s': dur,
                         'x_mean': float(np.nanmean(x)), 'x_ptp': float(np.ptp(x)),
                         'rate_mean': float(run['rate'].mean()),
                         'rate_sd': float(run['rate'].std()),
                         'n_spikes': int(run['n_sp'].sum()),
                         'as_outcome': meta.at[tn, 'as_outcome']})
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(['dur_s', 'x_ptp'], ascending=[False, True])
    return out.reset_index(drop=True)


def find_entries(rec, meta, side='hi', pre_s=2.0, post_s=2.0, min_rise_px=20.0,
                 outcomes=('as_off',)):
    """Movements into the target on ``outcomes`` trials (default as_off).

    The entry is the first frame at which x reaches the band's lower edge from
    below -- for a high-force target that is the force increase that turns the
    stimulus off. Requires MOVE frames in the 0.5 s before entry and a rise of
    at least ``min_rise_px`` over ``pre_s``, so a probe already parked at the
    edge is not counted. Ranked by rise, largest first.
    """
    trials = meta.index[(meta['pyasState'] == side)
                        & meta['as_outcome'].isin(outcomes)]
    rows = []
    for tn, g in rec[rec['trial'].isin(trials)].groupby('trial', sort=True):
        lo = meta.at[tn, 'tgt_lo']
        if not np.isfinite(lo):
            continue
        t = g['t'].to_numpy(dtype=float)
        x = g['x'].to_numpy(dtype=float)
        below = x < lo
        cross = np.where(below[:-1] & ~below[1:])[0]
        if not len(cross):
            continue
        i = int(cross[0]) + 1
        te = t[i]
        pre = (t >= te - pre_s) & (t < te)
        if pre.sum() < 3:
            continue
        rise = float(x[i] - np.nanmin(x[pre]))
        moving = bool((g['state'].to_numpy()[(t >= te - 0.5) & (t <= te)]
                       == kin.STATE_MOVE).any())
        if rise < min_rise_px or not moving:
            continue
        win = (t >= te - pre_s) & (t <= te + post_s)
        rows.append({'trial': int(tn), 't_entry': float(te),
                     't0': float(te - pre_s), 't1': float(te + post_s),
                     'rise_px': rise,
                     'rate_pre': float(g['rate'].to_numpy()[pre].mean()),
                     'rate_peak': float(np.nanmax(g['rate'].to_numpy()[win])),
                     'n_spikes': int(g['n_sp'].to_numpy()[win].sum())})
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values('rise_px', ascending=False)
    return out.reset_index(drop=True)


def find_candidates(rec, meta, hold_kwargs=None, entry_kwargs=None):
    """All three conditions at once: ``{'lo_hold', 'hi_hold', 'hi_move'}``."""
    hk, ek = hold_kwargs or {}, entry_kwargs or {}
    return {'lo_hold': find_holds(rec, meta, 'lo', **hk),
            'hi_hold': find_holds(rec, meta, 'hi', **hk),
            'hi_move': find_entries(rec, meta, 'hi', **ek)}


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def set_style():
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 10,
                         'svg.fonttype': 'none', 'axes.spines.top': False,
                         'axes.spines.right': False})


def window_frames(rec, trial, t0, t1):
    g = rec[rec['trial'] == trial]
    return g[(g['t'] >= t0) & (g['t'] <= t1)]


def draw_example(ax_x, ax_r, rec, meta, trial, t0, t1, color='k',
                 raster=True, t_mark=None, rate_ylim=RATE_YLIM, spikes=None,
                 vm=None):
    """Position over rate for one window, time re-zeroed at ``t0``.

    The trial's target band is shaded on the position axis. ``raster`` draws
    a tick per spike along the top of the rate axis, at the exact times in
    ``spikes`` (from ``load_spike_times``). Without ``spikes`` it falls back to
    one tick per frame holding a spike -- ~5 ms resolution, so a doublet inside
    one frame reads as one tick. ``t_mark`` (absolute trial time) draws a
    dashed line on both axes, e.g. the target entry.

    ``vm = (t, v)`` (from ``trial_vm``) draws the membrane potential on the
    lower axis in place of the rate, with headroom left above it for the ticks.
    """
    w = window_frames(rec, trial, t0, t1)
    tt = w['t'].to_numpy(dtype=float) - t0
    lo, hi = meta.at[trial, 'tgt_lo'], meta.at[trial, 'tgt_hi']
    if np.isfinite(lo) and np.isfinite(hi):
        ax_x.axhspan(lo, hi, color='#9467bd', alpha=0.15, lw=0)
    ax_x.plot(tt, w['x'].to_numpy(dtype=float), color='k', lw=1)
    if vm is not None:
        ax_r.plot(vm[0] - t0, vm[1], color=color, lw=0.4)
        v_lo, v_hi = np.nanmin(vm[1]), np.nanmax(vm[1])
        span = max(v_hi - v_lo, 1.0)
        ax_r.set_ylim(v_lo - 0.05 * span, v_hi + 0.3 * span)
    else:
        ax_r.plot(tt, w['rate'].to_numpy(dtype=float), color=color, lw=1)
        ax_r.set_ylim(*rate_ylim)
    if raster:
        if spikes is not None:
            st = spikes.loc[spikes['trial'] == trial, 't'].to_numpy(dtype=float)
            ticks = st[(st >= t0) & (st <= t1)] - t0
        else:
            ticks = tt[w['n_sp'].to_numpy() > 0]
        # y in axes fraction, so the ticks sit at the top whatever the y range
        ax_r.vlines(ticks, 0.90, 0.98, color=color, lw=0.6,
                    transform=ax_r.get_xaxis_transform())
    for ax in (ax_x, ax_r):
        ax.set_xlim(0, t1 - t0)
        if t_mark is not None and np.isfinite(t_mark):
            ax.axvline(t_mark - t0, color='0.5', ls='--', lw=0.8)
    ax_x.tick_params(labelbottom=False)
    return w


def plot_candidates(rec, meta, cands, cell_id, condition, n=12, ncols=4,
                    max_window_s=6.0, spikes=None, save=True, fig_dir=FIG_DIR,
                    vm_protocol=None):
    """Thumbnails of the top ``n`` candidates, to choose examples by eye.

    Hold windows longer than ``max_window_s`` show their first ``max_window_s``.
    Each thumbnail is titled with its row number in ``cands``, the trial and
    the window, which is what goes into ``EXAMPLES``. ``save`` writes
    ``candidates_{cell_id}_{condition}.png`` to ``fig_dir``. ``vm_protocol``
    (the cell's protocol) draws the membrane potential instead of the rate,
    saved with a ``_vm`` suffix.
    """
    sub = cands.head(n)
    if not len(sub):
        print(f'{cell_id} {condition}: no candidates')
        return None
    nrows = int(np.ceil(len(sub) / ncols))
    fig = plt.figure(figsize=(3.2 * ncols, 2.4 * nrows), layout='constrained')
    gs = fig.add_gridspec(nrows * 2, ncols)
    for k, (i, row) in enumerate(sub.iterrows()):
        r, c = divmod(k, ncols)
        ax_x = fig.add_subplot(gs[2 * r, c])
        ax_r = fig.add_subplot(gs[2 * r + 1, c], sharex=ax_x)
        t1 = min(row['t1'], row['t0'] + max_window_s)
        vm = (trial_vm(cell_id, vm_protocol, row['trial'], row['t0'], t1)
              if vm_protocol else None)
        draw_example(ax_x, ax_r, rec, meta, int(row['trial']), row['t0'], t1,
                     t_mark=row.get('t_entry'), spikes=spikes, vm=vm)
        ax_x.set_title(f"[{i}] tr {int(row['trial'])}  {row['t0']:.2f}-{t1:.2f} s",
                       fontsize=8)
        ax_x.tick_params(labelsize=7)
        ax_r.tick_params(labelsize=7)
    fig.suptitle(f'{cell_id}: {CONDITION_TITLES[condition]} -- top {len(sub)}',
                 fontsize=10)
    if save:
        sfx = '_vm' if vm_protocol else ''
        save_fig(fig, f'candidates_{cell_id}_{condition}{sfx}', fig_dir=fig_dir,
                 exts=('png',))
    return fig


def stable_bouts(rec, meta, min_s=MIN_STABLE_S, premove_s=PREMOVE_S):
    """Frames and per-bout means for the crunch: REST bouts >= ``min_s`` on
    task trials, with the imposed frames and the pre-movement run-up removed
    (the run-up is the motor command for the next movement, not a hold).

    Returns ``(frames, bouts)``; bouts carry the trial's ``pyasState``.
    """
    task = meta.index[~meta['as_outcome'].isin(('rest', 'info'))]
    ok = ((rec['state'].to_numpy() == kin.STATE_REST)
          & ~imposed_mask(rec)
          & ~(rec['t_to_move'].to_numpy() <= premove_s)
          & rec['trial'].isin(task).to_numpy())
    frames, rows = [], []
    for tn, g in rec.assign(_ok=ok).groupby('trial', sort=True):
        dt = _dt(g)
        for run in ba._mask_runs(g, g['_ok'].to_numpy()):
            if len(run) * dt < min_s:
                continue
            frames.append(run[['trial', 't', 'x', 'rate']])
            rows.append({'trial': int(tn), 'dur_s': len(run) * dt,
                         'x': float(run['x'].mean()),
                         'rate': float(run['rate'].mean()),
                         'pyasState': meta.at[tn, 'pyasState']})
    frames = (pd.concat(frames, ignore_index=True) if frames
              else pd.DataFrame(columns=['trial', 't', 'x', 'rate']))
    return frames, pd.DataFrame(rows)


STATE_COLORS = {'lo': '#1f77b4', 'hi': '#d62728', 'no_state': '0.5'}


def draw_crunch(ax, frames, bouts, rate_xlim=RATE_YLIM, max_frames=40000,
                seed=0):
    """Stable-period position vs rate: frames grey, bout means coloured by the
    trial's target (blue low, red high), and the bout-level fit of rate on
    position drawn back on these axes.

    Rate on position, not the reverse: with rate on x, regressing position on
    rate is diluted by the rate estimate's noise (the A2 kernel comparison
    confirmed exactly that), while rate-on-position is kernel-invariant.
    """
    f = frames
    if len(f) > max_frames:
        f = f.sample(max_frames, random_state=seed)
    ax.scatter(f['rate'], f['x'], s=1, color='0.75', alpha=0.3, lw=0,
               rasterized=True)
    stats = {'n_bouts': len(bouts)}
    if len(bouts):
        for st, b in bouts.groupby('pyasState'):
            ax.scatter(b['rate'], b['x'], s=9, color=STATE_COLORS.get(st, '0.5'),
                       edgecolor='white', lw=0.3, zorder=3, label=f'{st} target')
        if len(bouts) > 2:
            slope, icpt = np.polyfit(bouts['x'], bouts['rate'], 1)
            xs = np.linspace(bouts['x'].min(), bouts['x'].max(), 50)
            ax.plot(icpt + slope * xs, xs, color='k', lw=1.2, zorder=4)
            r = float(np.corrcoef(bouts['x'], bouts['rate'])[0, 1])
            stats.update(slope_hz_per_px=float(slope), r=r)
            ax.text(0.98, 0.03, f'r = {r:.2f}\n{slope:.2f} Hz/px\nn = {len(bouts)} bouts',
                    transform=ax.transAxes, ha='right', va='bottom', fontsize=8)
    ax.set_xlim(*rate_xlim)
    ax.set_xlabel('spike rate (Hz)')
    return stats


def make_figure(data, examples, cells, rate_ylim=RATE_YLIM, raster=True,
                pos_ylim=None, figsize=None, vm_cells=()):
    """The full figure.

    ``data[label] = {'cell_id', 'rec', 'meta', 'frames', 'bouts'}`` (plus
    ``'spikes'`` for exact raster ticks) and
    ``examples[(label, condition)] = (trial, t0, t1)``, or
    ``(trial, t0, t1, t_mark)`` for the movement column. ``cells`` is the row
    order, e.g. ``['A2', 'A4', 'F1']``. Missing examples leave the panel blank
    and labelled, so the figure can be built while choices are still open.
    ``pos_ylim`` maps a label to a fixed position range; otherwise each row
    shares one range across its three columns, so the low and high holds of a
    cell are read on the same axis.

    Rows in ``vm_cells`` draw the membrane potential (``trial_vm``, at
    ``VM_FS``) in place of the rate; they need ``data[label]['protocol']``.
    Their Vm axes share one range across the row, like the position axes.
    """
    set_style()
    n = len(cells)
    figsize = figsize or (10, 2.6 * n + 2.8)
    fig = plt.figure(figsize=figsize, layout='constrained')
    ratios = [1.0, 1.0] * n + [2.4]
    gs = fig.add_gridspec(2 * n + 1, 3, height_ratios=ratios)
    axes = {}
    for i, lab in enumerate(cells):
        d = data[lab]
        show_vm = lab in vm_cells
        row_x, row_vm = [], []
        for j, cond in enumerate(CONDITIONS):
            ax_x = fig.add_subplot(gs[2 * i, j])
            ax_r = fig.add_subplot(gs[2 * i + 1, j], sharex=ax_x)
            axes[(lab, cond)] = (ax_x, ax_r)
            ex = examples.get((lab, cond))
            if ex is None:
                ax_x.text(0.5, 0.5, 'choose example', transform=ax_x.transAxes,
                          ha='center', va='center', color='0.5')
                ax_r.set_ylim(*rate_ylim)
            else:
                trial, t0, t1 = ex[:3]
                t_mark = ex[3] if len(ex) > 3 else None
                vm = (trial_vm(d['cell_id'], d['protocol'], trial, t0, t1)
                      if show_vm else None)
                draw_example(ax_x, ax_r, d['rec'], d['meta'], trial, t0, t1,
                             raster=raster, t_mark=t_mark, rate_ylim=rate_ylim,
                             spikes=d.get('spikes'), vm=vm)
                ax_x.set_title(f'tr {trial}', fontsize=8, loc='right')
                row_x.append(ax_x)
                if vm is not None:
                    row_vm.append(ax_r)
            if i == 0:
                ax_x.set_title(CONDITION_TITLES[cond], fontsize=10, loc='left')
            if j == 0:
                ax_x.set_ylabel(f'{lab}\nposition (px)')
                ax_r.set_ylabel('Vm (mV)' if show_vm else 'rate (Hz)')
            else:
                ax_r.tick_params(labelleft=False)
            if i == n - 1:
                ax_r.set_xlabel('time (s)')
        if pos_ylim and lab in pos_ylim:
            lims = pos_ylim[lab]
        elif row_x:
            lims = (min(a.get_ylim()[0] for a in row_x),
                    max(a.get_ylim()[1] for a in row_x))
        else:
            lims = None
        if lims is not None:
            for cond in CONDITIONS:
                axes[(lab, cond)][0].set_ylim(*lims)
        if row_vm:
            vlims = (min(a.get_ylim()[0] for a in row_vm),
                     max(a.get_ylim()[1] for a in row_vm))
            for cond in CONDITIONS:
                axes[(lab, cond)][1].set_ylim(*vlims)
    crunch_stats = {}
    for j, lab in enumerate(cells[:3]):
        ax = fig.add_subplot(gs[2 * n, j])
        crunch_stats[lab] = draw_crunch(ax, data[lab]['frames'], data[lab]['bouts'],
                                        rate_xlim=rate_ylim)
        ax.set_title(f"{lab} ({data[lab]['cell_id']}): stable periods", fontsize=9)
        if j == 0:
            ax.set_ylabel('position (px)')
            ax.legend(frameon=False, fontsize=7, loc='upper left')
    return fig, axes, crunch_stats


def make_crunch_figure(groups, data, rate_xlim=RATE_YLIM, pos_ylim=None,
                       panel_size=(2.6, 2.6)):
    """Stable-period crunch for whole cell types: one row per type, one panel
    per cell, and a last column overlaying every cell's fit line -- the A2
    panels D/E arrangement, for the other neurons.

    ``groups`` maps a row label to its cells, e.g. ``{'A4': [...], 'F1': [...]}``;
    ``data[cell_id] = {'frames', 'bouts'}`` (from ``stable_bouts``). Every
    panel shares one rate axis and, within a row, one position axis, so cells
    are compared directly. Returns ``(fig, stats)`` with one stats row per cell.
    """
    set_style()
    ncols = max(len(c) for c in groups.values()) + 1
    nrows = len(groups)
    fig, axes = plt.subplots(nrows, ncols, squeeze=False, layout='constrained',
                             figsize=(panel_size[0] * ncols, panel_size[1] * nrows))
    rows = []
    for i, (lab, cells) in enumerate(groups.items()):
        ax_all = axes[i, -1]
        for j, cid in enumerate(cells):
            ax = axes[i, j]
            d = data[cid]
            st = draw_crunch(ax, d['frames'], d['bouts'], rate_xlim=rate_xlim)
            ax.set_title(f'{lab} {cid}', fontsize=9)
            rows.append({'type': lab, 'cell': cid, **st})
            b = d['bouts']
            if 'slope_hz_per_px' in st:
                xs = np.linspace(b['x'].min(), b['x'].max(), 50)
                icpt = np.mean(b['rate']) - st['slope_hz_per_px'] * np.mean(b['x'])
                ax_all.plot(icpt + st['slope_hz_per_px'] * xs, xs, lw=1.2,
                            label=f"{cid} (r {st['r']:.2f})")
            ax_all.scatter(b['rate'], b['x'], s=4, alpha=0.4, lw=0)
            if j == 0:
                ax.set_ylabel(f'{lab}\nposition (px)')
                if i == 0:
                    ax.legend(frameon=False, fontsize=7, loc='upper left')
        for ax in axes[i, len(cells):-1]:
            ax.set_visible(False)
        ax_all.set_xlim(*rate_xlim)
        ax_all.set_xlabel('spike rate (Hz)')
        ax_all.set_title(f'{lab}: all cells, bout means + fits', fontsize=9)
        ax_all.legend(frameon=False, fontsize=7, loc='lower right')
        row_axes = [axes[i, j] for j in range(len(cells))] + [ax_all]
        lims = (pos_ylim or {}).get(lab) or (
            min(a.get_ylim()[0] for a in row_axes),
            max(a.get_ylim()[1] for a in row_axes))
        for a in row_axes:
            a.set_ylim(*lims)
    return fig, pd.DataFrame(rows).set_index('cell')


def save_fig(fig, name, fig_dir=FIG_DIR, exts=('svg', 'png')):
    """Save under ``fig_dir`` and print each full path."""
    Path(fig_dir).mkdir(parents=True, exist_ok=True)
    for ext in exts:
        p = (Path(fig_dir) / f'{name}.{ext}').resolve()
        fig.savefig(p, dpi=300 if ext == 'png' else None)
        print(f'saved {p}')
