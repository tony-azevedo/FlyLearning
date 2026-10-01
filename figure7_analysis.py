"""Figure-7 plotting and per-cell orchestration.

Imports analysis primitives from :mod:`mapd.bout_analysis` and renders /
saves the panels used by ``op_cond_Figure7_MN_spike_rate_vs_position.ipynb``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Figure defaults for the panels in this module. mapd.table / mapd.trial set
# Arial 11 at import, so this has to run *after* them to win — which it does,
# since figure7_analysis imports mapd. Text stays as text in SVG and PDF so the
# panels can be edited in Illustrator.
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.size'] = 10

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from mapd import ephys
from mapd import bout_analysis as ba
from mapd.bout_analysis import (
    SIGMA_S, RATE_WINDOWS_S,
    collect_cell_records, boxcar_rate_specs,
    windowed_state_samples, window_sweep, conditional_spread,
    variance_decomposition,
    per_trial_ccfs, permutation_null, cv_lag, signed_v, positive_v,
    force_bin_edges, force_bin_edges_3,
    collect_bouts_with_meta, collect_trial_meta_as_off,
    bout_aligned_rate_segments, bout_aligned_position_segments,
    stim_aligned_segments,
    mean_sem,
)

MIN_BOUTS = 5
PRE_S, POST_S = 1.0, 0.5
STIM_PRE_S, STIM_POST_S = 1.0, 3.0
VEL_LABELS = ('Q1', 'Q2', 'Q3', 'Q4')

STATE_STYLES = (
    (ba.kin.STATE_MOVE,  'MOVE',  '#d62728'),
    (ba.kin.STATE_DRIFT, 'DRIFT', '#ff7f0e'),
    (ba.kin.STATE_REST,  'REST',  '#666666'),
)


def tag_suffix(sinq, cell_id):
    """Return a filename-safe ``_tag1_tag2`` suffix for ``cell_id`` based on
    ``sinq.notes_tags()``. Empty string if ``sinq`` is None or the cell has
    no tags.
    """
    if sinq is None:
        return ''
    try:
        raw = sinq.notes_tags().get(cell_id, '')
    except Exception:
        return ''
    if not raw:
        return ''
    parts = []
    for t in str(raw).split(','):
        t = t.strip().replace(' ', '_')
        t = ''.join(ch for ch in t if ch.isalnum() or ch in '_-')
        if t:
            parts.append(t)
    return ('_' + '_'.join(parts)) if parts else ''


# ---------------------------------------------------------------------------
# Interactive plotly: spike rate vs. probe position by movement state
# ---------------------------------------------------------------------------

def _nan_joined(runs, xk, yk):
    """Concatenate per-run arrays with a NaN between them, so one plotly trace
    draws many disconnected polylines instead of one trace per run."""
    xs, ys = [], []
    for r in runs:
        xs.append(np.append(r[xk], np.nan))
        ys.append(np.append(r[yk], np.nan))
    if not xs:
        return np.zeros(0), np.zeros(0)
    return np.concatenate(xs), np.concatenate(ys)


def _trajectory_runs(sub, rate_col='rate', max_points=120000):
    """Contiguous same-state stretches of one state's samples, in time order.

    ``sub`` is already restricted to one state, so a stretch breaks wherever the
    trial changes or a frame is missing from the sequence. Longest runs are kept
    first if the point budget is exceeded — a truncated set of long trajectories
    is more readable than a complete set of stubs.
    """
    runs = []
    for tn, g in sub.groupby('trial', sort=True):
        g = g.sort_values('t')
        t = g['t'].to_numpy()
        if len(t) < 2:
            continue
        # Break on a time gap rather than on the index, so this holds however
        # the caller indexed the frame. Frames of another state are missing from
        # ``sub``, and a gap is exactly what they leave behind.
        dt = float(np.median(np.diff(t)))
        brk = np.where(np.diff(t) > 1.5 * dt)[0] + 1
        for part in np.split(np.arange(len(g)), brk):
            if len(part) < 2:
                continue
            runs.append({'trial': tn,
                         't': g['t'].to_numpy()[part],
                         'x': g['x'].to_numpy()[part],
                         'rate': g[rate_col].to_numpy()[part]})
    runs.sort(key=lambda r: -len(r['t']))
    kept, budget = [], max_points
    for r in runs:
        if budget <= 0:
            break
        kept.append(r)
        budget -= len(r['t'])
    return kept, len(runs)


def _falling_rate_segments(runs, min_samples=3):
    """Sub-stretches where position rises while rate falls, frame to frame.

    Exploratory only — no smoothing, no threshold beyond a sign test and a
    minimum length, so at 25 ms smoothing plenty of these are noise. It marks
    where to look; it does not claim the events are real.
    """
    segs = []
    for r in runs:
        dx = np.diff(r['x'])
        dr = np.diff(r['rate'])
        m = (dx > 0) & (dr < 0)
        if not m.any():
            continue
        d = np.diff(np.concatenate([[0], m.astype(np.int8), [0]]))
        for s, e in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
            if e - s + 1 >= min_samples:
                segs.append({'trial': r['trial'],
                             't': r['t'][s:e + 1],
                             'x': r['x'][s:e + 1],
                             'rate': r['rate'][s:e + 1]})
    return segs


def plot_rate_vs_position(records, cell_id, max_points_per_state=20000,
                          marker_size=3, opacity=0.4, rate_col='rate',
                          connect=True, highlight_falling=True,
                          max_line_points=120000):
    """Plotly scatter of spike rate (x) vs. probe position (y), one subplot
    per movement state. Hover shows trial + trial time.

    ``connect`` overlays the frame-to-frame trajectory, so the cloud can be read
    as paths rather than as independent points. ``highlight_falling`` picks out
    the stretches where position rises while rate falls. Both are separate,
    legend-toggleable traces; markers are subsampled to
    ``max_points_per_state`` but the lines are drawn from the full sequence
    (longest trajectories first, up to ``max_line_points``), because a
    subsampled path is not a path.
    """
    fig = make_subplots(
        rows=1, cols=3,
        shared_xaxes=True, shared_yaxes=True,
        subplot_titles=[s[1] for s in STATE_STYLES],
        horizontal_spacing=0.04,
    )
    rng = np.random.default_rng(0)
    for col, (state, label, color) in enumerate(STATE_STYLES, start=1):
        full = records[records['state'] == state]
        n = len(full)
        sub = full
        if n > max_points_per_state:
            sub = full.iloc[rng.choice(n, size=max_points_per_state, replace=False)]
        customdata = np.column_stack([sub['trial'].to_numpy(), sub['t'].to_numpy()])
        fig.add_trace(
            go.Scattergl(
                x=sub[rate_col], y=sub['x'],
                mode='markers',
                name=f'{label} (n={n:,})',
                showlegend=False,
                marker=dict(size=marker_size, color=color, opacity=opacity, line=dict(width=0)),
                customdata=customdata,
                hovertemplate=(
                    'rate: %{x:.1f} Hz<br>'
                    'pos: %{y:.1f} um<br>'
                    'trial: %{customdata[0]}<br>'
                    't: %{customdata[1]:.2f} s<extra></extra>'
                ),
            ),
            row=1, col=col,
        )

        if (connect or highlight_falling) and n > 1:
            runs, n_runs = _trajectory_runs(full, rate_col=rate_col,
                                            max_points=max_line_points)
            if connect and runs:
                lx, ly = _nan_joined(runs, 'rate', 'x')
                fig.add_trace(
                    go.Scattergl(
                        x=lx, y=ly, mode='lines',
                        name=f'{label} path ({len(runs)}/{n_runs} runs)',
                        legendgroup='paths', showlegend=(col == 1),
                        line=dict(color=color, width=0.5),
                        opacity=0.35, hoverinfo='skip',
                    ),
                    row=1, col=col,
                )
            if highlight_falling and runs:
                segs = _falling_rate_segments(runs)
                if segs:
                    sx, sy = _nan_joined(segs, 'rate', 'x')
                    fig.add_trace(
                        go.Scattergl(
                            x=sx, y=sy, mode='lines',
                            name=f'{label} pos up / rate down ({len(segs)})',
                            legendgroup='falling', showlegend=(col == 1),
                            line=dict(color='#1f77b4', width=1.2),
                            opacity=0.9, hoverinfo='skip',
                        ),
                        row=1, col=col,
                    )
        fig.update_xaxes(title_text='spike rate (Hz)', row=1, col=col,
                         zeroline=True, zerolinecolor='black', zerolinewidth=0.5)
    fig.update_yaxes(title_text='probe position (um)', row=1, col=1,
                     zeroline=True, zerolinecolor='black', zerolinewidth=0.5)
    fig.update_layout(
        title=f'{cell_id} - smoothed rate (sigma={int(SIGMA_S*1000)} ms) vs. probe position',
        height=460, width=1150,
        margin=dict(l=60, r=20, t=70, b=50),
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0),
    )
    return fig


# ---------------------------------------------------------------------------
# REST: does averaging rate over longer windows tighten the position relation?
# ---------------------------------------------------------------------------

def plot_rest_windows_scatter(windows, cell_id, spread=None, fig_dir='./Figure7',
                              save=True, sinq=None, ncols=3, state_label='REST',
                              value_col='rate'):
    """Position vs. rate, one panel per averaging window, axes shared.

    Both axes are averages over the *same* window, so the panels are like-for-
    like: any narrowing is the measurement getting quieter, not position being
    compared against a slower signal. The horizontal bar on each panel is
    +/-1 sd of the Poisson counting noise at that window — the width the cloud
    would have if the spike train carried no information beyond its mean rate.
    A cloud that shrinks down to the bar and no further has been explained.
    """
    ws = sorted(windows['window_s'].unique())
    if not ws:
        return None
    nrows = int(np.ceil(len(ws) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.4 * nrows),
                             sharex=True, sharey=True, squeeze=False)
    sp = spread.set_index('window_s') if spread is not None else None
    for ax, w in zip(axes.ravel(), ws):
        g = windows[windows['window_s'] == w]
        ax.scatter(g[value_col], g['x_mean'], s=6, alpha=0.35, color='#666666',
                   linewidths=0)
        title = f'W = {w * 1000:.0f} ms   n = {len(g):,}'
        if sp is not None and w in sp.index:
            r = sp.loc[w]
            y_bar = np.nanpercentile(g['x_mean'], 5)
            floor_sd = r.get('sd_allan', r['sd_poisson'])
            ax.errorbar(r['rate_mean'], y_bar, xerr=floor_sd,
                        fmt='o', color='#2ca02c', capsize=3, ms=4, lw=1.5,
                        zorder=5, label='fast-noise floor')
            ax.errorbar(r['rate_mean'], y_bar, xerr=r['sd_within_bin'],
                        fmt='none', ecolor='#1f77b4', capsize=6, lw=1.0,
                        zorder=4, label='observed sd')
            title += (f'\nsd {r["sd_within_bin"]:.1f} vs floor {floor_sd:.1f} Hz'
                      f'   rho = {r["rho"]:+.2f}')
        ax.set_title(title, fontsize=9)
    for ax in axes.ravel()[len(ws):]:
        ax.set_visible(False)
    # The bottom row can be partly empty; give the x axis to whichever panel is
    # actually lowest in each column, or a shared axis leaves it unlabelled.
    for c in range(ncols):
        visible = [r for r in range(nrows) if axes[r, c].get_visible()]
        if not visible:
            continue
        ax = axes[visible[-1], c]
        ax.set_xlabel('spike rate (Hz)')
        ax.tick_params(labelbottom=True)
    for ax in axes[:, 0]:
        if ax.get_visible():
            ax.set_ylabel('probe position (um)')
    axes[0, 0].legend(fontsize=7, loc='upper right')
    fig.suptitle(f'{cell_id} — {state_label}: rate and position averaged over '
                 f'the same window')
    fig.tight_layout()
    if save:
        fig_dir = Path(fig_dir)
        fig_dir.mkdir(exist_ok=True)
        fig.savefig(fig_dir / f'rest_windows_{cell_id}{tag_suffix(sinq, cell_id)}.svg')
    return fig


#: Hex -> the word a caption should use for it. Only the palette entries that
#: ever encode a *direction* need to be here; anything unlisted falls back to the
#: hex string, which is ugly enough to notice and fix.
_COLOR_NAMES = {'#ff7f0e': 'orange', '#1f77b4': 'blue', '#d62728': 'red',
                '#2ca02c': 'green', '#9467bd': 'purple', '#7f7f7f': 'grey',
                '#666666': 'grey', '#444444': 'dark grey'}


def _color_name(c):
    """The word for a colour, for captions that must track the colours used."""
    return _COLOR_NAMES.get(str(c).lower(), str(c))


def _plain_log_ticks(ax, values=None, axis='x'):
    """Label a log axis with plain numbers. ``values`` pins the ticks to the
    windows actually used, which reads better than decade minor ticks when the
    whole range spans less than two decades.
    """
    a = ax.xaxis if axis == 'x' else ax.yaxis
    fmt = mticker.FuncFormatter(lambda v, _: f'{v:g}')
    if values is not None:
        a.set_ticks(np.asarray(values, dtype=float))
        a.set_ticks([], minor=True)
        a.set_major_formatter(fmt)
    else:
        # Under a decade of range a log axis can have no major ticks at all, so
        # the minor ones have to carry the labels.
        a.set_major_formatter(fmt)
        a.set_minor_formatter(fmt)


def plot_rate_vs_position_kernels(records, cell_id, sigmas_s=(0.025, 0.25),
                                  rate_cols=None, state=ba.kin.STATE_REST,
                                  min_duration_s=1.0, max_x_ptp=None,
                                  fig_dir='./Figure7', save=True, sinq=None,
                                  show_paths=True, show_means=True,
                                  max_bouts=400, xlim=None, ylim=None,
                                  fit_lines=None,
                                  value_label='spike rate (Hz)',
                                  from_sigma_s=None, base_col=None,
                                  premove_s=0.5, fname_tag='kernels_rate',
                                  highlight_col=None,
                                  highlight_sign_col='next_dx_init',
                                  highlight_label=None,
                                  highlight_colors=('#ff7f0e', '#1f77b4')):
    """Probe position vs. spike rate at two or more kernel widths, frame by frame.

    The point this makes, which the windowed analysis cannot: during REST the
    probe is nearly still, so each bout is a near-*horizontal* line on this plot
    and the rate estimate's fluctuation is that line's **length**. A 15 s bout
    holding position to within a micron can still smear across 30 Hz of x purely
    from spike-timing jitter passing through a short kernel. Widening the kernel
    should collapse each streak toward its own bout mean without moving the mean.

    Each panel draws the per-bout path (thin line) plus, optionally, a marker at
    the bout's mean rate — so the streak and the point it collapses to are
    visible together. The annotation reports the median within-bout rate sd,
    which is the streak length in numbers.

    ``rate_cols`` names existing columns; otherwise columns are derived for each
    of ``sigmas_s`` via :func:`mapd.bout_analysis.add_smoothed_rate`, reusing the
    exact per-frame counts already in ``records`` — no second pass over the trial
    files. The narrowest sigma reuses the exact ``rate`` column when it matches,
    since at a kernel only a frame or two wide the discretisation to frame bins
    would matter.
    """
    rec = records
    if rate_cols is None:
        # Derive one column per sigma. ``base_col`` decides how: a spike rate is
        # re-smoothed from the exact per-frame counts, while a continuous signal
        # (Vm) is re-smoothed from the column itself, which needs to know the
        # width it already carries — hence ``from_sigma_s``.
        base = base_col or ('rate' if from_sigma_s is None else 'vm')
        rate_cols, labels = [], []
        for s in sigmas_s:
            if abs(s - (from_sigma_s or SIGMA_S)) < 1e-9 and base in rec.columns:
                rate_cols.append(base)            # already at this width
            elif from_sigma_s is None:
                col = f'rate_g{int(round(s * 1000))}ms'
                if col not in rec.columns:
                    rec = ba.add_smoothed_rate(rec, s, out_col=col)
                rate_cols.append(col)
            else:
                col = f'{base}_g{int(round(s * 1000))}ms'
                if col not in rec.columns:
                    rec = ba.add_smoothed_column(rec, base, s,
                                                 from_sigma_s=from_sigma_s,
                                                 out_col=col)
                rate_cols.append(col)
            labels.append(f'sigma = {s * 1000:.0f} ms')
    else:
        rate_cols = list(rate_cols)
        labels = list(rate_cols)

    # Frames in the run-up to a movement. The rate often dips there while the
    # probe is still REST, so those frames add rate variation at an unchanged
    # position — a real effect, but not a position effect. Flagged by timing (see
    # add_movement_timing), highlighted below, and the spread is reported both
    # with and without them so the size of the contamination is visible.
    # ``highlight_col`` generalises this: any boolean column can drive the
    # highlight, split by the sign of ``highlight_sign_col``. It exists because
    # the run-up is not the only stretch worth marking -- the frames just AFTER
    # a rest begins carry the previous movement's decay, which is a different
    # question with the same shape on this plot. Default None keeps the
    # pre-movement behaviour exactly.
    if highlight_col is None:
        if premove_s and 't_to_move' not in rec.columns:
            rec = ba.add_movement_timing(rec)
        premove = (rec['t_to_move'].to_numpy() <= premove_s
                   if premove_s and 't_to_move' in rec.columns
                   else np.zeros(len(rec), dtype=bool))
    else:
        premove = rec[highlight_col].to_numpy(dtype=bool)
    rec = rec.assign(_premove=premove)

    spread = ba.bout_rate_spread(rec, rate_cols, state=state,
                                 min_duration_s=min_duration_s)
    spread_clean = ba.bout_rate_spread(rec[~rec['_premove']], rate_cols,
                                       state=state,
                                       min_duration_s=min_duration_s)
    if max_x_ptp is not None and len(spread):
        spread = spread[spread['x_ptp'] <= max_x_ptp]
        if len(spread_clean):
            spread_clean = spread_clean[spread_clean['x_ptp'] <= max_x_ptp]

    # Bouts to draw, longest first, shared across panels so the comparison is
    # of the same bouts.
    bouts = []
    for tn, g in rec.groupby('trial', sort=True):
        g = g.reset_index(drop=True)
        runs = ([g] if state is None
                else list(ba._mask_runs(g, g['state'].to_numpy() == state)))
        for i_epoch, run in enumerate(runs):
            t = run['t'].to_numpy()
            if len(t) < 3:
                continue
            dt = float(np.median(np.diff(t)))
            if len(t) * dt < min_duration_s:
                continue
            if max_x_ptp is not None and np.ptp(run['x'].to_numpy()) > max_x_ptp:
                continue
            bouts.append((len(t), tn, i_epoch, run))
    bouts.sort(key=lambda b: -b[0])
    n_eligible = len(bouts)
    bouts = bouts[:max_bouts]
    if not bouts:
        return None
    # Annotate the bouts actually drawn, not every eligible one — with max_bouts
    # biting, the two differ and a median over bouts absent from the panel would
    # misdescribe it.
    drawn_keys = {(tn, ep) for _, tn, ep, _ in bouts}
    spread_drawn = spread[
        [(t, e) in drawn_keys
         for t, e in zip(spread['trial'], spread['epoch'])]] if len(spread) else spread

    # Units for the annotation follow the axis label, so a Vm panel does not
    # report millivolts as Hz.
    unit = (value_label.rsplit('(', 1)[-1].rstrip(')').strip()
            if '(' in value_label else '')
    n = len(rate_cols)
    fig, axes = plt.subplots(1, n, figsize=(5.0 * n, 4.6), sharex=True, sharey=True,
                             squeeze=False)
    axes = axes[0]
    n_premove_frames = 0
    for ax, col, label in zip(axes, rate_cols, labels):
        for _, _, _, run in bouts:
            v = run[col].to_numpy(dtype=float)
            x = run['x'].to_numpy()
            if show_paths:
                ax.plot(v, x, '-', lw=0.4, alpha=0.35, color='#666666',
                        solid_capstyle='butt')
            pm = (run['_premove'].to_numpy() if '_premove' in run.columns
                  else np.zeros(len(run), dtype=bool))
            if pm.any():
                # Split the highlight by where the probe is about to go: a rate
                # decrease precedes most relaxations, while about half of the
                # force-increasing movements are preceded by a rate *increase*, so
                # one colour for both would show only their sum.
                dxi = (run[highlight_sign_col].to_numpy()
                       if highlight_sign_col in run.columns
                       else np.full(len(run), np.nan))
                # Draw only the contiguous pre-movement stretches, so the
                # highlight follows the path rather than jumping across gaps.
                d = np.diff(np.concatenate([[0], pm.astype(np.int8), [0]]))
                for a, b in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
                    dx = np.nanmedian(dxi[a:b]) if b > a else np.nan
                    color = (highlight_colors[0] if dx > 0 else
                             highlight_colors[1] if dx < 0 else '#7f7f7f')
                    ax.plot(v[a:b], x[a:b], '-', lw=1.2, alpha=0.9,
                            color=color, solid_capstyle='butt', zorder=4)
                n_premove_frames += int(pm.sum())
            if show_means:
                ax.plot(np.nanmean(v), np.mean(x), 'o', ms=3.5, alpha=0.8,
                        color='#d62728', mew=0)
        s = spread_drawn[spread_drawn['rate_col'] == col]
        title = label
        if len(s):
            title += (f'\nwithin-bout sd: median {s["rate_sd"].median():.2f} {unit}'
                      f'   ptp {s["rate_ptp"].median():.1f} {unit}')
            sc = (spread_clean[spread_clean['rate_col'] == col]
                  if len(spread_clean) else spread_clean)
            if len(sc):
                title += (f'\nexcluding pre-movement: '
                          f'{sc["rate_sd"].median():.2f} {unit}'
                          f'   ptp {sc["rate_ptp"].median():.1f} {unit}')
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(value_label)
    axes[0].set_ylabel('probe position (um)')
    # Regression lines drawn last, on every panel, in the panel's own coordinates
    # (rate on x, position on y). Overlaying several cells' fits on one cell's
    # scatter is how a shared relation is shown to be shared.
    for spec in (fit_lines or []):
        for ax in axes[:len(rate_cols)]:
            lo, hi = ax.get_xlim() if xlim is None else xlim
            xs = np.linspace(lo, hi, 50)
            ax.plot(xs, spec['intercept'] + spec['slope'] * xs, '-',
                    color=spec.get('color', '#d62728'),
                    lw=spec.get('lw', 1.2), alpha=spec.get('alpha', 0.9),
                    zorder=spec.get('zorder', 6), label=spec.get('label'))
    if fit_lines:
        axes[0].legend(fontsize=6.5, loc='lower right', frameon=False)

    # Shared limits so panels from different cells can be compared directly, and
    # so one wild bout cannot autoscale a cell out of the set.
    if xlim is not None:
        axes[0].set_xlim(*xlim)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    state_name = {ba.kin.STATE_REST: 'REST', ba.kin.STATE_DRIFT: 'DRIFT',
                  ba.kin.STATE_MOVE: 'MOVE'}.get(state, 'all states')
    capped = (f' of {n_eligible}' if n_eligible > len(bouts) else '')
    pm_txt = ''
    if premove_s or highlight_col is not None:
        n_pm = n_premove_frames // max(len(rate_cols), 1)
        if highlight_label is not None:
            pm_txt = f'\n{highlight_label} ({n_pm:,} frames)'
        else:
            # Names come from the colours actually in use, so swapping
            # ``highlight_colors`` cannot leave the caption describing the old
            # scheme — a wrong legend on a direction-coded plot is worse than no
            # legend, and this one has caused a misreading before.
            pm_txt = (f'\nwithin {premove_s*1e3:.0f} ms of a movement '
                      f'({n_pm:,} frames): {_color_name(highlight_colors[0])} '
                      f'= about to move toward target, '
                      f'{_color_name(highlight_colors[1])} = away')
    fig.suptitle(f'{cell_id} — {state_name} bouts >= {min_duration_s:g} s '
                 f'({len(bouts)}{capped}, longest first); '
                 f'grey = path, red = bout mean{pm_txt}', fontsize=10)
    fig.tight_layout()
    if save:
        fig_dir = Path(fig_dir)
        fig_dir.mkdir(exist_ok=True)
        fig.savefig(fig_dir / f'{fname_tag}_vs_position_{cell_id}'
                              f'{tag_suffix(sinq, cell_id)}.svg')
    return fig, spread


def plot_bout_rate_spread(spread, cell_id, fig_dir='./Figure7', save=True,
                          sinq=None, unit='Hz', what='rate',
                          fname_tag='kernels_bout_spread'):
    """Distribution of within-bout rate sd, one histogram per kernel.

    The companion number to :func:`plot_rate_vs_position_kernels`: if the long
    kernel is doing its job, its distribution sits well left of the short one
    while the *bout means* stay put (right panel), which is what distinguishes
    "removed timing jitter" from "smoothed away real signal".
    """
    if not len(spread):
        return None
    cols = list(dict.fromkeys(spread['rate_col']))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    colors = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd']
    for c, color in zip(cols, colors):
        s = spread[spread['rate_col'] == c]
        axes[0].hist(s['rate_sd'], bins=40, histtype='step', lw=1.5,
                     color=color, label=f'{c}  (median {s["rate_sd"].median():.2f})')
        axes[1].hist(s['rate_mean'], bins=40, histtype='step', lw=1.5,
                     color=color, label=f'{c}  (median {s["rate_mean"].median():.1f})')
    axes[0].set_xlabel(f'within-bout {what} sd ({unit})')
    axes[0].set_ylabel('bouts')
    axes[0].set_title('streak length — should shrink', fontsize=10)
    axes[0].legend(fontsize=7)
    axes[1].set_xlabel(f'bout mean {what} ({unit})')
    axes[1].set_title('bout means — should NOT move', fontsize=10)
    axes[1].legend(fontsize=7)
    fig.suptitle(f'{cell_id} — within-bout {what} scatter by kernel width')
    fig.tight_layout()
    if save:
        fig_dir = Path(fig_dir)
        fig_dir.mkdir(exist_ok=True)
        fig.savefig(fig_dir / f'{fname_tag}_{cell_id}'
                              f'{tag_suffix(sinq, cell_id)}.svg')
    return fig


def plot_bout_signal_correlation(corr, cell_id, fig_dir='./Figure7', save=True,
                                 sinq=None, bins=40):
    """Distribution of within-bout rho, one histogram per pair/kernel.

    The question this answers: while the probe sits still, does the firing rate
    fluctuate *with* the membrane potential? One number per bout, so no
    between-bout or between-trial offset can contribute — those are what make a
    pooled correlation untrustworthy for Vm.

    A distribution centred well away from 0 means the coupling is present bout by
    bout. Centred *on* 0 means the two fluctuate independently at that timescale,
    whatever a pooled correlation says.
    """
    if not len(corr):
        return None
    labels = list(dict.fromkeys(corr['label']))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.8))
    colors = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#8c564b']
    for lab, color in zip(labels, colors):
        c = corr[corr['label'] == lab]
        med = c['rho'].median()
        frac_pos = float((c['rho'] > 0).mean())
        axes[0].hist(c['rho'], bins=np.linspace(-1, 1, bins + 1), histtype='step',
                     lw=1.6, color=color,
                     label=f'{lab}\n  median {med:+.2f}, {frac_pos:.0%} > 0, n={len(c)}')
        axes[1].hist(c['slope'], bins=bins, histtype='step', lw=1.6, color=color,
                     label=f'{lab}  median {c["slope"].median():+.2f}')
    axes[0].axvline(0, color='k', lw=0.8)
    axes[0].set_xlabel('within-bout rho')
    axes[0].set_ylabel('bouts')
    axes[0].set_title('do they fluctuate together within a bout?', fontsize=10)
    axes[0].legend(fontsize=6.5)
    axes[1].axvline(0, color='k', lw=0.8)
    axes[1].set_xlabel('within-bout slope (Hz per mV)')
    axes[1].set_title('gain', fontsize=10)
    axes[1].legend(fontsize=7)
    fig.suptitle(f'{cell_id} — within-bout coupling, one point per bout')
    fig.tight_layout()
    if save:
        fig_dir = Path(fig_dir)
        fig_dir.mkdir(exist_ok=True)
        fig.savefig(fig_dir / f'bout_coupling_{cell_id}{tag_suffix(sinq, cell_id)}.svg')
    return fig


def plot_rate_vs_dvm(records, cell_id, pairs=None, trials=None, center='median',
                     by_state=True, n_bins=15, fig_dir='./Figure7', save=True,
                     sinq=None, marker_size=6, fname_tag='rate_vs_dvm'):
    """Firing rate against *change* in Vm, one panel per kernel width.

    ``dVm`` is Vm minus its own per-trial ``center`` (median by default). Vm has
    no meaningful absolute zero across a session — access resistance and cell
    health wander by tens of mV — so only the deviation within a trial is
    comparable. Centring per trial is what lets several trials share a panel
    without the drift smearing them apart.

    Each pair must carry the **same** kernel on both signals, or the relation is
    limited by the wider one and the slope means something different per panel.

    ``pairs`` is ``[(vm_col, rate_col, label), ...]``; the default uses
    ``('vm', 'rate')`` for the narrow kernel and ``('vm_g250ms', 'rate_g250ms')``
    for the wide one when those columns exist (see
    ``bout_analysis.add_smoothed_column`` / ``add_smoothed_rate`` to create them).

    Points are coloured by movement state, and the fit is reported per state as
    well as overall: MOVE samples span a far wider range of both signals, so a
    single fit over everything is dominated by them and says little about the
    rest plateaus.
    """
    rec = records if trials is None else records[
        records['trial'].isin(np.atleast_1d(trials))]
    if not len(rec):
        return None
    if pairs is None:
        pairs = []
        if {'vm', 'rate'} <= set(rec.columns):
            pairs.append(('vm', 'rate', f'sigma = {SIGMA_S*1e3:.0f} ms'))
        for c in ('vm_g250ms', 'vm_exact_g250'):
            if c in rec.columns:
                r = ('rate_g250ms' if 'rate_g250ms' in rec.columns
                     else 'rate_exact_g250' if 'rate_exact_g250' in rec.columns
                     else None)
                if r:
                    pairs.append((c, r, 'sigma = 250 ms'))
                break
    pairs = [p for p in pairs if p[0] in rec.columns and p[1] in rec.columns]
    if not pairs:
        return None

    rec = rec.copy()
    for vm_col, _, _ in pairs:
        cen = rec.groupby('trial')[vm_col].transform(center)
        rec[f'd_{vm_col}'] = rec[vm_col] - cen

    n = len(pairs)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.4), squeeze=False,
                             sharey=True)
    axes = axes[0]
    rows = []
    for ax, (vm_col, rate_col, label) in zip(axes, pairs):
        g = rec.dropna(subset=[f'd_{vm_col}', rate_col])
        if not len(g):
            continue
        groups = (STATE_STYLES if by_state
                  else [(None, 'all', '#666666')])
        for st, sname, color in groups:
            sub = g if st is None else g[g['state'] == st]
            if len(sub) < 5:
                continue
            dv = sub[f'd_{vm_col}'].to_numpy()
            r = sub[rate_col].to_numpy()
            ax.scatter(dv, r, s=marker_size, alpha=0.3, color=color,
                       linewidths=0, label=f'{sname} (n={len(sub):,})')
            if np.std(dv) > 0:
                rows.append({'label': label, 'state': sname, 'n': len(sub),
                             'rho': float(np.corrcoef(dv, r)[0, 1]),
                             'slope': float(np.polyfit(dv, r, 1)[0])})
        # binned mean over everything, so the curve's shape is visible
        dv, r = g[f'd_{vm_col}'].to_numpy(), g[rate_col].to_numpy()
        edges = np.unique(np.quantile(dv, np.linspace(0, 1, n_bins + 1)))
        if len(edges) > 2:
            idx = np.clip(np.searchsorted(edges, dv, 'right') - 1, 0, len(edges) - 2)
            ctr = 0.5 * (edges[:-1] + edges[1:])
            means = np.array([r[idx == b].mean() if (idx == b).any() else np.nan
                              for b in range(len(ctr))])
            ax.plot(ctr, means, 'o-', color='k', ms=3, lw=1.3, zorder=5)
        rho_all = float(np.corrcoef(dv, r)[0, 1]) if np.std(dv) > 0 else np.nan
        slope_all = float(np.polyfit(dv, r, 1)[0]) if np.std(dv) > 0 else np.nan
        rest = [x for x in rows if x['label'] == label and x['state'] == 'REST']
        extra = (f'   REST {rest[0]["rho"]:+.2f} / {rest[0]["slope"]:+.2f} Hz/mV'
                 if rest else '')
        ax.set_title(f'{label}\nall: rho {rho_all:+.2f}, '
                     f'{slope_all:+.2f} Hz/mV{extra}', fontsize=9)
        ax.axvline(0, color='k', lw=0.6)
        ax.set_xlabel(f'dVm (mV, per-trial {center} subtracted)')
    axes[0].set_ylabel('spike rate (Hz)')
    axes[0].legend(fontsize=7, loc='upper left')
    tn = np.unique(rec['trial'])
    which = (f'trial {tn[0]}' if len(tn) == 1 else f'{len(tn)} trials')
    fig.suptitle(f'{cell_id} — firing rate vs. dVm, {which}')
    fig.tight_layout()
    if save:
        fig_dir = Path(fig_dir)
        fig_dir.mkdir(exist_ok=True)
        suffix = f'_tr{tn[0]}' if len(tn) == 1 else ''
        fig.savefig(fig_dir / f'{fname_tag}_{cell_id}'
                              f'{tag_suffix(sinq, cell_id)}{suffix}.svg')
    return fig, pd.DataFrame(rows)


def plot_cue_response(records, cell_id, pre_s=0.3, post_s=1.2, align='onset',
                      cols=('rate', 'vm'), baseline_s=0.2, fig_dir='./Figure7',
                      save=True, sinq=None, windows=None, ramp_s=0.07):
    """Cue-aligned response by commanded displacement, over a generous window.

    Top row is the time course — mean +/- sem per displacement, baseline
    subtracted, with the cue window shaded and its ramps marked. This is what says
    how *long* the response lasts and whether the piezo's return ramp produces a
    separate off-response, neither of which a window clipped to the 300 ms cue
    could show.

    Bottom row is the window sweep: the regression of the response on displacement
    within each window. Where that slope is non-zero the window contains a
    displacement-dependent response; where it flips sign, the response has
    reversed — which is the signature of an off-response.

    Curves are split by displacement and never pooled: the commanded value
    alternates in sign, so a pooled mean cancels most of the effect.

    ``records`` should be built with ``trim_edges_s=0`` and ``pad_trials``, or the
    window will not be covered — the cue sits ~200 ms into the trial.
    """
    t_rel, stacks, meta = ba.cue_aligned_segments(
        records, pre_s=pre_s, post_s=post_s, align=align, cols=cols,
        baseline_s=baseline_s)
    if not len(meta):
        return None
    sweep = ba.cue_response_windows(records, windows=windows, align=align,
                                    cols=cols, baseline_s=baseline_s)

    cols = [c for c in cols if c in stacks]
    n = len(cols)
    fig, axes = plt.subplots(2, n, figsize=(6.0 * n, 7.4), squeeze=False)
    disps = sorted(meta['cue_displacement'].dropna().unique())
    cmap = plt.get_cmap('coolwarm')
    norm = (lambda d: 0.5 if len(disps) < 2 else
            (d - min(disps)) / (max(disps) - min(disps)))
    cue_dur = float(meta['cue_dur'].median())
    units = {'rate': 'Hz', 'vm': 'mV'}

    for j, c in enumerate(cols):
        ax = axes[0, j]
        arr = stacks[c]
        for d in disps:
            sel = (meta['cue_displacement'] == d).to_numpy()
            if sel.sum() < 2:
                continue
            m = np.nanmean(arr[sel], axis=0)
            se = np.nanstd(arr[sel], axis=0, ddof=1) / np.sqrt(sel.sum())
            color = cmap(norm(d))
            ax.plot(t_rel, m, color=color, lw=1.5, label=f'{d:+g} (n={sel.sum()})')
            ax.fill_between(t_rel, m - se, m + se, color=color, alpha=0.2, lw=0)
        # cue extent and its ramps, in the aligned frame
        c0 = 0.0 if align == 'onset' else -cue_dur
        c1 = cue_dur if align == 'onset' else 0.0
        ax.axvspan(c0, c1, color='0.7', alpha=0.25, lw=0, label='cue')
        for edge in (c0, c1):
            ax.axvspan(edge, edge + (ramp_s if edge == c0 else -ramp_s),
                       color='0.4', alpha=0.25, lw=0)
        ax.axhline(0, color='k', lw=0.7)
        ax.axvline(0, color='k', lw=0.7, ls=':')
        ax.set_xlabel(f'time from cue {align} (s)')
        ax.set_ylabel(f'delta {c} ({units.get(c, "")})')
        ax.set_title(f'{c}: cue-aligned, by commanded displacement', fontsize=9)
        ax.legend(fontsize=6.5, ncol=2, title='piezo', title_fontsize=6.5)

        ax = axes[1, j]
        s = sweep[sweep['col'] == c] if len(sweep) else sweep
        # A single trial (or one displacement value) yields no slope, so this panel
        # has nothing to show — say so rather than raising on a missing column.
        if len(s) and s['slope_vs_disp'].notna().any():
            w = (s.drop_duplicates('window')
                  .sort_values('t_start')[['window', 't_start', 't_end',
                                           'slope_vs_disp']].dropna())
            xs = np.arange(len(w))
            ax.bar(xs, w['slope_vs_disp'], color='#8c564b')
            ax.set_xticks(xs)
            ax.set_xticklabels([f'{lab}\n{a:.2f}-{b:.2f}' for lab, a, b
                                in zip(w['window'], w['t_start'], w['t_end'])],
                               fontsize=6.5, rotation=45, ha='right')
            ax.axhline(0, color='k', lw=0.8)
            ax.set_ylabel(f'slope ({units.get(c, "")} per displacement unit)')
            ax.set_title('window sweep: where is the response, and its sign?',
                         fontsize=9)
        else:
            ax.text(0.5, 0.5, 'no slope vs displacement\n'
                              '(needs >1 displacement across trials)',
                    ha='center', va='center', fontsize=8, color='0.4',
                    transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(f'{cell_id} — cue response, {len(meta)} trials '
                 f'(cue {cue_dur*1e3:.0f} ms, aligned to {align})')
    fig.tight_layout()
    if save:
        fig_dir = Path(fig_dir)
        fig_dir.mkdir(exist_ok=True)
        fig.savefig(fig_dir / f'cue_response_{align}_{cell_id}'
                              f'{tag_suffix(sinq, cell_id)}.svg')
    return fig, sweep


def plot_rate_vs_vm_trial(records, cell_id, trial=None, vm_col='vm',
                          rate_col='rate', fig_dir='./Figure7', save=True,
                          sinq=None, marker_size=9, connect_cue=True,
                          center=False, cue_pre_s=0.0, cue_post_s=0.2,
                          cue_window=None):
    """Rate against Vm for one trial, split by bout state, with the cue highlighted.

    The cue is the only moment in a trial where the perturbation is *known*, so it
    calibrates everything else: it says how far Vm and rate move for an imposed
    change in load, against which the spontaneous excursions during a hold can be
    judged as large or small. It is highlighted rather than removed for exactly
    that reason.

    Left panel is the state-split scatter with the cue samples drawn over the top
    and, with ``connect_cue``, joined in time order so the response's trajectory
    (and any hysteresis) is visible. Right panel compares the ranges directly:
    peak-to-peak Vm and rate during the cue versus during REST, over windows of
    the *same duration* — peak-to-peak grows with how long you watch, so the raw
    comparison would flatter the longer group.

    The highlighted window is ``[cue_t0 - cue_pre_s, cue_t1 + cue_post_s]``, and
    ``cue_post_s`` matters: the response outlasts the 300 ms cue. There is a
    distinct off-response on the piezo's return ramp, of the *opposite* sign to the
    on-response, decaying over ~300-400 ms. Clipping the highlight to the cue
    itself (``cue_post_s=0``) therefore hides the half of the trajectory that runs
    the other way. The default 200 ms carries the window to t = -0.3 s, which
    covers the off-response without reaching stimulus onset at t = 0.
    ``cue_window=(t0, t1)`` overrides the lot with explicit trial times.

    ``records`` should carry Vm computed with ``pad_trials`` — the cue sits ~200 ms
    into the trial and an unpadded wide kernel has no support there. ``center``
    subtracts the trial's median Vm, giving dVm.
    """
    rec = records if trial is None else records[records['trial'] == trial]
    rec = rec.dropna(subset=[vm_col, rate_col])
    if not len(rec):
        return None
    if cue_window is not None:
        c0, c1 = float(cue_window[0]), float(cue_window[1])
    elif {'cue_t0', 'cue_t1'} <= set(rec.columns):
        c0 = float(rec['cue_t0'].iloc[0]) - cue_pre_s
        c1 = float(rec['cue_t1'].iloc[0]) + cue_post_s
    elif 'in_cue' in rec.columns and rec['in_cue'].any():
        c0 = float(rec.loc[rec['in_cue'], 't'].min()) - cue_pre_s
        c1 = float(rec.loc[rec['in_cue'], 't'].max()) + cue_post_s
    else:
        raise KeyError("records need 'cue_t0'/'cue_t1' (or 'in_cue') — rebuild "
                       "with per_frame_records(..., mark_cue=True) (the default)")
    # Recomputed from the window rather than reusing the fixed ``in_cue`` flag, so
    # the highlight follows cue_pre_s / cue_post_s.
    rec = rec.assign(in_cue=(rec['t'] >= c0) & (rec['t'] <= c1))
    x_label = 'Vm (mV)'
    v_all = rec[vm_col].to_numpy(dtype=float)
    if center:
        v_all = v_all - np.median(v_all)
        x_label = 'dVm (mV, trial median subtracted)'
    rec = rec.assign(_v=v_all)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6),
                             gridspec_kw={'width_ratios': [2, 1]})
    ax = axes[0]
    rows = []
    for st, sname, color in STATE_STYLES:
        sub = rec[(rec['state'] == st) & (~rec['in_cue'])]
        if not len(sub):
            continue
        ax.scatter(sub['_v'], sub[rate_col], s=marker_size, alpha=0.35,
                   color=color, linewidths=0, label=f'{sname} (n={len(sub):,})')
        if len(sub) > 5 and np.std(sub['_v']) > 0:
            rows.append({'group': sname, 'n': len(sub),
                         'vm_ptp': float(np.ptp(sub['_v'])),
                         'rate_ptp': float(np.ptp(sub[rate_col])),
                         'vm_sd': float(np.std(sub['_v'], ddof=1)),
                         'rate_sd': float(np.std(sub[rate_col], ddof=1)),
                         'rho': float(np.corrcoef(sub['_v'], sub[rate_col])[0, 1]),
                         'slope': float(np.polyfit(sub['_v'], sub[rate_col], 1)[0])})
    cue = rec[rec['in_cue']].sort_values('t')
    if len(cue):
        if connect_cue:
            ax.plot(cue['_v'], cue[rate_col], '-', color='#1f77b4', lw=1.0,
                    alpha=0.8, zorder=4)
        ax.scatter(cue['_v'], cue[rate_col], s=marker_size + 14, alpha=0.95,
                   facecolors='none', edgecolors='#1f77b4', linewidths=1.1,
                   zorder=5, label=f'piezo cue (n={len(cue):,})')
        if len(cue) > 5 and np.std(cue['_v']) > 0:
            rows.append({'group': 'CUE', 'n': len(cue),
                         'vm_ptp': float(np.ptp(cue['_v'])),
                         'rate_ptp': float(np.ptp(cue[rate_col])),
                         'vm_sd': float(np.std(cue['_v'], ddof=1)),
                         'rate_sd': float(np.std(cue[rate_col], ddof=1)),
                         'rho': float(np.corrcoef(cue['_v'], cue[rate_col])[0, 1]),
                         'slope': float(np.polyfit(cue['_v'], cue[rate_col], 1)[0])})
    ax.set_xlabel(x_label)
    ax.set_ylabel('spike rate (Hz)')
    ax.legend(fontsize=7, loc='upper left')

    stats = pd.DataFrame(rows)
    # Duration-matched comparison. Peak-to-peak grows with how long you watch, so
    # the cue's 0.3 s excursion cannot be set against REST measured over the whole
    # trial. Instead REST is cut into windows the same length as the cue, and the
    # cue's excursion is placed in *that* distribution — which is what says whether
    # an imposed load moves this cell more than it moves on its own.
    cue_dur = float(cue['t'].max() - cue['t'].min()) if len(cue) else 0.0
    rest = rec[(rec['state'] == ba.kin.STATE_REST) & (~rec['in_cue'])]
    win_rows = []
    if cue_dur > 0 and len(rest) > 3:
        t = rest['t'].to_numpy()
        dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.0
        # Break on time gaps so a window never straddles a non-REST stretch.
        gaps = np.where(np.diff(t) > 1.5 * dt)[0] + 1 if dt > 0 else []
        for part in np.split(np.arange(len(rest)), gaps):
            if len(part) < 3:
                continue
            tp = t[part]
            n_win = int(np.floor((tp[-1] - tp[0]) / cue_dur))
            for k in range(max(n_win, 0)):
                m = (tp >= tp[0] + k * cue_dur) & (tp < tp[0] + (k + 1) * cue_dur)
                if m.sum() < 3:
                    continue
                sl = part[m]
                win_rows.append({
                    'vm_ptp': float(np.ptp(rest['_v'].to_numpy()[sl])),
                    'rate_ptp': float(np.ptp(rest[rate_col].to_numpy()[sl])),
                })
    wins = pd.DataFrame(win_rows)

    cue_vm_ptp = float(np.ptp(cue['_v'])) if len(cue) else np.nan
    cue_rate_ptp = float(np.ptp(cue[rate_col])) if len(cue) else np.nan
    pct = {}
    for key, val in (('vm_ptp', cue_vm_ptp), ('rate_ptp', cue_rate_ptp)):
        if len(wins) and val == val:
            pct[key] = float((wins[key] < val).mean())

    ax2 = axes[1]
    if len(wins):
        ax2.hist(wins['rate_ptp'], bins=30, color='#d62728', alpha=0.55,
                 label=f'REST {cue_dur*1e3:.0f} ms windows (n={len(wins)})')
        if cue_rate_ptp == cue_rate_ptp:
            ax2.axvline(cue_rate_ptp, color='#1f77b4', lw=2,
                        label=f'cue = {cue_rate_ptp:.0f} Hz'
                              + (f' ({pct["rate_ptp"]:.0%} pct)'
                                 if 'rate_ptp' in pct else ''))
        ax2.set_xlabel(f'rate peak-to-peak in {cue_dur*1e3:.0f} ms (Hz)')
        ax2.set_ylabel('windows')
        ax2.set_title('imposed cue vs spontaneous, same duration', fontsize=9)
        ax2.legend(fontsize=7)
        if 'vm_ptp' in pct:
            ax2.annotate(f'Vm ptp: cue {cue_vm_ptp:.2f} mV, '
                         f'{pct["vm_ptp"]:.0%} pct of REST windows\n'
                         f'(median REST {wins["vm_ptp"].median():.2f} mV)',
                         xy=(0.02, 0.72), xycoords='axes fraction', fontsize=7)

    tn = trial if trial is not None else 'pooled'
    disp = ''
    if len(cue) and 'cue_displacement' in rec.columns:
        disp = f", piezo {rec['cue_displacement'].iloc[0]:+g}"
    fig.suptitle(f'{cell_id} trial {tn} — rate vs Vm by bout state'
                 f' (sigma={SIGMA_S*1e3:.0f} ms{disp}); '
                 f'cue window {c0:+.3f}..{c1:+.3f} s '
                 f'({(c1 - c0) * 1e3:.0f} ms)')
    fig.tight_layout()
    if save:
        fig_dir = Path(fig_dir)
        fig_dir.mkdir(exist_ok=True)
        fig.savefig(fig_dir / f'rate_vs_vm_states_{cell_id}'
                              f'{tag_suffix(sinq, cell_id)}_tr{tn}'
                              f'_cue{(c1 - c0) * 1e3:.0f}ms.svg')
    if len(stats):
        stats = stats.assign(cue_t0=c0, cue_t1=c1, cue_dur=c1 - c0)
    return fig, stats


def plot_trial_records(records, cell_id=None, trial=None, min_duration_s=1.0,
                       premove_s=0.5, t_range=None, rate_cols=None,
                       vm_cols=None, fig_dir='./Figure7', save=True, sinq=None,
                       tag='', fname_prefix='records', show_spikes=True,
                       spike_times=None,
                       raw_voltage=None, channel='voltage_1', decimate=5,
                       target=None, trial_label=None):
    """Everything the analysis actually consumes, for one trial, drawn from
    ``records`` alone.

    The companion to :func:`plot_trial_vm_overlay`, and the difference matters:
    that one recomputes Vm and rate from the trial file, so it shows what the
    signals *look like*. This one plots only columns that are in ``records``,
    including the derived flags — so it shows what the analysis is *using*. If a
    state is mis-classified, a REST bout fails the duration cut, the cue window is
    off, or a pre-movement flag lands somewhere unexpected, it shows up here and
    nowhere else.

    Panels, top to bottom:

    1. probe position, coloured by kinematic state, with the REST bouts that
       *qualify* for the bout analyses shaded and the cue window marked
    2. every rate column present (``rate``, ``rate_g*``, ``rate_box_*``), with the
       pre-movement frames overdrawn in orange (about to move toward target) or
       blue (away) — the same colours as the kernel scatter
    3. every Vm column present (``vm``, ``vm_noblank``, ``vm_g*``)
    4. a strip of the boolean selections themselves: qualifying REST bout,
       ``in_cue``, pre-movement toward, pre-movement away

    A trial with no qualifying REST bout draws an empty band in panel 4 — which is
    the fastest way to see why a cell contributes few bouts (see 210604, whose
    rest is fragmented below the 1 s cut).

    ``show_spikes`` adds a raster above the rate trace. By default it is built from
    ``n_sp``, so the ticks sit at *frame* times (~5 ms bins) — coarser than the
    true spike times, but it is exactly the count the windowed analyses sum, so a
    disagreement between the raster and the rate trace is a real disagreement. A
    frame holding more than one spike gets a taller red tick. Pass
    ``spike_times=`` a Trial (or an array of times) for the exact raster instead;
    the label says which one is drawn.
    """
    rec = records if trial is None else records[records['trial'] == trial]
    if not len(rec):
        return None
    rec = rec.sort_values('t').reset_index(drop=True)
    if t_range is not None:
        rec = rec[(rec['t'] >= t_range[0]) & (rec['t'] <= t_range[1])]
        if not len(rec):
            return None
    t = rec['t'].to_numpy()
    x = rec['x'].to_numpy()
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.0

    if rate_cols is None:
        rate_cols = [c for c in rec.columns
                     if c == 'rate' or c.startswith(('rate_g', 'rate_box'))]
    if vm_cols is None:
        vm_cols = [c for c in rec.columns
                   if c == 'vm' or c == 'vm_noblank' or c.startswith('vm_g')]
    rate_cols = [c for c in rate_cols if c in rec.columns]
    vm_cols = [c for c in vm_cols if c in rec.columns]

    # qualifying REST bouts — the ones bout_rate_spread / premove would use
    rest_bouts = []
    for run in ba._mask_runs(rec, rec['state'].to_numpy() == ba.kin.STATE_REST):
        tt = run['t'].to_numpy()
        if len(tt) > 1 and (len(tt) * dt) >= min_duration_s:
            rest_bouts.append((tt[0], tt[-1]))
    # pre-movement frames, split by upcoming direction
    pm = (rec['t_to_move'].to_numpy() <= premove_s
          if 't_to_move' in rec.columns else np.zeros(len(rec), bool))
    dxi = (rec['next_dx_init'].to_numpy() if 'next_dx_init' in rec.columns
           else np.full(len(rec), np.nan))
    cue = (rec['in_cue'].to_numpy().astype(bool) if 'in_cue' in rec.columns
           else np.zeros(len(rec), bool))

    n_rows = 2 + int(bool(vm_cols)) + 1
    fig, axes = plt.subplots(n_rows, 1, figsize=(13, 2.3 * n_rows), sharex=True,
                             gridspec_kw={'height_ratios':
                                          [2] * (n_rows - 1) + [0.9]})

    def _shade(ax):
        for a, b in rest_bouts:
            ax.axvspan(a, b, color='#2ca02c', alpha=0.07, lw=0, zorder=0)
        for a, b in _true_spans(cue, t):
            ax.axvspan(a, b, color='#1f77b4', alpha=0.13, lw=0, zorder=0)

    # 1. probe, coloured by state
    ax = axes[0]
    for st, name, color in STATE_STYLES:
        seg = np.where(rec['state'].to_numpy() == st, x, np.nan)
        ax.plot(t, seg, color=color, lw=1.2, label=name)
    _shade(ax)
    # The target zone belongs on this axis: every "toward target" claim in the
    # analysis is relative to it, and without it the position trace cannot be
    # read against the task the fly was doing. Falls back to raw_voltage, which
    # is already a Trial whenever the raw trace is being drawn.
    # Fall back to raw_voltage only when it is a Trial; as an explicit
    # (t, v) pair it carries no target metadata.
    fallback = (None if isinstance(raw_voltage, (tuple, list, np.ndarray))
                else raw_voltage)
    band = _target_band(target if target is not None else fallback)
    drew_target = _draw_target(ax, band)
    ax.set_ylabel('probe (um)')
    ax.legend(fontsize=6.5, ncol=4, loc='upper right')
    ax.set_title(f'green = REST bout >= {min_duration_s:g} s (used by the bout '
                 f'analyses), blue = piezo cue'
                 + (f', purple = target [{band[0]:.0f}, {band[1]:.0f}]'
                    if drew_target else ''), fontsize=8)

    # 2. rate columns, with the pre-movement frames overdrawn
    ax = axes[1]
    for c, style in zip(rate_cols, ('-', '-', '--', ':', '-.') * 4):
        ax.plot(t, rec[c], style, lw=1.1, label=c, alpha=0.85)
    for a, b in _true_runs_idx(pm):
        col = ('#ff7f0e' if np.nanmedian(dxi[a:b]) > 0
               else '#1f77b4' if np.nanmedian(dxi[a:b]) < 0 else '#7f7f7f')
        ax.plot(t[a:b], rec[rate_cols[0]].to_numpy()[a:b] if rate_cols else x[a:b],
                '-', color=col, lw=2.4, alpha=0.9, zorder=4)
    _shade(ax)
    ax.set_ylabel('rate (Hz)')

    spike_note = ''
    if show_spikes:
        tt, multi, spike_note = _spike_ticks(rec, t, spike_times)
        if len(tt):
            lo, hi = ax.get_ylim()
            span = (hi - lo) or 1.0
            y0, h = hi + 0.04 * span, 0.05 * span
            ax.vlines(tt, y0, y0 + h, color='#1f77b4', lw=0.5, alpha=0.75,
                      label=spike_note)
            if len(multi):
                # a frame that holds more than one spike — the rate trace cannot
                # show this, but the count the analysis sums does
                ax.vlines(multi, y0, y0 + 1.9 * h, color='#d62728', lw=0.8,
                          label=f'{len(multi)} frames with >1 spike')
            ax.set_ylim(lo, y0 + 2.2 * h)
    ax.legend(fontsize=6.5, ncol=3, loc='upper right')
    ax.set_title(f'thick orange/blue = within {premove_s*1e3:.0f} ms of a movement '
                 f'toward / away from target', fontsize=8)

    row = 2
    if vm_cols:
        ax = axes[row]
        # Raw voltage first, behind everything: the continuous trace the Vm column
        # is derived from, so the blanking and the frame sampling can be judged
        # against the real data rather than taken on trust. Needs the Trial, which
        # records cannot carry.
        if raw_voltage is not None:
            vv = _raw_voltage_trace(raw_voltage, channel)
            if vv is not None:
                t_v, v_raw = vv
                m = (t_v >= t[0]) & (t_v <= t[-1])
                sl = slice(None, None, max(1, int(decimate)))
                ax.plot(t_v[m][sl], v_raw[m][sl], '-', color='0.75', lw=0.4,
                        zorder=0, label=f'raw {channel}')
        for c, style in zip(vm_cols, ('-', '--', ':', '-.') * 4):
            ax.plot(t, rec[c], style, lw=1.1, label=c, alpha=0.9, zorder=3)
        _shade(ax)
        ax.set_ylabel('Vm (mV)')
        ax.legend(fontsize=6.5, ncol=3, loc='upper right')
        row += 1

    # last row: the selections as bars, so "what is included" is explicit
    ax = axes[row]
    bands = [('REST bout used', np.zeros(len(t), bool), '#2ca02c'),
             ('in_cue', cue, '#1f77b4'),
             ('premove toward', pm & (dxi > 0), '#ff7f0e'),
             ('premove away', pm & (dxi < 0), '#1f77b4')]
    used = np.zeros(len(t), bool)
    for a, b in rest_bouts:
        used |= (t >= a) & (t <= b)
    bands[0] = ('REST bout used', used, '#2ca02c')
    for i, (name, mask, color) in enumerate(bands):
        y = len(bands) - 1 - i
        for a, b in _true_spans(mask, t):
            ax.barh(y, b - a, left=a, height=0.7, color=color, alpha=0.8)
    ax.set_yticks(range(len(bands)))
    ax.set_yticklabels([b[0] for b in bands][::-1], fontsize=7)
    ax.set_xlabel('trial time (s)')
    ax.set_ylim(-0.6, len(bands) - 0.4)

    # trial_label names a span when several trials were concatenated; without
    # it a multi-trial frame would title and save itself as 'pooled'.
    tn = (trial_label if trial_label is not None
          else (trial if trial is not None else 'pooled'))
    fig.suptitle(f'{cell_id} trial {tn} — what the analysis sees '
                 f'(all traces drawn from records, not recomputed)', fontsize=10)
    fig.tight_layout()
    if save:
        fig_dir = Path(fig_dir)
        fig_dir.mkdir(exist_ok=True)
        sfx = f'_{tag}' if tag else ''
        fig.savefig(fig_dir / f'{fname_prefix}_{cell_id}'
                              f'{tag_suffix(sinq, cell_id)}_tr{tn}{sfx}.svg')
    return fig


#: Colours shared by the excursion panels. Blue is reserved for movement *away*
#: from the target (negative-going), so the cue gets green rather than the blue it
#: had in the exploratory plots — the scheme has to stay consistent across panels.
EXC_COLORS = {'rest': '#7f7f7f', 'cue': '#2ca02c',
              'premove_toward': '#ff7f0e', 'premove_away': '#1f77b4',
              'current_step': '#d62728'}

#: A short hyperpolarizing current step is delivered near the start of every
#: trial. Measured on 210915_F1_C1 (60 trials, trial-averaged): onset ~-0.99 s,
#: i.e. ~10 ms after the recording starts, lasting ~55-60 ms at about -6.6 pA and
#: pulling Vm down ~3.9 mV, fully recovered by -0.925 s. It is *not* described in
#: ``trial.params`` (mode/gain/Vm_id are all zero) — only ``current_1`` shows it.
#: 0.2 s is a conservative default: it covers the step and its recovery and ends
#: exactly where the cue begins (-0.8 s).
CURRENT_STEP_S = 0.2


#: Metrics carried by ``bout_signal_correlation``, with display settings. ``sd_a``
#: and ``sd_b`` are the within-bout sd of the two signals of the pair, so the same
#: frame answers "how tightly do they couple" and "how much does each wander".
BOUT_METRICS = {
    'rho':   {'label': 'within-bout rho (Vm vs rate)', 'range': (-1, 1)},
    'slope': {'label': 'gain (Hz per mV)', 'range': (-5, 20)},
    'sd_a':  {'label': 'within-bout Vm sd (mV)', 'range': (0, 3)},
    'sd_b':  {'label': 'within-bout rate sd (Hz)', 'range': (0, 20)},
}


def plot_bout_coupling_across_cells(corr, highlight=None, labels=None,
                                    metrics=('rho', 'slope', 'sd_a'),
                                    kind='hist', bins=40, fig_dir='./Figure7',
                                    save=True, sinq=None,
                                    highlight_color='#d62728',
                                    other_color='0.6',
                                    fname_tag='bout_coupling_cells',
                                    metric_specs=None, title=None):
    """Per-bout coupling metrics, one trace per cell, one row per metric.

    Rows are the metrics of ``BOUT_METRICS`` — by default the correlation, the
    gain in Hz/mV, and the within-bout Vm sd. The last is the one that shows the
    kernel doing its job: the same bouts measured at 250 ms should sit at a
    visibly lower sd than at 25 ms, with the *coupling* unchanged or better.

    Columns are the kernel widths, which must be matched within a pair (a 25 ms
    rate against a 250 ms Vm would measure the wider kernel, not the cell).

    ``kind='hist'`` draws density histograms — comparable across cells even
    though bout counts differ by 3x — and ``'cdf'`` draws the cumulative version,
    which is steadier when the traces overlap heavily.

    ``highlight`` names the cell drawn in colour and bold; the rest are grey.

    ``metric_specs`` overrides entries of ``BOUT_METRICS`` — needed whenever the
    pair is not (Vm, rate), because the default axis labels name Vm and its units
    and would otherwise mislabel, say, a velocity-vs-rate gain. Each entry is
    ``{'label': str, 'range': (lo, hi) or None}``; ``None`` autoscales to the
    1st-99th percentile, which is what a slope in unknown units needs. Note the
    x axis is shared per row, so pairs whose slopes live on different scales
    belong in separate calls. ``title`` overrides the suptitle for the same
    reason.
    """
    if not len(corr):
        return None
    labels = list(labels) if labels else list(dict.fromkeys(corr['label']))
    specs = {**BOUT_METRICS, **(metric_specs or {})}
    metrics = [m for m in metrics if m in corr.columns]
    cells = list(dict.fromkeys(corr['cell']))
    fig, axes = plt.subplots(len(metrics), len(labels),
                             figsize=(4.6 * len(labels), 2.9 * len(metrics)),
                             squeeze=False, sharex='row')
    rows = []
    for i, met in enumerate(metrics):
        spec = specs.get(met, {'label': met, 'range': None})
        lo_hi = spec['range']
        vals_all = corr[met].replace([np.inf, -np.inf], np.nan).dropna()
        if lo_hi is None:
            lo_hi = (float(vals_all.quantile(0.01)), float(vals_all.quantile(0.99)))
        edges = np.linspace(lo_hi[0], lo_hi[1], bins + 1)
        for j, lab in enumerate(labels):
            ax = axes[i, j]
            sub = corr[corr['label'] == lab]
            for cell in cells:
                v = sub.loc[sub['cell'] == cell, met]
                v = v.replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
                if not len(v):
                    continue
                is_hi = (cell == highlight)
                style = dict(color=highlight_color if is_hi else other_color,
                             lw=2.2 if is_hi else 1.0,
                             alpha=1.0 if is_hi else 0.75,
                             zorder=5 if is_hi else 2)
                if kind == 'cdf':
                    x = np.sort(np.clip(v, edges[0], edges[-1]))
                    ax.step(x, np.arange(1, len(x) + 1) / len(x), where='post',
                            label=f'{cell} (n={len(v)})' if is_hi else None,
                            **style)
                else:
                    # density, so cells with 269 and 920 bouts are comparable
                    ax.hist(np.clip(v, edges[0], edges[-1]), bins=edges,
                            histtype='step', density=True,
                            label=f'{cell} (n={len(v)})' if is_hi else None,
                            **style)
                rows.append({'metric': met, 'label': lab, 'cell': cell,
                             'n': len(v), 'median': float(np.median(v))})
            if met == 'rho' or lo_hi[0] < 0 < lo_hi[1]:
                ax.axvline(0, color='k', lw=0.8)
            ax.set_xlim(*lo_hi)
            # Every row, not just the last: sharex is per-row, so each row has its
            # own scale and would otherwise go unlabelled.
            ax.set_xlabel(spec['label'])
            if i == 0:
                ax.set_title(lab, fontsize=10)
            if j == 0:
                ax.set_ylabel('density' if kind == 'hist'
                              else 'cumulative fraction')
            med = sub.groupby('cell')[met].median()
            ax.annotate('median\n' + '\n'.join(
                f'{c}: {med.get(c, np.nan):+.2f}' for c in cells),
                xy=(0.98, 0.97), xycoords='axes fraction', va='top', ha='right',
                fontsize=6.5)
    if highlight is not None:
        axes[0, 0].legend(fontsize=8, loc='upper left', frameon=False)
    fig.suptitle((title or 'Within-bout Vm-rate coupling, one line per cell')
                 + (f' ({highlight} in colour)' if highlight else ''),
                 fontsize=10)
    fig.tight_layout()
    if save:
        fig_dir = Path(fig_dir)
        fig_dir.mkdir(exist_ok=True)
        fig.savefig(fig_dir / f'{fname_tag}_{kind}.svg')
    return fig, pd.DataFrame(rows)


def plot_bout_rho_across_cells(corr, **kwargs):
    """Back-compatible alias: rho only. See plot_bout_coupling_across_cells."""
    kwargs.setdefault('metrics', ('rho',))
    kwargs.setdefault('fname_tag', 'bout_rho_cells')
    return plot_bout_coupling_across_cells(corr, **kwargs)


def plot_drate_vs_dvm_trial(records, cell_id, trial=None, cue_post_s=0.2,
                            cue_pre_s=0.0, premove_s=0.5, rate_col='rate',
                            vm_col='vm', connect=True, fig_dir='./Figure7',
                            save=True, sinq=None, tag='', marker_size=10,
                            annotate=True, current_step_s=CURRENT_STEP_S):
    """Change in firing rate against change in Vm, one trial, by excursion type.

    Both axes are differences from the same reference: the median over the
    trial's *steady* REST frames — REST that is neither inside the cue window nor
    in the run-up to a movement. So the origin is "this trial's holding state" and
    every point is a departure from it, which is what makes the excursions
    comparable in size.

    Colours (``EXC_COLORS``), chosen to stay consistent with the position panels:

    ``red``    the hyperpolarizing current step at the start of the trial (see
               ``CURRENT_STEP_S``) — an imposed Vm displacement with a known
               cause, so it belongs on this plot as a reference rather than being
               quietly dropped, but it must not enter the steady-REST baseline
    ``grey``   steady REST — the spontaneous scatter
    ``green``  the cue window plus ``cue_post_s`` (default 200 ms, which is what
               it takes to include the off-response on the piezo's return ramp)
    ``orange`` pre-movement, about to move *toward* the target
    ``blue``   pre-movement, about to move *away*

    Blue means away-from-target throughout, which is why the cue is green here
    even though the exploratory plots drew it blue.

    ``connect`` joins the cue and pre-movement points in time order, so each
    excursion reads as a trajectory (and its hysteresis is visible) rather than a
    cloud. Returns ``(fig, stats)`` with one row per group: n, the peak-to-peak
    and max absolute excursion on each axis, and rho / slope in Hz/mV — both
    annotated on the panel, since the point of comparing groups is how tightly
    each couples and at what gain.
    """
    rec = records if trial is None else records[records['trial'] == trial]
    rec = rec.dropna(subset=[rate_col, vm_col])
    if not len(rec):
        return None
    rec = rec.sort_values('t')
    t = rec['t'].to_numpy()

    # cue window, widened by cue_post_s the same way plot_rate_vs_vm_trial does
    if {'cue_t0', 'cue_t1'} <= set(rec.columns):
        c0 = float(rec['cue_t0'].iloc[0]) - cue_pre_s
        c1 = float(rec['cue_t1'].iloc[0]) + cue_post_s
    elif 'in_cue' in rec.columns and rec['in_cue'].any():
        c0 = float(rec.loc[rec['in_cue'], 't'].min()) - cue_pre_s
        c1 = float(rec.loc[rec['in_cue'], 't'].max()) + cue_post_s
    else:
        c0 = c1 = np.nan
    in_cue = (t >= c0) & (t <= c1) if np.isfinite(c0) else np.zeros(len(t), bool)

    pm = (rec['t_to_move'].to_numpy() <= premove_s
          if 't_to_move' in rec.columns else np.zeros(len(rec), bool))
    dxi = (rec['next_dx_init'].to_numpy() if 'next_dx_init' in rec.columns
           else np.full(len(rec), np.nan))
    is_rest = (rec['state'].to_numpy() == ba.kin.STATE_REST)
    step = ((t < t.min() + current_step_s) if current_step_s
            else np.zeros(len(t), bool))
    # steady rest = REST, not cue, not pre-movement, not the current step. This is
    # the reference, so anything with an imposed cause has to be kept out of it.
    steady = is_rest & ~in_cue & ~pm & ~step
    if steady.sum() < 5:
        steady = is_rest & ~in_cue & ~step
    if steady.sum() < 5:
        return None
    base_r = float(np.median(rec[rate_col].to_numpy()[steady]))
    base_v = float(np.median(rec[vm_col].to_numpy()[steady]))
    d_r = rec[rate_col].to_numpy() - base_r
    d_v = rec[vm_col].to_numpy() - base_v

    groups = [
        ('steady REST', steady, EXC_COLORS['rest'], False),
        ('current step %.0f ms' % (current_step_s * 1e3), step,
         EXC_COLORS['current_step'], True),
        ('cue + %.0f ms' % (cue_post_s * 1e3), in_cue, EXC_COLORS['cue'], True),
        ('pre-move toward', pm & (dxi > 0), EXC_COLORS['premove_toward'], True),
        ('pre-move away', pm & (dxi < 0), EXC_COLORS['premove_away'], True),
    ]

    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    rows = []
    for name, mask, color, is_exc in groups:
        if mask.sum() < 2:
            continue
        x, y = d_v[mask], d_r[mask]
        ax.scatter(x, y, s=marker_size + (6 if is_exc else 0),
                   alpha=0.75 if is_exc else 0.3, color=color, linewidths=0,
                   zorder=4 if is_exc else 2, label=f'{name} (n={mask.sum():,})')
        if connect and is_exc:
            order = np.argsort(t[mask])
            ax.plot(x[order], y[order], '-', color=color, lw=0.9, alpha=0.9,
                    zorder=3)
        rows.append({'group': name, 'n': int(mask.sum()),
                     'dvm_ptp': float(np.ptp(x)), 'drate_ptp': float(np.ptp(y)),
                     'dvm_absmax': float(np.max(np.abs(x))),
                     'drate_absmax': float(np.max(np.abs(y))),
                     'rho': float(np.corrcoef(x, y)[0, 1])
                     if np.std(x) > 0 and np.std(y) > 0 else np.nan,
                     'slope': float(np.polyfit(x, y, 1)[0])
                     if np.std(x) > 0 else np.nan})
    ax.axhline(0, color='k', lw=0.6)
    ax.axvline(0, color='k', lw=0.6)
    ax.set_xlabel('$\\Delta$Vm (mV, from steady REST)')
    ax.set_ylabel('$\\Delta$rate (Hz, from steady REST)')
    ax.legend(fontsize=8, loc='upper left', frameon=False)
    stats = pd.DataFrame(rows)
    if annotate and len(stats):
        lines = [f'{r["group"]}:  rho {r["rho"]:+.2f}   {r["slope"]:+.1f} Hz/mV'
                 f'   max {r["drate_absmax"]:.0f} Hz / {r["dvm_absmax"]:.1f} mV'
                 for _, r in stats.iterrows()]
        ax.annotate('\n'.join(lines), xy=(0.98, 0.02),
                    xycoords='axes fraction', ha='right', va='bottom',
                    fontsize=7.5, linespacing=1.4)
    tn = trial if trial is not None else 'pooled'
    fig.suptitle(f'{cell_id} trial {tn} — rate vs Vm excursions '
                 f'(reference = steady REST median)', fontsize=10)
    fig.tight_layout()
    if save:
        fig_dir = Path(fig_dir)
        fig_dir.mkdir(exist_ok=True)
        sfx = f'_{tag}' if tag else ''
        fig.savefig(fig_dir / f'drate_dvm_{cell_id}{tag_suffix(sinq, cell_id)}'
                              f'_tr{tn}{sfx}.svg')
    return fig, stats


def _target_band(source):
    """``(lo, hi)`` of the target zone, from a Trial or an explicit pair.

    The zone is in the flipped frame -- ``-(probe_position - probeZero)``,
    positive = toward target -- which is the frame ``x`` is already in, so the
    bounds go straight onto a position axis with no conversion. Returns None
    when the trial carries no usable target metadata, so callers skip the band
    rather than draw a NaN span.
    """
    if source is None:
        return None
    if isinstance(source, (tuple, list, np.ndarray)) and len(source) == 2:
        # Must be a scalar pair. A raw-voltage source is ALSO a length-2
        # sequence -- (t_array, v_array) -- and float() on an array raises, so
        # anything non-scalar is simply not a target band.
        a, b = np.asarray(source[0]), np.asarray(source[1])
        if a.ndim or b.ndim:
            return None
        lo, hi = float(a), float(b)
    else:
        # ba.target_bounds catches KeyError/TypeError/ValueError but not
        # AttributeError, which is what a Trial predating the pyas target
        # metadata raises -- and the 2021 cells are exactly that vintage. A
        # missing target must skip the band, never take the figure down.
        try:
            lo, hi = ba.target_bounds(source)
        except AttributeError:
            return None
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return None
    return (lo, hi) if lo <= hi else (hi, lo)


def _draw_target(ax, band, color='#9467bd', alpha=0.13, label='target zone'):
    """Shade the target zone across a position axis. True if it was drawn."""
    if band is None:
        return False
    ax.axhspan(band[0], band[1], color=color, alpha=alpha, lw=0, zorder=0)
    for y in band:
        ax.axhline(y, color=color, lw=0.8, alpha=0.55, zorder=1)
    ax.plot([], [], color=color, lw=6, alpha=min(alpha * 2.4, 1.0), label=label)
    return True


def _raw_voltage_trace(source, channel='voltage_1'):
    """``(t, v)`` for the raw ephys trace, from a Trial or an explicit pair."""
    if source is None:
        return None
    if isinstance(source, (tuple, list)) and len(source) == 2:
        return np.asarray(source[0], float), np.asarray(source[1], float)
    vv = ephys._trial_voltage(source, channel)
    if vv is None:
        return None
    v, fs, t0 = vv
    return t0 + np.arange(v.size) / fs, v


def _spike_ticks(rec, t, spike_times=None):
    """``(tick_times, multi_spike_frame_times, label)`` for the records raster.

    With ``spike_times=None`` the ticks come from ``n_sp`` and therefore sit at
    frame times — the same counts the windowed analyses sum, which is the point:
    the raster and the statistics cannot disagree. Pass a Trial or an array of
    times to draw the exact raster instead, which is finer but no longer purely
    "what the analysis sees".
    """
    if spike_times is not None:
        st = (ephys.trial_spike_times(spike_times)
              if hasattr(spike_times, 'params') else np.asarray(spike_times, float))
        if st is None:
            return np.zeros(0), np.zeros(0), 'no spikes'
        st = st[(st >= t[0]) & (st <= t[-1])]
        return st, np.zeros(0), f'{len(st)} spikes (exact times)'
    if 'n_sp' not in rec.columns:
        return np.zeros(0), np.zeros(0), ''
    n_sp = rec['n_sp'].to_numpy(dtype=float)
    return (t[n_sp > 0], t[n_sp > 1],
            f'{int(n_sp.sum())} spikes (frame bins, from n_sp)')


def _true_runs_idx(mask):
    """(start, stop) index pairs of contiguous True."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return []
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def _true_spans(mask, t):
    """(t_start, t_end) pairs of contiguous True, in the units of ``t``."""
    return [(t[a], t[min(b, len(t) - 1)]) for a, b in _true_runs_idx(mask)]


def plot_trial_vm_overlay(trial, cell_id=None, blank=None, sigma_s=SIGMA_S,
                          sigma_long_s=None, channel='voltage_1',
                          t_range=None, fig_dir='./Figure7', save=True,
                          sinq=None, decimate=1, show_cue=True, tag=None,
                          show_target=True):
    """Raw voltage with the spike-blanked Vm drawn over it, plus probe and rate.

    The visual check on ``subthreshold_vm``: the Vm trace should ride the base of
    the raw trace, ignoring the spikes, and it should not dip every time a spike
    is blanked out. Panels, top to bottom — probe position; raw voltage with Vm
    overlaid; firing rate with the same two kernels as Vm, so a shared
    fluctuation can be read off directly.

    ``blank`` is ``(pre_s, post_s)``; ``None`` measures it from this trial's own
    STA. ``sigma_long_s`` adds a second, wider Vm and rate.
    """
    if blank is None:
        sta = ephys.spike_triggered_average(trial, channel=channel)
        blank = ephys.sta_blank_window(sta)

    v = getattr(trial, channel, None)
    if v is None:
        return None
    v = np.asarray(v, dtype=float).ravel()
    fs = float(trial.params['sampratein'])
    t_v = np.asarray(trial.time).ravel()

    res = ephys.subthreshold_vm(trial, t_axis=t_v, channel=channel,
                                blank_pre_s=blank[0], blank_post_s=blank[1],
                                sigma_s=sigma_s)
    if res is None:
        return None
    _, vm, info = res
    vm_long = None
    if sigma_long_s:
        r2 = ephys.subthreshold_vm(trial, t_axis=t_v, channel=channel,
                                   blank_pre_s=blank[0], blank_post_s=blank[1],
                                   sigma_s=sigma_long_s)
        vm_long = None if r2 is None else r2[1]

    spike_s = ephys.trial_spike_times(trial)
    ds = trial.downsample_probe
    t_f = np.asarray(trial.time)[ds].squeeze()
    x_f = -(np.asarray(trial.probe_position).squeeze()[ds] - trial.probeZero)
    rate = ephys.gaussian_rate(spike_s, t_f, sigma_s=sigma_s)
    rate_long = (ephys.gaussian_rate(spike_s, t_f, sigma_s=sigma_long_s)
                 if sigma_long_s else None)

    sel_v = np.ones(len(t_v), dtype=bool)
    sel_f = np.ones(len(t_f), dtype=bool)
    if t_range is not None:
        sel_v = (t_v >= t_range[0]) & (t_v <= t_range[1])
        sel_f = (t_f >= t_range[0]) & (t_f <= t_range[1])

    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True,
                             gridspec_kw={'height_ratios': [1, 2, 1.2]})
    axes[0].plot(t_f[sel_f], x_f[sel_f], color='#1f77b4', lw=1)
    # Target zone, from this trial's own metadata. Same flipped frame as x_f,
    # so it needs no conversion; silently skipped if the trial has no target.
    if show_target:
        band = _target_band(trial)
        if _draw_target(axes[0], band):
            axes[0].legend(fontsize=6.5, loc='upper right')
    axes[0].set_ylabel('probe (um)')

    sv = slice(None, None, max(1, int(decimate)))
    axes[1].plot(t_v[sel_v][sv], v[sel_v][sv], color='0.75', lw=0.4,
                 label=f'raw {channel}')
    axes[1].plot(t_v[sel_v][sv], vm[sel_v][sv], color='#d62728', lw=1.3,
                 label=f'Vm blanked, sigma={sigma_s*1e3:.0f} ms')
    if vm_long is not None:
        axes[1].plot(t_v[sel_v][sv], vm_long[sel_v][sv], color='#000000', lw=1.6,
                     label=f'Vm blanked, sigma={sigma_long_s*1e3:.0f} ms')
    if spike_s is not None:
        m = (spike_s >= t_v[sel_v][0]) & (spike_s <= t_v[sel_v][-1])
        y = np.nanpercentile(v[sel_v], 99.5)
        axes[1].plot(spike_s[m], np.full(m.sum(), y), '|', ms=4, color='#1f77b4',
                     alpha=0.6, label=f'{int(m.sum())} spikes')
    axes[1].set_ylabel('voltage (mV)')
    axes[1].legend(fontsize=7, loc='lower right', ncol=2)

    axes[2].plot(t_f[sel_f], rate[sel_f], color='#d62728', lw=1,
                 label=f'rate sigma={sigma_s*1e3:.0f} ms')
    if rate_long is not None:
        axes[2].plot(t_f[sel_f], rate_long[sel_f], color='#000000', lw=1.6,
                     label=f'rate sigma={sigma_long_s*1e3:.0f} ms')
    axes[2].set_ylabel('rate (Hz)')
    axes[2].set_xlabel('time (s)')
    axes[2].legend(fontsize=7, loc='lower right')

    # Shade where the widest kernel lacks full support. This figure deliberately
    # plots the untrimmed trial — unlike per_frame_records, which drops these
    # samples — so the ramp has to be marked rather than silently shown: there
    # are no spikes before the recording starts, so a smoothed rate necessarily
    # climbs out of zero over the first ~3 sigma and falls back at the end.
    edge = 3.0 * max(sigma_s, sigma_long_s or 0.0)
    t0, t1 = float(t_v[sel_v][0]), float(t_v[sel_v][-1])
    for ax in axes:
        for a, b in ((t0, t0 + edge), (t1 - edge, t1)):
            ax.axvspan(a, b, color='#d62728', alpha=0.07, zorder=0, lw=0)
    # The piezo cue moves Vm and rate every trial and is easy to read as
    # spontaneous activity, so it is marked rather than left to be noticed.
    if show_cue:
        c0, c1 = ba.cue_window(trial)
        for ax in axes:
            ax.axvspan(c0, c1, color='#1f77b4', alpha=0.10, zorder=0, lw=0)
        axes[0].text(c0, axes[0].get_ylim()[1], ' piezo cue', fontsize=6.5,
                     va='top', color='#1f77b4')
    axes[2].text(t0 + edge, axes[2].get_ylim()[1], f' <{edge*1e3:.0f} ms: kernel'
                 ' lacks support', fontsize=6.5, va='top', color='#d62728')

    tn = int(trial.params['trial'])
    name = cell_id or getattr(trial, '_dfc', '')
    fig.suptitle(f'{name} trial {tn} — blanked Vm over raw ephys {'' if tag is None else tag}'
                 f'(blank -{blank[0]*1e3:.1f}/+{blank[1]*1e3:.1f} ms, '
                 f'{info["frac_blanked"]:.0%} of samples interpolated; '
                 f'shaded = incomplete kernel support)')
    fig.tight_layout()
    if save:
        fig_dir = Path(fig_dir)
        fig_dir.mkdir(exist_ok=True)
        # ``tag`` labels which demo category this trial came from (hold, cue+,
        # premove_up, ...), so one trial can be saved under several categories
        # without the files overwriting each other.
        suffix = f'_{tag}' if tag else ''
        fig.savefig(fig_dir / f'vm_overlay_{name}_tr{tn}{suffix}.svg')
    return fig


def plot_rest_window_summary(spread, decomp, cell_id, fig_dir='./Figure7',
                             save=True, sinq=None):
    """Three panels that together answer "is the spread real fluctuation?".

    left    cloud width vs. window, against the measured window-to-window
            (Allan) floor, the Poisson reference, and a 1/sqrt(W) guide.
            Tracking the guide down to the floor => the spread was fluctuation
            at the measurement timescale. Flattening above the floor =>
            something slower that averaging cannot remove. The Poisson curve is
            a reference only: it lies *above* the data for a regularly firing
            cell, and the annotated Fano factor says by how much.
    middle  the excess over the floor, with its one-sided 95% lower bound. The
            bound is the honest reading: sqrt() of a difference of two noisy
            variances is biased upward, so a small excess with a zero bound is
            not evidence of anything.
    right   where the variance lives — within a rest epoch (fast, averages out),
            between epochs, or between trials (an offset no window length can
            touch).
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    s = spread.sort_values('window_s')
    w = s['window_s'].to_numpy()

    ax = axes[0]
    ax.loglog(w, s['sd_within_bin'], 'o-', color='#1f77b4', label='observed (within position bin)')
    if 'sd_allan' in s:
        fano = s['fano_emp'].median()
        ax.loglog(w, s['sd_allan'], '^-', color='#2ca02c',
                  label=f'measured fast floor (Allan, Fano~{fano:.2f})')
    ax.loglog(w, s['sd_poisson'], 's--', color='#d62728',
              label='Poisson reference  sqrt(r/W)')
    guide = s['sd_within_bin'].iloc[0] * np.sqrt(w[0] / w)
    ax.loglog(w, guide, ':', color='0.5', label='1/sqrt(W) from first point')
    ax.set_xlabel('averaging window W (s)')
    ax.set_ylabel('rate sd (Hz)')
    ax.set_title('cloud width vs. window')
    ax.legend(fontsize=7)
    _plain_log_ticks(ax, w, axis='x')
    _plain_log_ticks(ax, None, axis='y')

    ax = axes[1]
    ax.plot(w, s['sd_excess'], 'o-', color='#2ca02c', label='excess over Poisson')
    ax.plot(w, s['sd_excess_lo95'], 'v--', color='#2ca02c', alpha=0.6,
            label='95% lower bound')
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xscale('log')
    ax.set_xlabel('averaging window W (s)')
    ax.set_ylabel('excess sd (Hz)')
    ax.set_title('real rate variation at fixed position')
    ax.legend(fontsize=7, loc='lower right')
    _plain_log_ticks(ax, w, axis='x')
    ax2 = ax.twinx()
    ax2.plot(w, s['rho'].abs(), 'd-', color='#9467bd', alpha=0.7)
    ax2.set_ylabel('|rho| rate vs. position', color='#9467bd')
    ax2.tick_params(axis='y', labelcolor='#9467bd')
    # Fixed range: autoscaling a correlation that never leaves ~0.01 draws a
    # dramatic-looking trend across a panel of noise.
    ax2.set_ylim(0, 1)

    ax = axes[2]
    d = decomp.sort_values('window_s')
    labels = [f'{v * 1000:.0f}' for v in d['window_s']]
    bottom = np.zeros(len(d))
    for col, color, name in (('frac_within_epoch', '#8c564b', 'within epoch'),
                             ('frac_between_epoch', '#ff7f0e', 'between epochs'),
                             ('frac_between_trial', '#1f77b4', 'between trials')):
        ax.bar(labels, d[col], bottom=bottom, color=color, label=name)
        bottom = bottom + d[col].to_numpy()
    ax.plot(labels, d['poisson_frac_var'].clip(upper=1.0), 'k_', ms=18,
            label='Poisson share of total')
    ax.set_xlabel('window (ms)')
    ax.set_ylabel('fraction of rate variance')
    ax.set_title('where the variance lives')
    ax.legend(fontsize=7, loc='upper right')

    fig.suptitle(f'{cell_id} — REST rate: measurement noise vs. real variation')
    fig.tight_layout()
    if save:
        fig_dir = Path(fig_dir)
        fig_dir.mkdir(exist_ok=True)
        fig.savefig(fig_dir / f'rest_window_summary_{cell_id}{tag_suffix(sinq, cell_id)}.svg')
    return fig


def plot_rate_by_trial(records, cell_id, fig_dir='./Figure7', save=True,
                       sinq=None, state=ba.kin.STATE_REST, silent_hz=1.0):
    """Per-trial firing rate across the session — the check to run before
    reading anything into the width of the rate-vs-position cloud.

    A cloud that is bimodal in rate (a stripe at 0 Hz plus a cloud at tens of
    Hz) has two very different explanations, and they are told apart per trial,
    not per frame:

    - **Biology.** The neuron is recruited in some trials and silent in others.
      Silent and active trials should then be interleaved and should line up
      with something — position, outcome, stimulus state.
    - **Spike-detection dropout.** Detection was tuned on part of the session
      and fails elsewhere. That shows up as *contiguous blocks* of trials at
      ~0 Hz, often at one end of the session, with no behavioural correlate.

    Left panel plots each trial's rate against trial number (both the chosen
    state and the whole trial), with silent trials marked. Right panel is the
    distribution. Trials with no detected spikes at all never reach ``records``,
    so they cannot appear here — the printed count of missing trials is part of
    the diagnostic.
    """
    per_trial = (records.groupby('trial')
                        .apply(lambda g: pd.Series({
                            'rate_all': g['n_sp'].sum() / (len(g) * np.median(np.diff(g['t']))),
                            'rate_state': (g.loc[g['state'] == state, 'n_sp'].sum()
                                           / max((g['state'] == state).sum(), 1)
                                           / np.median(np.diff(g['t']))),
                            'n_state': int((g['state'] == state).sum()),
                        }), include_groups=False)
                        .reset_index())
    if per_trial.empty:
        return None
    silent = per_trial['rate_all'] < silent_hz

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6),
                             gridspec_kw={'width_ratios': [2.5, 1]})
    ax = axes[0]
    ax.plot(per_trial['trial'], per_trial['rate_all'], '.-', ms=3, lw=0.5,
            color='0.6', label='whole trial')
    ax.plot(per_trial.loc[per_trial['n_state'] > 0, 'trial'],
            per_trial.loc[per_trial['n_state'] > 0, 'rate_state'], '.',
            ms=4, color='#1f77b4', label='REST only')
    if silent.any():
        ax.plot(per_trial.loc[silent, 'trial'], per_trial.loc[silent, 'rate_all'],
                'x', ms=5, color='#d62728',
                label=f'< {silent_hz:g} Hz ({int(silent.sum())} trials)')
    ax.set_xlabel('trial number')
    ax.set_ylabel('rate (Hz)')
    ax.set_title('per-trial rate across the session', fontsize=10)
    ax.legend(fontsize=7)

    ax = axes[1]
    ax.hist(per_trial['rate_all'], bins=30, color='0.6', orientation='horizontal')
    ax.set_xlabel('trials')
    ax.set_ylabel('rate (Hz)')
    ax.set_title('distribution', fontsize=10)

    frac_silent = float(silent.mean())
    fig.suptitle(f'{cell_id} — {len(per_trial)} trials with detected spikes, '
                 f'{frac_silent:.0%} below {silent_hz:g} Hz')
    fig.tight_layout()
    if save:
        fig_dir = Path(fig_dir)
        fig_dir.mkdir(exist_ok=True)
        fig.savefig(fig_dir / f'rate_by_trial_{cell_id}{tag_suffix(sinq, cell_id)}.svg')
    return fig, per_trial


# ---------------------------------------------------------------------------
# Membrane potential: spike-triggered average, Vm vs. rate, Vm vs. position
# ---------------------------------------------------------------------------

def plot_spike_sta(sta, blank, cell_id, fig_dir='./Figure7', save=True,
                   sinq=None):
    """Spike-triggered average voltage with the measured blanking window shaded.

    This is what justifies the blanking window used for Vm: the shaded span is
    where the average spike still deflects the trace. An after-hyperpolarization
    extending past the shading would be left in "subthreshold" Vm and would
    scale with rate all by itself.
    """
    if sta is None:
        return None
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    lag_ms = sta['lags_s'] * 1e3
    ax.fill_between(lag_ms, sta['mean'] - sta['sd'], sta['mean'] + sta['sd'],
                    color='0.8', label='+/- sd')
    ax.plot(lag_ms, sta['mean'], color='k', lw=1.2, label='mean')
    pre_s, post_s = blank
    ax.axvspan(-pre_s * 1e3, post_s * 1e3, color='#d62728', alpha=0.15,
               label=f'blank -{pre_s*1e3:.1f} / +{post_s*1e3:.1f} ms')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel('time from spike peak (ms)')
    ax.set_ylabel('voltage (baseline-subtracted)')
    ax.set_title(f'{cell_id} — spike-triggered average (n = {sta["n_spikes"]:,})',
                 fontsize=10)
    ax.legend(fontsize=7)
    fig.tight_layout()
    if save:
        fig_dir = Path(fig_dir)
        fig_dir.mkdir(exist_ok=True)
        fig.savefig(fig_dir / f'sta_{cell_id}{tag_suffix(sinq, cell_id)}.svg')
    return fig


def plot_vm_vs_rate(records, cell_id, fig_dir='./Figure7', save=True,
                    sinq=None, rate_col='rate', n_bins=20, center_per_trial=False,
                    vm_col='vm'):
    """Vm (x) against firing rate (y), one panel per movement state.

    The binned curve is the f-I-like relation; the annotated slope is its linear
    summary in Hz/mV. ``center_per_trial`` subtracts each trial's median Vm
    first, which is the version to trust when electrode drift makes absolute Vm
    incomparable across a session — at the cost of discarding any real
    between-trial Vm difference, so read the two together rather than picking
    one.

    ``vm_col='vm_noblank'`` plots the unblanked control (see
    ``vm_control`` in :func:`mapd.bout_analysis.per_frame_records`). Blanking
    biases Vm downward in proportion to rate and leaving spikes in biases it
    upward, so the two versions bracket the truth: a slope that keeps its sign
    and rough size in both is real, one that flips was made by the processing.
    """
    if vm_col not in records.columns or records[vm_col].isna().all():
        return None
    df = records.dropna(subset=[vm_col, rate_col]).copy()
    if vm_col != 'vm':
        df['vm'] = df[vm_col]
    if center_per_trial:
        df['vm'] = df['vm'] - df.groupby('trial')['vm'].transform('median')
    frac = float(np.nanmean(df['vm_frac_blanked'])) if 'vm_frac_blanked' in df else np.nan

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharex=True, sharey=True)
    for ax, (state, label, color) in zip(axes, STATE_STYLES):
        g = df[df['state'] == state]
        if len(g) < 10:
            ax.set_title(f'{label}: n={len(g)}')
            continue
        v, r = g['vm'].to_numpy(), g[rate_col].to_numpy()
        ax.scatter(v, r, s=4, alpha=0.25, color=color, linewidths=0)
        edges = np.quantile(v, np.linspace(0, 1, n_bins + 1))
        edges = np.unique(edges)
        if len(edges) > 2:
            idx = np.clip(np.searchsorted(edges, v, side='right') - 1, 0, len(edges) - 2)
            centers = 0.5 * (edges[:-1] + edges[1:])
            means = np.array([r[idx == b].mean() if (idx == b).any() else np.nan
                              for b in range(len(centers))])
            ax.plot(centers, means, 'o-', color='k', ms=3, lw=1.2)
        rho = float(np.corrcoef(v, r)[0, 1])
        slope = float(np.polyfit(v, r, 1)[0])
        ax.set_title(f'{label}: n={len(g):,}  rho={rho:+.2f}  {slope:+.2f} Hz/mV',
                     fontsize=9)
        ax.set_xlabel('Vm (mV)' + (' , per-trial centered' if center_per_trial else ''))
    axes[0].set_ylabel('spike rate (Hz)')
    what = ('spikes left in (control)' if vm_col == 'vm_noblank'
            else f'mean {frac:.0%} of samples blanked')
    fig.suptitle(f'{cell_id} — subthreshold Vm vs. firing rate ({what})')
    fig.tight_layout()
    if save:
        fig_dir = Path(fig_dir)
        fig_dir.mkdir(exist_ok=True)
        sfx = tag_suffix(sinq, cell_id)
        tag = ('_centered' if center_per_trial else '') + \
              ('_noblank' if vm_col == 'vm_noblank' else '')
        fig.savefig(fig_dir / f'vm_vs_rate_{cell_id}{sfx}{tag}.svg')
    return fig


def plot_vm_drift(records, cell_id, fig_dir='./Figure7', save=True, sinq=None,
                  state=ba.kin.STATE_REST):
    """Per-trial Vm and rate across the session — the drift check for Vm.

    Vm has no stable zero over an hour of recording: access resistance and cell
    health wander, and per-trial mean Vm can move tens of mV for reasons that
    have nothing to do with firing. If Vm marches with trial number while rate
    drifts the other way, the session-pooled Vm-rate correlation inverts and
    reports the opposite of what happens inside each trial. This panel makes
    that visible, and the annotated within-trial rho is the number to quote.
    """
    if 'vm' not in records.columns or records['vm'].isna().all():
        return None
    stats = ba.vm_rate_correlations(records, state=state)
    g = records.dropna(subset=['vm', 'rate'])
    g = g[g['state'] == state]
    if not len(g):
        return None
    pt = g.groupby('trial').agg(vm=('vm', 'median'), rate=('rate', 'median'))

    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.plot(pt.index, pt['vm'], '.-', ms=3, lw=0.6, color='#1f77b4')
    ax.set_xlabel('trial number')
    ax.set_ylabel('per-trial median Vm (mV)', color='#1f77b4')
    ax.tick_params(axis='y', labelcolor='#1f77b4')
    ax2 = ax.twinx()
    ax2.plot(pt.index, pt['rate'], '.-', ms=3, lw=0.6, color='#d62728')
    ax2.set_ylabel('per-trial median rate (Hz)', color='#d62728')
    ax2.tick_params(axis='y', labelcolor='#d62728')
    if stats:
        warn = ('  <-- drift: quote the within-trial value'
                if abs(stats['rho_vm_vs_trial']) > 0.5 else '')
        ax.set_title(
            f'{cell_id} — rho(Vm,rate): within-trial {stats["rho_within_trial"]:+.2f}, '
            f'between-trial {stats["rho_between_trial"]:+.2f}, '
            f'pooled {stats["rho_pooled"]:+.2f}\n'
            f'Vm vs trial number {stats["rho_vm_vs_trial"]:+.2f} over '
            f'{stats["vm_range"]:.1f} mV{warn}', fontsize=9)
    fig.tight_layout()
    if save:
        fig_dir = Path(fig_dir)
        fig_dir.mkdir(exist_ok=True)
        fig.savefig(fig_dir / f'vm_drift_{cell_id}{tag_suffix(sinq, cell_id)}.svg')
    return fig, stats


def plot_vm_vs_position_rest(windows, cell_id, window_s=None, fig_dir='./Figure7',
                             save=True, sinq=None):
    """During REST, at one averaging window: rate vs. position, Vm vs. position,
    and Vm vs. rate — the three pairings of the same windowed samples.

    If Vm tracks position more tightly than rate does, the rate estimate is the
    noisy link in the chain rather than the relation being weak.
    """
    if 'vm' not in windows.columns or windows['vm'].isna().all():
        return None
    if window_s is None:
        window_s = sorted(windows['window_s'].unique())[-1]
    g = windows[(windows['window_s'] == window_s)].dropna(subset=['vm'])
    if len(g) < 10:
        return None

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    pairs = [('x_mean', 'rate', 'probe position (um)', 'spike rate (Hz)'),
             ('x_mean', 'vm', 'probe position (um)', 'Vm (mV)'),
             ('vm', 'rate', 'Vm (mV)', 'spike rate (Hz)')]
    for ax, (xk, yk, xl, yl) in zip(axes, pairs):
        xv, yv = g[xk].to_numpy(), g[yk].to_numpy()
        ax.scatter(xv, yv, s=10, alpha=0.5, color='#666666', linewidths=0)
        if np.std(xv) > 0 and np.std(yv) > 0:
            rho = float(np.corrcoef(xv, yv)[0, 1])
            fit = np.polyfit(xv, yv, 1)
            xs = np.linspace(xv.min(), xv.max(), 50)
            ax.plot(xs, np.polyval(fit, xs), '-', color='#d62728', lw=1.2)
            ax.set_title(f'rho = {rho:+.2f}   slope = {fit[0]:+.3g}', fontsize=9)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
    fig.suptitle(f'{cell_id} — REST, {window_s * 1000:.0f} ms windows (n = {len(g):,})')
    fig.tight_layout()
    if save:
        fig_dir = Path(fig_dir)
        fig_dir.mkdir(exist_ok=True)
        fig.savefig(fig_dir / f'vm_rest_{cell_id}{tag_suffix(sinq, cell_id)}.svg')
    return fig


# ---------------------------------------------------------------------------
# Per-state velocity histograms
# ---------------------------------------------------------------------------

def plot_velocity_histograms(records, cell_id, fig_dir='./Figure7',
                             v_range=(-200, 200), bins=80, save=True,
                             sinq=None):
    """Smoothed signed velocity histograms, one panel per movement state.

    DRIFT panel answers "are flies ever slowly ramping force?" — if the
    distribution is meaningfully right-of-zero (frac > 0 well above 0%),
    sub-threshold positive movement is real and not just probe relaxation.
    Each panel annotates the fraction of samples with v > 0, median v,
    and a vertical line at v=0.
    """
    parts = []
    for _, g in records.groupby('trial', sort=True):
        if len(g) < 4:
            continue
        v = signed_v(g)
        parts.append(pd.DataFrame({'v': v, 'state': g['state'].to_numpy()}))
    if not parts:
        return None
    df = pd.concat(parts, ignore_index=True)

    state_styles = [
        (ba.kin.STATE_MOVE,  'MOVE',  '#d62728'),
        (ba.kin.STATE_DRIFT, 'DRIFT', '#ff7f0e'),
        (ba.kin.STATE_REST,  'REST',  '#666666'),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
    edges = np.linspace(v_range[0], v_range[1], bins + 1)
    for ax, (st, label, color) in zip(axes, state_styles):
        v = df.loc[df['state'] == st, 'v'].to_numpy()
        n = len(v)
        if n == 0:
            ax.set_title(f'{label}: n=0')
            continue
        ax.hist(np.clip(v, v_range[0], v_range[1]),
                bins=edges, color=color, alpha=0.85, log=True)
        ax.axvline(0, color='k', lw=0.8)
        frac_pos = float(np.mean(v > 0))
        med = float(np.median(v))
        ax.set_title(f'{label}: n={n:,}  frac(v>0)={frac_pos:.2%}  median={med:+.1f}')
        ax.set_xlabel('smoothed v (um/s)')
    axes[0].set_ylabel('count (log)')
    fig.suptitle(f'{cell_id} — per-state velocity histograms')
    fig.tight_layout()
    if save:
        fig_dir = Path(fig_dir)
        fig_dir.mkdir(exist_ok=True)
        sfx = tag_suffix(sinq, cell_id)
        fig.savefig(fig_dir / f'velocity_hist_{cell_id}{sfx}.svg')
    return fig


# ---------------------------------------------------------------------------
# Rate vs. velocity CCF panel
# ---------------------------------------------------------------------------

def plot_cell_ccf(records, cell_id, fig_dir='./Figure7',
                  lag_window_s=0.5, n_null=0, save=True,
                  state=ba.kin.STATE_MOVE, sinq=None):
    """Two-panel CCF figure: smoothed signed v and rectified v_+.

    For each signal: per-trial Pearson CCF, K=5 trial-level CV for an
    unbiased held-out r at tau*, optional permutation null band.

    By default the CCF is restricted to MOVE-state samples — each trial
    contributes one CCF averaged across its contiguous MOVE runs. Pass
    ``state=None`` to use every frame regardless of bout state.
    """
    fig_dir = Path(fig_dir)
    if save:
        fig_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    results = {}
    for ax, (sig_fn, label, key) in zip(
            axes,
            [(signed_v, 'smoothed signed v', 'signed_v'),
             (positive_v, 'smoothed v_+', 'positive_v')]):
        lags_s, ccfs = per_trial_ccfs(records, lag_window_s=lag_window_s,
                                      signal_fn=sig_fn, state=state)
        if not ccfs:
            ax.set_title(f'{label}: no usable trials')
            continue
        null = (permutation_null(records, lag_window_s=lag_window_s,
                                 signal_fn=sig_fn, n_iter=n_null, state=state)
                if n_null else [])
        taus, r_hold, mean_ccf = cv_lag(ccfs, lags_s, k=5)

        if len(null):
            lo, hi = np.nanpercentile(null, [5, 95], axis=0)
            ax.fill_between(lags_s * 1000, lo, hi, color='0.85',
                            label=f'5-95% null (n={n_null})')

        ax.plot(lags_s * 1000, mean_ccf, lw=1.5, color='C0', label='mean CCF')
        ax.axvline(0, color='k', lw=0.5)
        ax.axhline(0, color='k', lw=0.5)

        tau_med_ms = 1000 * np.median(taus)
        r_mean = np.nanmean(r_hold)
        r_sd = np.nanstd(r_hold)
        ax.axvline(tau_med_ms, color='C3', lw=1, ls='--',
                   label=f'tau* (CV med) = {tau_med_ms:+.0f} ms')
        ax.set_title(f'{label}: held-out r @ tau* = {r_mean:+.3f} +/- {r_sd:.3f}')
        ax.set_xlabel('lag (ms)   [+ = rate leads]')
        ax.legend(loc='best', fontsize=8)

        results[key] = {'lags_s': lags_s, 'mean_ccf': mean_ccf,
                        'taus': taus, 'r_hold': r_hold}

    axes[0].set_ylabel('cross-correlation')
    state_tag = {None: 'all frames',
                 ba.kin.STATE_MOVE: 'MOVE runs only',
                 ba.kin.STATE_DRIFT: 'DRIFT runs only',
                 ba.kin.STATE_REST: 'REST runs only'}.get(state, f'state={state}')
    state_sfx = {None: 'all',
                 ba.kin.STATE_MOVE: 'move',
                 ba.kin.STATE_DRIFT: 'drift',
                 ba.kin.STATE_REST: 'rest'}.get(state, f'state{state}')
    fig.suptitle(f'{cell_id} — rate(t) vs. v(t+tau)  (50 ms boxcar on v, {state_tag})')
    fig.tight_layout()
    if save:
        sfx = tag_suffix(sinq, cell_id)
        # fig.savefig(fig_dir / f'CCF_{state_sfx}_{cell_id}{sfx}.png')
        fig.savefig(fig_dir / f'CCF_{state_sfx}_{cell_id}{sfx}.svg')
    return results


# ---------------------------------------------------------------------------
# CCF stratified by bout strength
# ---------------------------------------------------------------------------

def plot_cell_ccf_by_strength(records, cell_id, fig_dir='./Figure7',
                              lag_window_s=0.25, n_bins=3, save=True,
                              sinq=None):
    """Two-panel CCF figure (signed v, v_+), MOVE-only, stratified by per-bout
    peak |smoothed v| into ``n_bins`` strength bins.

    Each panel overlays ``n_bins`` mean CCFs — one per peak-|v| bin. Tests
    whether the rate-leads peak shrinks (or the v-leads peak grows) as
    movements get stronger.
    """
    from mapd.bout_analysis import per_bout_ccfs

    fig_dir = Path(fig_dir)
    if save:
        fig_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, n_bins))
    results = {}
    for ax, (sig_fn, label, key) in zip(
            axes,
            [(signed_v, 'smoothed signed v', 'signed_v'),
             (positive_v, 'smoothed v_+', 'positive_v')]):
        lags_s, items = per_bout_ccfs(records, lag_window_s=lag_window_s,
                                      signal_fn=sig_fn,
                                      state=ba.kin.STATE_MOVE)
        if len(items) < n_bins:
            ax.set_title(f'{label}: too few bouts ({len(items)})')
            continue
        peaks = np.array([p for p, _, _ in items])
        ccfs = np.array([c for _, c, _ in items])
        # Rank-based assignment so the bins always have ~equal counts.
        ranks = np.argsort(np.argsort(peaks))
        bin_idx = np.clip((ranks * n_bins) // len(peaks), 0, n_bins - 1)
        bin_records = []
        for b in range(n_bins):
            sel = bin_idx == b
            if not sel.any():
                continue
            mean_ccf = np.nanmean(ccfs[sel], axis=0)
            vmin, vmax = peaks[sel].min(), peaks[sel].max()
            ax.plot(lags_s * 1000, mean_ccf, lw=1.5, color=cmap[b],
                    label=f'|v|={vmin:.0f}-{vmax:.0f} (n={int(sel.sum())})')
            bin_records.append({'bin': b, 'mean_ccf': mean_ccf,
                                'n_bouts': int(sel.sum()),
                                'v_min': float(vmin), 'v_max': float(vmax)})
        ax.axvline(0, color='k', lw=0.5)
        ax.axhline(0, color='k', lw=0.5)
        ax.set_title(label)
        ax.set_xlabel('lag (ms)   [+ = rate leads]')
        ax.legend(loc='best', fontsize=7)
        results[key] = {'lags_s': lags_s, 'bins': bin_records}
    axes[0].set_ylabel('cross-correlation')
    fig.suptitle(f'{cell_id} — CCF by bout peak |v|, MOVE-only ({n_bins} bins)')
    fig.tight_layout()
    if save:
        sfx = tag_suffix(sinq, cell_id)
        fig.savefig(fig_dir / f'CCF_by_strength_{cell_id}{sfx}.svg')
    return results


# ---------------------------------------------------------------------------
# CCF sign-split inside MOVE (signed-v restricted to v>0 vs. v<0 sub-runs)
# ---------------------------------------------------------------------------

def plot_cell_ccf_by_sign(records, cell_id, fig_dir='./Figure7',
                          lag_window_s=0.15, save=True, sinq=None):
    """Signed-v CCF restricted to MOVE+v>0 vs. MOVE+v<0 contiguous sub-runs,
    with the union (all MOVE) overlaid as reference.

    If the near-zero negative dip vanishes in the v>0 partition, the
    "return-movement de-recruitment" reading is supported. If the dip
    survives in v>0, peak-isometric anti-correlation (rate high at the
    v ≈ 0 force peak) is the more likely driver.

    A shorter ``lag_window_s`` than the all-MOVE CCF helps because sub-runs
    of single-sign v inside a bout are short.
    """
    from mapd.bout_analysis import per_trial_ccfs_masked

    fig_dir = Path(fig_dir)
    if save:
        fig_dir.mkdir(exist_ok=True)

    def mask_move(g):
        return g['state'].to_numpy() == ba.kin.STATE_MOVE

    def mask_move_pos(g):
        return mask_move(g) & (signed_v(g) > 0)

    def mask_move_neg(g):
        return mask_move(g) & (signed_v(g) < 0)

    fig, ax = plt.subplots(figsize=(7.5, 4))
    out = {}
    for label, mask_fn, color, key in [
        ('MOVE all',  mask_move,     '#888888', 'all'),
        ('MOVE, v>0', mask_move_pos, '#d62728', 'pos'),
        ('MOVE, v<0', mask_move_neg, '#1f77b4', 'neg'),
    ]:
        lags_s, ccfs = per_trial_ccfs_masked(
            records, lag_window_s=lag_window_s,
            signal_fn=signed_v, mask_fn=mask_fn)
        if not ccfs:
            continue
        mat = np.stack(list(ccfs.values()))
        mean_ccf = np.nanmean(mat, axis=0)
        n = np.sum(np.all(np.isfinite(mat), axis=1))
        sem = (np.nanstd(mat, axis=0) / np.sqrt(max(n, 1)))
        ax.plot(lags_s * 1000, mean_ccf, lw=1.5, color=color,
                label=f'{label} (n_trials={len(ccfs)})')
        ax.fill_between(lags_s * 1000, mean_ccf - sem, mean_ccf + sem,
                        color=color, alpha=0.2)
        out[key] = {'lags_s': lags_s, 'mean_ccf': mean_ccf,
                    'n_trials': len(ccfs)}
    ax.axvline(0, color='k', lw=0.5)
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel('lag (ms)   [+ = rate leads]')
    ax.set_ylabel('cross-correlation (signed v)')
    ax.set_title(f'{cell_id} — sign-split CCF inside MOVE  '
                 f'(lag ±{int(lag_window_s*1000)} ms)')
    ax.legend(fontsize=8)
    fig.tight_layout()
    if save:
        sfx = tag_suffix(sinq, cell_id)
        fig.savefig(fig_dir / f'CCF_sign_split_{cell_id}{sfx}.svg')
    return out


# ---------------------------------------------------------------------------
# Diagnostic: list bouts that landed in each (force, vel) bin
# ---------------------------------------------------------------------------

def print_bouts_per_bin(as_off, force_labels, vel_labels=VEL_LABELS):
    for fb in force_labels:
        fb_sub = as_off[as_off['force_bin'] == fb]
        if fb_sub.empty:
            print(f'\n{fb}: (no bouts)')
            continue
        print(f'\n{fb}:')
        for vq in vel_labels:
            bin_sub = fb_sub[fb_sub['vel_q'] == vq].sort_values(['trial', 'start_time'])
            if bin_sub.empty:
                print(f'   {vq} (n=0): -')
                continue
            entries = [f'{int(r.trial)}@{r.start_time:.2f}s' for r in bin_sub.itertuples()]
            print(f'   {vq} (n={len(bin_sub)}): {", ".join(entries)}')


# ---------------------------------------------------------------------------
# Quartile-analysis save helpers + driver
# ---------------------------------------------------------------------------

def _save_count_heatmap(counts, cell_id, n_bouts, fig_dir, name_suffix=''):
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    im = ax.imshow(counts.values, cmap='viridis', aspect='auto')
    mx = counts.values.max() if counts.values.size else 0
    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            v = counts.values[i, j]
            ax.text(j, i, str(v), ha='center', va='center',
                    color='white' if v < mx / 2 else 'black', fontsize=9)
    ax.set_xticks(range(counts.shape[1])); ax.set_xticklabels(counts.columns)
    ax.set_yticks(range(counts.shape[0])); ax.set_yticklabels(counts.index)
    ax.set_xlabel('init |v| quartile'); ax.set_ylabel('init position regime')
    ax.set_title(f'{cell_id}: as_off bout counts (n={n_bouts})')
    fig.colorbar(im, ax=ax, label='# bouts')
    fig.tight_layout()
    fig.savefig(fig_dir / f'counts_{cell_id}{name_suffix}.svg')
    plt.close(fig)


def _save_marginals_and_grid(t_rel, segs, cell_id, fb_labels,
                             ylabel, title_tag, fname_tag,
                             grid_color, fig_dir, min_bouts=MIN_BOUTS,
                             name_suffix=''):
    if not segs:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    cmap_f = plt.cm.viridis(np.linspace(0.15, 0.85, len(fb_labels)))
    for color, fb in zip(cmap_f, fb_labels):
        sub = [s for s in segs if s[0] == fb]
        if len(sub) < min_bouts:
            continue
        _, mean, sem = mean_sem(t_rel, sub)
        axes[0].plot(t_rel, mean, color=color, label=f'{fb} (n={len(sub)})')
        axes[0].fill_between(t_rel, mean - sem, mean + sem, color=color, alpha=0.2)
    axes[0].set_title('by initiation force')
    axes[0].set_xlabel('time from bout onset (s)')
    axes[0].set_ylabel(ylabel)
    axes[0].axvline(0, color='k', lw=0.5)
    axes[0].legend(fontsize=8)

    cmap_v = plt.cm.plasma(np.linspace(0.15, 0.85, 4))
    for color, vq in zip(cmap_v, VEL_LABELS):
        sub = [s for s in segs if s[1] == vq]
        if len(sub) < min_bouts:
            continue
        _, mean, sem = mean_sem(t_rel, sub)
        axes[1].plot(t_rel, mean, color=color, label=f'{vq} (n={len(sub)})')
        axes[1].fill_between(t_rel, mean - sem, mean + sem, color=color, alpha=0.2)
    axes[1].set_title('by initial |v| quartile')
    axes[1].set_xlabel('time from bout onset (s)')
    axes[1].axvline(0, color='k', lw=0.5)
    axes[1].legend(fontsize=8)
    fig.suptitle(f'{cell_id}: {title_tag} (as_off, mean ± SEM)')
    fig.tight_layout()
    fig.savefig(fig_dir / f'{fname_tag}_marginals_{cell_id}{name_suffix}.svg')
    plt.close(fig)

    fig, gax = plt.subplots(5, 4, figsize=(11, 11), sharex=True, sharey=True)
    for i, fb in enumerate(fb_labels):
        for j, vq in enumerate(VEL_LABELS):
            ax = gax[i, j]
            sub = [s for s in segs if s[0] == fb and s[1] == vq]
            if len(sub) >= min_bouts:
                _, mean, sem = mean_sem(t_rel, sub)
                ax.plot(t_rel, mean, color=grid_color)
                ax.fill_between(t_rel, mean - sem, mean + sem,
                                color=grid_color, alpha=0.25)
            ax.axvline(0, color='k', lw=0.4)
            ax.text(0.02, 0.95, f'n={len(sub)}', transform=ax.transAxes,
                    ha='left', va='top', fontsize=8)
            if i == 0:
                ax.set_title(vq)
            if j == 0:
                ax.set_ylabel(fb, rotation=0, ha='right', va='center')
    for ax in gax[-1]:
        ax.set_xlabel('t from bout onset (s)')
    fig.suptitle(f'{cell_id}: {title_tag}, force × init|v| quartile (as_off)')
    fig.tight_layout()
    fig.savefig(fig_dir / f'{fname_tag}_grid_{cell_id}{name_suffix}.svg')
    plt.close(fig)


def run_cell_quartile_analysis(sinq, cell_id, fig_dir='./Figure7',
                               init_window_s=1.0):
    """Restore T, bin as_off bouts into 5 force regimes x 4 |v| quartiles,
    save count + bout-aligned rate/position figures. Returns the binned
    bouts DataFrame.
    """
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(exist_ok=True)

    T = sinq.restore_table(dayflycell=cell_id)
    T.exclude_trials()
    T.extract_trial_properties()

    bouts_df = collect_bouts_with_meta(T, init_window_s=init_window_s)
    edges, fb_labels = force_bin_edges(T)
    as_off = bouts_df[bouts_df['as_outcome'] == 'as_off'].copy()
    print(f'   {len(bouts_df)} bouts total, {len(as_off)} from as_off')
    if as_off.empty:
        print('   (skipping — no as_off bouts)')
        return None

    as_off['force_bin'] = pd.cut(as_off['init_pos'], bins=edges, labels=fb_labels)
    as_off['vel_q'] = pd.qcut(as_off['init_abs_speed'], q=4, labels=list(VEL_LABELS))

    sfx = tag_suffix(sinq, cell_id)
    counts = (as_off.groupby(['force_bin', 'vel_q'], observed=False)
                    .size().unstack('vel_q').reindex(fb_labels).fillna(0).astype(int))
    _save_count_heatmap(counts, cell_id, len(as_off), fig_dir, name_suffix=sfx)

    t_rel_r, segs_r = bout_aligned_rate_segments(T, as_off,
                                                 pre_s=PRE_S, post_s=POST_S)
    _save_marginals_and_grid(t_rel_r, segs_r, cell_id, fb_labels,
                             ylabel='rate (Hz)',
                             title_tag='bout-aligned rate',
                             fname_tag='rate', grid_color='C0',
                             fig_dir=fig_dir, name_suffix=sfx)

    t_rel_x, segs_x = bout_aligned_position_segments(T, as_off,
                                                     pre_s=PRE_S, post_s=POST_S)
    _save_marginals_and_grid(t_rel_x, segs_x, cell_id, fb_labels,
                             ylabel='probe position (um)',
                             title_tag='bout-aligned probe position',
                             fname_tag='position', grid_color='C2',
                             fig_dir=fig_dir, name_suffix=sfx)
    return as_off


# ---------------------------------------------------------------------------
# Driver: REST window sweep (+ Vm) for one cell
# ---------------------------------------------------------------------------

def measure_blank_window(T, n_trials=10, channel='voltage_1', tol_frac=0.05,
                         verbose=True):
    """Spike-triggered average over the first ``n_trials`` usable trials, and
    the blanking window it implies. Returns ``(sta, (pre_s, post_s), info)``.

    Thin wrapper over :func:`mapd.ephys.blank_window_from_trials`, which the
    trial browser also uses, so the notebook and the browser cannot disagree
    about a cell's blank window. A clipped window (the STA still deflected at the
    edge of the measured span) is reported rather than silently accepted: it
    means the window is a lower bound and the span should be widened.
    """
    sta, (pre_s, post_s), info = ephys.blank_window_from_trials(
        T.df['Trial'], n_trials=n_trials, channel=channel, tol_frac=tol_frac)
    if verbose and info.get('clipped'):
        print(f'   WARNING: STA still deflected at the edge of the measured '
              f'span; blank window {pre_s*1e3:.1f}/{post_s*1e3:.1f} ms is a '
              f'lower bound — re-measure with a longer pre_s/post_s')
    return sta, (pre_s, post_s), info


def build_cell_records(sinq, cell_id, sigma_s=SIGMA_S, with_vm=True,
                       vm_control=False, pad_spikes=True, trim_edges_s=0.0,
                       rate_specs=None, vm_specs=None, sta_trials=8,
                       add_timing=True, drop_table=True, keep_table=False,
                       raw_channels=None, verbose=True, keep_silent=False,
                       **bout_kwargs):
    """Gather one cell's per-frame records. No analysis, no figures.

    This is the expensive step — restoring the Table, reading voltage, smoothing —
    and it is deliberately separate from what is done with the result, so a cell
    can be loaded once and several analyses run off the same ``records``.

    What it does: restore the Table, drop excluded and bad-ephys trials, measure
    the cell's own spike blanking window from its STA, then build padded per-frame
    records with ``t / x / state / rate / n_sp`` (+ Vm, + cue timing) and the
    movement-timing columns.

    ``pad_spikes`` splices the neighbouring trials' spikes *and* voltage, which
    both removes the kernel edge trim (no data lost to 3 sigma per end) and is
    what makes the piezo cue measurable at wide kernels, since the cue sits only
    ~200 ms into the trial.

    Returns a dict: ``cell_id``, ``records``, ``blank``, ``sta``, ``blank_info``,
    ``n_usable``, ``n_no_spikes``, and ``table`` when ``keep_table``. ``records``
    is None if the cell yielded nothing.
    """
    T = sinq.restore_table(dayflycell=cell_id)
    T.exclude_trials()
    T.exclude_ephys_trials(verbose=verbose)
    T.extract_trial_properties()
    n_usable = sum(1 for _, tr in T.df['Trial'].items()
                   if tr is not None and not getattr(tr, 'excluded', False))
    sta, blank, binfo = ((None, (0.0, 0.0), {}) if not with_vm else
                         measure_blank_window(T, n_trials=sta_trials,
                                              verbose=verbose))
    if verbose and with_vm:
        print(f'   blank window: -{blank[0]*1e3:.2f} / +{blank[1]*1e3:.2f} ms'
              + (f' (STA n={sta["n_spikes"]:,})' if sta else ' (no STA — defaults)'))

    frames, n_no_spikes = [], 0
    try:
        for tn in list(T.df.index):
            trial = T.df.at[tn, 'Trial']
            if trial is None or getattr(trial, 'excluded', False):
                continue
            # A trial whose detection ran and found nothing is a real,
            # analysable trial on a quiet cell; one with no detection result is
            # missing data. keep_silent=True keeps the first and still drops
            # the second. Default False preserves the previous behaviour --
            # which discarded roughly half the A4 trials, so flip it and rebuild
            # before quoting any rate for these cells.
            if ephys.trial_spike_times(trial, allow_empty=keep_silent) is None:
                n_no_spikes += 1
                continue
            rec = ba.per_frame_records(
                trial, sigma_s=sigma_s, trim_edges_s=trim_edges_s,
                rate_specs=rate_specs, with_vm=with_vm, vm_control=vm_control,
                vm_specs=vm_specs,
                vm_kwargs={'blank_pre_s': blank[0], 'blank_post_s': blank[1]}
                if with_vm else None,
                pad_trials=ba.neighbour_trials(T, tn) if pad_spikes else None,
                raw_channels=raw_channels, keep_silent=keep_silent,
                **bout_kwargs)
            if rec is not None:
                rec['trial'] = tn
                frames.append(rec)
    finally:
        if drop_table and not keep_table:
            sinq.drop_tables(index=[cell_id])

    out = {'cell_id': cell_id, 'records': None, 'blank': blank, 'sta': sta,
           'blank_info': binfo, 'n_usable': n_usable, 'n_no_spikes': n_no_spikes,
           'table': T if keep_table else None}
    if not frames:
        if verbose:
            print('   (no records)')
        return out
    records = pd.concat(frames, ignore_index=True)
    if add_timing:
        records = ba.add_movement_timing(records)
    out['records'] = records
    if verbose:
        dt = float(np.median(np.diff(records['t'].to_numpy()[:50])))
        print(f'   {len(records):,} frames / {records["trial"].nunique()} trials '
              f'({n_usable} usable, {n_no_spikes} without spikes); '
              f'mean rate {records["n_sp"].sum() / (len(records) * dt):.1f} Hz'
              + (f'; {records["vm_frac_blanked"].mean():.0%} of Vm interpolated'
                 if with_vm and 'vm_frac_blanked' in records else ''))
    return out


def find_demo_trials(records, n=3, hold_min_s=6.0, hold_max_x_ptp=5.0):
    """Trials worth looking at, by category — for spot-checking in the browser.

    ``hold`` a long steady REST bout with a large rate fluctuation (the probe
            barely moves while the rate swings), ``cue+`` / ``cue-`` the largest
            cue responses in each direction, ``premove_up`` / ``premove_dn`` the
            largest pre-movement rate increase before a force-increasing movement
            and decrease before a relaxation.

    A trial appearing under both ``hold`` and ``premove_up`` is the most
    informative kind: a steady hold that ends in a rate rise and a movement
    toward the target.
    """
    rows = []
    bs = ba.bout_rate_spread(records, ['rate'], state=ba.kin.STATE_REST,
                             min_duration_s=1.0)
    if len(bs):
        hold = bs[(bs['duration'] > hold_min_s)
                  & (bs['x_ptp'] < hold_max_x_ptp)].nlargest(n, 'rate_sd')
        rows += [{'kind': 'hold', 'trial': int(t), 'metric': round(float(s), 2)}
                 for t, s in zip(hold['trial'], hold['rate_sd'])]
    if 'in_cue' in records.columns:
        cue_rows = []
        for tn, g in records.groupby('trial'):
            c = g[g['in_cue']]
            c0 = g['cue_t0'].iloc[0]
            pre = g[(g['t'] > c0 - 0.2) & (g['t'] < c0)]
            if len(c) < 5 or len(pre) < 3:
                continue
            cue_rows.append({'trial': tn,
                             'd_rate': float(c['rate'].mean() - pre['rate'].mean())})
        cue = pd.DataFrame(cue_rows)
        if len(cue):
            for kind, sub in (('cue+', cue.nlargest(2, 'd_rate')),
                              ('cue-', cue.nsmallest(2, 'd_rate'))):
                rows += [{'kind': kind, 'trial': int(t), 'metric': round(float(d), 2)}
                         for t, d in zip(sub['trial'], sub['d_rate'])]
    tr = ba.premovement_transitions(records, lead_s=0.2, base_to_s=0.4,
                                    base_from_s=1.0, min_rest_s=1.0)
    if len(tr):
        for kind, sub in (
                ('premove_up', tr[tr['dx_init'] > 0].nlargest(2, 'd_rate')),
                ('premove_dn', tr[tr['dx_init'] < 0].nsmallest(2, 'd_rate'))):
            rows += [{'kind': kind, 'trial': int(t), 'metric': round(float(d), 2)}
                     for t, d in zip(sub['trial'], sub['d_rate'])]
    return pd.DataFrame(rows, columns=['kind', 'trial', 'metric'])


def run_cell_rest_analysis(built, sigmas_s=(0.025, 0.25), premove_s=0.5,
                           min_duration_s=1.0, max_bouts=600, cue_post_s=0.2,
                           fig_dir='./Figure7', save=True, sinq=None,
                           verbose=True, do_cue=True):
    """The REST-state analyses, off records already built.

    Takes the dict from :func:`build_cell_records` (or a bare records frame) and
    produces the kernel comparison with pre-movement stretches coloured by the
    direction of the upcoming movement, the cue-aligned response, the
    pre-movement transition table, and a demo-trial shortlist.

    Deliberately does *not* run the window sweep — see
    :func:`run_cell_window_analysis` for that. The two were one function, which
    meant paying for the sweep whenever you wanted any of this.
    """
    records = built['records'] if isinstance(built, dict) else built
    cell_id = built['cell_id'] if isinstance(built, dict) else '?'
    if records is None or not len(records):
        return None
    if 't_to_move' not in records.columns:
        records = ba.add_movement_timing(records)

    figs = {}
    out = plot_rate_vs_position_kernels(
        records, cell_id, sigmas_s=sigmas_s, min_duration_s=min_duration_s,
        max_bouts=max_bouts, premove_s=premove_s, fig_dir=fig_dir, save=save,
        sinq=sinq)
    spread = None
    if out is not None:
        figs['kernels'], spread = out
    if do_cue:
        try:
            cue = plot_cue_response(records, cell_id, pre_s=0.19, post_s=0.8,
                                    align='onset', baseline_s=0.15,
                                    fig_dir=fig_dir, save=save, sinq=sinq)
            if cue is not None:
                figs['cue'], _ = cue
        except Exception as e:
            if verbose:
                print(f'   cue response failed: {e}')

    premove = ba.premovement_transitions(records, lead_s=0.2, base_to_s=0.4,
                                         base_from_s=1.0, min_rest_s=1.0)
    if verbose and len(premove):
        up = premove[premove['dx_init'] > 0]
        dn = premove[premove['dx_init'] < 0]
        print(f'   premove: {len(premove)} transitions | '
              f'toward n={len(up)} med {up["d_rate"].median():+.2f} Hz '
              f'{100*(up["d_rate"] > 0).mean():.0f}% up | '
              f'away n={len(dn)} med {dn["d_rate"].median():+.2f} Hz '
              f'{100*(dn["d_rate"] > 0).mean():.0f}% up')
    demos = find_demo_trials(records)
    return {'cell_id': cell_id, 'spread': spread, 'premove': premove,
            'demos': demos, 'figs': figs}


def run_cell_window_analysis(built, windows_s=RATE_WINDOWS_S,
                             state=ba.kin.STATE_REST, x_bin_width=10.0,
                             min_per_bin=5, fig_dir='./Figure7', save=True,
                             sinq=None, verbose=True):
    """The window sweep, off records already built — optional and standalone.

    Needs boxcar columns for the display panels only; the statistics come from
    ``n_sp``, so records built without ``rate_specs`` still work.
    """
    records = built['records'] if isinstance(built, dict) else built
    cell_id = built['cell_id'] if isinstance(built, dict) else '?'
    if records is None or not len(records):
        return None
    mean_cols = ('vm',) if 'vm' in records.columns else ()
    windows = window_sweep(records, windows_s=windows_s, state=state,
                           mean_cols=mean_cols)
    if not len(windows):
        if verbose:
            print('   (no REST windows long enough)')
        return None
    spread = conditional_spread(windows, x_bin_width=x_bin_width,
                                min_per_bin=min_per_bin)
    decomp = variance_decomposition(windows)
    if verbose:
        print(spread[['window_s', 'n_windows', 'rate_mean', 'sd_within_bin',
                      'sd_allan', 'sd_poisson', 'sd_excess_lo95', 'rho']]
              .to_string(index=False, float_format='%.2f'))
    figs = {
        'windows': plot_rest_windows_scatter(windows, cell_id, spread=spread,
                                             fig_dir=fig_dir, save=save, sinq=sinq),
        'summary': plot_rest_window_summary(spread, decomp, cell_id,
                                           fig_dir=fig_dir, save=save, sinq=sinq),
    }
    return {'cell_id': cell_id, 'windows': windows, 'spread': spread,
            'decomp': decomp, 'figs': figs}


def run_cell_rest_window_analysis(sinq, cell_id, windows_s=RATE_WINDOWS_S,
                                  with_vm=True, fig_dir='./Figure7',
                                  state=ba.kin.STATE_REST, x_bin_width=10.0,
                                  min_per_bin=5, sta_trials=10, save=True,
                                  drop_table=True, verbose=True,
                                  vm_control=True, **bout_kwargs):
    """Build records, run the window sweep, and make the Vm figures.

    Kept as the original one-call entry point, but now a thin composition of
    :func:`build_cell_records` + :func:`run_cell_window_analysis` + the Vm panels.
    Prefer calling those separately: the gathering step is the expensive one, and
    the window sweep is optional.

    Returns a dict with ``records``, ``windows``, ``spread``, ``decomp``, ``sta``,
    ``blank``, ``vm_stats``, ``figs`` — or ``None`` if the cell yielded nothing.
    """
    built = build_cell_records(
        sinq, cell_id, with_vm=with_vm, vm_control=vm_control,
        rate_specs=boxcar_rate_specs(windows_s),
        # The window sweep wants the historical behaviour: no spike padding, and
        # the edge trim that goes with it.
        pad_spikes=False, trim_edges_s=0.075, sta_trials=sta_trials,
        drop_table=drop_table, verbose=verbose, **bout_kwargs)
    records, blank, sta = built['records'], built['blank'], built['sta']
    if records is None:
        return None

    win = run_cell_window_analysis(built, windows_s=windows_s, state=state,
                                   x_bin_width=x_bin_width,
                                   min_per_bin=min_per_bin, fig_dir=fig_dir,
                                   save=save, sinq=sinq, verbose=verbose)
    if win is None:
        return None
    figs = dict(win['figs'])

    vm_stats = {}
    if with_vm:
        figs['sta'] = plot_spike_sta(sta, blank, cell_id, fig_dir=fig_dir,
                                     save=save, sinq=sinq)
        figs['vm_rate'] = plot_vm_vs_rate(records, cell_id, fig_dir=fig_dir,
                                         save=save, sinq=sinq)
        figs['vm_rate_centered'] = plot_vm_vs_rate(
            records, cell_id, fig_dir=fig_dir, save=save, sinq=sinq,
            center_per_trial=True)
        figs['vm_rate_noblank'] = plot_vm_vs_rate(
            records, cell_id, fig_dir=fig_dir, save=save, sinq=sinq,
            vm_col='vm_noblank')
        figs['vm_rest'] = plot_vm_vs_position_rest(win['windows'], cell_id,
                                                  fig_dir=fig_dir, save=save,
                                                  sinq=sinq)
        drift = plot_vm_drift(records, cell_id, fig_dir=fig_dir, save=save,
                              sinq=sinq, state=state)
        if drift is not None:
            figs['vm_drift'], vm_stats = drift
            if verbose and vm_stats:
                print(f'   Vm-rate rho: within-trial {vm_stats["rho_within_trial"]:+.3f}  '
                      f'between-trial {vm_stats["rho_between_trial"]:+.3f}  '
                      f'pooled {vm_stats["rho_pooled"]:+.3f}')
                if abs(vm_stats['rho_vm_vs_trial']) > 0.5:
                    print(f'   WARNING: Vm drifts with trial number '
                          f'(rho={vm_stats["rho_vm_vs_trial"]:+.2f} over '
                          f'{vm_stats["vm_range"]:.1f} mV) — the pooled and '
                          f'between-trial values are not interpretable')
    return {'records': records, 'windows': win['windows'],
            'spread': win['spread'], 'decomp': win['decomp'], 'sta': sta,
            'blank': blank, 'vm_stats': vm_stats, 'figs': figs}


def pooled_spread_table(results, value_col='sd_excess_lo95'):
    """One row per cell x window from a dict of ``run_cell_rest_window_analysis``
    results — the across-cell view of whether the excess survives averaging."""
    rows = []
    for cid, res in results.items():
        if res is None:
            continue
        s = res['spread'].copy()
        s.insert(0, 'cell', cid)
        rows.append(s)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# Stim-onset aligned save helpers + driver
# ---------------------------------------------------------------------------

def _save_stim_count_bar(meta, cell_id, fb_labels, fig_dir, name_suffix=''):
    counts = meta['force_bin'].value_counts().reindex(fb_labels).fillna(0).astype(int)
    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(fb_labels)))
    bars = ax.bar(range(len(fb_labels)), counts.values, color=colors)
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width()/2, v, str(v),
                ha='center', va='bottom', fontsize=9)
    ax.set_xticks(range(len(fb_labels)))
    ax.set_xticklabels(fb_labels)
    ax.set_ylabel('# as_off trials')
    ax.set_title(f'{cell_id}: as_off trials by initial force regime (n={int(counts.sum())})')
    fig.tight_layout()
    fig.savefig(fig_dir / f'stim_counts_{cell_id}{name_suffix}.svg')
    plt.close(fig)


def _save_stim_marginal(t_rel, segs, cell_id, fb_labels,
                        ylabel, title_tag, fname_tag, fig_dir,
                        min_trials=3, name_suffix=''):
    if not segs:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(fb_labels)))
    for color, fb in zip(colors, fb_labels):
        sub = [s for fb_s, s in segs if fb_s == fb]
        if len(sub) < min_trials:
            continue
        arr = np.stack(sub)
        mean = arr.mean(axis=0)
        sem = arr.std(axis=0) / np.sqrt(arr.shape[0])
        ax.plot(t_rel, mean, color=color, label=f'{fb} (n={len(sub)})')
        ax.fill_between(t_rel, mean - sem, mean + sem, color=color, alpha=0.2)
    ax.axvline(0, color='k', lw=0.5)
    ax.set_xlabel('time from stim onset (s)')
    ax.set_ylabel(ylabel)
    ax.set_title(f'{cell_id}: {title_tag} (as_off, mean ± SEM)')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / f'stim_{fname_tag}_{cell_id}{name_suffix}.svg')
    plt.close(fig)


def run_cell_stim_analysis(sinq, cell_id, fig_dir='./Figure7',
                           pre_window_s=0.5):
    """Stim-onset aligned analysis for one cell: count bar + rate/position
    means by initial force regime. Returns the per-trial meta DataFrame.
    """
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(exist_ok=True)

    T = sinq.restore_table(dayflycell=cell_id)
    T.exclude_trials()
    T.extract_trial_properties()

    meta = collect_trial_meta_as_off(T, pre_window_s=pre_window_s)
    if meta.empty:
        print(f'   (skipping {cell_id} — no as_off trials)')
        return None
    edges, fb_labels = force_bin_edges_3(T)
    meta['force_bin'] = pd.cut(meta['init_pos'], bins=edges, labels=fb_labels)
    print(f'   {len(meta)} as_off trials  |  ' +
          ', '.join(f'{fb}={int((meta.force_bin == fb).sum())}' for fb in fb_labels))

    sfx = tag_suffix(sinq, cell_id)
    _save_stim_count_bar(meta, cell_id, fb_labels, fig_dir, name_suffix=sfx)
    t_rel_r, segs_r = stim_aligned_segments(T, meta, signal='rate',
                                            pre_s=STIM_PRE_S, post_s=STIM_POST_S)
    _save_stim_marginal(t_rel_r, segs_r, cell_id, fb_labels,
                        ylabel='rate (Hz)',
                        title_tag='stim-aligned firing rate',
                        fname_tag='rate', fig_dir=fig_dir, name_suffix=sfx)
    t_rel_x, segs_x = stim_aligned_segments(T, meta, signal='position',
                                            pre_s=STIM_PRE_S, post_s=STIM_POST_S)
    _save_stim_marginal(t_rel_x, segs_x, cell_id, fb_labels,
                        ylabel='probe position (um)',
                        title_tag='stim-aligned probe position',
                        fname_tag='position', fig_dir=fig_dir, name_suffix=sfx)
    return meta


# ---------------------------------------------------------------------------
# Lead / lag panels
# ---------------------------------------------------------------------------

#: One colour per condition, kept consistent between the CCF and latency panels.
COND_COLORS = {'cue': '#2ca02c', 'premove': '#ff7f0e',
               'premove_toward': '#ff7f0e', 'premove_away': '#1f77b4',
               'rest': '#7f7f7f', 'move': '#d62728',
               'current_step': '#9467bd'}


def plot_ccf_by_condition(ccfs, cell_id, fig_dir='./Figure7', save=True,
                          sinq=None, order=('cue', 'current_step', 'premove', 'rest', 'move'),
                          xlim_ms=(-500, 500), fname_tag='ccf_by_condition'):
    """Rate-vs-position cross-correlation, one panel per condition.

    Positive lag = **rate leads position**. Read the panels as a calibration
    before an answer: ``cue`` should trough at a *negative* lag (the piezo moves
    the probe first, and flexion hyperpolarizes a flexor, so the correlation is
    negative); ``premove`` should peak at a *positive* lag; ``rest`` should show
    nothing at short lags, because a probe that is not moving cannot be driven by
    a fast rate fluctuation. Only if those three come out as expected does the
    ``move`` panel mean anything.

    The grey band is the circular-rotation null (5-95%), which preserves both
    signals' autocorrelation — a CCF between two smooth signals is wide and
    non-zero by construction, so the band, not the peak height, is the reference.

    ``lag_cv`` is annotated alongside ``lag_peak``: the cross-validated value
    picks the lag on training trials and scores it on held-out ones, so it is not
    inflated by choosing the peak and reporting its own height.
    """
    names = [n for n in order if n in ccfs] + \
            [n for n in ccfs if n not in order]
    # Report every condition, including ones with no usable run. Filtering them
    # out silently here is what hid the rejected calibration conditions before.
    for n in [n for n in names if ccfs[n].get('ccf') is None]:
        d = ccfs[n]
        print(f'  WARNING [{n}] no usable runs: {d["n_too_short"]} were shorter '
              f'than {d.get("min_run_frames", "?")} frames (lag window '
              f'{d.get("lag_window_s", float("nan"))*1e3:.0f} ms) - shorten '
              f'lag_window_s or min_run_factor for this condition')
    names = [n for n in names if ccfs[n].get('ccf') is not None]
    if not names:
        return None
    # One shared x-axis so the conditions are directly comparable. The lag windows
    # differ per condition (short epochs cannot support a wide window), so a
    # condition's trace simply stops earlier - which shows that difference honestly
    # instead of hiding it behind per-panel rescaling.
    fig, axes = plt.subplots(1, len(names), figsize=(3.5 * len(names), 3.6),
                             squeeze=False, sharey=True, sharex=True)
    axes = axes[0]
    rows = []
    for ax, name in zip(axes, names):
        d = ccfs[name]
        lags, c = d['lags_s'] * 1e3, d['ccf']
        color = COND_COLORS.get(name, '#333333')
        if 'null_lo' in d:
            ax.fill_between(lags, d['null_lo'], d['null_hi'], color='0.8',
                            alpha=0.7, lw=0, label='rotation null 5-95%')
        ax.plot(lags, c, '-', color=color, lw=1.8)
        ax.axvline(0, color='k', lw=0.8, ls=':')
        ax.axhline(0, color='k', lw=0.6)
        lag_pk = d.get('lag_peak', np.nan)
        ax.plot([lag_pk * 1e3], [d.get('r_peak', np.nan)], 'v', color=color, ms=6)
        txt = (f"peak {lag_pk*1e3:+.0f} ms  r={d.get('r_peak', np.nan):+.2f}")
        if 'lag_cv' in d:
            txt += f"\nCV {d['lag_cv']*1e3:+.0f} ms  r={d['r_cv']:+.2f}"
        txt += f"\n{d['n_runs']} runs / {d['n_trials']} trials"
        if 'x_ptp_median' in d:
            # r is bounded by how far the probe actually moved. During rest and
            # the run-up to a movement it is nearly still, so a low r there is
            # expected and says nothing about whether the peak lag is real.
            txt += (f"\nprobe ptp {d['x_ptp_median']:.1f} um"
                    f"  (sd {d['x_sd_median']:.2f})")
        if d['n_too_short']:
            txt += f"\n{d['n_too_short']} runs too short"
        ax.annotate(txt, xy=(0.03, 0.03), xycoords='axes fraction', fontsize=7,
                    va='bottom', ha='left')
        exp = ba.CCF_EXPECTATION.get(name.split('_')[0] if name.startswith('premove')
                                     else name, {})
        ax.set_title(f'{name}\n' + (f"expect: {exp.get('lag','')}" if exp else ''),
                     fontsize=8.5)
        ax.set_xlabel('lag (ms)   +ve = rate leads')
        rows.append({'condition': name, 'lag_peak_ms': lag_pk * 1e3,
                     'r_peak': d.get('r_peak', np.nan),
                     'lag_cv_ms': d.get('lag_cv', np.nan) * 1e3,
                     'r_cv': d.get('r_cv', np.nan), 'sign': d.get('sign', np.nan),
                     'n_runs': d['n_runs'], 'n_trials': d['n_trials'],
                     'n_too_short': d['n_too_short']})
    axes[0].set_ylabel('correlation (rate vs position)')
    axes[0].set_xlim(*xlim_ms)
    axes[0].legend(fontsize=6.5, loc='upper right', frameon=False)
    fig.suptitle(f'{cell_id} — rate vs position lead/lag by condition', fontsize=10)
    fig.tight_layout()
    if save:
        fig_dir = Path(fig_dir)
        fig_dir.mkdir(exist_ok=True)
        fig.savefig(fig_dir / f'{fname_tag}_{cell_id}{tag_suffix(sinq, cell_id)}.svg')
    return fig, pd.DataFrame(rows)


def plot_event_latency(latencies, cell_id, fig_dir='./Figure7', save=True,
                       sinq=None, bins=35, xlim=(-400, 400),
                       fname_tag='event_latency'):
    """Per-event rate-vs-position latency, one histogram per condition.

    ``latencies`` is ``{condition: DataFrame}`` from
    :func:`mapd.bout_analysis.event_latency`. Positive = the rate moved first.

    This is the check on the CCF rather than a restatement of it: it uses no lag
    grid and produces one number per event, so if the two disagree the CCF peak
    was not measuring what it appeared to.
    """
    names = [n for n, d in latencies.items() if d is not None and len(d)]
    if not names:
        return None
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    edges = np.linspace(xlim[0], xlim[1], bins + 1)
    rows = []
    for name in names:
        d = latencies[name]
        v = d['latency_ms'].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
        if not len(v):
            continue
        color = COND_COLORS.get(name, '#333333')
        ax.hist(np.clip(v, edges[0], edges[-1]), bins=edges, histtype='step',
                lw=1.8, color=color, density=True,
                label=f'{name}  n={len(v)}  median {np.median(v):+.0f} ms')
        rows.append({'condition': name, 'n': len(v),
                     'median_ms': float(np.median(v)),
                     'mean_ms': float(np.mean(v)),
                     'frac_rate_first': float((v > 0).mean())})
    ax.axvline(0, color='k', lw=0.9)
    ax.set_xlabel('latency (ms):  position crossing - rate crossing\n'
                  '+ve = rate moved first')
    ax.set_ylabel('density')
    ax.legend(fontsize=7.5, frameon=False)
    fig.suptitle(f'{cell_id} — per-event rate/position latency', fontsize=10)
    fig.tight_layout()
    if save:
        fig_dir = Path(fig_dir)
        fig_dir.mkdir(exist_ok=True)
        fig.savefig(fig_dir / f'{fname_tag}_{cell_id}{tag_suffix(sinq, cell_id)}.svg')
    return fig, pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Lead/lag: onset latency (calibrated) + CCF (descriptive)
# ---------------------------------------------------------------------------
# Lifted verbatim out of op_cond_Figure7_MN_spike_rate_vs_position.ipynb, where
# it was defined inline, so the A4 notebook runs the SAME code rather than a
# copy that can drift. The A2 notebook still defines its own copy in-cell and is
# unaffected by this; delete that cell's definitions when convenient.
#
# The calibration conditions and the reason each one is here are in
# ``batch_lead_lag``'s own comments. The headline caveat, which governs how any
# of this may be reported: the ONSET LATENCY is the estimator. A CCF peak
# measures how similar two traces are at a shift, not which one started first,
# and on this data it gives the wrong sign even at the cue, where the piezo
# demonstrably moves first. Report ``lat_tbl``; treat ``ccf_tbl`` as descriptive.

LAG_WINDOWS = {'cue': 0.10, 'premove': 0.10, 'rest': 0.50, 'move': 0.40}

LAT_SPECS = (
    # name           events   x_col  pre   post   base_from base_to  dRate  dX
    ('cue',      'cue', 'cue_cmd', 0.10, 0.15,  0.10,     0.0,     2.0,   1.0),
    ('current_step', 'step',  'x',   0.0,  0.085, -0.09,    -0.17,   2.0,   1.5),
    ('premove',      'move',  'x',   0.50, 0.50,  1.00,     0.50,    3.0,   4.0),
)
# NOTE on the two negative values above: event_latency builds its baseline as
# [t0 - base_from_s, t0 - base_to_s), so negative arguments place the window
# AFTER t0. The current step needs that -- there are only ~10 ms of recording
# before it -- and getting the signs wrong yields an empty window, which skips
# every event and looks exactly like a control that passed.


def step_response(rec, window=None, base_s=(-0.90, -0.82)):
    """Signal excursions at the current step, against a post-recovery baseline."""
    lo, hi = ba.CURRENT_STEP_WINDOW if window is None else window
    rows = []
    for tn, g in rec.groupby('trial', sort=True):
        t = g['t'].to_numpy(dtype=float)
        inw = (t >= lo) & (t <= hi)
        inb = (t >= base_s[0]) & (t <= base_s[1])
        if inw.sum() < 5 or inb.sum() < 5:
            continue
        row = {'trial': tn}
        for col in ('rate', 'x', 'vm'):
            if col not in g.columns:
                continue
            y = g[col].to_numpy(dtype=float)
            seg = y[inw] - np.nanmean(y[inb])
            if np.isfinite(seg).any():
                row[f'd_{col}'] = float(seg[int(np.nanargmax(np.abs(seg)))])
        rows.append(row)
    return pd.DataFrame(rows)


def batch_lead_lag(rest_results, cell_list, sinq=None, save=True,
                   n_null=100, verbose=True, fname_tag=None):
    """Onset latencies + CCFs for every cell."""
    lat_rows, ccf_rows, step_rows, per_cell = [], [], [], {}
    for cid in cell_list:
        built = rest_results.get(cid)
        if built is None or built.get('records') is None:
            continue
        if verbose:
            print(f'-- {cid} --', flush=True)
        rec = built['records']
        if 't_to_move' not in rec.columns:
            rec = ba.add_movement_timing(rec)
        if 'cue_cmd' not in rec.columns:
            rec = ba.add_cue_command(rec)
        events = {'cue': ba.cue_events(rec),
                  'step': ba.current_step_events(rec),
                  'move': ba.movement_onset_events(rec, min_move_ptp=8.0)}

        # Coverage, printed before any latency. A column can exist and still be
        # unfilled -- and "no finite samples in the window" is not the same
        # finding as "no lead", so it has to be visible rather than silently
        # producing an empty condition.
        for c in ('rate', 'x', 'cue_cmd', 'sgs'):
            if c in rec.columns:
                f = float(np.isfinite(rec[c].to_numpy(dtype=float)).mean())
                if f < 0.999:
                    print(f'   coverage {c}: {100 * f:.2f}% finite'
                          f'{"  <-- unusable" if f < 0.5 else ""}')

        lats = {}
        for (name, ekey, x_col, pre, post, bf, bt, dr, dx) in LAT_SPECS:
            ev = events[ekey]
            if not len(ev) or x_col not in rec.columns:
                print(f'   {name}: skipped '
                      f'({"no events" if not len(ev) else x_col + " missing"})')
                continue
            d = ba.event_latency(rec, ev, mode='onset', rate_col='rate',
                                 x_col=x_col, pre_s=pre, post_s=post,
                                 base_from_s=bf, base_to_s=bt, n_sd=3.0,
                                 min_d_rate=dr, min_d_x=dx)
            lats[name] = d
            if not len(d):
                print(f'   {name}: 0 of {len(ev)} events timed '
                      f'(check the {x_col} coverage above and the '
                      f'baseline window signs)')
            v = (d['latency_ms'].replace([np.inf, -np.inf], np.nan).dropna()
                 if len(d) else pd.Series(dtype=float))
            lat_rows.append({
                'cell': cid, 'condition': name, 'x_col': x_col,
                'n_events': len(ev), 'n_timed': len(v),
                'median_ms': float(np.median(v)) if len(v) else np.nan,
                'iqr_ms': (float(np.subtract(*np.percentile(v, [75, 25])))
                           if len(v) else np.nan),
                'frac_rate_first': float((v > 0).mean()) if len(v) else np.nan})

        # CCF, descriptive. current_step is left out: its epochs are ~75 ms,
        # far too short to support any lag window.
        conds = ba.condition_masks(rec, cue_post_s=0.2, premove_s=0.5)
        conds.pop('current_step', None)
        ccfs = ba.ccf_by_condition(rec, conditions=conds, rate_col='rate',
                                   signal_col='x', lag_window_s=LAG_WINDOWS,
                                   min_run_factor=1.0, n_null=n_null)
        for name, d in ccfs.items():
            if d.get('ccf') is None:
                continue
            ccf_rows.append({'cell': cid, 'condition': name,
                             'lag_peak_ms': d.get('lag_peak', np.nan) * 1e3,
                             'r_peak': d.get('r_peak', np.nan),
                             'lag_cv_ms': d.get('lag_cv', np.nan) * 1e3,
                             'r_cv': d.get('r_cv', np.nan),
                             'n_runs': d['n_runs'],
                             'n_too_short': d['n_too_short']})

        sr = step_response(rec)
        if len(sr):
            row = {'cell': cid, 'n_trials': len(sr)}
            for c in ('d_rate', 'd_x', 'd_vm'):
                if c in sr.columns:
                    row[f'median_{c}'] = float(sr[c].median())
            step_rows.append(row)

        per_cell[cid] = {'lats': lats, 'ccfs': ccfs, 'step': sr}
        if save:
            o = plot_ccf_by_condition(
                ccfs, cid, sinq=sinq,
                fname_tag=f'{fname_tag}_ccf' if fname_tag else 'ccf_batch')
            if o:
                plt.close(o[0])
            o = plot_event_latency(lats, cid, sinq=sinq,
                                   fname_tag=(f'{fname_tag}_latency'
                                              if fname_tag else 'latency_batch'))
            if o:
                plt.close(o[0])
    return (pd.DataFrame(lat_rows), pd.DataFrame(ccf_rows),
            pd.DataFrame(step_rows), per_cell)


def bout_vm_spread(rest_results, cells, vm_cols=('vm', 'vm_noblank'),
                   state=None, min_duration_s=1.0, cue_post_s=0.2,
                   max_x_ptp=None, group=None):
    """Per-REST-bout membrane-potential scatter, for a set of cells.

    Written to be run identically on the A2 and A4 sets so the two are
    comparable: same bout definition, same exclusions, same columns. The cue,
    its off-response and the injected pre-cue current step are removed first --
    all three are imposed Vm displacements and would otherwise dominate the
    within-bout scatter they are being measured against.

    THE CONFOUND THIS FUNCTION CANNOT REMOVE, and the reason ``vm_noblank`` is
    returned alongside ``vm``: ``vm`` is spike-blanked, i.e. each spike is cut
    out and interpolated across. A2 cells fire continuously -- 40% of Vm samples
    are interpolated on 210915_F1_C1 -- while A4 cells barely fire at all, so
    their ``vm`` is nearly untouched. Blanking removes the fastest excursions,
    which biases the blanked sd DOWNWARD for the spiking set specifically.
    Comparing A2's blanked Vm with A4's is therefore biased in favour of "A4
    fluctuates more". ``vm_noblank`` has the opposite bias -- it leaves the
    spikes in, inflating A2 -- so a difference that holds in BOTH is not an
    artifact of either. Report both or neither.

    Returns one row per bout per column, with ``frac_blanked`` carried along so
    the size of that bias is visible per cell.
    """
    state = ba.kin.STATE_REST if state is None else state
    out = []
    for cid in cells:
        built = rest_results.get(cid)
        if built is None or built.get('records') is None:
            continue
        rec = built['records']
        cols = [c for c in vm_cols if c in rec.columns]
        if not cols:
            continue
        drop = ba.cue_mask(rec, post_s=cue_post_s) | ba.current_step_mask(rec)
        rec = rec.copy()
        for c in cols:
            v = rec[c].to_numpy(dtype=float).copy()
            v[drop] = np.nan          # bout_rate_spread drops non-finite samples
            rec[c] = v
        sp = ba.bout_rate_spread(rec, cols, state=state,
                                 min_duration_s=min_duration_s)
        if not len(sp):
            continue
        if max_x_ptp is not None:
            sp = sp[sp['x_ptp'] <= max_x_ptp]
        sp = sp.assign(
            cell=cid,
            group=group,
            frac_blanked=(float(np.nanmean(rec['vm_frac_blanked']))
                          if 'vm_frac_blanked' in rec.columns else np.nan))
        out.append(sp)
    if not out:
        return pd.DataFrame()
    d = pd.concat(out, ignore_index=True)
    # name the columns for what they now hold: millivolts, not hertz
    return d.rename(columns={'rate_col': 'vm_col', 'rate_mean': 'vm_mean',
                             'rate_sd': 'vm_sd', 'rate_ptp': 'vm_ptp',
                             'rate_iqr': 'vm_iqr'})


def records_in_trial_frame(records, trial, trial_number, x_col='x', tol=0.5):
    """One trial's records with ``x`` recomputed from the Trial's CURRENT probeZero.

    Guards a mismatch that is invisible until it is drawn. ``x`` is baked into
    the records at build time as ``probeZero - probe_position``; anything read
    live from the Trial afterwards -- the target band above all -- uses whatever
    probeZero says NOW. If probeZero changed in between, the trace and the band
    end up in different frames and the figure is wrong while both halves are
    individually correct. That happened here: 241115_F1_C1's records were built
    on the conventional 630 while target_bounds returned the re-measured 549.5,
    putting the band 17 px below a trace it should have overlapped.

    Returns ``(rows, offset_px)``. ``offset_px`` is what the stored ``x`` was
    carrying relative to the trial's current frame, so 0.0 means the records are
    already consistent and nothing was changed. Non-zero means the records are
    stale for this cell and everything else computed from them -- absolute
    positions, shared axes, target-relative measures -- is off by that much.
    """
    rows = records[records['trial'] == trial_number]
    if not len(rows):
        return rows, np.nan
    try:
        ds = trial.downsample_probe
        raw = np.asarray(trial.probe_position).squeeze()[ds]
        t_tr = np.asarray(trial.time)[ds].squeeze()
        pz = float(np.asarray(trial.probeZero).squeeze())
    except (AttributeError, KeyError, TypeError, ValueError):
        return rows, np.nan
    if not np.isfinite(pz):
        return rows, np.nan
    x_now = np.interp(rows['t'].to_numpy(dtype=float), t_tr, pz - raw)
    offset = float(np.nanmedian(rows[x_col].to_numpy(dtype=float) - x_now))
    if not np.isfinite(offset) or abs(offset) <= tol:
        return rows, 0.0
    return rows.assign(**{x_col: x_now}), offset
