"""Scratch: state-set-aware version of plot_in_target_durations.

Paste into op_cond_Figure5_mutant_explanations.ipynb, or import the functions
from a terminal (see debug_plot.py). Expects runs/trials frames from
ba.collect_in_target_moves_sinq(..., state_sets=(kin.STATE_MOVE, ba.STATES_ACTIVE)).

The calls at the bottom sit under `if __name__ == '__main__':`, which is True in
a notebook cell but False on import -- so importing this module gives you the
functions without running anything, and pasting it still runs the figures.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Distribution of in-target run durations (MOVE, or DRIFT+MOVE)
# ---------------------------------------------------------------------------
C_MAIN, C_ALT, C_GRAY, C_TIME = '#1f6f9e', '#b3d4e8', '#9e9e9e', '#c0504d'
C_DRIFT = '#e8a33d'

# Non-task trials have a nominal target but no stimulus -- drop them.
NON_TASK_OUTCOMES = ('rest', 'info')


def _split_alt(runs, alt_dfcs=None, alt_marker='TeTx'):
    """Flies to draw separately. Defaults to any genotype carrying alt_marker,
    so the single +;iav TeTx fly does not sit inside the Kir2.1 pooled curve.
    """
    if alt_dfcs is not None:
        return set(alt_dfcs)
    if 'genotype' not in runs.columns:
        return set()
    hit = runs['genotype'].fillna('').str.contains(alt_marker)
    return set(runs.loc[hit, 'dfc'].unique())


def _ecdf(v):
    v = np.sort(np.asarray(v, dtype=float))
    if not len(v):
        return v, v
    return v, np.arange(1, len(v) + 1) / len(v)


def _pick_state_set(df, state_set, what):
    """One state set's rows. Refuses to pool two sets by accident."""
    if 'state_set' not in df.columns:
        return df, 'move'
    have = list(pd.unique(df['state_set'].dropna()))
    if state_set is None:
        if len(have) > 1:
            raise ValueError(
                f'{what} carries {len(have)} state sets {have}; pass '
                f'state_set= to choose one, else the panels pool them.')
        return df, (have[0] if have else 'move')
    out = df[df['state_set'] == state_set]
    if out.empty:
        raise ValueError(f'no {what} rows with state_set={state_set!r}; have {have}')
    return out, state_set


def plot_in_target_durations(runs, trials, group_name='+;iav',
                             state_set=None, compare_state_set=None,
                             exclude_outcomes=NON_TASK_OUTCOMES,
                             alt_dfcs=None, alt_label='TeTx', savepath=None):
    """Four-panel view of how long in-target runs last.

    state_set          which set to draw ('move', 'drift+move'). Required when
                       the frames carry both; the labels follow it.
    compare_state_set  optional second set, drawn as a dashed overlay in a-c so
                       the with-/without-drift versions sit on one axis.

    a  per-fly ECDF of run durations (log x), pooled curve in bold
    b  the same pooled data weighted by count vs by time -- these point in
       opposite directions, which is the whole story
    c  per-fly median + IQR, ordered
    d  in-target time split into MOVE and DRIFT (same fly order as c). This
       panel is state-set independent -- the components come from
       in_target_move_only_time / in_target_drift_only_time -- so it answers
       'with or without drift' directly rather than once per set.

    exclude_outcomes is applied to BOTH frames, so panel d's denominator
    matches the runs in panels a-c.
    """
    if exclude_outcomes and 'as_outcome' in runs.columns:
        runs = runs[~runs['as_outcome'].isin(exclude_outcomes)]
        trials = trials[~trials['as_outcome'].isin(exclude_outcomes)]

    all_runs = runs                      # keep both sets for the overlay
    runs, label = _pick_state_set(runs, state_set, 'runs')
    trials, _ = _pick_state_set(trials, state_set, 'trials')
    cmp_runs = cmp_label = None
    if compare_state_set:
        cmp_runs, cmp_label = _pick_state_set(all_runs, compare_state_set, 'runs')

    alt = _split_alt(runs, alt_dfcs)
    dt = float(trials['dt'].median())
    what = f'in-target {label.upper()} run'

    fig, ((axA, axB), (axC, axD)) = plt.subplots(2, 2, figsize=(10, 7.2), dpi=150)

    # (a) per-fly ECDFs + pooled
    for dfc, g in runs.groupby('dfc'):
        if dfc in alt:
            axA.step(*_ecdf(g['duration']), where='post', color=C_ALT, lw=1.6,
                     ls='--', zorder=3, label=f'{dfc} ({alt_label})')
        else:
            axA.step(*_ecdf(g['duration']), where='post', color=C_GRAY,
                     lw=0.8, alpha=0.8)
    pooled = runs.loc[~runs['dfc'].isin(alt), 'duration']
    axA.step(*_ecdf(pooled), where='post', color=C_MAIN, lw=2.2, zorder=4,
             label=f'{group_name} {label} (n={len(pooled)})')
    if cmp_runs is not None:
        cmp_pooled = cmp_runs.loc[~cmp_runs['dfc'].isin(alt), 'duration']
        axA.step(*_ecdf(cmp_pooled), where='post', color=C_TIME, lw=2.0,
                 ls='--', zorder=4,
                 label=f'{group_name} {cmp_label} (n={len(cmp_pooled)})')
    axA.axvline(dt, color='k', lw=0.8, ls=':', zorder=1)
    axA.annotate(f'1 frame ({dt*1e3:.0f} ms)', xy=(dt * 1.15, 0.02),
                 fontsize=8, color='#555')
    axA.set_xscale('log')
    axA.set_xlim(dt * 0.8, 12)
    axA.set_ylim(0, 1)
    axA.set_xlabel(f'{what} duration (s)')
    axA.set_ylabel('cumulative fraction of runs')
    axA.set_title('a  per-fly ECDF (thin grey = one fly)', loc='left', fontsize=11)
    axA.legend(frameon=False, fontsize=8, loc='lower right')

    # (b) count-weighted vs time-weighted
    edges = np.logspace(np.log10(dt * 0.8), np.log10(12), 22)
    ctr = np.sqrt(edges[:-1] * edges[1:])
    d = pooled.to_numpy()
    n_h, _ = np.histogram(d, bins=edges)
    t_h, _ = np.histogram(d, bins=edges, weights=d)
    axB.step(ctr, n_h / n_h.sum(), where='mid', color=C_MAIN, lw=2,
             label='fraction of runs')
    axB.step(ctr, t_h / t_h.sum(), where='mid', color=C_TIME, lw=2,
             label=f'fraction of in-target {label} time')
    if cmp_runs is not None:
        dc = cmp_pooled.to_numpy()
        tc, _ = np.histogram(dc, bins=edges, weights=dc)
        axB.step(ctr, tc / tc.sum(), where='mid', color=C_TIME, lw=1.2, ls='--',
                 label=f'fraction of in-target {cmp_label} time')
    axB.set_xscale('log')
    axB.set_xlim(dt * 0.8, 12)
    axB.set_xlabel(f'{what} duration (s)')
    axB.set_ylabel('fraction')
    axB.set_title('b  many brief runs, but the time sits in the long ones',
                  loc='left', fontsize=11)
    axB.legend(frameon=False, fontsize=8)

    # (c) per-fly median + IQR
    g = runs.groupby('dfc')['duration']
    stats = pd.DataFrame({'med': g.median(), 'q1': g.quantile(.25),
                          'q3': g.quantile(.75), 'n': g.size()}).sort_values('med')
    stats['state_set'] = label
    y = np.arange(len(stats))
    cols = [C_ALT if d_ in alt else C_MAIN for d_ in stats.index]
    axC.hlines(y, stats['q1'], stats['q3'], color=cols, lw=2.5, alpha=0.55)
    axC.scatter(stats['med'], y, s=34, color=cols, zorder=3, label=label)
    if cmp_runs is not None:
        cmp_med = cmp_runs.groupby('dfc')['duration'].median().reindex(stats.index)
        axC.scatter(cmp_med.values, y, s=30, facecolors='none',
                    edgecolors=C_TIME, zorder=4, label=cmp_label)
        axC.legend(frameon=False, fontsize=8, loc='upper left')
    axC.set_yticks(y)
    axC.set_yticklabels([f'{d_}{"  (" + alt_label + ")" if d_ in alt else ""}'
                         for d_ in stats.index], fontsize=8)
    axC.axvline(dt, color='k', lw=0.8, ls=':')
    axC.set_xscale('log')
    axC.set_xlim(dt * 0.8, 13)
    axC.set_xlabel('run duration (s) -- median, IQR')
    axC.set_title(f'c  per-fly median {what}', loc='left', fontsize=11)
    for yy, n in zip(y, stats['n']):
        axC.annotate(f'n={int(n)}', xy=(11.5, yy), fontsize=7, color='#666',
                     va='center', ha='right')

    # (d) in-target time split into MOVE and DRIFT
    tg = trials.groupby('dfc')
    denom = tg['in_target_time'].sum()
    if 'in_target_move_only_time' in trials.columns:
        f_move = (tg['in_target_move_only_time'].sum() / denom).reindex(stats.index)
        f_drift = (tg['in_target_drift_only_time'].sum() / denom).reindex(stats.index)
    else:                      # older cache: only the requested set is available
        f_move = (tg['in_target_move_time'].sum() / denom).reindex(stats.index)
        f_drift = pd.Series(0.0, index=stats.index)
    hatches = ['//' if d_ in alt else None for d_ in stats.index]
    axD.barh(y, f_move.values, color=C_MAIN, height=0.62, label='move')
    axD.barh(y, f_drift.values, left=f_move.values, color=C_DRIFT, height=0.62,
             label='drift')
    for bars in (axD.containers[0], axD.containers[1]):
        for bar, h in zip(bars, hatches):
            bar.set_hatch(h)
    axD.set_yticks(y)
    axD.set_yticklabels([])
    axD.set_xlim(0, 1)
    axD.set_xlabel('fraction of in-target time')
    axD.set_title('d  in-target time: move vs drift', loc='left', fontsize=11)
    axD.legend(frameon=False, fontsize=8, loc='lower right')
    for yy, m, dr in zip(y, f_move.values, f_drift.values):
        axD.annotate(f'{m:.2f}+{dr:.2f}', xy=(min(m + dr + 0.02, 0.82), yy),
                     fontsize=7, color='#444', va='center')

    for ax in (axA, axB, axC, axD):
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle(f'{group_name}: duration of {label} states inside the current target',
                 x=0.01, ha='left', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    if savepath:
        fig.savefig(savepath, bbox_inches='tight')
        print('wrote', savepath)
    return fig, stats


# ---------------------------------------------------------------------------
# Whole-experiment time budget
# ---------------------------------------------------------------------------

def state_time_budget(trials, by='dfc', exclude_outcomes=NON_TASK_OUTCOMES,
                      move_set='move', active_set='drift+move'):
    """Total move / drift / in-target time per fly, from the trials frame.

    Needs BOTH state sets in ``trials`` (collect with
    ``state_sets=(kin.STATE_MOVE, ba.STATES_ACTIVE)``), because the whole-trace
    totals are the ``move_time`` column read from each set:

        move_time   (move rows)        total time in MOVE, anywhere
        move_time   (drift+move rows)  total time in MOVE or DRIFT, anywhere
        difference                     total time in DRIFT, anywhere

    The in-target components (``in_target_move_only_time`` /
    ``in_target_drift_only_time``) are state-set independent, so they are read
    from the move rows only -- summing across both sets would double count.

    ``by=None`` collapses to a single row for the whole group.

    Ratio columns:

        in_target_active_per_move    in-target (drift+move) / total MOVE
                                     -- can exceed 1, since the numerator counts
                                     drift that the denominator does not
        in_target_active_per_active  in-target (drift+move) / total (drift+move)
        in_target_move_per_move      in-target MOVE / total MOVE
        active_per_window            total (drift+move) / window
        in_target_per_window         in-target time / window
    """
    if exclude_outcomes and 'as_outcome' in trials.columns:
        trials = trials[~trials['as_outcome'].isin(exclude_outcomes)]
    if 'state_set' not in trials.columns:
        raise ValueError('trials has no state_set column -- recollect with '
                         'state_sets=(kin.STATE_MOVE, ba.STATES_ACTIVE)')
    have = list(pd.unique(trials['state_set'].dropna()))
    for s in (move_set, active_set):
        if s not in have:
            raise ValueError(f'trials lacks state_set={s!r}; have {have}')

    m = trials[trials['state_set'] == move_set]
    a = trials[trials['state_set'] == active_set]
    if by is None:
        gm = m.groupby(pd.Series('all', index=m.index), observed=True)
        ga = a.groupby(pd.Series('all', index=a.index), observed=True)
    else:
        gm, ga = m.groupby(by, observed=True), a.groupby(by, observed=True)

    out = pd.DataFrame({
        'n_trials':         gm.size(),
        'window_time':      gm['window_time'].sum(),
        'in_target_time':   gm['in_target_time'].sum(),
        'move_time':        gm['move_time'].sum(),
        'active_time':      ga['move_time'].sum(),
        'in_target_move':   gm['in_target_move_only_time'].sum(),
        'in_target_drift':  gm['in_target_drift_only_time'].sum(),
    })
    out['drift_time'] = out['active_time'] - out['move_time']
    out['in_target_active'] = out['in_target_move'] + out['in_target_drift']

    # consistency: the active pass's own in-target total must agree
    check = ga['in_target_move_time'].sum()
    bad = ((out['in_target_active'] - check).abs()
           > 1e-6 * out['in_target_active'].clip(lower=1))
    if bad.any():
        print(f'warning: in-target active time disagrees with the {active_set} '
              f'pass for {int(bad.sum())} group(s) -- max diff '
              f'{float((out["in_target_active"] - check).abs().max()):.4g} s')

    out['in_target_active_per_move'] = out['in_target_active'] / out['move_time']
    out['in_target_active_per_active'] = out['in_target_active'] / out['active_time']
    out['in_target_move_per_move'] = out['in_target_move'] / out['move_time']
    out['active_per_window'] = out['active_time'] / out['window_time']
    out['in_target_per_window'] = out['in_target_time'] / out['window_time']
    return out


# ---------------------------------------------------------------------------
# Time-weighted ECDF (the cumulative version of panel b)
# ---------------------------------------------------------------------------

def _weighted_ecdf(dur, w=None):
    """(sorted durations, cumulative fraction of weight at or below each)."""
    dur = np.asarray(dur, dtype=float)
    w = np.ones_like(dur) if w is None else np.asarray(w, dtype=float)
    o = np.argsort(dur, kind='mergesort')
    d, w = dur[o], w[o]
    c = np.cumsum(w)
    return d, (c / c[-1] if c[-1] > 0 else c)


def _dur_at_frac(d, cum, frac=0.5):
    """Run duration by which ``frac`` of the weight has accumulated."""
    if not len(d):
        return np.nan
    return float(np.interp(frac, cum, d))


def plot_time_weighted_ecdf(runs_by_group, state_set='drift+move',
                            weight='duration', exclude_outcomes=NON_TASK_OUTCOMES,
                            alt_dfcs=None, alt_marker='TeTx',
                            show_count_ecdf=True, per_fly=False,
                            colors=None, xmax=12, savepath=None, ax=None):
    """Cumulative version of panel b: what share of total in-target time is
    contributed by runs at or below a given duration.

    runs_by_group : {label: runs_df}, e.g.
                    {'+;iav': iav_runs, 'op_cond': op_cond_runs}
    weight        : 'duration' for total run time; 'move_time' for the MOVE
                    component only, which on a drift+move pass answers 'how much
                    of the move time is in long runs'.
    show_count_ecdf : also draw the unweighted (per-run) ECDF, dashed. The gap
                    between the two curves is the point -- runs are mostly short,
                    time is mostly long.
    per_fly       : thin per-fly time-weighted curves underneath.

    Returns (fig, summary) with per-group D50/D90 (duration by which 50 / 90 % of
    the weight has accumulated) and the fraction of weight in runs >= 1 s.
    """
    default_colors = [C_MAIN, C_TIME, C_DRIFT, '#5b8c5a', '#8d6bb1']
    colors = colors or default_colors
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=150)
    else:
        fig = ax.figure

    rows, dt_all = [], []
    for i, (label, runs) in enumerate(runs_by_group.items()):
        col = colors[i % len(colors)]
        if exclude_outcomes and 'as_outcome' in runs.columns:
            runs = runs[~runs['as_outcome'].isin(exclude_outcomes)]
        if 'state_set' in runs.columns and state_set is not None:
            runs = runs[runs['state_set'] == state_set]
        if runs.empty:
            raise ValueError(f'{label}: no runs left after filtering')
        alt = set(alt_dfcs) if alt_dfcs is not None else (
            set(runs.loc[runs.get('genotype', pd.Series('', index=runs.index))
                         .fillna('').str.contains(alt_marker), 'dfc'].unique())
            if 'genotype' in runs.columns else set())
        keep = runs[~runs['dfc'].isin(alt)]

        if per_fly:
            for _, g in keep.groupby('dfc'):
                d_i, c_i = _weighted_ecdf(g['duration'], g[weight])
                ax.step(d_i, c_i, where='post', color=col, lw=0.7, alpha=0.35,
                        zorder=2)

        d, cum = _weighted_ecdf(keep['duration'], keep[weight])
        ax.step(d, cum, where='post', color=col, lw=2.2, zorder=4,
                label=f'{label} -- time ({len(keep)} runs)')
        if show_count_ecdf:
            dn, cn = _weighted_ecdf(keep['duration'])
            ax.step(dn, cn, where='post', color=col, lw=1.2, ls='--', alpha=0.85,
                    zorder=3, label=f'{label} -- runs')

        d50 = _dur_at_frac(d, cum, 0.5)
        ax.plot([d50], [0.5], marker='o', ms=5, color=col, zorder=5)
        w = keep[weight].to_numpy(dtype=float)
        long_m = keep['duration'].to_numpy(dtype=float) >= 1.0
        rows.append({
            'group': label, 'state_set': state_set, 'weight': weight,
            'n_runs': len(keep), 'total_time': float(w.sum()),
            'D50': d50, 'D90': _dur_at_frac(d, cum, 0.9),
            'median_run': float(np.median(keep['duration'])),
            'frac_time_ge_1s': float(w[long_m].sum() / w.sum()) if w.sum() else np.nan,
            'frac_runs_ge_1s': float(long_m.mean()),
        })
        if 'dt' in runs.columns:
            dt_all.append(float(runs['dt'].median()))

    ax.axhline(0.5, color='#bbb', lw=0.8, ls=':', zorder=1)
    if dt_all:
        ax.set_xlim(min(dt_all) * 0.8, xmax)
    ax.set_xscale('log')
    ax.set_ylim(0, 1)
    ax.set_xlabel(f'in-target {state_set or ""} run duration (s)')
    ax.set_ylabel(f'cumulative fraction of {weight.replace("_", " ")}')
    ax.set_title('cumulative in-target time by run duration\n'
                 '(solid = time-weighted, dashed = per-run; dot = D50)',
                 loc='left', fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc='upper left')
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, bbox_inches='tight')
        print('wrote', savepath)
    return fig, pd.DataFrame(rows)


if __name__ == '__main__':          # True in a notebook cell, False on import
    fig_itm, iav_stats = plot_in_target_durations(
        iav_runs, iav_trials, group_name='+;iav',
        state_set='drift+move', compare_state_set='move',
        savepath=f'{mutant_fig_dir}/in_target_durations_iav.svg')

    fig_op_cond, op_cond_stats = plot_in_target_durations(
        op_cond_runs, op_cond_trials, group_name='op_cond',
        state_set='drift+move', compare_state_set='move',
        savepath=f'{mutant_fig_dir}/in_target_durations_op_cond.svg')


    # --- whole-experiment budget -----------------------------------------------
    iav_budget = state_time_budget(iav_trials, by='dfc')
    op_cond_budget = state_time_budget(op_cond_trials, by='dfc')

    TIME_COLS = ['n_trials', 'window_time', 'move_time', 'drift_time', 'active_time',
                 'in_target_move', 'in_target_drift', 'in_target_active']
    RATIO_COLS = ['in_target_active_per_move', 'in_target_active_per_active',
                  'in_target_move_per_move', 'active_per_window',
                  'in_target_per_window']

    for name, bud in (('+;iav', iav_budget), ('op_cond', op_cond_budget)):
        print(f'\n=== {name} ===')
        print(bud[TIME_COLS].round(1).to_string())
        print(bud[RATIO_COLS].round(3).to_string())

    print('\n=== pooled ===')
    print(pd.concat([state_time_budget(iav_trials, by=None).assign(group='+;iav'),
                     state_time_budget(op_cond_trials, by=None).assign(group='op_cond')])
          .set_index('group')[TIME_COLS + RATIO_COLS].round(3).to_string())


    # --- cumulative time by run duration ---------------------------------------
    fig_ecdf, ecdf_summary = plot_time_weighted_ecdf(
        {'+;iav': iav_runs, 'op_cond': op_cond_runs},
        state_set='drift+move', weight='duration', per_fly=True,
        savepath=f'{mutant_fig_dir}/in_target_time_ecdf.svg')
    print(ecdf_summary.round(3).to_string(index=False))

    fig_ecdf_mv, ecdf_summary_mv = plot_time_weighted_ecdf(
        {'+;iav': iav_runs, 'op_cond': op_cond_runs},
        state_set='drift+move', weight='move_time',
        savepath=f'{mutant_fig_dir}/in_target_move_time_ecdf.svg')
    print(ecdf_summary_mv.round(3).to_string(index=False))
