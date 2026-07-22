"""Block-folded outcome analysis.

Given a Table's trial DataFrame, fold trials by their ordinal position within
each ``op_cnd_blocks`` block and count outcome categories separately for
``hi``- and ``lo``-target states. The resulting ``(pyasState, trial_in_block)``
-indexed counts DataFrame is the same shape whether it came from one fly, one
tagged group, or a pooled cross-fly aggregate, so the same plotting code works
at every level.
"""
from __future__ import annotations

import os
import pickle
from collections import defaultdict
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


DEFAULT_CACHE_DIR = './folded_blocks'

DEFAULT_STATES = ('hi', 'lo')

# Coarse-category → list of fine as_outcome columns that get summed into it.
_FINE_TO_COARSE = {
    'as_off': ['as_off', 'as_off_late'],
    'no_as':  ['no_as_no_mv', 'no_as_mv'],
    'to':     ['timeout', 'timeout_fail'],
}

DEFAULT_COARSE_COLORS = {
    'as_off':                 '#169400',
    'no_as':                  '#000cad',
    'no_as_mv':               '#6068cb',
    'success':                '#009480',
    'to':                     '#ff0000',
    'probe_as':               '#ff30fc',
    'sudden_punishment':      '#8b0000',
    'soft_sudden_punishment': '#ff8c00',
}

DEFAULT_OUTCOME_TO_AX = {
    'as_off':                 1,
    'to':                     1,
    'no_as':                  2,
    'no_as_mv':               2,
    'success':                3,
    'sudden_punishment':      3,
    'soft_sudden_punishment': 3,
    'probe_as':               3,
}

DEFAULT_AX_YLABELS = {1: 'as_off / to', 2: 'no_as', 3: 'success / sudden_pun'}


_BOOL_FLAG_COLS = ('success', 'probe_as', 'soft_sudden_punishment', 'sudden_punishment')


def fold_outcome_counts(
    trial_df: pd.DataFrame,
    min_trial: int = 100,
    max_trial_in_block: int = 50,
    states: Iterable[str] = DEFAULT_STATES,
) -> pd.DataFrame:
    """Fold a Table's trial DataFrame into per-trial-in-block outcome counts.

    Returns a DataFrame indexed by ``(pyasState, trial_in_block)`` with columns
    ``as_off``, ``no_as``, ``no_as_mv``, ``to``, ``success``, ``probe_as``,
    ``max_cnts``. If the trial_df has ``sudden_punishment`` / ``soft_sudden_punishment``
    columns (populated by ``Table.find_sudden_punishment_trials()``), those are
    folded in too; otherwise they are silently omitted.
    Missing outcome categories are treated as zero.
    """
    required = ['pyasState', 'op_cnd_blocks', 'as_outcome', 'success', 'as_duration']
    missing = [c for c in required if c not in trial_df.columns]
    if missing:
        raise KeyError(f'trial_df is missing required columns: {missing}')

    optional_bool_cols = [c for c in ('soft_sudden_punishment', 'sudden_punishment')
                          if c in trial_df.columns]
    cols_to_keep = required + optional_bool_cols

    df = trial_df.loc[trial_df.index > min_trial, cols_to_keep].copy()
    df['trial_in_block'] = df.groupby('op_cnd_blocks').cumcount()
    df['probe_as'] = (df['as_duration'] > 0) & (df['as_outcome'] == 'probe')
    df = df.loc[df['trial_in_block'] < max_trial_in_block]

    states = list(states)

    outcome_counts = (
        df.groupby(['pyasState', 'trial_in_block'], observed=True)['as_outcome']
          .value_counts()
          .unstack(fill_value=0)
    )
    # ``pyasState`` is an ordered CategoricalDtype, and ``.loc[['hi','lo']]``
    # against a MultiIndex whose first level is a CategoricalIndex trips a
    # pandas quirk where the level still carries the full categories list
    # while the codes point somewhere else — the label lookup raises even
    # though the values are observed. Filter by boolean mask on the
    # string-cast level values instead to sidestep it.
    state_level = outcome_counts.index.get_level_values('pyasState').astype(str)
    observed = set(state_level.tolist())
    present_states = [s for s in states if s in observed]
    if not present_states:
        raise ValueError(
            f'No trials past min_trial={min_trial} in any of states={list(states)}. '
            f'Observed states after filter: {sorted(observed)}. '
            f'Check trial_df.loc[trial_df.index > {min_trial}, "pyasState"].value_counts()'
        )
    outcome_counts = outcome_counts[state_level.isin(present_states)]

    outcome_counts['max_cnts'] = outcome_counts.sum(axis=1, skipna=True)

    def _sum_cols(frame, cols):
        return sum((frame[c] for c in cols if c in frame.columns), start=pd.Series(0, index=frame.index))

    folded = pd.DataFrame(index=outcome_counts.index)
    for coarse, fine_cols in _FINE_TO_COARSE.items():
        folded[coarse] = _sum_cols(outcome_counts, fine_cols)
    # Preserve no_as_mv as its own line (also included in no_as).
    folded['no_as_mv'] = outcome_counts.get('no_as_mv', 0)

    bool_flag_cols = ['success', 'probe_as'] + optional_bool_cols
    for flag in bool_flag_cols:
        flag_counts = (
            df.groupby(['pyasState', 'trial_in_block', flag], observed=True)
              .size()
              .unstack(fill_value=0)
        )
        folded[flag] = (flag_counts.get(True, pd.Series(0, index=folded.index))
                                   .reindex(folded.index, fill_value=0))

    folded['max_cnts'] = outcome_counts['max_cnts']
    return folded


def cache_path(dfc: str, cache_dir: str = DEFAULT_CACHE_DIR) -> str:
    return os.path.join(cache_dir, f'folded_outcome_counts_{dfc}.pkl')


def save_folded_counts(counts: pd.DataFrame, dfc: str, cache_dir: str = DEFAULT_CACHE_DIR) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    path = cache_path(dfc, cache_dir)
    counts.to_pickle(path)
    return path


def load_folded_counts(dfc: str, cache_dir: str = DEFAULT_CACHE_DIR) -> pd.DataFrame:
    return pd.read_pickle(cache_path(dfc, cache_dir))


def load_folded_counts_many(
    dfcs: Iterable[str],
    cache_dir: str = DEFAULT_CACHE_DIR,
    skip_missing: bool = False,
) -> dict[str, pd.DataFrame]:
    """Load cached per-fly counts for a set of dfcs.

    If ``skip_missing`` is True, dfcs without a cache file are silently
    dropped; otherwise a FileNotFoundError propagates.
    """
    out: dict[str, pd.DataFrame] = {}
    for dfc in dfcs:
        path = cache_path(dfc, cache_dir)
        if not os.path.exists(path):
            if skip_missing:
                continue
            raise FileNotFoundError(path)
        out[dfc] = pd.read_pickle(path)
    return out


def aggregate_folded_counts(counts_iter: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Sum per-fly folded counts DataFrames into a single group aggregate."""
    total: pd.DataFrame | None = None
    for c in counts_iter:
        total = c.copy() if total is None else total.add(c, fill_value=0)
    if total is None:
        raise ValueError('aggregate_folded_counts got no DataFrames')
    return total


def _stitched_x(counts: pd.DataFrame, states: Iterable[str]) -> dict[str, pd.Index]:
    """Build the x-axis positions for each state so that hi trials come first
    and lo trials continue past the end of hi."""
    x: dict[str, pd.Index] = {}
    offset = 0
    for st in states:
        idx = counts.xs(st, level='pyasState').index
        x[st] = idx + offset
        if len(idx):
            offset = int(x[st][-1]) + 1
    return x


def plot_folded_outcomes(
    counts: pd.DataFrame,
    norm: bool = False,
    fig: Figure | None = None,
    figsize=(8, 8),
    colors: Mapping[str, str] | None = None,
    outcome_to_ax: Mapping[str, int] | None = None,
    ax_ylabels: Mapping[int, str] | None = None,
    states: Iterable[str] = DEFAULT_STATES,
    linewidth: float = 1.0,
    alpha: float = 0.9,
    title: str | None = None,
) -> Figure:
    """Plot a single folded-counts DataFrame (per-fly OR aggregate)."""
    colors = dict(DEFAULT_COARSE_COLORS if colors is None else colors)
    outcome_to_ax = dict(DEFAULT_OUTCOME_TO_AX if outcome_to_ax is None else outcome_to_ax)
    ax_ylabels = dict(DEFAULT_AX_YLABELS if ax_ylabels is None else ax_ylabels)
    states = [s for s in states if s in counts.index.get_level_values('pyasState')]

    if fig is None:
        fig = Figure(figsize=figsize)
        FigureCanvas(fig)

    ax_ids = sorted(set(outcome_to_ax.values()))
    ax_id_to_ax = {ax_id: fig.add_subplot(len(ax_ids), 1, i)
                   for i, ax_id in enumerate(ax_ids, start=1)}

    x = _stitched_x(counts, states)

    y_max_per_ax = {ax_id: 0.0 for ax_id in ax_ids}
    for outcome, ax_id in outcome_to_ax.items():
        ax = ax_id_to_ax[ax_id]
        labelled = False
        for st in states:
            state_counts = counts.xs(st, level='pyasState')
            if outcome not in state_counts.columns:
                continue
            y = state_counts[outcome]
            if norm:
                y = y / state_counts['max_cnts']
            ax.plot(x[st], y,
                    label=outcome if not labelled else None,
                    linewidth=linewidth, alpha=alpha,
                    color=colors.get(outcome))
            labelled = True
            if not norm:
                y_max_per_ax[ax_id] = max(y_max_per_ax[ax_id], float(state_counts['max_cnts'].max()))

    for ax_id, ax in ax_id_to_ax.items():
        if norm:
            ax.set_ylim(0, 1)
        else:
            ax.set_ylim(0, y_max_per_ax[ax_id] or 1)
        if ax_id in ax_ylabels:
            ax.set_ylabel(ax_ylabels[ax_id])
        if ax.get_lines():
            ax.legend(fontsize=8, loc='best')

    # Divider between states on every axis, plus a state label centered over
    # each state's x-range on the top axis.
    if any(len(x[s]) for s in states):
        top_ax = ax_id_to_ax[ax_ids[0]]
        for st in states:
            if not len(x[st]):
                continue
            mid = 0.5 * (float(x[st][0]) + float(x[st][-1]))
            top_ax.text(mid, 1.02, st,
                        transform=top_ax.get_xaxis_transform(),
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
    if len(states) > 1 and all(len(x[s]) for s in states):
        boundary = int(x[states[0]][-1]) + 0.5
        for ax in ax_id_to_ax.values():
            ax.axvline(boundary, color='0.6', linewidth=0.5, linestyle='--')

    if title:
        fig.suptitle(title)
    return fig


def table_folded_outcome_counts(self, min_trial=100, max_trial_in_block=50,
                                states=DEFAULT_STATES, save=True,
                                cache_dir=DEFAULT_CACHE_DIR):
    """Table method: fold this fly's trials into per-trial-in-block outcome counts.

    Returns a DataFrame indexed by (pyasState, trial_in_block). This is NOT
    a Sinq scalar — the result is a time series, not a single number, and it
    lives outside the ``compute_*`` convention on purpose. When ``save`` is
    True, caches to ``{cache_dir}/folded_outcome_counts_{dfc}.pkl`` so
    cross-fly aggregation can skip reloading Tables.
    """
    counts = fold_outcome_counts(
        self.df,
        min_trial=min_trial,
        max_trial_in_block=max_trial_in_block,
        states=states,
    )
    if save:
        save_folded_counts(counts, self.dfc, cache_dir=cache_dir)
    return counts


def plot_folded_outcomes_overlay(
    per_fly_counts: Mapping[str, pd.DataFrame],
    aggregate: pd.DataFrame | None = None,
    norm: bool = True,
    fig: Figure | None = None,
    figsize=(8, 8),
    colors: Mapping[str, str] | None = None,
    outcome_to_ax: Mapping[str, int] | None = None,
    ax_ylabels: Mapping[int, str] | None = None,
    states: Iterable[str] = DEFAULT_STATES,
    scatter_size: float = 5.0,
    scatter_alpha: float = 0.7,
    title: str | None = None,
) -> Figure:
    """Overlay per-fly folded counts as scatter + a group aggregate as line.

    If ``aggregate`` is None it will be computed from ``per_fly_counts.values()``.
    """
    if aggregate is None:
        aggregate = aggregate_folded_counts(per_fly_counts.values())

    colors = dict(DEFAULT_COARSE_COLORS if colors is None else colors)
    outcome_to_ax = dict(DEFAULT_OUTCOME_TO_AX if outcome_to_ax is None else outcome_to_ax)
    ax_ylabels = dict(DEFAULT_AX_YLABELS if ax_ylabels is None else ax_ylabels)
    states_agg = [s for s in states if s in aggregate.index.get_level_values('pyasState')]

    if fig is None:
        fig = Figure(figsize=figsize)
        FigureCanvas(fig)

    ax_ids = sorted(set(outcome_to_ax.values()))
    ax_id_to_ax = {ax_id: fig.add_subplot(len(ax_ids), 1, i)
                   for i, ax_id in enumerate(ax_ids, start=1)}

    x_agg = _stitched_x(aggregate, states_agg)
    # Per-state offset the aggregate applied to each state's trial_in_block.
    state_offsets = {st: (int(x_agg[st][0]) - int(aggregate.xs(st, level='pyasState').index[0]))
                     if len(x_agg[st]) else 0
                     for st in states_agg}

    # Per-fly scatter, mapping each fly's trial_in_block into the aggregate's
    # stitched x-axis so all flies align regardless of individual block lengths.
    for counts in per_fly_counts.values():
        for st in states_agg:
            if st not in counts.index.get_level_values('pyasState'):
                continue
            state_counts = counts.xs(st, level='pyasState')
            positions = state_counts.index + state_offsets[st]
            for outcome, ax_id in outcome_to_ax.items():
                if outcome not in state_counts.columns:
                    continue
                y = state_counts[outcome]
                if norm:
                    y = y / state_counts['max_cnts']
                ax_id_to_ax[ax_id].scatter(positions, y.values,
                                           marker='.', s=scatter_size,
                                           alpha=scatter_alpha,
                                           color=colors.get(outcome))

    # Aggregate line on top.
    for outcome, ax_id in outcome_to_ax.items():
        ax = ax_id_to_ax[ax_id]
        labelled = False
        for st in states_agg:
            state_counts = aggregate.xs(st, level='pyasState')
            if outcome not in state_counts.columns:
                continue
            y = state_counts[outcome]
            if norm:
                y = y / state_counts['max_cnts']
            ax.plot(x_agg[st], y,
                    label=outcome if not labelled else None,
                    linewidth=1.5, alpha=0.95,
                    color=colors.get(outcome))
            labelled = True

    for ax_id, ax in ax_id_to_ax.items():
        if norm:
            ax.set_ylim(0, 1)
        if ax_id in ax_ylabels:
            ax.set_ylabel(ax_ylabels[ax_id])
        if ax.get_lines():
            ax.legend(fontsize=8, loc='best')

    if any(len(x_agg[s]) for s in states_agg):
        top_ax = ax_id_to_ax[ax_ids[0]]
        for st in states_agg:
            if not len(x_agg[st]):
                continue
            mid = 0.5 * (float(x_agg[st][0]) + float(x_agg[st][-1]))
            top_ax.text(mid, 1.02, st,
                        transform=top_ax.get_xaxis_transform(),
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
    if len(states_agg) > 1 and all(len(x_agg[s]) for s in states_agg):
        boundary = int(x_agg[states_agg[0]][-1]) + 0.5
        for ax in ax_id_to_ax.values():
            ax.axvline(boundary, color='0.6', linewidth=0.5, linestyle='--')
    if title:
        fig.suptitle(title)
    return fig
