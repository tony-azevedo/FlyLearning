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


DEFAULT_CACHE_DIR = './data_cache/folded_blocks'

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
    drop_rest: bool = True,
) -> pd.DataFrame:
    """Fold a Table's trial DataFrame into per-trial-in-block outcome counts.

    Returns a DataFrame indexed by ``(pyasState, trial_in_block)`` with columns
    ``as_off``, ``no_as``, ``no_as_mv``, ``to``, ``success``, ``probe_as``,
    ``max_cnts``. If the trial_df has ``sudden_punishment`` / ``soft_sudden_punishment``
    columns (populated by ``Table.find_sudden_punishment_trials()``), those are
    folded in too; otherwise they are silently omitted.
    Missing outcome categories are treated as zero.

    ``drop_rest`` (default True): rest trials (acquisition ndf=0, aversive light
    blocked) are excluded from every outcome / flag line so they don't inflate
    the no_as line at end-of-block positions, but they still contribute to
    ``max_cnts`` — so normalized fractions reflect their share of the total
    block. Requires ``T.extract_trial_properties()`` to have run so ``is_rest``
    is a column.
    """
    required = ['pyasState', 'op_cnd_blocks', 'as_outcome', 'success', 'as_duration']
    missing = [c for c in required if c not in trial_df.columns]
    if missing:
        raise KeyError(f'trial_df is missing required columns: {missing}')

    optional_bool_cols = [c for c in ('soft_sudden_punishment', 'sudden_punishment')
                          if c in trial_df.columns]
    cols_to_keep = required + optional_bool_cols
    if drop_rest and 'is_rest' in trial_df.columns:
        cols_to_keep = cols_to_keep + ['is_rest']

    df = trial_df.loc[trial_df.index > min_trial, cols_to_keep].copy()
    df['trial_in_block'] = df.groupby('op_cnd_blocks').cumcount()
    df['probe_as'] = (df['as_duration'] > 0) & (df['as_outcome'] == 'probe')
    df = df.loc[df['trial_in_block'] < max_trial_in_block]

    states = list(states)

    # max_cnts is the total trial count per (state, trial_in_block), rest
    # included. This is the denominator for normalized fractions.
    max_cnts_full = df.groupby(['pyasState', 'trial_in_block'], observed=True).size()

    # ``pyasState`` is an ordered CategoricalDtype, and ``.loc[['hi','lo']]``
    # against a MultiIndex whose first level is a CategoricalIndex trips a
    # pandas quirk where the level still carries the full categories list
    # while the codes point somewhere else — the label lookup raises even
    # though the values are observed. Filter by boolean mask on the
    # string-cast level values instead to sidestep it.
    max_state_level = max_cnts_full.index.get_level_values('pyasState').astype(str)
    observed = set(max_state_level.tolist())
    present_states = [s for s in states if s in observed]
    if not present_states:
        raise ValueError(
            f'No trials past min_trial={min_trial} in any of states={list(states)}. '
            f'Observed states after filter: {sorted(observed)}. '
            f'Check trial_df.loc[trial_df.index > {min_trial}, "pyasState"].value_counts()'
        )
    max_cnts_full = max_cnts_full[max_state_level.isin(present_states)]

    # For every outcome / flag line, use only the operant (non-rest) trials so
    # rest trials don't inflate no_as_*, but preserve max_cnts_full above.
    if drop_rest and 'is_rest' in df.columns:
        df_ops = df.loc[~df['is_rest'].fillna(False).astype(bool)]
    else:
        df_ops = df

    outcome_counts = (
        df_ops.groupby(['pyasState', 'trial_in_block'], observed=True)['as_outcome']
          .value_counts()
          .unstack(fill_value=0)
    )
    # Reindex to max_cnts_full's index so positions where only rest trials
    # occurred are still present (with all-zero outcome counts).
    outcome_counts = outcome_counts.reindex(max_cnts_full.index, fill_value=0)

    def _sum_cols(frame, cols):
        return sum((frame[c] for c in cols if c in frame.columns), start=pd.Series(0, index=frame.index))

    folded = pd.DataFrame(index=max_cnts_full.index)
    for coarse, fine_cols in _FINE_TO_COARSE.items():
        folded[coarse] = _sum_cols(outcome_counts, fine_cols)
    # Preserve no_as_mv as its own line (also included in no_as).
    folded['no_as_mv'] = outcome_counts.get('no_as_mv', 0)

    bool_flag_cols = ['success', 'probe_as'] + optional_bool_cols
    for flag in bool_flag_cols:
        flag_counts = (
            df_ops.groupby(['pyasState', 'trial_in_block', flag], observed=True)
              .size()
              .unstack(fill_value=0)
        )
        folded[flag] = (flag_counts.get(True, pd.Series(0, index=folded.index))
                                   .reindex(folded.index, fill_value=0))

    folded['max_cnts'] = max_cnts_full
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


def plot_folded_outcomes_overlay_plotly(
    per_fly_counts: Mapping[str, pd.DataFrame],
    aggregate: pd.DataFrame | None = None,
    norm: bool = True,
    colors: Mapping[str, str] | None = None,
    outcome_to_ax: Mapping[str, int] | None = None,
    ax_ylabels: Mapping[int, str] | None = None,
    states: Iterable[str] = DEFAULT_STATES,
    title: str | None = None,
    scatter_size: int = 6,
    scatter_opacity: float = 0.6,
    row_height: int = 260,
):
    """Interactive plotly version of ``plot_folded_outcomes_overlay``.

    Same layout as the matplotlib overlay, but each per-fly scatter point
    carries hover text with the dfc, outcome, and value — useful for
    tracking down which flies drive an unusual aggregate. Aggregate lines
    have hover disabled so the markers stay clickable underneath.

    Save with ``fig.write_html('path.html')``. Returns a plotly Figure.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if aggregate is None:
        aggregate = aggregate_folded_counts(per_fly_counts.values())

    colors = dict(DEFAULT_COARSE_COLORS if colors is None else colors)
    outcome_to_ax = dict(DEFAULT_OUTCOME_TO_AX if outcome_to_ax is None else outcome_to_ax)
    ax_ylabels = dict(DEFAULT_AX_YLABELS if ax_ylabels is None else ax_ylabels)

    agg_state_level = aggregate.index.get_level_values('pyasState').astype(str)
    states_agg = [s for s in states if s in set(agg_state_level.tolist())]

    ax_ids = sorted(set(outcome_to_ax.values()))
    fig = make_subplots(
        rows=len(ax_ids), cols=1, shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=[ax_ylabels.get(a, '') for a in ax_ids],
    )
    ax_id_to_row = {ax_id: i + 1 for i, ax_id in enumerate(ax_ids)}

    x_agg = _stitched_x(aggregate, states_agg)
    # Offset each state's trial_in_block into the stitched x-axis, matching
    # the matplotlib overlay so per-fly points align even when fly and
    # aggregate have different lengths within a state.
    state_offsets = {}
    for st in states_agg:
        agg_state = aggregate[agg_state_level == st]
        agg_state_idx = agg_state.index.get_level_values('trial_in_block')
        if len(x_agg[st]) and len(agg_state_idx):
            state_offsets[st] = int(x_agg[st][0]) - int(agg_state_idx[0])
        else:
            state_offsets[st] = 0

    # Collect per-fly scatter points into one trace per outcome per row so the
    # legend is compact and hover ties each dot back to its dfc.
    scatter_pool: dict[tuple[str, int], dict[str, list]] = {}
    for dfc, counts in per_fly_counts.items():
        fly_state_level = counts.index.get_level_values('pyasState').astype(str)
        for st in states_agg:
            state_mask = fly_state_level == st
            if not state_mask.any():
                continue
            state_counts = counts[state_mask]
            tib = state_counts.index.get_level_values('trial_in_block')
            positions = tib + state_offsets[st]
            for outcome, ax_id in outcome_to_ax.items():
                if outcome not in state_counts.columns:
                    continue
                y = state_counts[outcome]
                if norm:
                    y = y / state_counts['max_cnts']
                key = (outcome, ax_id_to_row[ax_id])
                scatter_pool.setdefault(key, {'x': [], 'y': [], 'dfc': []})
                scatter_pool[key]['x'].extend(positions.tolist())
                scatter_pool[key]['y'].extend(y.tolist())
                scatter_pool[key]['dfc'].extend([dfc] * len(y))

    first_row = min(ax_id_to_row.values())
    for (outcome, row), data in scatter_pool.items():
        fig.add_trace(
            go.Scatter(
                x=data['x'], y=data['y'],
                mode='markers',
                marker=dict(size=scatter_size,
                            color=colors.get(outcome),
                            opacity=scatter_opacity),
                text=data['dfc'],
                hovertemplate=(
                    f'<b>{outcome}</b><br>'
                    'x=%{x}<br>y=%{y:.3f}<br>%{text}<extra></extra>'
                ),
                name=outcome,
                legendgroup=outcome,
                showlegend=(row == first_row),
            ),
            row=row, col=1,
        )

    # Aggregate line on top of the scatter for each outcome.
    for outcome, ax_id in outcome_to_ax.items():
        for st in states_agg:
            state_counts = aggregate[agg_state_level == st]
            if outcome not in state_counts.columns:
                continue
            state_counts = state_counts.copy()
            state_counts.index = state_counts.index.get_level_values('trial_in_block')
            y = state_counts[outcome]
            if norm:
                y = y / state_counts['max_cnts']
            fig.add_trace(
                go.Scatter(
                    x=x_agg[st], y=y,
                    mode='lines',
                    line=dict(color=colors.get(outcome), width=2),
                    name=outcome,
                    legendgroup=outcome,
                    showlegend=False,
                    hoverinfo='skip',
                ),
                row=ax_id_to_row[ax_id], col=1,
            )

    if len(states_agg) > 1 and all(len(x_agg[s]) for s in states_agg):
        boundary = int(x_agg[states_agg[0]][-1]) + 0.5
        for row in ax_id_to_row.values():
            fig.add_vline(x=boundary, line=dict(color='gray', width=1, dash='dash'),
                          row=row, col=1)

    # Plotly's first subplot axes are 'x' and 'y' (no suffix); only extras
    # get 'x2', 'y2', etc.
    x_ref = 'x' if first_row == 1 else f'x{first_row}'
    y_ref = 'y domain' if first_row == 1 else f'y{first_row} domain'
    for st in states_agg:
        if not len(x_agg[st]):
            continue
        mid = 0.5 * (float(x_agg[st][0]) + float(x_agg[st][-1]))
        fig.add_annotation(
            x=mid, y=1.08, xref=x_ref,
            yref=y_ref, text=f'<b>{st}</b>',
            showarrow=False, font=dict(size=13),
        )

    if norm:
        for row in ax_id_to_row.values():
            fig.update_yaxes(range=[0, 1], row=row, col=1)

    fig.update_layout(
        title=title,
        height=row_height * len(ax_ids) + 80,
        hovermode='closest',
        template='simple_white',
    )
    return fig


def _state_frame(counts: pd.DataFrame, state: str) -> pd.DataFrame | None:
    """Single-state slice of a folded-counts frame, indexed by trial_in_block.

    Uses a string-cast boolean mask rather than ``.xs`` for the same
    CategoricalIndex reason documented in ``fold_outcome_counts``.
    """
    mask = counts.index.get_level_values('pyasState').astype(str) == state
    if not mask.any():
        return None
    frame = counts[mask].copy()
    frame.index = frame.index.get_level_values('trial_in_block')
    return frame.sort_index()


def _summed_fraction(frame: pd.DataFrame, cols: Iterable[str], norm: bool) -> pd.Series:
    """Sum a set of coarse outcome columns, optionally as a fraction of max_cnts."""
    present = [c for c in cols if c in frame.columns]
    y = sum((frame[c] for c in present), start=pd.Series(0.0, index=frame.index))
    if norm:
        y = y / frame['max_cnts'].replace(0, np.nan)
    return y


# Two-trace publication view: "stay in the target zone" vs "turn the light off".
PUBLICATION_TRACES = {
    'stay':     dict(cols=('no_as',),       label='Stay',     color='#000000',
                     linewidth=1.8, fontweight='bold'),
    # 'turn_off': dict(cols=('as_off', 'to'), label='Turn off', color='#d98a63',
    #                  linewidth=1.0, fontweight='normal'),
    'turn_off': dict(cols=('as_off',), label='Turn off', color='#d98a63',
                         linewidth=1.0, fontweight='normal'),
}

PUBLICATION_STATE_SHADING = {'hi': '#f4a3aa', 'lo': '#fbd3c6'}

PUBLICATION_STATE_LABELS = {'hi': 'Hi blocks', 'lo': 'Lo blocks'}


def plot_folded_overlay_publication(
    per_fly_counts: Mapping[str, pd.DataFrame],
    aggregate: pd.DataFrame | None = None,
    norm: bool = True,
    fig: Figure | None = None,
    ax=None,
    figsize=(6, 2.2),
    traces: Mapping[str, Mapping] | None = None,
    states: Iterable[str] = DEFAULT_STATES,
    state_shading: Mapping[str, str] | None = None,
    state_labels: Mapping[str, str] | None = None,
    n_task: int = 45,
    shade_alpha: float = 1.0,
    show_flies: bool = False,
    fly_alpha: float = 0.25,
    fly_linewidth: float = 0.5,
    annotate: bool = True,
    legend: bool = False,
    title: str | None = None,
    ylabel: str | None = None,
    xlabel: str = 'trial in block',
    xtick_step: int = 25,
) -> Figure:
    """Single-axis publication view of block-folded outcomes.

    Only two traces are drawn: ``no_as`` ("Stay") and ``as_off`` + ``to``
    ("Turn off"). Both target states share one axis — hi trials occupy x
    1..N, lo trials continue at N+1..2N — with each state's task trials
    shaded (hi darker than lo) and the trailing rest trials left unshaded.

    ``per_fly_counts`` is a ``{dfc: counts}`` mapping as returned by
    ``load_folded_counts_many``; ``aggregate`` defaults to the sum over its
    values. The aggregate is the plotted line — pass ``show_flies=True`` to
    underlay each fly as a faint line.

    ``n_task`` is how many trials at the start of each block carry the
    aversive stimulus (blocks are 45 task + 5 rest), and only controls the
    shading extent.
    """
    if aggregate is None:
        aggregate = aggregate_folded_counts(per_fly_counts.values())

    traces = dict(PUBLICATION_TRACES if traces is None else traces)
    state_shading = dict(PUBLICATION_STATE_SHADING if state_shading is None else state_shading)
    state_labels = dict(PUBLICATION_STATE_LABELS if state_labels is None else state_labels)

    agg_frames = {st: _state_frame(aggregate, st) for st in states}
    states = [st for st in states if agg_frames.get(st) is not None]
    if not states:
        raise ValueError(f'aggregate has none of states={list(state_shading)}')

    if ax is None:
        if fig is None:
            fig = Figure(figsize=figsize)
            FigureCanvas(fig)
        ax = fig.add_subplot(1, 1, 1)
    else:
        fig = ax.get_figure()

    # Stitch the states onto one x-axis: each state starts where the previous
    # one ended, and x is 1-based so hi reads 1..50 and lo reads 51..100.
    x_start: dict[str, int] = {}
    tib_min: dict[str, int] = {}
    offset = 1
    for st in states:
        frame = agg_frames[st]
        x_start[st] = offset
        tib_min[st] = int(frame.index[0])
        offset += len(frame)

    def _x(frame: pd.DataFrame, st: str) -> np.ndarray:
        return frame.index.to_numpy() - tib_min[st] + x_start[st]

    # Shaded task region per state, drawn first so traces sit on top.
    for st in states:
        color = state_shading.get(st)
        if color is None:
            continue
        left = x_start[st] - 0.5
        right = left + min(n_task, len(agg_frames[st]))
        ax.axvspan(left, right, facecolor=color, alpha=shade_alpha,
                   edgecolor='none', zorder=0)

    if show_flies:
        for counts in per_fly_counts.values():
            for st in states:
                frame = _state_frame(counts, st)
                if frame is None:
                    continue
                x = _x(frame, st)
                for spec in traces.values():
                    ax.plot(x, _summed_fraction(frame, spec['cols'], norm),
                            color=spec['color'], linewidth=fly_linewidth,
                            alpha=fly_alpha, zorder=1)

    for key, spec in traces.items():
        labelled = False
        for st in states:
            frame = agg_frames[st]
            y = _summed_fraction(frame, spec['cols'], norm)
            ax.plot(_x(frame, st), y,
                    color=spec['color'],
                    linewidth=spec.get('linewidth', 1.5),
                    solid_capstyle='round',
                    label=spec['label'] if (legend and not labelled) else None,
                    zorder=3)
            labelled = True

        if annotate:
            # Label each trace at the end of its task run in the last state.
            st = states[-1]
            frame = agg_frames[st]
            y = _summed_fraction(frame, spec['cols'], norm)
            i = min(n_task, len(frame)) - 1
            ax.annotate(spec['label'],
                        xy=(_x(frame, st)[i], y.iloc[i]),
                        xytext=(-4, 6), textcoords='offset points',
                        ha='right', va='bottom',
                        fontsize=8, color=spec['color'],
                        fontweight=spec.get('fontweight', 'normal'),
                        zorder=4)

    if norm:
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.5, 1])
        ax.set_ylabel('fraction of trials' if ylabel is None else ylabel)
    elif ylabel:
        ax.set_ylabel(ylabel)

    ax.set_xlim(0.5, offset - 0.5)
    if xtick_step:
        ticks = [1] + list(range(xtick_step, offset, xtick_step))
        ax.set_xticks([t for t in ticks if t < offset])
    if xlabel:
        ax.set_xlabel(xlabel)

    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)

    for st in states:
        label = state_labels.get(st, st)
        mid = x_start[st] + 0.5 * (len(agg_frames[st]) - 1)
        ax.text(mid, 1.02, label, transform=ax.get_xaxis_transform(),
                ha='center', va='bottom', fontsize=9)

    if legend:
        ax.legend(fontsize=8, loc='center right', frameon=False)
    if title:
        # Sits above the per-state labels.
        ax.set_title(title, fontsize=10, pad=18)
    return fig


def per_fly_asymptote_fractions(
    per_fly_counts: Mapping[str, pd.DataFrame],
    positions: tuple = (-24, None),
    outcomes: list | None = None,
    states: Iterable[str] = DEFAULT_STATES,
) -> pd.DataFrame:
    """Per-fly summary of block-folded asymptote fractions.

    For each fly × state × outcome, returns the mean of
    ``counts[outcome] / counts['max_cnts']`` over ``positions`` — a slice
    ``(start, stop)`` into the sorted ``trial_in_block`` axis. Default
    ``(-24, None)`` takes the last 24 folded positions per state, which
    tracks the "asymptote" regardless of how ``max_trial_in_block`` is set.
    Pass e.g. ``(20, 44)`` for an absolute range.

    Returns a tidy DataFrame with columns ``dfc``, ``pyasState``, ``outcome``,
    ``fraction`` — one row per (dfc, state, outcome). Feeds directly into
    ``plot_asymptote_boxes`` or any seaborn/groupby analysis.
    """
    sl = slice(*positions)
    states = list(states)
    rows = []
    for dfc, counts in per_fly_counts.items():
        state_level = counts.index.get_level_values('pyasState').astype(str)
        for st in states:
            mask = state_level == st
            if not mask.any():
                continue
            state_counts = counts[mask].copy()
            state_counts.index = state_counts.index.get_level_values('trial_in_block')
            state_counts = state_counts.sort_index()
            window = state_counts.iloc[sl]
            if not len(window):
                continue
            outcome_cols = (outcomes if outcomes is not None
                            else [c for c in window.columns if c != 'max_cnts'])
            max_cnts = window['max_cnts'].replace(0, np.nan)
            for outcome in outcome_cols:
                if outcome not in window.columns:
                    continue
                frac = (window[outcome] / max_cnts).mean(skipna=True)
                rows.append({
                    'dfc': dfc,
                    'pyasState': st,
                    'outcome': outcome,
                    'fraction': float(frac) if pd.notna(frac) else np.nan,
                })
    return pd.DataFrame(rows)


def plot_asymptote_boxes(
    fractions_df: pd.DataFrame,
    outcomes_to_ax: Mapping[str, int] | None = None,
    ax_ylabels: Mapping[int, str] | None = None,
    states: Iterable[str] = DEFAULT_STATES,
    group_col: str | None = None,
    colors: Mapping[str, str] | None = None,
    group_colors: Mapping[str, str] | None = None,
    figsize=(10, 8),
    jitter: float = 0.08,
    title: str | None = None,
    box_alpha: float = 0.35,
    scatter_size: float = 12,
    seed: int = 0,
):
    """Multi-axis boxplot of per-fly asymptote fractions.

    Layout mirrors the folded plots: one axis per outcome group in
    ``outcomes_to_ax``. Within each axis, one box per (outcome, state) —
    with a jittered scatter of the per-fly values on top. When
    ``group_col`` is given (e.g. ``'group'``), boxes cluster by group so
    learner / control / mutant classes sit side-by-side.

    ``fractions_df`` is expected to be the output of
    ``per_fly_asymptote_fractions`` (columns: dfc, pyasState, outcome,
    fraction), optionally with a ``group_col`` added by the caller.
    """
    import matplotlib.patches as mpatches
    from matplotlib import colormaps

    outcomes_to_ax = dict(DEFAULT_OUTCOME_TO_AX if outcomes_to_ax is None else outcomes_to_ax)
    ax_ylabels = dict(DEFAULT_AX_YLABELS if ax_ylabels is None else ax_ylabels)
    colors = dict(DEFAULT_COARSE_COLORS if colors is None else colors)
    states = list(states)

    df = fractions_df[fractions_df['outcome'].isin(outcomes_to_ax.keys())]
    df = df[df['pyasState'].astype(str).isin(states)]

    groups = (sorted(df[group_col].dropna().unique().tolist())
              if group_col else [None])
    n_groups = max(1, len(groups))

    if group_col and group_colors is None:
        cmap = colormaps['tab10']
        group_colors = {g: cmap(i % 10) for i, g in enumerate(groups)}
    elif group_colors is None:
        group_colors = {}

    ax_ids = sorted(set(outcomes_to_ax.values()))
    fig = Figure(figsize=figsize)
    FigureCanvas(fig)
    ax_id_to_ax = {ax_id: fig.add_subplot(len(ax_ids), 1, i)
                   for i, ax_id in enumerate(ax_ids, start=1)}

    box_width = 0.7 / n_groups
    group_offset_step = 0.8 / n_groups
    outcome_state_gap = 0.6
    rng = np.random.default_rng(seed)

    for ax_id, ax in ax_id_to_ax.items():
        outcomes_here = [o for o in outcomes_to_ax if outcomes_to_ax[o] == ax_id]
        tick_positions = []
        tick_labels = []
        x_center = 0.0
        for outcome in outcomes_here:
            for st in states:
                for gi, group in enumerate(groups):
                    mask = (df['outcome'] == outcome) & (df['pyasState'].astype(str) == st)
                    if group is not None:
                        mask &= (df[group_col] == group)
                    values = df.loc[mask, 'fraction'].dropna().tolist()
                    if not values:
                        continue
                    box_x = x_center + (gi - (n_groups - 1) / 2) * group_offset_step
                    face = (group_colors.get(group)
                            if group is not None else colors.get(outcome, 'gray'))
                    ax.boxplot(
                        [values], positions=[box_x], widths=box_width,
                        patch_artist=True,
                        boxprops=dict(facecolor=face, alpha=box_alpha, edgecolor='black'),
                        medianprops=dict(color='black'),
                        whiskerprops=dict(color='black'),
                        capprops=dict(color='black'),
                        flierprops=dict(marker='.', markersize=3),
                    )
                    xs = box_x + rng.uniform(-jitter, jitter, len(values))
                    ax.scatter(xs, values, s=scatter_size, color=face,
                               edgecolor='black', linewidth=0.3, alpha=0.85, zorder=3)
                tick_positions.append(x_center)
                tick_labels.append(f'{outcome}\n{st}')
                x_center += 1.0
            x_center += outcome_state_gap

        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontsize=8)
        ax.set_ylabel(ax_ylabels.get(ax_id, ''))
        ax.set_ylim(0, 1)
        ax.axhline(0, color='0.85', linewidth=0.5, zorder=0)

    if group_col and groups:
        handles = [mpatches.Patch(facecolor=group_colors[g], alpha=box_alpha,
                                  edgecolor='black', label=str(g)) for g in groups]
        ax_id_to_ax[ax_ids[0]].legend(handles=handles, fontsize=8,
                                      loc='upper right', title=group_col)

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig
