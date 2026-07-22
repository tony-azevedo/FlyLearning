"""Figure-7 plotting and per-cell orchestration.

Imports analysis primitives from :mod:`mapd.bout_analysis` and renders /
saves the panels used by ``op_cond_Figure7_MN_spike_rate_vs_position.ipynb``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from mapd import bout_analysis as ba
from mapd.bout_analysis import (
    SIGMA_S,
    collect_cell_records,
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

def plot_rate_vs_position(records, cell_id, max_points_per_state=20000,
                          marker_size=3, opacity=0.4):
    """Plotly scatter of spike rate (x) vs. probe position (y), one subplot
    per movement state. Hover shows trial + trial time.
    """
    state_styles = [
        (ba.kin.STATE_MOVE,  'MOVE',  '#d62728'),
        (ba.kin.STATE_DRIFT, 'DRIFT', '#ff7f0e'),
        (ba.kin.STATE_REST,  'REST',  '#666666'),
    ]
    fig = make_subplots(
        rows=1, cols=3,
        shared_xaxes=True, shared_yaxes=True,
        subplot_titles=[s[1] for s in state_styles],
        horizontal_spacing=0.04,
    )
    rng = np.random.default_rng(0)
    for col, (state, label, color) in enumerate(state_styles, start=1):
        sub = records[records['state'] == state]
        n = len(sub)
        if n > max_points_per_state:
            sub = sub.iloc[rng.choice(n, size=max_points_per_state, replace=False)]
        customdata = np.column_stack([sub['trial'].to_numpy(), sub['t'].to_numpy()])
        fig.add_trace(
            go.Scattergl(
                x=sub['rate'], y=sub['x'],
                mode='markers',
                name=f'{label} (n={n:,})',
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
        fig.update_xaxes(title_text='spike rate (Hz)', row=1, col=col,
                         zeroline=True, zerolinecolor='black', zerolinewidth=0.5)
    fig.update_yaxes(title_text='probe position (um)', row=1, col=1,
                     zeroline=True, zerolinecolor='black', zerolinewidth=0.5)
    fig.update_layout(
        title=f'{cell_id} - smoothed rate (sigma={int(SIGMA_S*1000)} ms) vs. probe position',
        height=440, width=1100,
        margin=dict(l=60, r=20, t=70, b=50),
        showlegend=False,
    )
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
