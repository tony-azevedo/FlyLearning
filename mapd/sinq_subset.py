"""
mapd/sinq_subset.py
===================
Sinqs whose per-fly scalars are recomputed on a SUBSET of each fly's trials.

``SubsetSinq(source, selector)`` takes an existing Sinq and a ``TrialSelector``
and, for every fly, rebuilds the Table from its parquet, restricts ``T.df`` to
the selected trials, and recomputes the same scalar columns the source Sinq
stores (via the inherited ``Sinq._get_table_attr``). Tags and notes are copied
from the source, so ``has_tag`` / ``rows_with_tags`` / ``display_df`` work.

It never writes the real Sinq pickles in ``{data_root}/Sinqs``. Results are
cached as one parquet per (source, selector) under ``cache_dir`` and reloaded
on the next construction.

Selectors
---------
``trials_after(n)``  trials with index > n (drop the early, pre-learning trials)
``vnc_intact()``     trials with ``vnc_status == 'intact'`` (VNC-cut flies)

Usage
-----
    late = SubsetSinq(Sinq(sinqname='dataset_1'), trials_after(100))
    late.display_df()

    vnc = SubsetSinq(Sinq(sinqname='dataset_2_controls'), vnc_intact(),
                     rows=sinq.rows_with_tags('vnc_cut'))
"""
from __future__ import annotations

import contextlib
import gc
import io
import json
import os
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

import numpy as np
import pandas as pd

from .sinq import Sinq, _reorder_columns_helper


# ── Trial selectors ───────────────────────────────────────────────────────────

@dataclass
class TrialSelector:
    """
    name : short identifier, used in the cache file name
    mask : function(Table) -> boolean Series aligned to T.df.index, or None when
           the rule doesn't apply to this fly (the row is then dropped)
    """
    name: str
    mask: Callable


def trials_after(n: int) -> TrialSelector:
    """Trials with trial number (T.df index) > n."""
    return TrialSelector(f'trials_after_{n}', lambda T: pd.Series(T.df.index > n, index=T.df.index))


def all_trials() -> TrialSelector:
    """Every trial. Used to check that a SubsetSinq reproduces its source exactly."""
    return TrialSelector('all_trials', lambda T: pd.Series(True, index=T.df.index))


def vnc_intact() -> TrialSelector:
    """
    Trials with vnc_status == 'intact'. None (fly dropped) when the fly has no
    vnc_status column; a fly that is intact throughout is kept, and its subset
    equals the full session.
    """
    def _mask(T):
        if 'vnc_status' not in T.df.columns:
            return None
        return T.df['vnc_status'].astype(str).eq('intact')
    return TrialSelector('vnc_intact', _mask)


# ── Table restriction ─────────────────────────────────────────────────────────

# Table caches that would otherwise hand back full-session values after T.df is
# restricted. probe_position_distribution caches on the filter dict alone and
# never checks `index`.
_TABLE_CACHES = ('_hi_state_median_position', '_lo_state_median_position', '_state_time_counts_cache')


def restrict_table(T, mask):
    """
    Restrict a Table to the trials in `mask`, in place, and clear its caches.

    Call T.find_successful_trials() BEFORE this: success detection needs
    consecutive trial numbers, so computing it on the full session keeps
    successes at the subset boundary correct; the success columns then ride
    along into the subset. T.targets also stays as computed from the full
    session (same task, same targets).
    """
    T.df = T.df.loc[mask.reindex(T.df.index, fill_value=False).astype(bool)]
    T._df_filter = None
    T._probe_bins = None
    T._total_counts = None
    for attr in _TABLE_CACHES:
        if hasattr(T, attr):
            delattr(T, attr)
    return T


# ── SubsetSinq ────────────────────────────────────────────────────────────────

_SKIP_COLUMNS = ('Table', 'notes')


class SubsetSinq(Sinq):
    """
    A Sinq whose scalars are recomputed on a subset of each fly's trials.

    source     : Sinq to take the flies, parquet paths, notes and column list from
    selector   : TrialSelector
    rows       : dfcs to include (default: every row of source)
    columns    : scalar columns to compute (default: source.df.columns)
    cache_dir  : where the parquet cache lives
    recompute  : ignore the cache and rebuild
    min_trials : rows with fewer selected trials get low_n=True (kept, not dropped)
    """

    def __init__(self, source: Sinq, selector: TrialSelector, rows: Optional[Iterable[str]] = None,
                 columns: Optional[Iterable[str]] = None, cache_dir: str = './data_cache/subset_sinqs',
                 recompute: bool = False, min_trials: int = 50, verbose: bool = True):
        # Deliberately not Sinq.__init__: that would load (and possibly write) a pickle.
        self.source = source
        self.selector = selector
        self.sinqname = f'{source.sinqname}__{selector.name}'
        self.cache_dir = cache_dir
        self.file_name = os.path.join(cache_dir, self.sinqname + '.parquet')
        self.file_location = cache_dir
        self.T = None
        self._sources = [source]
        self.min_trials = min_trials
        self.verbose = verbose

        rows = list(source.df.index if rows is None else rows)
        self.scalar_columns = [c for c in (source.df.columns if columns is None else columns) if c not in _SKIP_COLUMNS]

        # Cache: build only the flies it doesn't hold yet, merge, and save. Flies the selector
        # doesn't apply to are cached too (applicable=False) so they aren't rebuilt every time.
        cached = None
        if not recompute and os.path.exists(self.file_name):
            cached = self._load_cache()
        todo = [r for r in rows if cached is None or r not in cached.index]
        if todo:
            # save after every fly, so an interrupted build keeps the flies already done
            for dfc in todo:
                new = self._build([dfc])
                cached = new if cached is None else pd.concat([cached.drop(index=new.index, errors='ignore'), new])
                self.df = cached
                self.save()
        else:
            self._log(f'{self.sinqname}: all {len(rows)} rows from cache {self.file_name}')
        keep = [r for r in rows if bool(cached.loc[r, 'applicable'])]
        self.df = cached.loc[keep]

    # ---------- build ----------

    def _log(self, msg):
        if self.verbose:
            print(msg, flush=True)

    def _build(self, rows):
        from .table import Table
        out = []
        for dfc in rows:
            src_row = self.source.df.loc[dfc]
            self._log(f'{self.sinqname}: {dfc}')
            with contextlib.redirect_stdout(io.StringIO()):
                T = Table(src_row['parquet'])
                T.extract_trial_properties()
                T.find_successful_trials()          # on the full session, before restricting
                n_full = len(T.df)
                mask = self.selector.mask(T)
                row = pd.Series({'Table': T, 'parquet': src_row['parquet']}, dtype=object)
                n_sub = 0
                if mask is not None:
                    restrict_table(T, mask)
                    n_sub = len(T.df)
                if n_sub > 0:
                    done_outcomes = False
                    for attr in self.scalar_columns:
                        if attr.startswith('outcome_fractions_'):
                            if done_outcomes:
                                continue
                            done_outcomes = True
                        row, _ = self._get_table_attr(row, attr)
            row['Table'] = None
            row['notes'] = src_row.get('notes')
            row['n_trials_full'] = n_full
            row['n_trials_subset'] = n_sub
            row['low_n'] = n_sub < self.min_trials
            row['subset_rule'] = self.selector.name
            row['applicable'] = mask is not None
            row.name = dfc
            out.append(row)
            del T
            gc.collect()
        df = pd.DataFrame(out)
        df.index.name = None
        for c in self.scalar_columns:
            if c in df.columns:
                try:
                    df[c] = pd.to_numeric(df[c])
                except (ValueError, TypeError):
                    pass
        return _reorder_columns_helper(df)

    # ---------- persistence: parquet cache, never the Sinq pickles ----------

    def save(self):
        os.makedirs(self.cache_dir, exist_ok=True)
        df = self.df.drop(columns=['Table'], errors='ignore').copy()
        df['notes'] = df['notes'].map(lambda n: json.dumps(n, default=str) if n is not None else None)
        # write then rename, so a kill mid-write can't corrupt the flies already cached
        tmp = self.file_name + '.tmp'
        df.to_parquet(tmp)
        os.replace(tmp, self.file_name)

    def _load_cache(self):
        df = pd.read_parquet(self.file_name)
        df['notes'] = df['notes'].map(lambda s: json.loads(s) if isinstance(s, str) else None)
        df.insert(0, 'Table', None)
        return _reorder_columns_helper(df)

    def _ensure_notes_column(self):
        # The base class saves when notes are missing; a subset never writes that way.
        if self.df is not None and self.NOTES_COLUMN not in self.df.columns:
            self.df[self.NOTES_COLUMN] = None

    def load_sinq(self, fresh: bool = False):
        raise NotImplementedError('SubsetSinq loads from its parquet cache; construct a new one instead.')

    # ---------- tables ----------

    def restore_table(self, dayflycell=None, overwrite=False):
        """The fly's Table, restricted to the same subset of trials."""
        from .table import Table
        if dayflycell not in self.df.index:
            print(f'No row for {dayflycell}.')
            return None
        T = Table(self.df.loc[dayflycell, 'parquet'])
        T.extract_trial_properties()
        T.find_successful_trials()
        mask = self.selector.mask(T)
        if mask is not None:
            restrict_table(T, mask)
        self.df.at[dayflycell, 'Table'] = T
        self.T = T
        return T
