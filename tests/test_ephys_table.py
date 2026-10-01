"""Table-level tests for ephys_status: set_ephys_status / exclude_ephys_trials.

The write test swaps a few of the Table's Trial objects for ones pointing at
sandbox copies, so the real ``set_ephys_status(write=True)`` code path runs
without touching the data. Everything else here is pure DataFrame work.

Run with:  conda run -n flop_py312 python tests/test_ephys_table.py
"""
import shutil
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mapd                                                        # noqa: E402
from mapd.table import _ephys_status_cat                           # noqa: E402
from mapd.trial import (Trial, EPHYS_STATUS_KEY, EPHYS_NOTE_KEY,   # noqa: E402
                        EPHYS_STATUS_DEFAULT)
from mapd.paths import default_data_directory                      # noqa: E402

DAY, FLY, CELL = '210908', 3, 1
DFC = f'{DAY}_F{FLY}_C{CELL}'
CELL_DIR = Path(default_data_directory(verbose=False)) / DAY / DFC


def _check(name, cond, detail=''):
    print(f'{"PASS" if cond else "FAIL"}  {name}' + (f'  — {detail}' if detail else ''))
    return bool(cond)


def _meta_keys(path):
    with h5py.File(path, 'r') as f:
        return sorted(f['meta'].keys()) if 'meta' in f else []


PROTOCOL = 'LEDFlashTriggerPiezoControl'   # this cell's analysis protocol; the
# directory also holds parquets for other protocols, and those tables do not load.


def load_table():
    parquet = CELL_DIR / f'{PROTOCOL}_{DFC}_Table.parquet'
    T = mapd.Table(str(parquet))
    T.exclude_trials()
    return T


def test_dry_run_updates_df_but_not_disk(T):
    trials = list(T.df.index[:5])
    paths = [Path(T.df.at[tn, 'Trial'].file_path) for tn in trials]
    before = [_meta_keys(p) for p in paths]
    sel = T.set_ephys_status('bad', index=trials, note='dry run', write=False,
                             verbose=False)
    after = [_meta_keys(p) for p in paths]
    df_set = (T.df.loc[trials, EPHYS_STATUS_KEY] == 'bad').all()
    disk_clean = all(EPHYS_STATUS_KEY not in k for k in after) and before == after
    is_cat = isinstance(T.df[EPHYS_STATUS_KEY].dtype, pd.CategoricalDtype)
    return _check('dry run sets the df column (categorical), writes nothing',
                  len(sel) == 5 and df_set and disk_clean and is_cat,
                  f'df_set={df_set} disk_clean={disk_clean} categorical={is_cat}')


def test_rejects_bad_status_and_missing_selection(T):
    bad_status = missing_sel = False
    try:
        T.set_ephys_status('rubbish', trial_min=1, verbose=False)
    except ValueError:
        bad_status = True
    try:
        T.set_ephys_status('bad', verbose=False)       # no selection at all
    except ValueError:
        missing_sel = True
    return _check('rejects an unknown status and a table-wide call',
                  bad_status and missing_sel,
                  f'status={bad_status} selection={missing_sel}')


def test_write_touches_only_the_selected_trials(T, tmp):
    """Real write path, against sandbox copies."""
    trials = list(T.df.index[:4])
    sel_trials, untouched = trials[:2], trials[2:]
    copies = {}
    for tn in trials:
        src = Path(T.df.at[tn, 'Trial'].file_path)
        dst = Path(tmp) / src.name
        shutil.copy2(src, dst)
        copies[tn] = dst
        T.df.at[tn, 'Trial'] = Trial(str(dst))       # points at the copy now
    assert all(Path(T.df.at[tn, 'Trial'].file_path) == copies[tn] for tn in trials)

    T.set_ephys_status('bad', index=sel_trials, note='patch failure',
                       write=True, verbose=False)

    wrote = all(EPHYS_STATUS_KEY in _meta_keys(copies[tn]) for tn in sel_trials)
    skipped = all(EPHYS_STATUS_KEY not in _meta_keys(copies[tn]) for tn in untouched)
    reread = all(Trial(str(copies[tn])).ephys_status == 'bad' for tn in sel_trials)
    note_ok = Trial(str(copies[sel_trials[0]])).ephys_note == 'patch failure'
    return _check('write=True writes only the selected trials',
                  wrote and skipped and reread and note_ok,
                  f'wrote={wrote} others_skipped={skipped} reread={reread} '
                  f'note={note_ok}')


def test_exclude_moves_rows_and_leaves_excluded_alone(T):
    n_before = len(T.df)
    n_excl_before = 0 if T._excluded_df is None else len(T._excluded_df)
    bad = list(T.df.index[:6])
    T.set_ephys_status('bad', index=bad, note='test', write=False, verbose=False)
    excluded_col_before = T.df['excluded'].copy()

    moved = T.exclude_ephys_trials(verbose=False)
    gone = all(tn not in T.df.index for tn in bad)
    in_excluded = all(tn in T._excluded_df.index for tn in bad)
    counts_ok = (len(T.df) == n_before - 6
                 and len(T._excluded_df) == n_excl_before + 6)
    # the /excluded-derived column must be untouched for the rows that remain
    remaining = T.df.index
    col_ok = (T.df['excluded'] == excluded_col_before.loc[remaining]).all()
    source_ok = (T._excluded_df.loc[bad, 'exclusion_source'] == 'ephys').all()

    again = T.exclude_ephys_trials(verbose=False)     # idempotent
    return _check('exclude_ephys_trials moves rows to _excluded_df, not /excluded',
                  gone and in_excluded and counts_ok and col_ok and source_ok
                  and len(again) == 0,
                  f'gone={gone} in_excluded={in_excluded} counts={counts_ok} '
                  f'excluded_col_intact={col_ok} source={source_ok} '
                  f'second_call_moved={len(again)}')


def test_exclude_is_a_noop_without_the_column(T):
    if EPHYS_STATUS_KEY in T.df.columns:
        T.df = T.df.drop(columns=[EPHYS_STATUS_KEY])
    n = len(T.df)
    moved = T.exclude_ephys_trials(verbose=False)
    return _check('exclude_ephys_trials is a no-op when no trial is annotated',
                  len(moved) == 0 and len(T.df) == n)


def test_summary_counts_unannotated_as_good(T):
    T.set_ephys_status('bad', index=list(T.df.index[:3]), write=False, verbose=False)
    s = T.ephys_summary()
    n_bad = int(s.get(('in_df', 'bad'), 0))
    n_good = int(s.get(('in_df', EPHYS_STATUS_DEFAULT), 0))
    return _check('ephys_summary reports NaN trials as good',
                  n_bad == 3 and n_good == len(T.df) - 3,
                  f'bad={n_bad} good={n_good} of {len(T.df)}')


def test_bootstrap_cast_survives_nan():
    """``_bootstrap_meta_columns`` does ``vals.astype(_category_dict[key])``; a
    partially annotated cell must survive that with NaN intact."""
    vals = pd.Series(['good', np.nan, 'bad', 'redetect', np.nan], dtype=object)
    cast = vals.astype(_ephys_status_cat)
    ok = (cast.isna().sum() == 2 and list(cast.dropna()) == ['good', 'bad', 'redetect']
          and cast.dtype.ordered)
    return _check('the categorical cast keeps NaN for unannotated trials', ok,
                  f'{list(cast)}')


def test_originals_untouched():
    hits, unreadable = [], 0
    for p in sorted(CELL_DIR.glob('*_Raw_*.mat')):
        try:
            keys = _meta_keys(p)
        except OSError:
            # Not every raw file in a cell directory is HDF5 (v7 .mat files exist);
            # they are equally not writable by us, so they cannot hide a hit.
            unreadable += 1
            continue
        if EPHYS_STATUS_KEY in keys or EPHYS_NOTE_KEY in keys:
            hits.append(p.name)
    return _check(f'no ephys key written to any of the {DFC} data files',
                  not hits, f'{hits[:5]} ({unreadable} non-HDF5 files skipped)')


if __name__ == '__main__':
    if not CELL_DIR.is_dir():
        print(f'SKIP — {CELL_DIR} not available')
        sys.exit(0)
    results = [test_bootstrap_cast_survives_nan()]

    print(f'\nloading {DFC} ...')
    T = load_table()
    print(f'  {len(T.df)} trials in df\n')

    results.append(test_dry_run_updates_df_but_not_disk(T))
    results.append(test_rejects_bad_status_and_missing_selection(T))
    with tempfile.TemporaryDirectory() as tmp:
        results.append(test_write_touches_only_the_selected_trials(T, tmp))

    T = load_table()          # fresh table for the exclusion tests
    results.append(test_exclude_moves_rows_and_leaves_excluded_alone(T))
    results.append(test_summary_counts_unannotated_as_good(T))
    results.append(test_exclude_is_a_noop_without_the_column(T))
    results.append(test_originals_untouched())

    print(f'\n{sum(results)}/{len(results)} passed')
    sys.exit(0 if all(results) else 1)
