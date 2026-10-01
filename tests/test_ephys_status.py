"""Tests for the ephys_status annotation, on sandbox copies of real trial files.

These copy trial .mat files into a temp directory and operate there. That is only
possible because ``Trial.__init__`` now honours the directory it is given: before
that it kept the basename and rebuilt the path under the data root, so a test
meaning to write to a copy silently modified the original.

Every test asserts on the *copy*, and the final test asserts the originals were
not touched.

Run with:  conda run -n flop_py312 python tests/test_ephys_status.py
"""
import shutil
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mapd.trial import (Trial, EPHYS_STATUS_KEY, EPHYS_NOTE_KEY,  # noqa: E402
                        EPHYS_STATUS_DEFAULT)
from mapd.paths import default_data_directory                      # noqa: E402

CELL_DIR = Path(default_data_directory(verbose=False)) / '210602' / '210602_F1_C1'
PROTOCOL = 'LEDFlashWithPiezoCueControl'
TRIALS = (100, 101, 102)


def _check(name, cond, detail=''):
    print(f'{"PASS" if cond else "FAIL"}  {name}' + (f'  — {detail}' if detail else ''))
    return bool(cond)


def _meta_keys(path):
    with h5py.File(path, 'r') as f:
        return sorted(f['meta'].keys()) if 'meta' in f else []


def sandbox(tmp):
    """Copy the test trials into ``tmp`` and return their paths."""
    out = []
    for n in TRIALS:
        src = CELL_DIR / f'{PROTOCOL}_Raw_210602_F1_C1_{n}.mat'
        dst = Path(tmp) / src.name
        shutil.copy2(src, dst)
        out.append(dst)
    return out


def test_trial_honours_the_given_directory(tmp):
    """The whole safety story: a Trial built from a copy must resolve to the copy."""
    paths = sandbox(tmp)
    tr = Trial(str(paths[0]))
    on_copy = Path(tr.file_path).resolve() == paths[0].resolve()
    # ...and a bare filename must still resolve under the data root, as before.
    tr_bare = Trial(paths[0].name)
    on_root = Path(tr_bare.file_path).resolve() == (CELL_DIR / paths[0].name).resolve()
    return _check('Trial(path) uses the given dir; Trial(name) uses the data root',
                  on_copy and on_root,
                  f'copy={on_copy} root={on_root}')


def test_default_status_when_unannotated(tmp):
    paths = sandbox(tmp)
    tr = Trial(str(paths[0]))
    ok = (tr.ephys_status == EPHYS_STATUS_DEFAULT and tr.ephys_ok
          and tr.ephys_note == ''
          and EPHYS_STATUS_KEY not in _meta_keys(paths[0]))
    return _check('unannotated trial reads as good, with no key on disk', ok,
                  f'status={tr.ephys_status!r} note={tr.ephys_note!r}')


def test_write_and_read_back(tmp):
    paths = sandbox(tmp)
    tr = Trial(str(paths[0]))
    tr.write_string_if_changed(EPHYS_STATUS_KEY, 'bad')
    tr.write_string_if_changed(EPHYS_NOTE_KEY, 'elevated Vm, patch failure')
    fresh = Trial(str(paths[0]))            # new object: reads from disk
    ok = (fresh.ephys_status == 'bad' and not fresh.ephys_ok
          and fresh.ephys_note == 'elevated Vm, patch failure'
          and EPHYS_STATUS_KEY in _meta_keys(paths[0]))
    return _check('status round-trips through /meta', ok,
                  f'status={fresh.ephys_status!r} note={fresh.ephys_note!r}')


def test_status_is_not_cached_stale(tmp):
    """``__getattr__`` caches /meta values as attributes; the property must not."""
    paths = sandbox(tmp)
    tr = Trial(str(paths[0]))
    first = tr.ephys_status                       # 'good'
    tr.write_string_if_changed(EPHYS_STATUS_KEY, 'redetect')
    second = tr.ephys_status                      # must see the new value
    return _check('ephys_status re-reads after a write (not cached stale)',
                  first == 'good' and second == 'redetect',
                  f'{first!r} -> {second!r}')


def test_write_leaves_other_meta_alone(tmp):
    paths = sandbox(tmp)
    before = _meta_keys(paths[0])
    tr = Trial(str(paths[0]))
    values_before = {k: tr._read_value_from_meta(k) for k in before}
    tr.write_string_if_changed(EPHYS_STATUS_KEY, 'bad')
    after = _meta_keys(paths[0])
    fresh = Trial(str(paths[0]))
    values_after = {k: fresh._read_value_from_meta(k) for k in before}
    same = all(np.all(values_before[k] == values_after[k]) for k in before)
    ok = set(after) == set(before) | {EPHYS_STATUS_KEY} and same
    return _check('writing the status adds one key and changes no other', ok,
                  f'added={set(after) - set(before)} others_equal={same}')


def test_meta_needs_no_matlab_fields(tmp):
    """/meta is a plain group, not a MATLAB struct — so a new key needs no
    MATLAB_fields bookkeeping, unlike /params."""
    paths = sandbox(tmp)
    Trial(str(paths[0])).write_string_if_changed(EPHYS_STATUS_KEY, 'bad')
    with h5py.File(paths[0], 'r') as f:
        meta_attrs = dict(f['meta'].attrs)
        params_attrs = dict(f['params'].attrs)
    ok = ('MATLAB_fields' not in meta_attrs) and ('MATLAB_fields' in params_attrs)
    return _check('/meta carries no MATLAB_fields (but /params does)', ok,
                  f'meta={list(meta_attrs)} params_has_fields='
                  f'{"MATLAB_fields" in params_attrs}')


def test_originals_untouched(tmp):
    """Nothing above may have reached the real data files."""
    bad = []
    for n in TRIALS:
        src = CELL_DIR / f'{PROTOCOL}_Raw_210602_F1_C1_{n}.mat'
        keys = _meta_keys(src)
        if EPHYS_STATUS_KEY in keys or EPHYS_NOTE_KEY in keys:
            bad.append((n, keys))
    return _check('the real trial files were not modified', not bad, str(bad))


if __name__ == '__main__':
    if not CELL_DIR.is_dir():
        print(f'SKIP — {CELL_DIR} not available')
        sys.exit(0)
    results = []
    for fn in (test_trial_honours_the_given_directory,
               test_default_status_when_unannotated,
               test_write_and_read_back,
               test_status_is_not_cached_stale,
               test_write_leaves_other_meta_alone,
               test_meta_needs_no_matlab_fields):
        with tempfile.TemporaryDirectory() as tmp:
            results.append(fn(tmp))
    results.append(test_originals_untouched(None))
    print(f'\n{sum(results)}/{len(results)} passed')
    sys.exit(0 if all(results) else 1)
