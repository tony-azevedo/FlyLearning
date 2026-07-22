"""Restructure 210908_F3_C1 trial files to match 210604-style layout.

Steps performed (Phase 1 steps 2, 4, 5 of the plan):
- 2. Concatenate each top-level channel with its `intertrial/<channel>` counterpart
     so the main signal contains trial samples followed by inter-trial samples.
- 4. Add the missing scalar/vector params to /params:
       samples, sampratein_control, target_location, startsample, starttime
- 5. Leave existing top-level datasets (target_location, startsample, starttime,
     excluded, tags, name, timestamp, spikes*, spikeDetectionParams, etc.) alone.

By default, writes a sibling file `<name>_restructured.mat` for validation. Pass
``--inplace`` to overwrite originals (after the user has validated trial 100).
"""

from __future__ import annotations

import argparse
import os
import shutil
from typing import Iterable

import h5py
import hdf5storage as h5s
import numpy as np

DATA_DIR = r'D:\Data\210908\210908_F3_C1'
PROTO = 'LEDFlashTriggerPiezoControl'
DFC = '210908_F3_C1'

CHANNELS = (
    'arduino_output', 'current_1', 'current_2', 'current_extEMG',
    'probe_position', 'sgsmonitor', 'voltage_1', 'voltage_2',
)


def restructure(src_path: str, dst_path: str) -> dict:
    """Restructure one trial. Returns a small report dict for logging."""
    if src_path != dst_path:
        shutil.copy2(src_path, dst_path)

    # Read everything we need in a single open.
    with h5py.File(dst_path, 'r') as f:
        sig = {ch: f[ch][...] for ch in CHANNELS}
        inter = {ch: f[f'intertrial/{ch}'][...] for ch in CHANNELS}
        sampratein = float(np.asarray(f['params/sampratein'][...]).ravel()[0])
        target_location = np.asarray(f['target_location'][...]).reshape(1, 2).astype('float64')
        startsample = float(np.asarray(f['startsample'][...]).ravel()[0])
        starttime = float(np.asarray(f['starttime'][...]).ravel()[0])
        trial_only = sig['voltage_1'].shape[0]
        intertrial_len = inter['voltage_1'].shape[0]

    # Sanity: every channel must have matching trial-vs-intertrial shapes.
    for ch in CHANNELS:
        assert sig[ch].shape == (trial_only, 1), f'{ch} trial shape {sig[ch].shape}'
        assert inter[ch].shape == (intertrial_len, 1), f'{ch} intertrial shape {inter[ch].shape}'

    # Concatenate (trial first, then intertrial) → (N, 1) column vectors.
    concat = {ch: np.concatenate([sig[ch], inter[ch]], axis=0) for ch in CHANNELS}
    new_samples = concat['voltage_1'].shape[0]
    assert new_samples == trial_only + intertrial_len

    # Replace each top-level channel: delete then re-write via hdf5storage so
    # the new dataset gets MATLAB-compatible attrs. hdf5storage transposes for
    # MATLAB column-major layout, so to make h5py see (N, 1) we pass (1, N).
    for ch in CHANNELS:
        with h5py.File(dst_path, 'r+') as f:
            if ch in f:
                del f[ch]
        h5s.write(
            data=concat[ch].reshape(1, -1),
            path='/' + ch,
            filename=dst_path,
            matlab_compatible=True,
            store_python_metadata=False,
        )

    # Add the five missing params. samples must reflect the new total length.
    # target_location: pass (2, 1) so h5py sees (1, 2) like 210604.
    new_params = {
        'samples': float(new_samples),
        'sampratein_control': float(sampratein),
        'target_location': target_location.reshape(2, 1),
        'startsample': float(startsample),
        'starttime': float(starttime),
    }
    for k, v in new_params.items():
        with h5py.File(dst_path, 'r+') as f:
            grp = f.require_group('params')
            if k in grp:
                del grp[k]
        h5s.write(
            data=v,
            path=f'/params/{k}',
            filename=dst_path,
            matlab_compatible=True,
            store_python_metadata=False,
        )
        # MATLAB needs H5PATH on /params/* datasets to expose them as struct fields.
        with h5py.File(dst_path, 'r+') as f:
            f[f'/params/{k}'].attrs.create(
                'H5PATH', '/params', dtype=h5py.string_dtype(encoding='ascii')
            )

    # MATLAB enumerates struct fields from the parent group's MATLAB_fields
    # attribute (vlen array of vlen-S1). Datasets not listed there are
    # invisible to load/whos. Append any new field names.
    vlen_s1 = h5py.vlen_dtype(np.dtype('S1'))
    with h5py.File(dst_path, 'r+') as f:
        attrs = f['params'].attrs
        existing = attrs['MATLAB_fields']
        existing_names = {b''.join(e).decode('ascii') for e in existing}
        to_add = [n for n in new_params if n not in existing_names]
        if to_add:
            out = np.empty(len(existing) + len(to_add), dtype=object)
            for i, e in enumerate(existing):
                out[i] = e.astype('S1')
            for j, n in enumerate(to_add):
                out[len(existing) + j] = np.frombuffer(n.encode('ascii'), dtype='S1')
            del attrs['MATLAB_fields']
            attrs.create('MATLAB_fields', out, dtype=vlen_s1)

    return {
        'src': src_path,
        'dst': dst_path,
        'trial_only_samples': trial_only,
        'intertrial_samples': intertrial_len,
        'new_samples': new_samples,
        'sampratein': sampratein,
    }


def iter_trial_files(data_dir: str = DATA_DIR) -> Iterable[str]:
    prefix = f'{PROTO}_Raw_{DFC}_'
    for name in sorted(os.listdir(data_dir)):
        if name.startswith(prefix) and name.endswith('.mat') and '_restructured' not in name:
            yield os.path.join(data_dir, name)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--trial', type=int, default=None,
                   help='Restructure only this trial number (default: all).')
    p.add_argument('--inplace', action='store_true',
                   help='Overwrite originals. Without this, writes <name>_restructured.mat.')
    args = p.parse_args()

    if args.trial is not None:
        files = [os.path.join(DATA_DIR, f'{PROTO}_Raw_{DFC}_{args.trial}.mat')]
    else:
        files = list(iter_trial_files())

    for src in files:
        dst = src if args.inplace else src.replace('.mat', '_restructured.mat')
        report = restructure(src, dst)
        print(f"trial={os.path.basename(src)} new_samples={report['new_samples']} "
              f"(trial={report['trial_only_samples']} + intertrial={report['intertrial_samples']})")


if __name__ == '__main__':
    main()
