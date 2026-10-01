"""Scan every trial of the A4 cells for the probe's physical rest stop.

x = probeZero - raw_probe, so the resting (zero-force) end of the trace is the
*maximum* raw probe_position.  When the probe reaches its stop the trace clips:
a flat, sustained run at a fixed raw value.  That clipping is the signature we
use to identify which trials actually reached the stop -- most trials never
relax that far, so their own minimum says nothing about where zero is.

Emits one row per trial plus a per-trial raw histogram so the pooled floor and
each trial's dwell-at-floor can be derived afterwards without re-reading.
"""
import glob, re, sys
import h5py
import numpy as np
import pandas as pd

STRIDE = 25          # probe_position is a held step function at camera rate
HIST_LO, HIST_HI = 200, 700
CELLS = {'210917_F2_C1': 655.0, '241115_F1_C1': 630.0, '241203_F2_C1': 630.0}

bins = np.arange(HIST_LO, HIST_HI + 1, 1.0)
rows, hists = [], {}

for cell, pz in CELLS.items():
    day = cell.split('_')[0]
    fs = glob.glob(f'D:/Data/{day}/{cell}/*_Raw_*.mat')
    fs.sort(key=lambda p: int(re.search(r'_(\d+)\.mat$', p).group(1)))
    print(f'== {cell}: {len(fs)} files', flush=True)
    for k, f in enumerate(fs):
        tn = int(re.search(r'_(\d+)\.mat$', f).group(1))
        try:
            with h5py.File(f, 'r') as h:
                raw = np.asarray(h['probe_position'][::STRIDE]).squeeze().astype(float)
                exc = bool(np.asarray(h['excluded']).squeeze()) if 'excluded' in h else False
        except Exception as e:
            rows.append({'cell': cell, 'trial': tn, 'ok': False, 'err': repr(e)[:60]})
            continue
        raw = raw[np.isfinite(raw)]
        if raw.size < 100:
            rows.append({'cell': cell, 'trial': tn, 'ok': False, 'err': 'too few samples'})
            continue
        c, _ = np.histogram(raw, bins=bins)
        hists[(cell, tn)] = c.astype(np.int32)
        rows.append({'cell': cell, 'trial': tn, 'ok': True, 'excluded': exc,
                     'n': raw.size, 'pz_conv': pz,
                     'raw_max': raw.max(), 'raw_p999': np.percentile(raw, 99.9),
                     'raw_p99': np.percentile(raw, 99), 'raw_med': np.median(raw)})
        if k % 100 == 0:
            print(f'   {k}/{len(fs)}', flush=True)

df = pd.DataFrame(rows)
df.to_csv('./Figure7/probezero_floor_scan.csv', index=False)
H = pd.DataFrame({f'{c}|{t}': v for (c, t), v in hists.items()},
                 index=bins[:-1].astype(int))
H.to_parquet('./Figure7/probezero_floor_hists.parquet')
print('wrote', len(df), 'rows;', H.shape[1], 'histograms')
