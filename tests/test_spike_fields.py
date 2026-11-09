"""Tests for per-field spike results (``spikes`` vs ``EMGspikes``), on sandbox
copies of real trial files.

An EMG / extracellular result must live beside a cell's intracellular result,
never on top of it, under the names MATLAB's spikeDetection.m uses for its
'alt_spike_field': EMGspikes, EMGspikes_uncorrected, EMGspikeDetectionParams,
EMGspikeSpotChecked. These tests write to copies only, then check the
originals were not touched.

Run with:  conda run -n flop_py312 python tests/test_spike_fields.py
"""
import hashlib
import shutil
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mapd.trial import Trial                                        # noqa: E402
from mapd.paths import default_data_directory                       # noqa: E402
from mapd import ephys, spike_detection as sds                      # noqa: E402

ROOT = Path(default_data_directory(verbose=False))
# 210319 trial 1 has intracellular spikes; trial 102 is a silent trial whose
# /spikes is MATLAB's empty encoding ([0, 0] flagged MATLAB_empty).
SRC_SPIKES = ROOT / '210319' / '210319_F2_C1' / 'LEDFlashWithPiezoCueControl_Raw_210319_F2_C1_1.mat'
SRC_SILENT = ROOT / '210319' / '210319_F2_C1' / 'LEDFlashWithPiezoCueControl_Raw_210319_F2_C1_102.mat'
# 261019 trial 20: an EMG recording with no spike results yet.
SRC_EMG = ROOT / '261019' / '261019_F2_C1' / 'LEDFlashTriggerPiezoControl_Raw_261019_F2_C1_20.mat'
SOURCES = (SRC_SPIKES, SRC_SILENT, SRC_EMG)


def _check(name, cond, detail=''):
    print(f'{"PASS" if cond else "FAIL"}  {name}' + (f'  — {detail}' if detail else ''))
    return bool(cond)


def _md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def _copy(src, tmp):
    dst = Path(tmp) / src.name
    shutil.copy2(src, dst)
    tr = Trial(str(dst))
    assert Path(tr.file_path).resolve() == dst.resolve(), 'Trial resolved off the copy'
    return tr


def test_field_keys():
    k = sds.field_keys('EMGspikes')
    ok = k == {'spikes': 'EMGspikes', 'uncorrected': 'EMGspikes_uncorrected',
               'params': 'EMGspikeDetectionParams', 'spot': 'EMGspikeSpotChecked'}
    ok &= sds.default_field('voltage_1') == 'spikes'
    ok &= all(sds.default_field(c) == 'EMGspikes'
              for c in ('current_1', 'current_2', 'current_extEMG'))
    return _check('field names match MATLAB alt_spike_field naming', ok, str(k))


def test_matlab_empty_reads_as_silent(tmp):
    tr = _copy(SRC_SILENT, tmp)
    res = sds.load_spikes_from_trial(tr)
    st = ephys.trial_spike_times(tr, allow_empty=True)
    return _check('MATLAB-empty /spikes reads as zero spikes',
                  res is not None and res.n_spikes == 0 and st is not None and len(st) == 0,
                  f'n={None if res is None else res.n_spikes}')


def test_emg_beside_intracellular(tmp):
    tr = _copy(SRC_SPIKES, tmp)
    before = sds.load_spikes_from_trial(tr)
    emg = sds.load_spikes_from_trial(tr, field='EMGspikes')
    ok = _check('intracellular result present, no EMG result yet',
                before is not None and before.n_spikes > 0 and emg is None,
                f'{before.n_spikes} spikes')

    fake = type(before)(spike_times=np.array([1000, 2000, 3000]),
                        spike_times_uncorrected=np.array([1010, 2010, 3010]),
                        params=before.params, spot_checked=False)
    sds.save_spikes_to_trial(tr, fake, field='EMGspikes', channel='current_2')
    with h5py.File(tr.file_path, 'r') as f:
        names = {'EMGspikes', 'EMGspikes_uncorrected', 'EMGspikeSpotChecked',
                 'EMGspikeDetectionParams'}
        ok &= _check('EMG datasets written under the MATLAB names', names <= set(f.keys()))
    emg = sds.load_spikes_from_trial(tr, field='EMGspikes')
    ok &= _check('EMG result reads back',
                 emg is not None and emg.spike_times.tolist() == [1000, 2000, 3000]
                 and emg.spike_times_uncorrected.tolist() == [1010, 2010, 3010])
    ok &= _check('channel recorded with the EMG result',
                 sds.result_channel(tr, 'EMGspikes') == 'current_2',
                 repr(sds.result_channel(tr, 'EMGspikes')))
    after = sds.load_spikes_from_trial(tr)
    ok &= _check('intracellular /spikes untouched by the EMG write',
                 np.array_equal(after.spike_times, before.spike_times))

    # a re-save without channel (what spot-check does) keeps the channel
    sds.save_spikes_to_trial(tr, emg, spot_checked=True, field='EMGspikes')
    ok &= _check('re-save keeps the recorded channel and sets the spot-check flag',
                 sds.result_channel(tr, 'EMGspikes') == 'current_2'
                 and sds.load_spikes_from_trial(tr, field='EMGspikes').spot_checked)

    t_emg = ephys.trial_spike_times(tr, field='EMGspikes')
    fs = float(tr.params['sampratein'])
    t0 = float(np.asarray(tr.time).ravel()[0])
    ok &= _check('trial_spike_times reads the EMG field in seconds',
                 t_emg is not None and np.allclose(t_emg, np.array([1000, 2000, 3000]) / fs + t0))

    # re-tuning EMG invalidates only the EMG spikes
    sds.save_params_to_trial(tr, emg.params, field='EMGspikes')
    with h5py.File(tr.file_path, 'r') as f:
        ok &= _check('EMG re-tune clears EMG spikes, keeps /spikes',
                     'EMGspikes' not in f and 'spikes' in f
                     and 'EMGspikeDetectionParams' in f)
    return ok


def test_emg_initial_params_scale(tmp):
    tr = _copy(SRC_EMG, tmp)
    p_emg = sds.initial_params(tr, field='EMGspikes', channel='current_extEMG')
    p_vm = sds.initial_params(tr, field='spikes', channel='voltage_1')
    return _check('EMG start threshold is set on the EMG channel\'s own scale',
                  0 < p_emg.peak_threshold != p_vm.peak_threshold,
                  f'EMG {p_emg.peak_threshold:.3g} vs Vm default {p_vm.peak_threshold:.3g}')


def main():
    missing = [s for s in SOURCES if not s.exists()]
    if missing:
        print('SKIP — source trials missing:', *missing, sep='\n  ')
        return 0
    hashes = {s: _md5(s) for s in SOURCES}
    results = [test_field_keys()]
    with tempfile.TemporaryDirectory() as tmp:
        for i, t in enumerate((test_matlab_empty_reads_as_silent,
                               test_emg_beside_intracellular,
                               test_emg_initial_params_scale)):
            sub = Path(tmp) / str(i)
            sub.mkdir()
            results.append(t(sub))
    results.append(_check('original trial files untouched',
                          all(_md5(s) == h for s, h in hashes.items())))
    n_fail = results.count(False)
    print(f'\n{len(results) - n_fail}/{len(results)} passed')
    return 1 if n_fail else 0


if __name__ == '__main__':
    sys.exit(main())
