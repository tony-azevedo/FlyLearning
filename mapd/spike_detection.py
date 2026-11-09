"""
Spike detection bridge for mapd.Trial objects.

Uses the `spikedetect` package (spikeDetection_py/spikedetect) to detect
motor-neuron / EMG spikes on an ephys channel of a mapd.Trial, and writes
the results back into the trial's .mat file in a MATLAB-compatible layout
(top-level /spikes, /spikes_uncorrected, /spikeSpotChecked, and a
/spikeDetectionParams/ group). All other trial data is left untouched.

`spikedetect` is imported lazily, so `import mapd` still works without it.

Typical workflow
----------------
    import mapd
    from mapd import spike_detection as sds
    import spikedetect as sd

    T = mapd.Table("241203_F2_C1_Table.parquet")
    trials = list(T.df['Trial'].dropna())

    # 1) Seed params on one trial with the GUIs (see spikeDetection_py README)
    from spikedetect.io.config import load_params
    params = load_params("241203_F2_C1")

    # 2) Tune thresholds on a subset by inspecting candidate distribution
    cands = sds.collect_candidates(trials[:20], params)
    # plt.scatter(cands.dtw_distance, cands.amplitude, c=cands.accepted)
    # adjust params.distance_threshold / params.amplitude_threshold

    # 3) Full detection on the same subset (save=True writes to .mat files)
    results, summary = sds.batch_detect_spikes(trials[:20], params, save=True)

    # 4) When happy, batch the whole cell
    results, summary = sds.batch_detect_spikes(trials, params, save=True)
"""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Iterable, Optional

import h5py
import hdf5storage as h5s
import numpy as np
import pandas as pd


_DEFAULTS_PATH = Path(__file__).with_name("spike_detection_defaults.json")

# Scalar params persisted to /spikeDetectionParams/<mat_name>. The full
# SpikeDetectionParams dataclass has two more fields (spike_template,
# likely_inflection_point_peak) that are written separately.
_PARAM_SCALAR_FIELDS = [
    ("fs", "fs"),
    ("spike_template_width", "spikeTemplateWidth"),
    ("hp_cutoff", "hp_cutoff"),
    ("lp_cutoff", "lp_cutoff"),
    ("diff_order", "diff"),
    ("peak_threshold", "peak_threshold"),
    ("distance_threshold", "Distance_threshold"),
    ("amplitude_threshold", "Amplitude_threshold"),
    ("polarity", "polarity"),
]


def _import_spikedetect():
    try:
        import spikedetect as sd
        return sd
    except ImportError as e:
        raise ImportError(
            "mapd.spike_detection requires the `spikedetect` package.\n"
            "Install it with:\n"
            '  pip install -e "<path/to/spikeDetection_py/spikedetect>[io,fast]"'
        ) from e


# ---------------------------------------------------------------------------
# Trial <-> Recording adapters
# ---------------------------------------------------------------------------

def trial_to_recording(trial, channel: str = "voltage_1"):
    """Wrap one channel of a mapd.Trial as a spikedetect.Recording."""
    sd = _import_spikedetect()
    v = np.asarray(getattr(trial, channel), dtype=np.float64).ravel()
    fs = float(trial.params["sampratein"])
    name = f"{trial._dfc}_tr{int(trial.params['trial'])}_{channel}"
    return sd.Recording(name=name, voltage=v, sample_rate=fs)


# ---------------------------------------------------------------------------
# Result fields: intracellular spikes vs EMG / extracellular spikes
# ---------------------------------------------------------------------------
# A trial can carry two independent results, named as the MATLAB
# spikeDetection.m names them with its 'alt_spike_field' argument:
#
#   field        uncorrected              params group              spot-check flag
#   spikes       spikes_uncorrected       spikeDetectionParams      spikeSpotChecked
#   EMGspikes    EMGspikes_uncorrected    EMGspikeDetectionParams   EMGspikeSpotChecked
#
# so EMG detection never overwrites the intracellular result, and files stay
# readable by FlyAnalysis. The params group also records ``channel``, the
# electrode the result was detected on (current_extEMG, current_1, ...).

def field_keys(field: str = "spikes") -> dict:
    """HDF5 dataset / group names that hold one spike result ``field``."""
    if not field.endswith("s"):
        raise ValueError(f"spike field must end in 's' (e.g. 'EMGspikes'): {field!r}")
    stem = field[:-1]
    return {"spikes": field, "uncorrected": f"{field}_uncorrected",
            "params": f"{stem}DetectionParams", "spot": f"{stem}SpotChecked"}


def default_field(channel: str) -> str:
    """``spikes`` for an intracellular voltage channel, ``EMGspikes`` for any
    current / extracellular channel."""
    return "spikes" if channel.startswith("voltage") else "EMGspikes"


def load_spikes_from_trial(trial, field: str = "spikes"):
    """Read the ``field`` spike result from trial.file_path, or None if absent.

    Reads the HDF5 directly rather than through ``sd.load_recording``, which
    insists on a ``voltage_1`` dataset and only knows the ``spikes`` field.
    MATLAB's empty-array encoding (a ``[0, 0]`` payload flagged MATLAB_empty)
    reads as zero spikes. Pre-v7.3 (non-HDF5) files fall back to
    ``sd.load_recording`` for ``spikes``.
    """
    sd = _import_spikedetect()
    from spikedetect.io.mat import _load_params_h5, _read_h5_array_or_empty

    k = field_keys(field)
    try:
        f = h5py.File(trial.file_path, "r")
    except OSError:
        return sd.load_recording(trial.file_path).result if field == "spikes" else None
    with f:
        if k["params"] not in f:
            return None
        params = _load_params_h5(f[k["params"]])
        st = (_read_h5_array_or_empty(f[k["spikes"]], dtype=np.int64)
              if k["spikes"] in f else np.array([], dtype=np.int64))
        stu = (_read_h5_array_or_empty(f[k["uncorrected"]], dtype=np.int64)
               if k["uncorrected"] in f else np.array([], dtype=np.int64))
        spot = bool(np.asarray(f[k["spot"]]).ravel()[0]) if k["spot"] in f else False
    return sd.SpikeDetectionResult(spike_times=st, spike_times_uncorrected=stu,
                                   params=params, spot_checked=spot)


def result_channel(trial, field: str = "spikes") -> Optional[str]:
    """The channel a stored result was detected on, or None if not recorded
    (results written before the channel was stored, and MATLAB results)."""
    _import_spikedetect()
    from spikedetect.io.mat import _read_h5_string

    k = field_keys(field)
    try:
        with h5py.File(trial.file_path, "r") as f:
            ds = f.get(f"{k['params']}/channel")
            return None if ds is None else _read_h5_string(ds)
    except OSError:
        return None


def print_trial_params(trial) -> None:
    """Pretty-print the /spikeDetectionParams stored on a mapd.Trial.

    In a notebook, ``trial.spikeDetectionParams`` returns an opaque HDF5
    group proxy. This reads the group via ``load_spikes_from_trial`` and
    prints the full dataclass (including whether a spike template exists
    and how many spikes were detected).
    """
    res = load_spikes_from_trial(trial)
    header = f"Trial {trial.params['trial']} ({trial._dfc}):"
    if res is None:
        print(f"{header} no /spikeDetectionParams")
        return
    p = res.params
    print(header)
    print(f"  fs                   = {p.fs:g} Hz")
    print(f"  hp_cutoff            = {p.hp_cutoff:g}")
    print(f"  lp_cutoff            = {p.lp_cutoff:g}")
    print(f"  diff_order           = {p.diff_order}")
    print(f"  polarity             = {p.polarity:+d}")
    print(f"  peak_threshold       = {p.peak_threshold:g}")
    print(f"  distance_threshold   = {p.distance_threshold:g}")
    print(f"  amplitude_threshold  = {p.amplitude_threshold:g}")
    print(f"  spike_template_width = {p.spike_template_width}")
    if p.spike_template is not None:
        print(f"  spike_template       = <array len={len(p.spike_template)}>")
    if p.likely_inflection_point_peak is not None:
        print(f"  likely_inflection_point_peak = {p.likely_inflection_point_peak}")
    print(f"  spikes detected      = {res.n_spikes}"
          f"  (spot_checked={res.spot_checked})")


def find_trial_path(cell_id: str, trial_num: int) -> Path:
    """Locate the Raw .mat for a given cell / trial.

    Tries D:/Data/<day>/<cell>/ then C:/Users/Tony/Data/<day>/<cell>/.
    The protocol prefix (``LEDFlashTriggerPiezoControl`` etc.) is
    discovered by globbing, so this works for any protocol.
    """
    day = cell_id.split("_")[0]
    roots = [Path(r"D:/Data"), Path(r"C:/Users/Tony/Data")]
    for root in roots:
        cell_dir = root / day / cell_id
        if not cell_dir.is_dir():
            continue
        matches = sorted(cell_dir.glob(f"*_Raw_{cell_id}_{trial_num}.mat"))
        if matches:
            return matches[0]
    raise FileNotFoundError(
        f"Could not find trial {trial_num} of {cell_id} under "
        + ", ".join(str(r) for r in roots)
    )


def list_cell_trials(cell_id: str):
    """Return sorted ``[(trial_num, path), ...]`` for every Raw .mat
    in the cell dir. Tries D:/Data then C:/Users/Tony/Data.
    """
    day = cell_id.split("_")[0]
    roots = [Path(r"D:/Data"), Path(r"C:/Users/Tony/Data")]
    tail_re = re.compile(rf"_Raw_{re.escape(cell_id)}_(\d+)$")
    for root in roots:
        cell_dir = root / day / cell_id
        if not cell_dir.is_dir():
            continue
        out = []
        for p in cell_dir.glob(f"*_Raw_{cell_id}_*.mat"):
            m = tail_re.search(p.stem)
            if m:
                out.append((int(m.group(1)), p))
        if out:
            return sorted(out)
    raise FileNotFoundError(
        f"No Raw trial files found for {cell_id} under D:/Data or C:/Users/Tony/Data"
    )


def plot_trial_spikes(trial, result=None, channel: str = "voltage_1",
                      figsize=(12, 7), show: bool = True):
    """Plot voltage + filtered + waveforms + ISI histogram for a mapd.Trial.

    Panels:
      - top (wide): raw voltage on ``channel`` with red marks at spike samples
      - middle (wide): filtered signal (using ``result.params``) with same marks
      - bottom-left: spike-triggered waveforms overlaid + mean
      - bottom-right: ISI histogram

    Parameters
    ----------
    trial : mapd.Trial
    result : spikedetect.SpikeDetectionResult | None
        If None, loaded from the trial's .mat via ``load_spikes_from_trial``.
        If still missing, only the voltage traces are drawn.
    channel : ephys channel to read off the Trial (default ``voltage_1``)
    show : if True, call ``plt.show()`` before returning
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    sd = _import_spikedetect()
    from spikedetect import SignalFilter

    if result is None:
        result = load_spikes_from_trial(trial, field=default_field(channel))

    rec = trial_to_recording(trial, channel=channel)
    t = np.arange(rec.n_samples) / rec.sample_rate

    fig = plt.figure(figsize=figsize)
    try:
        fig.canvas.manager.set_window_title(
            f"{trial._dfc} trial {int(trial.params['trial'])} — {channel}"
        )
    except Exception:
        pass
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1.0, 1.0, 1.2])
    ax_raw = fig.add_subplot(gs[0, :])
    ax_filt = fig.add_subplot(gs[1, :], sharex=ax_raw)
    ax_wf = fig.add_subplot(gs[2, 0])
    ax_isi = fig.add_subplot(gs[2, 1])

    ax_raw.plot(t, rec.voltage, linewidth=0.5, color="0.3")
    ax_raw.set_ylabel(channel)
    title = (f"{rec.name} — {rec.duration:.1f} s @ {rec.sample_rate:g} Hz")
    if result is not None:
        title += f" — {result.n_spikes} spikes"
        if getattr(result, "spot_checked", False):
            title += "  (spot-checked)"
    ax_raw.set_title(title)

    # Filtered trace (requires params)
    if result is not None and result.params.spike_template is not None:
        p = result.params
        filtered = SignalFilter.filter_data(
            rec.voltage, fs=rec.sample_rate,
            hp_cutoff=p.hp_cutoff, lp_cutoff=p.lp_cutoff,
            diff_order=p.diff_order, polarity=p.polarity,
        )
        ax_filt.plot(t, filtered, linewidth=0.5, color="0.3")
        ax_filt.set_ylabel(
            f"filtered (hp={p.hp_cutoff:g}, lp={p.lp_cutoff:g}, "
            f"diff={p.diff_order}, pol={p.polarity:+d})"
        )
    else:
        ax_filt.text(0.5, 0.5,
                     "filtered trace unavailable\n(no params / no template)",
                     ha="center", va="center", transform=ax_filt.transAxes,
                     color="gray")
        ax_filt.set_axis_off()
    ax_filt.set_xlabel("time (s)")

    # Spike markers
    if result is not None and result.n_spikes > 0:
        spike_t = result.spike_times / result.params.fs
        for ax in (ax_raw, ax_filt):
            if ax.has_data():
                ylo, yhi = ax.get_ylim()
                ax.vlines(spike_t, ylo, yhi,
                          colors="red", alpha=0.25, linewidth=0.5)

    # Waveform overlay
    if result is not None and result.n_spikes > 0:
        stw = result.params.spike_template_width
        half = stw // 2
        wfs = []
        for s in result.spike_times:
            a, b = int(s) - half, int(s) + half + 1
            if a >= 0 and b <= rec.n_samples:
                wfs.append(rec.voltage[a:b])
        if wfs:
            W = np.vstack(wfs)
            t_wf = (np.arange(W.shape[1]) - half) / rec.sample_rate * 1000
            ax_wf.plot(t_wf, W.T, linewidth=0.3, color="0.3", alpha=0.2)
            ax_wf.plot(t_wf, W.mean(0), linewidth=1.5, color="red")
            ax_wf.set_xlabel("ms from spike")
            ax_wf.set_ylabel("voltage")
            ax_wf.set_title(f"waveforms (n={len(wfs)})")
        else:
            ax_wf.text(0.5, 0.5, "no full-width waveforms", ha="center",
                       va="center", transform=ax_wf.transAxes, color="gray")
            ax_wf.set_axis_off()
    else:
        ax_wf.text(0.5, 0.5, "no spikes", ha="center", va="center",
                   transform=ax_wf.transAxes, color="gray")
        ax_wf.set_axis_off()

    # ISI histogram
    if result is not None and result.n_spikes >= 2:
        isi_ms = np.diff(result.spike_times) / result.params.fs * 1000
        upper = max(20.0, np.percentile(isi_ms, 99) * 1.2)
        ax_isi.hist(isi_ms, bins=np.linspace(0, upper, 60), color="0.4")
        ax_isi.set_xlabel("ISI (ms)")
        ax_isi.set_ylabel("count")
        ax_isi.set_title(
            f"ISIs — mean {isi_ms.mean():.1f} ms "
            f"({1000 / isi_ms.mean():.1f} Hz)"
        )
    else:
        ax_isi.text(0.5, 0.5, "<2 spikes", ha="center", va="center",
                    transform=ax_isi.transAxes, color="gray")
        ax_isi.set_axis_off()

    fig.tight_layout()
    if show:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# Defaults + precedence-chain initialization
# ---------------------------------------------------------------------------

def load_default_params() -> dict:
    """Read ``mapd/spike_detection_defaults.json`` and return the scalar
    params dict.

    Returns an empty dict if the file is missing. Only scalar fields are
    stored here — ``fs`` and ``spike_template`` are intentionally excluded
    because they are per-recording / per-electrode.
    """
    if not _DEFAULTS_PATH.exists():
        return {}
    with open(_DEFAULTS_PATH) as f:
        blob = json.load(f)
    return dict(blob.get("params", {}))


def _cell_dir(cell_id: str) -> Path:
    """Return the cell's data directory: ``<data_root>/<day>/<cell_id>/``."""
    from mapd.paths import default_data_directory
    day = cell_id.split("_")[0]
    return Path(default_data_directory(verbose=False)) / day / cell_id


def _cell_params_path(cell_id: str, field: str = "spikes") -> Path:
    """Sidecar JSON for per-cell params.

    Mirrors the ``Acquisition_<cell>.mat`` naming convention and lives
    alongside the trial .mat files in the cell's data directory. Travels
    with the data rather than living in the user's home. A field other than
    ``spikes`` gets its own file, ``spike_detection_<cell>_<field>.json``.
    """
    sfx = "" if field == "spikes" else f"_{field}"
    return _cell_dir(cell_id) / f"spike_detection_{cell_id}{sfx}.json"


def save_cell_params(cell_id: str, params, field: str = "spikes",
                     channel: Optional[str] = None) -> Path:
    """Persist per-cell SpikeDetectionParams into the cell's data directory.

    Written as ``spike_detection_<cell_id>.json`` via ``params.to_dict()``
    — loaded back by :func:`load_cell_params`. Stored in the cell's data
    folder so it travels with the recording and any collaborator reading
    the directory sees the tuning.
    """
    path = _cell_params_path(cell_id, field)
    if not path.parent.is_dir():
        raise FileNotFoundError(
            f"Cell directory does not exist: {path.parent}. "
            "save_cell_params is meant to be called on a real cell that has "
            "already been recorded."
        )
    d = params.to_dict()
    if channel is not None:
        d["channel"] = channel
    path.write_text(json.dumps(d, indent=2))
    return path


def load_cell_params(cell_id: str, field: str = "spikes"):
    """Load per-cell SpikeDetectionParams or return None if not saved."""
    sd = _import_spikedetect()
    path = _cell_params_path(cell_id, field)
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    d.pop("channel", None)
    return sd.SpikeDetectionParams.from_dict(d)


def cell_params_channel(cell_id: str, field: str = "spikes") -> Optional[str]:
    """The channel last tuned for ``field`` on this cell, from its JSON."""
    path = _cell_params_path(cell_id, field)
    if not path.exists():
        return None
    return json.loads(path.read_text()).get("channel")


def _robust_peak_threshold(trial, channel: str, p, k: float = 5.0) -> float:
    """``k`` robust SDs (MAD) of ``channel`` after ``p``'s filter.

    ``peak_threshold`` is an absolute offset in filtered units, so a value
    tuned on Vm in mV means nothing on an EMG channel whose raw scale is
    ~100x smaller; this gives an EMG channel a starting point on its own scale.
    """
    from spikedetect import SignalFilter
    v = np.asarray(getattr(trial, channel), dtype=np.float64).ravel()
    filt = SignalFilter.filter_data(v, fs=p.fs, hp_cutoff=p.hp_cutoff,
                                    lp_cutoff=p.lp_cutoff,
                                    diff_order=p.diff_order, polarity=p.polarity)
    filt = filt[max(100, round(0.01 * p.fs)):]
    mad = float(np.median(np.abs(filt - np.median(filt))))
    return k * 1.4826 * mad


def initial_params(trial, fs: Optional[float] = None,
                   cell_id: Optional[str] = None, field: str = "spikes",
                   channel: Optional[str] = None):
    """Build a starting ``SpikeDetectionParams`` with sensible precedence.

    Precedence, highest first:
      1. ``/spikeDetectionParams`` already saved on the trial (MATLAB or
         a previous Python tuning run) — also carries ``spike_template``
         if present.
      2. ``<data_root>/<day>/<cell_id>/spike_detection_<cell_id>.json``
         (written by :func:`save_cell_params`), if ``cell_id`` is given.
      3. ``mapd/spike_detection_defaults.json`` (project-wide defaults).
      4. ``SpikeDetectionParams.default(fs)`` — stock fallback.

    ``fs`` is always stamped from the recording (or from the ``fs`` kwarg
    if passed) so you never inherit a stale sample rate.

    ``field`` picks which stored result and which per-cell JSON steps 1-2
    read. For a field other than ``spikes`` (EMG), step 3's peak threshold
    -- tuned on Vm -- is replaced by 5 robust SDs of ``channel`` filtered.
    """
    sd = _import_spikedetect()

    if fs is None:
        fs = float(trial.params["sampratein"])

    # 1) trial's own params group for this field
    res = load_spikes_from_trial(trial, field=field)
    if res is not None:
        p = copy.copy(res.params)
        p.fs = float(fs)
        return p

    # 2) per-cell JSON
    if cell_id is not None:
        p = load_cell_params(cell_id, field=field)
        if p is not None:
            p.fs = float(fs)
            return p

    # 3) project-level defaults
    default = load_default_params()
    p = sd.SpikeDetectionParams.default(fs=float(fs))
    for k, v in default.items():
        if hasattr(p, k):
            setattr(p, k, v)
    if field != "spikes" and channel is not None:
        p.peak_threshold = _robust_peak_threshold(trial, channel, p)
    return p


def _write_params_group(fp: str, p, field: str = "spikes",
                        channel: Optional[str] = None) -> None:
    """Delete and rewrite ``field``'s params group in the trial .mat file
    (/spikeDetectionParams/ for ``spikes``, /EMGspikeDetectionParams/ for
    ``EMGspikes``). ``channel`` is stored in the group when given.

    Does not touch the spike datasets themselves.
    """
    grp = field_keys(field)["params"]
    with h5py.File(fp, "r+") as f:
        if grp in f:
            if channel is None and "channel" in f[grp]:
                from spikedetect.io.mat import _read_h5_string
                channel = _read_h5_string(f[grp]["channel"])
            del f[grp]

    if channel is not None:
        h5s.write(data=str(channel), path=f"/{grp}/channel", filename=fp,
                  matlab_compatible=True, store_python_metadata=False)
    for py_name, mat_name in _PARAM_SCALAR_FIELDS:
        h5s.write(
            data=float(getattr(p, py_name)),
            path=f"/{grp}/{mat_name}",
            filename=fp,
            matlab_compatible=True,
            store_python_metadata=False,
        )
    if p.spike_template is not None:
        h5s.write(
            data=np.asarray(p.spike_template, dtype=np.float64).reshape(-1, 1),
            path=f"/{grp}/spikeTemplate",
            filename=fp,
            matlab_compatible=True,
            store_python_metadata=False,
        )
    if p.likely_inflection_point_peak is not None:
        h5s.write(
            data=float(p.likely_inflection_point_peak),
            path=f"/{grp}/likelyiflpntpeak",
            filename=fp,
            matlab_compatible=True,
            store_python_metadata=False,
        )
    # Newer SpikeDetectionParams fields. Persist them when set so that
    # template freshness checks and refractory tuning survive a round trip
    # through the trial .mat file.
    if getattr(p, "template_updated_at", None) is not None:
        h5s.write(
            data=p.template_updated_at.isoformat(),
            path=f"/{grp}/templateUpdatedAt",
            filename=fp,
            matlab_compatible=True,
            store_python_metadata=False,
        )
    if getattr(p, "min_isi_samples", None) is not None:
        h5s.write(
            data=float(p.min_isi_samples),
            path=f"/{grp}/min_isi_samples",
            filename=fp,
            matlab_compatible=True,
            store_python_metadata=False,
        )


def _delete_spike_datasets(fp: str, field: str) -> None:
    k = field_keys(field)
    with h5py.File(fp, "r+") as f:
        for key in (k["spikes"], k["uncorrected"], k["spot"]):
            if key in f:
                del f[key]


def save_params_to_trial(trial, params, invalidate_spikes: bool = True,
                         field: str = "spikes",
                         channel: Optional[str] = None) -> None:
    """Persist only ``field``'s params group into the trial .mat file.

    Use during tuning, when you have good params but haven't run full
    detection yet.

    By default (``invalidate_spikes=True``), that field's pre-existing
    spikes, uncorrected spikes and spot-check flag are also deleted — the
    rationale being that if params have changed, the old spikes are
    stale. Pass ``invalidate_spikes=False`` to preserve them (e.g. when
    the params you're writing match the ones the existing spikes were
    detected with). Other fields are never touched.
    """
    fp = trial.file_path
    if invalidate_spikes:
        _delete_spike_datasets(fp, field)
    _write_params_group(fp, params, field=field, channel=channel)


def save_spikes_to_trial(trial, result, spot_checked: Optional[bool] = None,
                         field: str = "spikes",
                         channel: Optional[str] = None) -> None:
    """Write a SpikeDetectionResult into trial.file_path in place.

    Writes MATLAB-compatible datasets at the top level of the trial's HDF5,
    named after ``field`` (see :func:`field_keys`); for ``spikes``:

        /spikes                 (Nx1 double, 0-based sample indices)
        /spikes_uncorrected     (Nx1 double)
        /spikeSpotChecked       (1x1 double, 0 or 1)
        /spikeDetectionParams/  (group: fs, hp_cutoff, lp_cutoff, diff,
                                 peak_threshold, Distance_threshold,
                                 Amplitude_threshold, polarity,
                                 spikeTemplateWidth, spikeTemplate,
                                 channel, ...)

    and EMGspikes / EMGspikes_uncorrected / EMGspikeSpotChecked /
    EMGspikeDetectionParams for ``EMGspikes``. :func:`load_spikes_from_trial`
    reads them back. ``channel=None`` keeps the channel already recorded.
    """
    fp = trial.file_path
    k = field_keys(field)

    # Drop any prior spike data so we don't mix old and new
    _delete_spike_datasets(fp, field)

    spikes = np.asarray(result.spike_times, dtype=np.float64).reshape(-1, 1)
    spikes_u = np.asarray(
        result.spike_times_uncorrected, dtype=np.float64,
    ).reshape(-1, 1)
    sc = bool(result.spot_checked if spot_checked is None else spot_checked)

    h5s.write(data=spikes, path=f"/{k['spikes']}", filename=fp,
              matlab_compatible=True, store_python_metadata=False)
    h5s.write(data=spikes_u, path=f"/{k['uncorrected']}", filename=fp,
              matlab_compatible=True, store_python_metadata=False)
    h5s.write(data=float(sc), path=f"/{k['spot']}", filename=fp,
              matlab_compatible=True, store_python_metadata=False)

    _write_params_group(fp, result.params, field=field, channel=channel)


# ---------------------------------------------------------------------------
# Single-trial and batch detection
# ---------------------------------------------------------------------------

def _params_with_fs(params, fs: float):
    q = copy.copy(params)
    q.fs = float(fs)
    return q


def tune_params_on_trials(seed_trial, tune_trials, *, channel: str = "voltage_1",
                           cell_id: Optional[str] = None, save: bool = True,
                           field: Optional[str] = None):
    """Run interactive tuning on the concatenated voltage of ``tune_trials``.

    Walks the user through ``FilterGUIQt → TemplateSelectionGUIQt →
    ThresholdGUIQt`` on one synthetic Recording made by concatenating
    ``tune_trials``. ``seed_trial`` seeds the param precedence chain
    (its own /spikeDetectionParams → per-cell JSON → project defaults).
    All trials must share a sample rate.

    When ``save=True``: the tuned params are written to the per-cell
    JSON (if ``cell_id`` is given) and to the seed trial's
    /spikeDetectionParams. Spikes are NOT detected here — call
    :func:`batch_detect_spikes` or :func:`detect_spikes_for_trial`
    afterwards.

    Mirrors the ``scripts/tune_trials.py`` workflow but takes already-
    loaded Trial objects so callers (e.g. the trial browser) don't have
    to re-walk the file system.

    ``field`` (default: :func:`default_field` of ``channel``) is where the
    params are read from and saved to, so tuning an EMG channel never
    touches the intracellular ``spikes`` result.
    """
    sd = _import_spikedetect()
    field = field or default_field(channel)
    from spikedetect import Recording, SignalFilter
    from spikedetect.gui import (
        FilterGUIQt, TemplateSelectionGUIQt, ThresholdGUIQt,
    )

    if not tune_trials:
        raise ValueError("tune_trials must contain at least one trial")

    voltages = []
    fs = None
    for tr in tune_trials:
        rec_i = trial_to_recording(tr, channel=channel)
        if fs is None:
            fs = rec_i.sample_rate
        elif rec_i.sample_rate != fs:
            raise ValueError(
                f"Sample-rate mismatch: trial {int(tr.params['trial'])} is "
                f"{rec_i.sample_rate} Hz, expected {fs} Hz"
            )
        voltages.append(rec_i.voltage)
    voltage = np.concatenate(voltages)

    first_tn = int(tune_trials[0].params["trial"])
    last_tn = int(tune_trials[-1].params["trial"])
    rec = Recording(
        name=f"{seed_trial._dfc}_tune_tr{first_tn}-{last_tn}",
        voltage=voltage, sample_rate=float(fs),
    )

    params = initial_params(seed_trial, fs=fs, cell_id=cell_id, field=field,
                            channel=channel)

    params = FilterGUIQt(rec.voltage, params).run()
    filtered = SignalFilter.filter_data(
        rec.voltage, fs=float(fs),
        hp_cutoff=params.hp_cutoff, lp_cutoff=params.lp_cutoff,
        diff_order=params.diff_order, polarity=params.polarity,
    )
    # Direct ``params.spike_template = ...`` assignment bypasses the
    # template-freshness clock per spikedetect's contract — the GUI
    # instance stamps the timestamp on itself, and the caller is
    # responsible for copying it onto params before persisting.
    template_gui = TemplateSelectionGUIQt(filtered, params)
    params.spike_template = template_gui.run()
    params.template_updated_at = template_gui.template_updated_at
    if params.spike_template is None:
        raise RuntimeError(
            "No spike template selected — aborting before threshold tuning"
        )
    match = match_candidates(rec, params)
    params = ThresholdGUIQt(match_result=match, params=params).run()

    if save:
        if cell_id is not None:
            save_cell_params(cell_id, params, field=field, channel=channel)
        save_params_to_trial(seed_trial, params, field=field, channel=channel)
    return params


def detect_spikes_for_trial(trial, params, channel: str = "voltage_1",
                             save: bool = False, field: Optional[str] = None,
                             correct_times: bool = True):
    """Run the full detection pipeline on a single mapd.Trial.

    `params.fs` is snapped to the trial's actual sample rate; the caller's
    params object is not modified. ``field`` defaults to
    :func:`default_field` of ``channel`` (EMGspikes for a current channel).

    ``correct_times=False`` keeps the filtered-peak times as the spike times
    instead of moving each one to the inflection point of its rising phase.
    That correction was built for the intracellular spike shape; an EMG unit
    can be biphasic, so check it in the spot-check before trusting it there.
    Both versions are always stored (``*_uncorrected`` holds the peaks).
    """
    sd = _import_spikedetect()
    field = field or default_field(channel)
    rec = trial_to_recording(trial, channel=channel)
    result = sd.detect_spikes(rec, _params_with_fs(params, rec.sample_rate))
    if not correct_times:
        result.spike_times = result.spike_times_uncorrected.copy()
    if save:
        save_spikes_to_trial(trial, result, field=field, channel=channel)
    return result


def match_candidates(rec, params, start_offset: float = 0.01):
    """Run filter → peak-find → template-match on one Recording.

    Returns a ``spikedetect.pipeline.template.TemplateMatchResult`` — the
    intermediate object consumed by ``ThresholdGUI``. Use this when you
    want to tune DTW / amplitude thresholds interactively without running
    the full detection pipeline twice.

    Same preprocessing conventions as ``spikedetect.SpikeDetector.detect``:
    skips ``round(start_offset * fs)`` samples at the start and clamps an
    absurd ``peak_threshold`` to ``3 * std(filtered)``.
    """
    _import_spikedetect()
    from spikedetect import SignalFilter, PeakFinder, TemplateMatcher

    if params.spike_template is None:
        raise ValueError(
            "params.spike_template is None — run TemplateSelectionGUI first."
        )
    stw = len(params.spike_template)
    fs = rec.sample_rate
    start_point = round(start_offset * fs)
    unfiltered = rec.voltage[start_point:]
    filtered = SignalFilter.filter_data(
        unfiltered, fs=fs,
        hp_cutoff=params.hp_cutoff, lp_cutoff=params.lp_cutoff,
        diff_order=params.diff_order, polarity=params.polarity,
    )
    pthr = params.peak_threshold
    if pthr > 1e4 * np.std(filtered):
        pthr = 3 * np.std(filtered)
    locs = PeakFinder.find_spike_locations(
        filtered, peak_threshold=pthr, fs=fs, spike_template_width=stw,
    )
    return TemplateMatcher.match(
        locs, params.spike_template, filtered, unfiltered, stw, fs,
    )


def collect_candidates(trials: Iterable, params, channel: str = "voltage_1",
                        start_offset: float = 0.01) -> pd.DataFrame:
    """Gather every candidate peak across a list of trials.

    Runs the filter, peak finder, and template matcher on each trial but
    does NOT apply the DTW / amplitude acceptance thresholds. Use this on
    a subset of ~10–50 trials to see the real spread of
    (dtw_distance, amplitude) in this cell before choosing thresholds.

    Same preprocessing conventions as ``spikedetect.SpikeDetector.detect``:
    - skips ``round(start_offset * fs)`` samples at the start of each trial
    - clamps an absurd ``peak_threshold`` to ``3 * std(filtered)``

    Returns a DataFrame with columns:
        trial          (int) — trial.params['trial']
        sample         (int) — 0-based index into the full voltage trace
        dtw_distance   (float)
        amplitude      (float)
        accepted       (bool) — would pass the current thresholds
    """
    _import_spikedetect()
    from spikedetect import SignalFilter, PeakFinder, TemplateMatcher

    if params.spike_template is None:
        raise ValueError(
            "params.spike_template is None — seed it first by running "
            "TemplateSelectionGUI on one trial."
        )
    stw = len(params.spike_template)
    dthr = params.distance_threshold
    athr = params.amplitude_threshold

    rows = []
    for tr in trials:
        rec = trial_to_recording(tr, channel=channel)
        fs = rec.sample_rate
        start_point = round(start_offset * fs)
        unfiltered = rec.voltage[start_point:]
        filtered = SignalFilter.filter_data(
            unfiltered, fs=fs,
            hp_cutoff=params.hp_cutoff, lp_cutoff=params.lp_cutoff,
            diff_order=params.diff_order, polarity=params.polarity,
        )
        pthr = params.peak_threshold
        if pthr > 1e4 * np.std(filtered):
            pthr = 3 * np.std(filtered)
        locs = PeakFinder.find_spike_locations(
            filtered, peak_threshold=pthr, fs=fs, spike_template_width=stw,
        )
        if len(locs) == 0:
            continue
        m = TemplateMatcher.match(
            locs, params.spike_template, filtered, unfiltered, stw, fs,
        )
        tnum = int(tr.params["trial"])
        for loc, d, a in zip(m.spike_locs, m.dtw_distances, m.amplitudes):
            rows.append((tnum, int(loc) + start_point, float(d), float(a)))

    df = pd.DataFrame(
        rows, columns=["trial", "sample", "dtw_distance", "amplitude"],
    )
    df["accepted"] = (df["dtw_distance"] < dthr) & (df["amplitude"] > athr)
    return df


def batch_detect_spikes(trials: Iterable, params, channel: str = "voltage_1",
                         save: bool = False, verbose: bool = True,
                         field: Optional[str] = None, correct_times: bool = True):
    """Run full detection on a list of mapd.Trials (see
    :func:`detect_spikes_for_trial` for ``field`` and ``correct_times``).

    Returns
    -------
    results : dict[int, spikedetect.SpikeDetectionResult] keyed by trial number
    summary : DataFrame with columns trial, n_spikes, rate_hz, duration_s
    """
    results = {}
    rows = []
    for tr in trials:
        r = detect_spikes_for_trial(tr, params, channel=channel, save=save,
                                    field=field, correct_times=correct_times)
        tnum = int(tr.params["trial"])
        dur = float(tr.params["samples"]) / float(tr.params["sampratein"])
        rate = r.n_spikes / dur if dur > 0 else np.nan
        results[tnum] = r
        rows.append((tnum, r.n_spikes, rate, dur))
        if verbose:
            print(f"  trial {tnum:>4}: {r.n_spikes:>5} spikes  ({rate:.1f} Hz)")
    summary = pd.DataFrame(
        rows, columns=["trial", "n_spikes", "rate_hz", "duration_s"],
    )
    return results, summary
