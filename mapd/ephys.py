"""
mapd/ephys.py
=============
Signal-processing primitives for electrophysiology time series stored on
``Trial`` objects.

Currently provides smoothed firing-rate estimation from the spike trains
written by :mod:`mapd.spike_detection`. Mirrors the role of
:mod:`mapd.kinematics` for probe-position traces.
"""
from __future__ import annotations

import numpy as np

from . import spike_detection as sds


# ── Default smoothing parameters ──────────────────────────────────────────────
SIGMA_S = 0.025   # 25 ms — full kernel ≈ 100 ms (±2σ)
GRID_DT = 0.001   # 1 ms internal grid for binning before convolution

# ── Subthreshold Vm defaults ──────────────────────────────────────────────────
# Blanking window around each spike, in seconds relative to the detected peak.
# These are a fallback: prefer ``spike_blank_window`` / ``sta_blank_window``,
# which measure the window a cell's own spike-triggered average actually needs.
VM_BLANK_PRE_S  = 0.0015  # 1.5 ms before the peak
VM_BLANK_POST_S = 0.005   # 5 ms after — covers the after-hyperpolarization
VM_SIGMA_S      = 0.025   # smoothing of the blanked trace, matched to SIGMA_S


def _gaussian_kernel(sigma_samples: float, truncate: float = 4.0) -> np.ndarray:
    """Discretized, unit-area Gaussian. ``truncate`` is in units of σ."""
    radius = int(np.ceil(truncate * sigma_samples))
    x = np.arange(-radius, radius + 1)
    k = np.exp(-0.5 * (x / sigma_samples) ** 2)
    return k / k.sum()


def gaussian_rate(
    spike_times_s: np.ndarray,
    t_axis: np.ndarray,
    sigma_s: float = SIGMA_S,
    grid_dt: float = GRID_DT,
) -> np.ndarray:
    """Gaussian-smoothed firing rate (Hz) sampled at ``t_axis``.

    Spike times are binned onto a uniform internal grid of width ``grid_dt``,
    convolved with a Gaussian of standard deviation ``sigma_s``, divided by
    the bin width to convert counts → Hz, then linearly interpolated to
    ``t_axis``. The internal grid is independent of ``t_axis`` so the result
    is unbiased even when ``t_axis`` is irregular (e.g. camera frame times).

    Parameters
    ----------
    spike_times_s : array-like
        Spike times in seconds, in the same time frame as ``t_axis``.
    t_axis : array-like
        Output sample times in seconds (monotonic, but spacing may vary).
    sigma_s : float
        Gaussian standard deviation in seconds. Default 25 ms (≈100 ms FWHM).
    grid_dt : float
        Internal binning resolution in seconds. Should be ≪ ``sigma_s``.
    """
    t_axis = np.asarray(t_axis, dtype=float)
    spike_times_s = np.asarray(spike_times_s, dtype=float)
    if t_axis.size == 0:
        return np.zeros(0)

    t0, t1 = float(t_axis[0]), float(t_axis[-1])
    if t1 <= t0:
        return np.zeros_like(t_axis)

    # Pad the grid by ~4σ on each side so edge spikes contribute correctly.
    pad = 4.0 * sigma_s
    edges = np.arange(t0 - pad, t1 + pad + grid_dt, grid_dt)
    if spike_times_s.size:
        counts, _ = np.histogram(spike_times_s, bins=edges)
    else:
        counts = np.zeros(edges.size - 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    kernel = _gaussian_kernel(sigma_s / grid_dt)
    smoothed = np.convolve(counts, kernel, mode="same") / grid_dt  # Hz

    return np.interp(t_axis, centers, smoothed)


def spike_rate_from_trial(
    trial,
    t_axis: np.ndarray | None = None,
    sigma_s: float = SIGMA_S,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Load detected spikes from ``trial`` and return ``(t, rate_hz)``.

    Returns ``None`` when the trial has no detected spikes (or detection
    has not been run).

    The trial-time frame is stimulus-centered (``trial.time[0] = -pre_dur``);
    spike sample indices are converted to seconds and shifted into that
    same frame before smoothing.

    Parameters
    ----------
    trial : Trial
    t_axis : array-like, optional
        Output time axis in seconds. Defaults to ``trial.time``.
    sigma_s : float
        Gaussian σ in seconds. Default 25 ms.
    """
    try:
        result = sds.load_spikes_from_trial(trial)
    except Exception:
        return None
    if result is None or getattr(result, "n_spikes", 0) == 0:
        return None

    if t_axis is None:
        t_axis = np.asarray(trial.time).ravel()
    else:
        t_axis = np.asarray(t_axis, dtype=float)

    if t_axis.size == 0:
        return None

    t0 = float(np.asarray(trial.time).ravel()[0])
    spike_s = np.asarray(result.spike_times) / float(result.params.fs) + t0
    rate = gaussian_rate(spike_s, t_axis, sigma_s=sigma_s)
    return t_axis, rate


# ── Boxcar (sliding-count) rate ───────────────────────────────────────────────

def boxcar_rate(
    spike_times_s: np.ndarray,
    t_axis: np.ndarray,
    window_s: float,
) -> np.ndarray:
    """Spike count in a centered window of width ``window_s``, divided by that
    width — i.e. the plain average rate over ±``window_s``/2, in Hz.

    Computed exactly by ``searchsorted`` on the spike times rather than by
    convolving a binned grid, so the result is the true count with no binning
    error. That matters for the noise-floor comparison: for a Poisson process
    of rate *r*, the count in a window of width *W* has variance *rW*, so this
    estimator has sd ``sqrt(r / W)`` — the reference curve any "does more
    averaging tighten the cloud?" analysis has to beat.

    Unlike :func:`gaussian_rate`, no smoothing kernel is applied, so the
    effective averaging time is exactly ``window_s``.
    """
    t_axis = np.asarray(t_axis, dtype=float)
    if t_axis.size == 0:
        return np.zeros(0)
    st = np.sort(np.asarray(spike_times_s, dtype=float).ravel())
    half = 0.5 * float(window_s)
    lo = np.searchsorted(st, t_axis - half, side='left')
    hi = np.searchsorted(st, t_axis + half, side='right')
    return (hi - lo) / float(window_s)


def spikes_per_sample(spike_times_s: np.ndarray, t_axis: np.ndarray) -> np.ndarray:
    """Spike *count* attributed to each sample of ``t_axis``.

    Bin edges are the midpoints between consecutive samples (the first and last
    edges extrapolated by half a step), so every spike inside the axis span
    lands in exactly one bin and the counts sum to the number of spikes in that
    span. Downstream, the exact count over any window is the sum of these — which
    is what makes the Poisson noise floor exact rather than approximate. Spikes
    outside the axis span are not counted.
    """
    t_axis = np.asarray(t_axis, dtype=float)
    if t_axis.size == 0:
        return np.zeros(0, dtype=float)
    st = np.asarray(spike_times_s, dtype=float).ravel()
    if t_axis.size == 1:
        return np.array([float(st.size)])
    mid = 0.5 * (t_axis[:-1] + t_axis[1:])
    edges = np.concatenate([[t_axis[0] - (mid[0] - t_axis[0])],
                            mid,
                            [t_axis[-1] + (t_axis[-1] - mid[-1])]])
    counts, _ = np.histogram(st, bins=edges)
    return counts.astype(float)


def trial_spike_times(trial, allow_empty: bool = False) -> np.ndarray | None:
    """Detected spike times of ``trial``, in the trial's own (stimulus-centered)
    time frame.

    Returns ``None`` when detection has not been run. By default it ALSO returns
    None when detection ran and found nothing, which conflates two very
    different states -- and the conflation is not harmless on a quiet cell.
    ``build_cell_records`` drops any trial this returns None for, so on the A4
    set that discarded 306 of 620 usable trials on 241115_F1_C1, 324 of 542 on
    210917_F2_C1 and 373 of 584 on 241203_F2_C1. Every one of those had
    detection run and genuinely find zero spikes (checked on 241115: 123 of the
    first 250 trials, none missing a detection result). They are real trials in
    which the neuron never fired, so dropping them conditions every rate and
    every "fraction of bouts silent" on "trials where the cell fired at least
    once" -- which biases exactly the question an A4 analysis is asking.

    ``allow_empty=True`` returns an empty array for "ran, found none" and keeps
    None for "never ran", so a caller can keep the silent trials without
    silently swallowing undetected ones.
    """
    try:
        result = sds.load_spikes_from_trial(trial)
    except Exception:
        return None
    if result is None:
        return None                      # detection never ran
    if getattr(result, 'n_spikes', 0) == 0:
        return np.empty(0, dtype=float) if allow_empty else None
    t0 = float(np.asarray(trial.time).ravel()[0])
    return np.asarray(result.spike_times) / float(result.params.fs) + t0


def padded_spike_times(trial, prev_trial=None, next_trial=None,
                       allow_empty: bool = False):
    """``trial``'s spike times with the neighbours' spikes appended, in *its* clock.

    Recording is continuous across consecutive trials, so the spikes either side
    of a trial boundary are real data that simply live in another file. Adding
    them gives a smoothing kernel full support right up to the trial edges — the
    alternative is trimming, which throws away good samples (3 sigma is 750 ms
    per end for a 250 ms kernel, ~11% of a 14 s trial).

    Each trial's clock is stimulus-centred, and consecutive trials butt together
    the way :func:`mapd.bout_analysis.trace_across_trials` assumes: trial *k+1*
    begins where trial *k* ends. So in ``trial``'s frame the previous trial's
    spikes sit at ``s - prev.total_duration`` and the next trial's at
    ``s + trial.total_duration``.

    Pass only genuinely adjacent, usable neighbours — a neighbour that was
    excluded, has bad ephys, or is not the consecutive trial number would splice
    unrelated spikes across the boundary. ``mapd.bout_analysis.neighbour_trials``
    applies those rules.

    Returns the concatenated, sorted times, or ``None`` if ``trial`` has no
    spikes of its own.

    ``allow_empty`` matters once silent trials are kept (``keep_silent`` in
    per_frame_records). Without it a trial that fired nothing returns None here,
    the caller falls back to that trial's own empty spike list, and the
    NEIGHBOURS' spikes are dropped -- while ``rate_supported`` stays True, so no
    trim is applied either. The trial's rate then reads zero at its edges where
    it should carry the tail of a neighbour's spike. With it, a silent trial
    still gets its neighbours' support.
    """
    own = trial_spike_times(trial, allow_empty=allow_empty)
    if own is None:
        return None
    parts = [own]
    if prev_trial is not None:
        s = trial_spike_times(prev_trial, allow_empty=allow_empty)
        if s is not None:
            parts.append(s - float(prev_trial.total_duration))
    if next_trial is not None:
        s = trial_spike_times(next_trial, allow_empty=allow_empty)
        if s is not None:
            parts.append(s + float(trial.total_duration))
    return np.sort(np.concatenate(parts))


def _trial_voltage(trial, channel: str = 'voltage_1'):
    """``(voltage, fs, t0)`` for one ephys channel, or ``None``."""
    v = getattr(trial, channel, None)
    if v is None:
        return None
    v = np.asarray(v, dtype=float).ravel()
    if v.size == 0:
        return None
    fs = float(trial.params['sampratein'])
    t0 = float(np.asarray(trial.time).ravel()[0])
    return v, fs, t0


def padded_voltage(trial, prev_trial=None, next_trial=None,
                   channel: str = 'voltage_1'):
    """Voltage of ``(prev, trial, next)`` concatenated, on ``trial``'s clock.

    Safe because the acquisition is one continuous recording that is split into
    per-trial records afterwards — verified via ``startsample``, where each trial
    begins exactly one sample after the previous one ends. So there is no DC step
    to match at the joins and no missing sample: concatenation reproduces the
    original continuous trace.

    Neighbours are skipped if their sample rate differs. Returns
    ``(v, fs, t_axis, own_slice)`` where ``own_slice`` indexes ``trial``'s own
    samples — needed to report statistics (e.g. the blanked fraction) over the
    trial itself rather than over the padding.
    """
    vv = _trial_voltage(trial, channel)
    if vv is None:
        return None
    v, fs, t0 = vv
    before, after = [], []
    t_start = t0
    if prev_trial is not None:
        pv = _trial_voltage(prev_trial, channel)
        if pv is not None and abs(pv[1] - fs) < 1e-6:
            before.append(pv[0])
            t_start = pv[2] - float(prev_trial.total_duration)
    if next_trial is not None:
        nv = _trial_voltage(next_trial, channel)
        if nv is not None and abs(nv[1] - fs) < 1e-6:
            after.append(nv[0])
    n_before = sum(a.size for a in before)
    v_all = np.concatenate(before + [v] + after) if (before or after) else v
    t_all = t_start + np.arange(v_all.size) / fs
    return v_all, fs, t_all, slice(n_before, n_before + v.size)


# ── Spike-triggered average of the voltage (to size the blanking window) ──────

def spike_triggered_average(
    trials,
    channel: str = 'voltage_1',
    pre_s: float = 0.006,
    post_s: float = 0.030,
    max_spikes: int = 2000,
    baseline_s: float = 0.002,
):
    """Average voltage waveform aligned to detected spike peaks.

    The point is measurement rather than display: the returned waveform is what
    :func:`sta_blank_window` reads to decide how much of the trace a spike
    actually contaminates, instead of guessing a blanking window. Spike waveform
    width and the depth/duration of the after-hyperpolarization differ between
    cells, and an AHP that outlasts the blank leaks straight into "subthreshold"
    Vm as a rate-dependent bias — exactly the artifact that would manufacture a
    Vm-vs-rate correlation.

    Parameters
    ----------
    trials      : a Trial, or an iterable of Trials (waveforms are pooled).
    channel     : ephys channel to read.
    pre_s/post_s: window around the peak, in seconds.
    max_spikes  : cap on the number of waveforms pooled (evenly spaced draw,
                  not the first N, so a drifting recording isn't represented
                  only by its beginning).
    baseline_s  : the earliest ``baseline_s`` of the pre-window defines each
                  waveform's baseline, which is subtracted before averaging.
                  Per-waveform subtraction keeps slow electrode drift out of
                  the average.

    Returns
    -------
    dict with ``lags_s``, ``mean`` (mV, baseline-subtracted), ``sd``, ``sem``,
    ``n_spikes``, ``fs``. ``None`` if no usable spikes/voltage were found.
    """
    if not isinstance(trials, (list, tuple)):
        trials = [trials]
    segs, fs_seen = [], None
    for trial in trials:
        vv = _trial_voltage(trial, channel)
        st = trial_spike_times(trial)
        if vv is None or st is None:
            continue
        v, fs, t0 = vv
        fs_seen = fs if fs_seen is None else fs_seen
        n_pre, n_post = int(round(pre_s * fs)), int(round(post_s * fs))
        idx = np.round((st - t0) * fs).astype(int)
        idx = idx[(idx - n_pre >= 0) & (idx + n_post + 1 <= v.size)]
        if idx.size == 0:
            continue
        n_base = max(1, int(round(baseline_s * fs)))
        for i in idx:
            w = v[i - n_pre: i + n_post + 1]
            segs.append(w - np.mean(w[:n_base]))
    if not segs:
        return None
    if len(segs) > max_spikes:                     # evenly spaced draw
        pick = np.linspace(0, len(segs) - 1, max_spikes).round().astype(int)
        segs = [segs[i] for i in np.unique(pick)]
    W = np.stack(segs)
    n = W.shape[0]
    fs = float(fs_seen)
    n_pre = int(round(pre_s * fs))
    lags = (np.arange(W.shape[1]) - n_pre) / fs
    sd = W.std(axis=0, ddof=1) if n > 1 else np.zeros(W.shape[1])
    return {'lags_s': lags, 'mean': W.mean(axis=0), 'sd': sd,
            'sem': sd / np.sqrt(max(n, 1)), 'n_spikes': n, 'fs': fs}


def blank_window_from_trials(trials, n_trials: int = 8, channel: str = 'voltage_1',
                             tol_frac: float = 0.05, **sta_kwargs):
    """Blank window measured from an STA pooled over the first ``n_trials``.

    Pool rather than use one trial: a single trial's STA is noisy enough to
    widen the measured window substantially — on a 210915 trial it gave
    -3.0/+16.1 ms (54% of samples blanked) where eight trials gave -1.6/+9.8 ms
    (34%). Blanking half the trace to remove noise in the estimate of how much
    to blank is the wrong trade.

    Returns ``(sta, (pre_s, post_s), info)``.
    """
    usable = []
    for trial in trials:
        if trial is None or getattr(trial, 'excluded', False):
            continue
        usable.append(trial)
        if len(usable) >= n_trials:
            break
    sta = spike_triggered_average(usable, channel=channel, **sta_kwargs)
    pre_s, post_s, info = sta_blank_window(sta, tol_frac=tol_frac,
                                           return_info=True)
    return sta, (pre_s, post_s), info


def sta_blank_window(sta, tol_frac: float = 0.05, min_post_s: float = 0.001,
                     tail_frac: float = 0.25, return_info: bool = False):
    """Blanking window implied by a spike-triggered average.

    The spike is taken to contaminate the trace wherever the STA departs by more
    than ``tol_frac`` of its peak excursion, and only the *contiguous* excursion
    straddling the peak is taken (so a late unrelated wiggle cannot inflate the
    window).

    Each side is measured against its own local level, because the two differ.
    A spike-triggered average of Vm sits on a slow depolarization — the
    depolarization that *caused* the spike — which does not decay within a few
    ms, so the trace after the spike settles to a level offset from the one
    before it. The post-spike extent is therefore measured from the STA's late
    tail (median of the last ``tail_frac`` of the post window) and the pre-spike
    extent from the pre-spike baseline (zero, by construction of the STA).

    Using one level for both sides fails in both directions: the pre-spike
    baseline leaves the post side tripped forever, so the window runs to the
    edge of whatever span was measured and blanks a quarter of the recording to
    remove signal; the tail level makes the *rising phase pass through it*, so
    the walk back from the peak stops at that crossing and reports a pre-window
    of zero, leaving the spike's rising edge in "subthreshold" Vm.

    Returns ``(pre_s, post_s)``, both positive, for
    :func:`subthreshold_vm`. With ``return_info=True`` returns
    ``(pre_s, post_s, info)`` where ``info['clipped']`` flags an excursion still
    above threshold at the edge of the measured window — the window is then a
    lower bound and the STA should be re-measured over a longer span.
    ``sta=None`` gives the module defaults back.
    """
    info = {'clipped': False, 'peak_mv': np.nan, 'tail_level': np.nan,
            'from_defaults': False}
    if sta is None:
        info['from_defaults'] = True
        out = (VM_BLANK_PRE_S, VM_BLANK_POST_S)
        return (*out, info) if return_info else out
    lags, m = np.asarray(sta['lags_s']), np.asarray(sta['mean'])
    post = lags > 0
    n_tail = max(1, int(round(tail_frac * post.sum())))
    tail_level = float(np.median(m[post][-n_tail:])) if post.any() else 0.0
    pre_level = 0.0                       # the STA is baseline-subtracted there
    i_peak = int(np.argmax(np.abs(m - pre_level)))
    amp_pre = abs(m[i_peak] - pre_level)
    amp_post = abs(m[i_peak] - tail_level)
    info.update(peak_mv=float(m[i_peak]), tail_level=tail_level)
    if not np.isfinite(amp_pre) or min(amp_pre, amp_post) <= 0:
        info['from_defaults'] = True
        out = (VM_BLANK_PRE_S, VM_BLANK_POST_S)
        return (*out, info) if return_info else out

    over_pre = np.abs(m - pre_level) > tol_frac * amp_pre
    i0 = i_peak
    while i0 > 0 and over_pre[i0 - 1]:
        i0 -= 1
    over_post = np.abs(m - tail_level) > tol_frac * amp_post
    i1 = i_peak
    while i1 < len(m) - 1 and over_post[i1 + 1]:
        i1 += 1
    info['clipped'] = bool(i1 == len(m) - 1 or i0 == 0)
    pre_s = max(0.0, -float(lags[i0]))
    post_s = max(float(lags[i1]), min_post_s)
    out = (pre_s, post_s)
    return (*out, info) if return_info else out


# ── Subthreshold membrane potential ───────────────────────────────────────────

def subthreshold_vm(
    trial,
    t_axis: np.ndarray | None = None,
    channel: str = 'voltage_1',
    blank_pre_s: float = VM_BLANK_PRE_S,
    blank_post_s: float = VM_BLANK_POST_S,
    sigma_s: float = VM_SIGMA_S,
    grid_dt: float = GRID_DT,
    pad_trials=None,
):
    """Spike-blanked, smoothed membrane potential sampled at ``t_axis``.

    Pipeline: read the channel, blank ``[-blank_pre_s, +blank_post_s]`` around
    every detected spike peak, linearly interpolate across each blank, average
    onto a ``grid_dt`` grid, smooth with a Gaussian of sd ``sigma_s``, then
    interpolate to ``t_axis``. Smoothing on the coarse grid rather than at the
    acquisition rate is what keeps this cheap enough to run over every trial of
    a cell.

    Matching ``sigma_s`` to the firing-rate ``sigma_s`` matters when the two are
    to be correlated: if one signal is smoothed harder than the other, the
    correlation is limited by the wider kernel's bandwidth and the number means
    something different for every pair of settings.

    Returns ``(t_axis, vm, info)``, or ``None`` if the channel or the spike
    detection is missing. ``info`` carries

        ``frac_blanked``   fraction of samples replaced by interpolation — the
                           number that decides whether "Vm" is a measurement or
                           mostly a guess. At 100 Hz with a 6 ms window it is
                           already ~0.6, and any Vm-vs-rate relation is then
                           partly interpolation geometry.
        ``blank_pre_s`` / ``blank_post_s`` / ``sigma_s``   what was used
        ``n_spikes``, ``fs``

    With ``blank_pre_s = blank_post_s = 0`` nothing is blanked, which is the
    honest control: if the Vm-vs-rate slope survives blanking unchanged, spike
    waveform bleed-through was not driving it.
    """
    # ``pad_trials=(prev, next)`` splices the neighbours' voltage and spikes so
    # the smoothing kernel has full support at the trial edges. That matters here
    # beyond tidiness: the piezo cue lands ~210 ms into the trial, well inside the
    # 750 ms a 250 ms kernel needs, so without padding the cue response is
    # measured on the kernel's own ramp out of nothing.
    own_slice = None
    if pad_trials is not None:
        prev_trial, next_trial = pad_trials
        padded = padded_voltage(trial, prev_trial, next_trial, channel=channel)
        if padded is not None:
            v, fs, t_v_all, own_slice = padded
            t0 = float(t_v_all[0])
        else:
            own_slice = None
    if own_slice is None:
        vv = _trial_voltage(trial, channel)
        if vv is None:
            return None
        v, fs, t0 = vv
    st = (trial_spike_times(trial) if pad_trials is None
          else padded_spike_times(trial, *pad_trials))
    if st is None:
        return None

    if t_axis is None:
        t_axis = np.asarray(trial.time).ravel()
    else:
        t_axis = np.asarray(t_axis, dtype=float)
    if t_axis.size == 0:
        return None

    keep = np.ones(v.size, dtype=bool)
    n_pre = int(round(blank_pre_s * fs))
    n_post = int(round(blank_post_s * fs))
    if n_pre or n_post:
        centers = np.round((st - t0) * fs).astype(int)
        for i in centers:                      # spans overlap at high rates
            a, b = max(0, i - n_pre), min(v.size, i + n_post + 1)
            if b > a:
                keep[a:b] = False
    # Report the blanked fraction over the trial's own samples, not the padding.
    frac_blanked = 1.0 - float(keep[own_slice].mean() if own_slice is not None
                               else keep.mean())

    t_v = t0 + np.arange(v.size) / fs
    if keep.any():
        v_clean = np.interp(t_v, t_v[keep], v[keep])
    else:
        v_clean = np.full(v.size, np.nan)       # every sample blanked

    # Average onto the coarse grid, then smooth there.
    edges = np.arange(t_v[0], t_v[-1] + grid_dt, grid_dt)
    if edges.size < 3:
        vm = np.interp(t_axis, t_v, v_clean)
        return t_axis, vm, {'frac_blanked': frac_blanked, 'fs': fs,
                            'blank_pre_s': blank_pre_s,
                            'blank_post_s': blank_post_s,
                            'sigma_s': sigma_s, 'n_spikes': int(st.size)}
    idx = np.clip(np.searchsorted(edges, t_v, side='right') - 1, 0, edges.size - 2)
    nbin = edges.size - 1
    ssum = np.bincount(idx, weights=v_clean, minlength=nbin)
    scnt = np.bincount(idx, minlength=nbin).astype(float)
    centers = 0.5 * (edges[:-1] + edges[1:])
    occupied = scnt > 0
    binned = np.interp(centers, centers[occupied], ssum[occupied] / scnt[occupied])

    if sigma_s and sigma_s > 0:
        kernel = _gaussian_kernel(sigma_s / grid_dt)
        pad = len(kernel) // 2
        padded = np.concatenate([np.full(pad, binned[0]), binned,
                                 np.full(pad, binned[-1])])
        binned = np.convolve(padded, kernel, mode='same')[pad: pad + nbin]

    vm = np.interp(t_axis, centers, binned)
    return t_axis, vm, {'frac_blanked': frac_blanked, 'fs': fs,
                        'blank_pre_s': blank_pre_s, 'blank_post_s': blank_post_s,
                        'sigma_s': sigma_s, 'n_spikes': int(st.size)}
