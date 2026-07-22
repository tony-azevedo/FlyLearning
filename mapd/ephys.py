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
