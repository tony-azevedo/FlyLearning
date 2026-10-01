"""Target-triggered stimulus analysis for the ``LEDFlashTriggerPiezoControl`` protocol.

In this protocol every trial delivers the *same* mechanical stimulus twice:

1. as a **cue**, at a fixed time in the pre-period, before the aversive
   stimulus comes on, and
2. as a **target-triggered stimulus**, delivered by the Arduino at the moment
   the fly reaches the target and switches the aversive stimulus off.

Comparing the neuron's response to the two deliveries of an identical stimulus
is what makes the reflex reversal visible: the cue response is the baseline
reflex, the triggered response is the reflex during an ongoing reach.

The MATLAB original is ``Script_PlotTargetTriggeredStimuli.m`` (called from
``Script_R01figures_211014.m``); this module reproduces its measurements on top
of :class:`mapd.Trial`.
"""

import numpy as np

__all__ = [
    'cue_onset',
    'window_duration',
    'as_off_time',
    'triggered_stim_onset',
    'align_trials',
    'firing_rate',
    'matlab_smooth',
    'DEFAULT_PRE',
]

#: Baseline kept before each stimulus onset, in seconds.  Together with
#: ``cueStimDurInSec + cueDelayDurInSec`` this makes the MATLAB ``cuewin``.
DEFAULT_PRE = 0.2


def cue_onset(trial):
    """Time of the cue displacement, relative to aversive-stimulus onset.

    The piezo trigger fires ``posttriggerdelay`` before the ramp, and the
    stimulus waveform itself starts with a matching pad of zeros, so the
    displacement begins exactly ``-(cueDelay + cueStim)`` seconds before t=0.
    """
    p = trial.params
    return -(p['cueDelayDurInSec'] + p['cueStimDurInSec'])


def window_duration(trial):
    """Length of the analysis window in seconds: 0.2 s pre + cue + delay."""
    p = trial.params
    return DEFAULT_PRE + p['cueStimDurInSec'] + p['cueDelayDurInSec']


def as_off_time(trial):
    """Time at which the aversive stimulus (LED) went off, in seconds.

    ``arduino_output`` carries the LED TTL, so this is its last high sample.
    Returns ``np.nan`` if the LED never came on.
    """
    ard = np.asarray(trial.arduino_output).ravel()
    t = trial.time[:len(ard)]
    high = np.flatnonzero(ard > 0.5)
    if high.size == 0:
        return np.nan
    return float(t[high[-1]])


def triggered_stim_onset(trial, search_from=0.05):
    """Onset of the target-triggered piezo displacement, in seconds.

    Found the way the MATLAB script did it: fit a line through the 10-90%
    portion of the ``sgsmonitor`` ramp and extrapolate back to the
    pre-stimulus baseline.  That beats a threshold crossing because it does
    not depend on where on the ramp the threshold happens to sit.

    Returns ``np.nan`` when no displacement follows the aversive stimulus.
    """
    sgs = np.asarray(trial.sgsmonitor).ravel()
    t = trial.time[:len(sgs)]

    m = t > search_from
    s, tt = sgs[m], t[m]
    if s.size < 2000:
        return np.nan

    base = np.median(s[:2000])
    # The displacement can go either way; work in the direction of the peak.
    sign = 1.0 if (s.max() - base) >= (base - s.min()) else -1.0
    amp = sign * ((s.max() if sign > 0 else s.min()) - base)
    if amp < 0.5:          # no stimulus was delivered
        return np.nan

    above = sign * (s - base)
    i10 = int(np.argmax(above > 0.1 * amp))
    i90 = int(np.argmax(above > 0.9 * amp))
    if i90 <= i10 + 1:
        return np.nan

    slope, intercept = np.polyfit(tt[i10:i90], s[i10:i90], 1)
    if slope == 0:
        return np.nan
    return float((base - intercept) / slope)


def _window_start(trial, onset, pre):
    """First sample index of the window ``[onset - pre, ...)`` in trial time."""
    fs = trial.params['sampratein']
    return int(round((onset - pre + trial.params['preDurInSec']) * fs))


def align_trials(trials, pre=DEFAULT_PRE, dur=None, probe_zeroed=True):
    """Cut cue-aligned and trigger-aligned windows out of a set of trials.

    Parameters
    ----------
    trials : sequence of :class:`mapd.Trial`
    pre : float
        Seconds of baseline kept before each onset.
    dur : float or None
        Total window length.  Defaults to :func:`window_duration` of the first
        trial, which is the window the MATLAB script used.
    probe_zeroed : bool
        Return probe position in the display convention,
        ``-(probe_position - probeZero)``, so that flexion is up.

    Returns
    -------
    dict
        ``t`` is the window time base with 0 at stimulus onset; ``cue`` and
        ``stim`` are dicts of ``(n_trials, n_samples)`` arrays -- ``raster``
        (bool), ``sgs``, ``probe``, ``vm`` -- plus per-trial ``onset`` and
        ``probe_at_onset``.
    """
    trials = list(trials)
    if not trials:
        raise ValueError('no trials given')

    fs = trials[0].params['sampratein']
    if dur is None:
        dur = window_duration(trials[0])
    n = int(round(dur * fs))
    t_win = np.arange(n) / fs - pre

    out = {'t': t_win, 'trials': [int(tr.params['trial']) for tr in trials]}
    for name in ('cue', 'stim'):
        out[name] = {
            'raster': np.zeros((len(trials), n), dtype=bool),
            'sgs': np.full((len(trials), n), np.nan),
            'probe': np.full((len(trials), n), np.nan),
            'vm': np.full((len(trials), n), np.nan),
            'onset': np.full(len(trials), np.nan),
            'probe_at_onset': np.full(len(trials), np.nan),
        }

    for r, tr in enumerate(trials):
        sgs = np.asarray(tr.sgsmonitor).ravel()
        probe = np.asarray(tr.probe_position).ravel()
        vm = np.asarray(tr.voltage_1).ravel()
        if probe_zeroed:
            probe = -(probe - tr.probeZero)
        # ``spikes`` are 1-based sample indices into the full acquisition,
        # which runs past the end of the trial into the intertrial period.
        spk = np.asarray(tr.spikes).ravel().astype(np.int64) - 1

        for name, onset in (('cue', cue_onset(tr)),
                            ('stim', triggered_stim_onset(tr))):
            d = out[name]
            d['onset'][r] = onset
            if not np.isfinite(onset):
                continue
            start = _window_start(tr, onset, pre)
            stop = start + n
            if start < 0 or start >= len(sgs):
                continue
            take = min(stop, len(sgs)) - start
            d['sgs'][r, :take] = sgs[start:start + take]
            d['probe'][r, :take] = probe[start:start + take]
            d['vm'][r, :take] = vm[start:start + take]

            inwin = spk[(spk >= start) & (spk < stop)] - start
            d['raster'][r, inwin] = True

            at = int(round((onset + tr.params['preDurInSec']) * fs))
            if 0 <= at < len(probe):
                d['probe_at_onset'][r] = probe[at]

    return out


def matlab_smooth(y, span):
    """MATLAB's ``smooth(y, span)``: moving average with shrinking endpoints."""
    y = np.asarray(y, dtype=float)
    span = int(span)
    if span % 2 == 0:
        span -= 1
    if span < 3:
        return y.copy()
    half = (span - 1) // 2

    n = len(y)
    c = np.concatenate(([0.0], np.cumsum(y)))
    i = np.arange(n)
    # MATLAB shrinks the window symmetrically at the edges.
    k = np.minimum(np.minimum(i, n - 1 - i), half)
    return (c[i + k + 1] - c[i - k]) / (2 * k + 1)


def firing_rate(t, raster, DT=0.08, edge='normalize'):
    """Trial-averaged firing rate, replicating FlyAnalysis ``firingRate.m``.

    A ``DT``-wide moving sum (7/8 of it causal), divided by ``DT`` and by the
    number of trials, then smoothed with a ``DT/4``-wide moving average.

    Parameters
    ----------
    edge : {'normalize', 'matlab'}
        How to treat the first ``7/8 * DT`` of the window, where the moving
        sum has no history to draw on.  ``'matlab'`` reproduces ``firingRate.m``
        exactly and ramps up from zero -- an artefact, not a silent period.
        The cue window cannot be padded (the trial begins only 0.2 s before the
        cue), so the default instead divides by the width of the window that
        actually exists, which is unbiased and merely noisier at the edge.
    """
    raster = np.atleast_2d(np.asarray(raster, dtype=float))
    n_trials = raster.shape[0]
    counts = np.nansum(raster, axis=0) / max(n_trials, 1)

    wind = int(np.sum(t <= t[0] + DT))
    nb, nf = int(round(wind * 7 / 8)), int(round(wind * 1 / 8))

    n = len(counts)
    c = np.concatenate(([0.0], np.cumsum(counts)))
    i = np.arange(n)
    lo = np.clip(i - nb, 0, n)
    hi = np.clip(i + nf + 1, 0, n)
    if edge == 'matlab':
        width = DT
    elif edge == 'normalize':
        width = (hi - lo) * (t[1] - t[0])
    else:
        raise ValueError("edge must be 'normalize' or 'matlab'")
    return matlab_smooth((c[hi] - c[lo]) / width, round(wind / 4))
