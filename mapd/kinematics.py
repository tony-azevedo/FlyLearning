"""
mapd/kinematics.py
==================
Signal-processing primitives for probe-position traces and movement-bout detection.

Sign convention
---------------
Functions here operate on a position array ``x`` where **positive = toward target**
(i.e. ``x = -(probe_position - probeZero)``).  This matches the convention used
throughout ``Trial`` methods.  The ``_build_trace`` helper returns the same sign.

Bout detection
--------------
Three-state classifier (defaults tuned on 250304_F3_C1):
  REST  (0): |v| < v_rest (80 um/s), sustained for >= Dt (0.2 s)
  DRIFT (1): |v| in [v_rest, v_th), OR any quiet period shorter than Dt
  MOVE  (2): |v| >= v_th (120 um/s), peak-to-peak >= x_excursion_min (8 um);
             short gaps < move_gap (0.3 s) between MOVE runs are filled in

A movement bout is a contiguous span containing at least one MOVE sample,
bounded by REST periods.
"""
from __future__ import annotations

import numpy as np

# ── Physical constant ─────────────────────────────────────────────────────────
k_spring_constant = 0.0829  # uN/um

# ── Default detection thresholds (tuned on 250304_F3_C1) ─────────────────────
V_TH   = 60.0   # um/s — enter MOVE
V_REST =  30.0   # um/s — exit MOVE / enter REST


# ── Signal primitives ─────────────────────────────────────────────────────────

def velocity(t, x):
    """Instantaneous velocity via np.gradient."""
    return np.gradient(x, t)


def acceleration(t, x):
    """Second derivative of position."""
    return np.gradient(velocity(t, x), t)


def mean_velocity(t, x):
    """Mean of absolute velocity."""
    return np.mean(np.abs(velocity(t, x)))


def rms_velocity(t, x):
    """RMS velocity — weights faster movements more."""
    v = velocity(t, x)
    return np.sqrt(np.mean(v ** 2))


def jerk_energy(t, x):
    """Integrated squared jerk."""
    jerk = np.gradient(acceleration(t, x), t)
    return np.sum(jerk ** 2) * np.diff(t[0:2])[0]


def power(t, x):
    """Instantaneous mechanical power against the spring."""
    v = velocity(t, x)
    return -k_spring_constant * x * v


def work(t, x):
    """Work done against the spring (signed)."""
    return np.trapezoid(power(t, x), t)


def holding_cost(t, x):
    """Integrated potential energy (cost of holding position)."""
    U = 0.5 * k_spring_constant * x ** 2
    return np.trapezoid(U, t)


def positive_effort(t, x):
    """One-sided effort: only counts work done moving toward target (x > 0 direction)."""
    v = velocity(t, x)
    pos_vel = np.clip(v, a_min=0, a_max=None)
    pwr = k_spring_constant * x * pos_vel
    return np.trapezoid(np.clip(pwr, a_min=0, a_max=None), t)


def bout_cumulative_metrics(bouts, t, x, trials=None):
    """
    Cumulative v_rms and positive effort from the first bout start to the
    last bout end before the final prolonged rest within the trial-1 window.

    When *trials* is provided the window is capped at the end of trials[0].
    All bouts that start within that window are included, so the metric
    spans REST and DRIFT gaps between bouts — only ending where the last
    bout's trailing REST begins.

    When *trials* is ``None`` only the first bout is used.

    Parameters
    ----------
    bouts  : list of bout dicts (from detect_movement_bouts)
    t, x   : full trace arrays (from detect_movement_bouts)
    trials : list of Trial objects or None

    Returns
    -------
    dict with keys:
        bt          — time array for the window
        cum_v_rms   — cumulative RMS velocity at each sample
        cum_effort  — cumulative positive effort at each sample
        v_rms       — scalar final v_rms
        effort      — scalar final effort
    All arrays are empty (length-0) if no bouts were found.
    """
    t = np.asarray(t)
    x = np.asarray(x)

    if not bouts:
        empty = np.array([])
        return dict(bt=empty, cum_v_rms=empty, cum_effort=empty,
                    v_rms=0.0, effort=0.0)

    si = bouts[0]['start_idx']

    if trials is not None and len(trials) >= 1:
        # Find the last bout that starts within trial 1
        t1_dur = trials[0].total_duration - trials[1].params['preDurInSec']
        last_bout = bouts[0]
        for b in bouts:
            if b['start_time'] < t1_dur:
                last_bout = b
            else:
                break
        # Cap at trial-1 end in sample space
        t1_end_idx = int(np.searchsorted(t, t1_dur))
        ei = min(last_bout['end_idx'], t1_end_idx)
    else:
        ei = bouts[0]['end_idx']

    ei = min(ei, len(t))
    bt = t[si:ei]
    bx = x[si:ei]

    if len(bt) < 2:
        empty = np.array([])
        return dict(bt=empty, cum_v_rms=empty, cum_effort=empty,
                    v_rms=0.0, effort=0.0)

    bv = velocity(bt, bx)
    cum_v_rms = np.sqrt(np.cumsum(bv ** 2) / np.arange(1, len(bv) + 1))

    bp_pwr = np.clip(k_spring_constant * bx * np.clip(bv, 0, None), 0, None)
    bdt = np.diff(bt)
    cum_effort = np.concatenate([[0.0],
                     np.cumsum(0.5 * (bp_pwr[:-1] + bp_pwr[1:]) * bdt)])

    return dict(
        bt=bt,
        cum_v_rms=cum_v_rms,
        cum_effort=cum_effort,
        v_rms=float(cum_v_rms[-1]),
        effort=float(cum_effort[-1]),
    )


def effort(t, x, alpha=1e-6, beta=1e-2):
    """Symmetric effort cost function (not currently in active use)."""
    v = velocity(t, x)
    return np.trapezoid(alpha * v ** 2 + beta * x ** 2, t)


# ── State constants ───────────────────────────────────────────────────────────

STATE_REST  = 0
STATE_DRIFT = 1
STATE_MOVE  = 2
_STATE_LABEL = {STATE_REST: 'rest', STATE_DRIFT: 'drift', STATE_MOVE: 'move'}
_STATE_COLORS = {STATE_REST: 'steelblue', STATE_DRIFT: 'gray', STATE_MOVE: 'tomato'}


# ── Bout detection internals ──────────────────────────────────────────────────

def _zero_short_dt_velocity(v, t, smooth_window, fs):
    """Zero velocity in a window around any sample where dt < 0.8 * median dt."""
    dt_arr = np.diff(t)
    dt_nominal = np.median(dt_arr)
    short_dt = np.where(dt_arr < 0.8 * dt_nominal)[0]
    if len(short_dt):
        n_sm = max(1, int(round(smooth_window * fs)))
        for idx in short_dt:
            lo = max(0, idx - n_sm)
            hi = min(len(v), idx + n_sm + 2)
            v[lo:hi] = 0.0
    return v


def _rolling_rms(v, n):
    """Rolling RMS of v over a window of n samples (centred).

    The first ``n-1`` output samples are zeroed: the window is not full there,
    so the estimate is unreliable and the fly should be treated as at rest.
    """
    v2 = v ** 2
    kernel = np.ones(n) / n
    out = np.sqrt(np.convolve(v2, kernel, mode='same'))
    out[:n - 1] = 0.0
    return out


def _classify_states_schmitt(v, x=None, v_th=V_TH, v_rest=V_REST, Dt=0.2, fs=None,
                             smooth_window=0.05, x_excursion_min=8.0):
    """
    Schmitt-trigger (hysteretic) state classifier.

    Enter MOVE when smoothed speed >= v_th.
    Exit MOVE only when speed < v_rest sustained for >= Dt seconds.
    Short dips below v_th (but above v_rest) do not exit MOVE — hysteresis
    makes move_gap filling unnecessary.

    Non-MOVE regions are then resolved to REST (speed < v_rest for >= Dt)
    or DRIFT (everything else).

    Compared to _classify_states, bout boundaries are cleaner: the exit
    threshold is v_rest (not v_th), so there is no spurious splitting on
    brief velocity dips within a continuous movement.
    """
    if fs is None:
        raise ValueError("fs must be provided (sampling rate in Hz)")
    n_smooth = max(1, int(round(smooth_window * fs)))
    speed    = _rolling_rms(v, n_smooth)
    n_rest   = max(1, int(round(Dt * fs)))
    n        = len(speed)

    state   = np.full(n, STATE_DRIFT, dtype=np.int8)
    in_move = False
    i       = 0

    # ── First pass: find MOVE regions via Schmitt trigger ────────────────────
    while i < n:
        if not in_move:
            if speed[i] >= v_th:
                in_move    = True
                move_start = i
                state[i]   = STATE_MOVE
            i += 1
        else:
            state[i] = STATE_MOVE
            if speed[i] < v_rest:
                quiet_start = i
                j = i + 1
                while j < n and speed[j] < v_rest:
                    j += 1
                quiet_len = j - quiet_start
                if quiet_len >= n_rest:
                    # Genuine rest — exit MOVE; unmark the quiet run
                    state[quiet_start:j] = STATE_DRIFT  # resolved below
                    if x_excursion_min is not None and x is not None:
                        if np.ptp(x[move_start:quiet_start]) < x_excursion_min:
                            state[move_start:quiet_start] = STATE_DRIFT
                    in_move = False
                    i = j
                else:
                    # Short dip — hysteresis: keep as MOVE
                    state[quiet_start:j] = STATE_MOVE
                    i = j
            else:
                i += 1

    if in_move:
        if x_excursion_min is not None and x is not None:
            if np.ptp(x[move_start:]) < x_excursion_min:
                state[move_start:] = STATE_DRIFT

    # ── Second pass: resolve non-MOVE runs to REST or DRIFT ──────────────────
    i = 0
    while i < n:
        if state[i] != STATE_MOVE:
            j = i + 1
            while j < n and state[j] != STATE_MOVE:
                j += 1
            # scan non-MOVE region [i:j] for sustained quiet runs
            k = i
            while k < j:
                if speed[k] < v_rest:
                    rest_start = k
                    m = k + 1
                    while m < j and speed[m] < v_rest:
                        m += 1
                    if m - rest_start >= n_rest:
                        state[rest_start:m] = STATE_REST
                    k = m
                else:
                    k += 1
            i = j
        else:
            i += 1

    return state


def _classify_states(v, x=None, v_th=V_TH, v_rest=V_REST, Dt=0.2, fs=None,
                     smooth_window=0.05, x_excursion_min=8.0, move_gap=0.3):
    """
    Classify a velocity array into REST / DRIFT / MOVE per sample.

    Speed is computed as rolling RMS over ``smooth_window`` seconds before
    thresholding, so brief zero-crossings at movement reversals do not
    spuriously break a bout into separate pieces.

    Short REST runs (< Dt seconds) are then promoted to DRIFT.

    If ``x_excursion_min`` is set, any MOVE run whose peak-to-peak position
    range is below that threshold is demoted to DRIFT.  This rejects
    high-frequency, low-amplitude jitter (e.g. 250304_F3_C1 trial 6) that has high velocity
    but negligible displacement.

    If ``move_gap`` is set, non-MOVE gaps between two MOVE segments shorter
    than this duration (s) are promoted back to MOVE.  This prevents brief
    dips below v_th from splitting a single bout.

    # Alternative: Schmitt trigger (hysteresis)
    # Enter MOVE when speed >= v_th; exit only when speed < v_rest sustained
    # for Dt.  Cleaner bout boundaries, but requires a stateful loop and an
    # explicit exit threshold (v_rest makes a natural choice).  Consider if
    # rolling-RMS alone leaves too many spurious REST patches inside bouts.
    """
    if fs is None:
        raise ValueError("fs must be provided (sampling rate in Hz)")
    n_smooth = max(1, int(round(smooth_window * fs)))
    speed  = _rolling_rms(v, n_smooth)
    n_rest = max(1, int(round(Dt * fs)))

    raw = np.where(speed >= v_th,  STATE_MOVE,
          np.where(speed <  v_rest, STATE_REST, STATE_DRIFT)).astype(np.int8)

    state = raw.copy()
    i, n  = 0, len(raw)
    while i < n:
        if raw[i] == STATE_REST:
            j = i + 1
            while j < n and raw[j] == STATE_REST:
                j += 1
            if j - i < n_rest:
                state[i:j] = STATE_DRIFT
            i = j
        else:
            i += 1

    # Excursion guard: demote low-displacement MOVE runs to DRIFT
    if x_excursion_min is not None and x is not None:
        i = 0
        while i < n:
            if state[i] == STATE_MOVE:
                j = i + 1
                while j < n and state[j] == STATE_MOVE:
                    j += 1
                if np.ptp(x[i:j]) < x_excursion_min:
                    state[i:j] = STATE_DRIFT
                i = j
            else:
                i += 1

    # Move-gap fill: merge short non-MOVE gaps between MOVE segments into MOVE
    if move_gap is not None:
        n_gap = max(1, int(round(move_gap * fs)))
        i = 0
        while i < n:
            if state[i] == STATE_MOVE:
                j = i + 1
                while j < n and state[j] == STATE_MOVE:
                    j += 1
                # j is the first non-MOVE after this run; find start of next MOVE
                k = j
                while k < n and state[k] != STATE_MOVE:
                    k += 1
                if k < n and (k - j) < n_gap:
                    state[j:k] = STATE_MOVE
                i = k
            else:
                i += 1

    return state


def _build_trace(trial1, trial2=None):
    """
    Concatenate downsampled probe traces from one or two Trial objects.

    Returns (t, x) where x = -(probe_position - probeZero), i.e. positive
    toward target, matching the convention used in Trial methods.

    Trial 2's time is offset by trial1.total_duration.  Any samples in the
    concatenated trace closer together than the nominal downsampled step
    (trial1.frame_length_mode / sampratein) are dropped — this prevents
    spurious velocity spikes at the trial junction.
    """
    xs, ts = [], []
    for tr in ([trial1] if trial2 is None else [trial1, trial2]):
        x_raw = -(tr.probe_position - tr.probeZero).squeeze()
        xs.append(x_raw[tr.downsample_probe])
        ts.append(tr.time[tr.downsample_probe].squeeze())
    if trial2 is not None:
        ts[1] = ts[1] + trial1.total_duration
    t = np.concatenate(ts)
    x = np.concatenate(xs)
    if trial2 is not None:
        # Drop samples that are too close together at the junction
        nominal_dt = trial1.frame_length_mode / trial1.params['sampratein']
        dt_arr = np.diff(t)
        keep = np.ones(len(t), dtype=bool)
        keep[1:] = dt_arr >= 0.5 * nominal_dt
        t = t[keep]
        x = x[keep]
    return t, x


def _rle(states, offset=0):
    """Run-length encode a state array → list of {state, start, end} dicts."""
    if len(states) == 0:
        return []
    segs, cur, start = [], states[0], 0
    for i in range(1, len(states)):
        if states[i] != cur:
            segs.append({'state': int(cur), 'start': start + offset, 'end': i + offset})
            cur, start = states[i], i
    segs.append({'state': int(cur), 'start': start + offset, 'end': len(states) + offset})
    return segs


def _resolve_bout_end(segs, t, fallback):
    """
    Find the sample index where a bout ends (first REST segment).

    fallback : 'none'  → (None, False) when no REST found
               'drift' → start of first DRIFT, or (None, False)
               'end'   → last sample of the last segment
    """
    for seg in segs:
        if seg['state'] == STATE_REST:
            return seg['start'], True
    if fallback == 'none':
        return None, False
    elif fallback == 'drift':
        for seg in segs:
            if seg['state'] == STATE_DRIFT:
                return seg['start'], False
        return None, False
    else:  # 'end'
        return segs[-1]['end'], False


# ── Public API ────────────────────────────────────────────────────────────────

def detect_movement_bouts(
    t, x,
    start_time=0.0,
    v_th=V_TH,
    v_rest=V_REST,
    Dt=0.2,
    fallback='end',
    smooth_window=0.05,
    x_excursion_min=8.0,
    move_gap=0.3,
    classifier=_classify_states_schmitt,
    junction_indices=None,
):
    """
    Detect movement bouts in a probe position trace.

    Parameters
    ----------
    t, x             : 1-D arrays — time (s) and position (um, positive toward target)
    start_time       : float or None — search only from this time onward; default 0.0
                       so that bouts before stimulus onset (t=0) are never returned.
                       Pass None to search the entire trace.
    v_th             : MOVE velocity threshold (um/s), default 120
    v_rest           : REST velocity threshold (um/s), default 80
    Dt               : minimum REST duration (s) for a genuine rest, default 0.2
    fallback         : 'none' | 'drift' | 'end' — behaviour when no trailing REST found
    smooth_window    : rolling-RMS window (s) applied to speed before thresholding,
                       default 0.05 — handles brief zero-crossings at reversals
    x_excursion_min  : float or None — MOVE runs with peak-to-peak position range
                       below this (um) are demoted to DRIFT; default 8.0
    move_gap         : float or None — only used by _classify_states;
                       non-MOVE gaps shorter than this (s) are filled in as MOVE;
                       default 0.3
    classifier       : callable or None — state classifier function with signature
                           f(v, x, v_th, v_rest, Dt, fs, smooth_window, x_excursion_min)
                       Defaults to _classify_states_schmitt (hysteretic classification).
                       Pass _classify_states for threshold-based with move_gap.
    junction_indices : list of int or None — sample indices where trial boundaries
                       fall in the concatenated trace; velocity is zeroed there to
                       prevent spurious spikes from position discontinuities.

    Returns
    -------
    bouts  : list of dicts with keys:
               start_idx, end_idx, start_time, end_time,
               duration, started_from_rest, has_trailing_rest
    states : int8 array, length == len(t) — per-sample state
    t      : echo of input t (squeezed)
    x      : echo of input x (squeezed)
    """
    t = np.asarray(t).squeeze()
    x = np.asarray(x).squeeze()

    fs = 1.0 / np.median(np.diff(t))
    v  = velocity(t, x)

    v = _zero_short_dt_velocity(v, t, smooth_window, fs)

    if junction_indices:
        n_sm = max(1, int(round(smooth_window * fs)))
        for j in junction_indices:
            lo = max(0, j - n_sm)
            hi = min(len(v), j + n_sm + 1)
            v[lo:hi] = 0.0

    shared_kwargs = dict(v_th=v_th, v_rest=v_rest, Dt=Dt, fs=fs,
                         smooth_window=smooth_window, x_excursion_min=x_excursion_min)

    if classifier is _classify_states:
        states = classifier(v, x, move_gap=move_gap, **shared_kwargs)
    else:
        states = classifier(v, x, **shared_kwargs)

    search_start = 0
    if start_time is not None:
        search_start = int(np.searchsorted(t, start_time))
    segs = _rle(states[search_start:], offset=search_start)

    bouts   = []
    seg_idx = 0
    while seg_idx < len(segs):
        seg = segs[seg_idx]
        if seg['state'] == STATE_MOVE:
            bout_start        = seg['start']
            started_from_rest = (seg_idx == 0 or segs[seg_idx - 1]['state'] == STATE_REST)

            end_idx, has_rest = _resolve_bout_end(segs[seg_idx:], t, fallback)
            if end_idx is None:
                seg_idx += 1
                continue

            ei = end_idx - 1 if end_idx > bout_start else bout_start
            bouts.append({
                'start_idx':         bout_start,
                'end_idx':           end_idx,
                'start_time':        float(t[bout_start]),
                'end_time':          float(t[ei]),
                'duration':          float(t[ei] - t[bout_start]),
                'started_from_rest': started_from_rest,
                'has_trailing_rest': has_rest,
            })
            seg_idx = next(
                (k for k, s in enumerate(segs) if s['start'] >= end_idx),
                len(segs),
            )
        else:
            seg_idx += 1

    return bouts, states, t, x


def detect_bouts_across_trials(trials, **kwargs):
    """
    Detect movement bouts across a list of contiguous Trial objects.

    Concatenates their downsampled traces (via ``_build_trace``) and calls
    ``detect_movement_bouts``.  All ``**kwargs`` are forwarded to
    ``detect_movement_bouts`` (v_th, v_rest, Dt, fallback, start_time).

    Parameters
    ----------
    trials : list of Trial objects in chronological order

    Returns
    -------
    Same as detect_movement_bouts: (bouts, states, t, x)
    """
    if not trials:
        raise ValueError("trials list is empty")
    t, x = _build_trace(trials[0], trials[1] if len(trials) > 1 else None)
    junctions = []
    if len(trials) > 1:
        junctions.append(int(np.searchsorted(t, trials[0].total_duration)))
    if len(trials) > 2:
        for tr in trials[2:]:
            junctions.append(len(t))
            dt_step = t[-1] - t[-2]
            t_extra = tr.time[tr.downsample_probe].squeeze()
            t_extra = t_extra + (t[-1] + dt_step - t_extra[0])
            x_extra = -(tr.probe_position - tr.probeZero).squeeze()[tr.downsample_probe]
            t = np.concatenate([t, t_extra])
            x = np.concatenate([x, x_extra])
    return detect_movement_bouts(t, x, junction_indices=None, **kwargs)


def plot_bouts(
    bouts,
    states,
    t,
    x,
    trials=None,
    v_th=V_TH,
    v_rest=V_REST,
    smooth_window=0.05,
    start_time=None,
    title=None,
    show_target=True,
    show_grid=False,
    show_drift_patches=True,
    cum_vrms_all_bouts=False,
    context_trial=None,
):
    """
    Interactive Plotly figure for inspecting bout detection results.

    Accepts the output of detect_movement_bouts (or detect_bouts_across_trials)
    directly, so detection and plotting are decoupled.

    Parameters
    ----------
    bouts, states, t, x : output tuple from detect_movement_bouts
    trials      : Trial object OR list of Trial objects — optional, used only
                  for the target-zone band (show_target) and auto-title
    v_th, v_rest        : thresholds drawn as horizontal lines on the speed panel
    smooth_window       : rolling-RMS window (s) used to compute the speed trace
    start_time  : float or None — vertical dotted line marking the search start
    title       : str or None — figure title
    show_target : bool — draw the pyasState target band from trials[0]
    show_grid   : bool — show plotly background grid lines (default True)
    show_drift_patches : bool — draw light gray patches for DRIFT periods (row 1)
    cum_vrms_all_bouts : bool — if True, row 3 shows cumulative v_rms across all
                         MOVE bouts within trial 1, resetting on REST; if False
                         (default), shows cumulative v_rms over the first bout only
    context_trial : Trial or None — preceding trial shown on rows 1 & 2 with
                    negative time (shifted so it ends at t[0]).  State-colored
                    on row 1, speed shown on row 2.  Does not affect bout
                    detection or cumulative metric panels (rows 3 & 4).

    Returns
    -------
    fig : plotly.graph_objects.Figure  — call fig.show() or fig.write_html(...)

    Row 1 — probe position coloured by REST/DRIFT/MOVE; bouts highlighted gold
    Row 2 — rolling-RMS speed with v_th / v_rest lines
    Row 3 — cumulative v_rms over the first bout (or all bouts in trial 1)
    Row 4 — cumulative positive effort over the first bout (or all bouts in trial 1)

    Typical usage
    -------------
    bouts, states, t, x = detect_bouts_across_trials(
        [trial1, trial2], start_time=trial1.as_duration, v_th=60, v_rest=40)
    fig = plot_bouts(bouts, states, t, x,
                     trials=[trial1, trial2], v_th=60, v_rest=40,
                     start_time=trial1.as_duration)
    fig.show()
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    t = np.asarray(t)
    x = np.asarray(x)

    # ── Normalise trials for target / title ──────────────────────────────────
    if trials is not None and not isinstance(trials, list):
        trials = [trials]

    # ── Speed trace ───────────────────────────────────────────────────────────
    fs    = 1.0 / np.median(np.diff(t))
    v     = velocity(t, x)
    v     = _zero_short_dt_velocity(v, t, smooth_window, fs)
    n_sm  = max(1, int(round(smooth_window * fs)))
    speed = _rolling_rms(v, n_sm)

    # ── Plotly colours (match _STATE_COLORS) ─────────────────────────────────
    _plotly_colors = {
        STATE_REST:  'steelblue',
        STATE_DRIFT: 'gray',
        STATE_MOVE:  'tomato',
    }

    # ── Figure ────────────────────────────────────────────────────────────────
    if title is None and trials is not None:
        nums  = ', '.join(str(tr.params.get('trial', '?')) for tr in trials)
        title = f'Trial {nums}  |  v_th={v_th}  v_rest={v_rest}'
    elif title is None:
        title = f'v_th={v_th}  v_rest={v_rest}'

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        row_heights=[0.46, 0.18, 0.18, 0.18],
        vertical_spacing=0.04,
        subplot_titles=[
            'Probe position (um, +toward target)',
            'Rolling-RMS speed (um/s)',
            'Cum v_rms — all bouts to trial-1 end (um/s)' if cum_vrms_all_bouts
                else 'Cumulative v_rms from t=0 (um/s)',
            'Cum effort — all bouts to trial-1 end (uN·um)' if cum_vrms_all_bouts
                else 'Cumulative positive effort from t=0 (uN·um)',
        ],
    )

    # Row 1: position, coloured by state
    for state_val, color in _plotly_colors.items():
        mask = states == state_val
        if not mask.any():
            continue
        fig.add_trace(
            go.Scattergl(
                x=t[mask], y=x[mask],
                mode='markers',
                marker=dict(color=color, size=3, opacity=0.8),
                name=_STATE_LABEL[state_val],
                legendgroup=_STATE_LABEL[state_val],
            ),
            row=1, col=1,
        )

    # Context trial — full state-colored trace on rows 1 & 2 with negative time
    if context_trial is not None:
        ctx = context_trial
        ctx_idx = ctx.downsample_probe
        ctx_t = ctx.time[ctx_idx].squeeze()
        ctx_x = -(ctx.probe_position[ctx_idx].squeeze() - ctx.probeZero)
        # Shift so context ends at t[0]
        ctx_t = ctx_t - ctx_t[-1] + t[0]

        # State-classify the context trace
        ctx_fs = 1.0 / np.median(np.diff(ctx_t))
        ctx_v = velocity(ctx_t, ctx_x)
        ctx_v = _zero_short_dt_velocity(ctx_v, ctx_t, smooth_window, ctx_fs)
        ctx_n_sm = max(1, int(round(smooth_window * ctx_fs)))
        ctx_speed = _rolling_rms(ctx_v, ctx_n_sm)
        ctx_states = _classify_states_schmitt(
            ctx_v, x=ctx_x, v_th=v_th, v_rest=v_rest, Dt=0.2,
            fs=ctx_fs, smooth_window=smooth_window, x_excursion_min=8.0)

        # Row 1: position colored by state (faded)
        for state_val, color in _plotly_colors.items():
            mask = ctx_states == state_val
            if not mask.any():
                continue
            fig.add_trace(
                go.Scattergl(
                    x=ctx_t[mask], y=ctx_x[mask],
                    mode='markers',
                    marker=dict(color=color, size=2, opacity=0.3),
                    name=_STATE_LABEL[state_val],
                    legendgroup=_STATE_LABEL[state_val],
                    showlegend=False,
                ),
                row=1, col=1,
            )

        # Row 2: speed trace (faded)
        fig.add_trace(
            go.Scattergl(
                x=ctx_t, y=ctx_speed,
                mode='lines',
                line=dict(color='gray', width=1),
                opacity=0.3,
                name='speed (context)',
                showlegend=False,
            ),
            row=2, col=1,
        )

    # Bout highlight bands (row 1)
    for b in bouts:
        si = b['start_idx']
        ei = min(b['end_idx'], len(t) - 1)
        fig.add_vrect(
            x0=float(t[si]), x1=float(t[ei]),
            fillcolor='gold', opacity=0.2, line_width=0,
            row=1, col=1,
        )

    # Drift patches (row 1)
    if show_drift_patches:
        drift_mask = states == STATE_DRIFT
        in_drift = False
        drift_start = 0
        for i in range(len(drift_mask)):
            if drift_mask[i] and not in_drift:
                drift_start = i
                in_drift = True
            elif not drift_mask[i] and in_drift:
                fig.add_vrect(
                    x0=float(t[drift_start]), x1=float(t[i - 1]),
                    fillcolor='lightgray', opacity=0.3, line_width=0,
                    row=1, col=1,
                )
                in_drift = False
        if in_drift:
            fig.add_vrect(
                x0=float(t[drift_start]), x1=float(t[-1]),
                fillcolor='lightgray', opacity=0.3, line_width=0,
                row=1, col=1,
            )

    # Row 2: rolling-RMS speed
    fig.add_trace(
        go.Scattergl(
            x=t, y=speed,
            mode='lines',
            line=dict(color='gray', width=1),
            name='speed (RMS)',
            showlegend=True,
        ),
        row=2, col=1,
    )
    fig.add_hline(y=v_th,   line=dict(color='tomato',    width=1.5),
                  annotation_text=f'v_th={v_th}',
                  annotation_position='top right', row=2, col=1)
    fig.add_hline(y=v_rest, line=dict(color='steelblue', width=1.5, dash='dash'),
                  annotation_text=f'v_rest={v_rest}',
                  annotation_position='bottom right', row=2, col=1)

    # Target zone band (row 1)
    if show_target and trials is not None:
        tr0 = trials[0]
        try:
            tgt_hi   = -(float(tr0.pyasXPosition) - float(tr0.probeZero))
            x_width  = float(tr0.pyasWidth)
            tgt_lo   = tgt_hi - x_width
            state_str = str(getattr(tr0, 'pyasState', ''))
            tgt_color = 'rgba(214,39,40,0.15)' if state_str == 'hi' else 'rgba(31,119,180,0.15)'
            fig.add_hrect(
                y0=tgt_lo, y1=tgt_hi,
                fillcolor=tgt_color, line_width=0,
                annotation_text=f'target ({state_str})',
                annotation_position='top right',
                row=1, col=1,
            )
        except Exception:
            pass   # silently skip if trial lacks target metadata

    # Rows 3 & 4: cumulative v_rms and effort
    metrics = bout_cumulative_metrics(
        bouts, t, x,
        trials=trials if cum_vrms_all_bouts else None,
    )
    bt         = metrics['bt']
    cum_v_rms  = metrics['cum_v_rms']
    cum_effort = metrics['cum_effort']

    # Row 3: cumulative v_rms over bout
    fig.add_trace(
        go.Scattergl(
            x=bt, y=cum_v_rms,
            mode='lines',
            line=dict(color='mediumpurple', width=1),
            name='cum v_rms',
            showlegend=True,
        ),
        row=3, col=1,
    )

    # Row 4: cumulative positive effort over bout
    fig.add_trace(
        go.Scattergl(
            x=bt, y=cum_effort,
            mode='lines',
            line=dict(color='darkorange', width=1),
            name='cum effort',
            showlegend=True,
        ),
        row=4, col=1,
    )

    # Start-time marker on all rows
    if start_time is not None:
        for row in (1, 2, 3, 4):
            fig.add_vline(
                x=start_time,
                line=dict(color='black', width=1, dash='dot'),
                annotation_text='AS off' if row == 1 else '',
                annotation_position='top left',
                row=row, col=1,
            )

    fig.update_layout(
        title=title,
        height=900,
        legend=dict(orientation='v', x=1.02, y=1),
        hovermode='x unified',
        plot_bgcolor='white' if not show_grid else None,
    )
    t_min = float(ctx_t[0]) if context_trial is not None else float(t[0])
    fig.update_xaxes(range=[t_min, float(t[-1])], title_text='time (s)', row=4, col=1)
    fig.update_yaxes(title_text='position (um)',    range=[-10, 500], row=1, col=1)
    fig.update_yaxes(title_text='speed (um/s)',     range=[0, 200], row=2, col=1)
    fig.update_yaxes(title_text='cum v_rms (um/s)', row=3, col=1)
    fig.update_yaxes(title_text='cum effort (uN·um)', row=4, col=1)

    if not show_grid:
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)

    return fig
