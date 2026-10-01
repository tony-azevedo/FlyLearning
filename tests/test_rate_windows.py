"""Synthetic-data checks for the windowed rate machinery in mapd.

The interpretation of the REST rate-vs-position sweep rests on three claims:
exact spike counts, a Poisson floor that a constant-rate train sits *on*, and a
decomposition that puts genuine slow variability where it belongs. Each is
checked here against a train whose truth is known by construction.

Run with:  conda run -n flop_py312 python tests/test_rate_windows.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mapd import ephys                       # noqa: E402
from mapd import bout_analysis as ba         # noqa: E402
from mapd import kinematics as kin           # noqa: E402


FS_FRAME = 50.0          # camera frame rate used for the synthetic records
DT = 1.0 / FS_FRAME


def _check(name, cond, detail=''):
    print(f'{"PASS" if cond else "FAIL"}  {name}' + (f'  — {detail}' if detail else ''))
    return bool(cond)


# ---------------------------------------------------------------------------
# ephys primitives
# ---------------------------------------------------------------------------

def test_boxcar_rate_is_exact():
    spikes = np.array([0.10, 0.20, 0.30, 0.90])
    t = np.array([0.20, 0.50, 1.00])
    r = ephys.boxcar_rate(spikes, t, window_s=0.20)   # +/- 100 ms
    # 0.20 -> spikes in [0.10, 0.30] = 3 ;  0.50 -> none ;  1.00 -> 1 (0.90)
    expect = np.array([3, 0, 1]) / 0.20
    return _check('boxcar_rate counts exactly', np.allclose(r, expect),
                  f'{r} vs {expect}')


def test_spikes_per_sample_conserves_count():
    rng = np.random.default_rng(0)
    t = np.arange(0, 10, DT)
    spikes = np.sort(rng.uniform(t[0], t[-1], size=500))
    n_sp = ephys.spikes_per_sample(spikes, t)
    inside = ((spikes >= t[0] - DT / 2) & (spikes <= t[-1] + DT / 2)).sum()
    return _check('spikes_per_sample conserves spike count',
                  n_sp.sum() == inside, f'{n_sp.sum()} vs {inside}')


def test_sta_blank_window_recovers_a_synthetic_spike():
    """A boxcar 'spike' 3 ms wide should be measured as ~3 ms of contamination."""
    fs = 10_000.0
    lags = np.arange(-int(0.006 * fs), int(0.020 * fs) + 1) / fs
    mean = np.where((lags >= 0.0) & (lags <= 0.003), -10.0, 0.0)  # 3 ms deflection
    pre, post = ephys.sta_blank_window({'lags_s': lags, 'mean': mean})
    ok = (pre <= 2 / fs) and (0.0028 < post < 0.0032)
    return _check('sta_blank_window recovers a 3 ms deflection', ok,
                  f'pre={pre*1e3:.2f} ms post={post*1e3:.2f} ms')


def test_sta_blank_window_handles_a_post_spike_offset():
    """A real Vm STA settles to a *different* level than it started from (the slow
    depolarization that caused the spike). The window must still bracket only the
    fast waveform: it must not run to the edge of the measured span, and it must
    not report a zero pre-window because the rising phase crossed the tail level.
    """
    fs = 10_000.0
    lags = np.arange(-int(0.006 * fs), int(0.030 * fs) + 1) / fs
    # 2 ms rise to a peak of 10, decay by ~6 ms, settling on a +2 plateau.
    spike = 10.0 * np.exp(-np.maximum(lags, 0) / 0.0015) * (lags >= 0)
    rise = np.clip((lags + 0.001) / 0.001, 0, 1) * (lags < 0)
    plateau = 2.0 / (1.0 + np.exp(-(lags - 0.0) / 0.0005))
    mean = spike + 10.0 * rise * 0 + plateau
    pre, post, info = ephys.sta_blank_window({'lags_s': lags, 'mean': mean},
                                             return_info=True)
    ok = (not info['clipped']) and (0.0 < post < 0.015) and (pre > 0.0)
    return _check('sta_blank_window brackets the spike despite a post-spike offset',
                  ok, f'pre={pre*1e3:.2f} ms post={post*1e3:.2f} ms '
                      f'clipped={info["clipped"]} tail={info["tail_level"]:.2f}')


def test_allan_floor_tracks_a_regular_train_that_poisson_overestimates():
    """A near-regular train has counts far less variable than Poisson. The Allan
    floor must follow the data; the Poisson reference must sit above it.
    """
    rng = np.random.default_rng(11)
    # Regular spiking with small jitter: ISI = 25 ms +/- 2 ms => CV ~ 0.08
    n = 4000
    isi = np.abs(rng.normal(0.025, 0.002, size=n))
    st = np.cumsum(isi)
    t = np.arange(0, st[-1] - 0.1, DT)
    rec = pd.DataFrame({'t': t, 'x': 50.0 + rng.normal(0, 0.3, len(t)),
                        'state': kin.STATE_REST,
                        'n_sp': ephys.spikes_per_sample(st, t)})
    rec['trial'] = 1
    rec['rate'] = rec['n_sp'].rolling(5, center=True, min_periods=1).mean() / DT
    w = ba.window_sweep(rec, windows_s=(0.25, 1.0), state=kin.STATE_REST)
    cs = ba.conditional_spread(w, x_bin_width=1000.0, min_per_bin=5)  # one bin
    print(cs[['window_s', 'rate_mean', 'sd_within_bin', 'sd_allan', 'sd_poisson',
              'fano_emp', 'sd_excess_lo95']].to_string(index=False, float_format='%.3f'))
    ok = bool((cs['sd_poisson'] > cs['sd_within_bin']).all()      # Poisson too high
              and (cs['fano_emp'] < 0.5).all()                    # sub-Poisson
              and (cs['sd_excess_lo95'] == 0).all())              # no slow structure
    return _check('Allan floor fits a regular train where Poisson overestimates',
                  ok, f'fano={np.round(cs["fano_emp"].to_numpy(), 3)} '
                      f'excess_lo95={np.round(cs["sd_excess_lo95"].to_numpy(), 2)}')


# ---------------------------------------------------------------------------
# Synthetic records: Poisson spikes whose rate depends only on position
# ---------------------------------------------------------------------------

def make_records(n_epochs=40, epoch_s=6.0, gap_s=0.5, rate_of_x=None,
                 epoch_rate_sd=0.0, seed=0):
    """REST epochs separated by short MOVE gaps, Poisson spikes throughout.

    ``rate_of_x`` maps position -> true rate (Hz). ``epoch_rate_sd`` adds a
    per-epoch rate offset that position knows nothing about — genuine slow
    variability, the thing ``sd_excess`` is supposed to detect.
    """
    rng = np.random.default_rng(seed)
    if rate_of_x is None:
        rate_of_x = lambda x: 20.0 + 0.4 * x
    frames = []
    t_cursor = 0.0
    for k in range(n_epochs):
        x0 = rng.uniform(0.0, 100.0)
        n = int(round(epoch_s / DT))
        t = t_cursor + np.arange(n) * DT
        x = x0 + rng.normal(0.0, 0.3, size=n)          # nearly still: REST
        true_rate = np.maximum(rate_of_x(x0) + rng.normal(0.0, epoch_rate_sd), 0.1)
        n_sp = rng.poisson(true_rate * DT, size=n).astype(float)
        frames.append(pd.DataFrame({
            't': t, 'x': x, 'state': kin.STATE_REST, 'n_sp': n_sp,
            'true_rate': true_rate,
        }))
        t_cursor = t[-1] + DT
        n_gap = int(round(gap_s / DT))                 # MOVE gap breaks the epoch
        t = t_cursor + np.arange(n_gap) * DT
        frames.append(pd.DataFrame({
            't': t, 'x': x0 + np.linspace(0, 5, n_gap), 'state': kin.STATE_MOVE,
            'n_sp': rng.poisson(20 * DT, size=n_gap).astype(float),
            'true_rate': 20.0,
        }))
        t_cursor = t[-1] + DT
    rec = pd.concat(frames, ignore_index=True)
    rec['trial'] = 1
    # A smoothed-rate column, so rate_col='rate' has something to average.
    rec['rate'] = rec['n_sp'].rolling(5, center=True, min_periods=1).mean() / DT
    return rec


def test_windows_stay_inside_rest_epochs():
    rec = make_records(n_epochs=6, epoch_s=6.0, seed=1)
    w = ba.windowed_state_samples(rec, window_s=1.0, state=kin.STATE_REST)
    # 6 s epochs -> 6 windows each, 6 epochs -> 36; and every window's frames
    # must be REST frames only, which the epoch labelling enforces implicitly.
    ok_n = len(w) == 36
    ok_epochs = sorted(w['epoch'].unique().tolist()) == list(range(6))
    ok_dur = np.allclose(w['duration'], 1.0, atol=DT)
    return _check('windows tile REST epochs without straddling',
                  ok_n and ok_epochs and ok_dur,
                  f'n={len(w)} epochs={w["epoch"].nunique()}')


def test_window_rate_equals_count_over_duration():
    rec = make_records(n_epochs=4, seed=2)
    w = ba.windowed_state_samples(rec, window_s=0.5, state=kin.STATE_REST)
    ok = np.allclose(w['rate'], w['n_spikes'] / w['duration'])
    return _check('window rate == exact count / covered duration', ok)


def test_constant_rate_train_sits_on_the_poisson_floor():
    """Rate a fixed function of position => residual scatter is counting noise
    only, so sd_within should track sd_poisson at every window length."""
    rec = make_records(n_epochs=60, epoch_s=6.0, epoch_rate_sd=0.0, seed=3)
    w = ba.window_sweep(rec, windows_s=(0.25, 0.5, 1.0, 2.0),
                        state=kin.STATE_REST)
    cs = ba.conditional_spread(w, x_bin_width=10.0, min_per_bin=5)
    ratio = cs['sd_within_bin'] / cs['sd_poisson']
    print(cs[['window_s', 'n_windows', 'rate_mean', 'sd_within_bin',
              'sd_poisson', 'sd_excess', 'sd_excess_lo95', 'rho']]
          .to_string(index=False, float_format='%.3f'))
    # The bounded statistic is the one that must not fire: no real excess here.
    ok = (bool(((ratio > 0.75) & (ratio < 1.3)).all())
          and bool((cs['sd_excess_lo95'] == 0).all()))
    return _check('constant-rate train: sd_within ~ sd_poisson, no excess claimed',
                  ok, f'ratios={np.round(ratio.to_numpy(), 2)} '
                      f'excess_lo95={np.round(cs["sd_excess_lo95"].to_numpy(), 2)}')


def test_real_slow_variability_shows_as_excess_and_survives_averaging():
    """A per-epoch rate offset of 6 Hz that position cannot explain should read
    out as sd_excess ~ 6 Hz, roughly flat in window length, and land in the
    between-epoch term of the decomposition."""
    rec = make_records(n_epochs=80, epoch_s=6.0, epoch_rate_sd=6.0, seed=4)
    w = ba.window_sweep(rec, windows_s=(0.25, 0.5, 1.0, 2.0),
                        state=kin.STATE_REST)
    cs = ba.conditional_spread(w, x_bin_width=10.0, min_per_bin=5)
    print(cs[['window_s', 'n_windows', 'rate_mean', 'sd_within_bin',
              'sd_poisson', 'sd_excess', 'sd_excess_lo95', 'rho']]
          .to_string(index=False, float_format='%.3f'))
    ok_excess = bool(((cs['sd_excess'] > 3.5) & (cs['sd_excess'] < 9.0)).all()
                     and (cs['sd_excess_lo95'] > 2.0).all())

    vd = ba.variance_decomposition(w, value_col='rate', levels=('epoch', 'i_win'))
    print(vd.to_string(index=False, float_format='%.3f'))
    # levels=('epoch','i_win') makes 'between_trial' the between-epoch term and
    # 'within_epoch' the counting noise: each window is its own i_win group, so
    # the middle term is 0 by construction. At 2 s windows the epoch offsets
    # should dominate.
    long_w = vd.loc[vd['window_s'] == 2.0].iloc[0]
    ok_split = long_w['frac_between_trial'] > 0.6
    return _check('slow variability -> sd_excess ~ 6 Hz, attributed between epochs',
                  ok_excess and ok_split,
                  f'excess={np.round(cs["sd_excess"].to_numpy(), 1)} '
                  f'between_epoch@2s={long_w["frac_between_trial"]:.2f}')


def test_averaging_helps_when_the_noise_is_fast():
    """The headline claim: with no slow variability, the width of the cloud at a
    given position must fall as 1/sqrt(W)."""
    rec = make_records(n_epochs=80, epoch_s=8.0, epoch_rate_sd=0.0, seed=5)
    w = ba.window_sweep(rec, windows_s=(0.25, 1.0, 4.0), state=kin.STATE_REST)
    cs = ba.conditional_spread(w, x_bin_width=10.0, min_per_bin=5).set_index('window_s')
    obs = cs.loc[0.25, 'sd_within_bin'] / cs.loc[4.0, 'sd_within_bin']
    expect = np.sqrt(4.0 / 0.25)
    ok = 0.7 * expect < obs < 1.4 * expect
    return _check('cloud width falls as 1/sqrt(window) for fast noise', ok,
                  f'observed x{obs:.2f}, sqrt ratio x{expect:.2f}')


def _premove_fixture():
    """REST, then a movement toward target, then REST, then a movement away.

    Built so the two transitions have opposite ``dx_init`` and opposite
    pre-movement rate changes, which is the split the analysis must recover.
    """
    dt = DT
    parts = []

    def add(n, x0, dx_per, state, rate):
        t0 = parts[-1]['t'].iloc[-1] + dt if parts else 0.0
        t = t0 + np.arange(n) * dt
        parts.append(pd.DataFrame({
            't': t, 'x': x0 + dx_per * np.arange(n),
            'state': state, 'rate': float(rate),
            'n_sp': float(rate) * dt}))

    # Frame durations matter here: at DT = 20 ms a "0.2 s lead" is 10 frames, not
    # 40. Make the lead block longer than lead_s and it reaches into the baseline
    # window, which dilutes the very step the test is checking for.
    n_lead = int(round(0.2 / dt))
    add(300, 100.0, 0.0, kin.STATE_REST, 30)          # 6 s rest, baseline
    add(n_lead, 100.0, 0.0, kin.STATE_REST, 45)       # 0.2 s lead: rate UP
    add(60, 100.0, 1.0, kin.STATE_MOVE, 60)           # moves +60 um: toward target
    add(300, 160.0, 0.0, kin.STATE_REST, 30)          # rest again
    add(n_lead, 160.0, 0.0, kin.STATE_REST, 15)       # lead: rate DOWN
    add(60, 160.0, -1.0, kin.STATE_MOVE, 10)          # moves -60 um: away
    add(100, 100.0, 0.0, kin.STATE_REST, 30)
    rec = pd.concat(parts, ignore_index=True)
    rec['trial'] = 1
    return rec


def test_movement_timing_reports_the_upcoming_direction():
    rec = ba.add_movement_timing(_premove_fixture())
    lead = rec[(rec['state'] == kin.STATE_REST) & (rec['t_to_move'] <= 0.2)]
    signs = set(np.sign(lead['next_dx_init'].dropna().unique()).astype(int))
    # both directions must be represented, and never NaN in a lead window
    ok = signs == {1, -1} and lead['next_dx_init'].notna().all()
    return _check('add_movement_timing labels the upcoming direction', ok,
                  f'signs={sorted(signs)}')


def test_premovement_transitions_recovers_both_signs():
    """The headline claim: a rate increase before a force-up movement and a
    decrease before a relaxation, each recovered with the right sign."""
    tr = ba.premovement_transitions(_premove_fixture(), lead_s=0.2,
                                    base_to_s=0.4, base_from_s=1.4,
                                    min_rest_s=1.0, cols=('rate',))
    if len(tr) != 2:
        return _check('premovement_transitions recovers both signs', False,
                      f'expected 2 transitions, got {len(tr)}')
    toward = tr[tr['dx_init'] > 0].iloc[0]
    away = tr[tr['dx_init'] < 0].iloc[0]
    ok = (toward['d_rate'] > 10) and (away['d_rate'] < -10)
    print(f'   toward: dx_init {toward["dx_init"]:+.1f} um, '
          f'd_rate {toward["d_rate"]:+.1f} Hz | '
          f'away: dx_init {away["dx_init"]:+.1f} um, d_rate {away["d_rate"]:+.1f} Hz')
    return _check('premovement_transitions recovers both signs', ok)


def test_premovement_skips_drift_between_rest_and_move():
    """A DRIFT run between the rest and the movement must not hide the
    transition — requiring MOVE to follow REST directly found 3 of 79."""
    rec = _premove_fixture()
    # turn the first 10 frames of each MOVE run into DRIFT
    st = rec['state'].to_numpy().copy()
    for a, b, v in ba._rle_states(st):
        if v == kin.STATE_MOVE:
            st[a:a + 10] = kin.STATE_DRIFT
    rec['state'] = st
    tr = ba.premovement_transitions(rec, lead_s=0.2, base_to_s=0.4,
                                    base_from_s=1.4, min_rest_s=1.0,
                                    cols=('rate',))
    return _check('premovement_transitions sees through an intervening DRIFT',
                  len(tr) == 2, f'{len(tr)} transitions')


if __name__ == '__main__':
    results = [
        test_movement_timing_reports_the_upcoming_direction(),
        test_premovement_transitions_recovers_both_signs(),
        test_premovement_skips_drift_between_rest_and_move(),
        test_boxcar_rate_is_exact(),
        test_spikes_per_sample_conserves_count(),
        test_sta_blank_window_recovers_a_synthetic_spike(),
        test_sta_blank_window_handles_a_post_spike_offset(),
        test_allan_floor_tracks_a_regular_train_that_poisson_overestimates(),
        test_windows_stay_inside_rest_epochs(),
        test_window_rate_equals_count_over_duration(),
        test_constant_rate_train_sits_on_the_poisson_floor(),
        test_real_slow_variability_shows_as_excess_and_survives_averaging(),
        test_averaging_helps_when_the_noise_is_fast(),
    ]
    print(f'\n{sum(results)}/{len(results)} passed')
    sys.exit(0 if all(results) else 1)
