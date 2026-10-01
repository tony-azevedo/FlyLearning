"""Terminal harness for debugging the plotting code in scratch.py.

Loads the cached parquets instead of re-collecting from the Sinq, so it starts
in about a second. Run it from the repo root:

    conda activate flop_py312

    python debug_plot.py                 # just run it
    python -i debug_plot.py              # run, then land in a REPL with the frames
    python -m pdb debug_plot.py          # step from the very first line
    python -m pdb -c continue debug_plot.py   # run; on a crash, post-mortem prompt

Inside pdb the useful ones are:

    b scratch.py:250   set a breakpoint (tab-completion off; use the line number)
    c                  continue to the next breakpoint / to the end
    n / s              next line / step into the call
    l / ll             list source around here / the whole function
    p x / pp x         print / pretty-print an expression
    w                  where am I -- the call stack
    u / d              move up / down a stack frame
    interact           full Python REPL with this frame's locals (Ctrl-D to exit)
    q                  quit

You can also drop `breakpoint()` on any line of scratch.py and just run
`python debug_plot.py` -- execution stops there with no pdb flags needed.
"""
import os

import matplotlib
matplotlib.use('Agg')      # no GUI window; figures go to files. Comment out and
                           # use plt.show() if you want an interactive window --
                           # but note show() blocks the pdb prompt.
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, 'Figure5_mutant_plots')
OUT = os.path.join(HERE, 'debug_out')
os.makedirs(OUT, exist_ok=True)


def load(tag):
    runs = pd.read_parquet(f'{FIG}/in_target_moves_{tag}_runs.parquet')
    trials = pd.read_parquet(f'{FIG}/in_target_moves_{tag}_trials.parquet')
    print(f'{tag:8s} runs {str(runs.shape):14s} trials {str(trials.shape):14s} '
          f'state_sets {sorted(runs["state_set"].unique())}')
    return runs, trials


iav_runs, iav_trials = load('iav')
op_cond_runs, op_cond_trials = load('op_cond')

# scratch.py's figure calls are under `if __name__ == '__main__'`, so importing
# it here gives the functions without running them.
import scratch


def durations(tag='iav', **kw):
    """plot_in_target_durations on one group. kw overrides the defaults."""
    runs, trials = (iav_runs, iav_trials) if tag == 'iav' else (op_cond_runs, op_cond_trials)
    opts = dict(group_name=tag, state_set='drift+move', compare_state_set='move',
                savepath=f'{OUT}/durations_{tag}.png')
    opts.update(kw)
    return scratch.plot_in_target_durations(runs, trials, **opts)


def ecdf(**kw):
    """plot_time_weighted_ecdf on both groups."""
    opts = dict(state_set='drift+move', weight='duration', per_fly=True,
                savepath=f'{OUT}/time_ecdf.png')
    opts.update(kw)
    return scratch.plot_time_weighted_ecdf(
        {'+;iav': iav_runs, 'op_cond': op_cond_runs}, **opts)


def budget(tag='iav', **kw):
    trials = iav_trials if tag == 'iav' else op_cond_trials
    return scratch.state_time_budget(trials, **kw)


if __name__ == '__main__':
    fig, stats = durations('iav')
    print(stats.to_string())

    fig, summ = ecdf()
    print(summ.round(3).to_string(index=False))

    b = budget('iav', by=None)
    print(b.round(3).to_string())

    print(f'\nfigures in {OUT}')
    print('try:  python -i debug_plot.py   then  durations("op_cond"), '
          'ecdf(weight="move_time"), budget("op_cond")')
