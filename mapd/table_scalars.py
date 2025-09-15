# table comparison functions for flylearning analysis objects
from os import path
import re
from scipy.io import loadmat
import os
import subprocess
import pandas as pd
# import modin.pandas as pd
import swifter
import numpy as np
from datetime import datetime, timedelta
from collections import Counter

from functools import wraps


def compute_duration(self):
    if not 'total_duration' in self.df.columns:
        self.extract_trial_properties(prop_list=['total_duration'])
    return self.df['total_duration'].sum()

def compute_rms_velocity(self):
    self.compute_trial_method('probe_rms_velocity')

    return self.df['probe_rms_velocity'].sum()

def compute_outcome_fractions(self):
    if not 'as_outcome' in self.df.columns:
        self.extract_trial_properties(self,prop_list=['as_outcome'])
    val_cnts = self.df['as_outcome'].value_counts()
    fracs = val_cnts / val_cnts.sum()
    # print(fracs)
    return fracs


def _median_probe_position(T, df_filter=None):
    """
    Compute the median probe position.
    """
    total_counts, total_N, probe_bins = T.probe_position_distribution(binwidth=2,bin_min=-500,bin_max=10,filter=df_filter)
    cum_cnts = np.cumsum(total_counts)
    median_position = probe_bins[np.argmax(cum_cnts >= total_N / 2)] # first occurence
    return median_position


def compute_lo_state_median_position(self):
    """
    Compute the median position of the fly in the lo state.
    """
    df_filter = {'pyasState':'lo'}
    median = _median_probe_position(self,df_filter=df_filter)
    self._lo_state_median_position = median
    return median


def compute_hi_state_median_position(self):
    """
    Compute the median position of the fly in the hi state.
    """
    df_filter = {'pyasState':'hi'}
    median = _median_probe_position(self,df_filter=df_filter)
    self._hi_state_median_position = median
    return median


def compute_hi_lo_shift(self,hi=None, lo=None):
    """
    Compute the shift in median position between hi and lo states.
    """
    if hi is None:
        if hasattr(self, '_hi_state_median_position'):
            hi_median = self._hi_state_median_position
        elif not hasattr(self, '_hi_state_median_position'):
            hi_median = self.hi_state_median_position()
        
    if lo is None:
        if hasattr(self, '_lo_state_median_position'):
            lo_median = self._lo_state_median_position
        elif not hasattr(self, '_lo_state_median_position'):
            lo_median = self.lo_state_median_position()

    shift = hi_median - lo_median
    return shift


def _on_target_fraction_wrapper(T, df_filter=None,index=None,state_switch = 'on'):
    """
    Compute the on-target fraction.
    """
    if df_filter is None:
        raise KeyError('Need a filter')    
    binwidth = 2
    state_filter = df_filter.copy()
    if state_switch == 'off':
        state_filter['pyasState'] = 'hi' if df_filter['pyasState'] == 'lo' else 'lo'
    total_counts, total_N, probe_bins = T.probe_position_distribution(binwidth=binwidth,bin_min=-500,bin_max=10,filter=state_filter, index=index)

    if df_filter['pyasState'] not in T.targets:
        raise ValueError("Filter state doesn't match most common target states.")
    target = T.targets[df_filter['pyasState']]
    return T._on_target_fraction(target,total_counts,probe_bins)


def compute_hi_state_on_target(self):
    """
    Compute the on-target fraction for the hi state.
    """
    df_filter = {'pyasState':'hi'}
    return _on_target_fraction_wrapper(self, df_filter=df_filter)


def compute_hi_target_off_state(self):
    """
    Compute the fraction of time in hi target when not hi state, for comparison.
    """
    df_filter = {'pyasState':'hi'}
    return _on_target_fraction_wrapper(self, df_filter=df_filter,state_switch='off')


def compute_lo_state_on_target(self):
    """
    Compute the on-target fraction for the lo state.
    """
    df_filter = {'pyasState':'lo'}
    return _on_target_fraction_wrapper(self, df_filter=df_filter)


def compute_lo_target_off_state(self):
    """
    Compute the fraction of time in lo target when not lo state, for comparison.
    """
    df_filter = {'pyasState':'lo'}
    return _on_target_fraction_wrapper(self, df_filter=df_filter,state_switch='off')


def compute_num_trials(self):
    return self.df.shape[0]


def compute_blue_fraction(self):
    """What is the cube status of each trial?"""
    if 'filtercube_status' in self.df.columns:
        print('filter cube status present')
    else:
        print('filter cube status not yet assigned')
        trials = self.df['Trial']
        def get_filterstatus_from_meta(trial):
            try:
                fcs = trial.filtercube_status()
            except AttributeError as e:
                fcs = np.nan
            
            return fcs
        self.df['filtercube_status'] = trials.apply(get_filterstatus_from_meta)

    blue_fraction = (self.df['filtercube_status']=='blue').sum()/self.df['filtercube_status'].shape[0]
    return blue_fraction


def compute_blue_toggle_fraction(self):
    """
    Blue_toggle_fraction of <1 should have epi_only as the fiberLED
    """
    if not 'is_rest' in self.df.columns:
        self.extract_trial_properties()
    mask = (self.df['blueToggle'] == 1) & (~self.df['is_rest'])
    non_rest = (~self.df['is_rest']).sum()
    blue_fraction = mask.sum()/non_rest
    return blue_fraction


def compute_most_common_fiberLED(self):
    return self.df['fiberLED'].value_counts().idxmax()