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

from functools import wraps


def compute_duration(self):
    if not 'total_duration' in self.df.columns:
        self.add_trial_properties(prop_list=['total_duration'])
    return self.df['total_duration'].sum()

def compute_rms_velocity(self):
    self.compute_trial_method('probe_rms_velocity')

    return self.df['probe_rms_velocity'].sum()

def compute_outcome_fractions(self):
    val_cnts = self.df['as_outcome'].value_counts()
    fracs = val_cnts / val_cnts.sum()
    print(fracs)
    return fracs


def compute_lo_state_median_position(self):
    """
    Compute the median position of the fly in the lo state.
    """
    df_filter = {'pyasState':'lo'}
    median = self.median_probe_position(df_filter=df_filter)
    return median


def compute_hi_state_median_position(self):
    """
    Compute the median position of the fly in the hi state.
    """
    df_filter = {'pyasState':'hi'}
    median = self.median_probe_position(df_filter=df_filter)
    return median


def median_probe_position(self, df_filter=None):
    """
    Compute the median probe position.
    """
    total_counts, total_N, probe_bins = self.probe_position_distribution(binwidth=2,bin_min=-500,bin_max=10,filter=df_filter)
    cum_cnts = np.cumsum(total_counts)
    median_position = probe_bins[np.argmax(cum_cnts >= total_N / 2)] # first occurence
    return median_position


def compute_hi_lo_shift(self,hi=None, lo=None):
    """
    Compute the shift in median position between hi and lo states.
    """
    if hi is None:
        hi_median = self.compute_hi_state_median_position()
    if lo is None:
        lo_median = self.compute_lo_state_median_position()
    shift = hi_median - lo_median
    return shift


