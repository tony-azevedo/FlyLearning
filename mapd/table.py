from .helpers import get_day_fly_cell, get_file, default_data_directory
# from .table_plotters import plot_some_trials, plot_outcomes, plot_probe_distribution, probe_position_heatmap
import types
from . import table_plotters
from . import table_movie_maker
from . import table_export_methods
from . import table_scalars
from mapd.trial import Trial
from mapd.trial import TRIAL_METADATA_GROUP

import importlib
importlib.reload(table_plotters)
importlib.reload(table_movie_maker)
importlib.reload(table_export_methods)

import os
import warnings
import subprocess
import pandas as pd
pd.options.mode.chained_assignment = 'raise'
# import modin.pandas as pd
import swifter
import numpy as np
from datetime import datetime, timedelta
from functools import cached_property
# from matplotlib.figure import Figure
# from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from collections import Counter
from scipy.io import loadmat

# from matplotlib import pyplot as plt
# import matplotlib.patches as patches
# import seaborn as sns

import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)  # reset to defaults
mpl.rcParams['pdf.fonttype'] = 42         # embed fonts as text, not paths
mpl.rcParams['svg.fonttype'] = 'none'     # keep text editable in SVG
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.size'] = 11


_outcomes_dict = {
    'no_as_no_mv': 'no aversive stimulus, no movement',
    'no_as_mv': 'no aversive stimulus, probe moves',
    'as_off': 'fly turns off aversive stimulus during trial',
    'as_off_late': 'fly turns off aversive stimulus in intertrial period',
    'timeout_fail': 'aversive stimulus never turned off and the probe was more flexed than the target',
    'timeout': 'aversive stimulus never turned off, no movement',
    'probe': 'probe_trial',
    'rest': 'rest trial',
    'info': 'info',
}

_as_outcomes_cat =             pd.api.types.CategoricalDtype(categories=_outcomes_dict.keys(), ordered=True)
_pyasState_cat =                     pd.api.types.CategoricalDtype(categories=['no_state','lo','hi'], ordered=True)
_vnc_status_cat =           pd.api.types.CategoricalDtype(categories=['intact','cut'], ordered=True)
_filtercube_status_cat =    pd.api.types.CategoricalDtype(categories=['green','blue'], ordered=False)
_fiberLED_cat =    pd.api.types.CategoricalDtype(categories=['epi_only','740_ir','625_red','590_orange','470_blue','405_uv'], ordered=True)

_category_dict = {
    'as_outcome': _as_outcomes_cat,
    'pyasState': _pyasState_cat,
    'vnc_status': _vnc_status_cat,
    'filtercube_status': _filtercube_status_cat,
    'fiberLED': _fiberLED_cat,
}


def pyas_state(df):
    states = df.pyasState.unique()
    if len(states) > 1:
        return None
    else:
        return 'hi' if states=='hi' else 'lo'


class Table:
    """
    Table represents a set of trials from the FlySoundAcquisition software.
    """

    def __init__(self,fn,fig_folder='./figpanels',progress_bar=False,add_probestate = False):
        self.topdir = default_data_directory(verbose=True)
        self.day,self.fly,self.cell = get_day_fly_cell(fn)
        self._dfc = '{}_F{}_C{}'.format(self.day,self.fly,self.cell)
        self.parquet = get_file(fn)

        self.fig_folder = fig_folder
        os.makedirs(self.fig_folder, exist_ok=True)
        
        self.progress_bar = progress_bar  # Set to False to disable progress bar in swifter
        self._outcomes_dict = {
            'no_as_no_mv': 'no aversive stimulus, no movement',
            'no_as_mv': 'no aversive stimulus, probe moves',
            'as_off': 'fly turns off aversive stimulus during trial',
            'as_off_late': 'fly turns off aversive stimulus in intertrial period',
            'timeout_fail': 'aversive stimulus never turned off and the probe was more flexed than the target',
            'timeout': 'aversive stimulus never turned off, no movement',
            'probe': 'probe_trial',
            'rest': 'rest trial',
            'info': 'info',
        }
        
        self.flycelldir = self.day + '_F' + str(self.fly) + '_C' + str(self.cell)
        self.path = os.path.join(self.topdir,self.day,self.flycelldir)

        # placeholders for computed properties
        self._df_filter = None
        self._probe_bins = None
        self._total_counts = None
        self._total_N = None
        self.ppdf = None  # probe positions dataframe
        self.hmdf = None
        self.df = None
        self._excluded_df = None
        self._tfn_template = None

        self._load_parquet()
        self._trial_file_name_template()
        self.get_trials()
        self.exclude_trials()
        self._bootstrap_meta_columns()
        self.get_target_positions()

        if not 'pyasState' in self.df.columns:
            if not add_probestate:
                print('Need to write to pyasState')
                return
            print('Need to write to pyasState')
            self._assume_probeZero_from_lo_target()
            self.write_column_to_trial_files('probeZero')
            self._map_assumed_pyasState()
            self.write_column_to_trial_files('pyasState')
            print('Re-adding pyasState and probeZero meta columns')
            self._bootstrap_meta_columns()
            self.get_target_positions()

        self.df = self.df.copy()
        self.df['op_cnd_blocks'] = (self.df['pyasState'] != self.df['pyasState'].shift(1)).cumsum()
        self.df['pyasState'] = self.df['pyasState'].astype(_pyasState_cat)


    def _assume_probeZero_from_lo_target(self):      
        target_x = list(self.targets.keys())
        if all([isinstance(i,str) for i in target_x]):
            raise KeyError('Targets are already assigned names: {}'.format(', '.join(target_x)))
        lo_x = (np.array(target_x)).max()       # the target with the higher number is lo force
        probeZero = lo_x+180                    # assume probeZero is 180 um to the right of the high-force, lo value end of the target (pyasXPosition)
        mapping = {(np.array(target_x)).max():probeZero, (np.array(target_x)).min(): probeZero}
        self.df['probeZero'] = self.df['pyasXPosition'].apply(
            lambda x: mapping.get(x, np.nan)
        )
        print('ProbeZero is calculated as {}. {} Trials do not have an appropriate value.'.format(probeZero,int(self.df['probeZero'].isna().sum())))
        print('Recommend running T.plot_probe_distribution() after assigning state')


    def _map_assumed_pyasState(self):     
        target_x = list(self.targets.keys())
        mapping  = {(np.array(target_x)).max():'lo', (np.array(target_x)).min(): 'hi'}
        
        self.df['pyasState'] = self.df['pyasXPosition'].map(mapping).astype('string')
        print('Writing assumed pyasState. {} Trials do not have an appropriate value.'.format(
            int(self.df['pyasState'].isna().sum())))
        print('Recommend running T.plot_probe_distribution(binwidth=2,bin_min=100,bin_max=800,filter=None,index=None) after assigning state')        


    @cached_property
    def genotype(self):
        return self._get_genotype()


    @property
    def dfc(self):
        return self._dfc


    def exclude_trials(self):
        if not 'Trial' in self.df.columns:
            self.get_trials()

        tru_arr = np.array([[1]])
        self.df.loc[self.df.index,'excluded'] = self.df.Trial.apply(lambda tr: np.array_equal(tru_arr,tr.excluded))
        to_exclude = self.df[self.df['excluded'] == True]
        if self._excluded_df is None:
            self._excluded_df = to_exclude
            print('Excluding trials:')
            print(self._excluded_df['Trial'].to_list())
        elif not to_exclude.empty:
            print('Concatenating to_exclude df ({}) to self._excluded_df ({})'.format(to_exclude.shape,self._excluded_df.shape))
            self._excluded_df = pd.concat([self._excluded_df,to_exclude],axis=0)
            self._excluded_df = self._excluded_df.sort_index()
        else:
            print('Excluding trials:')
            print(self._excluded_df['Trial'].to_list())

        self.df = self.df[self.df['excluded'] == False]


    def get_trials(self):
        if 'Trial' in self.df.columns:
            return self.df['Trial']

        def create_trial(row):
            trial_number = row.name  # Use the index (trial_number)
            # print(trial_number)
            file_name = self._generate_filename(trial_number)
            return Trial(file_name)
        
        print('Getting trials')
        self.df['Trial'] = self.df.swifter.progress_bar(self.progress_bar).apply(create_trial,axis=1,result_type='expand')
        # self.df['Trial'] = self.df.apply(create_trial,axis=1,result_type='expand')
        return self.df['Trial']
        

    def get_target_positions(self):
        """
        Get the target positions for the table.
        """
        self.targets = {}
        if 'pyasXPosition' not in self.df.columns or 'pyasWidth' not in self.df.columns:
            print(self.df.columns)
            raise ValueError("DataFrame does not contain 'pyasXPosition' or 'pyasWidth' columns.")
        if 'probeZero' in self.df.columns and 'pyasState' in self.df.columns:
            tempdf = self.df[['pyasState','probeZero','pyasXPosition','pyasWidth',]].copy()
            tempdf.loc[self.df['probeZero'].isna(),'probeZero'] = 0
            tempdf.loc[self.df['pyasState'].isna(),'pyasState'] = 'no_state'
            target_tuples = list(zip(tempdf.pyasState,
                                    tempdf.probeZero,
                                    tempdf.pyasXPosition-tempdf.probeZero, 
                                    tempdf.pyasWidth,))
            tuple_counter = Counter(target_tuples) # Assume two most common target positions
            for mct_item in tuple_counter.most_common(2):
                mct = mct_item[0]
                
                target_dict = {'probeZero': mct[1],
                            'pyasXPosition': mct[2],
                            'pyasWidth': mct[3],
                            'pyasState': mct[0]}
                self.targets[mct[0]] = target_dict
        else: 
            print('ProbeZero or pyasState not yet computed, collecting pyasXPosition and pyasWidth')
            target_tuples = list(zip(self.df.pyasXPosition, 
                                    self.df.pyasWidth))
            tuple_counter = Counter(target_tuples) # Assume two most common target positions
            for mct_item in tuple_counter.most_common(2):
                mct = mct_item[0]
                
                target_dict = {'pyasXPosition': mct[0],
                            'pyasWidth': mct[1]}
                self.targets[mct[0]] = target_dict
            
        print('Found {} target positions - {}'.format(len(tuple_counter), tuple_counter))

        return self.targets


    def extract_trial_properties(self,
                                prop_list: list =['total_duration','is_rest','is_probe','as_duration','as_outcome'],
                                persist: bool = False,
                                group_name: str = TRIAL_METADATA_GROUP,
                                rewrite_attrs: bool = False):
        """ Compute properties for each trial and add them to the DataFrame.
        """
        def compute_property(tr, prop):
            if not hasattr(tr, prop):
                print('Trial is {}'.format(tr))
                raise ValueError(f"Call T.extract_trial_properties() or T.extract_trial_properties(''{prop}'')")
            if callable(getattr(tr, prop)):
                return getattr(tr, prop)()
            else:
                return getattr(tr, prop)

        newcols = {}
        for prop in prop_list:
            s = self.df['Trial'].map(lambda tr: compute_property(tr, prop))
            if prop in _category_dict:
                s = s.astype(_category_dict[prop])
            newcols[prop] = s
        self.df = self.df.assign(**newcols)
        return self.df[prop_list]
    

    def as_outcome_recompute(self,
                                group_name: str = TRIAL_METADATA_GROUP,
                                rewrite_attrs: bool = False):

        # newcols = {}
        # for prop in prop_list:
        s = self.df['Trial'].map(lambda tr: tr.get_as_outcome(rerun=True,verbose=True))
        s = s.astype(_category_dict['as_outcome'])
        newcols = {'as_outcome':s}
        s = self.df['Trial'].map(lambda tr: tr.is_probe)
        newcols['is_probe'] = s
        s = self.df['Trial'].map(lambda tr: tr.is_rest)
        newcols['is_rest'] = s
        
        print(newcols.keys())
        self.df = self.df.assign(**newcols)
        return self.df[['as_outcome','is_probe','is_rest']]

    
    def probe_positions_df(self,df=None):
        # Collect a dataframe of trial information
        # Once trial.probe_position is called, that data has been loaded
        # Then its quicker to call this again, 
        # so the probe_positions are not saved to the df
        
        if df is None:
            df = self.df
            if not self.ppdf is None:
                print('Using existing probe positions dataframe')
                return self.ppdf            

        def get_probe_positions(row):
            trial = row.Trial
            if any(trial.probe_position>1000):
                trial.exclude(reason='probe values too high')
                ValueError('Have to exclude trial, probe is too high, rerun')
            pps = trial.probe_position - trial.probeZero
            return {'probe_positions': pps,
                    'probe_min': pps.min(),  # Max flexion
                    'probe_max': pps.max(),  # Should be close to ProbeZero
                    'probe_zero': trial.probeZero
                    }

        ppdf = df.swifter.progress_bar(self.progress_bar).apply(get_probe_positions, axis=1, result_type='expand')
        if ppdf.index.equals(self.df.index):
            self.df['probe_min'] = ppdf['probe_min']
            self.df['probe_max'] = ppdf['probe_max']
            self.df['probe_zero'] = ppdf['probe_zero']

        self.ppdf = ppdf
        print('Probe positions dataframe created with {} trials'.format(len(ppdf)))
        return ppdf


    def downsample_probe_df(self,ppdf=None):
        if ppdf is None:
            ppdf = self.probe_positions_df()

        ppdf['Trial'] = self.df.loc[ppdf.index,'Trial']
        totsmps = ppdf['probe_positions'].apply(lambda x: len(x)).sum(axis=0)

        def downsample(row):
            ds_idx = row.Trial.downsample_probe
            pp_dwnsmpl = row.probe_positions[ds_idx]
            return pp_dwnsmpl
        
        ppdf['probe_positions'] = ppdf.apply(downsample,axis=1)
        dssmps = ppdf['probe_positions'].apply(lambda x: len(x)).sum(axis=0)

        # print('Downsampled probe positions from {} to {} more unique samples'.format(totsmps,dssmps))
        return ppdf


    def ds_and_align_probe_position_hmdf(self, overwrite = False):
        # first test that all of the stimulus duration is the same
        trial_dur = self.df.eval("preDurInSec + stimDurInSec + postDurInSec")
        if trial_dur.nunique() != 1:
            min = trial_dur.argmin()
            tr = self.df.loc[self.df.index[min],'Trial']
            print("Trial duration is different, assuming the length of the shortest trial: Trial {} - {} s".format(min,trial_dur.iloc[min]))
        else:
            tr = self.df.loc[self.df.index[0],'Trial']

        if (not self.hmdf is None) and (not overwrite):
            return self.hmdf
        
        probe_positions = self.probe_positions_df()
        ds_time = tr.time[tr.downsample_probe]
        ds_trpridx = tr.downsample_probe[ds_time < tr.trialtime[-1]]
        ds_trtime = ds_time[ds_time < tr.trialtime[-1]]

        try:
            probe_positions['probe_positions'] = probe_positions['probe_positions'].apply(lambda x: x[ds_trpridx])
        except IndexError as e:
            raise(e)
        probe_positions['length'] = probe_positions['probe_positions'].apply(lambda x: len(x))

        self.hmdf = pd.DataFrame(probe_positions['probe_positions'].apply(lambda a: a.ravel()).to_list(), index=probe_positions.index, columns=ds_trtime)
        return self.hmdf


    def probe_position_distribution(self,binwidth=2,bin_min=None,bin_max=None,filter=None,index=None):

        if index is None:
            index = self.df.index

        filtered_df = self.df.loc[index]
        if not filter is None:
            for key in filter:
                print(key)
                filtered_df = filtered_df.loc[filtered_df[key]==filter[key],:]

        if bin_min is None:
            ValueError('bin_min must be specified') # bin_min = probe_positions.probe_min.min() # Max flexion
        if bin_max is None:
            ValueError('bin_max must be specified') # bin_max = probe_positions.probe_max.max() # Should be ProbeZero
        probe_bins = np.arange(bin_min, bin_max, binwidth)

        if self._df_filter == filter and (not self._probe_bins is None):
            if np.array_equal(self._probe_bins, probe_bins):
                print('Using existing probe bins')
                return self._total_counts, self._total_N, self._probe_bins
        else:
            print('New filter or probe bins, recomputing distributions')

        def trial_probe_position_ds(row):
            trial = row.Trial
            if any(trial.probe_position>1000):
                trial.exclude(reason='probe values too high')
                ValueError('Have to exclude trial, probe is too high, rerun')
            pps = trial.probe_position - trial.probeZero
            pps = pps[row.Trial.downsample_probe]

            counts, _ = np.histogram(pps, bins=probe_bins)
            # return counts, len(pps)
            return {'counts': counts,
                    'N': len(pps)
                    }

        print('Counting downsampled probe positions in bins')
        counts_per_bin = filtered_df.swifter.progress_bar(self.progress_bar).apply(trial_probe_position_ds, axis=1, result_type='expand')
 
        stacked_counts = np.stack(counts_per_bin['counts'].values)
        total_counts = np.sum(stacked_counts, axis=0)
        total_N = np.sum(counts_per_bin['N'].values)

        self._df_filter = filter
        self._total_counts = total_counts
        self._total_N = total_N
        self._probe_bins = probe_bins
        
        return total_counts, total_N, probe_bins


    def _on_target_fraction(self,target,total_counts,probe_bins):
        binwidth = np.diff(probe_bins[2:4])
        tbin_max = target['pyasXPosition']
        tbin_min = target['pyasXPosition'] + target['pyasWidth']

        target_bins = (probe_bins[:-1] <= tbin_min+2*binwidth) & (probe_bins[:-1] >= tbin_max-2*binwidth)

        ontarget = np.sum(total_counts[target_bins]) / np.sum(total_counts)
        print('On target ({} - {}): {:.2f} ({}/{})'.format(
            tbin_max-2*binwidth,
            tbin_min+2*binwidth,
            ontarget,
            np.sum(total_counts[target_bins]),
            np.sum(total_counts)
        ))
        return ontarget
    

    def assign_column_value(self, column_name, value, trial_min=None, trial_max=None, index=None, dtype=None, write_to_hdf5=False):
        # add tag like VNC cut
        if all(x is None for x in [trial_min, trial_max, index]):
            raise ValueError("Must specify trial_min, trial_max, or trial_index to select rows.")

        if index is not None:
            index = self.df.index[index] if isinstance(index, (list, np.ndarray, pd.Index)) else [index]
        else:
            index = self.df.index
            if not trial_min is None:
                index = index[index>=trial_min]

            if not trial_max is None:
                index = index[index<=trial_max]

        # Assign the value
        # Validate value length if assigning a sequence
        if hasattr(value, '__len__') and not isinstance(value, str):
            if len(index) != len(value):
                raise ValueError(f"Length of value ({len(value)}) does not match length of index ({len(index)})")
        
        if column_name not in self.df.columns:
            self.df[column_name] = np.nan
        self.df.loc[index, column_name] = value

        # Try to infer dtype if not provided
        if dtype is None and column_name in _category_dict:
            dtype = _category_dict[column_name]

        # Validate category membership if dtype was inferred
        if isinstance(dtype, pd.CategoricalDtype):
            if value not in dtype.categories:
                raise ValueError(f"Value '{value}' is not a valid category in {dtype.categories}")
            self.df[column_name] = self.df[column_name].astype(dtype)

        if write_to_hdf5:
            # print('Writing column {} to trial files.'.format(column_name))
            self.write_column_to_trial_files(column_name)
        else: 
            print('Not writing column {} to trial files. Call T.write_column_to_trial_files(''{}'')'.format(column_name,column_name))

        return pd.Series([value], dtype=dtype).iloc[0]

        
    def write_column_to_trial_files(self, column_name):
        """
        Write the value in `column_name` to each Trial's HDF5 file,
        skipping if it's already present and matches.
        """
        print('Writing {} to each trial'.format(column_name))
        def writer(row):
            trial :Trial = row['Trial']
            value = row[column_name]
            if isinstance(value,str):
                return trial.write_string_if_changed(column_name, value)
            elif isinstance(value,float):
                return trial.write_scalar_if_changed(column_name, value)
            else:
                return np.nan

        return self.df.swifter.progress_bar(self.progress_bar).apply(writer, axis=1)
        

    def compute_trial_method(self, method_name: str,*,trial_col: str = 'Trial',debug: bool = False):
        """Compute a function on the trial object and add to the df"""
        if not hasattr(self.df[trial_col].iloc[0], method_name):
            raise AttributeError(f"Trial objects do not have a method called '{method_name}'")

        trial_ser = self.df[trial_col].copy().swifter.progress_bar(self.progress_bar).apply(lambda trial: getattr(trial,method_name)(debug))
        self.df[method_name] = trial_ser


    def open_notes_files(self):
        """
        Finds all text files in the directory that start with 'notes' 
        and opens them in VS Code.
        """
        # Validate if the directory exists
        if not os.path.isdir(self.path):
            raise FileNotFoundError(f"Directory not found: {self.path}")

        # Find text files starting with 'notes'
        files_to_open = [
            os.path.join(self.path, file)
            for file in os.listdir(self.path)
            if file.startswith("notes") and file.endswith(".txt")
        ]
        # print(files_to_open)

        # Check if there are files to open
        if not files_to_open:
            print("No 'notes' files found in the directory.")
            return

        # Open each file in a new VS Code window
        pathtocode = 'C:\\Users\\tony\\AppData\\Local\\Programs\\Microsoft VS Code\\bin\\code.cmd'
        for file_path in files_to_open:
            subprocess.run([pathtocode, file_path], check=True)
            # print(f"Opened file: {file_path}")


    def exclude_list_of_trials(self, index=None, reason=None):
        if reason is None:
            raise ValueError('better have a good reason to exclude many trials')
        subdf = self.df.loc[index]
        subdf['Trial'].apply(lambda tr: tr.exclude(reason=reason))
        self.exclude_trials()


    def find_outcome_sequence(self,outcome_sequence:list = None):
        """
        A successful trial is a trial where the fly enters the target and stays at least 2 trials.
        """
        if outcome_sequence is None:
            KeyError('Need to input a sequence')

        if 'as_outcome' not in self.df.columns:
            self.extract_trial_properties(prop_list=['as_outcome'])
        
        if 'on_target' not in self.df.columns:
            self.extract_trial_properties(prop_list=['on_target'])
        
        as_off = (self.df['as_outcome'] == 'as_off') | (self.df['as_outcome'] == 'as_off_late')
        no_as_no_mv = (self.df['as_outcome'] == 'no_as_no_mv')
        no_as = no_as_no_mv | (self.df['as_outcome'] == 'no_as_mv')
        rest = (self.df['as_outcome'] == 'rest')
        probe = (self.df['as_outcome'] == 'probe')

        on_target = ((self.df['on_target']>.9) & rest) | ((self.df['on_target']>.9) & probe)

        s = pd.Index(self.df.index).to_series(index=self.df.index)
        next_contig = s.shift(-1).sub(s).eq(1)

        outcome_list=[]
        shft=0
        for outcome in outcome_sequence:
            # take as_off trials. Ask if the next two trials are either no_as or on_target
            if outcome == 'as_off':
                outcome_list.append(as_off.shift(shft))
            elif outcome== 'no_as_no_mv':
                outcome_list.append(no_as_no_mv.shift(shft))
            elif outcome=='no_as_mv':
                outcome_list.append(no_as_no_mv.shift(shft))
            shft = shft-1
            # no_as_1 = no_as.shift(-1) | on_target.shift(-1)
            # no_as_no_mv_1 = no_as_no_mv.shift(-1) | on_target.shift(-1)
            # no_as_no_mv_2 = no_as_no_mv.shift(-2) | on_target.shift(-2)
        matches=None
        for ol in outcome_list:
            if matches is None:
                matches=ol
            else:
                matches = matches & ol


        sequence_matches = self.df[['as_outcome','on_target']].copy()
        sequence_matches['matches'] = matches
        
        print('no_as_no_mv: {}; no_as_mv: {}; as_off: {}; matches: {}'.format(
            (self.df['as_outcome']=='no_as_no_mv').sum(),
            (self.df['as_outcome']=='no_as_mv').sum(),
            (self.df['as_outcome']=='as_off').sum(),
            sequence_matches['matches'].sum(),
        ))
        return sequence_matches

    
    def find_successful_trials(self):
        """
        A successful trial is a trial where the fly enters the target and stays at least 2 trials.
        """
        if 'as_outcome' not in self.df.columns:
            self.extract_trial_properties(prop_list=['as_outcome'])
        
        if 'on_target' not in self.df.columns:
            self.extract_trial_properties(prop_list=['on_target'])
        
        as_off = (self.df['as_outcome'] == 'as_off') | (self.df['as_outcome'] == 'as_off_late')
        no_as_no_mv = (self.df['as_outcome'] == 'no_as_no_mv')
        no_as = no_as_no_mv | (self.df['as_outcome'] == 'no_as_mv')
        rest = (self.df['as_outcome'] == 'rest')
        probe = (self.df['as_outcome'] == 'probe')

        on_target = ((self.df['on_target']>.9) & rest) | ((self.df['on_target']>.9) & probe)

        s = pd.Index(self.df.index).to_series(index=self.df.index)
        next_contig = s.shift(-1).sub(s).eq(1)

        # take as_off trials. Ask if the next two trials are either no_as or on_target
        no_as_1 = no_as.shift(-1) | on_target.shift(-1)
        no_as_no_mv_1 = no_as_no_mv.shift(-1) | on_target.shift(-1)
        no_as_no_mv_2 = no_as_no_mv.shift(-2) | on_target.shift(-2)
        self.df['soft_success'] = next_contig & as_off & no_as_1 # & no_as_2
        self.df['hard_success'] = next_contig & as_off & no_as_no_mv_1 & no_as_no_mv_2
        self.df['success'] = self.df['soft_success'] | self.df['hard_success']

        print('no_as_no_mv: {}; no_as_mv: {}; as_off: {}; success: {}; hard_success: {}'.format(
            (self.df['as_outcome']=='no_as_no_mv').sum(),
            (self.df['as_outcome']=='no_as_mv').sum(),
            (self.df['as_outcome']=='as_off').sum(),
            self.df['success'].sum(),
            self.df['hard_success'].sum(),
        ))

    def classify_successful_trials(self):
        """ Classify successful trials into settle or enter.
        """

        self.df['success_type'] = 'unsuccessful'
        
        trials = self.df.loc[['Trial','success']].copy()
        trials['next_trial'] = trials['Trial'].shift(-1)
        trials = trials.loc[trials['success']]

        if 'next_trial' not in self.df.columns:
            self.df['next_trial'] = self.df['Trial'].shift(-1)
        def classify_success(row):
            if row['success']:
                tr1 = row['Trial']
                tr2 = row['next_trial']
                
                state = 'lo' if tr1.pyasState == 'lo' else 'hi'
                xpos = tr1.params['pyasXPosition']
                wdth = tr1.params['pyasWidth']

                if row['next_trial'] is not None:
                    next_pp = row['next_trial'].probe_position
                    target = next

                    if np.all(pp == next_pp):
                        return 'successful'

            elif not row['success']:
                return 'unsuccessful'

        self.df['success_type'] = self.df.copy().apply(classify_success, axis=1)

        if 'target_enter_side' not in self.df.columns:
            self.extract_trial_properties(prop_list=['on_target'])
        # what kind of success?
        self.df['success_type'] = 'unsuccessful'

        # settle

        # enter

    

    # ---------------------------------------------------------
    # Helper (Internal) Methods
    # ---------------------------------------------------------
    def _load_parquet(self):
        self.df = pd.read_parquet(os.path.join(self.path,self.parquet))

        self.df['trial'] = self.df['trial'].astype(int)
        self.df.rename(columns={'trial': 'trial_number'}, inplace=True)
        self.df.set_index('trial_number',inplace=True)
        
        def matlab_datenum_to_datetime(matlab_datenum):
            # Offset: MATLAB datenum for 1970-01-01 is 719529
            offset = 719529
            # Convert MATLAB datenum to Python datetime
            return datetime(1970, 1, 1) + timedelta(days=(matlab_datenum - offset))
        self.df['timestamp'] = self.df['timestamp'].apply(matlab_datenum_to_datetime)
        print('T = pd.read_parquet("{}")'.format(os.path.join(self.path,self.parquet).replace('\\','\\\\')))
        # print(self.df['samples'])
        try:
            self.df = self.df[self.df['samples'].apply(lambda x: x.size > 0)]
        except KeyError:
            raise(KeyError('No samples variable. There are probably some empty trials that are not in the continuous file. Delete them, remake the table, and run again'))


    def _trial_file_name_template(self):
        original_filename = self.parquet
        parts = original_filename.split('_')
        parts.insert(1, "Raw")
        parts[-1] = parts[-1].replace("Table.parquet", "{x}.mat")
        self._tfn_template = "_".join(parts)


    def _generate_filename(self,tn):
        return self._tfn_template.format(x=tn)
    

    def _get_genotype(self):
        matdata = loadmat(os.path.join(self.path,'Acquisition_{}_F{}_C{}.mat'.format(self.day,self.fly,self.cell)))
        gntyp = matdata['acqStruct']['flygenotype'].ravel()
        self._genotype = str(gntyp[0][0])
        self._genotype = self._genotype.replace('.', '_')
        self._genotype = self._genotype.replace('>', '_')
        self._genotype = self._genotype.replace('/', '_')
        self._genotype = self._genotype.replace('[', '')
        self._genotype = self._genotype.replace(']', '')
        self._genotype = self._genotype.replace('*', '')
        self._standardize_genotype()

        return self._genotype


    def _bootstrap_meta_columns(self) -> None:
        """Create df columns for every /meta key across all trials and fill values."""
        trials = self.df['Trial']

        # 1) discover union of keys
        get_keys = (lambda tr: tr.list_meta_keys())
        print('Getting all meta keys')
        key_lists = trials.swifter.progress_bar(self.progress_bar).apply(get_keys)
        keys = sorted({k for ks in key_lists for k in (ks or [])})

        print(keys)
        if not keys:
            return

        # 2) pull each key’s value into a Series and install
        for key in keys:
            self.extract_trial_properties([key])



    def _standardize_genotype(self):
        geno_map = {
            '31H05_pJFRC7':                    'w;pJFRC7;31H05-GAL4',
            '+;31H05-Gal4_pJFRC7':             '+;pJFRC7;31H05-GAL4',
            'SS61350_pJFRC7':                  '+;SS61350_pJFRC7',
            '+;pJFRC7;SS61350':                '+;SS61350_pJFRC7',
            'ss61350_pJFRC7':                  '+;SS61350_pJFRC7',
            'iav_Kir2_1':                      'w;iav-GAL4_UAS-Kir21',
            'iav-Gal4_+;+;UAS-Kir2_1':         '+;iav-GAL4_UAS-Kir21',
            '+;iav-Gal4_UAS-Kir2_1':           '+;iav-GAL4_UAS-Kir21',
            '+;TH-Gal4_UAS-Kir2_1':            '+;UAS-Kir21;TH-GAL4',
            '+;TH-Gal4_pJFRC7':                '+;pJFRC7;TH-GAL4',
            'w, NorpA':                        'w,NorpA',
            'ppk-Gal4;10XUAS-ChR in WT':       '+;ppk-GAL4;10XUAS-ChR',
            'norpAE55':                        '+,NorpA',
            '+;pFJRC7;+':                      '+;pJFRC7;+',
            'norpA':                           'w,NorpA',
            'Hot-Cell-Gal4 (test)':            '+;Hot-Cell-GAL4_pJFRC7;10XUAS-ChR',
            'Hot-Cell-LexA_Chr;81A06_pJFRC7':  '+;Hot-Cell-LexA_Chr;81A06-GAL4_pJFRC7',
            'Hot-Cell-LexA_Chr;35c09_pJFRC7':  '+;Hot-Cell-LexA_Chr;35C09-GAL4_pJFRC7',
            'Hot-Cell-LexA_Chr;35C09_pJFRC7':  '+;Hot-Cell-LexA_Chr;35C09-GAL4_pJFRC7',
            'Hot-Cell-LexA_Chr;78E05_pJFRC7':  '+;Hot-Cell-LexA_Chr;78E05-GAL4_pJFRC7',
            'Hot-Cell-LexA_Chr;31H05_pJFRC7':  '+;Hot-Cell-LexA_Chr;31H05-GAL4_pJFRC7',
        }
        if self._genotype in geno_map.keys():
            self._genotype = geno_map[self._genotype]
        elif self._genotype in geno_map.keys():
            print('Genotype {} is standardized'.format(self._genotype))
        else:
            warnings.warn('Genotype {} not standardized'.format(self._genotype))

        return self._genotype

    # ---------------------------------------------------------
    # Dunder Methods
    # ---------------------------------------------------------
    def __dir__(self):
        # Filter out attributes that start with an underscore
        return [attr for attr in super().__dir__() if not attr.startswith("_")]
    

    def __repr__(self):
        return('Day {}, F{}, C{}: {}'.format(self.day,self.fly,self.cell,self.parquet))
    



# Table.plot_some_trials = plot_some_trials
# Table.plot_outcomes = plot_outcomes
# Table.probe_position_heatmap = probe_position_heatmap
# Table.plot_probe_distribution = plot_probe_distribution
# Table.plot_outcomes = plot_outcomes

for name in dir(table_plotters):
    obj = getattr(table_plotters, name)
    if isinstance(obj, types.FunctionType) and name.startswith("plot_"):
        setattr(Table, name, obj)

for name in dir(table_movie_maker):
    obj = getattr(table_movie_maker, name)
    if isinstance(obj, types.FunctionType) and name.startswith("make_"):
        setattr(Table, name, obj)

for name in dir(table_export_methods):
    obj = getattr(table_export_methods, name)
    if isinstance(obj, types.FunctionType) and name.startswith("export_"):
        setattr(Table, name, obj)   

for name in dir(table_scalars):
    obj = getattr(table_scalars, name)
    if isinstance(obj, types.FunctionType) and name.startswith("compute_"):
        setattr(Table, name[len("compute_"):], obj)   