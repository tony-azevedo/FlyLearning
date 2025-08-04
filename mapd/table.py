from scipy.io import loadmat
from .helpers import get_day_fly_cell, get_file, default_data_directory
# from .table_plotters import plot_some_trials, plot_outcomes, plot_probe_distribution, probe_position_heatmap
import types
from . import table_plotters
from . import table_movie_maker
from . import table_export_methods
from . import table_scalars
from .trial import Trial

import importlib
importlib.reload(table_plotters)
importlib.reload(table_movie_maker)
importlib.reload(table_export_methods)

import os
import subprocess
import pandas as pd
# import modin.pandas as pd
import swifter
import numpy as np
from datetime import datetime, timedelta
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# from matplotlib import pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

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

_outcomes_cat =             pd.api.types.CategoricalDtype(categories=_outcomes_dict.keys(), ordered=True)
_pyas_state_cat =                     pd.api.types.CategoricalDtype(categories=['lo','hi'], ordered=True)
_vnc_status_cat =           pd.api.types.CategoricalDtype(categories=['intact','cut'], ordered=True)
_filtercube_status_cat =    pd.api.types.CategoricalDtype(categories=['green','blue'], ordered=False)

_categories_list = [_outcomes_cat,_pyas_state_cat,_vnc_status_cat,_filtercube_status_cat]


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

    def __init__(self,fn,fig_folder='./figpanels'):
        self.topdir = default_data_directory(verbose=True)
        self.day,self.fly,self.cell = get_day_fly_cell(fn)
        self._dfc = '{}_F{}_C{}'.format(self.day,self.fly,self.cell)
        self.parquet = get_file(fn)

        self.fig_folder = fig_folder
        os.makedirs(self.fig_folder, exist_ok=True)
        
        self.progress_bar = True  # Set to False to disable progress bar in swifter
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

        self.df = None
        self._load_parquet()

        self._tfn_template = None
        self._trial_file_name_template()
        
        self._excluded_df = None
        self.exclude_trials()
        
        self._genotype = None
        
        self.ppdf = None  # probe positions dataframe
        self.hmdf = None


    @property
    def trial_series(self):
        if not 'Trial' in self.df.columns:
            self.get_trials()
        return self.df['Trial']
    

    @property
    def _outcomes_cat(self):
        return _outcomes_cat

    @property
    def genotype(self):
        if self._genotype is None:
            self._get_genotype()
        return self._genotype

    def exclude_trials(self):
        if not 'Trial' in self.df.columns:
            self.get_trials()

        tru_arr = np.array([[1]])
        self.df['excluded'] = self.df.Trial.apply(lambda tr: np.array_equal(tru_arr,tr.excluded))
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
            file_name = self._generate_filename(trial_number)
            return Trial(file_name)
        
        print('Getting trials')
        self.df['Trial'] = self.df.swifter.progress_bar(self.progress_bar).apply(create_trial,axis=1,result_type='expand')
        return self.df['Trial']
        

    def add_trial_properties(self,prop_list=['total_duration','is_rest','is_probe','as_duration','as_outcome'],rerun=False):
        """ Compute properties for each trial and add them to the DataFrame.
        """
        def compute_property(tr, prop, rerun=False):
            if not hasattr(tr, prop):
                raise ValueError(f"Trial object does not have a property called '{prop}'")
            if callable(getattr(tr, prop)):
                return getattr(tr, prop)(rerun=rerun)
            else:
                return getattr(tr, prop)

        for prop in prop_list:
            print('Computing trial {}'.format(prop))
            self.df[prop] = self.df['Trial'].swifter.progress_bar(self.progress_bar).apply(lambda tr: compute_property(tr, prop))
            
            if prop == 'as_outcome':
                # Convert to categorical type
                self.df['as_outcome'] = self.df['as_outcome'].astype(_outcomes_cat)

        if 'op_cnd_blocks' not in self.df.columns:
            self.df['op_cnd_blocks'] = (self.df['pyasState'] != self.df['pyasState'].shift(1)).cumsum()
            self.df['pyasState'] = self.df['pyasState'].astype(_pyas_state_cat)
        
        return self.df[prop_list]
    

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
            pps = trial.probe_position - trial.params['probeZero']
            return {'probe_positions': pps,
                    'probe_min': pps.min(),  # Max flexion
                    'probe_max': pps.max(),  # Should be close to ProbeZero
                    'probe_zero': trial.params['probeZero'],
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


    def get_probe_position_df(self):
        # first test that all of the stimulus duration is the same
        def trial_duration(row):
            return row.preDurInSec+row.stimDurInSec+row.postDurInSec
        trial_dur = self.df.apply(trial_duration,axis=1)
        if not trial_dur.nunique() == 1:
            raise NotImplementedError('What if the trial duration is different?')

        if not self.hmdf is None:
            return self.hmdf
        
        probe_positions = self.probe_positions_df()
        tr = self.df.loc[probe_positions.index[0],'Trial']
        ds_time = tr.time[tr.downsample_probe]
        ds_trpridx = tr.downsample_probe[ds_time < tr.trialtime[-1]]
        ds_trtime = ds_time[ds_time < tr.trialtime[-1]]

        probe_positions['probe_positions'] = probe_positions['probe_positions'].apply(lambda x: x[ds_trpridx])
        probe_positions['length'] = probe_positions['probe_positions'].apply(lambda x: len(x))

        self.hmdf = pd.DataFrame(probe_positions['probe_positions'].apply(lambda a: a.ravel()).to_list(), index=probe_positions.index, columns=ds_trtime)
        return self.hmdf


    def probe_position_distribution(self,binwidth=2,bin_min=None,bin_max=None,filter=None,index=None,savefig=False,format=None):

        if index is None:
            index = self.df.index

        filtered_df = self.df.loc[index]
        if not filter is None:
            # for key in filter:
            #     probe_positions.loc[ppi,key] = self.df.loc[ppi,key]
            for key in filter:
                filtered_df = filtered_df.loc[filtered_df[key]==filter[key],:]

        if bin_min is None:
            ValueError('bin_min must be specified') # bin_min = probe_positions.probe_min.min() # Max flexion
        if bin_max is None:
            ValueError('bin_max must be specified') # bin_max = probe_positions.probe_max.max() # Should be ProbeZero
        probe_bins = np.arange(bin_min, bin_max, binwidth)

        def trial_probe_position_ds(row):
            trial = row.Trial
            if any(trial.probe_position>1000):
                trial.exclude(reason='probe values too high')
                ValueError('Have to exclude trial, probe is too high, rerun')
            pps = trial.probe_position - trial.params['probeZero']
            pps = pps[row.Trial.downsample_probe]

            counts, _ = np.histogram(pps, bins=probe_bins)
            # return counts, len(pps)
            return {'counts': counts,
                    'N': len(pps)
                    }

        print('Calculating downsampled probe positions')
        counts_per_bin = filtered_df.swifter.progress_bar(self.progress_bar).apply(trial_probe_position_ds, axis=1, result_type='expand')
        stacked_counts = np.stack(counts_per_bin['counts'].values)
        total_counts = np.sum(stacked_counts, axis=0)
        total_N = np.sum(counts_per_bin['N'].values)
        return total_counts, total_N, probe_bins

    
    def add_df_category(self, category, trial_min=None, trial_max=None, trial_index=None, categories=None):
        # add tag like VNC cut
        if all([x is None for x in [trial_min,trial_max,trial_index]]):
               ValueError('Need some indication of which trials')
        
        index = self.df.index
        if not trial_min is None:
            index = index[index.get_loc(trial_min):]

        if not trial_max is None:
            index = index[:index.get_loc(trial_max)]

        if not categories is None:
            if isinstance(categories,pd.CategoricalDtype):
                raise NotImplementedError('what if input is a categorical type?')

            else:
                cat_name=categories
                try:
                    scope = globals()
                    categories =  scope.get(categories, None)
                except ModuleNotFoundError:
                    raise ImportError(f"Module '{categories}' not found.")
                if not category in categories.categories:
                    raise ValueError(f"Category '{category}' not found in {categories.categories}.")

        else:
            # First check if category is a category in the _category_list
            for dtype in _categories_list:
                if category in dtype.categories:
                    categories=dtype
                    print(categories.categories)
                    break
            if categories is None:
                raise NotImplementedError('What if there still is no categories?')
            
        assert(isinstance(categories,pd.CategoricalDtype))
        assert(category in categories.categories)
        
        # first_category = categories.categories[0]  # 'intact'
        # if not categories in self.df.columns:
        #     print(first_category)
        #     self.df.loc[self.df.index,cat_name] = pd.Series([first_category] * len(self.df), dtype=categories)
        self.df.loc[index,cat_name] = category

        return index


    def compute_trial_method(self, method_name: str,trial_col: str = 'Trial'):
        """Compute a function on the trial object and add to the df"""
        if not hasattr(self.df[trial_col].iloc[0], method_name):
            raise AttributeError(f"Trial objects do not have a method called '{method_name}'")

        self.df[method_name] = self.df[trial_col].swifter.progress_bar(self.progress_bar).apply(lambda trial: getattr(trial,method_name)())


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


    def find_successful_trials(self):
        """
        A successful trial is a trial where the fly enters the target and stays at least 2 trials.
        """
        if 'as_outcome' not in self.df.columns:
            self.add_trial_properties(prop_list=['as_outcome'])
        
        if 'on_target' not in self.df.columns:
            self.add_trial_properties(prop_list=['on_target'])

        as_off = (self.df['as_outcome'] == 'as_off') | (self.df['as_outcome'] == 'as_off_late')
        no_as = (self.df['as_outcome'] == 'no_as_no_mv') | (self.df['as_outcome'] == 'no_as_mv')
        rest = (self.df['as_outcome'] == 'rest')
        probe = (self.df['as_outcome'] == 'probe')

        on_target = ((self.df['on_target']>.9) & rest) | ((self.df['on_target']>.9) & probe)

        # take as_off trials. Ask if the next two trials are either no_as or on_target
        no_as_1 = no_as.shift(-1) | on_target.shift(-1)
        no_as_2 = no_as.shift(-2) | on_target.shift(-2)
        self.df['success'] = as_off & no_as_1 & no_as_2

    
    def classify_successful_trials(self):
        """ Classify successful trials into settle or enter.
        """

        self.df['success_type'] = 'unsuccessful'

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

            else:
                return 'unsuccessful'

        self.df['success_type'] = self.df.apply(classify_success, axis=1)

        if 'target_enter_side' not in self.df.columns:
            self.add_trial_properties(prop_list=['on_target'])
        # what kind of success?
        self.df['success_type'] = 'unsuccessful'

        # settle




    def compute_successful_enter_trials(self):
        """
        Compute the number of successful enter trials.
        """
        if 'as_outcome' not in self.df.columns:
            raise ValueError("DataFrame does not contain 'as_outcome' column.")
        
        successful_enter_trials = self.df[self.df['as_outcome'] == 'enter'].shape[0]
        return successful_enter_trials

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