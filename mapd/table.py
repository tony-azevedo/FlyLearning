from scipy.io import loadmat
from .helpers import get_day_fly_cell, get_file
# from .table_plotters import plot_some_trials, plot_outcomes, plot_probe_distribution, probe_position_heatmap
import types
from . import table_plotters
from . import probe_movie_maker
from .trial import Trial
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
    Trial represents a trial from the FlySoundAcquisition software.

    For now, trials are saved in the old format, not as h5 files, and the data has to be loaded with scipy

    Attributes:
    -----------
    keys : list
        list of 

    Methods:
    --------
    greet() -> str:
        Returns a greeting message including the name of the entity.
    """

    def __init__(self,fn,fig_folder='./figpanels'):
        self.topdir = 'D:\\Data'
        self.day,self.fly,self.cell = get_day_fly_cell(fn)
        self._dfc = '{}_F{}_C{}'.format(self.day,self.fly,self.cell)
        self.fn = get_file(fn)

        self.fig_folder = fig_folder
        os.makedirs(self.fig_folder, exist_ok=True)
        
        self.flycelldir = self.day + '_F' + str(self.fly) + '_C' + str(self.cell)
        self.path = os.path.join(self.topdir,self.day,self.flycelldir)

        self.df = None
        self._load_parquet()

        self._tfn_template = None
        self._trial_file_name_template()
        
        self._excluded_df = None
        self.exclude_trials()
        
        self._genotype = None
        
        self.hmdf = None


    @property
    def trial_list(self):
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
            return self.df.loc[:,['Trial','is_rest','is_probe','as_duration','as_outcome','op_cnd_blocks','pyasState']]

        def create_trial(row):
            trial_number = row.name  # Use the index (trial_number)
            file_name = self._generate_filename(trial_number)
            tr = Trial(file_name)
            try: 
                trial_dict = {
                'Trial':    tr,
                'is_rest':  tr.is_rest,
                'is_probe':  tr.is_probe,
                'as_duration':  tr.as_duration,
                'as_outcome':  tr.as_outcome,
                }
            except ValueError as e:
                print(f'{tr}: ',e)
                trial_dict = {
                    'Trial':    tr,
                    'is_rest':  None,
                    'is_probe':  None,
                    'as_duration':  None,
                    'as_outcome':  None,
                    }

            return trial_dict
        
        print('Getting trials')
        new_cols = self.df.swifter.apply(create_trial,axis=1,result_type='expand')
        new_cols['as_outcome'] = new_cols['as_outcome'].astype(_outcomes_cat)
        
        # numbering blocks where learning is engaged.
        self.df = self.df.join(new_cols)
        self.df['op_cnd_blocks'] = (self.df['pyasState'] != self.df['pyasState'].shift(1)).cumsum()
        self.df['pyasState'] = self.df['pyasState'].astype(_pyas_state_cat)
        return new_cols
        

    def probe_positions_df(self,df=None):
        # Collect a dataframe of trial information
        # Once trial.probe_position is called, that data has been loaded
        # Then its quicker to call this again, 
        # so the probe_positions are not saved to the df
        
        if df is None:
            df = self.df

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
        
        ppdf = df.swifter.progress_bar(False).apply(get_probe_positions, axis=1, result_type='expand')
        if ppdf.index.equals(self.df.index):
            self.df['probe_min'] = ppdf['probe_min']
            self.df['probe_max'] = ppdf['probe_max']
            self.df['probe_zero'] = ppdf['probe_zero']

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
        
        self.df[method_name] = self.df[trial_col].apply(lambda trial: getattr(trial,method_name)())


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


    # ---------------------------------------------------------
    # Helper (Internal) Methods
    # ---------------------------------------------------------
    def _load_parquet(self):
        self.df = pd.read_parquet(os.path.join(self.path,self.fn))

        self.df['trial'] = self.df['trial'].astype(int)
        self.df.rename(columns={'trial': 'trial_number'}, inplace=True)
        self.df.set_index('trial_number',inplace=True)
        
        def matlab_datenum_to_datetime(matlab_datenum):
            # Offset: MATLAB datenum for 1970-01-01 is 719529
            offset = 719529
            # Convert MATLAB datenum to Python datetime
            return datetime(1970, 1, 1) + timedelta(days=(matlab_datenum - offset))
        self.df['timestamp'] = self.df['timestamp'].apply(matlab_datenum_to_datetime)
        print('T = pd.read_parquet("{}")'.format(os.path.join(self.path,self.fn).replace('\\','\\\\')))
        # print(self.df['samples'])
        try:
            self.df = self.df[self.df['samples'].apply(lambda x: x.size > 0)]
        except KeyError:
            raise(KeyError('No samples variable. There are probably some empty trials that are not in the continuous file. Delete them, remake the table, and run again'))


    def _trial_file_name_template(self):
        original_filename = self.fn
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
        return('Day {}, F{}, C{}: {}'.format(self.day,self.fly,self.cell,self.fn))
    



# Table.plot_some_trials = plot_some_trials
# Table.plot_outcomes = plot_outcomes
# Table.probe_position_heatmap = probe_position_heatmap
# Table.plot_probe_distribution = plot_probe_distribution
# Table.plot_outcomes = plot_outcomes

for name in dir(table_plotters):
    obj = getattr(table_plotters, name)
    if isinstance(obj, types.FunctionType) and name.startswith("plot_"):
        setattr(Table, name, obj)

for name in dir(probe_movie_maker):
    obj = getattr(probe_movie_maker, name)
    if isinstance(obj, types.FunctionType) and name.startswith("make_"):
        setattr(Table, name, obj)