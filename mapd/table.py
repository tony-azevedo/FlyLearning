from scipy.io import loadmat
from .helpers import get_day_fly_cell, get_file
from .trial import Trial
import os
import subprocess
import pandas as pd
import swifter
import numpy as np
from datetime import datetime, timedelta
from matplotlib import pyplot as plt
import matplotlib.patches as patches

_outcomes_dict = {
    'no_as_no_mv': 'no aversive stimulus, no movement',
    'no_as_mv': 'no aversive stimulus, probe moves',
    'as_off': 'fly turns off aversive stimulus during trial',
    'as_off_late': 'fly turns off aversive stimulus in intertrial period',
    'timeout': 'aversive stimulus never turned off, no movement',
    'timeout_fail': 'aversive stimulus never turned off, even though there was movement',
    'probe': 'probe_trial',
    'rest': 'rest trial',
}
_outcomes_cat =     pd.api.types.CategoricalDtype(categories=_outcomes_dict.keys(), ordered=True)
_pyas_state_cat =        pd.api.types.CategoricalDtype(categories=['lo','hi'], ordered=True)
_vnc_status_cat =   pd.api.types.CategoricalDtype(categories=['intact','cut'], ordered=True)

_categories_list = [_outcomes_cat,_pyas_state_cat,_vnc_status_cat]

_force_clrs = [(.92, .6, .7),(1,.9,1)]


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

    def __init__(self,fn):
        self.topdir = 'D:\\Data'
        self.day,self.fly,self.cell = get_day_fly_cell(fn)
        self._dfc = '{}_F{}_C{}'.format(self.day,self.fly,self.cell)
        self.fn = get_file(fn)
        
        self.flycelldir = self.day + '_F' + str(self.fly) + '_C' + str(self.cell)
        self.path = os.path.join(self.topdir,self.day,self.flycelldir)

        self.df = None
        self._load_parquet()

        self._tfn_template = None
        self._trial_file_name_template()

        self.genotype = None
        self._get_genotype()
        
        self._excluded_df = None
        self.exclude_trials()


    @property
    def trial_list(self):
        if not 'Trial' in self.df.columns:
            self.get_trials()
        return self.df['Trial']
    

    @property
    def _outcomes_cat(self):
        return _outcomes_cat


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
            print(self._excluded_df)
            raise AttributeError('Handle the case of excluded files with index')
            self._excluded_df = pd.concat([self._excluded_df,to_exclude])
        else:
            print('Excluding trials:')
            print(self._excluded_df['Trial'].to_list())

        self.df = self.df[self.df['excluded'] == False]


    def get_trials(self):
        if 'Trial' in self.df.columns:
            return self.df.loc[['Trial','is_rest','is_probe','as_duration','as_outcome']]

        def create_trial(row):
            trial_number = row.name  # Use the index (trial_number)
            file_name = self._generate_filename(trial_number)
            tr = Trial(file_name)
            return {
                'Trial':    tr,
                'is_rest':  tr.is_rest,
                'is_probe':  tr.is_probe,
                'as_duration':  tr.as_duration,
                'as_outcome':  tr.as_outcome,
                }
        
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
        
        if df is None:
            df = self.df

        def get_probe_positions(row):
            trial = row.Trial
            pps = trial.probe_position - trial.params['probeZero']
            return {'probe_positions': pps,
                    'probe_min': pps.min(),  # Max flexion
                    'probe_max': pps.max(),  # Should be close to ProbeZero
                    'probe_zero': trial.params['probeZero'],
                    }
        
        new_cols = df.swifter.apply(get_probe_positions, axis=1, result_type='expand')
        if new_cols.index.equals(self.df.index):
            if (not 'probe_min' in self.df.columns) or (not 'probe_max' in self.df.columns) or (not 'probe_zero' in self.df.columns):
                self.df = self.df.join(new_cols[['probe_min','probe_max','probe_zero']])
                print('Probe positions found for all trials, indices are the same, joining dfs.')
        return new_cols

    
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
        
        first_category = categories.categories[0]  # 'intact'
        self.df[cat_name] = pd.Series([first_category] * len(self.df), dtype=categories)
        self.df.loc[index,cat_name] = category

        return index


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
        print(files_to_open)

        # Check if there are files to open
        if not files_to_open:
            print("No 'notes' files found in the directory.")
            return

        # Open each file in a new VS Code window
        pathtocode = 'C:\\Users\\tony\\AppData\\Local\\Programs\\Microsoft VS Code\\bin\\code.cmd'
        for file_path in files_to_open:
            subprocess.run([pathtocode, file_path], check=True)
            print(f"Opened file: {file_path}")


    # ---------------------------------------------------------
    # Plotting Methods
    # ---------------------------------------------------------
    def plot_outcomes(self,savefig=False,format='png'):
        # Convert categories to their integer codes (0 for 'Low', 1 for 'Medium', 2 for 'High')
        y_positions = self.df['as_outcome'].cat.codes
        x_positions = self.df.index

        # Plot each row as a vertical tick mark at its categorical position
        
        fig, ax = plt.subplots(figsize=(8,4))
        rec_min = self.df['as_outcome'].cat.categories.get_loc('no_as_no_mv')
        rec_max = self.df['as_outcome'].cat.categories.get_loc('probe')

        # Non rest trials
        T = self.df[~self.df['is_rest']]
        for ocb in T.op_cnd_blocks.unique():
            T_rows = T[T.op_cnd_blocks==ocb]
            
            pyasstate = pyas_state(T_rows)

            tgt_clr = _force_clrs[0 if pyasstate=='lo' else 1]

            tr_min = T_rows.index.min()
            tr_max = T_rows.index.max()
            rect = patches.Rectangle(
                    (tr_min, rec_min-0.5),        # Bottom-left corner of the rectangle
                    (tr_max - tr_min + 1),       # Width (covers the specified rows)
                    rec_max,                        # Height (covers all categories)
                    edgecolor=tgt_clr,
                    facecolor=tgt_clr,
                    alpha=1
                )
            ax.add_patch(rect)

        ax.scatter(x_positions, y_positions, marker='|', s=200, color='black')


        # Label the y-axis with the category names
        ax.set_yticks(range(len(self.df['as_outcome'].cat.categories)), self.df['as_outcome'].cat.categories)
        ax.invert_yaxis()

        ax.set_xlabel("Trial Index")
        ax.set_ylabel("Outcome")
        ax.set_title(f"{self._dfc} {self.genotype} outcomes")

        
        plt.show()
        if savefig:
            fig.savefig(f'./figpanels/{self._dfc}_{self._genotype}_as_outcomes.{format}',format=format)
    

    def plot_probe_distribution(self,binwidth=2,bin_min=None,bin_max=None,filter=None,savefig=False,format='png'):
        from collections import Counter

        probe_positions = self.probe_positions_df()
        if bin_min is None:
            bin_min = probe_positions.probe_min.min() # Max flexion
        if bin_max is None:
            bin_max = probe_positions.probe_max.max() # Should be ProbeZero
        probe_bins = np.arange(bin_min, bin_max, binwidth)

        ppi = probe_positions.index
        if not filter is None:
            for key in filter:
                probe_positions.loc[ppi,key] = self.df.loc[ppi,key]
                # print('Adding {}'.format(key))

            for key in filter:
                # print('Keeping rows with {}: {}'.format(key,filter[key]))
                # print(new_cols.columns)
                probe_positions = probe_positions.loc[probe_positions[key]==filter[key],:]

        print('Histogram for {} rows'.format(probe_positions.shape[0]))

        # Define a function to calculate the histogram
        def calculate_histogram(array, bins):
            counts, _ = np.histogram(array, bins=bins)
            return counts
        
        probe_positions['histogram'] = probe_positions['probe_positions'].apply(lambda arr: calculate_histogram(arr, probe_bins))
        summed_histogram = np.sum(np.vstack(probe_positions['histogram'].to_numpy()), axis=0)

        target_tuples = list(zip(self.df.pyasXPosition-self.df.probeZero, self.df.pyasWidth, self.df.pyasState))
        most_common_tuples = Counter(target_tuples).most_common(2)
        print(most_common_tuples)

        fig, ax = plt.subplots(figsize=(8, 6))
        for mct_item in most_common_tuples:
            mct = mct_item[0]
            tgt_clr = _force_clrs[0 if mct[2]=='lo' else 1]
            rect = patches.Rectangle(
                    (0, mct[0]),        # Bottom-left corner of the rectangle
                    (summed_histogram.max()),       # Width (covers the specified rows)
                    mct[1],                        # Height (covers all categories)
                    edgecolor=tgt_clr,
                    facecolor=tgt_clr,
                    alpha=1
                )
            ax.add_patch(rect)

        ax.step(summed_histogram, probe_bins[:-1], where='post', color='blue', label='Histogram')
        ax.set_title('Histogram of Probe Positions')
        ax.set_xlabel('Probe Position')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.5)
        plt.show()

        if savefig:
            fig.savefig(f'./figpanels/{self._dfc}_{self._genotype}_as_outcomes.{format}',format=format)


    def plot_hi_lo_probe_distribution(self,binwidth=2,bin_min=None,bin_max=None,savefig=False,format='png'):

        probe_positions = self.probe_positions_df()

        probe_positions = self.df['Trial'].swifter.apply(lambda trial: trial.probe_position - trial.params['ProbeZero'])
        if bin_min is None:
            bin_min = probe_positions.min() # Max flexion
        if bin_max is None:
            bin_max = probe_positions.max() # Should be ProbeZero
        probe_bins = np.arange(bin_min, bin_max, binwidth)
        # hist, bin_edges = np.histogram(probe_positions, bins=probe_bins)




        if savefig:
            fig.savefig(f'./figpanels/{self._dfc}_{self._genotype}_as_outcomes.{format}',format=format)


    def probe_position_heatmap(self):
        # first test if all of the stimulus duration is the same
        def trial_duration(row):
            return row.preDurInSec+row.stimDurInSec+ row.postDurInSec
        trial_dur = self.df.apply(trial_duration,axis=1)
        if not trial_dur.nunique() == 1:
            raise NotImplementedError('What if the trial duration is different?')

        probe_positions = self.probe_positions_df()
        
        return probe_positions

        


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

        self.df = self.df[self.df['samples'].apply(lambda x: x.size > 0)]


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
        self.genotype = str(gntyp[0][0])
        self._genotype = self.genotype.replace('.', '_')
        self._genotype = self._genotype.replace('>', '_')


    # ---------------------------------------------------------
    # Dunder Methods
    # ---------------------------------------------------------
    def __dir__(self):
        # Filter out attributes that start with an underscore
        return [attr for attr in super().__dir__() if not attr.startswith("_")]
    

    def __repr__(self):
        return('Day {}, F{}, C{}: {}'.format(self.day,self.fly,self.cell,self.fn))
    













    ## Potentially useful code
#     import importlib

# def find_variable_in_module(module_name, variable_name):
#     """
#     Finds a variable by name in the specified module.

#     Args:
#         module_name (str): The name of the module (e.g., 'table').
#         variable_name (str): The name of the variable to find.

#     Returns:
#         Any: The value of the variable if found, or None if not found.
#     """
#     try:
#         # Import the module dynamically
#         module = importlib.import_module(module_name)
        
#         # Check if the variable exists in the module's namespace
#         if hasattr(module, variable_name):
#             return getattr(module, variable_name)
#         else:
#             return None
#     except ModuleNotFoundError:
#         raise ImportError(f"Module '{module_name}' not found.")