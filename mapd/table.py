from scipy.io import loadmat
from .helpers import get_day_fly_cell, get_file
from .trial import Trial
from os.path import join
import pandas as pd
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
_outcomes_cat = pd.api.types.CategoricalDtype(categories=_outcomes_dict.keys(), ordered=True)

_state_cat = ['lo','hi']

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
        self.path = join(self.topdir,self.day,self.flycelldir)

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
            def create_trial(row):
                trial_number = row.name  # Use the index (trial_number)
                file_name = self._generate_filename(trial_number)
                return Trial(file_name)
            self.df['Trial'] = self.df.apply(create_trial,axis=1)

        return self.df['Trial']
    

    @property
    def _outcomes_cat(self):
        return _outcomes_cat


    def exclude_trials(self):
        if not 'Trial' in self.df.columns:
            self.trial_list
            print('Got trial list')

        tru_arr = np.array([[1]])
        self.df['excluded'] = self.df.Trial.apply(lambda tr: np.array_equal(tru_arr,tr.excluded))
        to_exclude = self.df[self.df['excluded'] == False]
        if self._excluded_df is None:
            self._excluded_df = to_exclude
            print('Excluding trials:')
            print(self._excluded_df['Trial'].to_list())
        elif not to_exclude.empty:
            raise AttributeError('Handle the case of excluded files with index')
            self._excluded_df = pd.concat([self._excluded_df,to_exclude])
        else:
            print('Excluding trials:')
            print(self._excluded_df['Trial'].to_list())

        self.df = self.df[self.df['excluded'] == False]


    def get_trial_metrics(self):
        if not 'Trial' in self.df.columns:
            self.trial_list
            print('Got trial list')

        # rest trials
        self.df['is_rest'] = self.df['Trial'].apply(lambda tr: tr.is_rest)

        # rest trials
        self.df['is_probe'] = self.df['Trial'].apply(lambda tr: tr.is_probe)

        # add as_duration 
        self.df['as_duration'] = self.df['Trial'].apply(lambda tr: tr.as_duration)

        # add ordered outcomes    
        self.df['as_outcome'] = self.df['Trial'].apply(lambda tr: tr.as_outcome).astype(_outcomes_cat)

        # numbering blocks where learning is engaged.
        self.df['op_cnd_blocks'] = (self.df['pyasState'] != self.df['pyasState'].shift(1)).cumsum()
        

    # ---------------------------------------------------------
    # Plotting Methods
    # ---------------------------------------------------------
    def plot_outcomes(self):
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


    # ---------------------------------------------------------
    # Helper (Internal) Methods
    # ---------------------------------------------------------
    def _load_parquet(self):
        self.df = pd.read_parquet(join(self.path,self.fn))

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
        matdata = loadmat(join(self.path,'Acquisition_{}_F{}_C{}.mat'.format(self.day,self.fly,self.cell)))
        gntyp = matdata['acqStruct']['flygenotype'].ravel()
        self.genotype = str(gntyp[0][0])


    # ---------------------------------------------------------
    # Dunder Methods
    # ---------------------------------------------------------
    def __dir__(self):
        # Filter out attributes that start with an underscore
        return [attr for attr in super().__dir__() if not attr.startswith("_")]
    

    def __repr__(self):
        return('Day {}, F{}, C{}: {}'.format(self.day,self.fly,self.cell,self.fn))