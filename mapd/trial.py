from .helpers import get_day_fly_cell, get_file
from os.path import join
import h5py
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

class Trial:
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
        self.fn = get_file(fn)
        
        self.flycelldir = self.day + '_F' + str(self.fly) + '_C' + str(self.cell)
        self.path = join(self.topdir,self.day,self.flycelldir)
        self.file_path = join(self.path,self.fn)
        # print('Day {}, F{}, C{}: {}'.format(self.day,self.fly,self.cell,self.fn))
        
        self._load_params()

        with h5py.File(self.file_path,'r') as hdf5_file:
            self.groups = [key for key in hdf5_file.keys() if key != '#refs#']
            if '#refs#' in hdf5_file:
                self._refs = self._load_refs(hdf5_file['#refs#'])

        self._time = None
        self._trialtime = None
        self._downsample_probe = None

        self._probegroups = ['probe_position']
        self._ephysgroups = ['voltage_1','voltage_2','current_extEMG','current_1','current_2']
        self._tgt_clrs = [(0.8,0.8,0.8),(0.6,0.6,1)]

        self._as_duration = None
        self._as_outcome = None
        self._is_rest = None
        self._is_probe = None

    @property
    def time(self):
        if self._time is None:
            self.create_time()
        return self._time


    @property
    def trialtime(self):
        if self._trialtime is None:
            self.create_time()
        return self._trialtime

    @property
    def downsample_probe(self):
        if self._downsample_probe is None:
            self.create_time()
        return self._downsample_probe

    @property
    def as_duration(self):
        if self._as_duration is None:
            # Calculate duration where the array is 1
            sample_rate = self.params['sampratein'] if self.params else 1  # Default to 1 if params not available
            try:
                self._as_duration = np.sum(self.arduino_output == 1) / sample_rate
            except AttributeError as e:
                print("AttributeError:", e)
                self.exclude()
                self._as_duration = None
                raise ValueError('Run Table.exclude_trials: {}'.format(self))
        return self._as_duration


    @property
    def as_outcome(self):
        if self.as_duration is None:
            self._as_outcome == None
            raise ValueError('Run Table.exclude_trials: {}'.format(self))
            # return self._as_outcome
        if self._as_outcome is None:
            self._classify_as_outcome()
        return self._as_outcome
    

    @property
    def is_rest(self):
        if self._is_rest is None:
            self._is_rest = (self.params['ndf'] == 0) 
        return self._is_rest


    @property
    def is_probe(self):
        if self._is_probe is None:
            try:
                self._is_probe = (self.params['probeToggle'] == 0) 
            except KeyError:
                self._is_probe = False
        return self._is_probe


    def exclude(self):
        if not self.excluded:
            print('excluding: {}'.format(self))
            with h5py.File(join(self.path,self.fn),'r+') as hdf5_file:
                flipped = 1-self.excluded
                hdf5_file['excluded'][...] = flipped
                setattr(self, 'excluded', flipped)
            print('self.excluded: {}'.format(self.excluded))

    def include(self):
        if self.excluded:
            print('including: {}'.format(self))
            with h5py.File(join(self.path,self.fn),'r+') as hdf5_file:
                flipped = 1-self.excluded
                hdf5_file['excluded'][...] = flipped
                setattr(self, 'excluded', flipped)
            print('self.excluded: {}'.format(self.excluded))


    def create_time(self):
        pre_dur = self.params['preDurInSec']
        samprate = self.params['sampratein']
        samples = self.params['samples']
        trial_dur =  self.params['preDurInSec'] + self.params['stimDurInSec'] + self.params['postDurInSec'] 
        trialsamps = trial_dur*samprate

        total_dur = samples / samprate
        self._time = np.linspace(-pre_dur, total_dur - pre_dur, np.int64(samples))
        self._trialtime = np.linspace(-pre_dur, trial_dur - pre_dur, np.int64(trialsamps))
        self.create_probe_ds_time()

    def create_probe_ds_time(self):
        from scipy.stats import mode

        arr = self.probe_position.ravel()
        change_points = np.where(np.diff(arr) != 0)[0] + 1

        # Calculate lengths of sequences
        sequence_lengths = np.diff(np.concatenate(([0], change_points, [len(arr)])))
        sequence_mode = mode(sequence_lengths).mode

        time_zero_index = self.params['preDurInSec'] * self.params['sampratein']
        trialsamps = len(self._trialtime)

        indices = np.arange(time_zero_index, trialsamps, sequence_mode)
        indices = np.concatenate((np.arange(time_zero_index, -1, -sequence_mode)[::-1], indices)).astype(int)
        self._downsample_probe = indices


    def _classify_as_outcome(self):
        # outcomes_dict = {
        #     'no_as_no_mv': 'no aversive stimulus, no movement',
        #     'no_as_mv': 'no aversive stimulus, probe moves',
        #     'as_off': 'fly turns off aversive stimulus during trial',
        #     'as_off_late': 'fly turns off aversive stimulus in intertrial period',
        #     'timeout': 'aversive stimulus never turned off, no movement',
        #     'timeout_fail': 'aversive stimulus never turned off, even though there movement',
        #     'probe': 'probe_trial',
        #     'rest': 'rest trial',
        # }
        # outcomes_cat = pd.api.types.CategoricalDtype(categories=outcomes_dict.keys(), ordered=True)

        if self.is_rest:
            self._as_outcome = 'rest'
        
        elif self.is_probe:
            self._as_outcome = 'probe'

        elif self._as_duration==0:
            self._as_outcome = 'no_as_no_mv'
            # if self._post_stim_var > self._mv_thresh:
            #     self._as_outcome = 'no_as_mv'
        elif self.as_duration < self.params['samples']/self.params['sampratein'] - self.params['preDurInSec']:
            self._as_outcome = 'as_off_late'
            if self._as_duration < self.params['stimDurInSec']:
                self._as_outcome = 'as_off'
        
        else:
            self._as_outcome = 'timeout'
            # if self._post_stim_var > self._mv_thresh:
            #     self._as_outcome = 'timeout_fail'



    def plot_group(self, group_name, use_full_time=True):
        """Plot the data from a specified group over self.time or self.trialtime.

        Args:
            group_name (str): Name of the group or dataset to plot.
            use_full_time (bool): Whether to use self.time (True) or self.trialtime (False).
        """

        if group_name in self._probegroups:
            self.plot_probe_groups(use_full_time)
        elif group_name in self._ephysgroups:
            self._plot_ephys_groups(group_name,use_full_time)
        else:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{group_name}'")


    def plot_probe_groups(self,use_full_time=True):           # Additional plot for probe_position
        probe_position = getattr(self,'probe_position',None)
        arduino_output = getattr(self,'arduino_output',None)
        
        target_y = self.params['pyasXPosition']
        target_width = self.params['pyasWidth']
        if use_full_time:
            time_array = self.time 
        else:
            time_array = self.trialtime
            probe_position = probe_position[0:len(time_array)]
            arduino_output = arduino_output[0:len(time_array)]

        tgt_clr = self._tgt_clrs[int(self.params['blueToggle'])]
        
        arduino_output = -arduino_output*target_width + target_y + target_width

        # Plot
        plt.figure()
        plt.gca().add_patch(plt.Rectangle(
            (time_array[0], target_y),  # Bottom-left corner
            width=max(time_array) - min(time_array),  # Full time range
            height=target_width,  # Width of the target
            color=tgt_clr, alpha=0.3, label='Target'))
        plt.plot(time_array, arduino_output,color=(0.7,0.7,0.7),label='arduino_output')

        plt.plot(time_array, probe_position)
        plt.title(f"tr#={int(self.params['trial'])} probe_position: '{self.as_outcome}'")
        plt.xlabel("Time (s)")
        plt.ylabel("um")
        plt.xlim(time_array[0], time_array[-1])
        plt.ylim([self.params['probeZero'] - 500, self.params['probeZero'] + 20])
        plt.axhline(y=self.params['probeZero'], xmin=0, xmax=1, color=(0.5,0.5,0.5), linestyle='--', label='probe_zero')
        # plt.grid(True)
        plt.show()

    def _plot_ephys_groups(self,group_name,use_full_time=True):           # Additional plot for probe_position
        data = getattr(self,group_name,None)
        target_y = self.params['pyasXPosition']
        target_width = self.params['pyasWidth']
        time_array = self.time if use_full_time else self.trialtime
        
        # Plot the data

        plt.figure()
        plt.plot(time_array, data)
        plt.title(f"Plot of {group_name}")
        plt.xlabel("Time (s)")
        plt.ylabel("Value")
        plt.grid(True)
        plt.show()

        plt.legend()

        plt.show()


    def _load_refs(self, refs_group):
        """Load references from the '#refs#' group."""
        refs = {}
        for ref_name in refs_group:
            ref = refs_group[ref_name]
            if isinstance(ref, h5py.Reference):
                refs[ref_name] = ref
        return refs
    

    def _load_params(self):
        
        keys_as_strings = ['protocol','pyasState']

        def extract_hdf5_group_as_dict(group):
            """
            Converts an HDF5 group into a dictionary by inspecting its datasets and extracting their values.
            
            Parameters:
            group (h5py.Group): The HDF5 group to process.

            Returns:
            dict: A dictionary containing the keys and extracted values from the group.
            """
            # Convert arrays to lists
            result = {}

            for key in group.keys():
                item = group[key]

                # If the item is a dataset, extract its data
                if isinstance(item, h5py.Dataset):
                    shape = item.shape
                    dtype = item.dtype
                    # print('{} is {}'.format(key,dtype))

                    # Load the dataset
                    data = item[()].ravel()  # This reads the data into memory
                    # print('shape is {}'.format(data.shape))

                    if key in keys_as_strings:
                        # Convert ints to chars and join them
                        result[key] = ''.join(chr(v) for v in data)
            
                    # Convert scalars
                    elif shape == ():
                        # print('converting scalars')
                        result[key] = data.item() if hasattr(data, "item") else data
                    # Convert arrays
                    else:
                        # print('converting array')
                        if hasattr(data, "tolist"):
                            if data.shape==(1,):
                                # print('to scalar')
                                result[key] = float(data[0])
                            else: 
                                # print('to list')
                                result[key] = data.tolist()
                        else: 
                            result[key] = data

                # If the item is another group, recursively extract its contents
                elif isinstance(item, h5py.Group):
                    result[key] = extract_hdf5_group_as_dict(item)

            return result   

        with h5py.File(join(self.path,self.fn),'r') as hdf5_file:
            params = hdf5_file['params']
            self.params = extract_hdf5_group_as_dict(params)
    

    def _lazy_load_group(self, group):
        """Create a proxy object to lazily load an HDF5 group."""
        group_obj = type(group.name, (object,), {'_group': group, '_loaded_attrs': {}})()
        return group_obj


    def __getattr__(self, name):

        # if name in ['file_path', 'params', '_refs', 'groups']:
        #     return super().__getattribute__(name)
        
        """Lazy load the attribute from the HDF5 file."""
        with h5py.File(self.file_path, 'r') as file:
            if name in file:
                # print(name)
                group = file[name]
                if isinstance(group, h5py.Group):
                    group_obj = self._lazy_load_group(group)
                    setattr(self, name, group_obj)
                    return group_obj
                elif isinstance(group, h5py.Dataset):
                    dataset_value = group[()]
                    setattr(self, name, dataset_value)
                    return dataset_value
            elif name == 'params' and self.params is not None:
                return self.params

            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __dir__(self):
        # Filter out attributes that start with an underscore
        return [attr for attr in super().__dir__() if not attr.startswith("_")]    

    def __repr__(self):
        return "Trial(trial={t}, {d}_F{f}_C{c}, dT={dt}, ex={ex})".format(t=int(self.params['trial']),d=self.day,f=self.fly,c=self.cell,dt=self.params['samples']/self.params['sampratein'],ex=np.array_equal(self.excluded, np.array([[1]])))
        # print('Day {}, F{}, C{}: {}'.format(self.day,self.fly,self.cell,self.fn))


# Usage example
if __name__ == "__main__":
    trial = Trial('d:\\Data\\241205\\241205_F2_C1\\LEDFlashTriggerPiezoControl_Raw_241205_F2_C1_243.mat')
    print(trial.params)  # Access the params dictionary
    if hasattr(trial, 'some_group'):
        print(trial.some_group.some_dataset)  # Access dynamically created attributes

    trial.params
    trial.groups