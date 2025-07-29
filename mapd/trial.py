from .helpers import get_day_fly_cell, get_file, default_data_directory
from os.path import join
import h5py
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from scipy.stats import mode

k_spring_constant = 0.0829 #uN/um

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
        self.topdir = default_data_directory(verbose=False)
        self.day,self.fly,self.cell = get_day_fly_cell(fn)
        self._dfc = '{}_F{}_C{}'.format(self.day,self.fly,self.cell)
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
        self._frame_length_mode = None # this is the most likely number of samples per Pyas frame

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
            self.create_probe_downsample_idx()
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
                raise ValueError('Run Table.exclude_trials: {}'.format(self.fn))
        return self._as_duration


    @property
    def as_outcome(self,rerun=True):
        if self.as_duration is None:
            self._as_outcome == None
            raise ValueError('Run Table.exclude_trials: {}'.format(self.fn))
            # return self._as_outcome

        if self._as_outcome is None:
            if not rerun:
                if 'current_as_outcome' in self.groups:
                    self._as_outcom = self.current_as_outcome.decode('utf-8')
                    return self._as_outcome
                else:
                    self._classify_as_outcome(rerun=rerun)
            else:
                self._classify_as_outcome(rerun=rerun)

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
                self._is_probe = (self.params['controlToggle'] == 0) 
            except KeyError:
                self._is_probe = False
        return self._is_probe


    def exclude(self,reason=None):
        if not self.excluded:
            print('excluding: {}'.format(self))
            with h5py.File(join(self.path,self.fn),'r+') as hdf5_file:
                flipped = 1-self.excluded
                hdf5_file['excluded'][...] = flipped
                setattr(self, 'excluded', flipped)
                print('self.excluded: {}'.format(self.excluded))
        if not reason is None:
            print('Reason for excluding is: {}'.format(reason))
            reason = 'Exclusion: {}'.format(reason)
            # with h5py.File(join(self.path,self.fn),'r+') as hdf5_file:
                # raise NotImplementedError('try adding tags to trials to provide exclusion reason')
                # self.tag(reason)
            print('Excluded for {}: {}'.format(reason, self))
        


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


    def create_probe_downsample_idx(self):
        arr = self.probe_position.ravel()
        change_points = np.where(np.diff(arr) != 0)[0] + 1  # take the first element of tuple from where, and add 1 to get the next change point

        sequence_lengths = np.diff(np.concatenate(([0], change_points, [len(arr)])))
        self._frame_length_mode = mode(sequence_lengths).mode
        if self._frame_length_mode > 500:
            print('Most common frame length is {} samples. Implement tagging reason'.format(self._frame_length_mode))
            self.exclude() # self.exclude(reason='pyas stopped working')

        time_zero_index = self.params['preDurInSec'] * self.params['sampratein']
        totalsamps = len(arr)

        # start the down sample vector at time_zero_index, go to total samps
        after_indices = np.arange(time_zero_index, totalsamps, self._frame_length_mode)
        # add the last index if it is not there (if DT doesn't fit)
        if not totalsamps-1 in after_indices:
            after_indices = np.append(after_indices,totalsamps-1)

        before_indices = np.arange(time_zero_index-self._frame_length_mode, -1, -self._frame_length_mode)[::-1]
        if not 0 in before_indices:
            before_indices = np.insert(before_indices, 0, 0, axis=0)

        indices = np.concatenate((before_indices,after_indices)).astype(int)

        assert np.all(np.diff(indices) > 0)
        self._downsample_probe = indices


    def _classify_as_outcome(self,rerun=True):

        if (self._as_outcome is None) or rerun:
            if self.is_rest:
                self._as_outcome = 'rest'
            
            elif self.is_probe:
                self._as_outcome = 'probe'

            elif self._as_duration==0:
                self._as_outcome = 'no_as_no_mv'
                
                # but is the probe in the target the entire time?
                probe_position = self.probe_position.ravel()

                target_min = self.params['pyasXPosition']
                target_max = self.params['pyasXPosition'] + self.params['pyasWidth']
                if any(probe_position<target_min) | any(probe_position>target_max):
                    # print('Trial {}: probe leaves target'.format(self))
                    self._as_outcome = 'no_as_mv'

            elif self.as_duration < self.params['stimDurInSec']:
                self._as_outcome = 'as_off'
            else:
                probe_position = self.probe_position[self.time>0].ravel()
                target_min = self.params['pyasXPosition']
                target_max = self.params['pyasXPosition'] + self.params['pyasWidth']
                if any((probe_position>target_min) & (probe_position<target_max)):
                    self._as_outcome = 'as_off_late'
                elif all(probe_position<target_min):
                    self._as_outcome = 'timeout_fail'
                    # if self._post_stim_var > self._mv_thresh:
                    #     self._as_outcome = 'timeout_fail'
                else: 
                    self._as_outcome = 'timeout'
            self._write_string_to_hdf5('current_as_outcome',self._as_outcome)
        else:
            raise KeyError('Is the current as outcome correct?')

    ## Compute functions, using the downsampled probe
    def probe_velocity(self):
        x = self.probe_position[self.downsample_probe].squeeze()
        t = self.time[self.downsample_probe]
        # assert x.shape == t.shape
        # print(x.shape)
        # print(t.shape)
        v = np.gradient(x, t)
        # assert x.shape == t.shape
        return v

    
    def probe_acceleration(self):
        """A measure of effort in motor control"""
        ds_v = self.probe_velocity()
        ds_time = self.time[self.downsample_probe]
        a = np.gradient(ds_v, ds_time)
        return a


    def probe_mean_velocity(self,):
        """Integral of abs(velocity)"""
        vigor = np.mean(np.abs(self.probe_velocity))
        return vigor
    

    def probe_rms_velocity(self):
        """Weights faster movements more"""
        v = self.probe_velocity()
        rms_vigor = np.sqrt(np.mean(v**2))
        return rms_vigor


    def probe_jerk_energy(self):
        """A measure of effort in motor control"""
        ds_time = self.time[self.downsample_probe]
        jerk = np.gradient(self.probe_acceleration(), ds_time)
        jerk_energy = np.sum(jerk**2) * np.diff(ds_time[2:3])
        return jerk_energy


    def probe_power(self):
        x = self.probe_position[self.downsample_probe].squeeze()
        v = self.probe_velocity()
        power = -k_spring_constant * x * v
        return power


    def probe_work(self):
        """Work Done Against the Spring"""
        t = self.time[self.downsample_probe]
        power = self.probe_power()
        work = np.trapezoid(power,t)
        return work


    def probe_holding_cost(self):
        """Holding Cost (integrated potential energy)"""
        t = self.time[self.downsample_probe]
        x = self.probe_position[self.downsample_probe].squeeze()
        U = 0.5 * k_spring_constant * x**2
        holding_cost = np.trapezoid(U, t)
        return holding_cost


    def probe_positive_effort(self):
        power = self.probe_power()
        t = self.time[self.downsample_probe]
        effort = np.trapezoid(np.clip(power, a_min=0, a_max=None), t)
        return effort
    

    def probe_effort(self, alpha=1e-6, beta = 1e-2):
        """Effort Cost Function, symetric, Not in use"""
        x = self.probe_position[self.downsample_probe].squeeze()
        t = self.time[self.downsample_probe]
        v = self.probe_velocity()
        effort = np.trapezoid(alpha * v**2 + beta * x**2, t)
        return effort


    ## plotting functions
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


    def plot_probe_groups(self,use_full_time=True,use_full_y = False,from_zero = False,savefig=False,format=None):           # Additional plot for probe_position
        probe_position = getattr(self,'probe_position',None)
        arduino_output = getattr(self,'arduino_output',None)
        
        target_y = self.params['pyasXPosition']
        target_width = self.params['pyasWidth']
        probe_zero = self.params['probeZero']
        if from_zero:
            target_y = target_y-probe_zero
            probe_position = probe_position - probe_zero
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
        if not from_zero:
            if not use_full_y:
                plt.ylim([self.params['probeZero'] - 500, self.params['probeZero'] + 20])
            else:
                plt.ylim([0, 1280])
            plt.axhline(y=self.params['probeZero'], xmin=0, xmax=1, color=(0.5,0.5,0.5), linestyle='--', label='probe_zero')

        else:
            if not use_full_y:
                plt.ylim([0 - 500, 0 + 20])
            else:
                plt.ylim([0, 1280]-self.params['probeZero'])
            plt.axhline(y=0, xmin=0, xmax=1, color=(0.5,0.5,0.5), linestyle='--', label='probe_zero')

        # plt.grid(True)
        plt.show()

        if savefig or (not format is None):
            format = format or 'png'
            plt.savefig(f'./figpanels/{self._dfc}_Trial_{self.params['trial']}_probe_plot.{format}',format=format)


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

    def _write_string_to_hdf5(self, name, data):
        """Wrapper function to write data to HDF5 file."""
        dt = h5py.string_dtype(encoding='utf-8')

        encoded = data.encode('utf-16-le')
        data_array = np.frombuffer(encoded, dtype=np.uint16)
        dt = 'uint16'

        need_to_set_attrs = False
        with h5py.File(join(self.path,self.fn),'r+') as hdf5_file:
            if name in hdf5_file:
                # Overwrite contents of existing dataset
                hdf5_file[name][...] = data_array
            else:
                # Create new dataset
                dset = hdf5_file.create_dataset(name, data=data_array, dtype=dt)                
                ascii_dtype = h5py.string_dtype(encoding='ascii', length=4)  # 'char' is 4 letters
                dset.attrs.create("MATLAB_class", 'char', dtype=ascii_dtype)
                dset.attrs.create("MATLAB_int_decode", np.uint8(2))

    # ---------------------------------------------------------
    # Dunder Methods
    # ---------------------------------------------------------
    def __getattr__(self, name):

        # if name in ['file_path', 'params', '_refs', 'groups']:
        #     return super().__getattribute__(name)
        
        """Lazy load the attribute from the HDF5 file."""
        with h5py.File(self.file_path, 'r') as file:
            if name == 'params' and self.params is not None:
                return self.params
            
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
                else:
                    raise TypeError(f"Unsupported type for attribute '{name}': {type(group)}")
            
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