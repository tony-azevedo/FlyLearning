from .helpers import get_day_fly_cell, get_file, default_data_directory
from os.path import join
import h5py
import hdf5storage as h5s
import numpy as np
import os, shutil, time

# from matplotlib import pyplot as plt
import pandas as pd
from scipy.stats import mode

from functools import cached_property
from typing import Optional

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.patches as patches

import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)  # reset to defaults
mpl.rcParams['pdf.fonttype'] = 42         # embed fonts as text, not paths
mpl.rcParams['svg.fonttype'] = 'none'     # keep text editable in SVG
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.size'] = 11

TRIAL_METADATA_GROUP = 'meta'

import mapd.sentinels as s
from .kinematics import (
    k_spring_constant,
    velocity, acceleration, mean_velocity, rms_velocity, jerk_energy,
    power, work, holding_cost, positive_effort, effort,
    detect_movement_bouts, detect_bouts_across_trials,
    STATE_REST, STATE_DRIFT, STATE_MOVE, _STATE_LABEL, _STATE_COLORS,
)


def _decode_h5_string(arr):
    if isinstance(arr, np.ndarray):
        if np.issubdtype(arr.dtype, np.uint16):
            # Correct: keep both bytes of each code unit
            return arr.tobytes().decode('utf-16-le')
        if np.issubdtype(arr.dtype, np.uint8):
            return arr.tobytes().decode('utf-8')
    if isinstance(arr, (bytes, bytearray)):
        # If the dataset was stored as a byte string
        try:
            return arr.decode('utf-8')
        except UnicodeDecodeError:
            return arr.decode('utf-16-le')
    return str(arr)


def _decode_if_string(dset, val):
    """
    Try to coerce dataset contents to Python str when appropriate.
    - HDF5 fixed/vlen strings: use asstr()
    - MATLAB-style UTF-16LE stored as uint16 arrays: use your _decode_h5_string
    """
    # HDF5 strings (fixed-length 'S', unicode 'U', or vlen 'O' with base=str)
    try:
        if dset.dtype.kind in ('S', 'U') or h5py.check_dtype(vlen=dset.dtype) is str:
            return dset.asstr()[()]  # scalar or array of str
    except Exception:
        pass

    # MATLAB v7.3 char arrays often come as uint16
    if np.issubdtype(dset.dtype, np.uint16) and isinstance(val, np.ndarray):
        # Delegate to your existing decoder (expects the raw array)
        return _decode_h5_string(val)

    # bytes -> try utf-8, fallback to original bytes
    if isinstance(val, (bytes, bytearray)):
        try:
            return val.decode('utf-8')
        except Exception:
            return val

    return val


def _is_matlab_char_dset(dset: h5py.Dataset) -> bool:
    """Heuristically detect MATLAB char dataset."""
    cls = dset.attrs.get('MATLAB_class', None)
    if isinstance(cls, (bytes, np.bytes_)):
        cls = cls.decode('ascii', 'ignore')
    if cls == 'char':
        return True
    # Common fallback: char is stored as uint16 with MATLAB_int_decode
    return (dset.dtype == np.uint16) and ('MATLAB_int_decode' in dset.attrs)


def _decode_matlab_char(arr: np.ndarray) -> str:
    """uint16 code units → Python str (UTF-16LE)."""
    arr = np.asarray(arr, dtype=np.uint16)
    return arr.ravel().tobytes().decode('utf-16le')


def _char_shape_ok(dset) -> bool:
    """Accept common MATLAB-compatible char shapes."""
    if dset.ndim == 1:
        # e.g., (N,) — MATLAB tolerates this just fine
        return True
    if dset.ndim == 2:
        r, c = dset.shape
        return r == 1 or c == 1  # 1xN or Nx1
    return False  # 0-D or >2-D => not OK


def _as_python_scalar(x):
    """Return a Python scalar if x is a 0-D numpy scalar or a 1-element array; otherwise x."""
    if isinstance(x, np.ndarray):
        if x.size == 1:
            return x.reshape(()).item()
        return x
    return x.item() if hasattr(x, "item") else x


def _set_ascii_attr(h5obj, key: str, value: str):
    h5obj.attrs.create(key, value, dtype=h5py.string_dtype(encoding='ascii'))


# Signal primitives and bout detection are imported from mapd.kinematics above.



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
        
        self._time = None
        self._trialtime = None
        self._downsample_probe = None
        self.frame_length_mode = None # this is the most likely number of samples per Pyas frame

        self._probegroups = ['probe_position']
        self._ephysgroups = ['voltage_1','voltage_2','current_extEMG','current_1','current_2']
        self._tgt_clrs = [(0.8,0.8,0.8),(0.6,0.6,1)]

        self._as_duration = None
        self._as_outcome = None        
        # self._is_rest = None
        # self._is_probe = None
        # self._meta_keys = None

        self._probeZero = None
        self._pyasXPosition = None
        self._pyasWidth = None
        self._pyasState = None

        self.normalize_group_for_matlab(dry_run=False,add_h5path_attr=True)
        self._load_params()
        self._ensure_target_params()

        with h5py.File(self.file_path,'r') as hdf5_file:
            self.groups = [key for key in hdf5_file.keys() if key != '#refs#']
            if '#refs#' in hdf5_file:
                self._refs = self._load_refs(hdf5_file['#refs#'])

        self.total_duration = self.params['samples'] / self.params['sampratein']


    @cached_property
    def time(self):
        if self._time is None:
            self.create_time()
        return self._time


    @cached_property
    def trialtime(self):
        if self._trialtime is None:
            self.create_time()
        return self._trialtime


    @cached_property
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


    @property  # not cached, have to be able to change this.
    def as_outcome(self):
        return self.get_as_outcome(rerun=False)
    

    def get_as_outcome(self, rerun=False,verbose = False):
        if rerun:
            # print(f'Reclassifying trial outcomes: {rerun}')
            self._classify_as_outcome(rerun=rerun,verbose=verbose)

        elif self._as_outcome is None or not rerun:  #or self._as_outcome is s.MISSING
            try:
                # val = self._read_string_from_meta('as_outcome')
                val = self._read_value_from_meta('as_outcome')

                if val is s.MISSING:
                     self._classify_as_outcome(rerun=rerun)

                if (val is not None) and (not val is s.MISSING):
                    self._as_outcome = val
            
                if 'current_as_outcome' in self.groups:
                    self._remove_legacy_as_outcome_group()

            except AttributeError as e:

                if self._as_outcome is None:
                    self._classify_as_outcome(rerun=rerun)
        
        # if self._as_outcome is s.MISSING:
        #     print('Missing a loop')
        return self._as_outcome


    @cached_property
    def is_rest(self):
        return self.params['ndf'] == 0


    @cached_property
    def is_probe(self):
        try:
            _is_probe = (self.params['controlToggle'] == 0) 
        except KeyError:
            _is_probe = False
        return _is_probe


    @property
    def pyasState(self):
        px = self.params.get('pyasState',None)
        if px is None:
            px = self.refresh_meta('pyasState')
        
        if not px is None:
            return px
        else:
            return np.nan


    @cached_property
    def pyasXPosition(self):
        px = self.params.get('pyasXPosition',None)
        if px is None:
            px = self.refresh_meta('pyasXPosition')

        if not px is None:
            return px
        else:
            raise KeyError('PyasXPosition cannot be found')


    @cached_property
    def pyasWidth(self):
        pw = self.params.get('pyasWidth',None)
        if pw is None:
            pw = self.refresh_meta('pyasWidth')

        if not pw is None:
            return pw
        else:
            raise KeyError('pyasWidth cannot be found')


    @property
    def probeZero(self):
        ''''Need these property level definitions because the trial may not have this meta information
            Can't be cached because I need to change None values
        '''
        if not self._probeZero is None:
            return self._probeZero
        
        if 'probeZero' in self.params.keys():
            self._probeZero = self.params['probeZero']
            return self._probeZero
        
        elif 'probeZero' in self.meta_keys:
            self._probeZero = self._read_value_from_meta('probeZero')
            return self._probeZero
        
        else:
            # print('ProbeZero needs to be calculated at the Table level')
            return 0


    @cached_property    
    def meta_keys(self):
        # if self._meta_keys is None:
        #     self._meta_keys = self.list_meta_keys()
        return self.list_meta_keys()


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
            self.write_string_if_changed('exclude_reason',reason)
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

        self._total_duration = samples / samprate
        self._time = np.linspace(-pre_dur, self._total_duration - pre_dur, np.int64(samples))
        self._trialtime = np.linspace(-pre_dur, trial_dur - pre_dur, np.int64(trialsamps))


    def create_probe_downsample_idx(self):
        arr = self.probe_position.ravel()
        change_points = np.where(np.diff(arr) != 0)[0] + 1  # take the first element of tuple from where, and add 1 to get the next change point

        sequence_lengths = np.diff(np.concatenate(([0], change_points, [len(arr)])))
        self.frame_length_mode = mode(sequence_lengths).mode
        if self.frame_length_mode > 500:
            print('Most common frame length is {} samples. Pyas probably stopped working.'.format(self.frame_length_mode))
            self.exclude(reason='Pyas stopped working')

        time_zero_index = self.params['preDurInSec'] * self.params['sampratein']
        totalsamps = len(arr)

        # start the down sample vector at time_zero_index, go to total samps
        after_indices = np.arange(time_zero_index, totalsamps, self.frame_length_mode)
        # add the last index if it is not there (if DT doesn't fit)
        if not totalsamps-1 in after_indices:
            after_indices = np.append(after_indices,totalsamps-1)

        before_indices = np.arange(time_zero_index-self.frame_length_mode, -1, -self.frame_length_mode)[::-1]
        if not 0 in before_indices:
            before_indices = np.insert(before_indices, 0, 0, axis=0)

        indices = np.concatenate((before_indices,after_indices)).astype(int)

        assert np.all(np.diff(indices) > 0)
        self._downsample_probe = indices


    def _classify_as_outcome(self,rerun=False, verbose=False):
        if (self._as_outcome is None) or rerun:
            # if self.is_rest:
            #     self._as_outcome = 'rest'
            
            # elif self.is_probe:
            #     self._as_outcome = 'probe'
            
            probe_position = self.probe_position.ravel()
            t = self.time
            target_min = self.pyasXPosition
            target_max = self.pyasXPosition + self.pyasWidth

            # For cases when the fly is not under control
            if self.is_probe:
                if self._as_duration==0:
                    # but is the probe in the target the entire time?
                    if any(probe_position[t>0]<target_min) | any(probe_position[t>0]>target_max):
                        # print('Trial {}: probe leaves target'.format(self))
                        self._as_outcome = 'no_as_mv'
                    else:
                        self._as_outcome = 'no_as_no_mv'
                elif any(probe_position[t>0]>target_min) & any(probe_position[t>0]<target_max):
                    self._as_outcome = 'as_off'
                else:
                    self._as_outcome = 'timeout'

            
            else:
                if self._as_duration==0:
                    # but is the probe in the target the entire time?
                    if any(probe_position[t>0]<target_min) | any(probe_position[t>0]>target_max):
                        # print('Trial {}: probe leaves target'.format(self))
                        self._as_outcome = 'no_as_mv'
                    else:
                        self._as_outcome = 'no_as_no_mv'

                elif self.as_duration < self.params['stimDurInSec']:
                    self._as_outcome = 'as_off'

                else:
                    if any((probe_position>target_min) & (probe_position<target_max)):
                        self._as_outcome = 'as_off_late'
                    elif all(probe_position<target_min):
                        self._as_outcome = 'timeout_fail'
                        # if self._post_stim_var > self._mv_thresh:
                        #     self._as_outcome = 'timeout_fail'
                    else: 
                        self._as_outcome = 'timeout'
            self.write_string_if_changed('as_outcome',self._as_outcome)
            if verbose:
                print(f'{self._as_outcome}: {self.__repr__()}')
            return self._as_outcome
        elif (self._as_outcome is None) and not rerun:
            raise KeyError('Is the current as_outcome correct?')
        else: # both exist and are True
            return self._as_outcome


    def on_target(self, probe_position=None):
        """
        How much of the duration is the probe on the target.
        This is a helper function to quantify how much the probe is on the target during the trial.
        Args:
            probe_position (np.ndarray): The probe position data. If None, uses self.probe_position.
        
        Returns:
            float: The fraction of time the probe is on the target.
        """
        if probe_position is None:
            probe_position = self.probe_position.ravel()
        
        target_min = self.pyasXPosition
        target_max = self.pyasXPosition + self.pyasWidth
        
        return np.sum((probe_position >= target_min) & (probe_position <= target_max))/ len(probe_position)

    
    def time_on_target(self, probe_position=None):
        """
        How long is the probe on the target.
        This is a helper function to quantify how much the probe is on the target during the trial.
        Args:
            probe_position (np.ndarray): The probe position data. If None, uses self.probe_position.
        
        Returns:
            float: The fraction of time the probe is on the target.
        """
        on_target_frac = self.on_target()
        
        return on_target_frac * np.diff([self.time[0],self.time[-1]])[0]
    

    ## Compute functions, using the downsampled probe
    def probe_velocity(self,debug = False):
        x = -(self.probe_position[self.downsample_probe].squeeze() - self.probeZero)
        t = self.time[self.downsample_probe].squeeze()
        return velocity(t,x)

    
    def probe_acceleration(self,debug = False):
        """A measure of effort in motor control"""
        x = -(self.probe_position[self.downsample_probe].squeeze() - self.probeZero)
        t = self.time[self.downsample_probe].squeeze()
        return acceleration(t,x)


    # def probe_mean_velocity(self,debug = False):
    #     """Integral of abs(velocity)"""
    #     x = self.probe_position[self.downsample_probe].squeeze()
    #     t = self.time[self.downsample_probe].squeeze()
    #     return mean_velocity(t,x)
    

    def probe_rms_velocity(self,debug=False):
        """Weights faster movements more"""
        x = -(self.probe_position[self.downsample_probe].squeeze() - self.probeZero)
        t = self.time[self.downsample_probe].squeeze()
        if debug:
            print(f"Trial {self.params['trial']}")
        return rms_velocity(t,x)


    # def probe_jerk_energy(self,debug = False):
    #     """A measure of effort in motor control"""
    #     x = -(self.probe_position[self.downsample_probe].squeeze() - self.probeZero)
    #     t = self.time[self.downsample_probe].squeeze()
    #     return jerk_energy(t,x)


    def probe_power(self,debug = False):
        x = -(self.probe_position[self.downsample_probe].squeeze() - self.probeZero)
        t = self.time[self.downsample_probe].squeeze()
        return power(t,x)


    def probe_work(self,debug = False):       
        """Work Done Against the Spring"""
        x = -(self.probe_position[self.downsample_probe].squeeze() - self.probeZero)
        t = self.time[self.downsample_probe].squeeze()
        return work(t,x)


    def probe_holding_cost(self,debug = False):
        """Holding Cost (integrated potential energy)"""
        x = -(self.probe_position[self.downsample_probe].squeeze() - self.probeZero)
        t = self.time[self.downsample_probe].squeeze()
        return holding_cost(t,x)

    
    def probe_positive_effort(self,debug = False):
        '''Assumes p is never negative'''
        t = self.time[self.downsample_probe].squeeze()
        x = -(self.probe_position[self.downsample_probe].squeeze() - self.probeZero)
        if debug:
            print(f"Trial {self.params['trial']}")
        return positive_effort(t,x)


    def probe_effort(self,debug = False, alpha=1e-6, beta = 1e-2):
        """Effort Cost Function, symetric, Not in use"""
        x = -(self.probe_position[self.downsample_probe].squeeze() - self.probeZero)
        t = self.time[self.downsample_probe].squeeze()
        return effort(t,x)


    def detect_bouts(self, start_time=None, v_th=30.0, v_rest=5.0, Dt=0.5, fallback='none', smooth_window=0.05):
        """
        Detect movement bouts in this trial's downsampled probe trace.

        Thin wrapper around ``mapd.kinematics.detect_movement_bouts``.
        x is sign-corrected: positive = toward target.

        Parameters
        ----------
        start_time : float or None — search only from this time onward
                     (e.g. self.as_duration to find post-AS bouts)
        v_th       : MOVE threshold (um/s), default 30
        v_rest     : REST threshold (um/s), default 5
        Dt         : minimum REST duration (s), default 0.5
        fallback   : 'none' | 'drift' | 'end'

        Returns
        -------
        bouts, states, t, x  — see detect_movement_bouts for full spec
        """
        t = self.time[self.downsample_probe].squeeze()
        x = -(self.probe_position[self.downsample_probe].squeeze() - self.probeZero)
        return detect_movement_bouts(t, x, start_time=start_time,
                                     v_th=v_th, v_rest=v_rest, Dt=Dt, fallback=fallback,
                                     smooth_window=smooth_window)


    def prestim_v_rms(self,debug = False):
        x = -(self.probe_position[self.downsample_probe].squeeze() - self.probeZero)
        t = self.time[self.downsample_probe]
        prestim_x = x[t<0]
        prestim_t = t[t<0]
        prestim_rms = rms_velocity(prestim_t,prestim_x)
        return prestim_rms


    def prestim_holding_cost(self,debug = False):
        x = -(self.probe_position[self.downsample_probe].squeeze() - self.probeZero)
        t = self.time[self.downsample_probe]
        prestim_x = x[t<0]
        prestim_t = t[t<0]
        prestim_hold = holding_cost(prestim_t,prestim_x)
        return prestim_hold/np.diff([prestim_t[0],prestim_t[-1]])


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
        
        target_y = self.pyasXPosition
        target_width = self.pyasWidth
        probe_zero = self.probeZero
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
        fig = Figure(figsize=(6, 6), dpi=200)
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)

        ax.add_patch(patches.Rectangle(
            (time_array[0], target_y),  # Bottom-left corner
            width=max(time_array) - min(time_array),  # Full time range
            height=target_width,  # Width of the target
            color=tgt_clr, alpha=0.3, label='Target'))
        ax.plot(time_array, arduino_output,color=(0.7,0.7,0.7),label='arduino_output')

        ax.plot(time_array, probe_position)
        ax.set_title(f"tr#={int(self.params['trial'])} probe_position: '{self.as_outcome}'")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("um")
        ax.set_xlim(time_array[0], time_array[-1])
        if not from_zero:
            if not use_full_y:
                ax.set_ylim([self.probeZero - 500, self.probeZero + 20])
            else:
                ax.set_ylim([0, 1280])
            ax.axhline(y=self.probeZero, xmin=0, xmax=1, color=(0.5,0.5,0.5), linestyle='--', label='probe_zero')

        else:
            if not use_full_y:
                ax.set_ylim([0 - 500, 0 + 20])
            else:
                ax.set_ylim([0, 1280]-self.probeZero)
            ax.axhline(y=0, xmin=0, xmax=1, color=(0.5,0.5,0.5), linestyle='--', label='probe_zero')

        if savefig or (not format is None):
            format = format or 'png'
            fig.savefig(f'./figpanels/{self._dfc}_Trial_{self.params['trial']}_probe_plot.{format}',format=format)

        return fig, ax
    

    def _plot_ephys_groups(self,group_name,use_full_time=True):           # Additional plot for probe_position
        raise ValueError('Not implemented yet.')
        data = getattr(self,group_name,None)
        target_y = self.params['pyasXPosition']
        target_width = self.params['pyasWidth']
        time_array = self.time if use_full_time else self.trialtime
        
        # Plot the data
        # plt.figure()
        # plt.plot(time_array, data)
        # plt.title(f"Plot of {group_name}")
        # plt.xlabel("Time (s)")
        # plt.ylabel("Value")
        # plt.grid(True)
        # plt.legend()
        # plt.show()


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


    def _ensure_target_params(self):
        # print(self.params)
        if not 'pyasXPosition' in self.params.keys():
            self.write_scalar_if_changed('pyasXPosition',self.params['target_location'][0]) # tr.params['target_location'][0]
        if not 'pyasWidth' in self.params.keys():
            self.write_scalar_if_changed('pyasWidth',self.params['target_location'][1]) # tr.params['target_location'][0]


    def _lazy_load_group(self, group):
        """Create a proxy object to lazily load an HDF5 group."""
        group_obj = type(group.name, (object,), {'_group': group, '_loaded_attrs': {}})()
        return group_obj


    def write_scalar_if_changed(
        self,
        dataset_name: str,
        value,
        group_name=TRIAL_METADATA_GROUP,
        rewrite_attrs: bool = False,     # kept for API compatibility; see add_h5path_attr
        add_h5path_attr: bool = True,
        rtol: float = 0.0,
        atol: float = 0.0
    ) -> bool:
        """
        Write a numeric scalar to /<group>/<dataset_name> in MATLAB-friendly form (1x1 double)
        using hdf5storage, but only if changed.
        """
        mat_path = join(self.path, self.fn)
        existing_val = None

        # Read existing numeric value if present
        if os.path.exists(mat_path):
            with h5py.File(mat_path, "r") as f:
                if group_name in f and dataset_name in f[group_name]:
                    ds = f[group_name][dataset_name]
                    try:
                        arr = ds[...]
                        if np.isscalar(arr):
                            existing_val = float(arr)
                        elif getattr(arr, "size", 0) == 1:
                            existing_val = float(np.array(arr).reshape(-1)[0])
                    except Exception:
                        existing_val = None

        # Compare
        new_val = float(value)
        changed = (existing_val is None) or (not np.isclose(new_val, existing_val, rtol=rtol, atol=atol))
        if not changed:
            return False

        # Delete the old dataset to avoid dtype/shape conflicts, then write via hdf5storage
        with h5py.File(mat_path, "r+") as f:
            grp = f.require_group(group_name)
            if dataset_name in grp:
                del grp[dataset_name]

        h5s.write(
            data=new_val,                              # Python float -> 1x1 double
            path=f"{group_name}/{dataset_name}",
            filename=mat_path,
            matlab_compatible=True,
            store_python_metadata=False
        )

        if add_h5path_attr or rewrite_attrs:
            with h5py.File(mat_path, "r+") as f:
                f[group_name][dataset_name].attrs.create("H5PATH", group_name, dtype=h5py.string_dtype(encoding="ascii"))

        return True
    

    def write_string_if_changed(
        self,
        dataset_name: str,
        value: str,
        group_name=TRIAL_METADATA_GROUP,
        add_h5path_attr: bool = False
    ) -> bool:
        """
        Write a MATLAB-friendly char array (1xN, uint16 code units) using hdf5storage
        if the value changed. Pass Python str; we’ll compare to existing char.
        """
        mat_path = join(self.path, self.fn)
        new_val = "" if value is None else str(value)

        # Read existing as Python str, if present and char-like
        existing_val = None
        if os.path.exists(mat_path):
            with h5py.File(mat_path, "r") as f:
                if group_name in f and dataset_name in f[group_name]:
                    dset = f[group_name][dataset_name]
                    try:
                        if _is_matlab_char_dset(dset):
                            existing_val = _decode_matlab_char(dset[...])
                    except Exception:
                        existing_val = None

        if existing_val == new_val:
            return False

        # Delete old and write via hdf5storage
        with h5py.File(mat_path, "r+") as f:
            grp = f.require_group(group_name)
            if dataset_name in grp:
                del grp[dataset_name]

        h5s.write(
            data=new_val,                               # Python str -> 1xN char with correct attrs
            path=f"{group_name}/{dataset_name}",
            filename=mat_path,
            matlab_compatible=True,
            store_python_metadata=False
        )

        if add_h5path_attr:
            with h5py.File(mat_path, "r+") as f:
                f[group_name][dataset_name].attrs.create("H5PATH", group_name, dtype=h5py.string_dtype(encoding="ascii"))

        return True
    

    def normalize_group_for_matlab(
        self,
        group_name=TRIAL_METADATA_GROUP,
        dry_run: bool = True,
        add_h5path_attr: bool = False,
        make_backup: bool = False
    ) -> int:
        """
        Scan <group_name> for MATLAB-unfriendly shapes and rewrite via hdf5storage.
        - numeric scalar dataspace ()      -> 1x1
        - numeric rank-1 (N,)              -> 1xN
        - char not shaped (1,N)            -> re-emit from Python str
        Returns the number of datasets that would be (or were) modified.
        """
        mat_path = join(self.path, self.fn)
        if not os.path.isfile(mat_path):
            print(f"{mat_path}: not found.")
            return 0

        to_fix = []
        with h5py.File(mat_path, "r") as f:
            if group_name not in f:
                # print(f"{mat_path}: {group_name} not found.")
                return 0
            grp = f[group_name]
            for name, obj in grp.items():
                if not isinstance(obj, h5py.Dataset):
                    continue
                is_char = _is_matlab_char_dset(obj)
                arr = obj[...]
                plan = None
                payload = None

                if is_char:
                    # MATLAB wants 1xN for char
                    if not _char_shape_ok(obj):
                        plan = "char -> 1xN"
                        payload = _decode_matlab_char(arr)  # Python str
                else:
                    if obj.ndim == 0:
                        plan = "scalar dataspace -> 1x1"
                        payload = np.array(arr).item()      # Python scalar
                    elif obj.ndim == 1:
                        plan = "1-D -> 1xN"
                        payload = np.asarray(arr).reshape(1, -1)  # numeric row

                if plan:
                    to_fix.append((name, is_char, plan, payload))

        if not to_fix:
            # print("normalize: nothing to change 🎉")
            return 0

        print("normalize: planned changes:")
        for name, is_char, plan, _ in to_fix:
            t = "char" if is_char else "numeric"
            print(f"  - {group_name}/{name} [{t}] : {plan}")

        if dry_run:
            return len(to_fix)

        if make_backup:
            ts = time.strftime("%Y%m%d_%H%M%S")
            bak = f"{mat_path}.{ts}.bak"
            shutil.copy2(mat_path, bak)
            print(f"Backup: {bak}")

        # Apply rewrites
        for name, is_char, plan, payload in to_fix:
            # remove old dataset first
            with h5py.File(mat_path, "r+") as f:
                if group_name in f and name in f[group_name]:
                    del f[group_name][name]

            # write new payload in MATLAB-compatible manner
            h5s.write(
                data=(str(payload) if is_char else payload),
                path=f"{group_name}/{name}",
                filename=mat_path,
                matlab_compatible=True,
                store_python_metadata=False
            )

            if add_h5path_attr:
                with h5py.File(mat_path, "r+") as f:
                    f[group_name][name].attrs.create("H5PATH", group_name, dtype=h5py.string_dtype(encoding="ascii"))

        print("normalize: done.")
        return len(to_fix)


    def list_meta_keys(self):
        """Return list of dataset names under /meta (empty if none)."""
        with h5py.File(self.file_path, "r") as f:
            if TRIAL_METADATA_GROUP in f:
                return list(f[TRIAL_METADATA_GROUP].keys())
        return []
    
        
    def _read_value_from_meta(self, key, *, default=None, decode_strings=True, squeeze=True):
        """
        Unified reader for /meta datasets:
        - Returns Python scalar for scalar datasets (and for 1-element arrays if squeeze=True).
        - Returns NumPy array for multi-element arrays.
        - If decode_strings=True, tries to return Python str(s) for string datasets.
        """
        if default is None:
            def_val = s.MISSING

        try:
            with h5py.File(join(self.path, self.fn), 'r') as f:
                g = f.get(TRIAL_METADATA_GROUP)
                if g is None or key not in g:
                    if def_val is s.MISSING:
                        raise AttributeError(f"{TRIAL_METADATA_GROUP}/{key} not found in {self.fn}")
                    return def_val

                dset = g[key]
                val = dset[()]  # works for scalars and arrays

                if decode_strings:
                    val = _decode_if_string(dset, val)

                if squeeze:
                    val = _as_python_scalar(val)

                return val

        except Exception as e:
            # print(f"Failed to read {key} from trial {self.fn}: {e}")
            if not def_val is s.MISSING:
                return def_val
            raise AttributeError(f"Failed to read {key} from trial {self.fn}: {e}")


    def _remove_legacy_as_outcome_group(self):
        """Remove the old 'current_as_outcome' group from the trial HDF5 file, if it exists."""
        try:
            with h5py.File(join(self.path, self.fn), 'r+') as f:
                if 'current_as_outcome' in f:
                    if isinstance(f['current_as_outcome'], h5py.Dataset):
                        print(f"[{self.fn}] Deleting group 'current_as_outcome'")
                        del f['current_as_outcome']
                    else:
                        print(f"[{self.fn}] 'current_as_outcome' exists but is not a group — skipping.")
        except Exception as e:
            print(f"[{self.fn}] Failed to delete 'current_as_outcome': {e}")


    def refresh_meta(self, key, *, cache: bool = True):
        """
        Re-read /meta/<key>. Raises AttributeError if missing.
        If previously cached by __getattr__, drop it. Optionally re-cache.
        Never overwrites real class attributes/properties.
        """

        try:
            val = self._read_value_from_meta(key)
        except AttributeError as e:
            # print('No such key: {}'.format(e))
            if key in self.__dict__:
                del self.__dict__[key]
            return

        # 2) Remove only instance-level cache (what __getattr__ set)
        if key in self.__dict__:
            del self.__dict__[key]

        # 3) Optionally re-cache, but don't shadow real class attrs/properties
        if cache and not self._has_class_attr(key):
            setattr(self, key, val)

        return val


    def _has_class_attr(self, name: str) -> bool:
        # True if name is defined on the class (method/property/descriptor), not just instance
        return any(name in cls.__dict__ for cls in type(self).mro()) 


    # ---------------------------------------------------------
    # Dunder Methods
    # ---------------------------------------------------------    
    def __getattr__(self, name):
        """Lazy load: root datasets/groups first; then fall back to /meta/<name>; also expose meta_keys."""
        if name == "meta_keys":
            keys = self.list_meta_keys()
            setattr(self, "meta_keys", keys)
            return keys

        with h5py.File(self.file_path, "r") as file:
            # 1) root-level group/dataset
            if name in file:
                obj = file[name]
                if isinstance(obj, h5py.Group):
                    group_obj = self._lazy_load_group(obj)
                    setattr(self, name, group_obj)
                    return group_obj
                elif isinstance(obj, h5py.Dataset):
                    val = obj[()]
                    setattr(self, name, val)
                    return val

            # 2) fall back to /meta/<name>
            if TRIAL_METADATA_GROUP in file and name in file[TRIAL_METADATA_GROUP]:
                ds = file[TRIAL_METADATA_GROUP][name][:]
                val = _decode_h5_string(ds)
                setattr(self, name, val)  # cache decoded value as regular attr
                return val

        # If nothing matched:
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