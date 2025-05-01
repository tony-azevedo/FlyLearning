# table plotting functions for flylearning analysis objects
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
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from functools import wraps

# _force_clrs = [(.92, .6, .7),(1,.9,1)]
_force_clrs = [
    (np.float64(0.95447591), np.float64(0.47082238), np.float64(0.32310953)),
    (np.float64(0.7965014), np.float64(0.10506637), np.float64(0.31063031)),
    ]

def auto_ax_and_save(default_title=None):
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, ax=None, format=None, savefig=False, **kwargs):
            created_fig = None
            if ax is None:
                fig, ax = plt.subplots()
                created_fig = fig

            trial_ids = kwargs.get("index", args[0] if args else None)
            trial_str = ""
            if trial_ids is not None and hasattr(trial_ids, "min") and hasattr(trial_ids, "max"):
                try:
                    trial_str = f"_trials_{int(trial_ids.min())}_{int(trial_ids.max())}"
                except Exception:
                    pass  # Fail silently if something's weird with trial_ids
            
            # Call the actual plotting function
            result = func(self, *args, ax=ax, **kwargs)

            # Set title if a default is given and no title exists
            if default_title: # and not ax.get_title():
                ax.set_title(f'{self._dfc} {default_title}: {trial_ids.min()} - {trial_ids.max()}')

            # Save if needed
            if created_fig and (savefig or format):
                fmt = format if format else "png"
                filename = f"{func.__name__}{trial_str}.{fmt}"
                # fig.tight_layout()
                fig.savefig(f'{self.fig_folder}/{self._dfc}_{func.__name__}{trial_str}.{fmt}',transparent=True)
                # fig.savefig(filename, format=fmt, bbox_inches='tight')
                # plt.close(fig)


            return result
        return wrapper
    return decorator


def pyas_state(df):
    states = df.pyasState.unique()
    if len(states) > 1:
        return None
    else:
        return 'hi' if states=='hi' else 'lo'


def plot_some_trials(self,index,from_zero=True,savefig=False,format=None):
    fig, axs = plt.subplots(3, 1, figsize=(8, 10))
    self.plot_some_as_piezo(index,ax=axs[0])
    self.plot_some_probe_groups(index,ax=axs[1],from_zero=from_zero)
    self.plot_some_phys(index,ax=axs[2],from_zero=from_zero)
    if (savefig or format):
        fmt = format if format else "png"
        # fig.tight_layout()
        trial_str = f"_trials_{int(index.min())}_{int(index.max())}"
        figfn = f'{self.fig_folder}/{self._dfc}_plot_some_trials{trial_str}.{fmt}'
        fig.savefig(figfn)
        print(f'Saved: {figfn}')
        # fig.savefig(filename, format=fmt, bbox_inches='tight')
        # plt.close(fig)


@auto_ax_and_save(default_title="Probe position")
def plot_some_probe_groups(self,index=None,from_zero=True,ax=None,savefig=False,format=None):
    if index.empty:
        raise ValueError('No such trials. Try: T._excluded_df.loc[a:b]')

    is_sequential = index.is_monotonic_increasing and (index[1:] - index[:-1]).min() == 1
    if not is_sequential:
        KeyError('Index is not sequential')

    try:
        trial_df = self.df.loc[index]
    except KeyError as e:
        print(f'Index is incorrectly formatted. If using a single trial, enclose the number in brackets to get a dataframe []')
        raise
    
    time_list = []
    probe_position_list = []
    arduino_output_list = []

    tlims_list = []
    target_y_list = []
    target_width_list = []
    target_color_list = []
    probeZero_list = []

    cumulative_time_offset = 0

    for tr in trial_df.Trial:
        probeZero = tr.params['probeZero']
        probe_position = getattr(tr,'probe_position',None)
        
        arduino_output = getattr(tr,'arduino_output',None)
        time_array = tr.time 
    
        target_y = tr.params['pyasXPosition']
        target_width = tr.params['pyasWidth']
        if from_zero:
            probe_position = getattr(tr,'probe_position',None)-probeZero
            target_y = target_y-probeZero
        
        tgt_clr = tr._tgt_clrs[int(tr.params['blueToggle'])]
    
        arduino_output = -arduino_output*target_width + target_y + target_width

        time_array_adjusted = time_array + cumulative_time_offset
        cumulative_time_offset = cumulative_time_offset + time_array[-1]-time_array[0]

        time_list.append(time_array_adjusted)
        probe_position_list.append(probe_position)
        arduino_output_list.append(arduino_output)

        tlims_list.append([time_array_adjusted[0],time_array_adjusted[-1]])
        target_y_list.append(target_y)
        target_width_list.append(target_width)
        target_color_list.append(np.full(3, tgt_clr))  
        probeZero_list.append(np.full(2, probeZero))        
        
    cumulative_time = np.concatenate(time_list)  # Flatten to a single array
    cumulative_probe_position = np.concatenate(probe_position_list)
    cumulative_arduino_output = np.concatenate(arduino_output_list)
    tlims = [cumulative_time[0],cumulative_time[-1]]
    # cumulative_target_y = np.concatenate(cumulative_target_y)
    # cumulative_target_width = np.concatenate(cumulative_target_width)
    cumulative_probeZero = np.concatenate(probeZero_list)

    for tgt_y,tgt_w,tlm,tgt_clr in zip(target_y_list,target_width_list,tlims_list,target_color_list):
        ax.add_patch(plt.Rectangle(
            (tlm[0], tgt_y),  # Bottom-left corner
            width=max(tlm) - min(tlm),  # Full time range
            height=tgt_w,  # Width of the target
            color=tgt_clr, alpha=0.3, label='Target'))
        
    ax.plot(cumulative_time, cumulative_arduino_output,color=(0.7,0.7,0.7),label='arduino_output')

    ax.plot(cumulative_time, cumulative_probe_position)
    # ttl = f'tr_n_{index[0]}_to_{index[-1]}'
    # ax.set_title(ttl)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("um")
    ax.set_xlim(cumulative_time[0], cumulative_time[-1])
    if not from_zero:
        ax.set_ylim([cumulative_probeZero.min() - 500, cumulative_probeZero.max() + 20])
        ax.axhline(y=cumulative_probeZero.min(), xmin=0, xmax=1, color=(0.5,0.5,0.5), linestyle='--', label='probe_zero')
    else:
        ax.set_ylim([0 - 500, 0 + 20])
        ax.axhline(y=0, xmin=0, xmax=1, color=(0.5,0.5,0.5), linestyle='--', label='probe_zero')


@auto_ax_and_save(default_title="AS and peizo")
def plot_some_as_piezo(self,index=None,ax=None,savefig=False,format=None):
    if index.empty:
        raise ValueError('No such trials. Try: T._excluded_df.loc[a:b]')

    is_sequential = index.is_monotonic_increasing and (index[1:] - index[:-1]).min() == 1
    if not is_sequential:
        KeyError('Index is not sequential')
    time_list = []
    piezo_list = []
    arduino_output_list = []

    tlims_list = []

    try:
        trial_df = self.df.loc[index]
    except KeyError as e:
        print(f'Index is incorrectly formatted. If using a single trial, enclose the number in brackets to get a dataframe []')
        raise

    cumulative_time_offset = 0

    for tr in trial_df.Trial:
        piezo = getattr(tr,'sgsmonitor',None)
        arduino_output = getattr(tr,'arduino_output',None)
        time_array = tr.time 

        time_array_adjusted = time_array + cumulative_time_offset
        cumulative_time_offset = cumulative_time_offset + time_array[-1]-time_array[0]

        time_list.append(time_array_adjusted)
        piezo_list.append(piezo)
        arduino_output_list.append(arduino_output)

        tlims_list.append([time_array_adjusted[0],time_array_adjusted[-1]])
        
    cumulative_time = np.concatenate(time_list)  # Flatten to a single array
    cumulative_piezo = np.concatenate(piezo_list)
    cumulative_arduino_output = np.concatenate(arduino_output_list)
    tlims = [cumulative_time[0],cumulative_time[-1]]

        
    ax.plot(cumulative_time, cumulative_arduino_output,color=(0.7,0.7,0.7),label='arduino_output')

    ax.plot(cumulative_time, cumulative_piezo)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("V")
    ax.set_xlim(cumulative_time[0], cumulative_time[-1])


@auto_ax_and_save(default_title="E Phys")
def plot_some_phys(self,index=None,from_zero=False,ax=None,savefig=False,format=None):
    if index.empty:
        raise ValueError('No such trials. Try: T._excluded_df.loc[a:b]')

    is_sequential = index.is_monotonic_increasing and (index[1:] - index[:-1]).min() == 1
    if not is_sequential:
        KeyError('Index is not sequential')
    time_list = []
    voltage_1_list = []

    trial_df = self.df.loc[index]
    cumulative_time_offset = 0

    for tr in trial_df.Trial:
        voltage_1 = getattr(tr,'voltage_1',None)
        time_array = tr.time         
        time_array_adjusted = time_array + cumulative_time_offset
        cumulative_time_offset = cumulative_time_offset + time_array[-1]-time_array[0]

        time_list.append(time_array_adjusted)
        voltage_1_list.append(voltage_1)
        
    cumulative_time = np.concatenate(time_list)  # Flatten to a single array
    cumulative_voltage_1 = np.concatenate(voltage_1_list)

    # Plot
    ax.plot(cumulative_time, cumulative_voltage_1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("mV")
    ax.set_xlim(cumulative_time[0], cumulative_time[-1])


def plot_outcomes(self,savefig=False,format='png'):
    # Plot each row as a vertical tick mark at its categorical position
    
    fig, ax = plt.subplots(figsize=(8,4))
    rec_min = self.df['as_outcome'].cat.categories.get_loc('no_as_no_mv')
    rec_max = self.df['as_outcome'].cat.categories.get_loc('probe')

    # Non rest trials
    T = self.df[self.df['is_rest']==False]
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
                alpha=0.5
            )
        ax.add_patch(rect)

    # Indicate if the VNC was cut
    if '_vnc_status_cat' in self.df.columns:
        T_rows = self.df.loc[self.df['_vnc_status_cat']=='cut']
        rec_min = self.df['as_outcome'].cat.categories.get_loc('probe')
        tr_min = T_rows.index.min()
        tr_max = T_rows.index.max()
        rect = patches.Rectangle(
                (tr_min, rec_min-0.5),        # Bottom-left corner of the rectangle
                (tr_max - tr_min + 1),       # Width (covers the specified rows)
                1,                        # Height (covers all categories)
                edgecolor='none',
                facecolor='lightgray',
                alpha=1
            )
        ax.add_patch(rect)
        ax.text(tr_min,rec_min,'NC cut', fontsize=8, ha='left', va='center')
    
    # Indicate Blue or Green LED
    if '_filtercube_status_cat' in self.df.columns:
        rec_min = self.df['as_outcome'].cat.categories.get_loc('info')

        print(rec_min)
        for led in ['blue','green']:
            T_rows = self.df.loc[self.df['_filtercube_status_cat']==led]
            tr_min = T_rows.index.min()
            tr_max = T_rows.index.max()
            rect = patches.Rectangle(
                    (tr_min, rec_min-0.5),        # Bottom-left corner of the rectangle
                    (tr_max - tr_min + 1),       # Width (covers the specified rows)
                    1,                        # Height (covers all categories)
                    edgecolor='none',
                    facecolor=led,
                    alpha=0.2
                )
            ax.add_patch(rect)
            ax.text(tr_min,rec_min,f'{led} LED', fontsize=8, ha='left', va='center')

    y_positions = self.df['as_outcome'].cat.codes
    x_positions = self.df.index
    ax.scatter(x_positions, y_positions, marker='|', s=200, color='black',linewidths=0.5)

    # Label the y-axis with the category names
    ax.set_yticks(range(len(self.df['as_outcome'].cat.categories)), self.df['as_outcome'].cat.categories)
    ax.invert_yaxis()

    ax.set_xlabel("Trial Index")
    ax.set_ylabel("Outcome")
    ax.set_title(f"{self._dfc} {self.genotype} outcomes")
    
    timeout_trials = self.df.loc[self.df.as_outcome=='timeout']
    # timeout_trials = timeout_trials[timeout_trials['_vnc_status_cat']=='intact']
    blue_timeout_trials = timeout_trials.loc[timeout_trials.blueToggle==1]
    if not timeout_trials.empty:
        bstr = "{} of {} timeouts ({:.1f}%) during blue light".format(
            blue_timeout_trials.shape[0],
            timeout_trials.shape[0],
            blue_timeout_trials.shape[0]/timeout_trials.shape[0]*100)
    else:
        bstr = f"{timeout_trials.shape[0]} timeouts"

    print(bstr)
    if '_vnc_status_cat' in timeout_trials.columns:
        intact_timeouts = timeout_trials.loc[timeout_trials['_vnc_status_cat']=='intact']
        intact_blue_timeouts = intact_timeouts.loc[timeout_trials.blueToggle==1]
        if not timeout_trials.empty:
            bstr = "{} of {} timeouts ({:.1f}%) during blue light with vnc intact".format(
                intact_blue_timeouts.shape[0],
                intact_timeouts.shape[0],
                intact_blue_timeouts.shape[0]/intact_timeouts.shape[0]*100)
        else:
            bstr = f"{intact_timeouts.shape[0]} timeouts"

        print(bstr)
    
    probe_y = self.df['as_outcome'].cat.categories.get_loc('probe')
    ax.text(0, probe_y, bstr, fontsize=8, ha='left', va='center')

    if not blue_timeout_trials.empty:
        y_positions = blue_timeout_trials['as_outcome'].cat.codes
        x_positions = blue_timeout_trials.index
        blueclr = (0.339, 0.4235, 0.95)
        ax.scatter(x_positions, y_positions, marker='|', s=200, color=blueclr)

    if savefig:
        fig.savefig(f'{self.fig_folder}/{self._dfc}_{self._genotype}_as_outcomes.{format}',format=format)
    
    plt.show()


def plot_probe_distribution(self,binwidth=2,bin_min=None,bin_max=None,filter=None,index=None,savefig=False,format=None):
    from collections import Counter

    if index is None:
        index = self.df.index
    probe_positions = self.probe_positions_df(self.df.loc[index])
    probe_positions = self.downsample_probe_df(probe_positions)

    if bin_min is None:
        bin_min = probe_positions.probe_min.min() # Max flexion
    if bin_max is None:
        bin_max = probe_positions.probe_max.max() # Should be ProbeZero
    probe_bins = np.arange(bin_min, bin_max, binwidth)

    ppi = probe_positions.index
    if not filter is None:
        for key in filter:
            probe_positions.loc[ppi,key] = self.df.loc[ppi,key]

        for key in filter:
            probe_positions = probe_positions.loc[probe_positions[key]==filter[key],:]

    # print('Histogram for {} rows'.format(probe_positions.shape[0]))

    # Define a function to calculate the histogram
    def calculate_histogram(array, bins):
        counts, _ = np.histogram(array, bins=bins)
        return counts
    
    probe_positions['histogram'] = probe_positions['probe_positions'].apply(lambda arr: calculate_histogram(arr, probe_bins))
    summed_histogram = np.sum(np.vstack(probe_positions['histogram'].to_numpy()), axis=0)

    target_tuples = list(zip(self.df.pyasXPosition-self.df.probeZero, self.df.pyasWidth, self.df.pyasState))
    most_common_tuples = Counter(target_tuples).most_common(2)
    # print(most_common_tuples)

    fig, ax = plt.subplots(figsize=(6, 6))

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
    
    ax.set_xlabel('Probe Position')
    ax.set_ylabel('Frequency')
    ax.grid(True, alpha=0.5)

    if not filter is None:
        titl = '{flk} ({idx})'.format(
                    dfc = self._dfc,
                    gno = self._genotype,
                    flk = ', '.join([filter[key] for key in filter.keys()]),
                    idx = '-'.join([f'{probe_positions.index[0]}', f'{probe_positions.index[-1]}']),
                    fmt = format)
    else: titl = ''
    ax.set_title(f'Probe: {titl}')
    print(titl)

    if savefig or (not format is None):
        format = format or 'png'
        if not filter is None:
            fig.savefig('{self.fig_folder}/{dfc}_{gno}_probe_dist_{flk}_{idx}.{fmt}'.format(
                        dfc = self._dfc,
                        gno = self._genotype,
                        flk = '_'.join([filter[key] for key in filter.keys()]),
                        idx = '_'.join([f'{probe_positions.index[0]}', f'{probe_positions.index[-1]}']),
                        fmt = format),
                        format=format)
            return
        else:
            fig.savefig(f'{self.fig_folder}/{self._dfc}_{self._genotype}_probe_dist_all.{format}',format=format)
            return
        
    plt.show()


def plot_probe_position_heatmap(self,index=None,savefig=False,format=None,cmin=None,cmax=None):
    probe_position_hm_df = self.get_probe_position_df()

    fig, ax = plt.subplots(figsize=(8, 8))  # Adjust figsize as needed
    # sns.heatmap(probe_position_df, cmap="rocket", ax=ax, cbar_kws={'label': 'probe_position'})
    rocket_cmap = sns.color_palette("rocket", as_cmap=True)
    
    if cmin is None:
        cmin = np.min(probe_position_hm_df.values)
    if cmax is None:
        cmax = np.max(probe_position_hm_df.values)

    if not index is None:
        if index.empty:
            raise ValueError('No such trials. Try: T._excluded_df.loc[a:b]')

        is_sequential = index.is_monotonic_increasing and (index[1:] - index[:-1]).min() == 1
        if not is_sequential:
            KeyError('Index is not sequential')

        probe_position_hm_df = probe_position_hm_df.loc[index]

    im = ax.imshow(probe_position_hm_df.values, 
                    aspect='auto', 
                    cmap=rocket_cmap, 
                    vmin=cmin,
                    vmax=cmax,
                    extent=(probe_position_hm_df.columns.min(), 
                            probe_position_hm_df.columns.max(), 
                            probe_position_hm_df.shape[0],
                            0),
                    origin='upper')

    # Put on rectangles to indicate pyas state
    t = self.df.loc[probe_position_hm_df.index] #[['op_cnd_blocks','pyasState']]
    t = t[t['is_rest']==False]
    for ocb in t.op_cnd_blocks.unique():
        # print(ocb)
        T_rows = t[t.op_cnd_blocks==ocb]
        
        tr_min = T_rows.index.min() - t.index.min()
        tr_max = T_rows.index.max() - t.index.min()
        t_min = probe_position_hm_df.columns.min()
        t_max = probe_position_hm_df.columns[-3]
        width = t_max-t_min

        rect = patches.Rectangle(
                (t_min,tr_min-1),        # Bottom-left corner of the rectangle
                width,           # Width (covers the specified rows)
                (tr_max - tr_min+1), # Height (covers all categories)    
                linewidth=1,                        
                edgecolor='black',
                facecolor='none',
                alpha=1
            )
        ax.add_patch(rect)

        # target patches
        pyas_target = T_rows['pyasXPosition'].mean() + T_rows['pyasWidth'].mean() - T_rows['probeZero'].mean()
        normalized_value = (pyas_target - cmin) / (cmax - cmin)
        # print(normalized_value)
        tgt_clr = rocket_cmap(normalized_value)
        # print(tgt_clr)
        rect = patches.Rectangle(
                (t_min,tr_min-1),        # Bottom-left corner of the rectangle
                0.2,           # Width (covers the specified rows)
                (tr_max - tr_min+1), # Height (covers all categories)    
                linewidth=1,                        
                edgecolor='none',
                facecolor=tgt_clr,
                alpha=1
            )
        ax.add_patch(rect)

    # Customize labels and title
    cbar = fig.colorbar(im,ax=ax,label='probe_position')
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(f"{self._dfc} {self.genotype} probe_position")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("trial")

    tick_values = np.arange(-1.0, float(probe_position_hm_df.columns.max()), 0.5)
    ax.set_xticks(tick_values)
    ax.set_xticklabels([f"{tv:.1f}" for tv in tick_values])  # Format times as needed

    # tick_values = probe_position_hm_df.index.to_numpy()
    # print(tick_values)
    # ax.set_yticks(tick_values)
    # ax.set_yticklabels([f"{tv:.1f}" for tv in tick_values])  # Format times as needed
    
    # # Show the plot
    # plt.tight_layout()
    plt.show()

    if savefig or (not format is None):
        format = format or 'png'
        fig.savefig(f'{self.fig_folder}/{self._dfc}_{self._genotype}_heatmap.{format}',format=format)


    return probe_position_hm_df


def plot_sensory_responses(self,target=None,piezo_step=None):
    index = self.df.loc[(self.df['pyas_target'] == target) and (self.df['pyas_target'] == piezo_step)]


def plot_trial_computations(self,method_name: str,savefig=False,format='png'):
    """Plot a value from the dataframe across trials."""
    if method_name not in self.df.columns:
        raise ValueError(f"Column '{method_name}' not found in DataFrame. "
                         f"Call compute_trial_method('{method_name}') first.")
    
    fig, ax = plt.subplots()
    y = self.df[method_name]
    rec_min = 0
    rec_max = y.max()

    ax = self.plot_plotting_context(ax=ax,rec_min=rec_min,rec_max=rec_max)
    
    x_positions = self.df.index
    ax.plot(x_positions, y, linestyle='-', color='black',linewidth=0.5)

    # Label the y-axis with the category names
    ax.set_xlabel("Trial Index")
    ax.set_ylabel(method_name)
    ax.set_title(f"{self._dfc} {self.genotype} {method_name}")
    
    if savefig:
        fig.savefig(f'{self.fig_folder}/{self._dfc}_{self._genotype}_{method_name}.{format}',format=format)
    
    plt.show()


def plot_plotting_context(self,ax=None,rec_min=0,rec_max=1):
    # Non rest trials
    T = self.df[self.df['is_rest']==False]
    for ocb in T.op_cnd_blocks.unique():
        T_rows = T[T.op_cnd_blocks==ocb]
        
        pyasstate = pyas_state(T_rows)

        tgt_clr = _force_clrs[0 if pyasstate=='lo' else 1]

        tr_min = T_rows.index.min()
        tr_max = T_rows.index.max()
        rect = patches.Rectangle(
                (tr_min, rec_min),        # Bottom-left corner of the rectangle
                (tr_max - tr_min + 1),       # Width (covers the specified rows)
                rec_max,                        # Height (covers all categories)
                edgecolor=tgt_clr,
                facecolor=tgt_clr,
                alpha=0.5
            )
        ax.add_patch(rect)

    # Indicate if the VNC was cut
    if '_vnc_status_cat' in self.df.columns:
        T_rows = self.df.loc[self.df['_vnc_status_cat']=='cut']
        rec_min = self.df['as_outcome'].cat.categories.get_loc('probe')
        tr_min = T_rows.index.min()
        tr_max = T_rows.index.max()
        rect = patches.Rectangle(
                (tr_min, rec_min-(rec_max-rec_min)*.05),        # Bottom-left corner of the rectangle
                (tr_max - tr_min + 1),       # Width (covers the specified rows)
                (rec_max-rec_min)*.05,                        
                edgecolor='none',
                facecolor='lightgray',
                alpha=1
            )
        ax.add_patch(rect)
        ax.text(tr_min,rec_min,'NC cut', fontsize=8, ha='left', va='center')
    
    # Indicate Blue or Green LED
    if '_filtercube_status_cat' in self.df.columns:
        rec_min = self.df['as_outcome'].cat.categories.get_loc('info')

        # print(rec_min)
        for led in ['blue','green']:
            T_rows = self.df.loc[self.df['_filtercube_status_cat']==led]
            tr_min = T_rows.index.min()
            tr_max = T_rows.index.max()
            rect = patches.Rectangle(
                    (tr_min, rec_min-(rec_max-rec_min)*.05),        # Bottom-left corner of the rectangle
                    (tr_max - tr_min + 1),       # Width (covers the specified rows)
                    (rec_max-rec_min)*.05,                        # Height (covers all categories)
                    edgecolor='none',
                    facecolor=led,
                    alpha=0.2
                )
            ax.add_patch(rect)
            ax.text(tr_min,rec_min,f'{led} LED', fontsize=8, ha='left', va='center')

    return(ax)


