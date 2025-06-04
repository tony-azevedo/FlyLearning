import pandas as pd
import os
import matplotlib

matplotlib.use("Agg")  # Must be before importing any matplotlib.pyplot-dependent code
import matplotlib.patches as patches
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np
import skvideo.io
from scipy.stats import mode

def export_some_probe_groups(self,index=None,from_zero=True):
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
    target_x_list = []
    target_width_list = []
    probeZero_list = []

    cumulative_time_offset = 0

    for tr in trial_df.Trial:
        probeZero = tr.params['probeZero']
        
        tr_probe_position = tr.probe_position[tr.downsample_probe]
        tr_ardo = tr.arduino_output[tr.downsample_probe].astype(np.int16)
        tr_time = tr.time[tr.downsample_probe]


        fr = 1/mode(np.diff(tr.time[tr.downsample_probe])).mode
    
        target_x = tr.params['pyasXPosition']
        target_width = tr.params['pyasWidth']
        if from_zero:
            tr_probe_position = tr_probe_position-probeZero
            target_x = target_x-probeZero
        
        n = len(tr_time)
        tr_target_x = np.full(n, target_x)
        tr_target_width = np.full(n, tr.params['pyasWidth'])
        # tgt_clr = tr._tgt_clrs[int(tr.params['blueToggle'])]
    
        tr_ardo = -tr_ardo*target_width + target_x + target_width

        time_array_adjusted = tr_time + cumulative_time_offset
        cumulative_time_offset = cumulative_time_offset + tr_time[-1]-tr_time[0]

        time_list.append(time_array_adjusted)
        probe_position_list.append(tr_probe_position)
        arduino_output_list.append(tr_ardo)

        tlims_list.append([time_array_adjusted[0],time_array_adjusted[-1]])
        target_x_list.append(tr_target_x)
        target_width_list.append(tr_target_width)
        probeZero_list.append(np.full(2, probeZero))        
        
    cumulative_time = np.concatenate(time_list)  # Flatten to a single array
    cumulative_probe_position = np.ravel(np.concatenate(probe_position_list))
    cumulative_arduino_output = np.concatenate(arduino_output_list)
    cumulative_target_x = np.concatenate(target_x_list)
    cumulative_target_width = np.concatenate(target_width_list)
    tlims = [cumulative_time[0],cumulative_time[-1]]
    # cumulative_target_y = np.concatenate(cumulative_target_y)
    # cumulative_target_width = np.concatenate(cumulative_target_width)
    cumulative_probeZero = np.concatenate(probeZero_list)

    # # Assuming all vectors are 1D arrays of the same length
    # variables = {
    #     "cumulative_time": cumulative_time,
    #     "cumulative_probe_position": cumulative_probe_position,
    #     "cumulative_arduino_output": cumulative_arduino_output,
    #     "cumulative_target_x": cumulative_target_x,
    #     "cumulative_target_width": cumulative_target_width,
    # }   
    # for name, array in variables.items():
    #     print(f"{name}: shape = {array.shape}")
    #     if array.ndim != 1:
    #         raise ValueError("All vectors must be 1D arrays.")

    df = pd.DataFrame({
        "time_s": np.ravel(cumulative_time),
        "probe_position": np.ravel(cumulative_probe_position),
        "arduino_output": np.ravel(cumulative_arduino_output),
        "target_x": np.ravel(cumulative_target_x),
        "target_width": np.ravel(cumulative_target_width),
    })

    fname = f"./export_data/probe_position_{self.day}_F{self.fly}_C{self.cell}_{self.genotype}_T{index[0]}_{index[-1]}.pkl"
    df.to_pickle("{}".format(fname))

    return fname


import pandas as pd
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def plot_probe_trace_from_pickle(pkl_path, save_path=None):
    # Load data
    df = pd.read_pickle(pkl_path)

    # Extract columns
    time = df["time_s"].values
    probe = df["probe_position"].values
    arduino = df["arduino_output"].values
    target_x = df["target_x"].values
    target_width = df["target_width"].values
    target_x_end = target_x + target_width

    # Create figure and canvas
    fig = Figure(figsize=(10, 6), dpi=100)
    canvas = FigureCanvas(fig)

    # Subplots: only Arduino and Probe now
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212, sharex=ax1)

    # Plot Arduino output
    ax1.plot(time, arduino, color='orange')
    ax1.set_ylabel("Arduino")

    # Plot probe and target bounds on same axis
    ax2.plot(time, probe, label="Probe position", color='blue')
    ax2.plot(time, target_x, '--', color='gray', label="Target start")
    ax2.plot(time, target_x_end, '--', color='gray', label="Target end")

    ax2.set_ylabel("Probe")
    ax2.set_xlabel("Time (s)")
    ax2.legend(loc="upper right")

    # Save or return
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path)
        print(f"Saved figure to {save_path}")
    else:
        canvas.draw()
        print("Figure rendered (not shown interactively)")

    return fig