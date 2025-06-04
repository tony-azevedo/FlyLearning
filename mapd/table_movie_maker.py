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


def create_probe_frame(xlim=(0, 1280), ylim=(0, 128), radius=8):
    fig = Figure(figsize=(6.4, 0.64), dpi=200)
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    assert fig.canvas.get_width_height() == (1280, 128), "Unexpected canvas size"
    
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')
    ax.axis('off')

    # Placeholder target rectangle (initial position = 0, width = 0)
    rect = patches.Rectangle((0, 0), 0, ylim[1], edgecolor='white', facecolor='none')
    ax.add_patch(rect)

    # Placeholder probe dot
    dot = patches.Circle((0, ylim[1] // 2), radius, color='white')
    ax.add_patch(dot)

    time_text = ax.text(
        0.98, 0.02, "", transform=ax.transAxes,
        color='white', fontsize=10, ha='right', va='bottom'
    )

    return fig, ax, dot, rect, time_text


def init_video_writer(filename, frame_rate=30, crf=28):
    writer = skvideo.io.FFmpegWriter(
        filename,
        inputdict={'-framerate': str(frame_rate)},
        outputdict={
            '-vcodec': 'libx264',
            '-pix_fmt': 'yuv420p',
            '-crf': str(crf),
            '-preset': 'slow',
            '-movflags': '+faststart'
        }
    )
    return writer

def update_probe_frame(dot, rect, probe_position, target_x, target_width, 
                       time_text=None, time_seconds = None, light_on=False, 
                       edgewidth_inside=3, edgewidth_outside=1):
    """
    Updates the dot and rectangle in the frame.
    - dot: matplotlib Circle object
    - rect: matplotlib Rectangle object
    - probe_position: scalar or (x, y)
    - target_x: left edge of target region
    - target_width: width of target region
    - light_on: bool, adds yellow fill to rectangle if True
    - edgewidth_inside: line width if probe is inside the target
    - edgewidth_outside: line width if probe is outside the target
    """

    # Update dot position
    if np.ndim(probe_position) == 0 or len(probe_position) == 1:
        x = probe_position[0] if np.ndim(probe_position) else probe_position
        y = dot.center[1]
    else:
        x, y = probe_position[0], probe_position[1]
    dot.center = (x, y)

    # Update target rectangle
    rect.set_x(target_x)
    rect.set_width(target_width)
    rect.set_edgecolor('blue' if light_on else 'white')

    inside = target_x <= x <= (target_x + target_width)
    rect.set_linewidth(edgewidth_inside if inside else edgewidth_outside)

    # Update time text
    if time_text is not None and time_seconds is not None:
        time_text.set_text(f"{time_seconds:.2f} s")


def make_movie_from_index(self, index=None):
    fig, ax, dot, rect,time_text = create_probe_frame(radius=6)
    # writer = init_video_writer("debug_probe_video.mp4", frame_rate=30)

    tr = self.df.loc[index[0],'Trial'];

    tr_probe_position = tr.probe_position[tr.downsample_probe]
    tr_ardo = tr.arduino_output[tr.downsample_probe].astype(np.bool)
    tr_time = tr.time[tr.downsample_probe]

    fr = 1/mode(np.diff(tr.time[tr.downsample_probe])).mode
    # writer = init_video_writer("debug_probe_video.mp4", frame_rate=fr, crf=28)

    update_probe_frame(
        dot, rect,
        probe_position=tr_probe_position[0],
        target_x=tr.params['pyasXPosition'],
        target_width=tr.params['pyasWidth'],
        light_on=tr_ardo[0]
    )

    writer = init_video_writer(f"./probe_movies/pm_{self.day}_F{self.fly}_C{self.cell}_{self.genotype}_T{index[0]}_{index[-1]}.mp4", frame_rate=fr, crf=28)
    
    fig.canvas.draw()

    for tr in self.df.loc[index, 'Trial']:
        tr_probe_position = tr.probe_position[tr.downsample_probe]
        tr_ardo = tr.arduino_output[tr.downsample_probe].astype(np.bool)
        tr_time = tr.time[tr.downsample_probe]

        # Example loop
        for idx,pos,ao,tstep in zip(range(len(tr_probe_position)),tr_probe_position,tr_ardo,tr_time):
            # idx = 500
            # pos = tr_probe_position[idx]
            # ao = tr_ardo[idx]
            # tstep = tr_time[idx]
            # if idx == 500:   
            update_probe_frame(
                dot, rect,
                probe_position=pos,
                target_x=tr.params['pyasXPosition'],
                target_width=tr.params['pyasWidth'],
                time_text=time_text,
                time_seconds=tstep,
                light_on=ao
            )

            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image = np.asarray(fig.canvas.buffer_rgba())
            image_rgb = image[..., :3]  # drop alpha for ffmpeg compatibility
            writer.writeFrame(image_rgb)

    writer.close()
    