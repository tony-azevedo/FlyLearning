"""Spike-tick overlay.

Renders each detected spike as a short black tick at the top of the ephys
axis. Uses a blended transform (data x, axes y) so the ticks stay at the
same vertical position regardless of the ephys y-scale and remain visible
even when all voltage/current channels are unchecked.
"""
from __future__ import annotations

import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.transforms import blended_transform_factory

from mapd import spike_detection as sds

from ..overlay import Overlay, register_overlay


@register_overlay
class SpikesOverlay(Overlay):
    name = "Spikes"

    def draw(self, trial, axes):
        try:
            result = sds.load_spikes_from_trial(trial)
        except Exception:
            return
        if result is None:
            return
        spike_samples = np.asarray(getattr(result, "spike_times", []))
        if spike_samples.size == 0:
            return

        # result.spike_times are sample indices; trial.time[0] = -pre_dur, so
        # align spike times to the browser's stimulus-centered x-axis.
        spike_s = (spike_samples / float(result.params.fs)
                   + float(trial.time[0]))
        ax = axes["ephys"]
        trans = blended_transform_factory(ax.transData, ax.transAxes)
        segments = [[(t, 0.92), (t, 1.0)] for t in spike_s]
        lc = LineCollection(
            segments, transform=trans, colors="black",
            linewidths=0.8, zorder=5,
        )
        ax.add_collection(lc)
        self._artists.append(lc)
