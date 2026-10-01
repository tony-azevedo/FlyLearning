"""Subthreshold Vm overlay.

Draws the spike-blanked, smoothed membrane potential from
:func:`mapd.ephys.subthreshold_vm` directly on the ephys axis — same units as the
raw trace, so no twin axis is needed and the two are read against each other.

The blanking window is measured from the cell's own spike-triggered average
rather than assumed, and cached per cell: spike width and after-hyperpolarization
differ between cells, and an AHP outlasting the blank leaks into "subthreshold"
Vm in proportion to firing rate — the one artifact that can manufacture a
Vm-vs-rate relation out of nothing.
"""
from __future__ import annotations

import numpy as np

from mapd import ephys

from ..overlay import Overlay, register_overlay


@register_overlay
class SubthresholdVmOverlay(Overlay):
    name = "Subthreshold Vm"
    DEFAULT_PARAMS = {
        "sigma_s": 0.025,      # pushed by the browser's kernel selector
        "channel": "voltage_1",
        "blank": None,         # (pre_s, post_s); None -> measure from the STA
        "sta_trials": 1,       # trials pooled for the STA (the current one)
        "show_unblanked": False,
        "shade_edges": True,
    }

    def __init__(self):
        super().__init__()
        self._blank_cache: dict = {}   # dfc -> (pre_s, post_s)

    def _blank_window(self, trial):
        """Blank window for this trial's cell, measured once and cached.

        Cached per cell rather than per trial: measuring an STA on every
        navigation would make stepping through trials sluggish, and the spike
        shape is a property of the recording, not of one trial. Pass
        ``blank=(pre, post)`` via ``set_params`` to override — e.g. with a
        Table-wide measurement from ``figure7_analysis.measure_blank_window``.
        """
        override = self.params.get("blank")
        if override is not None:
            return tuple(override)
        dfc = getattr(trial, "_dfc", None) or "?"
        if dfc not in self._blank_cache:
            sta = ephys.spike_triggered_average(
                trial, channel=self.params["channel"])
            self._blank_cache[dfc] = ephys.sta_blank_window(sta)
        return self._blank_cache[dfc]

    def draw(self, trial, axes):
        ax = axes["ephys"]
        sigma_s = float(self.params["sigma_s"])
        pre_s, post_s = self._blank_window(trial)

        result = ephys.subthreshold_vm(
            trial, channel=self.params["channel"],
            blank_pre_s=pre_s, blank_post_s=post_s, sigma_s=sigma_s)
        if result is None:
            return
        t, vm, info = result

        (line,) = ax.plot(t, vm, color="#d62728", lw=1.4, zorder=4,
                          label=f"Vm blanked, sigma={sigma_s * 1e3:.0f} ms")
        self._artists.append(line)

        if self.params.get("show_unblanked"):
            ctl = ephys.subthreshold_vm(
                trial, channel=self.params["channel"],
                blank_pre_s=0.0, blank_post_s=0.0, sigma_s=sigma_s)
            if ctl is not None:
                (l2,) = ax.plot(t, ctl[1], color="#ff7f0e", lw=1.0, ls="--",
                                zorder=3, label="Vm unblanked (control)")
                self._artists.append(l2)

        # frac_blanked is the number that decides whether this trace is a
        # measurement or mostly interpolation, so it is always on screen.
        txt = ax.annotate(
            f"blank -{pre_s * 1e3:.1f}/+{post_s * 1e3:.1f} ms   "
            f"{info['frac_blanked']:.0%} interpolated",
            xy=(0.005, 0.02), xycoords="axes fraction", fontsize=7,
            color="#d62728", ha="left", va="bottom", zorder=5)
        self._artists.append(txt)

        if self.params.get("shade_edges"):
            self.shade_kernel_edges(trial, axes, sigma_s)
