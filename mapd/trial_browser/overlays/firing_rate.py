"""Firing rate overlay.

Plots the Gaussian-smoothed firing rate from :func:`mapd.ephys.spike_rate_from_trial`
on a twin y-axis sharing the ephys axis's x. The kernel width comes from the
browser's kernel selector, so it can be compared against the Vm overlay at a
matched bandwidth — an unmatched pair says more about the kernels than the cell.
"""
from __future__ import annotations

from mapd import ephys

from ..overlay import Overlay, register_overlay


@register_overlay
class FiringRateOverlay(Overlay):
    name = "Firing rate"
    SIGMA_S = 0.025  # kept for backwards compatibility; params['sigma_s'] wins
    DEFAULT_PARAMS = {
        "sigma_s": 0.025,   # 25 ms — full kernel ≈ 100 ms (±2σ)
        "shade_edges": True,
    }

    def __init__(self):
        super().__init__()
        self._twinx = None  # lazy-init on first draw

    def draw(self, trial, axes):
        sigma_s = float(self.params.get("sigma_s", self.SIGMA_S))
        result = ephys.spike_rate_from_trial(trial, sigma_s=sigma_s)
        if result is None:
            return
        t, rate = result

        ax_ephys = axes["ephys"]
        if self._twinx is None:
            self._twinx = ax_ephys.twinx()
            self._twinx.tick_params(axis="y", colors="purple")

        self._twinx.set_visible(True)
        self._twinx.set_ylabel(f"rate (Hz), sigma={sigma_s * 1e3:.0f} ms",
                               color="purple")
        (line,) = self._twinx.plot(t, rate, color="purple", lw=1)
        self._artists.append(line)
        self._twinx.relim()
        self._twinx.autoscale_view()

        if self.params.get("shade_edges"):
            self.shade_kernel_edges(trial, axes, sigma_s)

    def clear(self):
        super().clear()
        if self._twinx is not None:
            self._twinx.set_visible(False)
