"""Movement-bouts overlay.

Runs ``mapd.kinematics.detect_movement_bouts`` on the downsampled probe
trace and shades each detected MOVE bout on the probe axis.
"""
from __future__ import annotations

from mapd.kinematics import STATE_MOVE, _STATE_COLORS, detect_movement_bouts

from ..overlay import Overlay, register_overlay


@register_overlay
class MovementBoutsOverlay(Overlay):
    name = "Movement bouts"

    def draw(self, trial, axes):
        ds = trial.downsample_probe
        t = trial.time[ds].squeeze()
        x = -(trial.probe_position[ds].squeeze() - trial.probeZero)
        try:
            bouts, _states, _t, _x = detect_movement_bouts(t, x)
        except Exception:
            return
        ax_probe = axes["probe"]
        color = _STATE_COLORS.get(STATE_MOVE, "tomato")
        for b in bouts:
            span = ax_probe.axvspan(
                b["start_time"], b["end_time"],
                alpha=0.2, color=color, linewidth=0,
            )
            self._artists.append(span)
