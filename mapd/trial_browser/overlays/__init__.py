"""Auto-import every overlay module so the registry is populated on import.

Add new overlays by dropping a module in this package and importing it here.
"""
from . import firing_rate, movement_bouts, spikes, subthreshold_vm  # noqa: F401 — registry side-effect

__all__ = ["firing_rate", "movement_bouts", "spikes", "subthreshold_vm"]
