from .trial import Trial
from .table import Table
from .sinq import Sinq
from . import kinematics
from . import spike_detection
from . import ephys
from . import bout_analysis
from . import target_triggered
from . import trial_subclasses  # noqa: F401 — registers Trial subclasses

__all__ = ["Trial", "Table", "Sinq", "kinematics", "spike_detection", "ephys",
           "bout_analysis", "target_triggered"]