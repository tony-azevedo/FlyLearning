from .trial import Trial
from .table import Table
from .sinq import Sinq
from . import sentinels as s  # import module, not names
from . import kinematics

__all__ = ["Trial", "Table", "Sinq", "quickScanner", "s", "kinematics"]