# mapd/sentinels.py
from typing import Final
import pandas as pd

class _MissingType:
    __slots__ = ()
    def __repr__(self): return "MISSING"
    def __reduce__(self):
        import importlib
        mod = importlib.import_module(__name__)
        return getattr, (mod, "MISSING")

MISSING: Final = _MissingType()

def is_missing_like(x) -> bool:
    """Looser check for 'missing' values.

    Returns True if:
      - x is the MISSING sentinel
      - x is None
      - x is a pandas/NumPy NA/NaN
    """
    if x is MISSING:
        return True
    if x is None:
        return True
    try:
        if pd.isna(x):  # works for np.nan, pd.NA, None in object arrays
            return True
    except Exception:
        pass
    return False