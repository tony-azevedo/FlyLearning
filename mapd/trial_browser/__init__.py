"""Qt/PySide6 desktop app for walking through trials of a mapd.Table.

Launch from a terminal::

    python scripts/browse_trials.py 241203_F2_C1

See ``mapd.trial_browser.browser.TrialBrowser`` for the window class and
``scripts/browse_trials.py`` for the CLI entry point.
"""
from .browser import TrialBrowser, launch

__all__ = ["TrialBrowser", "launch"]
