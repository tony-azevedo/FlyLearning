"""TrialBrowser QMainWindow.

Matplotlib is embedded via ``FigureCanvasQTAgg`` — we never call
``pyplot``. Axes and line artists are created once in ``_build_figure``
and only have their data updated on each trial switch; drawing is
delegated to ``Trial.draw_in_browser`` so protocol-specific subclasses
can add overlays.

Phase-1 layout (left → right):

    QListWidget (trial numbers) | Figure + NavigationToolbar | Metadata panel

Bottom: Prev / Next / trial# QLineEdit, channel checkboxes, status bar.
Arrow keys step Prev/Next.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtCore import QObject, QThread, Qt, Signal
from PySide6.QtGui import (
    QBrush, QColor, QCursor, QDoubleValidator, QIntValidator, QKeySequence,
    QShortcut,
)
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QListWidget, QListWidgetItem, QMainWindow, QPushButton,
    QSplitter, QStatusBar, QVBoxLayout, QWidget,
)

from mapd.table import (PROBEZERO_CONV_OFFSET, PROBEZERO_CORRECTED_TOL,
                        probezero_provenance)
from . import overlays  # noqa: F401 — imports populate the overlay registry
from .overlay import available_overlays

PROBE_CHANNEL = "probe_position"
EPHYS_CHANNELS = ("voltage_1", "voltage_2", "current_1", "current_2",
                  "current_extEMG")
DEFAULT_ACTIVE = ("voltage_1",)
METADATA_FIELDS = ("trial", "as_outcome", "pyasState", "vnc_status",
                   "excluded", "ephys_status", "ephys_note",
                   "probeZero", "pyasXPosition", "pyasWidth")
# Read-only here by design: ephys quality is set from the sinq notebook, which is
# the record of how the data was processed. A click-to-edit control in the browser
# would make annotations that leave no trace in that record.
EPHYS_STATUS_COLORS = {"bad": "#c62828", "redetect": "#ef6c00"}
# Gaussian sd choices offered by the kernel selector, shared by every overlay
# that smooths (firing rate, subthreshold Vm).
KERNEL_SIGMAS_S = (0.005, 0.010, 0.025, 0.050, 0.100, 0.250, 0.500)
DEFAULT_KERNEL_S = 0.025
EXPORT_FOLDER = "figpanels_browser"
# Default text for each axis-limit edit at browser startup. Blank = autoscale.
DEFAULT_LIMITS = {
    "x":       ("", ""),
    "probe_y": ("-10", "510"),
    "ephys_y": ("-50", "-20"),
}


class _SpikeDetectWorker(QObject):
    """Runs ``detect_spikes_for_trial`` over a list of Trials in a worker
    thread. Emits ``progress(i, n)`` after every trial and ``finished(
    n_succeeded, n_failed)`` once the loop completes (or stops early).

    Lives in its own QThread via ``moveToThread``. Both signals are
    delivered to the main thread via Qt's queued connection, so slot
    handlers can safely touch widgets.
    """
    progress = Signal(int, int)
    finished = Signal(int, int)

    def __init__(self, trials, params, channel: str = "voltage_1"):
        super().__init__()
        self.trials = trials
        self.params = params
        self.channel = channel
        self._stop = False

    def stop(self):
        """Request the loop exit before its next iteration."""
        self._stop = True

    def run(self):
        from .. import spike_detection as sds

        n = len(self.trials)
        failed = 0
        processed = 0
        for tr in self.trials:
            if self._stop:
                break
            processed += 1
            try:
                sds.detect_spikes_for_trial(
                    tr, self.params, channel=self.channel, save=True,
                )
            except Exception as e:
                failed += 1
                print(f"  trial {int(tr.params['trial'])} failed: {e}")
            self.progress.emit(processed, n)
        self.finished.emit(processed - failed, failed)


class TrialBrowser(QMainWindow):
    def __init__(self, table, *, start_trial=None, include_excluded=False,
                 parent=None):
        super().__init__(parent)
        self.table = table
        self.include_excluded = include_excluded
        self._trials = self._compute_trial_order()

        self.setWindowTitle(
            f"Trial Browser — {table.day}_F{table.fly}_C{table.cell} "
            f"[{getattr(table, 'protocol', '?')}]"
        )
        self.resize(1400, 800)

        self._current_trial_number = None
        self._probe_line = None
        self._pz_convention = 'unset'
        self._pz_prefer_meta = True   # sticky across trials
        self._pz_effective = None     # value the axis is drawn against
        self._ephys_lines: dict = {}
        self._metadata_labels: dict = {}
        self._channel_boxes: dict = {}
        self._probe_box = None
        self._overlays: dict = {name: cls() for name, cls in available_overlays().items()}
        self._overlay_boxes: dict = {}
        # Axis-limit QLineEdits — blank means autoscale. Keys: 'x', 'probe_y',
        # 'ephys_y'; values: (min_edit, max_edit).
        self._limit_edits: dict = {}
        # Target-region axhspans, cleared and redrawn per trial.
        self._target_artists: list = []
        # While True, all interactive widgets are disabled and shortcut
        # handlers (Left/Right/F5) no-op. Set via :meth:`_set_busy`.
        self._busy = False
        # Spike-detection background thread + worker, while a run is live.
        self._spike_thread = None
        self._spike_worker = None

        self._build_ui()
        self._wire_shortcuts()

        initial = start_trial if start_trial in self._trials else (
            self._trials[0] if self._trials else None
        )
        if initial is not None:
            self.show_trial(initial)

    # -----------------------------------------------------------------
    # Trial order
    # -----------------------------------------------------------------
    def _compute_trial_order(self) -> list[int]:
        df = self.table.df
        if "excluded" in df.columns and not self.include_excluded:
            mask = df["excluded"].astype(bool) == False  # noqa: E712
            return list(df.index[mask])
        return list(df.index)

    def _trial_object(self, trial_number: int):
        trial = self.table.df.loc[trial_number, "Trial"]
        if trial is None:
            raise RuntimeError(f"Trial {trial_number} failed to load (see Table log)")
        return trial

    # -----------------------------------------------------------------
    # UI construction
    # -----------------------------------------------------------------
    def _build_ui(self):
        splitter = QSplitter(Qt.Horizontal)

        # Left: trial list
        self.trial_list = QListWidget()
        self.trial_list.setMinimumWidth(100)
        self.trial_list.setMaximumWidth(160)
        for n in self._trials:
            self.trial_list.addItem(QListWidgetItem(str(n)))
        self.trial_list.currentItemChanged.connect(self._on_list_select)
        splitter.addWidget(self.trial_list)

        # Center: figure + toolbar + controls
        center = QWidget()
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(4, 4, 4, 4)
        self._build_figure()
        center_layout.addWidget(NavigationToolbar2QT(self.canvas, center))
        center_layout.addWidget(self.canvas, stretch=1)
        center_layout.addLayout(self._build_controls())
        splitter.addWidget(center)

        # Right: metadata + channel toggles
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(4, 4, 4, 4)
        right_layout.addWidget(self._build_metadata_panel())
        right_layout.addWidget(self._build_channel_panel())
        right_layout.addWidget(self._build_limits_panel())
        overlays_box = self._build_overlays_panel()
        if overlays_box is not None:
            right_layout.addWidget(overlays_box)
        right_layout.addWidget(self._build_spike_panel())
        right_layout.addStretch(1)
        right.setMinimumWidth(240)
        right.setMaximumWidth(340)
        splitter.addWidget(right)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        self.setCentralWidget(splitter)
        self.setStatusBar(QStatusBar(self))

    def _build_figure(self):
        self.fig = Figure(figsize=(9, 5), constrained_layout=True)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.ax_probe, self.ax_ephys = self.fig.subplots(2, 1, sharex=True)
        self.ax_probe.set_ylabel("probe")
        self.ax_ephys.set_ylabel("ephys")
        self.ax_ephys.set_xlabel("time (s)")

        (self._probe_line,) = self.ax_probe.plot([], [], lw=1)
        for ch in EPHYS_CHANNELS:
            (line,) = self.ax_ephys.plot([], [], lw=0.5, label=ch)
            self._ephys_lines[ch] = line
        self._ephys_legend = self.ax_ephys.legend(loc="upper right", fontsize=8)
        self._ephys_legend.set_visible(False)

    def _build_controls(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self.prev_btn = QPushButton("◀ Prev")
        self.next_btn = QPushButton("Next ▶")
        self.prev_btn.clicked.connect(self.on_prev)
        self.next_btn.clicked.connect(self.on_next)

        self.goto_edit = QLineEdit()
        self.goto_edit.setPlaceholderText("trial #")
        self.goto_edit.setValidator(QIntValidator(0, 99999, self))
        self.goto_edit.setMaximumWidth(80)
        self.goto_edit.returnPressed.connect(self._on_goto)

        self.export_btn = QPushButton("Export SVG")
        self.export_btn.clicked.connect(self._on_export)

        self.reload_btn = QPushButton("↻ Reload")
        self.reload_btn.setToolTip("Reload the table from disk and re-show the current trial (F5)")
        self.reload_btn.clicked.connect(self._on_reload)

        row.addWidget(self.prev_btn)
        row.addWidget(self.next_btn)
        row.addSpacing(12)
        row.addWidget(QLabel("go to:"))
        row.addWidget(self.goto_edit)
        row.addStretch(1)
        row.addWidget(self.reload_btn)
        row.addWidget(self.export_btn)
        return row

    def _build_metadata_panel(self) -> QGroupBox:
        box = QGroupBox("Metadata")
        form = QFormLayout(box)
        for field in METADATA_FIELDS:
            lbl = QLabel("—")
            lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
            self._metadata_labels[field] = lbl
            form.addRow(field, lbl)
        return box

    def _build_channel_panel(self) -> QGroupBox:
        box = QGroupBox("Channels")
        layout = QVBoxLayout(box)
        probe_row = QHBoxLayout()
        self._probe_box = QCheckBox("probe_position")
        self._probe_box.setChecked(True)
        self._probe_box.stateChanged.connect(self._on_channel_toggled)
        probe_row.addWidget(self._probe_box)
        # Which probeZero the probe axis is drawn against. Checked = the
        # corrected /meta value; unchecked = the acquisition /params value (or
        # the lo-target convention when there is no /params). It changes what
        # the axis MEANS -- an uncorrected zero is synthesised, so absolute
        # positions are not comparable across cells -- and nothing on the trace
        # itself shows it. Amber when the value in use IS the convention.
        self._pz_box = QCheckBox("probeZero")
        self._pz_box.setEnabled(False)
        self._pz_box.stateChanged.connect(self._on_probezero_toggled)
        probe_row.addWidget(self._pz_box)
        probe_row.addStretch(1)
        layout.addLayout(probe_row)
        for ch in EPHYS_CHANNELS:
            cb = QCheckBox(ch)
            cb.setChecked(ch in DEFAULT_ACTIVE)
            cb.stateChanged.connect(self._on_channel_toggled)
            self._channel_boxes[ch] = cb
            layout.addWidget(cb)
        self._legend_box = QCheckBox("show legend")
        self._legend_box.setChecked(False)
        self._legend_box.stateChanged.connect(self._on_legend_toggled)
        layout.addWidget(self._legend_box)
        return box

    def _build_limits_panel(self) -> QGroupBox:
        box = QGroupBox("Axis limits (blank = auto)")
        layout = QVBoxLayout(box)
        for key, label in [("x", "x"), ("probe_y", "probe y"), ("ephys_y", "ephys y")]:
            row = QHBoxLayout()
            lbl = QLabel(label)
            lbl.setFixedWidth(54)
            row.addWidget(lbl)
            pair = []
            defaults = DEFAULT_LIMITS.get(key, ("", ""))
            for side, default in zip(("min", "max"), defaults):
                e = QLineEdit()
                e.setPlaceholderText(side)
                e.setText(default)
                e.setValidator(QDoubleValidator(self))
                e.setMaximumWidth(70)
                e.returnPressed.connect(self._on_limits_changed)
                row.addWidget(e)
                pair.append(e)
            row.addStretch(1)
            self._limit_edits[key] = tuple(pair)
            layout.addLayout(row)
        return box

    def _build_overlays_panel(self) -> QGroupBox | None:
        if not self._overlays:
            return None
        box = QGroupBox("Overlays")
        layout = QVBoxLayout(box)
        for name in self._overlays:
            cb = QCheckBox(name)
            cb.setChecked(False)
            cb.stateChanged.connect(self._on_channel_toggled)
            self._overlay_boxes[name] = cb
            layout.addWidget(cb)

        # One kernel for every overlay that smooths. Rate and Vm must share it:
        # comparing them at different widths measures the wider kernel, not the
        # cell.
        row = QHBoxLayout()
        lbl = QLabel("kernel:")
        lbl.setFixedWidth(54)
        self.kernel_combo = QComboBox()
        for sigma in KERNEL_SIGMAS_S:
            self.kernel_combo.addItem(f"{sigma * 1e3:g} ms", sigma)
        self.kernel_combo.setCurrentIndex(KERNEL_SIGMAS_S.index(DEFAULT_KERNEL_S))
        self.kernel_combo.setToolTip(
            "Gaussian sd used by the Firing rate and Subthreshold Vm overlays.\n"
            "Both use the same value so they can be compared directly.\n"
            "Shaded bands mark where the kernel lacks full support (+/-3 sigma)."
        )
        self.kernel_combo.currentIndexChanged.connect(self._on_kernel_changed)
        row.addWidget(lbl)
        row.addWidget(self.kernel_combo)
        row.addStretch(1)
        layout.addLayout(row)
        return box

    def _current_kernel_s(self) -> float:
        combo = getattr(self, "kernel_combo", None)
        if combo is None:
            return DEFAULT_KERNEL_S
        value = combo.currentData()
        return DEFAULT_KERNEL_S if value is None else float(value)

    def _on_kernel_changed(self, _index):
        if self._current_trial_number is not None:
            self.show_trial(self._current_trial_number)

    def _build_spike_panel(self) -> QGroupBox:
        box = QGroupBox("Spike detection")
        layout = QVBoxLayout(box)

        row1 = QHBoxLayout()
        lbl = QLabel("tune N:")
        lbl.setFixedWidth(54)
        self.tune_n_edit = QLineEdit("5")
        self.tune_n_edit.setValidator(QIntValidator(1, 99999, self))
        self.tune_n_edit.setMaximumWidth(70)
        self.tune_n_edit.setToolTip(
            "Number of trials (starting at the current trial) to "
            "concatenate for tuning the detection params."
        )
        row1.addWidget(lbl)
        row1.addWidget(self.tune_n_edit)
        row1.addStretch(1)
        layout.addLayout(row1)

        row2 = QHBoxLayout()
        lbl2 = QLabel("range:")
        lbl2.setFixedWidth(54)
        self.detect_range_edit = QLineEdit()
        self.detect_range_edit.setPlaceholderText("this or '50-200'")
        self.detect_range_edit.setMaximumWidth(140)
        self.detect_range_edit.setToolTip(
            "Inclusive trial-number range to detect on, e.g. 50-200. "
            "Leave blank to detect on every trial in the table."
        )
        row2.addWidget(lbl2)
        row2.addWidget(self.detect_range_edit)
        row2.addStretch(1)
        layout.addLayout(row2)

        btn_row = QHBoxLayout()
        self.spike_tune_btn = QPushButton("Tune")
        self.spike_tune_btn.setToolTip(
            "Run the Filter / Template / Threshold GUIs on the next N "
            "trials (seeded by the current trial) and save the tuned "
            "params, but don't detect spikes yet — useful for sanity-"
            "checking before a full Detect run."
        )
        self.spike_tune_btn.clicked.connect(self._on_spike_tune)
        self.spike_detect_btn = QPushButton("Detect")
        self.spike_detect_btn.setToolTip(
            "Tune detection params on the next N trials (seeded by the "
            "current trial), then batch-detect spikes across the range. "
            "Reloads the table when finished."
        )
        self.spike_detect_btn.clicked.connect(self._on_spike_detect)
        self.spike_check_btn = QPushButton("Check")
        self.spike_check_btn.setToolTip(
            "Open the spot-check GUI on the current trial's detected "
            "spikes — accept / reject one at a time, then save the "
            "edited result back to the trial. Enabled only when the "
            "current trial has spikes and a template timestamp."
        )
        self.spike_check_btn.clicked.connect(self._on_spike_check)
        # Disabled by default; `_update_spikes_overlay_label` flips it
        # on whenever the current trial has spikes + a timestamp.
        self.spike_check_btn.setEnabled(False)
        btn_row.addWidget(self.spike_tune_btn)
        btn_row.addWidget(self.spike_detect_btn)
        btn_row.addWidget(self.spike_check_btn)
        layout.addLayout(btn_row)
        return box

    def _wire_shortcuts(self):
        QShortcut(QKeySequence(Qt.Key_Left), self, activated=self.on_prev)
        QShortcut(QKeySequence(Qt.Key_Right), self, activated=self.on_next)
        QShortcut(QKeySequence(Qt.Key_F5), self, activated=self._on_reload)

    # -----------------------------------------------------------------
    # Navigation
    # -----------------------------------------------------------------
    def show_trial(self, trial_number: int):
        if trial_number not in self._trials:
            self.statusBar().showMessage(
                f"Trial {trial_number} not in current filter", 4000
            )
            return
        self._current_trial_number = trial_number
        try:
            trial = self._trial_object(trial_number)
        except Exception as e:
            self.statusBar().showMessage(f"Failed to load trial {trial_number}: {e}", 6000)
            return

        # Resolve probeZero first: both the trace and the target band are
        # drawn against it, and they must agree or the band lands off the trace.
        self._update_probezero_box(trial)

        trial.draw_in_browser(
            ax_probe=self.ax_probe,
            ax_ephys=self.ax_ephys,
            probe_line=self._probe_line,
            ephys_lines=self._ephys_lines,
            show_probe=self._probe_box.isChecked(),
            active_channels=self._active_channels(),
            probe_zero=self._pz_effective,
        )
        self._draw_target_region(trial)
        self._apply_overlays(trial)
        self._apply_axis_limits()
        self.canvas.draw_idle()
        self._update_metadata(trial_number, trial)
        self._update_spikes_overlay_label(trial)
        self._sync_list_selection(trial_number)

    def _apply_axis_limits(self):
        """Override autoscale with any user-specified limits. Blank = autoscale."""
        def _parse(edit):
            txt = edit.text().strip()
            try:
                return float(txt) if txt else None
            except ValueError:
                return None

        x_lo, x_hi = (_parse(e) for e in self._limit_edits["x"])
        if x_lo is not None or x_hi is not None:
            cur = self.ax_ephys.get_xlim()
            self.ax_ephys.set_xlim(
                cur[0] if x_lo is None else x_lo,
                cur[1] if x_hi is None else x_hi,
            )  # sharex propagates to ax_probe

        for key, ax in (("probe_y", self.ax_probe), ("ephys_y", self.ax_ephys)):
            y_lo, y_hi = (_parse(e) for e in self._limit_edits[key])
            if y_lo is None and y_hi is None:
                continue
            cur = ax.get_ylim()
            ax.set_ylim(
                cur[0] if y_lo is None else y_lo,
                cur[1] if y_hi is None else y_hi,
            )

    def _on_limits_changed(self):
        # set_xlim/set_ylim disabled autoscale on the affected axis, so an
        # autoscale_view() alone is a no-op. Re-enable autoscale first so
        # clearing an edit box actually reverts that side.
        for ax in (self.ax_probe, self.ax_ephys):
            ax.set_autoscale_on(True)
            ax.relim()
            ax.autoscale_view()
        self._apply_axis_limits()
        self.canvas.draw_idle()

    def _on_legend_toggled(self, _state):
        if self._ephys_legend is not None:
            self._ephys_legend.set_visible(self._legend_box.isChecked())
            self.canvas.draw_idle()

    def _on_reload(self):
        """Re-instantiate the table from its parquet, rebuild the trial list,
        and re-show the current trial (or the closest one still in the list).

        Blocks the UI for the duration: the Reload button is disabled, a
        wait cursor is shown, and the canvas is force-redrawn synchronously
        before this method returns so the user sees the new data on screen
        before regaining control.

        Use this after editing trial files outside the browser — e.g. running
        spike detection from a script — to pick up the new metadata.
        """
        if self._busy:
            return
        from ..table import Table

        prev_trial = self._current_trial_number
        parquet_path = os.path.join(self.table.path, self.table.parquet)

        # Repaint the disabled button + status before the freeze so the user
        # sees that something is happening. Without the processEvents() the
        # whole UI just sits with the button still enabled until the heavy
        # work finishes and Qt finally processes the paint events.
        self.reload_btn.setEnabled(False)
        self.statusBar().showMessage("Reloading table from disk...")
        QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
        QApplication.processEvents()

        old_reload_btn_style = self.reload_btn.styleSheet()

        self.reload_btn.setEnabled(False)
        self.reload_btn.setStyleSheet(old_reload_btn_style + """
        QPushButton {
            color: gray;
        }
        """)
        try:
            try:
                self.table = Table.for_path(parquet_path)
            except Exception as e:
                self.statusBar().showMessage(f"Reload failed: {e}", 6000)
                return

            self._trials = self._compute_trial_order()
            self.trial_list.blockSignals(True)
            self.trial_list.clear()
            for n in self._trials:
                self.trial_list.addItem(QListWidgetItem(str(n)))
            self.trial_list.blockSignals(False)

            if prev_trial in self._trials:
                target = prev_trial
            elif self._trials:
                # Land on the next-larger remaining trial, or the last one.
                target = next((n for n in self._trials if n >= (prev_trial or -1)),
                              self._trials[-1])
            else:
                target = None

            self._current_trial_number = None
            if target is not None:
                self.show_trial(target)
                # show_trial ends in canvas.draw_idle(), which only schedules
                # a paint. Force it to complete now so the figure on screen
                # reflects the reloaded data before this method returns.
                self.canvas.draw()
            self.statusBar().showMessage("Reloaded table from disk", 4000)
        finally:
            QApplication.restoreOverrideCursor()
            self.reload_btn.setEnabled(True)
            self.reload_btn.setStyleSheet(old_reload_btn_style)


    def _resolve_tune_subset(self):
        """Return ``(seed_trial, tune_trials, tune_subset_nums)`` for the
        current seed + ``tune N`` setting, or None on a user-visible error
        (already surfaced in the status bar).
        """
        if self._current_trial_number is None:
            self.statusBar().showMessage("No current trial — load one first", 4000)
            return None
        try:
            n_tune = int(self.tune_n_edit.text() or "20")
        except ValueError:
            self.statusBar().showMessage("Invalid tune N", 4000)
            return None
        seed_n = self._current_trial_number
        try:
            seed_idx = self._trials.index(seed_n)
        except ValueError:
            self.statusBar().showMessage(
                f"Seed trial {seed_n} not in current filter", 4000)
            return None
        tune_subset_nums = self._trials[seed_idx:seed_idx + max(1, n_tune)]
        df = self.table.df
        seed_trial = df.at[seed_n, "Trial"]
        tune_trials = [df.at[t, "Trial"] for t in tune_subset_nums]
        return seed_trial, tune_trials, tune_subset_nums

    def _run_tuning(self):
        """Run the GUI tuning workflow on the current seed/tune-N selection.

        Returns the tuned ``SpikeDetectionParams`` on success, or None if
        the user aborted or the inputs were invalid (status bar already
        informs the user). The caller is responsible for toggling
        :meth:`_set_busy` — this method does not touch the busy flag,
        so the same call can be a standalone Tune action or a prelude
        to a long-running Detect run.
        """
        from .. import spike_detection as sds

        prep = self._resolve_tune_subset()
        if prep is None:
            return None
        seed_trial, tune_trials, tune_subset_nums = prep
        cell_id = self.table.dfc

        self.statusBar().showMessage(
            f"Tuning on trials {tune_subset_nums[0]}–{tune_subset_nums[-1]} "
            f"({len(tune_trials)} trials) — complete the GUI dialogs to continue"
        )
        QApplication.processEvents()
        try:
            return sds.tune_params_on_trials(
                seed_trial, tune_trials, cell_id=cell_id, save=True,
            )
        except Exception as e:
            self.statusBar().showMessage(f"Tuning aborted: {e}", 8000)
            return None

    def _on_spike_tune(self):
        """Run the GUI tuning workflow only — useful for sanity-checking
        params before committing to a full Detect run. Saves params to
        the per-cell JSON and the seed trial's /spikeDetectionParams.
        """
        self._set_busy(True)
        try:
            params = self._run_tuning()
        finally:
            self._set_busy(False)
        if params is None:
            return
        # Re-show the seed trial so the Spikes label updates: the fresh
        # timestamp turns it green, and the cleared /spikes (Tune calls
        # save_params_to_trial with invalidate_spikes=True) un-bolds it.
        if self._current_trial_number is not None:
            self.show_trial(self._current_trial_number)
        self.statusBar().showMessage(
            f"Tuned params saved — distance_threshold={params.distance_threshold:g}, "
            f"amplitude_threshold={params.amplitude_threshold:g}",
            8000,
        )

    def _on_spike_check(self):
        """Open the spot-check GUI on the current trial.

        Loads the trial's detected spikes via
        ``mapd.spike_detection.load_spikes_from_trial``, hands them to
        ``SpotCheckGUIQt`` for spike-by-spike y/n review, and writes
        the (possibly edited) result back to the trial .mat with
        ``spot_checked=True`` so subsequent loads know it was vetted.
        Refreshes the current trial after save so any new/dropped spikes
        show up in the overlay and the bold/green label updates.
        """
        from .. import spike_detection as sds
        from spikedetect.gui import SpotCheckGUIQt

        if self._current_trial_number is None:
            self.statusBar().showMessage("No current trial to spot-check", 4000)
            return
        trial = self.table.df.at[self._current_trial_number, "Trial"]
        try:
            result = sds.load_spikes_from_trial(trial)
        except Exception as e:
            self.statusBar().showMessage(f"Failed to load spikes: {e}", 6000)
            return
        if result is None:
            self.statusBar().showMessage(
                "No /spikeDetectionParams on this trial — run Detect first",
                6000,
            )
            return

        rec = sds.trial_to_recording(trial)
        self._set_busy(True)
        try:
            try:
                edited = SpotCheckGUIQt(rec, result).run()
            except Exception as e:
                self.statusBar().showMessage(f"Spot-check aborted: {e}", 6000)
                return
            try:
                sds.save_spikes_to_trial(trial, edited, spot_checked=True)
            except Exception as e:
                self.statusBar().showMessage(f"Save failed: {e}", 6000)
                return
        finally:
            self._set_busy(False)

        # Re-show this trial so the spike overlay (if on) and the
        # bold/green Spikes label pick up the edits.
        self.show_trial(self._current_trial_number)
        self.statusBar().showMessage(
            f"Spot-checked trial {self._current_trial_number}: "
            f"{edited.n_spikes} spikes saved",
            6000,
        )

    def _on_spike_detect(self):
        """Tune detection params on N trials starting from the current
        trial, then batch-detect across the chosen range (or all trials
        if the range box is empty), and reload the table when done.

        Tuning happens on the main thread (the GUI dialogs require it),
        but the per-trial detection loop is offloaded to a QThread so
        the status bar updates live and the rest of the UI stays
        responsive during the run.
        """
        # Parse detection range up front so an invalid range fails fast
        # (before we open any GUIs).
        range_text = self.detect_range_edit.text().strip()
        if range_text:
            try:
                lo_str, hi_str = range_text.split("-", 1)
                lo, hi = int(lo_str), int(hi_str)
            except ValueError:
                self.statusBar().showMessage(
                    "Invalid range (use 'lo-hi', e.g. 50-200)", 4000)
                return
            detect_subset_nums = [t for t in self._trials if lo <= t <= hi]
        else:
            # detect_subset_nums = list(self._trials)
            detect_subset_nums = [self._current_trial_number]
        if not detect_subset_nums:
            self.statusBar().showMessage("No trials in detection range", 4000)
            return

        # Stays busy from here through `_on_spike_finished`. Tuning
        # opens its own modal dialogs; the worker thread runs after.
        self._set_busy(True)
        params = self._run_tuning()
        if params is None:
            self._set_busy(False)
            return

        df = self.table.df
        detect_trials = [df.at[t, "Trial"] for t in detect_subset_nums]

        # Spin up a worker thread for the per-trial loop. Holding refs to
        # both `_spike_thread` and `_spike_worker` is required — letting
        # either go out of scope before the run finishes would crash Qt.
        self._spike_thread = QThread(self)
        self._spike_worker = _SpikeDetectWorker(detect_trials, params)
        self._spike_worker.moveToThread(self._spike_thread)

        self._spike_thread.started.connect(self._spike_worker.run)
        self._spike_worker.progress.connect(self._on_spike_progress)
        self._spike_worker.finished.connect(self._on_spike_finished)
        # Tear down once finished — quit the loop, then schedule deletion.
        self._spike_worker.finished.connect(self._spike_thread.quit)
        self._spike_thread.finished.connect(self._spike_worker.deleteLater)
        self._spike_thread.finished.connect(self._spike_thread.deleteLater)

        n = len(detect_trials)
        self.statusBar().showMessage(f"Detecting spikes on {n} trials...")
        self._spike_thread.start()

    def _on_spike_progress(self, i: int, n: int):
        # Updating the status bar on every trial is cheap; let it scroll.
        self.statusBar().showMessage(f"Detecting spikes... {i}/{n}")

    def _on_spike_finished(self, n_done: int, n_failed: int):
        # Drop our refs so the QThread/worker can be garbage-collected
        # after deleteLater fires.
        self._spike_worker = None
        self._spike_thread = None
        # Clear busy before reload — `_on_reload` itself bails when busy.
        self._set_busy(False)
        self._on_reload()
        msg = f"Spike detection done: {n_done} trials"
        if n_failed:
            msg += f" ({n_failed} failed)"
        self.statusBar().showMessage(msg, 8000)

    def _on_export(self):
        if self._current_trial_number is None:
            self.statusBar().showMessage("No trial loaded — nothing to export", 4000)
            return
        out_dir = Path(EXPORT_FOLDER)
        out_dir.mkdir(parents=True, exist_ok=True)
        dfc = f"{self.table.day}_F{self.table.fly}_C{self.table.cell}"
        path = out_dir / f"{dfc}_trial{self._current_trial_number}.svg"
        try:
            self.fig.savefig(path, format="svg", bbox_inches="tight")
        except Exception as e:
            self.statusBar().showMessage(f"Export failed: {e}", 6000)
            return
        self.statusBar().showMessage(f"Saved {path}", 5000)

    def _draw_target_region(self, trial):
        """Shade this trial's target on ax_probe using *its own* pyasState,
        pyasXPosition and pyasWidth — not the table-wide values from
        ``table.targets``.

        Trial-level ``pyasXPosition``/``pyasWidth``/``probeZero`` are raw
        HDF5 values (no pre-subtraction). The browser's probe y-axis is
        ``-(raw_probe - probeZero)``, so the region spans
        ``[probeZero - (pyasXPosition + pyasWidth), probeZero - pyasXPosition]``.

        No target is drawn for rest/probe trials (``pyasState == 'no_state'``)
        or when the required meta is missing.
        """
        # Clear any previously-drawn target.
        for art in self._target_artists:
            try:
                art.remove()
            except (ValueError, NotImplementedError, AttributeError):
                pass
        self._target_artists.clear()

        state = getattr(trial, "pyasState", None)
        if not state or state == "no_state":
            return
        try:
            px = float(trial.pyasXPosition)
            pw = float(trial.pyasWidth)
            # the same zero the trace was drawn against, never trial.probeZero
            pz = self._pz_effective
            pz = float(getattr(trial, "probeZero", 0) or 0) if pz is None else float(pz)
        except (AttributeError, TypeError, ValueError):
            return

        y_low = pz - (px + pw)
        y_high = pz - px
        color = (0.6, 0.6, 1.0) if state == "hi" else (0.8, 0.8, 0.8)
        span = self.ax_probe.axhspan(
            y_low, y_high, color=color, alpha=0.3, zorder=0,
            label=f"{state} target",
        )
        # Don't let the target band clamp probe y-autoscale.
        span.sticky_edges.y[:] = []
        self._target_artists.append(span)

    def _table_blank_window(self):
        """The cell's spike blanking window, measured once per session.

        Measured from an STA pooled over several trials and pushed to the Vm
        overlay, because that overlay only ever sees one trial and a single-trial
        STA measures a noticeably wider window than the truth — which would blank
        (and interpolate) far more of the trace than necessary.
        """
        if getattr(self, "_blank_window", None) is None:
            from .. import ephys as _ephys
            try:
                _, blank, _ = _ephys.blank_window_from_trials(self.table.df["Trial"])
                self._blank_window = blank
            except Exception:
                self._blank_window = False   # measured and failed; don't retry
        return self._blank_window or None

    def _apply_overlays(self, trial):
        axes = {"probe": self.ax_probe, "ephys": self.ax_ephys, "fig": self.fig}
        sigma_s = self._current_kernel_s()
        blank = None
        vm_box = self._overlay_boxes.get("Subthreshold Vm")
        if vm_box is not None and vm_box.isChecked():
            blank = self._table_blank_window()
        for name, overlay in self._overlays.items():
            overlay.clear()
            cb = self._overlay_boxes.get(name)
            if cb is not None and cb.isChecked():
                try:
                    # Pushed to every overlay; those that don't smooth just store
                    # it (see Overlay.set_params).
                    overlay.set_params(sigma_s=sigma_s)
                    if blank is not None and name == "Subthreshold Vm":
                        overlay.set_params(blank=blank)
                    overlay.draw(trial, axes)
                except Exception as e:
                    self.statusBar().showMessage(
                        f"Overlay {name!r} failed: {e}", 6000
                    )

    def _update_spikes_overlay_label(self, trial):
        """Style the Spikes-overlay checkbox to reflect this trial's state,
        and gate the Check button on the same load.

        Bold  = ``/spikes`` is non-empty (detection has been run).
        Green = ``/spikeDetectionParams/templateUpdatedAt`` is < 24h old.

        The two are independent: green-but-not-bold means the params are
        fresh from a recent Tune but the trial hasn't had Detect run yet,
        which is the case the user wants to spot at a glance.

        The Check button is enabled only when there are spikes AND the
        params carry a timestamp — without spikes there's nothing to
        spot-check, and an absent timestamp means the params predate
        the timestamped tuning workflow and we don't know what
        produced these spikes.
        """
        from datetime import datetime, timedelta
        from .. import spike_detection as sds

        has_spikes = False
        has_timestamp = False
        is_fresh = False
        ts = None
        try:
            result = sds.load_spikes_from_trial(trial)
        except Exception:
            result = None
        if result is not None:
            has_spikes = result.n_spikes > 0
            ts = getattr(result.params, "template_updated_at", None)
            if ts is not None:
                has_timestamp = True
                is_fresh = (datetime.now() - ts) < timedelta(hours=24)

        cb = self._overlay_boxes.get("Spikes")
        if cb is not None:
            label = "Spikes"
            if ts is not None:
                label = f"Spikes ({ts.strftime('%Y-%m-%d')})"
            cb.setText(label)
            rules = []
            if has_spikes:
                rules.append("font-weight: bold")
            if is_fresh:
                rules.append("color: green")
            # ``QCheckBox { ... }`` selector keeps the styling on the
            # label only and survives Qt's hover/focus repaints; bare
            # property rules sometimes get dropped on state transitions.
            cb.setStyleSheet(
                f"QCheckBox {{ {'; '.join(rules)} }}" if rules else ""
            )

        if hasattr(self, "spike_check_btn"):
            self.spike_check_btn.setEnabled(has_spikes and has_timestamp)

        self._color_list_item_for_freshness(self._current_trial_number, is_fresh)

    def _color_list_item_for_freshness(self, trial_number, is_fresh: bool):
        """Tint the list item for ``trial_number`` green if its spike template
        is fresh (< 24h), otherwise restore the default text color. Called
        from :meth:`_update_spikes_overlay_label` so the list reveals its
        coloring lazily — only trials the user has actually visited get
        evaluated.
        """
        if trial_number is None:
            return
        for i in range(self.trial_list.count()):
            item = self.trial_list.item(i)
            if item.text() != str(trial_number):
                continue
            if is_fresh:
                item.setForeground(QBrush(QColor("green")))
            else:
                item.setData(Qt.ForegroundRole, None)
            return

    def _set_busy(self, busy: bool):
        """Disable the central widget (left list, figure, sidebar — every
        interactive control) while a long-running task is underway, and
        flip a ``_busy`` flag the shortcut handlers consult so Left/Right/
        F5 also no-op. The status bar stays alive for progress messages.
        """
        self._busy = busy
        self.centralWidget().setEnabled(not busy)

    def on_prev(self):
        if self._busy:
            return
        i = self._current_index()
        if i is None or i == 0:
            return
        self.show_trial(self._trials[i - 1])

    def on_next(self):
        if self._busy:
            return
        i = self._current_index()
        if i is None or i == len(self._trials) - 1:
            return
        self.show_trial(self._trials[i + 1])

    def _on_goto(self):
        txt = self.goto_edit.text().strip()
        if not txt:
            return
        try:
            n = int(txt)
        except ValueError:
            return
        self.show_trial(n)
        self.goto_edit.clear()

    def _on_list_select(self, current, _previous):
        if current is None:
            return
        try:
            n = int(current.text())
        except ValueError:
            return
        if n != self._current_trial_number:
            self.show_trial(n)

    def _on_channel_toggled(self, _state):
        if self._current_trial_number is not None:
            self.show_trial(self._current_trial_number)

    # -----------------------------------------------------------------
    # Small helpers
    # -----------------------------------------------------------------
    def _current_index(self) -> int | None:
        if self._current_trial_number is None:
            return None
        try:
            return self._trials.index(self._current_trial_number)
        except ValueError:
            return None

    def _active_channels(self) -> list[str]:
        return [ch for ch, cb in self._channel_boxes.items() if cb.isChecked()]

    def _sync_list_selection(self, trial_number: int):
        for i in range(self.trial_list.count()):
            if self.trial_list.item(i).text() == str(trial_number):
                if self.trial_list.currentRow() != i:
                    self.trial_list.blockSignals(True)
                    self.trial_list.setCurrentRow(i)
                    self.trial_list.blockSignals(False)
                return

    def _update_metadata(self, trial_number: int, trial):
        df = self.table.df
        for field in METADATA_FIELDS:
            value: object
            if field == "trial":
                value = trial_number
            elif field in ("ephys_status", "ephys_note"):
                # Ask the Trial, not the df: the property resolves an unannotated
                # trial to its default ('good'), where the df column holds NaN and
                # would display as "nan".
                value = getattr(trial, field, "—")
            elif field in df.columns:
                value = df.at[trial_number, field]
            else:
                value = getattr(trial, field, "—")
            lbl = self._metadata_labels[field]
            lbl.setText(_format_metadata_value(value))
            if field == "ephys_status":
                color = EPHYS_STATUS_COLORS.get(str(value))
                lbl.setStyleSheet(
                    f"color: {color}; font-weight: bold;" if color else "")

        self.statusBar().showMessage(
            f"trial {trial_number}   "
            f"{self._current_index() + 1}/{len(self._trials)}"
        )

    def _probezero_convention(self):
        """``max(pyasXPosition) + offset`` over the whole cell, or None.

        A cell-level quantity: pyasXPosition moves between trials as the target
        moves, so a single trial's value is not the convention.
        """
        if getattr(self, '_pz_convention', 'unset') != 'unset':
            return self._pz_convention
        self._pz_convention = None
        try:
            col = self.table.df['pyasXPosition']
            v = float(np.nanmax(col.to_numpy(dtype=float)))
            if np.isfinite(v):
                self._pz_convention = v + PROBEZERO_CONV_OFFSET
        except (KeyError, AttributeError, TypeError, ValueError):
            pass
        return self._pz_convention

    def _on_probezero_toggled(self, _state) -> None:
        """User chose which probeZero to draw against; remember and redraw."""
        self._pz_prefer_meta = self._pz_box.isChecked()
        if self._current_trial_number is not None:
            self.show_trial(self._current_trial_number)

    def _update_probezero_box(self, trial) -> None:
        """Set the probeZero checkbox for this trial and resolve the value.

        Checked  -> the corrected ``/meta`` value.
        Unchecked-> the acquisition ``/params`` value, or the lo-target
                    convention when the trial carries no ``/params``.

        The box is only enabled when ``/meta`` has something to switch to. The
        user's choice is sticky across trials, so a cell can be stepped through
        under one convention; it is forced off wherever ``/meta`` is absent.
        """
        try:
            prov = probezero_provenance(
                trial, convention=self._probezero_convention())
        except Exception as exc:            # never let this break browsing
            self._pz_box.setEnabled(False)
            self._pz_box.setToolTip(f"probeZero check failed: {exc}")
            self._pz_effective = None
            return

        meta_v = prov.get("meta_value", np.nan)
        params_v = prov.get("params_value", np.nan)
        conv = prov.get("convention", np.nan)
        has_meta = bool(prov.get("corrected")) and np.isfinite(meta_v)

        self._pz_box.blockSignals(True)
        self._pz_box.setEnabled(has_meta)
        self._pz_box.setChecked(has_meta and self._pz_prefer_meta)
        self._pz_box.blockSignals(False)

        # A trial can carry NO stored probeZero at all -- neither /meta nor
        # /params. Falling back to the convention draws a sensible-looking
        # trace, which is exactly the wrong thing for a diagnostic to do
        # quietly: Trial.probeZero returns 0 for that trial, 0 is finite so
        # per_frame_records' NaN guard never fires, and every OTHER analysis
        # puts x at -probe_position, hundreds of px out. Draw it, but say so.
        no_stored = not (np.isfinite(meta_v) or np.isfinite(params_v))
        if self._pz_box.isChecked():
            value, src = float(meta_v), "/meta"
        elif np.isfinite(params_v):
            value, src = float(params_v), "/params"
        elif np.isfinite(conv):
            value, src = float(conv), "lo-target convention"
        else:
            value, src = None, "missing"
        self._pz_effective = value

        tol = PROBEZERO_CORRECTED_TOL
        is_conv = (value is not None and np.isfinite(conv)
                   and abs(value - conv) <= tol)
        if no_stored:
            self._pz_box.setText(
                "probeZero NOT STORED"
                + (f" - drawn at {value:.1f}" if value is not None else ""))
            self._pz_box.setStyleSheet("color: #d62728; font-weight: bold;")
        elif value is None:
            self._pz_box.setText("probeZero MISSING")
            self._pz_box.setStyleSheet("color: #d62728; font-weight: bold;")
        else:
            self._pz_box.setText(f"probeZero {value:.1f} ({src})")
            self._pz_box.setStyleSheet(
                "color: #b8860b;" if is_conv               # synthesised
                else ("color: #2ca02c;" if src == "/meta"  # corrected
                      else "color: #d62728;"))             # acquisition value
        tip = [prov.get("reason", "")]
        for v, lab in ((meta_v, "/meta"), (params_v, "/params"),
                       (conv, "lo-target convention")):
            if v is not None and np.isfinite(v):
                tip.append(f"{lab}: {v:.1f}")
        if no_stored:
            tip.append("RED: this trial stores no probeZero in /meta OR "
                       "/params. The browser is drawing it at the convention, "
                       "but Trial.probeZero returns 0, so every other analysis "
                       "puts x at -probe_position. Fix the trial, do not trust "
                       "this trace elsewhere.")
        elif is_conv:
            tip.append("AMBER: the value in use equals the convention, so it "
                       "is synthesised from the target, not a measured stop.")
        if not has_meta:
            tip.append("Disabled: no /meta value to switch to.")
        self._pz_box.setToolTip(("\n").join(t for t in tip if t))


def _format_metadata_value(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, (np.ndarray,)):
        if value.size == 1:
            value = value.item()
        else:
            return f"array{tuple(value.shape)}"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def launch(table, *, start_trial=None, include_excluded=False) -> "TrialBrowser":
    """Create (or reuse) a QApplication, show a TrialBrowser, return the window.

    Caller is responsible for ``QApplication.instance().exec()`` — this lets
    callers launch multiple browsers in one app or drive the event loop from
    a REPL.
    """
    app = QApplication.instance() or QApplication([])
    win = TrialBrowser(table, start_trial=start_trial,
                       include_excluded=include_excluded)
    win.show()
    return win
