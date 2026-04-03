# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

Analysis scripts for *Drosophila* operant conditioning experiments. The experiment records fly leg probe movements in response to an aversive stimulus (heat or optogenetic). Data is acquired with a MATLAB system (FlySoundAcquisition) and analyzed here in Python.

## Data Layout

Raw data lives at `D:\Data` or `C:\Users\Tony\Data` (checked automatically by `mapd/helpers.py:default_data_directory()`).

Within the data root:
```
YYMMDD/                      # recording day
  YYMMDD_F{fly}_C{cell}/     # per-fly-per-cell directory
    YYMMDD_Table_F{n}_C{n}.parquet      # trial metadata table
    YYMMDD_Raw_F{n}_C{n}_{trial}.mat    # individual HDF5/MAT trial files
    Acquisition_YYMMDD_F{n}_C{n}.mat   # acquisition metadata (genotype, etc.)
    notes*.txt
Sinqs/                        # saved Sinq pickle files
```

## `mapd` Package Architecture

The `mapd` package is the analysis library. Three main classes:

### `Trial` (`mapd/trial.py`)
Wraps a single `.mat` trial file (HDF5 format). Key properties:
- `probe_position`, `time`, `trialtime` — raw time-series data
- `probeZero` — baseline probe position (read from HDF5 `/meta`)
- `pyasState` — stimulus state (`'hi'`/`'lo'`/`'no_state'`)
- `as_outcome` — trial outcome category (see `_outcomes_dict` in `table.py`)
- `excluded`, `downsample_probe` — quality/processing flags
- `write_string_if_changed()`, `write_scalar_if_changed()` — write metadata back to HDF5

### `Table` (`mapd/table.py`)
Wraps the parquet file for one day/fly/cell. `self.df` is the trial-indexed DataFrame.
- **Initializing:** `T = Table("YYMMDD_Table_F1_C1.parquet")` or any path containing the `YYMMDD_F#_C#` pattern
- Calls `get_trials()` (loads all Trial objects), `exclude_trials()`, `_bootstrap_meta_columns()`, `get_target_positions()` at init
- **Method injection:** `plot_*` methods come from `table_plotters.py`, `make_*` from `table_movie_maker.py`, `export_*` from `table_export_methods.py`, scalar computations from `table_scalars.py` (registered as `compute_*` → `*`)
- `extract_trial_properties(prop_list)` — pull Trial attributes into `self.df` columns
- `assign_column_value(column, value, trial_min, trial_max)` — annotate trials (e.g., VNC cut trial ranges)
- `write_column_to_trial_files(column_name)` — persist a column back to HDF5 trial files
- `probe_positions_df()` / `probe_position_distribution()` — probe kinematics analysis
- `find_successful_trials()` — computes `soft_success`, `hard_success`, `success` columns

### `Sinq` (`mapd/sinq.py`)
A persistent registry of Table objects across experiments (saved as pickle in `{data_root}/Sinqs/`).
- **Loading:** `sinq = Sinq(sinqname='all_Tables')` — loads or creates `all_Tables.pkl`
- `sinq.add_table(table_or_path)` — registers a Table, computes standard attributes, saves
- `sinq.restore_table(dayflycell)` — reconstructs a Table from its parquet path
- `sinq.sync()` — fills in missing computed values across all rows
- `sinq.merge_sinq(other)` — combines two Sinqs
- Indexed by `dayflycell` strings like `'241121_F1_C1'`
- `Table` objects are dropped before pickling (`_prepare_for_export()`); `restore_table()` reloads them
- `NotesMixin` provides structured `notes` column with `text`, `tags`, `author`, `timestamp` fields

### Supporting modules
- `mapd/sinq_builders.py` — `build_composite_sinq()`, `subset_sinq()` factory functions
- `mapd/sentinels.py` — `MISSING` sentinel and `is_missing_like()` helper
- `mapd/quickScanner.py` — `QuickScanner` for scanning raw `.mat` files directly

## Trial Outcome Categories

`as_outcome` values (ordered categorical): `no_as_no_mv`, `no_as_mv`, `as_off`, `as_off_late`, `timeout_fail`, `timeout`, `probe`, `rest`, `info`

## Categorical Columns

Several DataFrame columns use ordered `pd.CategoricalDtype`:
- `as_outcome`, `pyasState` (`lo`/`hi`/`no_state`), `vnc_status` (`intact`/`cut`), `filtercube_status`, `fiberLED`

## Notebooks

Figure-generating notebooks follow naming conventions:
- `sinq_Figure*.ipynb` — figures based on the Sinq aggregate view
- `op_cond_Figure*.ipynb` — operant conditioning figure panels
- `RPPR_figures.ipynb` — reporting figures
- `archive/` — deprecated earlier versions

`figpanels/` receives saved figure outputs (PNG/PDF/SVG).

## Working in Notebooks

The typical workflow:
```python
import mapd
T = mapd.Table("YYMMDD_Table_F1_C1.parquet")
T.extract_trial_properties()
T.plot_probe_distribution(binwidth=2, bin_min=-400, bin_max=100)

sinq = mapd.Sinq(sinqname='all_Tables')
sinq.restore_table('241121_F1_C1')
```
