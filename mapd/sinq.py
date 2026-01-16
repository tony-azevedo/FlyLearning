import os
import numpy as np
import pandas as pd
from .helpers import get_day_fly_cell, get_file, default_data_directory
from mapd.table import Table
import warnings
from collections import Counter


from datetime import datetime
from typing import Dict, List, Optional, Union


def _reorder_columns_helper(df):
    preferred = ["Table", "parquet", "genotype", "notes",'note_text',"tags"]
    existing = [c for c in preferred if c in df]
    rest = [c for c in df if c not in existing]

    return df[existing + rest]


class NotesMixin:
    NOTES_COLUMN = "notes"

    # ---------- internal helpers ----------

    def _ensure_notes_column(self):
        if self.df is None:
            return
        if self.NOTES_COLUMN not in self.df.columns:
            self.df[self.NOTES_COLUMN] = None
            self.save()

    def _normalize_note(self, note: Union[str, Dict, None]) -> Dict:
        """
        Normalize any legacy or partial note into a structured dict.
        """
        if note is None or (isinstance(note, float) and pd.isna(note)):
            return self._empty_note()

        if isinstance(note, str):
            return {
                "text": note,
                "tags": [],
                "author": None,
                "timestamp": None,
            }

        if isinstance(note, dict):
            return {
                "text": note.get("text", ""),
                "tags": list(note.get("tags", [])),
                "author": note.get("author"),
                "timestamp": note.get("timestamp"),
            }

        raise TypeError(f"Unsupported note type: {type(note)}")

    def _empty_note(self) -> Dict:
        return {
            "text": "",
            "tags": [],
            "author": 'TA',
            "timestamp": self._now(),
        }

    def _now(self) -> str:
        return datetime.now().isoformat(timespec="seconds")
    

    # ---------- public API ----------


    def set_note(
        self,
        dayflycell: str,
        text: str,
        *,
        tags: Optional[List[str]] = None,
        author: Optional[str] = 'Tony',
        append: bool = False,
        timestamp: bool = True,
    ):
        """
        Create or update a structured note.
        """
        self._ensure_notes_column()

        current = self._normalize_note(
            self.df.at[dayflycell, self.NOTES_COLUMN]
        )

        if append and current["text"]:
            current["text"] += "\n" + text
        else:
            current["text"] = text

        if tags:
            current["tags"] = sorted(set(current["tags"]).union(tags))

        if author:
            current["author"] = author

        if timestamp:
            current["timestamp"] = self._now()

        self.df.at[dayflycell, self.NOTES_COLUMN] = current
        self.save()


    def datetime_from_dayflycell(dayflycell: str) -> datetime:
        """
        Convert a dayflycell string (YYMMDD_Fx_Cy) into a datetime.date.

        Example
        -------
        >>> datetime_from_dayflycell("251107_F1_C1")
        datetime.datetime(2025, 11, 7, 0, 0)
        """
        try:
            date_part = dayflycell.split("_")[0]
            return datetime.strptime(date_part, "%y%m%d")
        except Exception as e:
            raise ValueError(f"Invalid dayflycell format: {dayflycell}") from e


    def get_note(self, dayflycell: str) -> Dict:
        """
        Always returns a structured note dict.
        """
        self._ensure_notes_column()
        return self._normalize_note(
            self.df.at[dayflycell, self.NOTES_COLUMN]
        )

    def clear_note(self, dayflycell: str):
        self._ensure_notes_column()
        self.df.at[dayflycell, self.NOTES_COLUMN] = self._empty_note()
        self.save()

    # ---------- plotting helpers ----------

    def notes_to_hovertext(self) -> pd.Series:
        """
        Returns a Series of strings suitable for hover labels.
        """
        self._ensure_notes_column()

        def fmt(note):
            n = self._normalize_note(note)
            parts = [n["text"]] if n["text"] else []
            if n["tags"]:
                parts.append(f"tags: {', '.join(n['tags'])}")
            if n["author"]:
                parts.append(f"by {n['author']}")
            return "<br>".join(parts)

        return self.df[self.NOTES_COLUMN].apply(fmt)

    def has_tag(self, tag: str) -> pd.Series:
        """
        Boolean mask: rows that contain a given tag.
        """
        self._ensure_notes_column()
        return self.df[self.NOTES_COLUMN].apply(
            lambda n: tag in self._normalize_note(n)["tags"]
        )


    def all_tags(self) -> list[str]:
        """
        Return a sorted list of all unique tags across all notes.
        """
        self._ensure_notes_column()

        tags = set()
        for note in self.df[self.NOTES_COLUMN]:
            n = self._normalize_note(note)
            tags.update(n.get("tags", []))

        return sorted(tags)


    def tag_counts(self) -> dict[str, int]:
        """
        Return a dict mapping tag -> number of rows containing that tag.
        """
        self._ensure_notes_column()

        counter = Counter()
        for note in self.df[self.NOTES_COLUMN]:
            n = self._normalize_note(note)
            counter.update(set(n.get("tags", [])))  # set avoids double-counting per row

        return dict(counter)


    def rows_with_tags(
        self,
        tags,
        *,
        mode: str = "all",
    ) -> list[str]:
        """
        Return list of indices (dayflycells) whose notes contain the given tag(s).

        Parameters
        ----------
        tags : str or iterable of str
            Tag or tags to search for.
        mode : {"all", "any"}
            - "all": row must contain all tags
            - "any": row must contain at least one tag
        """
        self._ensure_notes_column()

        # Normalize input tags
        if isinstance(tags, str):
            tags = {tags}
        else:
            tags = set(tags)

        if mode not in {"all", "any"}:
            raise ValueError("mode must be 'all' or 'any'")

        matches = []

        for idx, note in self.df[self.NOTES_COLUMN].items():
            row_tags = set(self._normalize_note(note).get("tags", []))

            if mode == "all" and tags.issubset(row_tags):
                matches.append(idx)
            elif mode == "any" and tags & row_tags:
                matches.append(idx)

        return matches


class Sinq(NotesMixin):
    def __init__(self, fresh: bool = False, **kwargs):
        self.sinqname = kwargs.get('sinqname', 'all_Tables')
        self.file_location = os.path.join(default_data_directory(),'Sinqs')
        os.makedirs(self.file_location, exist_ok=True)
        self.file_name = os.path.join(self.file_location, self.sinqname + '.pkl')
        self.T = None
        self.load_sinq(fresh=fresh)
        self._sources = None


    def load_sinq(self, fresh: bool = False):
        """Load the DataFrame from the file."""
        if fresh:
            self.df = None
            print(f"File {self.file_name} exists but starting fresh.")
            return
        if os.path.exists(self.file_name):
            self.df = pd.read_pickle(self.file_name)
            self._ensure_notes_column()
            self.reorder_columns()
        else:
            self.df = None
            print(f"File {self.file_name} does not exist. Starting with an empty DataFrame.")


    def reorder_columns(self):
        """
        Reorder columns for human-friendly display.
        """
        if self.df is None:
            return

        self.df = _reorder_columns_helper(self.df)


    def save(self):
        """Save the DataFrame to the file. remove tables first"""

        new_file_name = os.path.join(self.file_location, self.sinqname + '.pkl')
        if not new_file_name == self.file_name:
            warnings.warn(f"Saving as {new_file_name} (Old file: {self.file_name})")
            self.file_name = new_file_name

        if self.df is not None:
            stripped_df = self._prepare_for_export()
            stripped_df.to_pickle(self.file_name)
        else:
            raise ValueError("DataFrame is empty. Nothing to save.")


    def _prepare_for_export(self):
        """Prepare the DataFrame for export by removing Table objects."""
        stripped_df = self.df.copy() if self.df is not None else pd.DataFrame()
        if self.df is not None:
            # Remove Table objects from the DataFrame
            stripped_df['Table'] = stripped_df['Table'].apply(lambda x: None )
            return stripped_df
    
    
    def drop_tables(self):
        """Save the DataFrame to the file. remove tables first"""
        if self.df is not None:
            stripped_df = self._prepare_for_export()
            self.df = stripped_df
        else:
            raise ValueError("DataFrame is empty. Nothing to save.")

    @property
    def columns(self):
        """Return the columns of the DataFrame."""
        if self.df is not None:
            return self.df.columns.tolist()
        else:
            return []
    

    @property
    def dayflycells(self):
        """Return the index values (dayflycells) of the DataFrame."""
        if self.df is not None:
            return self.df.index.tolist()
        else:
            return []

        
    def add_table(self, table=None, overwrite=True):
        """
        Add a table to the Sinq object, either from a Table object or a string path.
        This is the main function for adding or updating a table.
        """
        if isinstance(table, Table):
            self.T = table
            dayflycell = self.T.flycelldir
        elif isinstance(table, str):
            day, fly, cell = get_day_fly_cell(table)
            dayflycell = f'{day}_F{fly}_C{cell}'
            if (self.df is not None) and (dayflycell in self.df.index):
                # Check if values exist before adding
                if not self.df.loc[dayflycell].apply(lambda x: isinstance(x, float) and np.isnan(x)).any():
                    print(f"Table and all computations for {dayflycell} already exists.")
                    if not overwrite:
                        print('Overwrite = {}. Skipping addition.'.format(overwrite))
                        return
                    else:
                        print('Overwrite: {}'.format(overwrite))
            self.T = Table(table)  # Create new table if it's a string
        else:
            raise ValueError("Data must be a Table or string. Cannot be none")

        # Use _add_row to handle the row addition or update
        row_data = {
            'parquet': table.parquet,
            'Table': table,
            'genotype': table.genotype
        }

        if self.df is None:
            self.df = pd.DataFrame([row_data], index=[dayflycell])
        elif not dayflycell in self.df.index or overwrite:
            # If the dayflycell exists and overwrite is True, update the row with row_data
            # Fill in the missing columns with existing values or NaN
            for column in self.df.columns:
                if column in row_data:
                    self.df.loc[dayflycell, column] = row_data[column]  # Assign new data
                else:
                    # If the column isn't in row_data, keep the existing value or set to NaN
                    print(column)
                    if not (isinstance(self.df.loc[dayflycell, column],str)) and np.isnan(self.df.loc[dayflycell, column]):
                        self.df.loc[dayflycell, column] = np.nan

        updated_row = self._add_row(self.df.loc[dayflycell].copy(), overwrite=overwrite)
        self.df.loc[dayflycell] = updated_row

        # After updating the row, save the Sinq object
        
        self.df = self.df.sort_index()
        self.save()
        return self.T


    def _add_row(self, row, table=None, overwrite=False):
        """
        Private function to add or update a row in the DataFrame based on the provided Table object.
        This function is called internally by add_table.
        """

        # Check if the row already exists and whether we should overwrite it
        if not row.apply(lambda x: isinstance(x, float) and np.isnan(x)).any():
            print(f"Table and all computations for {row.index} already exists.")
            if not overwrite:
                print('Overwrite = False. Skipping addition.')
                return row  # Return the existing row if no overwrite is needed

        if table is None:
            table = row['Table']

        # Update all computed attributes for the row
        table.extract_trial_properties()
        row['Table'] = table
        for attr in self.df.columns:
            if attr == self.NOTES_COLUMN:
                continue
            if (isinstance(row[attr], float) and not(np.isnan(row[attr]))) and not overwrite:
                print(f"Skipping {attr} for {row.name}: {row[attr]}")
                continue
            print(f"Adding {attr} for {row.name}")
            row, attr_list = self._get_table_attr(row,attr)

        # Sort the DataFrame to maintain order
        return row  # Return the updated row
    
    
    def delete_table(self, table=None, overwrite=True):
        """
        Add a table to the Sinq object, either from a Table object or a string path.
        This is the main function for adding or updating a table.
        """
        if isinstance(table, Table):
            self.T = table
            dayflycell = self.T.flycelldir
        elif isinstance(table, str):
            day, fly, cell = get_day_fly_cell(table)
            dayflycell = f'{day}_F{fly}_C{cell}'
            if (self.df is not None) and (dayflycell in self.df.index):
                # Check if values exist before adding
                if not self.df.loc[dayflycell].apply(lambda x: isinstance(x, float) and np.isnan(x)).any():
                    print(f"Table and all computations for {dayflycell} already exists.")
                    if not overwrite:
                        print('Overwrite = {}. Skipping addition.'.format(overwrite))
                        return
                    else:
                        print('Overwrite: {}'.format(overwrite))
            self.T = Table(table)  # Create new table if it's a string
        else:
            raise ValueError("Data must be a Table or string. Cannot be none")

        # Use _add_row to handle the row addition or update
        row_data = {
            'parquet': table.parquet,
            'Table': table,
            'genotype': table.genotype
        }

        if self.df is None:
            self.df = pd.DataFrame([row_data], index=[dayflycell])
        elif not dayflycell in self.df.index or overwrite:
            # If the dayflycell exists and overwrite is True, update the row with row_data
            # Fill in the missing columns with existing values or NaN
            for column in self.df.columns:
                if column in row_data:
                    self.df.loc[dayflycell, column] = row_data[column]  # Assign new data
                else:
                    # If the column isn't in row_data, keep the existing value or set to NaN
                    print(column)
                    if not (isinstance(self.df.loc[dayflycell, column],str)) and np.isnan(self.df.loc[dayflycell, column]):
                        self.df.loc[dayflycell, column] = np.nan

        updated_row = self._add_row(self.df.loc[dayflycell].copy(), overwrite=overwrite)
        self.df.loc[dayflycell] = updated_row

        # After updating the row, save the Sinq object
        
        self.df = self.df.sort_index()
        self.save()
        return self.T

    def add_column(self, column_name, overwrite=False):
        """
        Add a column to the DataFrame.
        If the DataFrame is None, initialize it first.
        Will not compute column for other rows.
        """

        if self.df is None:
            raise ValueError("DataFrame is not initialized. Call add_table() first.")
        
        if self.T is None:
            raise ValueError("No Table object is associated with this Sinq. Call add_table() or restore_table() first.")
        
        if column_name in self.df.columns and not overwrite:
            print(f"Column '{column_name}' already exists in the DataFrame. Call sinq.sync_column({column_name})")
            overwrite = all(self.df[column_name].isna())
            if not overwrite:
                return
        if column_name in self.df.columns and overwrite:
            print("Overwriting existing column.")
        
        day,fly,cell = get_day_fly_cell(self.T.parquet)
        dayflycell = f'{day}_F{fly}_C{cell}'

        # test if T has attribute
        if hasattr(self.T,column_name):
            self.df[column_name] = np.nan
            row, attr_list = self._get_table_attr(self.df.loc[dayflycell,:].copy(),column_name)
            self.df.loc[dayflycell] = row
            self.save()
            return self.df.loc[dayflycell,attr_list]
        else:
            raise AttributeError('Not sure Table has attribute {}'.format(column_name))
        

    def _get_table_attr(self,row,attr):
        """Call attr on the table"""

        if 'Table' in attr:
            return row, [attr]

        if 'outcome_fractions_' in attr:
            returned_attr = getattr(row['Table'], 'outcome_fractions')()
            for key, value in returned_attr.items():
                row[f"{'outcome_fractions'}_{key}"] = value
            return row, returned_attr.items()

        if callable(getattr(row['Table'], attr)):
            returned_attr = getattr(row['Table'], attr)()
        else:
            returned_attr = getattr(row['Table'], attr, np.nan)

        if pd.api.types.is_scalar(returned_attr):
            row[attr] = returned_attr
        
        return row, [attr]


    def sync(self, *, overwrite:bool=False) -> pd.Series:
        """
        Synchronize the current Sinq by filling in missing columns
        based on the existing Table objects in the DataFrame.
        """
        # Count total missing values
        df_no_table = self.df.drop(
            columns=['Table', self.NOTES_COLUMN],
            errors='ignore'
        )
        missing_count = df_no_table.isna().sum().sum()
        print('Syncing Sinq: filling in {} empyties - {}'.format(missing_count,self.__repr__()))
        
        
        if self.df.isna().sum().sum()==0 and not overwrite:
            print('Sinq is full, use overwrite=True to recompute')
            return
        
        def re_add_table_values(row):
            if not ((row.apply(lambda x: isinstance(x, float) and np.isnan(x))).any()) and (not overwrite):
                if row['Table'] is not None:
                    print('All values exist for {}'.format(row['Table']))
                else:
                    print('All values exist for {} except Table'.format(row['parquet']))
                return row  # Return row unmodified if no changes are needed

            if row['Table'] is not None:
                print('Adding all values for {}'.format(row['Table']))
                row = self._add_row(row, overwrite=overwrite)
            else:
                print('Restoring Table and all values for {}'.format(row['parquet']))
                table = self.restore_table(dayflycell=row.name)
                row = self._add_row(row, table=table, overwrite=overwrite)

            return row  # Ensure the row is returned after modification
        
        for dfc in self.df.index:
            print
            row = self.df.loc[dfc].copy()
            row = re_add_table_values(row)
            self.df.loc[dfc] = row
            # self.df = self.df.apply(re_add_table_values,axis=1)
            print(f'Saving sync {self.__repr__()}')
            self.save()
            self.drop_tables()
        return row


    def sync_column(self, column=None, overwrite=True):
        """
        Recalculate values for each row for a single column
        """
        if column is None or not column in self.df.columns:
            raise KeyError('No column {} to update'.format(column))
        
        if column == self.NOTES_COLUMN:
            raise ValueError("Notes column is metadata and cannot be synced.")

        # Count total missing values
        print('Syncing {}: filling in {} empyties'.format(column, self.df[column].isna().sum()))
        
        if self.df[column].isna().sum()==0 and not overwrite:
            print("Sinq.df['{}] is full, use overwrite=True to recompute".format(column))
            return
        
        def re_add_table_values(row: pd.Series):
            if row['Table'] is None:
                print('Restoring Table and all values for {}'.format(row['parquet']))
                table = self.restore_table(dayflycell=row.name)
                row['Table'] = table
            row, attr_list = self._get_table_attr(row,column)
            return row  # Ensure the row is returned after modification

        self.df = self.df.apply(re_add_table_values,axis=1)
        self.save()


    def restore_table(self, dayflycell=None,overwrite = False):
        """
        Restore a table from the DataFrame using the dayflycell index.
        If the table is not found, it will return None.
        """
        if dayflycell is None or self.df is None:
            raise KeyError('Sinq has no Tables or no dayflycell given')
                
        def make_table(row):
            if pd.isna(row['parquet']):
                raise KeyError('parquet_path is empty')
            
            if not row['Table'] is None and not overwrite:
                print(f'Table {dayflycell} in sinq')
                return row['Table']
            else:
                T = Table(row['parquet'])
                return T

        if dayflycell == 'All' or dayflycell == 'all':
            print('Restoring all tables: {}'.format(self.df.index.tolist() if self.df is not None else [],))
            self.df['Table'] = self.df[['parquet','Table']].copy().apply(make_table, axis=1)
            self.T = self.df['Table'][0]
            return self.T
        
        if dayflycell not in self.df.index:
            print(f"No table found for {dayflycell}.")
            return None
        
        else:
            T = make_table(self.df.loc[dayflycell,['parquet','Table']])
            self.df.at[dayflycell, 'Table'] = T
            self.T = T
            return T
        
    

    def merge_sinq(self, other_sinq,*, overwrite: bool = False, merge_notes: bool = False,):
        """
        Merge another Sinq object into the current Sinq.
        This will combine the DataFrames, ensuring no duplicates.
        """
        if not isinstance(other_sinq, Sinq):
            raise ValueError("other_sinq must be an instance of Sinq.")
        
        if self.df is None:
            self.df = other_sinq.df.copy()
            self.save()
            return
        
        combined = pd.concat([self.df, other_sinq.df])
        keep = "last" if overwrite else "first"
        combined = combined[~combined.index.duplicated(keep=keep)]

        if merge_notes and "notes" in combined.columns:
            def merge_notes_fn(group):
                if len(group) == 1:
                    return group.iloc[0]
                a, b = group.iloc[0], group.iloc[1]
                if isinstance(a["notes"], dict) and isinstance(b["notes"], dict):
                    b["notes"] = {
                        "text": "\n".join(filter(None, [a["notes"].get("text"), b["notes"].get("text")])),
                        "tags": sorted(set(a["notes"].get("tags", [])) | set(b["notes"].get("tags", []))),
                        "author": b["notes"].get("author") or a["notes"].get("author"),
                        "timestamp": b["notes"].get("timestamp") or a["notes"].get("timestamp"),
                    }
                return b        
            combined = (
                combined.groupby(level=0, sort=False)
                        .apply(merge_notes_fn)
            )

        self.df = combined
        self.save()


    def __str__(self):
        return "Sinq({}, {} rows x {} params)".format(
            self.sinqname,
            self.df.shape[0] if self.df is not None else 0,
            len(self.df.columns) if self.df is not None else 0
        )
    
    def __repr__(self):
        if self.df is not None:
            df_no_table = self.df.drop(columns=['Table'])

            # Count total missing values
            missing_count = df_no_table.isna().sum().sum()
        else:
            missing_count = 0
        repr_str = "{}: {} empties; {} x {} ".format(
            self.__str__(),
            missing_count,
            self.df.index.tolist() if self.df is not None else [],
            self.df.columns.tolist() if self.df is not None else []
        )

        return repr_str
    

    def notes_text(self) -> pd.Series:
        """
        Return a Series of human-readable note text.
        """
        def extract(note):
            if isinstance(note, dict):
                return note.get("text", "")
            if isinstance(note, str):
                return note
            return ""

        return self.df["notes"].apply(extract)


    def notes_tags(self) -> pd.Series:
        """
        Return a Series of human-readable note text.
        """
        def extract(note):
            if isinstance(note, dict):
                return f"{', '.join(note['tags'])}"
            if isinstance(note, str):
                return note
            return ""

        return self.df["notes"].apply(extract)


    def display_df(self,show_tags=False) -> pd.DataFrame:
        """
        Human-readable view of the Sinq DataFrame.
        """
        df = self.df.copy()
        if "notes" in df.columns:
            df["note_text"] = self.notes_text()
            if show_tags:
                df["tags"] = self.notes_tags()
            df = _reorder_columns_helper(df)
        return df
    

    def rebuild(self):
        """
        Rebuild this Sinq from its source Sinqs.
        """
        if not hasattr(self, "_sources") or self._sources is None:
            raise RuntimeError("This Sinq has no recorded sources to rebuild from.")

        # wipe current state
        self.df = None

        for src in self._sources:
            self.merge_sinq(src, **getattr(self, "_merge_policy", {}))

        self.save()


    def __call__(self, *args, **kwargs):
        return Sinq(*self.args, *args, **self.kwargs, **kwargs)
