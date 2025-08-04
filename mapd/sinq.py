import os
import numpy as np
import pandas as pd
from .helpers import get_day_fly_cell, get_file, default_data_directory
from .table import Table  # Assuming Table is defined in table.py


class Sinq(object):
    def __init__(self, **kwargs):
        self.sinqname = kwargs.get('sinqname', 'all_Tables')
        self.file_location = os.path.join(default_data_directory(),'Sinqs')
        os.makedirs(self.file_location, exist_ok=True)
        self.file_name = os.path.join(self.file_location, self.sinqname + '.pkl')
        self.T = None
        self.load_sinq()

    def load_sinq(self):
        """Load the DataFrame from the file."""
        if os.path.exists(self.file_name):
            self.df = pd.read_pickle(self.file_name)
        else:
            self.df = None
            print(f"File {self.file_name} does not exist. Starting with an empty DataFrame.")
            
    
    def save(self):
        """Save the DataFrame to the file. remove tables first"""

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
        Add a table to the Sinq object.
        If 'table' is a DataFrame, add it directly.
        If 'table' is a string, create the table and analyze to add to sinq table (empty or via helper).
        """
        if isinstance(table, Table):
            # If data is a string, create a new DataFrame (customize as needed)
            # Example: create an empty DataFrame with the string as a column
            self.T = table
            dayflycell = self.T.flycelldir
        elif isinstance(table, str):
            # Add/replace the table as a new attribute or merge as needed
            day,fly,cell = get_day_fly_cell(table)
            dayflycell = f'{day}_F{fly}_C{cell}'
            if (self.df is not None) and (dayflycell in self.df.index):                    
                # Check if all values in the row are not np.nan; if so, return early
                if not self.df.loc[dayflycell].isnull().any():
                    print(f"Table and all computations for {dayflycell} already exists. Skipping addition.")
                    return
            self.T = Table(table)
        else:
            raise ValueError("Data must be a Table or string. Cannot be none")
        
        row_data = {
            'parquet': self.T.parquet,
            'Table': self.T,
            'genotype': self.T.genotype
        }

        print(row_data)

        if self.df is None:
            self.df = pd.DataFrame([row_data], index=[dayflycell])
        else:
            self.df.loc[dayflycell] = pd.Series(row_data)

        for attr in self.df.columns:
            if not attr in row_data:
                print(f"Adding {attr} for {dayflycell}")
                if hasattr(self.T, attr) and callable(getattr(self.T, attr)):
                    if overwrite:
                        self.df.at[dayflycell, attr] = getattr(self.T, attr)() 
                else:
                    raise ValueError('Attribute {} not found in Table {}'.format(attr, self.T))
                    self.df.at[dayflycell, attr] = np.nan
        
        self.save()
        return self.T


    def restore_table(self, dayflycell):
        """
        Restore a table from the DataFrame using the dayflycell index.
        If the table is not found, it will return None.
        """
        if self.df is None or dayflycell not in self.df.index:
            print(f"No table found for {dayflycell}.")
            return None
        
        parquet_path = self.df.at[dayflycell, 'parquet']
        if pd.isna(parquet_path):
            print(f"No parquet path found for {dayflycell}.")
            return None
        
        self.T = Table(parquet_path)
        self.df.at[dayflycell, 'Table'] = self.T
        return self.T

    
    def add_column(self, column_name):
        """
        Add a column to the DataFrame.
        If the DataFrame is None, initialize it first.
        Will not compute column for other rows.
        """
        if self.df is None:
            raise ValueError("DataFrame is not initialized. Call add_table() first.")
        
        if column_name in self.df.columns:
            print(f"Column '{column_name}' already exists in the DataFrame.")
            return
        
        if self.T is None:
            raise ValueError("No Table object is associated with this Sinq. Call add_table() first.")
        
        day,fly,cell = get_day_fly_cell(self.T.parquet)
        dayflycell = f'{day}_F{fly}_C{cell}'


        if callable(getattr(self.T, column_name)):
            returned_attr = getattr(self.T, column_name)()
        else:
            returned_attr = getattr(self.T, column_name, np.nan)

        if pd.api.types.is_scalar(returned_attr):
            self.df.at[dayflycell, column_name] = returned_attr

        if pd.api.types.is_dict_like(returned_attr):
            for key, value in returned_attr.items():
                self.df.at[dayflycell, f"{column_name}_{key}"] = value

        self.save()
        return returned_attr

    def sync_sinq(self, overwrite=True):
        """
        Synchronize the current Sinq by filling in missing columns
        based on the existing Table objects in the DataFrame.
        """

        self.save()


    def merge_sinq(self, other_sinq):
        """
        Merge another Sinq object into the current Sinq.
        This will combine the DataFrames, ensuring no duplicates.
        """
        if not isinstance(other_sinq, Sinq):
            raise ValueError("other_sinq must be an instance of Sinq.")
        
        if self.df is None:
            self.df = other_sinq.df.copy()
        elif other_sinq.df is not None:
            self.df = pd.concat([self.df, other_sinq.df]).drop_duplicates()

        self.save()


    def __str__(self):
        return "Sinq({}, {} rows x {} params)".format(
            self.sinqname,
            self.df.shape[0] if self.df is not None else 0,
            len(self.df.columns) if self.df is not None else 0
        )
    
    def __repr__(self):
        repr_str = "{}: {} empties; {} x {} ".format(
            self.__str__(),
            self.df.isna().sum().sum() if self.df is not None else 0,
            self.df.index.tolist() if self.df is not None else [],
            self.df.columns.tolist() if self.df is not None else []
        )

        return repr_str

    def __call__(self, *args, **kwargs):
        return Sinq(*self.args, *args, **self.kwargs, **kwargs)
