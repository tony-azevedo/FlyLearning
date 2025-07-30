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
            self.df.to_pickle(self.file_name)
        else:
            raise ValueError("DataFrame is empty. Nothing to save.")
        
        
    def add_table(self, table=None):
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
            if dayflycell not in self.df.index:
                self.T = Table(table)
            else:
                print(f"Table for {dayflycell} already exists.")
            return
        else:
            raise ValueError("Data must be a Table or string. Cannot be none")
        
        if self.df is None:
            # Make a simple DataFrame with the genotype as a column
            row_data = {'parquet': self.T.parquet,'Table': self.T,'genotype': self.T.genotype}
            self.df = pd.DataFrame([row_data], index=[dayflycell])

        elif dayflycell not in self.df.index:
            self.df.loc[dayflycell] = [np.nan] * len(self.df.columns)

        self.df.at[dayflycell, 'parquet'] = os.path.join(self.T.path,self.T.parquet)
        for col in self.T.df.columns:
            if hasattr(self.T, col) and callable(getattr(self.T, col)):
                self.df.at[dayflycell, col] = getattr(self.T, col)()
            else:
                self.df.at[dayflycell, col] = np.nan
        
        self.save()

    
    def add_column(self, column_name):
        """
        Add a column to the DataFrame.
        If the DataFrame is None, initialize it first.
        """
        if self.df is None:
            raise ValueError("DataFrame is not initialized. Call add_table() first.")
        
        if column_name in self.df.columns:
            raise ValueError(f"Column '{column_name}' already exists in the DataFrame.")
        
        for dayflycell in self.df.index:
            self.T = self.df['Table'].loc[dayflycell]
            if self.T is None:
                self.T = Table(self.df['parquet'].iloc[0])
                self.df.at[dayflycell, 'Table'] = self.T

            if hasattr(self.T, column_name) and callable(getattr(self.T, column_name)):
                day,fly,cell = get_day_fly_cell(self.T)
                dayflycell = f'{day}_F{fly}_C{cell}'

                self.df.at[dayflycell, column_name] = getattr(self.T, column_name)()
                
            else:
                self.df.at[dayflycell, column_name] = np.nan

        self.save()


    def __str__(self):
        return "Sinq({}, {} rows x {} params)".format(
            self.sinqname,
            self.df.shape[0] if self.df is not None else 0,
            len(self.df.columns) if self.df is not None else 0
        )
    
    def __repr__(self):
        return f"Sinq(args={self.args}, kwargs={self.kwargs})"

    def __call__(self, *args, **kwargs):
        return Sinq(*self.args, *args, **self.kwargs, **kwargs)
