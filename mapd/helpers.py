# Helper functions for flylearning analysis objects
from os import path
import re


def get_path(filename):
    default_directory = default_data_directory()
    directory_path = path.dirname(filename)
    return directory_path if directory_path else default_directory


def default_data_directory(verbose=True):
    possible_paths = [
        r"D:\Data", 
        r"C:\Users\Tony\Data"
        ]
    
    for p in possible_paths:
        if path.isdir(p):
            if verbose: print(f"Found data directory: {p}") 
            return p

    raise FileNotFoundError("No valid data directory found in expected locations.")


def get_file(filename):
    return path.basename(filename)


def get_day_fly_cell(file_path):

    filename_pattern = r"_(\d{6})_F(\d)_C(\d)_"
    match = re.search(filename_pattern, file_path, re.IGNORECASE)
    
    if match:
        # Extract components from the matched groups
        yymmdd, x, y = match.groups()
        return yymmdd, x, y
    else:
        ValueError('No identifiers found')