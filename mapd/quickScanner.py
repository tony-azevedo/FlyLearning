import pandas as pd
import numpy as np 
import os
import shutil
import datetime as dt
from time import strftime
import sys
import seaborn as sns
import warnings
import requests
import matplotlib
from time import sleep
from .helpers import get_path


rng = np.random.default_rng()

data_dir = 'd:\\Data\\'


class QuickScanner:
    """A class for scanning through the raw data .mat files generated from continuous acquisition.

    Parameters
    ----------
    fly_cell_path (str): path of the fly to look at.


    Methods
    ---------
    scan
    get_state
    get_state_from_columns
    resolve_duplicates
    update_table
    
    """

    def __init__(self,
                 fly_cell_path,
                 client = None,
                 ):
        self.day = fly_cell_path[0:6]
        self.fly_cell = fly_cell_path
        
        self._daydir = os.path.join('d:\\Data\\',self.day)
        self._fly_cell_dir = os.path.join(self._daydir,self.fly_cell)
        

