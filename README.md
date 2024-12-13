Analysis scripts for operant conditioning experiments

Purpose: Separate data acquisition and pre-processing from analysis.
Previously, all analysis records were in MATLAB scripts. Moving to python for analysis scripts for some of the reasons listed below.


Matlab is limited in several ways for analysis: 
1. Stuck with MATLAB ide.
2. Iterators and indexing is outdated.
3. Pandas is better than tables.
4. Integrating with additional packages is limited.

There are some new developments that make python possible:
1. .mat files are now h5 files.
2. Datajoints might offer new advantages.
3. ipynb scripts are just as good as MATLAB scripts now
4. Visual Studio Code is an excellent IDE

Preferrences:
1. Python is just more readable and manipulable, e.g. can set default input values
2. I've just grown more accustomed to it.
3. MATLAB cell arrays are the stupidest thing ever.