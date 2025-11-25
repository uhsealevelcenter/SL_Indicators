import pandas as pd
import numpy as np
import scipy.io
from datetime import datetime, timedelta
from scipy.stats import chi2
from scipy.optimize import brentq
import matplotlib.pyplot as plt
import subprocess
import json
import os

from pathlib import Path
import xarray as xr
from scipy.stats import chi2

import seaborn as sns
import matplotlib.colors as mcolors
import plotly.graph_objects as go

# Set up directories using the same approach as 0_1_setup.ipynb
try:
    # Try to import paths from a local config file (not in version control)
    import sys
    # Add the notebooks directory to path so we can import config_env
    notebooks_dir = Path(__file__).resolve().parent.parent / 'notebooks'
    if str(notebooks_dir) not in sys.path:
        sys.path.insert(0, str(notebooks_dir))
    
    from config_env import DATA_DIR as data_dir_path
    from config_env import OUTPUT_DIR as output_dir_path
    print("Using custom paths from config_env.py")
except ImportError:
    # Fallback to default local directories if config_env.py doesn't exist
    print("config_env.py not found. Using default 'data' and 'output' directories.")
    data_dir_path = 'data'
    output_dir_path = 'output'

# Set up directories as Path objects
data_dir = Path(data_dir_path).expanduser()
output_dir = Path(output_dir_path).expanduser()

# Create directories if they don't exist to prevent errors later
data_dir.mkdir(parents=True, exist_ok=True)
output_dir.mkdir(parents=True, exist_ok=True)

print(f"Data directory: {data_dir}")
print(f"Output directory: {output_dir}")


param_names = ['Annual seasonal cycle',
    'Semiannual seasonal cycle',
    'Triannual seasonal cycle',
    'Long-term Trend in Location',
    'Covariate in Location',
    'Covariate in Scale',
    'Nodal Cycle']



def make_directoryDict(base_dir):

    base_dir = Path(base_dir)


    dirs = {
        'data_dir': data_dir, 
        'output_dir': output_dir / 'extremes',
        'input_dir': base_dir / 'model_input',
        'matrix_dir': base_dir / 'matrix',
        'model_output_dir': data_dir / 'GEV_model_output',
        'CI_dir': data_dir / 'climate_indices',
        'run_dir': data_dir / 'model_run'
    }
    
    return dirs


def define_limits(run_dir):
    limits = [
        (0.01, 7.5),
        (0.001, 2.5),
        (-0.35, 0.15),
        (-2.001, 2.001),
        (-2.001, 2.001),
        (-0.5, 0.5),
        (-0.5, 0.5),
        (-0.25, 0.25),
        (-0.25, 0.25),
        (-0.3, 0.3),
        (-0.5, 0.5),
        (-0.2, 0.2),
        (-0.2, 0.2),
        (-0.2, 0.2),
    ]

    limitsPath = run_dir / 'limits.txt'

    with open(limitsPath, "w") as file:
        for row in limits:
            file.write(f"{row[0]} {row[1]}\n")

    return limitsPath