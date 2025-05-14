# %% [markdown]
# # Setup
# 
# We first need to import the necessary libraries, access the data, and make a quick plot to ensure we will be analyzing the right thing.
# 
# ## Import necessary libraries.

# %%
# Standard libraries
import os, io, glob
import datetime as dt
from pathlib import Path

# Data manipulation libraries
import numpy as np
import pandas as pd
import xarray as xr

# Data retrieval libraries
from urllib.request import urlretrieve
import requests

# Data analysis libraries
import scipy.stats as stats

# HTML parsing library
from bs4 import BeautifulSoup

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Miscellaneous
from myst_nb import glue  # used for figure numbering when exporting to LaTeX

# import custom functions
import sys
notebooks_dir = Path.cwd()
if notebooks_dir not in sys.path:
    sys.path.append(str(notebooks_dir.parent))
    
import tseries_functions as tsf

# %%
#set up directories as environment variables
os.environ["DATA_DIR"] = "~/Documents/SL_Hawaii_data/data"
os.environ["OUTPUT_DIR"] = "~/Documents/SL_Hawaii_data/output"

#set up directories as Path objects
data_dir = Path(os.environ["DATA_DIR"]).expanduser()
output_dir = Path(os.environ["OUTPUT_DIR"]).expanduser()

# %%




