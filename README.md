# Sea Level Indicators: Hawaiian Island Region

Coastal flooding takes many forms, ranging from major flooding associated with surge and heavy rains of strong storms to minor tidal flooding associated with seasonally high tides and variations in ocean currents. This repository, developed using BIL funds, contains the documented codes necessary to produce a diverse set of water-level monitoring products for the Hawaiian Island region using either hourly tide gauge data from the UHSLC database or NOAA CO-OPS API. These outputs can be
translated for use in reports, bulletins, portals and dashboards used by NOAA and its partners.

Jupyter Book Page is available at:
 https://jwfiedler.github.io/SL_Hawaii/

# SLI311 Environment Setup

This guide explains how to set up the `SLI311` environment for the project. The environment is managed using `conda` and includes both `conda` and `pip` dependencies. At the moment, this environment is a do-all beast, so resolving conflicts when installing the environment (because there are so many packages) might take some time. The perks of having it all in one big environment is...debatable. But for now that's where things are at. 

### Why a Large Environment?
The `SLI311` environment is designed to support a wide range of tasks, including:
- Data processing and analysis (`numpy`, `pandas`, `xarray`, `dask`)
- Geospatial analysis (`cartopy`, `geopandas`)
- Visualization (`seaborn`, `plotly`)
- Tidal analysis (`utide`)
- Accessing external data sources (`copernicusmarine`)

---

## Prerequisites

Before you begin, ensure you have the following installed:
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/)
You can also do things in mamba if that's what you're into:
- [Mamba](https://mamba.readthedocs.io/en/latest/installation.html) or [Micromamba](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html)


---

## Installation Steps

### 1. Clone the Repository
First, clone the repository to your local machine:
```bash
git clone https://github.com/jwfiedler/SL_Hawaii.git
cd SL_Hawaii 
```

### 2. Create the Environment
```bash
mamba env create -f environment.yml
```
OR
```bash
conda env create -f environment.yml
```

This should install all the conda dependencies, and automatically install the pip dependencies listed in the environment.yml file.

### 3. Activate the Environment
```bash
conda activate SLI311
```

### 4. Make sure everything worked

```bash
mamba list
```
OR
```bash
conda list
```
---
# Data input and output directories
You'll want to configure your data input and output directories for your own machine. This is where we will save our downloaded raw and intermediate (calculated) data. The output directory will contain our figures, tables, and csv files produced by the notebooks. You can put these folders wherever makes sense to your workflow. These data directories are established in the [getting started](notebooks/0_0_gettingStarted.md) chapter.

# Get your Copernicus Marine credentials set up
In order to access and download the CMEMS data (altimetry), you'll need to have [Copernicus Marine credentials](https://help.marine.copernicus.eu/en/articles/4220332-how-to-sign-up-for-copernicus-marine-service) stored in a configuration file on your machine. This configuration file will automatically be read when requesting data from the copernicus marine toolbox. You can read about how to do that [here](https://help.marine.copernicus.eu/en/articles/8185007-copernicus-marine-toolbox-credentials-configuration). Note: the downloaded datasets can be big.  

# Running the Notebooks
In order to run the notebooks succesfully, we suggest you follow the order given in the [table of contents](https://jwfiedler.github.io/SL_Hawaii/intro.html). If you want to be a rebel, you can try them out of order, but you'll still need to run the [data wrangling](notebooks/0_2_SL_Data_Wrangling.ipynb) first. 


## Citation
If you use this project in your research, please cite:

