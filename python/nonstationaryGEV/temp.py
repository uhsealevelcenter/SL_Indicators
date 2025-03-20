#%% Import required libraries
from models import run_CI_models, run_noClimateIndex_models
from imports import *
from plotting import plotExtremeSeasonality, plotTimeDependentReturnValue
from helpers import make_directories, get_monthly_max_time_series, get_covariate
from IPython import get_ipython
import sys
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import t
import matplotlib.pyplot as plt
import xarray as xr
import os

#%%
def in_debug_mode():
    gettrace = getattr(sys, 'gettrace', None)
    if gettrace is None:
        return False
    return gettrace() is not None

#%%
def in_interactive_mode():
    try:
        get_ipython()
        return True
    except NameError:
        return False

#%%
# Determine base directory based on mode
cwd = os.getcwd()

if in_debug_mode():
    base_dir = cwd  # For debug mode
elif in_interactive_mode():
    base_dir = os.path.abspath(os.path.join(cwd, "../.."))  # For interactive mode (Jupyter/IPython)
else:
    base_dir = cwd  # For normal execution

print(f"Base directory: {base_dir}")

# Make directories
dirs = make_directoryDict(base_dir)

#%% Define climate indices and record ID
CI_dir = dirs['CI_dir']
climateIndex = ['AO','BEST','ONI','PDO','PMM','PNA','TNA']

# Initialize a list to store DataFrames for each station
dataframes_list = []

#%% Get dataset of hourly sea level data
rsl = xr.open_dataset(dirs['data_dir'] / 'rsl_hawaii.nc')

# Remove stations 547, 548, 14
rsl_hourly = rsl.sel(record_id=~rsl.record_id.isin([547,548,14]))

#%% Loop through each station
for recordID in rsl_hourly.record_id.values:  # Ensure recordID is a value
    # Get dataset of monthly max sea level data
    mm, STNDtoMHHW, station_name, year0 = get_monthly_max_time_series(recordID, rsl_hourly)
    mmax = mm['monthly_max'].to_numpy()
    CIcorr = np.zeros((len(climateIndex), 30))

    # Arrays to store peak correlation and lag for each climate index
    CIcorr_max_peaks = np.zeros(len(climateIndex))
    CIcorr_max_lag = np.zeros(len(climateIndex))

    # Loop through each climate index
    for indCI in range(len(climateIndex)):
        CIname = climateIndex[indCI]
        CI = get_covariate(mm['t_monthly_max'], CI_dir, CIname=CIname)

        # Define the number of lags
        orig_lag = 30
        lag = orig_lag

        if CIname == 'PDO': # choose a reasonable maximum lag for a given climate index based on literature
            lag = 20

        if CIname == 'PMM': # choose a reasonable maximum lag for a given climate index based on literature
            lag = 15

        corr = np.zeros(orig_lag)

        # Calculate lagged correlation
        for i in range(1, lag + 1):
            corr[i - 1] = np.corrcoef(CI[:-i], mmax[i:])[0, 1]

        CIcorr[indCI,:] = corr

        # # Find peaks in correlation
        # peaks, _ = find_peaks(np.abs(CIcorr[indCI,:]), width=2)
        
        # if len(peaks) > 1:
        #     CIcorr_max_peaks[indCI] = CIcorr[indCI,peaks[0]]
        #     CIcorr_max_lag[indCI] = peaks[0]
        # else:
        #     CIcorr_max_peaks[indCI] = CIcorr[indCI,-1]
        #     CIcorr_max_lag[indCI] = lag-1

        # if CIcorr_max_peaks[indCI] < CIcorr[indCI,0]:
        #     CIcorr_max_peaks[indCI] = CIcorr[indCI,0]
        #     CIcorr_max_lag[indCI] = 0

        # if np.abs(np.abs(CIcorr_max_peaks[indCI])-np.abs(CIcorr[indCI,0]))<0.5*np.std(CIcorr[indCI,:]):
        #     CIcorr_max_peaks[indCI] = CIcorr[indCI,0]
        #     CIcorr_max_lag[indCI] = 0

        # get max correlation and lag
        CIcorr_max_peaks[indCI] = CIcorr[indCI, np.argmax(np.abs(CIcorr[indCI,:]))]
        CIcorr_max_lag[indCI] = np.argmax(np.abs(CIcorr[indCI,:]))

        ##****######****### THE FOLLOWING LINES SET THE LAGS!!! ######****######****###
        ####****######### REMOVE THIS SECTION TO ALLOW FOR AUTOMATIC LAG DETECTION ####

        #if CIname is ONI or BEST, enforce 18-month lag
        # # This is based on Long et al 2020
        ## ONLY DO THIS FOR HAWAII STATIONS ###
        if recordID in [57,58,59,60,61,552]:
            if CIname in ['ONI', 'BEST']:
                CIcorr_max_lag[indCI] = 18
                CIcorr_max_peaks[indCI] = CIcorr[indCI, 18]

            # if CIname is PDO, enforce 16-month lag
            if CIname == 'PDO':
                CIcorr_max_lag[indCI] = 16
                CIcorr_max_peaks[indCI] = CIcorr[indCI, 16]

            # if CIname is PMM, enforce 10-month lag
            if CIname == 'PMM':
                CIcorr_max_lag[indCI] = 10
                CIcorr_max_peaks[indCI] = CIcorr[indCI, 10]

        ####****##### REMOVE THIS SECTION TO ALLOW FOR AUTOMATIC LAG DETECTION ####***
        ##****######****### THE ABOVE LINES SET THE LAGS!!!#######****######****###***

    #% Plot correlation for each climate index
    fig, ax = plt.subplots()
    
    # fill zeros with nan
    CIcorr[CIcorr == 0] = np.nan

    ax.plot(np.arange(1, orig_lag+1), CIcorr.T)

    for indCI in range(len(climateIndex)):
        if CIcorr_max_lag[indCI] is not None:
            ax.scatter(CIcorr_max_lag[indCI] + 1, CIcorr_max_peaks[indCI])

    ax.set_xlabel('Lag (months)')
    ax.set_ylabel('Correlation')
    ax.set_title(f'Correlation between climate index and sea level monthly max for {station_name}')
    ax.legend(climateIndex)

    # Save plot in output/CI directory
    savedir = dirs['output_dir'] / 'CI'
    savedir.mkdir(parents=True, exist_ok=True)
    fig.savefig(savedir / f'{station_name}_correlation_plot.png')

    #% Create DataFrame for current station
    CI_lags_df = pd.DataFrame({
        'climateIndex': climateIndex,
        'max_corr': CIcorr_max_peaks,
        'lag': CIcorr_max_lag
    })

    # Calculate p-values
    def calculate_p_value(r, n):
        if r is None:
            return None  # Handle None values for correlation
        t_stat = r * np.sqrt((n - 2) / (1 - r**2))
        return 2 * (1 - t.cdf(abs(t_stat), df=n - 2))  # Two-tailed test

    # Calculate p-values for each climate index correlation
    CI_lags_df['p_value'] = np.nan
    for indCI in range(len(climateIndex)):
        r = CIcorr_max_peaks[indCI]
        lag = CIcorr_max_lag[indCI]
        
        if not np.isnan(lag):
            n = len(mmax) - int(lag)  # Adjust for lag
        else: 
            n = len(mmax)
        p_value = calculate_p_value(r, n)
        CI_lags_df.loc[indCI, 'p_value'] = p_value

    # Add significance column
    CI_lags_df['significant'] = CI_lags_df['p_value'] < 0.05

    # Add station name and recordID
    CI_lags_df['station'] = station_name
    CI_lags_df['recordID'] = recordID

    # Append the DataFrame to the list
    dataframes_list.append(CI_lags_df)

# After the loop, concatenate all DataFrames into a master DataFrame
master_df = pd.concat(dataframes_list, ignore_index=True)


# master_df.to_csv(dirs['CI_dir'] / 'CI_correlation_results.csv', index=False)


# # Enforce 18-month lag for ONI, BEST

# master_df.loc[master_df['climateIndex'].isin(['ONI', 'BEST']), 'lag'] = 19

# # Enforce 16-month lag for PDO
# # average lag from bulk of HI stations
# master_df.loc[master_df['climateIndex'] == 'PDO', 'lag'] = 16

# # Enforce 10-month lag for PMM
# # Long et al 2020 uses 8-month lag for upper 100m 
# master_df.loc[master_df['climateIndex'] == 'PMM', 'lag'] = 10


# Now `master_df` contains all the results across stations
print(master_df)

#%%
# Save master DataFrame to CSV
master_df.to_csv(dirs['CI_dir'] / 'CI_correlation_results_v2.csv', index=False)

#%%
# look at PMM for all stations
CI_df = master_df[master_df['climateIndex'] == 'PMM']

# get CI_df for Honolulu
honolulu_CI = CI_df[CI_df['station'] == 'Honolulu, Hawaii']

# %%
