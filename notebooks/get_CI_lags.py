#%% Import required libraries

from IPython import get_ipython
import sys
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import t
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import xarray as xr
import os
from pathlib import Path
from setup import data_dir
base_dir = Path(data_dir).parent

nsGEVpath = Path(__file__).parent.parent / 'python' / 'nonstationaryGEV'

if str(nsGEVpath) not in sys.path:
    sys.path.append(str(nsGEVpath))


from models import run_CI_models, run_noClimateIndex_models
from imports import *
from plotting import plotExtremeSeasonality, plotTimeDependentReturnValue
from helpers import make_directories, get_monthly_max_time_series, get_covariate
#%%


print(f"Base directory: {base_dir}")

# Make directories
dirs = make_directoryDict(base_dir)

#%% Define climate indices and record ID
CI_dir = dirs['CI_dir']
climateIndex = ['AO','BEST','ONI','PDO','PMM','PNA','TNA']



#%% Get dataset of hourly sea level data
rsl_hourly = xr.open_dataset(dirs['data_dir'] / 'rsl_hawaii_noaa.nc')

#%% Loop through each station
def get_CI_lags_dataframe(rsl_hourly, set_lags=True):
    # Initialize a list to store DataFrames for each station
    dataframes_list = []

    for stationID in rsl_hourly.station_id.values:  # Ensure stationID is a value
        # Get dataset of monthly max sea level data
        stationID_str = str(stationID.item() if hasattr(stationID, 'item') else stationID)

        
        mm, STNDtoMHHW, station_name, year0 = get_monthly_max_time_series(stationID_str, rsl_hourly)
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

            # get max correlation and lag
            CIcorr_max_peaks[indCI] = CIcorr[indCI, np.argmax(np.abs(CIcorr[indCI,:]))]
            CIcorr_max_lag[indCI] = np.argmax(np.abs(CIcorr[indCI,:]))

            ##****######****### THE FOLLOWING LINES SET THE LAGS!!! ######****######****###
            ####****######### REMOVE THIS SECTION TO ALLOW FOR AUTOMATIC LAG DETECTION ####

            #if CIname is ONI or BEST, enforce 18-month lag
            # # This is based on Long et al 2020
            ## ONLY DO THIS FOR HAWAII STATIONS ###

            if set_lags:

                if int(stationID) in [1611400,1612340,1612480,1615680,1617433,1617760]:
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

        #% Plot correlation for each climate 
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
        plt.close(fig)  

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

        # Add station name and stationID
        CI_lags_df['station'] = station_name
        CI_lags_df['stationID'] = stationID

        # Append the DataFrame to the list
        dataframes_list.append(CI_lags_df)

    # After the loop, concatenate all DataFrames into a master DataFrame
    master_df = pd.concat(dataframes_list, ignore_index=True)

    return master_df



df_nosetlag = get_CI_lags_dataframe(rsl_hourly, set_lags=False)
df_nosetlag.to_csv(dirs['CI_dir'] / 'CI_correlation_results.csv', index=False)

df_setlag = get_CI_lags_dataframe(rsl_hourly, set_lags=True)
df_setlag.to_csv(dirs['CI_dir'] / 'CI_correlation_results_setLag.csv', index=False)

# print(df_nosetlag)

# %%
