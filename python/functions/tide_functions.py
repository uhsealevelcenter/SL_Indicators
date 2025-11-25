# run this code in the SLI311 environment
# This code is used to calculate the non-tidal residuals for the Hawaii tide gauge data


def calculate_ntr(ds):
    """"
    "This function calculates the non-tidal residuals (NTR) for the Hawaii tide gauge data"
    """
    import pandas as pd
    import xarray as xr
    import numpy as np
    from pathlib import Path
    from utide import solve, reconstruct
    import os
    from python.functions.tseries_functions import process_trend_with_nan
    from copy import deepcopy
    from scipy.interpolate import interp1d

    # #set up directories as Path objects, this assumes the environment variables
    # # DATA_DIR and OUTPUT_DIR are set
    # if 'DATA_DIR' not in os.environ:
    #     DATA_DIR = os.path.join(os.environ['HOME'], 'Documents', 'SL_Hawaii_data','data')
    #     os.environ["DATA_DIR"] = DATA_DIR
    # if 'OUTPUT_DIR' not in os.environ:
    #     OUTPUT_DIR = os.path.join(os.environ['HOME'], 'Documents', 'SL_Hawaii_data','output')
    #     os.environ["OUTPUT_DIR"] = OUTPUT_DIR

    # Set up directories using the same approach as 0_1_setup.ipynb
    
    from config_env import DATA_DIR as data_dir_path
    from config_env import OUTPUT_DIR as output_dir_path
    

    # Set up directories as Path objects
    data_dir = Path(data_dir_path).expanduser()
    output_dir = Path(output_dir_path).expanduser()


    savepath = Path(data_dir / 'ntr_data')
    if not savepath.exists():
        savepath.mkdir()

    # #load the data
    # ds = xr.open_dataset(data_dir / 'rsl_hawaii.nc')




    # # use NTDE 1983-2001 as epoch:
    epoch_start = np.datetime64('1983-01-01')
    epoch_end = np.datetime64('2001-12-31')
    ds_epoch = ds.sel(time=slice(epoch_start, epoch_end))

    # ds_epoch = ds.copy()

    # # use NTDE 2002-2020 as epoch
    # epoch_start = np.datetime64('2002-01-01')
    # epoch_end = np.datetime64('2020-12-31')
    # ds_epoch = ds.sel(time=slice(epoch_start, epoch_end))

    #extract time and sea level values for each station
    time = ds.time.values
    time_days = (time - time[0]) / np.timedelta64(1, 'D')

    stations = ds['station_id'].values

    # reduce stations if already processed
    processed_files = [
        int(f.stem.split('_')[1])
        for f in savepath.glob('ntr_*.csv')
        if f.stem.split('_')[1].isdigit()
    ]

    # name must match the pattern ntr_XXX.csv, remove any files that do not match this pattern

    print(f'Already processed {len(processed_files)} files:')
    
    #print the list of processed files
    print(processed_files)

    stations = [s for s in stations if s not in processed_files]

    for station in stations:
        t = pd.to_datetime(ds_epoch.time.values)
        sea_level = ds_epoch.sea_level.sel(station_id=station).values
        time_ALL = pd.to_datetime(ds.time.values)
        sea_level_ALL = ds.sea_level.sel(station_id=station).values
        # sea_level_df = pd.DataFrame({'sea_level': ds.sea_level.sel(station_id=station).values, 'time': pd.to_datetime(ds.time.values)})

        station_name = ds_epoch.station_name.sel(station_id=station).item()
        print(f'Working on station {station}: {station_name}')

        #check if enough data in sea_level
        if np.sum(~np.isnan(sea_level)) < 365:
            print(f'Not enough data for station {station_name}')
            continue

        #truncate dataset if gaps are longer than 3 months. keep the latter part of the record
        max_gap = np.timedelta64(90, 'D') # 3 months
        time_days = np.array(time_days, dtype='datetime64[D]')

        gaps = np.diff(time_days)
        if np.any(gaps > max_gap):
            # find the first gap
            first_gap = np.where(gaps > max_gap)[0][0]
            print(f'Found gap longer than 3 months for station {station_name}, truncating dataset')
            ds_epoch = ds_epoch.sel(time=slice(None, time_days[first_gap]))

        #remove trend from sea level data using linear regression
        # make sea_level a pandas dataframe
        sea_level_df = pd.DataFrame({'sea_level': sea_level, 'time': pd.to_datetime(ds_epoch.time.values)})
        sea_level_df_ALL = pd.DataFrame({'sea_level': sea_level_ALL,'time':pd.to_datetime(ds.time.values)})
        # truncate the data to the last 18.6 years for the tide analysis
        # sea_level_df = sea_level_df.iloc[-int(365*18.6*24):]
        

        
        trend_mag, trend, trend_rate = process_trend_with_nan(sea_level_df, time_column='time', weighted=False)
        #extend trend to the full time series

        time_index = pd.to_datetime(time_ALL).to_julian_date()
        f = interp1d(t.to_julian_date(), trend['sea_level'],kind='linear', fill_value='extrapolate')
        trend_epoch_extended = f(time_index)

        trend_magALL, trendALL, trend_rate = process_trend_with_nan(sea_level_df_ALL,time_column='time', weighted=False)

        trendDiff = trend_epoch_extended - trendALL['sea_level']

        # make trend same time period as sea_level
        # trend = trend[(trend.index >= pd.to_datetime(epoch_start)) & (trend.index < pd.to_datetime(epoch_end))]

        # sea_level_detrended = sea_level_df['sea_level'] - trend['sea_level']
        # # sea_level_detrended should only be the epoch period
        # sea_level_detrended = sea_level_detrended[(sea_level_df['time'] >= pd.to_datetime(epoch_start)) & (sea_level_df['time'] <= pd.to_datetime(epoch_end))]
        # sea_level_detrended = sea_level_detrended.values

        sea_level_detrended = sea_level_ALL - trendALL['sea_level']

        # print what type sea_level_detrended is
        print(f'sea_level_detrended is of type {type(sea_level_detrended)} and has shape {sea_level_detrended.shape}')
        
        # truncate the data to the last 18.6 years for the tide analysis, each datapoint is 1 hour
        # 18.6 years = 18.6 * 365 * 24 = 162768 hours
        # sea_level_detrended_186 = sea_level_detrended[-int(365*18.6*24):]
        # t = time[-int(365*18.6*24):]

        # These next two lines will take some time, especially if doing the full timeseries
        # Using OLS method instead of robust to speed things up
        coef_noNodal = solve(time_ALL, sea_level_detrended, nodal=False, trend=False, method='ols',lat=float(ds['lat'].sel(station_id=station).values))
        coef = solve(time_ALL, sea_level_detrended, nodal=True, trend=False, method='ols',lat=ds['lat'].sel(station_id=station).values)

        

        # find beginning of data in timeseries
        start = np.where(~np.isnan(sea_level_ALL))[0][0]
        time_ALL = time_ALL[start:]
        sea_level_ALL = sea_level_ALL[start:]

        
        seasonal_names = ['SA', 'SSA']
        
        seasonal_idx = [i for i, name in enumerate(coef['name']) if name in seasonal_names]
        nonseasonal_idx = [i for i, name in enumerate(coef['name']) if name not in seasonal_names]

        coef_seasonal = deepcopy(coef)
        coef_noSeasonal = deepcopy(coef)

        # Filter main fields
        for key in ['A', 'g', 'A_ci', 'g_ci', 'PE', 'SNR', 'name']:
            if key in coef:
                coef_seasonal[key] = np.array(coef[key])[seasonal_idx]
                coef_noSeasonal[key] = np.array(coef[key])[nonseasonal_idx]

        # Filter auxiliary fields
        for aux_key in coef['aux']:
            value = coef['aux'][aux_key]
            if isinstance(value, np.ndarray) and value.ndim == 1 and len(value) == len(coef['name']):
                coef_seasonal['aux'][aux_key] = value[seasonal_idx]
                coef_noSeasonal['aux'][aux_key] = value[nonseasonal_idx]
            else:
                coef_seasonal['aux'][aux_key] = value  
                coef_noSeasonal['aux'][aux_key] = value

        tide_ALL_withNodal_noSeasonal = reconstruct(time_ALL, coef_noSeasonal)
        tide_ALL_noNodal = reconstruct(time_ALL, coef_noNodal)
        tide_ALL_withNodal = reconstruct(time_ALL, coef)
        seasonal_cycle = reconstruct(time_ALL, coef_seasonal)

        # tide_ALL_withNodal_MHHWepoch = tide_ALL_withNodal.h-trendDiff[start:]
        sea_level_detrendedALL = sea_level_detrended[start:]

        # sea_level_detrendedALL = sea_level_ALL - np.nanmean(sea_level) + trend_rate['sea_level']/365.25
        # sea_level_detrendedALL = sea_level_ALL - np.nanmean(sea_level_ALL) 
        # sea_level_detrendedALL = sea_level_ALL - trendALL

        # ensure that the NTRs returned are relative to the epoch with trendDiff
        ntr = sea_level_detrended[start:] - tide_ALL_withNodal.h 
        ntr_withNodal = sea_level_detrended[start:] - tide_ALL_noNodal.h 
        ntr_noASA = sea_level_detrended[start:] - tide_ALL_withNodal_noSeasonal.h 
        # ntr = sea_level_detrended[start:] - tide_ALL_withNodal_noSeasonal.h 
        # ntr_withNodal = sea_level_detrended[start:] - tide_ALL_noNodal.h 
        # nodal_pred = tide_ALL_withNodal.h - tide_ALL_noNodal.h
        nodal_pred =  tide_ALL_noNodal.h - tide_ALL_withNodal.h
        trendALL = trendALL['sea_level'][start:]
        trendDiff = trendDiff[start:]

        ntr_data = pd.DataFrame({
            'time': time_ALL, 
            'ntr': ntr, 
            'ntr_noASA': ntr_noASA,
            'sea_level': sea_level_ALL, 
            'sea_level_detrended': sea_level_detrendedALL,
            'trend': trendALL,
            'trendDiffEpoch': trendDiff,
            'tide': tide_ALL_withNodal.h, 
            'nodal': nodal_pred, 
            'ntr_withNodal': ntr_withNodal, 
            'seasonal_cycle': seasonal_cycle.h
        })

        #ensure that the time is in datetime format
        ntr_data['time'] = pd.to_datetime(ntr_data['time'])
        ntr_data['time'] = ntr_data['time'].dt.tz_localize(None)

        datapath = Path(savepath, f'ntr_{station:03d}.csv')
        ntr_data.to_csv(datapath, index=False, date_format='%Y-%m-%d %H:%M:%S')
        print(f'Saved data for station {station_name} to {datapath}')

## run the function
# calculate_ntr(ds)

# set up a test function to test the function
if __name__ == '__main__':
    # This is a test function to test the calculate_ntr function
    # This is a test function to test the calculate_ntr function
    import xarray as xr
    from pathlib import Path
    import os
    
    print('Running tide_functions.py')

    from config_env import DATA_DIR as data_dir_path
    

    # Set up directories as Path objects
    data_dir = Path(data_dir_path).expanduser()
    ds = xr.open_dataset(data_dir / 'rsl_hawaii.nc')
    
    calculate_ntr(ds)