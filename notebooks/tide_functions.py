# run this code in the SLI39 environment
# This code is used to calculate the non-tidal residuals for the Hawaii tide gauge data
print('Running tide_functions.py')



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
    from tseries_functions import process_trend_with_nan
    from copy import deepcopy

    #set up directories as Path objects, this assumes the environment variables
    # DATA_DIR and OUTPUT_DIR are set
    if 'DATA_DIR' not in os.environ:
        DATA_DIR = os.path.join(os.environ['HOME'], 'Documents', 'SL_Hawaii_data','data')
        os.environ["DATA_DIR"] = DATA_DIR
    if 'OUTPUT_DIR' not in os.environ:
        OUTPUT_DIR = os.path.join(os.environ['HOME'], 'Documents', 'SL_Hawaii_data','output')
        os.environ["OUTPUT_DIR"] = OUTPUT_DIR

    data_dir = Path(os.environ["DATA_DIR"]).expanduser()
    output_dir = Path(os.environ["OUTPUT_DIR"]).expanduser()


    savepath = Path(data_dir / 'ntr_data')
    if not savepath.exists():
        savepath.mkdir()

    # #load the data
    # ds = xr.open_dataset(data_dir / 'rsl_hawaii.nc')




    # # use NTDE 1983-2001 as epoch:::NO USE FULL RECORD
    # epoch_start = np.datetime64('1983-01-01')
    # epoch_end = np.datetime64('2001-01-01')
    # ds_epoch = ds.sel(time=slice(epoch_start, epoch_end))

    #extract time and sea level values for each station
    time = ds.time.values
    time_days = (time - time[0]) / np.timedelta64(1, 'D')

    stations = ds['record_id'].values

    # reduce stations if already processed
    processed_files = [
        int(f.stem.split('_')[1])
        for f in savepath.glob('ntr_*.csv')
        if len(f.name) == 11
    ]

    # name must match the pattern ntr_XXX.csv, remove any files that do not match this pattern

    print(f'Already processed {len(processed_files)} files:')
    
    #print the list of processed files
    print(processed_files)

    stations = [s for s in stations if s not in processed_files]

    for station in stations:
        sea_level = ds.sea_level.sel(record_id=station
                           ).values
        station_name = ds.station_name.sel(record_id=station).item()
        print(f'Working on station {station}: {station_name}')

        #check if enough data in sea_level
        if np.sum(~np.isnan(sea_level)) < 365:
            print(f'Not enough data for station {station_name}')
            continue

        #remove trend from sea level data using linear regression
        # make sea_level a pandas dataframe
        sea_level_df = pd.DataFrame({'sea_level': sea_level, 'time': pd.to_datetime(ds.time.values)})

        # truncate the data to the last 18.6 years for the tide analysis
        # sea_level_df = sea_level_df.iloc[-int(365*18.6*24):]
        trend_mag, trend, trend_rate = process_trend_with_nan(sea_level_df, time_column='time', weighted=False)

        sea_level_detrended = sea_level - trend['sea_level']

        # truncate the data to the last 18.6 years for the tide analysis, each datapoint is 1 hour
        # 18.6 years = 18.6 * 365 * 24 = 162768 hours
        # sea_level_detrended_186 = sea_level_detrended[-int(365*18.6*24):]
        # t = time[-int(365*18.6*24):]

        coef_noNodal = solve(time, sea_level_detrended, nodal=False, trend=False, method='robust',lat=float(ds['lat'].sel(record_id=station).values))
        coef = solve(time, sea_level_detrended, nodal=True, trend=False, method='robust',lat=ds['lat'].sel(record_id=station).values)

        time_ALL = ds.time.values
        sea_level_ALL = ds.sea_level.sel(record_id=station).values

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

        

       
        ntr = sea_level_detrended[start:] - tide_ALL_withNodal_noSeasonal.h 
        ntr_withNodal = sea_level_detrended[start:] - tide_ALL_noNodal.h 
        nodal_pred = tide_ALL_withNodal.h - tide_ALL_noNodal.h

        ntr_data = pd.DataFrame({'time': time_ALL, 'ntr': ntr, 'sea_level': sea_level_ALL, 'sea_level_detrended': sea_level_detrended[start:],'trend': trend['sea_level'][start:],'tide': tide_ALL_withNodal_noSeasonal.h, 'nodal': nodal_pred, 'ntr_withNodal': ntr_withNodal, 'seasonal_cycle': seasonal_cycle.h})

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
    
    
    data_dir = Path('/Users/juliafiedler/Documents/SL_Hawaii_data/data')
    ds = xr.open_dataset(data_dir / 'rsl_pacific.nc')
    
    calculate_ntr(ds)

