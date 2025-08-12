# run this code in the SLI39 environment
# This code is used to calculate the non-tidal residuals for the Hawaii tide gauge data

import pandas as pd
import xarray as xr
import numpy as np
from pathlib import Path
from utide import solve, reconstruct
import os

#set up directories as Path objects, this assumes the environment variables
#DATA_DIR and OUTPUT_DIR are set

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

#load the data
ds = xr.open_dataset(data_dir / 'rsl_hawaii.nc')


def calculate_ntr(ds):

    # use NTDE 1983-2001 as epoch
    epoch_start = np.datetime64('1983-01-01')
    epoch_end = np.datetime64('2001-01-01')
    ds_epoch = ds.sel(time=slice(epoch_start, epoch_end))

    #extract time and sea level values for each station
    time = ds.time.values
    time_days = (time - time[0]) / np.timedelta64(1, 'D')

    stations = ds['record_id'].values

    # reduce stations if already processed
    processed_files = [int(f.stem.split('_')[1]) for f in savepath.glob('ntr_*.csv')]

    print(f'Already processed {len(processed_files)} files:')
    
    #print the list of processed files
    print(processed_files)

    stations = [s for s in stations if s not in processed_files]

    for station in stations:
        sea_level = ds_epoch.sea_level.sel(record_id=station
                           ).values
        station_name = ds_epoch.station_name.sel(record_id=station).item()
        print(f'Working on station {station}: {station_name}')

        #check if enough data in sea_level
        if np.sum(~np.isnan(sea_level)) < 365:
            print(f'Not enough data for station {station_name}')
            continue

        #remove trend from sea level data using linear regression
        # make sea_level a pandas dataframe
        sea_level_df = pd.DataFrame({'sea_level': sea_level, 'time': pd.to_datetime(ds_epoch.time.values)})

        # truncate the data to the last 18.6 years for the tide analysis
        # sea_level_df = sea_level_df.iloc[-int(365*18.6*24):]
        trend_rate = np.polyfit(sea_level_df['time'].astype('datetime64[ns]').values.astype(np.int64), sea_level_df['sea_level'], 1)
        trend = np.polyval(trend_rate, sea_level_df['time'].astype('datetime64[ns]').values.astype(np.int64))
        trend = pd.DataFrame({'sea_level': trend, 'time': sea_level_df['time']})

        sea_level_detrended = sea_level - trend['sea_level']

        
        # truncate the data to the last 18.6 years for the tide analysis, each datapoint is 1 hour
        # 18.6 years = 18.6 * 365 * 24 = 162768 hours
        # sea_level_detrended_186 = sea_level_detrended[-int(365*18.6*24):]
        # t = time[-int(365*18.6*24):]

        t = pd.to_datetime(ds_epoch.time.values)
        coef_noNodal = solve(t, sea_level_detrended, nodal=False, trend=False, method='robust',lat=float(ds['lat'].sel(record_id=station).values))
        coef = solve(t, sea_level_detrended, nodal=True, trend=False, method='robust',lat=ds['lat'].sel(record_id=station).values)

        time_ALL = ds.time.values
        sea_level_ALL = ds.sea_level.sel(record_id=station).values

        # find beginning of data in timeseries
        start = np.where(~np.isnan(sea_level_ALL))[0][0]
        time_ALL = time_ALL[start:]
        sea_level_ALL = sea_level_ALL[start:]

        #extend trend to the full time series
        time_index = pd.to_datetime(time_ALL).to_julian_date()
        trendALL = np.interp(time_index, t.to_julian_date(), trend['sea_level'])

        seasonal_names = ['SA', 'SSA']
        
        seasonal_idx = [i for i, name in enumerate(coef['name']) if name in seasonal_names]
        
        #create a new coef structure for seasonal terms
        coef_seasonal = {}
        coef_seasonal['name'] = [coef['name'][i] for i in seasonal_idx]
        coef_seasonal['freq'] = [coef['freq'][i] for i in seasonal_idx]
        coef_seasonal['A'] = [coef['A'][i] for i in seasonal_idx]
        coef_seasonal['g'] = [coef['g'][i] for i in seasonal_idx]
        
        # Remove seasonal terms from main coef
        non_seasonal_idx = [i for i, name in enumerate(coef['name']) if name not in seasonal_names]
        coef_no_seasonal = {}
        coef_no_seasonal['name'] = [coef['name'][i] for i in non_seasonal_idx]
        coef_no_seasonal['freq'] = [coef['freq'][i] for i in non_seasonal_idx]
        coef_no_seasonal['A'] = [coef['A'][i] for i in non_seasonal_idx]
        coef_no_seasonal['g'] = [coef['g'][i] for i in non_seasonal_idx]
        
        
        
        # Create coef arrays (need to be numpy arrays for reconstruct)
        for key in ['name', 'freq', 'A', 'g']:
            coef_seasonal[key] = np.array(coef_seasonal[key])
            coef_no_seasonal[key] = np.array(coef_no_seasonal[key])
        
        coef_seasonal['aux'] = coef['aux']
        coef_no_seasonal['aux'] = coef['aux']

        
        tide_ALL_noNodal = reconstruct(time_ALL, coef_noNodal)
        tide_ALL_withNodal_noSeasonal = reconstruct(time_ALL, coef_no_seasonal)
        tide_ALL_withNodal = reconstruct(time_ALL, coef)
        seasonal_cycle = reconstruct(time_ALL, coef_seasonal)


        sea_level_detrendedALL = sea_level_ALL - np.nanmean(sea_level) + trend_rate[0]/365.25

        ntr = sea_level_detrendedALL - tide_ALL_withNodal_noSeasonal.h 
        ntr_withNodal = sea_level_detrendedALL - tide_ALL_noNodal.h 
        # ntr = sea_level_detrended[start:] - tide_ALL_withNodal_noSeasonal.h 
        # ntr_withNodal = sea_level_detrended[start:] - tide_ALL_noNodal.h 
        # nodal_pred = tide_ALL_withNodal.h - tide_ALL_noNodal.h
        nodal_pred = tide_ALL_noNodal.h - tide_ALL_withNodal.h


        ntr_data = pd.DataFrame({'time': time_ALL, 'ntr': ntr, 'sea_level': sea_level_ALL, 'sea_level_detrended': sea_level_detrendedALL,'trend': trendALL,'tide': tide_ALL_withNodal_noSeasonal.h, 'nodal': nodal_pred, 'ntr_withNodal': ntr_withNodal, 'seasonal_cycle': seasonal_cycle.h})

        #ensure that the time is in datetime format
        ntr_data['time'] = pd.to_datetime(ntr_data['time'])
        ntr_data['time'] = ntr_data['time'].dt.tz_localize(None)

        datapath = Path(savepath, f'ntr_{station:03d}.csv')
        ntr_data.to_csv(datapath, index=False, date_format='%Y-%m-%d %H:%M:%S')
        print(f'Saved data for station {station_name} to {datapath}')

if __name__ == '__main__':
    from pathlib import Path
    import os
    
    print('Running tide_functions.py')


    ds = xr.open_dataset(data_dir / 'rsl_pacific.nc')
    
    calculate_ntr(ds)


