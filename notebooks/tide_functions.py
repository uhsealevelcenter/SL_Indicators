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
    time = ds_epoch.time.values
    time_days = (time - time[0]) / np.timedelta64(1, 'D')

    stations = ds_epoch['record_id'].values

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

        coef_noNodal = solve(time, sea_level, nodal=False, trend=True, method='robust',lat=ds['lat'].sel(record_id=station).values)
        coef_Nodal = solve(time, sea_level, nodal=True, trend=True, method='robust',lat=ds['lat'].sel(record_id=station).values)

        time_ALL = ds.time.values
        sea_level_ALL = ds.sea_level.sel(record_id=station).values

        # find beginning of data in timeseries
        start = np.where(~np.isnan(sea_level_ALL))[0][0]
        time_ALL = time_ALL[start:]
        sea_level_ALL = sea_level_ALL[start:]

        tide_ALL_withNodal = reconstruct(time_ALL, coef_Nodal)
        tide_ALL_noNodal = reconstruct(time_ALL, coef_noNodal)


        ntr = sea_level_ALL - tide_ALL_withNodal.h
        ntr_withNodal = sea_level_ALL - tide_ALL_noNodal.h
        nodal_pred = tide_ALL_withNodal.h - tide_ALL_noNodal.h
        ntr_data = pd.DataFrame({'time': time_ALL, 'ntr': ntr, 'sea_level': sea_level_ALL, 'tide': tide_ALL_withNodal.h, 'nodal': nodal_pred, 'ntr_withNodal': ntr_withNodal})

        #ensure that the time is in datetime format
        ntr_data['time'] = pd.to_datetime(ntr_data['time'])
        ntr_data['time'] = ntr_data['time'].dt.tz_localize(None)

        datapath = Path(savepath, f'ntr_{station:03d}.csv')
        ntr_data.to_csv(datapath, index=False, date_format='%Y-%m-%d %H:%M:%S')
        print(f'Saved data for station {station_name} to {datapath}')

## run the function
calculate_ntr(ds)