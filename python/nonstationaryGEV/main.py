#%%
from models import run_CI_models, run_noClimateIndex_models
from helpers import make_directories, make_directoryDict
import os
import xarray as xr
from config import RSL_FILENAME, STATION_IDS, RUN_WITHOUT_MODEL, RETURN_PERIOD, NUM_PROCESSES, CLIMATE_INDEX, BASE_DIR, LIMITS


#%%
def load_and_clean_data(data_dir,rsl_filename, station_ids):
    """Load and clean the sea level dataset."""
    with xr.open_dataset(data_dir / rsl_filename) as rsl:
        rsl_hourly = rsl.sel(record_id=rsl.record_id.isin(station_ids)).load()
    return rsl_hourly

def main(
        rsl_filename=RSL_FILENAME,
        station_ids=STATION_IDS, 
        runWithoutModel=RUN_WITHOUT_MODEL, 
        returnPeriod=RETURN_PERIOD, 
        climateIndex=CLIMATE_INDEX, 
        numProcesses=NUM_PROCESSES,
        base_dir=BASE_DIR,
        limits=LIMITS):
    """ Main function to run the nonstationary GEV models for sea level data. """
    
    # base_dir = os.getcwd()  # Get the current working directory
    print(f"Base directory: {base_dir}")  # This is where the script will run
    dirs = make_directoryDict(base_dir)  # Create directory structure
    
    # Load and clean the dataset
    rsl_hourly = load_and_clean_data(dirs['data_dir'], rsl_filename, station_ids)    
    make_directories(rsl_hourly, dirs)  # Create necessary directories  

    # write limits to a text file
    # check if limits.txt already exists, if not, create it
    if not os.path.exists(dirs['run_dir'] / 'limits.txt'):
        with open(dirs['run_dir'] / 'limits.txt', 'w') as f:
            for value in LIMITS.values():
                line = ' '.join(str(v) for v in value)
                f.write(f"{line}\n")

    # run the models for all recordIDs
    for recordID in STATION_IDS:
        _, _, _, _, _, _, x_N, w_N, wcomp, SignifN = run_noClimateIndex_models(rsl_hourly,recordID,runWithoutModel,dirs, returnPeriod, CIname='None', nproc=numProcesses)
        run_CI_models(rsl_hourly,recordID,False,dirs, returnPeriod, climateIndex,x_N, w_N, wcomp, SignifN, nproc=numProcesses)



#%%
if __name__ == "__main__":
    main(
        rsl_filename=RSL_FILENAME,
        station_ids=STATION_IDS, 
        runWithoutModel=RUN_WITHOUT_MODEL, 
        returnPeriod=RETURN_PERIOD, 
        climateIndex=CLIMATE_INDEX, 
        numProcesses=NUM_PROCESSES,
        base_dir=BASE_DIR,
        limits=LIMITS
    )

