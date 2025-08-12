RSL_FILENAME = 'rsl_pacific.nc'
# STATION_IDS = [50, 52, 57, 58, 59, 60, 61, 552]
STATION_IDS = [25,38]
RUN_WITHOUT_MODEL = True
RETURN_PERIOD = [2, 10, 50, 100]
NUM_PROCESSES = 8
CLIMATE_INDEX = ['AO', 'AAO', 'BEST', 'DMI', 'ONI', 'PDO', 'PMM', 'PNA', 'TNA']
BASE_DIR = '/Users/juliafiedler/Desktop/NSGEV_Test/'

# Define the limits for the parameters of the nonstationary GEV model
# These limits are based on the expected range of values for each parameter
# The indices correspond to the parameters in the model.
LIMITS = {
    1:  [0.01, 7.5], 
    2:  [0.001, 2.5], 
    3:  [-0.35, 0.15],
    4:  [-2.001, 2.001],
    5:  [-2.001, 2.001],
    6:  [-0.5, 0.5],
    7:  [-0.5, 0.5],
    8:  [-0.25, 0.25],
    9:  [-0.25, 0.25],
    10: [-0.3, 0.3],
    11: [-0.5, 0.5],
    12: [-0.2, 0.2],
    13: [-0.2, 0.2],
    14: [-0.2, 0.2]
}