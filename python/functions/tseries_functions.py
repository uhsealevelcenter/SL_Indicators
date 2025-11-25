#make sure this can be read as a module
#%%

import numpy as np
import pandas as pd
from scipy import stats

from scipy.fftpack import fft, ifft, fftfreq
from scipy.signal import butter, filtfilt


def segment_by_nan_runs(df, value_column='value', max_nan_run=24*7):
    """
    Splits a DataFrame into segments based on consecutive NaNs in the value column.
    Allows isolated NaNs (up to max_nan_run in a row) but breaks the segment when a block exceeds that.
    
    Parameters:
    - df: pandas DataFrame assumed to be sorted by time.
    - value_column: name of the column with the measurement data.
    - max_nan_run: maximum number of consecutive NaNs allowed in a segment.
    
    Returns:
    - segments: List of DataFrame segments (each segment is a subset of df).
    """
    segments = []
    current_indices = []  # list of row indices for the current segment
    consecutive_nans = 0

    # Iterate over the DataFrame rows (assuming the DataFrame is sorted by time)
    for idx, row in df.iterrows():
        if pd.isna(row[value_column]):
            consecutive_nans += 1
        else:
            consecutive_nans = 0
        
        # Always add the current index to the current segment
        current_indices.append(idx)
        
        # If we have exceeded the allowed consecutive NaNs,
        # then end the current segment just before the long NaN block.
        if consecutive_nans > max_nan_run:
            # Remove the trailing block of NaNs from the current segment.
            valid_segment = current_indices[:-consecutive_nans]
            if valid_segment:
                segments.append(df.loc[valid_segment].copy())
            # Reset: start a new segment after the NaN block.
            current_indices = []  
            consecutive_nans = 0

    # Add the final segment if there are any remaining indices.
    if current_indices:
        segments.append(df.loc[current_indices].copy())
        
    return segments

def filter_segments(segments, value_column='value', freq_cutoff=1/7, order=4):
    """
    Apply the Butterworth low-pass filter to each continuous segment.
    
    Parameters:
    - segments: List of DataFrame segments.
    - value_column: Name of the column containing the time series data.
    - freq_cutoff: Cutoff frequency in cycles per day.
    - order: Filter order.
    
    Returns:
    - filtered_segments: List of DataFrames with additional columns for the filtered signal and high-frequency residual.
    """
    filtered_segments = []
    
    for seg in segments:
        # Only process segments with enough data
        if len(seg) < 1/(freq_cutoff*5*24):
            continue
        
        # Create an array of time differences (assumes hourly data)
        time_diffs = np.ones(len(seg))
        
        # Get the raw data from the segment
        data = seg[value_column].values
        
        # Apply the Butterworth low-pass filter
        filtered, highFreq = butterworth_lowpass(data, time_diffs, freq_cutoff, order=order)
        
        # Optionally, you might want to store the filtered results in the segment DataFrame
        seg = seg.copy()  # To avoid modifying the original segment
        seg['filtered'] = filtered
        seg['highFreq'] = highFreq
        
        filtered_segments.append(seg)
    
    return filtered_segments

def filter_known_frequency_components(ntr, time_diffs, freq_to_filter, width=0.01):
    """
    Apply a soft band-stop filter to remove energy near a specified frequency.
    
    Parameters:
    - ntr: The non-tidal residuals time series (array).
    - time_diffs: Array of time step differences (in hours).
    - freq_to_filter: Target frequency to remove (in cycles per day).
    - width: Controls the width of the notch filter (default 0.01 cpd).
    
    Returns:
    - ntr_filtered: The filtered signal with reduced power at freq_to_filter.
    """
    # check if hourly data
    if np.any(np.array(time_diffs) != 1):
        raise ValueError("Input time_diffs must be in hours.")
    
    # Ensure ntr has no NaNs
    ntr = np.array(ntr)
    if np.any(np.isnan(ntr)):
        # raise ValueError("Input contains NaNs. Please remove or interpolate NaNs before filtering.")
        # ntr = np.nan_to_num(ntr)
        # fill nans with noise represenative of the data
        # Compute standard deviation ignoring NaNs
        noise_std = np.nanstd(ntr)

        # Identify NaN positions in ntr
        nans = np.isnan(ntr)
        nancount = np.sum(nans)
        nanfrac = nancount / len(ntr)

        # Replace each NaN with a random number drawn from a normal distribution
        # with mean 0 and standard deviation equal to the computed noise_std
        ntr[nans] = np.random.normal(0, noise_std, size=np.sum(nans))
        # print out what fraction of the data was filled with noise
        print(f"Filled {100*nanfrac:.0f} % of data with noise (std = {noise_std:.4f}).")

    # Compute the FFT of the non-tidal residuals
    n = len(ntr)
    dt = np.median(time_diffs)  # Should be 1
    freqs = fftfreq(n, d=dt) * 24  # Convert Hz to cpd (cycles per day)
    fft_values = fft(ntr)  # Compute FFT

    # Design a soft notch filter using a Gaussian function
    notch = np.exp(-((freqs - freq_to_filter) / width)**2)  # Gaussian taper
    notch += np.exp(-((freqs + freq_to_filter) / width)**2)  # Mirror for negative frequencies
    notch = 1 - notch  # Convert to a notch filter (1 at most frequencies, 0 at target)

    # Apply the notch filter in frequency space
    filtered_fft_values = fft_values * notch

    # Inverse FFT to reconstruct the filtered time series
    ntr_filtered = np.real(ifft(filtered_fft_values))
    if np.any(np.isnan(ntr)):
        ntr_filtered[nans] = np.nan  # Restore NaNs in filtered signal

    # known signal
    known_signal = ntr - ntr_filtered


    return ntr_filtered, known_signal

def butterworth_lowpass(ntr, time_diffs, freq_cutoff, order=4, padtype='odd', padlen=None):
    """
    Apply a Butterworth low-pass filter to remove high-frequency components.
    
    Parameters:
    - ntr: The non-tidal residuals time series (array).
    - time_diffs: Array of time step differences (in hours).
    - freq_cutoff: Cutoff frequency in cycles per day (cpd).
    - order: Order of the Butterworth filter (default = 4 for smooth roll-off).
    
    Returns:
    - ntr_filtered: The filtered signal with high frequencies attenuated.
    """
    # check if hourly data
    if np.any(np.array(time_diffs) != 1):
        raise ValueError("Input time_diffs must be in hours.")
    
    # Ensure ntr has no NaNs
    ntr = np.array(ntr)
    if np.any(np.isnan(ntr)):
        # raise ValueError("Input contains NaNs. Please remove or interpolate NaNs before filtering.")
        # Compute standard deviation ignoring NaNs
        noise_std = np.nanstd(ntr)
        # Identify NaN positions in ntr
        nans = np.isnan(ntr)
        nancount = np.sum(nans)
        nanfrac = nancount / len(ntr)
        # Replace each NaN with a random number drawn from a normal distribution
        # with mean 0 and standard deviation equal to the computed noise_std
        ntr[nans] = np.random.normal(0, noise_std, size=np.sum(nans))
        # print out what fraction of the data was filled with noise
        print(f"Filled {100*nanfrac:.0f} % of data with noise (std = {noise_std:.4f}).")

    # Compute sampling frequency (fs) in cycles per day
    fs = 1 / (np.median(time_diffs))  # cycles per hour
    fs *= 24  # Convert to cycles per day (cpd)

    # Normalize the cutoff frequency (must be between 0 and 1 for butter)
    nyquist = fs / 2  # Nyquist frequency (max meaningful frequency)
    normalized_cutoff = freq_cutoff / nyquist  # Convert to fraction of Nyquist


    # Design Butterworth low-pass filter
    b, a = butter(order, normalized_cutoff, btype='low', analog=False)

    # Apply the filter with filtfilt (zero-phase filtering)
    if padlen is None:
        ntr_filtered = filtfilt(b, a, ntr,padtype=padtype)
    else:
        ntr_filtered = filtfilt(b, a, ntr,padtype=padtype, padlen=padlen * (max(len(a), len(b))  - 1))

    highFreq = ntr - ntr_filtered
    if np.any(np.isnan(ntr)):
        highFreq[nans] = np.nan  # Restore NaNs in filtered signal

    return ntr_filtered, highFreq

def butterworth_bandpass(ntr,time_diffs,low_cutoff,high_cutoff,order=4):
    """
    Apply a Butterworth bandpass filter to retain only frequencies within a range.
    
    Parameters:
    - ntr: The non-tidal residuals time series (array).
    - time_diffs: Array of time step differences (in hours).
    - low_cutoff: Lower bound of frequency range (cycles per day, cpd).
    - high_cutoff: Upper bound of frequency range (cpd).
    - order: Order of the Butterworth filter (default = 4 for smooth roll-off).
    
    Returns:
    - ntr_filtered: The filtered signal with only desired frequencies retained.
    """
    # Ensure ntr has no NaNs
    ntr = np.array(ntr)
    if np.any(np.isnan(ntr)):
        raise ValueError("Input contains NaNs. Please remove or interpolate NaNs before filtering.")

    # check if hourly data
    if np.any(np.array(time_diffs) != 1):
        raise ValueError("Input time_diffs must be in hours.")

    # Compute sampling frequency (fs) in cycles per day
    fs = 1 / (np.median(time_diffs))  # cycles per hour
    fs *= 24  # Convert to cycles per day (cpd)

    # Normalize cutoff frequencies (must be between 0 and 1 for butter)
    nyquist = fs / 2  # Nyquist frequency (max meaningful frequency)
    low = low_cutoff / nyquist
    high = high_cutoff / nyquist

    # Ensure valid frequency range
    if low <= 0 or high >= 1:
        raise ValueError(f"Invalid frequency range: {low:.5f} - {high:.5f}. Adjust `width` or `notch_freq`.")


    # Design Butterworth bandpass filter
    b, a = butter(order, [low, high], btype='bandpass', analog=False)

    # Apply the filter with filtfilt (zero-phase filtering)
    ntr_filtered = filtfilt(b, a, ntr, padtype='odd')
    ntr_filtered_residuals = ntr - ntr_filtered

    return ntr_filtered, ntr_filtered_residuals

def process_trend_with_nan(sea_level_data, time_column='time', weighted=False):
    """
    Compute trends for sea level anomaly data, handling NaNs.
    
    Supports both pandas.DataFrame and xarray.DataArray.
    
    Parameters:
    - sea_level_data: pandas.DataFrame or xarray.DataArray
    - time_column: str, column name for time in a DataFrame (ignored for xarray)
    - weighted: bool, whether to apply cosine weighting (only applies to gridded xarray data)

    Returns:
    - trend_mag: Trend magnitude over time
    - sea_level_trend: Trend values
    - trend_rate: Rate of change per year
    """
    import xarray as xr


    # Handle xarray input
    if isinstance(sea_level_data, xr.DataArray):
        print("Processing xarray DataArray (vectorized polyfit, time in years)...")
        # Ensure time is the first dimension
        if sea_level_data.dims[0] != 'time':
            sea_level_data = sea_level_data.transpose('time', ...)
        # Add a time_in_years coordinate
        time_years = sea_level_data['time'].dt.year + (sea_level_data['time'].dt.dayofyear - 1) / 365.25
        sea_level_data = sea_level_data.assign_coords(time_in_years=("time", time_years.data))
        # Use polyfit along time_in_years
        pf = sea_level_data.polyfit(dim='time_in_years', deg=1, skipna=True)
        slope = pf.polyfit_coefficients.sel(degree=1)
        intercept = pf.polyfit_coefficients.sel(degree=0)
        # Compute trend at each time step
        trend = slope * sea_level_data['time_in_years'] + intercept
        # Detrended data
        detrended = sea_level_data - trend
        # Trend magnitude and rate
        trend_mag = (trend.isel(time=-1) - trend.isel(time=0))
        time_mag = float(sea_level_data['time_in_years'].isel(time=-1) - sea_level_data['time_in_years'].isel(time=0))
        trend_rate = trend_mag / time_mag
        # Weighted mean if requested
        if weighted:
            if 'latitude' in sea_level_data.dims:
                weights = np.cos(np.deg2rad(sea_level_data.latitude))
                weights.name = 'weights'
                if 'longitude' in sea_level_data.dims:
                    weights_broadcast, _ = xr.broadcast(weights, sea_level_data.longitude)
                else:
                    weights_broadcast = weights
                trend_mag_weighted = (trend_mag * weights_broadcast).mean().item()
                trend_rate_weighted = (trend_rate * weights_broadcast).mean().item()
                sea_level_trend_weighted = (trend * weights_broadcast).mean(dim=['latitude', 'longitude'])
                return trend_mag_weighted, sea_level_trend_weighted, trend_rate_weighted
            else:
                raise ValueError("Weighted trend calculation requires latitude coordinate.")
        return trend_mag, trend, trend_rate

    # Handle DataFrame input
    elif isinstance(sea_level_data, pd.DataFrame):
        if time_column not in sea_level_data.columns:
            raise ValueError(f"Time column '{time_column}' not found in DataFrame.")

        # Convert time to Julian date
        # time_index = sea_level_data[time_column].dt.to_julian_date()
        time_index = (sea_level_data[time_column] - sea_level_data[time_column].iloc[0]).dt.total_seconds() / (24 * 3600)

        # time_index = pd.to_datetime(sea_level_data[time_column]).dt.to_julian_date()
        # time_index = sea_level_data[time_column].to_julian_date().values

        # Exclude the time column from analysis
        data_columns = [col for col in sea_level_data.columns if col != time_column]

        trend_magnitudes = {}
        trend_series = {}
        trend_rates = {}

        for col in data_columns:
            y = sea_level_data[col].values
            mask = ~np.isnan(y)

            if np.any(mask):
                time_masked = time_index[mask]
                y_masked = y[mask]

                slope, intercept, _, _, _ = stats.linregress(time_masked, y_masked)
                trend = slope * time_index + intercept
                trend = trend.to_numpy()  # Convert to NumPy array
                detrended = y - trend
                trend_mag = trend[-1] - trend[0]
                time_mag = (pd.to_datetime(sea_level_data[time_column].iloc[-1]) - pd.to_datetime(sea_level_data[time_column].iloc[0])).days / 365.25
                trend_rate = trend_mag / time_mag

                trend_magnitudes[col] = trend_mag
                trend_series[col] = trend
                trend_rates[col] = trend_rate
            else:
                trend_magnitudes[col] = np.nan
                trend_series[col] = np.nan
                trend_rates[col] = np.nan

        return trend_magnitudes, trend_series, trend_rates

    else:
        raise TypeError("Input must be a pandas DataFrame or xarray DataArray.")



# write a test for process_trend_with_nan using rsl_daily_pacific.nc as input

#%% TESTING
if __name__ == "__main__":
    import xarray as xr
    from pathlib import Path
    import os

    #set up directories as Path objects
    data_dir = Path(os.environ["DATA_DIR"]).expanduser()
    output_dir = Path(os.environ["OUTPUT_DIR"]).expanduser()

    rsl_daily = xr.open_dataset(data_dir / 'rsl_daily_pacific.nc')
    trend_mag_rsl, trend_line_rsl, trend_rate_rsl = process_trend_with_nan(rsl_daily.rsl_anomaly, time_column='time',weighted=False)

# %%
