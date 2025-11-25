from pathlib import Path
import pandas as pd
import numpy as np


def _select_series(rsl, station_id, month=None):
    """Return a pandas Series of sea level values for a station.

    Supports xarray Dataset/DataArray shapes used across the notebooks.
    If `month` is provided, the series is filtered to that month.
    """
    # prefer these variable names if present
    var_names = ["sea_level_mhhw", "sea_level", "rsl_anomaly", "rsl"]

    # try to access the data array for sea level
    da = None
    for name in var_names:
        try:
            if name in rsl:
                da = rsl[name]
                break
        except Exception:
            # rsl might be a DataArray with .name set
            try:
                if hasattr(rsl, "name") and rsl.name == name:
                    da = rsl
                    break
            except Exception:
                pass

    if da is None:
        # last resort: if rsl looks like a DataArray or Series, try converting directly
        try:
            return pd.Series(rsl.sel(station_id=station_id).to_series())
        except Exception:
            raise ValueError("Could not find a sea-level variable in the provided dataset")

    # select station id (accepts int or string)
    try:
        sel = da.sel(station_id=station_id)
    except Exception:
        # try matching as string
        try:
            sel = da.sel(station_id=str(station_id))
        except Exception:
            # fallback: try to find matching index and use isel
            try:
                ids = list(da['station_id'].values)
                idx = ids.index(int(station_id))
                sel = da.isel(station_id=idx)
            except Exception:
                raise

    # if month filtering requested
    if month is not None:
        try:
            sel = sel.where(sel['time.month'] == int(month), drop=True)
        except Exception:
            # if selection above fails, try using the coordinating time array
            sel = sel.sel(time=sel['time'].sel(time=sel['time'].dt.month == int(month)))

    # convert to pandas Series
    try:
        series = sel.to_series()
    except Exception:
        # if sel is already a pandas-like object
        series = pd.Series(sel.values, index=pd.to_datetime(sel['time'].values))

    return series.dropna()


def get_top_ten(rsl_subset, station_id, month=None, mode='max', min_sep_days=3):
    """Return a DataFrame of the top 10 independent events for a station.

    Parameters
    - rsl_subset: xarray Dataset/DataArray or object containing sea level data
    - station_id: station identifier (int or str)
    - month: optional month (1-12) to filter to (used by intra-annual notebook)
    - mode: 'max' or 'min'
    - min_sep_days: minimum separation (days) between independent events
    """
    sea_level_series = _select_series(rsl_subset, station_id, month=month)

    if mode == 'max':
        top_values = sea_level_series.nlargest(100)
    elif mode == 'min':
        top_values = sea_level_series.nsmallest(100)
    else:
        raise ValueError('mode must be either "max" or "min"')

    filtered_dates = []
    top_10_values = []

    for date, value in top_values.items():
        if all(abs((pd.to_datetime(date) - pd.to_datetime(added_date)).days) > min_sep_days for added_date in filtered_dates):
            filtered_dates.append(date)
            top_10_values.append((pd.to_datetime(date), float(value)))
        if len(filtered_dates) == 10:
            break

    rank = np.arange(1, len(top_10_values) + 1)

    # try to get station name
    station_name = None
    try:
        if 'station_name' in rsl_subset:
            station_name = str(rsl_subset['station_name'].sel(station_id=str(station_id)).values)
    except Exception:
        try:
            station_name = str(rsl_subset['station_name'].sel(station_id=station_id).values)
        except Exception:
            station_name = ''

    df = pd.DataFrame({
        'rank': rank,
        'date': [t[0] for t in top_10_values],
        'sea level (m)': [t[1] for t in top_10_values]
    })

    if month is not None:
        # keep column name compatible with intra-annual notebook
        df = df.rename(columns={'rank': 'rank in month'})
        df['month'] = int(month)

    df['station_name'] = station_name
    df['station_id'] = station_id
    df['type'] = mode
    df['date'] = pd.to_datetime(df['date']).dt.round('h')

    return df


def _find_default_oni_path():
    """Look for an oni.csv file in a few common locations relative to the repo.

    Returns a pathlib.Path if found, otherwise None.
    """
    repo_root = Path(__file__).resolve().parents[3]
    candidates = [
        Path.cwd() / 'climate_indices' / 'oni.csv',
        Path.cwd() / 'data' / 'climate_indices' / 'oni.csv',
        repo_root / 'data' / 'climate_indices' / 'oni.csv',
        repo_root / 'climate_indices' / 'oni.csv',
        repo_root / 'data' / 'oni.csv',
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _prepare_oni(oni_df):
    """Ensure ONI DataFrame has needed columns and mode classification.

    Expects an index of datetimes (monthly). Adds columns 'El Nino', 'La Nina',
    and 'ONI Mode' similar to the notebooks.
    """
    oni = oni_df.copy()
    # ensure datetime index
    if not isinstance(oni.index, pd.DatetimeIndex):
        try:
            oni.index = pd.to_datetime(oni.index)
        except Exception:
            pass

    # Criteria from notebook: 5 consecutive periods
    # Previous approach used a trailing rolling sum which only marked the last
    # month of a qualifying run. Here we mark all months that belong to any
    # run of >=5 consecutive months meeting the ONI threshold.
    if 'ONI' in oni.columns:
        # positive runs (El Niño)
        mask_pos = oni['ONI'] > 0.5
        # create run ids for consecutive-equality groups
        run_id_pos = (mask_pos != mask_pos.shift()).cumsum()
        run_len_pos = mask_pos.groupby(run_id_pos).transform('size')
        oni['El Nino'] = mask_pos & (run_len_pos >= 5)

        # negative runs (La Niña)
        mask_neg = oni['ONI'] < -0.5
        run_id_neg = (mask_neg != mask_neg.shift()).cumsum()
        run_len_neg = mask_neg.groupby(run_id_neg).transform('size')
        oni['La Nina'] = mask_neg & (run_len_neg >= 5)

        oni['ONI Mode'] = 'Neutral'
        oni.loc[oni['La Nina'] == True, 'ONI Mode'] = 'La Nina'
        oni.loc[oni['El Nino'] == True, 'ONI Mode'] = 'El Nino'
    return oni


def get_top_10_table(rsl_subset, station_id, oni_df=None, oni_path: str = None):
    """Return a concatenated DataFrame of top 10 max and min events for a station.

    Parameters
    - rsl_subset, station_id: forwarded to get_top_ten
    - oni_df: optional pandas DataFrame of ONI indexed by datetime (monthly)
    - oni_path: optional path to a csv file containing ONI (overrides default search)

    If ONI data is available (either via ``oni_df`` or a csv at ``oni_path`` or
    a few common locations), the function will merge monthly ONI values and a
    derived 'ONI Mode' into the returned table.
    """
    top_10_values_max = get_top_ten(rsl_subset, station_id, mode='max')
    top_10_values_min = get_top_ten(rsl_subset, station_id, mode='min')

    top_10_table = pd.concat([top_10_values_max, top_10_values_min], ignore_index=True)

    # attempt to merge ONI if requested / available
    oni = None
    if oni_df is not None:
        oni = _prepare_oni(oni_df)
    else:
        # if oni_path explicitly provided, try that first
        if oni_path:
            p = Path(oni_path)
            if p.exists():
                try:
                    oni = pd.read_csv(p, index_col='time', parse_dates=True)
                    oni = _prepare_oni(oni)
                except Exception:
                    # try without index_col
                    try:
                        oni = pd.read_csv(p, parse_dates=True)
                        oni = _prepare_oni(oni)
                    except Exception:
                        oni = None
        else:
            # search a few default locations
            p = _find_default_oni_path()
            if p is not None:
                try:
                    oni = pd.read_csv(p, index_col='time', parse_dates=True)
                    oni = _prepare_oni(oni)
                except Exception:
                    try:
                        oni = pd.read_csv(p, parse_dates=True)
                        oni = _prepare_oni(oni)
                    except Exception:
                        oni = None

    if oni is not None and 'ONI' in oni.columns:
        # align oni to month-period; ensure we have a monthly timestamp index
        try:
            oni_month = oni.copy()
            oni_month.index = pd.to_datetime(oni_month.index).to_period('M').to_timestamp()
            oni_month = oni_month[~oni_month.index.duplicated(keep='first')]

            # create merge key in top_10_table: month start timestamp
            top_10_table['_oni_month'] = pd.to_datetime(top_10_table['date']).dt.to_period('M').dt.to_timestamp()

            top_10_table = top_10_table.merge(oni_month[['ONI', 'ONI Mode']],
                                              left_on='_oni_month', right_index=True,
                                              how='left')

            # ensure columns exist even if missing
            if 'ONI' not in top_10_table.columns:
                top_10_table['ONI'] = np.nan
            if 'ONI Mode' not in top_10_table.columns:
                top_10_table['ONI Mode'] = 'Neutral'

            # cleanup helper column
            top_10_table = top_10_table.drop(columns=['_oni_month'])
        except Exception:
            # if anything goes wrong, just return table without ONI
            pass

    return top_10_table
