"""
This script automates the downloading and pre-processing of all required datasets,
producing pre-packaged input-output pairs for streamlined model training and evaluation.
"""

import cdsapi
import os
import xarray as xr
from dask import delayed
import numpy as np
import torch
from datetime import datetime
import pandas as pd
import config


def generate_file_paths(prefix, years, months):
    """
    Generate a list of file paths for given prefix, years, and months.
    """
    return [f"{config.DATA_DIR}/{prefix}_{year}_{month:02d}.nc" 
            for year in years for month in months]


@delayed
def load_dataset(fp, chunks):
    """
    Lazily open an xarray dataset with specified chunking.
    """
    return xr.open_dataset(fp, chunks=chunks)


def load_and_concat(prefix, years, months, chunks, dim='valid_time'):
    """
    Load multiple netCDF files in parallel and concatenate along a dimension.
    """
    paths = generate_file_paths(prefix, years, months)
    delayed_ds = [load_dataset(path, chunks) for path in paths]
    computed = [ds.compute() for ds in delayed_ds]
    return xr.concat(computed, dim=dim)


# ================================================================================================
# ERA5 Data Fetching Functions
# ================================================================================================

def fetch_precipitation_data():
    """Fetch total precipitation data from ERA5."""
    client = cdsapi.Client(url=config.CDS_API_URL, key=config.CDS_API_KEY)
    days = [f"{i:02d}" for i in range(1, 32)]
    hourly_times = [f"{i:02d}:00" for i in range(24)]
    
    for year in config.YEARS:
        for month in config.MONTHS:
            request = {
                "product_type": ["reanalysis"],
                "variable": ["total_precipitation"],
                "year": [str(year)],
                "month": [f"{month:02d}"],
                "day": days,
                "time": hourly_times,
                "data_format": "netcdf",
                "download_format": "unarchived"
            }
            filename = f'{config.DATA_DIR}/era5_tp_data_{year}_{month:02d}.nc'
            client.retrieve("reanalysis-era5-single-levels", request).download(filename)


def fetch_2m_temperature_data():
    """Fetch 2m temperature data from ERA5."""
    client = cdsapi.Client(url=config.CDS_API_URL, key=config.CDS_API_KEY)
    days = [f"{i:02d}" for i in range(1, 32)]
    three_hourly_times = ["00:00", "03:00", "06:00", "09:00", 
                          "12:00", "15:00", "18:00", "21:00"]
    
    for year in config.YEARS:
        for month in config.MONTHS:
            request = {
                "product_type": ["reanalysis"],
                "variable": ["2m_temperature"],
                "year": [str(year)],
                "month": [f"{month:02d}"],
                "day": days,
                "time": three_hourly_times,
                "data_format": "netcdf",
                "download_format": "unarchived"
            }
            filename = f'{config.DATA_DIR}/era5_2m_temperature_{year}_{month:02d}.nc'
            client.retrieve("reanalysis-era5-single-levels", request).download(filename)


def fetch_sst_data():
    """Fetch sea surface temperature data from ERA5."""
    client = cdsapi.Client(url=config.CDS_API_URL, key=config.CDS_API_KEY)
    days = [f"{i:02d}" for i in range(1, 32)]
    three_hourly_times = ["00:00", "03:00", "06:00", "09:00", 
                          "12:00", "15:00", "18:00", "21:00"]
    
    for year in config.YEARS:
        for month in config.MONTHS:
            request = {
                "product_type": ["reanalysis"],
                "variable": ["sea_surface_temperature"],
                "year": [str(year)],
                "month": [f"{month:02d}"],
                "day": days,
                "time": three_hourly_times,
                "data_format": "netcdf",
                "download_format": "unarchived"
            }
            filename = f'{config.DATA_DIR}/era5_sst_data_{year}_{month:02d}.nc'
            client.retrieve("reanalysis-era5-single-levels", request).download(filename)


def fetch_pressure_level_data():
    """Fetch pressure level data from ERA5."""
    client = cdsapi.Client(url=config.CDS_API_URL, key=config.CDS_API_KEY)
    days = [f"{i:02d}" for i in range(1, 32)]
    three_hourly_times = ["00:00", "03:00", "06:00", "09:00", 
                          "12:00", "15:00", "18:00", "21:00"]
    
    for year in config.YEARS:
        for month in config.MONTHS:
            request = {
                "product_type": ["reanalysis"],
                "variable": [
                    "potential_vorticity",
                    "specific_humidity",
                    "temperature",
                    "u_component_of_wind",
                    "v_component_of_wind"
                ],
                "year": [str(year)],
                "month": [f"{month:02d}"],
                "day": days,
                "time": three_hourly_times,
                "pressure_level": ["500", "750"],
                "data_format": "netcdf",
                "download_format": "unarchived"
            }
            filename = f'{config.DATA_DIR}/era5_pressure_data_{year}_{month:02d}.nc'
            client.retrieve("reanalysis-era5-pressure-levels", request).download(filename)


def fetch_wave_data():
    """Fetch wave data from ERA5."""
    client = cdsapi.Client(url=config.CDS_API_URL, key=config.CDS_API_KEY)
    days = [f"{i:02d}" for i in range(1, 32)]
    three_hourly_times = ["00:00", "03:00", "06:00", "09:00", 
                          "12:00", "15:00", "18:00", "21:00"]
    
    for year in config.YEARS:
        for month in config.MONTHS:
            request = {
                "product_type": ["reanalysis"],
                "variable": [
                    "significant_height_of_combined_wind_waves_and_swell",
                    "maximum_individual_wave_height"
                ],
                "year": [str(year)],
                "month": [f"{month:02d}"],
                "day": days,
                "time": three_hourly_times,
                "data_format": "netcdf",
                "download_format": "unarchived"
            }
            filename = f'{config.DATA_DIR}/era5_wh_data_{year}_{month:02d}.nc'
            client.retrieve("reanalysis-era5-single-levels", request).download(filename)


def fetch_model_bathymetry_data():
    """Fetch model bathymetry data from ERA5."""
    client = cdsapi.Client(url=config.CDS_API_URL, key=config.CDS_API_KEY)
    days = [f"{i:02d}" for i in range(1, 32)]
    three_hourly_times = ["00:00", "03:00", "06:00", "09:00", 
                          "12:00", "15:00", "18:00", "21:00"]
    
    for year in config.YEARS:
        for month in config.MONTHS:
            request = {
                "product_type": ["reanalysis"],
                "variable": ["model_bathymetry"],
                "year": [str(year)],
                "month": [f"{month:02d}"],
                "day": days,
                "time": three_hourly_times,
                "data_format": "netcdf",
                "download_format": "unarchived",
            }
            filename = f'{config.DATA_DIR}/era5_bathymetry_data_{year}_{month:02d}.nc'
            client.retrieve("reanalysis-era5-single-levels", request).download(filename)


def fetch_all_era5_data():
    """Fetch all ERA5 data."""
    print("Fetching all ERA5 data...")
    fetch_precipitation_data()
    fetch_2m_temperature_data()
    fetch_sst_data()
    fetch_pressure_level_data()
    fetch_wave_data()
    fetch_model_bathymetry_data()
    print("All ERA5 data fetched successfully!")


# ================================================================================================
# Data Loading and Processing Functions
# ================================================================================================

def load_precipitation_dataset():
    """Load and process precipitation dataset."""
    # Load pressure-level data and extract levels
    ds_pressure = load_and_concat(
        config.ERA5_PREFIXES['pressure'], 
        config.YEARS, 
        config.MONTHS, 
        config.SMALL_CHUNKS
    )
    for level in (500, 750):
        for var in ('q', 't', 'u', 'v', 'pv'):
            ds_pressure[f"{var}_p{level}"] = ds_pressure[var].sel(pressure_level=level)
    ds_pressure = ds_pressure.drop_vars(['q', 't', 'u', 'v', 'pv', 'pressure_level'])

    # Load 2m temperature
    ds_2m = load_and_concat(
        config.ERA5_PREFIXES['temperature_2m'], 
        config.YEARS, 
        config.MONTHS, 
        config.SMALL_CHUNKS
    )

    # Load sea surface temperature and combine with 2m where missing
    ds_sst = load_and_concat(
        config.ERA5_PREFIXES['sst'], 
        config.YEARS, 
        config.MONTHS, 
        config.SMALL_CHUNKS
    )
    ds_sst['land_mask'] = xr.where(ds_sst['sst'].isnull(), 1, 0)
    ds_sst['sst_and_t2m'] = ds_sst['sst'].where(
        ds_sst['sst'].notnull(), 
        ds_2m['t2m'].where(ds_sst['sst'].isnull())
    )

    # Load total precipitation and compute rolling sum
    ds_tp = load_and_concat(
        config.ERA5_PREFIXES['precipitation'], 
        config.YEARS, 
        config.MONTHS, 
        config.LARGE_CHUNKS
    )
    ds_tp = ds_tp.assign(cum_tp=ds_tp.tp.rolling(valid_time=3, min_periods=3).sum())
    ds_tp = ds_tp.sel(valid_time=ds_tp.valid_time.dt.hour.isin(config.PRECIPITATION_HOURS))
    ds_tp = ds_tp.chunk(config.SMALL_CHUNKS)

    # Merge all datasets into one
    ds_combined = xr.merge([ds_pressure, ds_sst, ds_tp])
    
    return ds_combined


def load_storm_surge_dataset():
    """Load and process storm surge dataset (includes wave data)."""
    # Start with precipitation dataset
    ds_combined = load_precipitation_dataset()
    
    # Load wave and bathymetry data
    ds_wave = load_and_concat(
        config.ERA5_PREFIXES['wave'], 
        config.YEARS, 
        config.MONTHS, 
        config.SMALL_CHUNKS
    )
    ds_bathymetry = load_and_concat(
        config.ERA5_PREFIXES['bathymetry'], 
        config.YEARS, 
        config.MONTHS, 
        config.SMALL_CHUNKS
    )

    # Interpolate wave and bathymetry data to match resolution
    # Get wave data - try common variable names
    if 'swh' in ds_wave.data_vars:
        wave_data = ds_wave['swh'].values
    elif 'significant_height_of_combined_wind_waves_and_swell' in ds_wave.data_vars:
        wave_data = ds_wave['significant_height_of_combined_wind_waves_and_swell'].values
    else:
        # Get first data variable
        wave_var_name = list(ds_wave.data_vars)[0]
        wave_data = ds_wave[wave_var_name].values
    
    # Get bathymetry data
    if 'wmb' in ds_bathymetry.data_vars:
        bathymetry_data = ds_bathymetry['wmb'].values
    elif 'model_bathymetry' in ds_bathymetry.data_vars:
        bathymetry_data = ds_bathymetry['model_bathymetry'].values
    else:
        bathymetry_var_name = list(ds_bathymetry.data_vars)[0]
        bathymetry_data = ds_bathymetry[bathymetry_var_name].values
    
    # Convert to tensors and add channel dimension for interpolation
    # Interpolation expects (N, C, H, W) format
    wave_tensor = torch.tensor(wave_data, dtype=torch.float32)
    if wave_tensor.dim() == 3:  # (time, lat, lon) -> add channel dim
        wave_tensor = wave_tensor.unsqueeze(1)  # (time, 1, lat, lon)
    
    bathymetry_tensor = torch.tensor(bathymetry_data, dtype=torch.float32)
    if bathymetry_tensor.dim() == 3:
        bathymetry_tensor = bathymetry_tensor.unsqueeze(1)
    
    # Interpolate to target size (721, 1440) for lat/lon
    wave_interpolated = torch.nn.functional.interpolate(
        wave_tensor, size=(721, 1440), mode='nearest'
    ).squeeze(1).numpy()  # Remove channel dim, keep (time, lat, lon)
    
    bathymetry_interpolated = torch.nn.functional.interpolate(
        bathymetry_tensor, size=(721, 1440), mode='nearest'
    ).squeeze(1).numpy()

    ds_combined['swh'] = xr.DataArray(
        wave_interpolated, 
        dims=['valid_time', 'latitude', 'longitude']
    )
    ds_combined['wmb'] = xr.DataArray(
        bathymetry_interpolated, 
        dims=['valid_time', 'latitude', 'longitude']
    )

    return ds_combined


# ================================================================================================
# Storm Event Processing Functions
# ================================================================================================

def process_precipitation_storm_events(storm_events_df, ds_combined, output_suffix=""):
    """
    Process storm events for precipitation prediction.
    
    Args:
        storm_events_df: DataFrame with storm event information
        ds_combined: Combined xarray dataset
        output_suffix: Suffix for output filenames (e.g., "_2019_2020")
    """
    features_list, targets_list = [], []
    
    for idx, row in storm_events_df.iterrows():
        try:
            # Extract storm data
            storm_lat, storm_lon = row['lat'], row['lon']
            radius_45_225 = row['precipitation_radius1']
            radius_135_315 = row['precipitation_radius2']
            
            # Define latitude and longitude bounds centred on the storm
            lat_min = storm_lat - config.LAT_LON_RANGE/2
            lat_max = storm_lat + config.LAT_LON_RANGE/2
            lon_min = storm_lon - config.LAT_LON_RANGE/2
            lon_max = storm_lon + config.LAT_LON_RANGE/2
            
            # Convert time string to datetime
            time = np.datetime64(datetime.strptime(row['iso_time_str'], '%Y-%m-%d %H:%M:%S'))
            
            # Filter data within the lat-lon bounds
            subset_ds = ds_combined.sel(
                valid_time=time,
                latitude=slice(lat_max, lat_min),
                longitude=slice(lon_min, lon_max)
            )
            
            # Skip if subset is empty
            if subset_ds.sizes['latitude'] == 0 or subset_ds.sizes['longitude'] == 0:
                print(f"Skipping empty subset for storm at {storm_lat}, {storm_lon}, time {row['iso_time_str']}")
                continue
            
            # Extract feature variables
            feature_arrays = []
            for var in config.PRECIPITATION_FEATURES:
                if var in subset_ds:
                    feature_arrays.append(subset_ds[var].values)
                else:
                    print(f"Warning: Variable {var} not found in dataset")
                    continue
            
            # Ensure dimensions don't exceed num_points
            lat_values = subset_ds.latitude.values[:config.NUM_POINTS]
            lon_values = subset_ds.longitude.values[:config.NUM_POINTS]
            subset_ds = subset_ds.sel(latitude=lat_values, longitude=lon_values)
            
            # Calculate target variable: sum of cumulative rainfall within storm area
            if 'cum_tp' in subset_ds:
                lats, lons = subset_ds.latitude.values, subset_ds.longitude.values
                lon_grid, lat_grid = np.meshgrid(lons, lats)
                x_dist, y_dist = lon_grid - storm_lon, lat_grid - storm_lat
                
                # Convert radii from nautical miles to degrees
                radius_45_225_deg = radius_45_225 * 1.852 / 111.0
                radius_135_315_deg = radius_135_315 * 1.852 / 111.0
                
                # Rotate coordinates to align with ellipse axes
                cos_45, sin_45 = np.cos(np.radians(45)), np.sin(np.radians(45))
                x_rot = x_dist * cos_45 + y_dist * sin_45
                y_rot = -x_dist * sin_45 + y_dist * cos_45
                
                # Calculate normalised distances for elliptical shape
                normalised_dist = (x_rot**2 / (radius_45_225_deg**2)) + (y_rot**2 / (radius_135_315_deg**2))
                precipitation_mask = normalised_dist <= 1.0
                
                # Get precipitation values within the storm area
                precipitation_values = subset_ds['cum_tp'].values * precipitation_mask
                valid_precip = precipitation_values[precipitation_mask]
                
                if len(valid_precip) > 0:
                    # Calculate various statistics as target values
                    target_values = [
                        np.sum(valid_precip), 
                        np.max(valid_precip), 
                        np.min(valid_precip), 
                        np.mean(valid_precip),
                        np.median(valid_precip), 
                        np.percentile(valid_precip, 25), 
                        np.percentile(valid_precip, 75),
                        np.percentile(valid_precip, 90), 
                        np.percentile(valid_precip, 95), 
                        np.percentile(valid_precip, 99)
                    ]
                else:
                    target_values = [0.0] * 10
                
                # Stack the feature arrays along a new dimension (channels)
                if feature_arrays:
                    feature_arrays.append(precipitation_mask.astype(np.float32))
                    features_tensor = torch.tensor(np.stack(feature_arrays, axis=0), dtype=torch.float32)
                    features_list.append(features_tensor)
                    targets_list.append(torch.tensor(target_values, dtype=torch.float32))
                    
                    # Print progress every 20 rows
                    if idx % 20 == 0 and idx > 0:
                        print(f"Processed {idx} of {len(storm_events_df)} rows. Current features: {len(features_list)}")
        
        except Exception as e:
            print(f"Error processing row {idx} (date: {row['iso_time_str']}): {str(e)}")
            continue
    
    # Save processed data
    if features_list:
        X = torch.stack(features_list)
        y = torch.stack(targets_list)
        
        features_filename = os.path.join(
            config.PROCESSED_DATA_DIR, 
            config.FEATURES_PATTERN.format(f"precipitation{output_suffix}")
        )
        targets_filename = os.path.join(
            config.PROCESSED_DATA_DIR,
            config.TARGETS_PATTERN.format(f"precipitation{output_suffix}")
        )
        
        torch.save(X, features_filename)
        torch.save(y, targets_filename)
        print(f"Saved {len(features_list)} samples to {features_filename} and {targets_filename}")
    
    return features_list, targets_list


def process_storm_surge_events(storm_events_df, ds_combined, output_suffix=""):
    """
    Process storm events for storm surge prediction.
    
    Args:
        storm_events_df: DataFrame with storm event information
        ds_combined: Combined xarray dataset (with wave data)
        output_suffix: Suffix for output filenames
    """
    features_list, targets_list = [], []
    
    for idx, row in storm_events_df.iterrows():
        try:
            # Extract storm data
            storm_lat, storm_lon = row['lat'], row['lon']
            surge_radius_45_225 = row['chosen_length1'] * 3
            surge_radius_135_315 = row['chosen_length2'] * 3
            
            # Define latitude and longitude bounds
            lat_min = storm_lat - config.LAT_LON_RANGE/2
            lat_max = storm_lat + config.LAT_LON_RANGE/2
            lon_min = storm_lon - config.LAT_LON_RANGE/2
            lon_max = storm_lon + config.LAT_LON_RANGE/2
        
            # Convert time string to datetime
            time_obj = datetime.strptime(row['iso_time_str'], '%Y-%m-%d %H:%M:%S')
            closest_time = np.datetime64(time_obj)
            
            # Filter data within the lat-lon bounds
            subset_ds = ds_combined.sel(
                valid_time=closest_time,
                latitude=slice(lat_max, lat_min),
                longitude=slice(lon_min, lon_max)
            )
            
            # Skip if subset is empty
            if subset_ds.sizes['latitude'] == 0 or subset_ds.sizes['longitude'] == 0:
                print(f"Skipping empty subset for storm at {storm_lat}, {storm_lon}, time {row['iso_time_str']}")
                continue
            
            # Extract feature variables
            feature_arrays = []
            for var in config.STORM_SURGE_FEATURES:
                if var in subset_ds:
                    feature_arrays.append(subset_ds[var].values)
                else:
                    print(f"Warning: Variable {var} not found in dataset")
                    continue
            
            # Ensure dimensions don't exceed num_points
            lat_values = subset_ds.latitude.values[:config.NUM_POINTS]
            lon_values = subset_ds.longitude.values[:config.NUM_POINTS]
            if len(subset_ds.latitude) > config.NUM_POINTS or len(subset_ds.longitude) > config.NUM_POINTS:
                subset_ds = subset_ds.sel(latitude=lat_values, longitude=lon_values)
            
            # Calculate target variable: maximum wave height within storm area
            if 'swh' in subset_ds:
                lats, lons = subset_ds.latitude.values, subset_ds.longitude.values
                lon_grid, lat_grid = np.meshgrid(lons, lats)
                x_dist, y_dist = lon_grid - storm_lon, lat_grid - storm_lat
                
                # Convert radii from nautical miles to degrees
                surge_radius_45_225_deg = surge_radius_45_225 * 1.852 / 111.0
                surge_radius_135_315_deg = surge_radius_135_315 * 1.852 / 111.0
                
                # Rotate coordinates by 45 degrees
                cos_45, sin_45 = np.cos(np.radians(45)), np.sin(np.radians(45))
                x_rot = x_dist * cos_45 + y_dist * sin_45
                y_rot = -x_dist * sin_45 + y_dist * cos_45
                
                # Calculate normalised distances for elliptical shape
                normalised_surge_dist = (x_rot**2 / (surge_radius_45_225_deg**2)) + (y_rot**2 / (surge_radius_135_315_deg**2))
                surge_mask = normalised_surge_dist <= 1.0
                
                # Get wave height values within the storm area
                swh_values = subset_ds['swh'].values
                swh_masked = swh_values * surge_mask
                swh_masked_no_nan = np.nan_to_num(swh_masked, nan=0.0)
                
                max_surge_idx = np.unravel_index(np.argmax(swh_masked_no_nan), swh_masked_no_nan.shape)
                max_surge_value = swh_masked_no_nan[max_surge_idx]
                
                # Stack the feature arrays and add to lists
                if len(feature_arrays) > 0:
                    feature_arrays.append(surge_mask.astype(np.float32))
                    features = np.stack(feature_arrays, axis=0)
                    features_tensor = torch.tensor(features, dtype=torch.float32)
                    
                    features_list.append(features_tensor)
                    targets_list.append(torch.tensor([max_surge_value], dtype=torch.float32))
                    
                    # Print progress every 20 rows
                    if idx % 20 == 0 and idx > 0:
                        print(f"Processed {idx} of {len(storm_events_df)} rows. Current features: {len(features_list)}")
        
        except Exception as e:
            print(f"Error processing row {idx} (date: {row['iso_time_str']}): {str(e)}")
            continue
    
    # Save processed data
    if features_list:
        X = torch.stack(features_list)
        y = torch.stack(targets_list)
        
        features_filename = os.path.join(
            config.PROCESSED_DATA_DIR,
            config.FEATURES_PATTERN.format(f"wh{output_suffix}")
        )
        targets_filename = os.path.join(
            config.PROCESSED_DATA_DIR,
            config.TARGETS_PATTERN.format(f"swh{output_suffix}")
        )
        
        torch.save(X, features_filename)
        torch.save(y, targets_filename)
        print(f"Saved {len(features_list)} samples to {features_filename} and {targets_filename}")
    
    return features_list, targets_list


def compile_precipitation_data():
    """Main function to compile precipitation data."""
    print("Loading precipitation dataset...")
    ds_combined = load_precipitation_dataset()
    
    print("Processing storm events...")
    storm_events_path = os.path.join(config.STORM_EVENTS_DIR, config.STORM_EVENTS_PACIFIC)
    storm_events_df = pd.read_csv(storm_events_path)
    
    process_precipitation_storm_events(storm_events_df, ds_combined)
    print("Precipitation data compilation complete!")


def compile_storm_surge_data():
    """Main function to compile storm surge data."""
    print("Loading storm surge dataset...")
    ds_combined = load_storm_surge_dataset()
    
    print("Processing storm events...")
    storm_events_path = os.path.join(config.STORM_EVENTS_DIR, config.STORM_EVENTS_PACIFIC)
    storm_events_df = pd.read_csv(storm_events_path)
    
    process_storm_surge_events(storm_events_df, ds_combined)
    print("Storm surge data compilation complete!")


if __name__ == "__main__":
    # Example usage:
    fetch_all_era5_data()  # Uncomment to fetch ERA5 data
    compile_precipitation_data()
    compile_storm_surge_data()
    pass

