import xarray as xr
from dask import delayed
import numpy as np
import torch
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd

def generate_file_paths(prefix, years, months):
    """
    Generate a list of file paths for given prefix, years, and months.
    """
    return [f"data/{prefix}_{year}_{month:02d}.nc" for year in years for month in months]


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

# Main function to load the dataset and merge all the datasets into one
def main():
    years = [2019, 2020, 2021, 2022, 2023, 2024]
    months = range(1, 13)
    small_chunks = {'valid_time': 8}

    # Load pressure-level data and extract levels
    ds_pressure = load_and_concat('era5_pressure_data', years, months, small_chunks)
    for level in (500, 750):
        for var in ('q', 't', 'u', 'v', 'pv'):
            ds_pressure[f"{var}_p{level}"] = ds_pressure[var].sel(pressure_level=level)
    ds_pressure = ds_pressure.drop_vars(['q', 't', 'u', 'v', 'pv', 'pressure_level'])

    # Load 2m temperature
    ds_2m = load_and_concat('era5_2m_temperature', years, months, small_chunks)

    # Load sea surface temperature and combine with 2m where missing
    ds_sst = load_and_concat('era5_sst_data', years, months, small_chunks)
    ds_sst['land_mask'] = xr.where(ds_sst['sst'].isnull(), 1, 0) # 1 if land, 0 if ocean
    ds_sst['sst_and_t2m'] = ds_sst['sst'].where(ds_sst['sst'].notnull(), ds_2m['t2m'].where(ds_sst['sst'].isnull()))

    # Load total precipitation and compute rolling sum
    large_chunks = {'valid_time': 24}
    ds_tp = load_and_concat('era5_tp_data', years, months, large_chunks)
    ds_tp = ds_tp.assign(cum_tp=ds_tp.tp.rolling(valid_time=3, min_periods=3).sum())
    hours = [3, 6, 9, 12, 15, 18, 21, 0]
    ds_tp = ds_tp.sel(valid_time=ds_tp.valid_time.dt.hour.isin(hours))
    ds_tp = ds_tp.chunk(small_chunks)

    # Merge all datasets into one
    ds_combined = xr.merge([ds_pressure, ds_sst, ds_tp])

    return ds_combined

ds_combined = main()
print(ds_combined)




# Preprocess dataset for CNN model
storm_events_df = pd.read_csv('storm_events_2019to24_pacific.csv')
storm_events_df

# Initialize lists to store features and targets
features_list, targets_list = [], []

# Define the latitude and longitude range (in degrees)
LAT_LON_RANGE = 40
num_points = LAT_LON_RANGE * 4  

# Iterate over each row in storm_events_df
for idx, row in storm_events_df.iterrows():
    try:
        # Extract storm data
        storm_lat, storm_lon = row['lat'], row['lon']
        radius_45_225 = row['precipitation_radius1']
        radius_135_315 = row['precipitation_radius2']
        
        # Define latitude and longitude bounds centered on the storm
        lat_min, lat_max = storm_lat - LAT_LON_RANGE/2, storm_lat + LAT_LON_RANGE/2
        lon_min, lon_max = storm_lon - LAT_LON_RANGE/2, storm_lon + LAT_LON_RANGE/2
        
        # Convert time string to datetime and find time in dataset
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
        
        # Extract the feature variables
        feature_vars = [
            'q_p500', 't_p500', 'u_p500', 'v_p500', 'pv_p500',
            'q_p750', 't_p750', 'u_p750', 'v_p750', 'pv_p750',
            'sst_and_t2m', 'land_mask'
        ]
        
        # Create a list to store feature arrays
        feature_arrays = []
        
        # Ensure dimensions don't exceed num_points
        lat_values, lon_values = subset_ds.latitude.values, subset_ds.longitude.values
        lat_values, lon_values = lat_values[:num_points], lon_values[:num_points]
        
        # Re-select with trimmed coordinates if needed
        subset_ds = subset_ds.sel(latitude=lat_values, longitude=lon_values)
        
        # Extract each feature and add to the list
        for var in feature_vars:
            if var in subset_ds:
                feature_arrays.append(subset_ds[var].values)
            else:
                print(f"Warning: Variable {var} not found in dataset for storm at {storm_lat}, {storm_lon}")
                continue
        
        # Calculate the target variable: sum of cumulative rainfall within storm area
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
            
            # Calculate normalized distances for elliptical shape
            normalized_dist = (x_rot**2 / (radius_45_225_deg**2)) + (y_rot**2 / (radius_135_315_deg**2))
            precipitation_mask = normalized_dist <= 1.0
            
            # Get precipitation values within the storm area 
            precipitation_values = subset_ds['cum_tp'].values * precipitation_mask # inside the storm area, multiply by 1, outside the storm area, multiply by 0
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
                feature_vars.append('precipitation_mask')
                features_tensor = torch.tensor(np.stack(feature_arrays, axis=0), dtype=torch.float32)
                features_list.append(features_tensor)
                targets_list.append(torch.tensor(target_values, dtype=torch.float32))
                
                # Print progress every 20 rows
                if idx % 20 == 0 and idx > 0:
                    print(f"Processed {idx} of {len(storm_events_df)} rows. Current features: {len(features_list)}")
        else:
            print(f"Warning: Variable 'cum_tp' not found in dataset for storm at {storm_lat}, {storm_lon}")
    
    except Exception as e:
        print(f"Error processing row {idx} (date: {row['iso_time_str']}): {str(e)}")
        continue

# Print summary of processed data
print(f"Processed {len(features_list)} storm samples from storm_events_df")
if features_list:
    print(f"Feature tensor shape: {features_list[0].shape}")
    print(f"Feature channels: {feature_vars}")
    print(f"Target tensor shape: {targets_list[0].shape}")
    print(f"Target values include: sum, max, min, mean, median, p25, p75, p90, p95, p99")
    
    # Print range for each target statistic
    target_array = torch.stack(targets_list).numpy()
    for i, stat_name in enumerate(['sum', 'max', 'min', 'mean', 'median', 'p25', 'p75', 'p90', 'p95', 'p99']):
        print(f"{stat_name} range: {np.min(target_array[:, i])} to {np.max(target_array[:, i])}")

if features_list:
    X, y = torch.stack(features_list), torch.stack(targets_list)
    print(f"Final dataset shapes - X: {X.shape}, y: {y.shape}")
    torch.save(X, 'storm_features_precipitation.pt')
    torch.save(y, 'storm_targets_precipitation.pt')
    print("Dataset saved to storm_features_precipitation.pt and storm_targets_precipitation.pt")
