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
    ds_sst['land_mask'] = xr.where(ds_sst['sst'].isnull(), 1, 0)
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

    # Load wave and bathymetry data
    ds_wave = load_and_concat('era5_wh_data', years, months, small_chunks)
    ds_bathymetry = load_and_concat('era5_bathymetry_data', years, months, small_chunks)

    wave_tensor = torch.tensor(ds_wave.as_numpy().swh.data).unsqueeze(0)
    bathymetry_tensor = torch.tensor(ds_bathymetry.as_numpy().wmb.data).unsqueeze(0)
    wave_interpolated = torch.nn.functional.interpolate(wave_tensor, size=(721, 1440), mode='nearest').squeeze(0).numpy()
    bathymetry_interpolated = torch.nn.functional.interpolate(bathymetry_tensor, size=(721, 1440), mode='nearest').squeeze(0).numpy()

    ds_combined['swh'] = xr.DataArray(wave_interpolated, dims=['valid_time', 'latitude', 'longitude'])
    ds_combined['wmb'] = xr.DataArray(bathymetry_interpolated, dims=['valid_time', 'latitude', 'longitude'])

    return ds_combined

ds_combined = main()
print(ds_combined)


storm_events_from2019to24 = pd.read_csv('storm_events_2019to24_pacific.csv')
storm_events_from2019to24

# initialize lists to store features and targets
features_list = []
targets_list_swh = []  # For maximum significant wave height

# define the latitude and longitude range (in degrees)
LAT_LON_RANGE = 40
num_points = LAT_LON_RANGE * 4


# iterate over each row in storm_events_df
for idx, row in storm_events_from2019to24.iterrows():
    try:
        # extract storm data
        storm_lat, storm_lon = row['lat'], row['lon']
        # Calculate storm surge mask dimensions
        surge_radius_45_225 = row['chosen_length1'] * 3 # along 45°/225° direction
        surge_radius_135_315 = row['chosen_length2'] * 3  # along 135°/315° direction
        
        # define latitude and longitude bounds centered on the storm
        lat_min = storm_lat - LAT_LON_RANGE/2
        lat_max = storm_lat + LAT_LON_RANGE/2
        lon_min = storm_lon - LAT_LON_RANGE/2
        lon_max = storm_lon + LAT_LON_RANGE/2
    
        # convert time string to datetime and find closest time in dataset
        time_obj = datetime.strptime(row['iso_time_str'], '%Y-%m-%d %H:%M:%S')
        closest_time = np.datetime64(time_obj)
        
        # filter data within the lat-lon bounds
        subset_ds = ds_combined.sel(
            valid_time=closest_time,
            latitude=slice(lat_max, lat_min),  # Fixed: correct order for latitude slice
            longitude=slice(lon_min, lon_max)
        )
        
        # skip if subset is empty
        if subset_ds.sizes['latitude'] == 0 or subset_ds.sizes['longitude'] == 0:
            print(f"Skipping empty subset for storm at {storm_lat}, {storm_lon}, time {row['iso_time_str']}")
            continue
        
        # extract the feature variables
        feature_vars = [
            'q_p500', 't_p500', 'u_p500', 'v_p500', 'pv_p500',
            'q_p750', 't_p750', 'u_p750', 'v_p750', 'pv_p750',
            'sst_and_t2m', 'land_mask', 'wmb'
        ]
        
        # create a list to store feature arrays
        feature_arrays = []
        
        # ensure dimensions don't exceed num_points
        lat_values = subset_ds.latitude.values[:num_points] if len(subset_ds.latitude) > num_points else subset_ds.latitude.values
        lon_values = subset_ds.longitude.values[:num_points] if len(subset_ds.longitude) > num_points else subset_ds.longitude.values
        
        # re-select with trimmed coordinates if needed
        if len(subset_ds.latitude) > num_points or len(subset_ds.longitude) > num_points:
            subset_ds = subset_ds.sel(latitude=lat_values, longitude=lon_values)
        
        # extract each feature and add to the list
        for var in feature_vars:
            if var in subset_ds:
                feature_arrays.append(subset_ds[var].values)
            else:
                print(f"Warning: Variable {var} not found in dataset for storm at {storm_lat}, {storm_lon}")
                continue
        
        # calculate the target variables: maximum wave heights within storm area
        if 'swh' in subset_ds:
            # get latitude and longitude arrays
            lats = subset_ds.latitude.values
            lons = subset_ds.longitude.values
            
            # create meshgrid for calculating distances
            lon_grid, lat_grid = np.meshgrid(lons, lats)
            
            # calculate distances from storm center
            x_dist = lon_grid - storm_lon
            y_dist = lat_grid - storm_lat
            
            # convert radii from nautical miles to degrees (1 nautical mile = 1.852 km, 1 degree ≈ 111 km)
            surge_radius_45_225_deg = surge_radius_45_225 * 1.852 / 111.0
            surge_radius_135_315_deg = surge_radius_135_315 * 1.852 / 111.0
            
            # rotate coordinates by 45 degrees to align with the ellipse axes
            cos_45 = np.cos(np.radians(45))
            sin_45 = np.sin(np.radians(45))
            x_rot = x_dist * cos_45 + y_dist * sin_45
            y_rot = -x_dist * sin_45 + y_dist * cos_45
            
            # calculate normalized distances for elliptical shape
            normalized_surge_dist = (x_rot**2 / (surge_radius_45_225_deg**2)) + (y_rot**2 / (surge_radius_135_315_deg**2))
            
            # create surge mask (points inside the ellipse)
            surge_mask = normalized_surge_dist <= 1.0
            
            
            # Get wave height values within the storm area
            swh_values = subset_ds['swh'].values
            swh_masked = swh_values * surge_mask
            swh_masked_no_nan = np.nan_to_num(swh_masked, nan=0.0)

           
            max_surge_idx = np.unravel_index(np.argmax(swh_masked_no_nan), swh_masked_no_nan.shape)
            max_surge_lat = lat_grid[max_surge_idx]
            max_surge_lon = lon_grid[max_surge_idx]
            max_surge_value = swh_masked_no_nan[max_surge_idx]
            print(f"Storm at {storm_lat}, {storm_lon}: Max wave and swell height of {max_surge_value} at lat={max_surge_lat}, lon={max_surge_lon}")
            
        
            # Get the maximum SWH value for the target
            swh_max = max_surge_value
            
            # stack the feature arrays and add to lists
            if len(feature_arrays) > 0:
                # Add the storm surge mask to feature arrays
                feature_arrays.append(surge_mask.astype(np.float32))
                feature_vars.append('storm_surge_mask')
                
                features = np.stack(feature_arrays, axis=0)
                features_tensor = torch.tensor(features, dtype=torch.float32)
                
                features_list.append(features_tensor)
                targets_list_swh.append(torch.tensor([swh_max], dtype=torch.float32))
                
                # Print progress every 20 rows
                if idx % 20 == 0 and idx > 0:
                    print(f"Processed {idx} of {len(storm_events_from2019to24)} rows. Current features: {len(features_list)}")
        else:
            missing_vars = []
            if 'swh' not in subset_ds: missing_vars.append('swh')
            print(f"Warning: Variables {', '.join(missing_vars)} not found in dataset for storm at {storm_lat}, {storm_lon}")
    
    except Exception as e:
        print(f"Error processing row {idx} (date: {row['iso_time_str']}): {str(e)}")
        continue

# print summary of processed data
print(f"Processed {len(features_list)} storm samples")
if len(features_list) > 0:
    print(f"Feature tensor shape: {features_list[0].shape}")
    print(f"Feature channels: {feature_vars}")
    print(f"SWH target tensor shape: {targets_list_swh[0].shape}")
    
    # Print range for the targets
    swh_array = torch.stack(targets_list_swh).numpy()
    print(f"Maximum SWH range: {np.min(swh_array)} to {np.max(swh_array)}")

    # Save the features and targets as .pt files
    torch.save(torch.stack(features_list), 'storm_features_wh.pt')
    torch.save(torch.stack(targets_list_swh), 'storm_targets_swh.pt')