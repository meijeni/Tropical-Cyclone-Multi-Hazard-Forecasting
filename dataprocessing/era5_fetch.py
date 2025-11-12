import cdsapi, os

# Define common parameters
years = ["2019", "2020", "2021", "2022", "2023", "2024"]
months = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
days = [
    "01", "02", "03", "04", "05", "06", "07", "08", "09", "10",
    "11", "12", "13", "14", "15", "16", "17", "18", "19", "20",
    "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31"
]

hourly_times = [
    "00:00", "01:00", "02:00", "03:00", "04:00", "05:00",
    "06:00", "07:00", "08:00", "09:00", "10:00", "11:00",
    "12:00", "13:00", "14:00", "15:00", "16:00", "17:00",
    "18:00", "19:00", "20:00", "21:00", "22:00", "23:00"
]

three_hourly_times = [
    "00:00", "03:00", "06:00", "09:00", 
    "12:00", "15:00", "18:00", "21:00"
]

# Initialize client
client = cdsapi.Client(url="https://cds.climate.copernicus.eu/api", key="fbe8bfbd-087a-434c-a6bf-34c3451202e3")

# Create data/ dir if doesn't exist
os.makedirs("data", exist_ok=True)

# 1. Total precipitation data
def fetch_precipitation_data():
    for year in years:
        for month in months:
            request = {
                "product_type": ["reanalysis"],
                "variable": ["total_precipitation"],
                "year": [year],
                "month": [month],
                "day": days,
                "time": hourly_times,
                "data_format": "netcdf",
                "download_format": "unarchived"
            }
            
            # Download the data for the specific year and month
            client.retrieve("reanalysis-era5-single-levels", request).download(f'data/era5_tp_data_{year}_{month}.nc')

# 2. Temperature data
def fetch_2m_temperature_data():
    for year in years:
        for month in months:
            request = {
                "product_type": ["reanalysis"],
                "variable": ["2m_temperature"],
                "year": [year],
                "month": [month],
                "day": days,
                "time": three_hourly_times,
                "data_format": "netcdf",
                "download_format": "unarchived"
            }
            
            client.retrieve("reanalysis-era5-single-levels", request).download(f'data/era5_2m_temperature_{year}_{month}.nc')

# 3. Sea surface temperature data
def fetch_sst_data():
    dataset = "reanalysis-era5-single-levels"
    
    for year in years:
        for month in months:
            request = {
                "product_type": ["reanalysis"],
                "variable": ["sea_surface_temperature"],
                "year": [year],
                "month": [month],
                "day": days,
                "time": three_hourly_times,
                "data_format": "netcdf",
                "download_format": "unarchived"
            }
            
            client.retrieve(dataset, request).download(f'data/era5_sst_data_{year}_{month}.nc')

# 4. Pressure level data
def fetch_pressure_level_data():
    for year in years:
        for month in months:
            request = {
                "product_type": ["reanalysis"],
                "variable": [
                "potential_vorticity",
                "specific_humidity",
                "temperature",
                "u_component_of_wind",
                "v_component_of_wind"
                ],
                "year": [year],
                "month": [month],
                "day": days,
                "time": three_hourly_times,
                "pressure_level": ["500", "750"],
                "data_format": "netcdf",
                "download_format": "unarchived"
            }
            
            # Download the data for the specific year and month
            client.retrieve("reanalysis-era5-pressure-levels", request).download(f'data/era5_pressure_data_{year}_{month}.nc')

# 5. Wave data
def fetch_wave_data():
    dataset = "reanalysis-era5-single-levels"
    
    for year in years:
        for month in months:
            request = {
                "product_type": ["reanalysis"],
                "variable": [
                    #"mean_wave_direction",
                    #"mean_wave_period",
                    "significant_height_of_combined_wind_waves_and_swell",
                    #"significant_height_of_total_swell",
                    #"significant_height_of_wind_waves"
                    "maximum_individual_wave_height"
                ],
                "year": [year],
                "month": [month],
                "day": days,
                "time": three_hourly_times,
                "data_format": "netcdf",
                "download_format": "unarchived"
            }
            
            client.retrieve(dataset, request).download(f'data/era5_wh_data_{year}_{month}.nc')

# 6. Model bathymetry data
def fetch_model_bathymetry_data():
    dataset = "reanalysis-era5-single-levels"
    
    for year in years:
        for month in months:
            request = {
                "product_type": ["reanalysis"],
                "variable": ["model_bathymetry"],
                "year": [year],
                "month": [month],
                "day": days,
                "time": three_hourly_times,
                "data_format": "netcdf",
                "download_format": "unarchived",
            }
            
            client.retrieve(dataset, request).download(f'data/era5_bathymetry_data_{year}_{month}.nc')


# Execute all data fetching functions
fetch_precipitation_data()
fetch_2m_temperature_data()
fetch_sst_data()
fetch_pressure_level_data()
fetch_wave_data()
fetch_model_bathymetry_data()
