"""
Global configuration parameters for the storm prediction models.
"""

import torch
import os

# ================================================================================================
# Device Configuration
# ================================================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================================================================================================
# Directory Paths
# ================================================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "model", "model paths")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "model", "cleaned tensors")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
STORM_EVENTS_DIR = os.path.join(BASE_DIR, "dataprocessing")

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ================================================================================================
# Data Processing Parameters
# ================================================================================================
# Years and months for data fetching
YEARS = [2019, 2020, 2021, 2022, 2023, 2024]
MONTHS = list(range(1, 13))

# Latitude/longitude range for storm extraction (in degrees)
LAT_LON_RANGE = 40
NUM_POINTS = LAT_LON_RANGE * 4

# Data chunking parameters
SMALL_CHUNKS = {'valid_time': 8}
LARGE_CHUNKS = {'valid_time': 24}

# Hours to select for precipitation data
PRECIPITATION_HOURS = [3, 6, 9, 12, 15, 18, 21, 0]

# Normalization channels (indices)
NORMALIZE_CHANNELS_FIRST_11 = list(range(11))  # First 11 channels
NORMALIZE_CHANNEL_13 = [12]  # Channel 13 (index 12)

# ================================================================================================
# Model Hyperparameters
# ================================================================================================
# Training parameters
NUM_EPOCHS = 300
BATCH_SIZE = 16
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 1e-5
DROPOUT_RATE = 0.5

# Hyperparameter tuning ranges (for Optuna)
HYPERPARAMETER_RANGES = {
    'lr': (1e-5, 1e-2),
    'batch_size': [8, 16, 32],
    'weight_decay': (1e-6, 1e-3),
    'dropout_rate': (0.2, 0.6),
    'activation': ['relu', 'silu'],
    'optimizer': ['adam', 'sgd']
}

# Optuna tuning parameters
OPTUNA_N_TRIALS = 100
OPTUNA_TUNING_EPOCHS = 10  # Reduced epochs for hyperparameter tuning

# ================================================================================================
# Target Metrics
# ================================================================================================
TARGET_METRICS = ['sum', 'max', 'min', 'mean', 'median', 'p25', 'p75', 'p90', 'p95', 'p99']
TARGET_METRIC_INDEX = 0  # Default to 'sum'

# ================================================================================================
# Data Split Indices
# ================================================================================================
# Train/validation/test split indices
TRAIN_INDICES = list(range(0, 2660))
VAL_INDICES = list(range(2660, 2990))
TEST_INDICES = list(range(2990, 3313))

# For storm surge models (slightly different indices)
TRAIN_INDICES_SURGE = list(range(0, 2651))
VAL_INDICES_SURGE = list(range(2651, 2980))
TEST_INDICES_SURGE = list(range(2980, 3304))

# ================================================================================================
# File Names
# ================================================================================================
# Storm events CSV files
STORM_EVENTS_PACIFIC = "storm_events_2019to24_pacific.csv"
STORM_EVENTS_ATLANTIC = "storm_events_atlantic.csv"

# Model checkpoint names
MODEL_NAMES = {
    'precipitation_alexnet': 'best_model_precipitation_alexnet.pth',
    'precipitation_resnet': 'best_model_precipitation_resnet.pth',
    'precipitation_senet': 'best_model_precipitation_senet.pth',
    'stormsurge_alexnet': 'best_model_stormsurges_alexnet.pth',
    'stormsurge_resnet': 'best_model_stormsurges_resnet.pth',
    'stormsurge_senet': 'best_model_stormsurges_senet.pth'
}

# Processed data file patterns
FEATURES_PATTERN = "storm_features_{}.pt"
TARGETS_PATTERN = "storm_targets_{}.pt"

# ================================================================================================
# ERA5 API Configuration
# ================================================================================================
CDS_API_URL = "https://cds.climate.copernicus.eu/api" 
CDS_API_KEY = "########################" # Replace with own API key

# ERA5 data prefixes
ERA5_PREFIXES = {
    'pressure': 'era5_pressure_data',
    'temperature_2m': 'era5_2m_temperature',
    'sst': 'era5_sst_data',
    'precipitation': 'era5_tp_data',
    'wave': 'era5_wh_data',
    'bathymetry': 'era5_bathymetry_data'
}

# ================================================================================================
# Feature Variables
# ================================================================================================
# Feature variables for precipitation model
PRECIPITATION_FEATURES = [
    'q_p500', 't_p500', 'u_p500', 'v_p500', 'pv_p500',
    'q_p750', 't_p750', 'u_p750', 'v_p750', 'pv_p750',
    'sst_and_t2m', 'land_mask'
]

# Feature variables for storm surge model
STORM_SURGE_FEATURES = [
    'q_p500', 't_p500', 'u_p500', 'v_p500', 'pv_p500',
    'q_p750', 't_p750', 'u_p750', 'v_p750', 'pv_p750',
    'sst_and_t2m', 'land_mask', 'wmb'
]

# ================================================================================================
# Plot Configuration
# ================================================================================================
PLOT_DPI = 300
PLOT_FORMAT = 'png'

# Plot directory names
PLOT_DIRS = {
    'precipitation_alexnet': 'validation_plots_alexnet_precipitation',
    'precipitation_resnet': 'validation_plots_resnet_precipitation',
    'precipitation_senet': 'validation_plots_senet_precipitation',
    'stormsurge_resnet': 'validation_plots_resnet_stormsurge',
    'stormsurge_senet': 'validation_plots_senet_stormsurge',
    'stormsurge_alexnet': 'validation_plots_alexnet_stormsurge'
}
