import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import optuna
import shap
import math


# Choose target metric
target_metrics = ['sum', 'max', 'min', 'mean', 'median', 'p25', 'p75', 'p90', 'p95', 'p99']
TARGET_METRIC = 0

# Model Hyperparameters (will be tuned by Optuna) 
BATCH_SIZE = 16
LR = 0.0001
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 20

# Load the datasets from different years and concatenate them
X_2019_2020 = torch.load('storm_features_precipitation_2019_2020.pt', weights_only=True)
X_2021 = torch.load('storm_features_precipitation_2021.pt', weights_only=True)
X_2022 = torch.load('storm_features_precipitation_2022.pt', weights_only=True)
X_2023 = torch.load('storm_features_precipitation_2023.pt', weights_only=True)
X_2024 = torch.load('storm_features_precipitation_2024.pt', weights_only=True)

# Load targets
y_2019_2020 = torch.load('storm_targets_precipitation_2019_2020.pt', weights_only=True)
y_2021 = torch.load('storm_targets_precipitation_2021.pt', weights_only=True)
y_2022 = torch.load('storm_targets_precipitation_2022.pt', weights_only=True)
y_2023 = torch.load('storm_targets_precipitation_2023.pt', weights_only=True)
y_2024 = torch.load('storm_targets_precipitation_2024.pt', weights_only=True)

# Concatenate along the first dimension (samples)
X = torch.cat([X_2019_2020, X_2021, X_2022, X_2023, X_2024], dim=0)
y = torch.cat([y_2019_2020, y_2021, y_2022, y_2023, y_2024], dim=0)

print(X.shape, y.shape)

X = X[1:]  # Delete the first value along the first dimension
y = y[1:]  # Delete the first value along the first dimension
print(X.shape, y.shape)

# Check for NaN values in X and y
num_nan_X = torch.isnan(X).sum().item()
num_nan_y = torch.isnan(y).sum().item()

print(f"Number of NaN values in X: {num_nan_X}")
print(f"Number of NaN values in y: {num_nan_y}")

# Select target metric
target_metric_index = TARGET_METRIC
selected_metric = target_metrics[target_metric_index]
print(f"Training model on target metric: {selected_metric}")

# Split dataset
n_samples = len(X)
train_size, val_size = int(n_samples * 0.8), int(n_samples * 0.1)

# Get indices for splits
train_indices = list(range(0, 2660))  # Indices from 0 to 2659
val_indices = list(range(2660, 2990))  # Indices from 2660 to 2989
test_indices = list(range(2990, 3313))  # Indices from 2990 to 3312

# Split data
X_train, y_train = X[train_indices], y[train_indices, target_metric_index]
X_val, y_val = X[val_indices], y[val_indices, target_metric_index]

# Normalise only the first 11 channels, keep the rest unchanged
# Extract first 11 channels for normalisation
X_train_normalized = X_train[:, :11, :, :]          

# compute per-feature min, max and mean for the first 11 channels
xmin_train = torch.amin(X_train_normalized, dim=(0, 2, 3))
xmax_train = torch.amax(X_train_normalized, dim=(0, 2, 3))
xavg_train = torch.mean(X_train_normalized, dim=(0, 2, 3))

# expand them back to [1, 11, 1, 1] so they broadcast over X
xmin_train = xmin_train.view(1, -1, 1, 1)
xmax_train = xmax_train.view(1, -1, 1, 1)
xavg_train = xavg_train.view(1, -1, 1, 1)

# normalise the first 11 channels
X_train_normalized = (X_train_normalized - xavg_train) / (xmax_train - xmin_train)

print("Per-feature minima:", xmin_train.flatten())
print("Per-feature maxima:", xmax_train.flatten())
print("Per-feature means: ", xavg_train.flatten())

# Combine normalised first 11 features with the remaining unnormalised features
X_train = torch.cat((X_train_normalized, X_train[:, 11:, :, :]), dim=1)

# Apply the same normalisation to validation data
X_val_normalized = X_val[:, :11, :, :]   
X_val_normalized = (X_val_normalized - xavg_train) / (xmax_train - xmin_train)

# Combine normalised first 11 features with the remaining unnormalised features
X_val = torch.cat((X_val_normalized, X_val[:, 11:, :, :]), dim=1)

########################################  Test set  #########################################

X_test = X[test_indices]
y_test = y[test_indices, target_metric_index]  # Select only the target metric

# Apply the same normalisation to test data
X_test_normalized = X_test[:, :11, :, :]   
X_test_normalized = (X_test_normalized - xavg_train) / (xmax_train - xmin_train)

# Combine normalised first 11 features with the remaining unnormalised features
X_test = torch.cat((X_test_normalized, X_test[:, 11:, :, :]), dim=1)


#############################################  Model  ##############################################

"""
ResNet and SENet convolutional neural network architectures for precipitation
regression prediction.
"""


class Bottleneck(nn.Module):
    """Standard ResNet bottleneck block."""
    expansion = 4

    def __init__(self, in_channels, mid_channels, identity_downsample=None, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, mid_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(mid_channels * self.expansion)
        self.silu = nn.SiLU(inplace=False)
        self.identity_downsample = identity_downsample

    def forward(self, x):
        identity = x.clone()

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.silu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.silu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        out = out + identity
        out = self.silu(out)
        return out


class SELayer(nn.Module):
    """Squeeze-and-Excitation layer."""
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.SiLU(inplace=False),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class SEBottleneck(nn.Module):
    """Bottleneck block with Squeeze-and-Excitation."""
    expansion = 4

    def __init__(self, in_channels, mid_channels, identity_downsample=None, stride=1, reduction=16):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, mid_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(mid_channels * self.expansion)
        self.silu = nn.SiLU(inplace=False)
        self.se = SELayer(mid_channels * self.expansion, reduction)
        self.identity_downsample = identity_downsample

    def forward(self, x):
        identity = x.clone()

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.silu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.silu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.se(out)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        out = out + identity
        out = self.silu(out)
        return out


class ResNetBase(nn.Module):
    """Generic ResNet/SENet backbone for precipitation regression."""
    def __init__(self, block, layers, image_channels, dropout_rate=0.5):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.silu = nn.SiLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, layers[0], mid_channels=64, stride=1)
        self.layer2 = self._make_layer(block, layers[1], mid_channels=128, stride=2)
        self.layer3 = self._make_layer(block, layers[2], mid_channels=256, stride=2)
        self.layer4 = self._make_layer(block, layers[3], mid_channels=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(512 * block.expansion, 1)
        # No sigmoid for regression - output can be any real value

        self._initialise_weights()

    def _make_layer(self, block, num_blocks, mid_channels, stride):
        identity_downsample = None
        if stride != 1 or self.in_channels != mid_channels * block.expansion:
            identity_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, mid_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(mid_channels * block.expansion)
            )

        layers = []
        layers.append(block(self.in_channels, mid_channels, identity_downsample, stride))
        self.in_channels = mid_channels * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, mid_channels))
        return nn.Sequential(*layers)

    def _initialise_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.silu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        # No activation function for regression output
        return x


def SDS_ResNet(img_channel, dropout_rate=0.5):
    """Standard ResNet-50-style architecture for precipitation regression."""
    return ResNetBase(Bottleneck, [3, 4, 6, 3], img_channel, dropout_rate)


def SDS_SENet(img_channel, dropout_rate=0.5):
    """SENet-50-style architecture for precipitation regression."""
    return ResNetBase(SEBottleneck, [3, 4, 6, 3], img_channel, dropout_rate)


# Check for available device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Clear GPU cache
torch.cuda.empty_cache()

# Instantiate model
model = SDS_ResNet(img_channel=X_train.shape[1]).to(device)
model.eval()
model.load_state_dict(torch.load('best_model_precipitation_resnet.pth', map_location=device))

torch.cuda.empty_cache()

# Calculate difference between predicted and actual for all test set observations
print("Calculating difference between predicted and actual for all test set observations...")
with torch.no_grad():
    y_pred_all = model(X_test.to(device)).cpu().numpy().flatten()

y_true_all = y_test.cpu().numpy()

# Print difference for every value in the test set
print("\nDifference (Predicted - Actual) for every test set observation:")
for i in range(len(y_true_all)):
    difference = y_pred_all[i] - y_true_all[i]
    print(f"Test observation {i + 1}: Difference={difference:.4f}")

#############################################  SHAP  ##############################################

# Create folder for SHAP plots
shap_plots_folder = 'shap_plots_precipitation_resnet_paper'
os.makedirs(shap_plots_folder, exist_ok=True)

# Keep data on CPU to save GPU memory
X_train_subset = X_train[:500]  
explainer = shap.DeepExplainer(model, data=X_train_subset.to(device))

X_test_shap = X_test[::10]
shap_values_cal = explainer.shap_values(X_test_shap.to(device), check_additivity=False)

shap_values = shap_values_cal.squeeze(-1).reshape(-1, 13)

# Define feature names
feature_vars = ['Specific Humidity\n(500 hPa)', 
                'Temperature\n(500 hPa)', 
                'Zonal Wind\n(500 hPa)', 
                'Meridional Wind\n(500 hPa)', 
                'Potential Vorticity\n(500 hPa)',
                'Specific Humidity\n(750 hPa)', 
                'Temperature\n(750 hPa)', 
                'Zonal Wind\n(750 hPa)', 
                'Meridional Wind\n(750 hPa)', 
                'Potential Vorticity\n(750 hPa)',
                'Sea-surface Temperature\nand 2m Air Temperature\nover Land', 
                'Land Mask', 
                'Storm Size Mask']

plt.figure(figsize=(20, 13))
shap.summary_plot(shap_values, X_test_shap.reshape(-1, 13).cpu().numpy(), feature_names=feature_vars, show=False, alpha=0.7)

ax = plt.gca()
ax.tick_params(axis='y', labelsize=10) 
plt.subplots_adjust(left=0.3, right=0.95, top=0.95, bottom=0.1)
plt.savefig(os.path.join(shap_plots_folder, 'shap_summary_plot_precipitation_resnet.png'))
plt.close()

# Get model predictions for SHAP test subset
with torch.no_grad():
    y_pred_shap = model(X_test_shap.to(device)).cpu().numpy().flatten()

# Get ground truth values for SHAP test subset
y_true_shap = y_test[::10].cpu().numpy()

# Plot SHAP values overlaid on actual feature observations
# Define crop parameters to focus on storm area (center 100x100)
crop_size = 100
center = 80  # Assuming 160x160 grid, center is at 80
start_idx = center - crop_size // 2
end_idx = center + crop_size // 2

# Loop through each observation in the test set
for obs_idx in range(X_test_shap.shape[0]):
    # Calculate difference for this observation
    obs_true = y_true_shap[obs_idx]
    obs_pred = y_pred_shap[obs_idx]
    
    difference = obs_pred - obs_true
    
    print(f"SHAP Observation {obs_idx + 1}: Difference={difference:.4f}")
    
    plt.figure(figsize=(15, 10))
    
    # Get SHAP values and actual features for this observation
    obs_shap_values = shap_values_cal[obs_idx]
    obs_features = X_test_shap[obs_idx].cpu()
    
    # Create subplots for each feature channel
    for i in range(13):
        plt.subplot(3, 5, i+1)
        
        # Crop to focus on storm area
        cropped_features = obs_features[i][start_idx:end_idx, start_idx:end_idx]
        cropped_shap = obs_shap_values[i][start_idx:end_idx, start_idx:end_idx]
        
        # Plot the actual feature as background
        plt.imshow(cropped_features, cmap='gray', alpha=0.7)
        
        # Overlay SHAP values with transparency
        shap_max = np.abs(cropped_shap).max()
        if shap_max > 0:  # Only plot if there are non-zero SHAP values
            plt.imshow(cropped_shap, cmap='RdBu_r', alpha=0.8, 
                      vmin=-shap_max, vmax=shap_max)
        
        plt.title(f'{feature_vars[i]}', fontsize=11)
        cbar = plt.colorbar(shrink=0.6, pad=0.1)
        cbar.ax.tick_params(labelsize=8)
        cbar.ax.yaxis.get_offset_text().set_fontsize(7)  
        plt.axis('off')
    
    plt.suptitle(f'SHAP for Each Feature - Testing Observation {obs_idx + 1}', fontsize=15)
    plt.tight_layout()
    plt.savefig(os.path.join(shap_plots_folder, f'shap_overlay_observation_{obs_idx + 1}.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Also create a comparison plot showing features and SHAP values side by side
    fig, axes = plt.subplots(3, 10, figsize=(20, 8))
    
    obs_shap_values = shap_values_cal[obs_idx]
    obs_features = X_test_shap[obs_idx].cpu()
    
    for i in range(13):
        row = i // 5
        col_features = (i % 5) * 2
        col_shap = col_features + 1
        
        # Crop to focus on storm area
        cropped_features = obs_features[i][start_idx:end_idx, start_idx:end_idx]
        cropped_shap = obs_shap_values[i][start_idx:end_idx, start_idx:end_idx]
        
        # Plot original features
        if row < 3 and col_features < 10:
            axes[row, col_features].imshow(cropped_features, cmap='viridis')
            axes[row, col_features].set_title(f'{feature_vars[i]}', fontsize=8)
            axes[row, col_features].axis('off')
        
        # Plot SHAP values
        if row < 3 and col_shap < 10:
            shap_max = np.abs(cropped_shap).max()
            im = axes[row, col_shap].imshow(cropped_shap, cmap='RdBu_r', 
                                          vmin=-shap_max, vmax=shap_max)
            axes[row, col_shap].set_title(f'{feature_vars[i]} (SHAP)', fontsize=8)
            axes[row, col_shap].axis('off')
    
    # Remove any unused subplots
    for i in range(13, 15):
        row = i // 5
        col = (i % 5) * 2
        if row < 3 and col < 10:
            fig.delaxes(axes[row, col])
        if row < 3 and col + 1 < 10:
            fig.delaxes(axes[row, col + 1])
    
    plt.suptitle(f'Features vs SHAP Values Comparison - Observation {obs_idx + 1}', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(shap_plots_folder, f'features_vs_shap_observation_{obs_idx + 1}.png'), dpi=300, bbox_inches='tight')
    plt.close()