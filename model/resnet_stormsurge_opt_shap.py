import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import shap
import math

# === Model Hyperparameters (will be tuned by Optuna) ===
BATCH_SIZE = 16
LR = 0.0001
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 20

# load the datasets
X1 = torch.load('storm_features_wh_firsthalf.pt', weights_only=True)  # shape: (N, C, H, W)
y1 = torch.load('storm_targets_swh_firsthalf.pt', weights_only=True)  # shape: (N,)

X2 = torch.load('storm_features_wh_secondhalf.pt', weights_only=True)  # shape: (N, C, H, W)
y2 = torch.load('storm_targets_swh_secondhalf.pt', weights_only=True)  # shape: (N,)

# merge the datasets
X = torch.cat([X1, X2], dim=0)  # concatenate along the batch dimension
y = torch.cat([y1, y2], dim=0)  # concatenate along the batch dimension

X = X[1:]  # Delete the first value along the first dimension
y = y[1:]  # Delete the first value along the first dimension
print(X.shape, y.shape)

print(f"Original shapes: X1 {X1.shape}, X2 {X2.shape}, y1 {y1.shape}, y2 {y2.shape}")
print(f"Merged shapes: X {X.shape}, y {y.shape}")

# Find indices where y has 0 values
zero_indices = torch.where(y == 0)[0]
print(f"Found {len(zero_indices)} samples with zero targets")
print(f"Zero indices: {zero_indices}")

# Delete those indices from X and y
X = torch.index_select(X, 0, torch.tensor([i for i in range(X.shape[0]) if i not in zero_indices]))
y = torch.index_select(y, 0, torch.tensor([i for i in range(y.shape[0]) if i not in zero_indices]))

print(f"After removing zero targets: X {X.shape}, y {y.shape}")


# Check for NaN values in X and y
nan_in_X = torch.isnan(X).any()
nan_in_y = torch.isnan(y).any()

print(f"NaN values in X: {nan_in_X}")
print(f"NaN values in y: {nan_in_y}")

# Get more detailed information about NaNs if they exist
if nan_in_X:
    # Count NaNs per channel
    nan_count_per_channel = torch.isnan(X).sum(dim=(0, 2, 3))
    print(f"NaN count per channel: {nan_count_per_channel}")
    
    # Percentage of NaNs in the dataset
    total_elements = X.numel()
    nan_percentage = torch.isnan(X).sum().item() / total_elements * 100
    print(f"Percentage of NaNs in X: {nan_percentage:.4f}%")
    
    # Replace NaN values with 0 in X
    X = torch.nan_to_num(X, nan=0.0)
    print("Replaced NaN values with 0 in X")

if nan_in_y:
    # Count NaNs in targets
    nan_count_y = torch.isnan(y).sum().item()
    print(f"Number of NaN values in y: {nan_count_y}")
    print(f"Percentage of NaNs in y: {nan_count_y / y.numel() * 100:.4f}%")
    
    # Replace NaN values with 0 in y
    y = torch.nan_to_num(y, nan=0.0)
    print("Replaced NaN values with 0 in y")

# Verify NaNs have been removed
print(f"NaN values remaining in X: {torch.isnan(X).any()}")
print(f"NaN values remaining in y: {torch.isnan(y).any()}")

# Split dataset
n_samples = len(X)
train_size, val_size = int(n_samples * 0.8), int(n_samples * 0.1)

# Get indices for splits
train_indices = list(range(0, 2651))  # Indices from 0 to 2650
val_indices = list(range(2651, 2980))  # Indices from 2651 to 2979
test_indices = list(range(2980, 3304))  # Indices from 2980 to 3303

print("First 10 train indices:", train_indices[:10])
print("Last 10 train indices:", train_indices[-10:])
print("First 10 validation indices:", val_indices[:10])
print("Last 10 validation indices:", val_indices[-10:])
print("First 10 test indices:", test_indices[:10])
print("Last 10 test indices:", test_indices[-10:])

# Split data
X_train, y_train = X[train_indices], y[train_indices]
X_val, y_val = X[val_indices], y[val_indices]

# Normalize first 11 channels and channel 13, keep others unchanged
# First 11 channels
X_train_norm_first = X_train[:, :11, :, :]
xmin_first = torch.amin(X_train_norm_first, dim=(0, 2, 3)).view(1, -1, 1, 1)
xmax_first = torch.amax(X_train_norm_first, dim=(0, 2, 3)).view(1, -1, 1, 1)
xavg_first = torch.mean(X_train_norm_first, dim=(0, 2, 3)).view(1, -1, 1, 1)
X_train_norm_first = (X_train_norm_first - xavg_first) / (xmax_first - xmin_first)

# Channel 13
X_train_norm_ch13 = X_train[:, 12:13, :, :]
xmin_ch13 = torch.amin(X_train_norm_ch13, dim=(0, 2, 3)).view(1, -1, 1, 1)
xmax_ch13 = torch.amax(X_train_norm_ch13, dim=(0, 2, 3)).view(1, -1, 1, 1)
xavg_ch13 = torch.mean(X_train_norm_ch13, dim=(0, 2, 3)).view(1, -1, 1, 1)
X_train_norm_ch13 = (X_train_norm_ch13 - xavg_ch13) / (xmax_ch13 - xmin_ch13)

print("Per-feature minima (first 11):", xmin_first.flatten())
print("Per-feature maxima (first 11):", xmax_first.flatten())
print("Per-feature means (first 11): ", xavg_first.flatten())
print(f"Channel 13 - min: {xmin_ch13.item()}, max: {xmax_ch13.item()}, mean: {xavg_ch13.item()}")

# Combine all parts
X_train = torch.cat((
    X_train_norm_first,
    X_train[:, 11:12, :, :],
    X_train_norm_ch13,
    X_train[:, 13:, :, :]
), dim=1)

# Apply same normalization to validation data
X_val_norm_first = (X_val[:, :11, :, :] - xavg_first) / (xmax_first - xmin_first)
X_val_norm_ch13 = (X_val[:, 12:13, :, :] - xavg_ch13) / (xmax_ch13 - xmin_ch13)

X_val = torch.cat((
    X_val_norm_first,
    X_val[:, 11:12, :, :],
    X_val_norm_ch13,
    X_val[:, 13:, :, :]
), dim=1)

# Create datasets
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)


########################################  Test set  #########################################
X_test = X[test_indices]
y_test = y[test_indices] 

# Apply the same normalisation to test data
X_test_norm_first = X_test[:, :11, :, :]   
X_test_norm_first = (X_test_norm_first - xavg_first) / (xmax_first - xmin_first)

# Channel 13
X_test_norm_ch13 = X_test[:, 12:13, :, :]
X_test_norm_ch13 = (X_test_norm_ch13 - xavg_ch13) / (xmax_ch13 - xmin_ch13)

# Combine all parts
X_test = torch.cat((
    X_test_norm_first,
    X_test[:, 11:12, :, :],
    X_test_norm_ch13,
    X_test[:, 13:, :, :]
), dim=1)


#############################################  Model  ##############################################

"""
ResNet and SENet convolutional neural network architectures for storm surge
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
    """Generic ResNet/SENet backbone for storm surge regression."""
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
    """Standard ResNet-50-style architecture for storm surge regression."""
    return ResNetBase(Bottleneck, [3, 4, 6, 3], img_channel, dropout_rate)


def SDS_SENet(img_channel, dropout_rate=0.5):
    """SENet-50-style architecture for storm surge regression."""
    return ResNetBase(SEBottleneck, [3, 4, 6, 3], img_channel, dropout_rate)


# Check for available device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Clear GPU cache
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Instantiate model
model = SDS_ResNet(img_channel=X_train.shape[1]).to(device)
model.eval()
model.load_state_dict(torch.load('best_model_stormsurges_resnet_paper.pth'))

if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Compute predictions for the test set and calculate difference for each point
model.eval()
test_predictions = []
with torch.no_grad():
    for i in range(X_test.shape[0]):
        pred = model(X_test[i:i+1].to(device))
        test_predictions.append(pred.cpu().item())

test_predictions = torch.tensor(test_predictions)

# Calculate and print difference for each testing point
print("\nDifference (Predicted - Actual) for each testing point:")
for i in range(len(test_predictions)):
    difference = test_predictions[i] - y_test[i].item()
    print(f"Test point {i+1}: {difference:.4f}")

#############################################  SHAP  ##############################################

# Create folder for SHAP plots
shap_plots_folder = 'shap_plots_stormsurges_resnet_paper'
os.makedirs(shap_plots_folder, exist_ok=True)

# Keep data on CPU to save GPU memory
X_train_subset = X_train[:500]  
explainer = shap.DeepExplainer(model, data=X_train_subset.to(device))

X_test_shap = X_test[::10]
y_test_shap = y_test[::10]
shap_values_cal = explainer.shap_values(X_test_shap.to(device), check_additivity=False)

# Get predictions for the SHAP subset
shap_predictions = []
with torch.no_grad():
    for i in range(X_test_shap.shape[0]):
        pred = model(X_test_shap[i:i+1].to(device))
        shap_predictions.append(pred.cpu().item())

shap_predictions = torch.tensor(shap_predictions)

shap_values = shap_values_cal.squeeze(-1).reshape(-1, 14)

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
                'Model bathymetry',
                'Storm Size Mask']

plt.figure(figsize=(20, 13))
shap.summary_plot(shap_values, X_test_shap.reshape(-1, 14).cpu().numpy(), feature_names=feature_vars, show=False, alpha=0.7)

ax = plt.gca()
ax.tick_params(axis='y', labelsize=10) 
plt.subplots_adjust(left=0.3, right=0.95, top=0.95, bottom=0.1)
plt.savefig(os.path.join(shap_plots_folder, 'shap_summary_plot_stormsurge_resnet.png'))
plt.close()

# Plot SHAP values overlaid on actual feature observations
# Define crop parameters to focus on storm area (center 100x100)
crop_size = 100
center = 80  # Assuming 160x160 grid, center is at 80
start_idx = center - crop_size // 2
end_idx = center + crop_size // 2

# Loop through each observation in the test set
for obs_idx in range(X_test_shap.shape[0]):
    # Calculate metrics for this observation
    predicted_val = shap_predictions[obs_idx].item()
    actual_val = y_test_shap[obs_idx].item()
    difference = predicted_val - actual_val
    
    print(f"\nObservation {obs_idx + 1} difference:")
    print(f"  Difference (Predicted - Actual): {difference:.4f}")
    
    plt.figure(figsize=(15, 10))
    
    # Get SHAP values and actual features for this observation
    obs_shap_values = shap_values_cal[obs_idx]
    obs_features = X_test_shap[obs_idx].cpu()
    
    # Create subplots for each feature channel
    for i in range(14):
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
    
    for i in range(14):
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
    for i in range(14, 15):
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