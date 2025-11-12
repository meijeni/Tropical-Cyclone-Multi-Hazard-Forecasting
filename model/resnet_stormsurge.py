import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import optuna
import math

# Create folder for plots
plots_folder = 'validation_plots_resnet_stormsurge'
os.makedirs(plots_folder, exist_ok=True)

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

# Get indicies for splits
indices = list(range(n_samples))
train_indices = indices[:train_size]
val_indices = indices[train_size:train_size + val_size]
test_indices = indices[train_size + val_size:n_samples]

# Split data
X_train, y_train = X[train_indices], y[train_indices]
X_val, y_val = X[val_indices], y[val_indices]

print("First 10 train indices:", train_indices[:10])
print("Last 10 train indices:", train_indices[-10:])
print("First 10 validation indices:", val_indices[:10])
print("Last 10 validation indices:", val_indices[-10:])
print("First 10 test indices:", test_indices[:10])
print("Last 10 test indices:", test_indices[-10:])

print(zero_indices)

train_indices = list(range(0, 2651))  # Indices from 0 to 2660
val_indices = list(range(2651, 2980))  # Indices from 2661 to 2983
test_indices = list(range(2980, 3304))  # Indices from 2990 to 2990

print("First 10 train indices:", train_indices[:10])
print("Last 10 train indices:", train_indices[-10:])
print("First 10 validation indices:", val_indices[:10])
print("Last 10 validation indices:", val_indices[-10:])
print("First 10 test indices:", test_indices[:10])
print("Last 10 test indices:", test_indices[-10:])

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

"""
ResNet convolutional neural network architecture for storm surge
regression prediction.

"""



class Bottleneck(nn.Module):
    """Standard ResNet bottleneck block."""
    expansion = 4

    def __init__(self, in_channels, mid_channels, identity_downsample=None, stride=1, activation='silu'):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, mid_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(mid_channels * self.expansion)
        
        # Choose activation function
        if activation == 'silu':
            self.activation = nn.SiLU(inplace=True)
        elif activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        
        self.identity_downsample = identity_downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        out += identity
        out = self.activation(out)
        return out


class ResNetBase(nn.Module):
    """Generic ResNet backbone for storm surge regression."""
    def __init__(self, block, layers, image_channels, dropout_rate=0.5, activation='silu'):
        super().__init__()
        self.in_channels = 64
        self.activation_name = activation
        
        # Choose activation function
        if activation == 'silu':
            self.activation = nn.SiLU(inplace=True)
        elif activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
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
        layers.append(block(self.in_channels, mid_channels, identity_downsample, stride, activation=self.activation_name))
        self.in_channels = mid_channels * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, mid_channels, activation=self.activation_name))
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
        x = self.activation(x)
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


def SDS_ResNet(img_channel, dropout_rate=0.5, activation='silu'):
    """Standard ResNet-50-style architecture for storm surge regression."""
    return ResNetBase(Bottleneck, [3, 4, 6, 3], img_channel, dropout_rate, activation)

# Check for available device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class Trainer:
    def __init__(self, model, optimizer, criterion, train_loader, val_loader):
        self.model = model
        self.optimizer = optimizer 
        self.criterion = criterion

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.best_val_loss = float('inf')

    def train_ep(self):
        self.model.train()
        losses = []
        for inputs, targets in self.train_loader:
            self.optimizer.zero_grad()
            
            outputs = self.model(inputs.to(device))
            loss = self.criterion(outputs, targets.to(device))

            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())
        return np.mean(losses)

    @torch.no_grad()
    def eval(self):
        self.model.eval()
        losses, predictions, actuals = [], [], []

        for inputs, targets in self.val_loader:
            outputs = self.model(inputs.to(device))
            loss = self.criterion(outputs, targets.to(device))
            losses.append(loss.item())

            predictions.extend(outputs.cpu().numpy().flatten())
            actuals.extend(targets.cpu().numpy().flatten())
        
        return np.mean(losses), predictions, actuals
    
    def train(self, num_epochs):
        train_losses, eval_losses = [], []
        for epoch in range(1, num_epochs+1):
            train_loss = self.train_ep()
            train_losses.append(train_loss)
            val_loss, val_predictions, val_actuals = self.eval()
            eval_losses.append(val_loss)

            # Update best validation loss
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss

        return train_losses, eval_losses, self.best_val_loss


def objective(trial):
    # Suggest hyperparameters
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.7)
    activation = trial.suggest_categorical('activation', ['relu', 'silu'])
    
    # Create datasets with suggested batch size
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Create model with suggested parameters
    model = SDS_ResNet(img_channel=X_train.shape[1], dropout_rate=dropout_rate, activation=activation).to(device)
    
    # Create optimizer with suggested parameters (only Adam)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    criterion = nn.MSELoss()
    
    # Train model
    trainer = Trainer(model, optimizer, criterion, train_loader, val_loader)
    train_losses, eval_losses, best_val_loss = trainer.train(num_epochs=15)  # Reduced epochs for optimization
    
    return best_val_loss


# Run Optuna optimization
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

# Print best parameters
print("Best trial:")
print(f"  Value: {study.best_value}")
print("  Best params:")
for key, value in study.best_params.items():
    print(f"    {key}: {value}")

# Train final model with best parameters
best_params = study.best_params

# Create datasets with best batch size
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=best_params['batch_size'])

# Create final model
model = SDS_ResNet(
    img_channel=X_train.shape[1], 
    dropout_rate=best_params['dropout_rate'], 
    activation=best_params['activation']
).to(device)

# Create optimizer (only Adam)
optimizer = optim.Adam(model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])

criterion = nn.MSELoss()

# Extended trainer class for final training with saving
class ExtendedTrainer(Trainer):
    def __init__(self, model, optimizer, criterion, train_loader, val_loader):
        super().__init__(model, optimizer, criterion, train_loader, val_loader)
        self.best_model_path = 'best_model_stormsurges_resnet_paper.pth'

    def train(self, num_epochs):
        train_losses, eval_losses = [], []
        for epoch in range(1, num_epochs+1):
            train_loss = self.train_ep()
            train_losses.append(train_loss)
            val_loss, val_predictions, val_actuals = self.eval()
            eval_losses.append(val_loss)
            print(f'Epoch {epoch}/{num_epochs} -> Training Loss: {train_loss:.6f}, Validation Loss: {val_loss:.6f}')
            
            # Save model if best validation loss
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.best_model_path)

        return train_losses, eval_losses, (val_predictions, val_actuals)

# Train final model
trainer = ExtendedTrainer(model, optimizer, criterion, train_loader, val_loader)
train_losses, eval_losses, (val_predictions, val_actuals) = trainer.train(num_epochs=500)

def evaluate_model(num_epochs, train_losses, val_losses):
    # Find the best epoch (lowest validation loss)
    best_epoch_idx = np.argmin(val_losses)
    best_epoch = best_epoch_idx + 1  # +1 because epochs are 1-indexed in plots
    
    # plot training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs+1), train_losses, marker='o', label='Training Loss')
    plt.plot(range(1, num_epochs+1), val_losses, marker='x', label='Validation Loss')
    plt.axvline(x=best_epoch, color='g', linestyle='--', label=f'Best Epoch ({best_epoch})')
    plt.title('Training vs Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_folder, 'training_validation_loss_resnet_paper.png'), dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Model from epoch {best_epoch} with validation loss: {val_losses[best_epoch_idx]:.6f}")

evaluate_model(
    num_epochs=500,
    train_losses=train_losses,
    val_losses=eval_losses,
)

X_test = X[test_indices]
y_test = y[test_indices]  # Select only the target metric

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

test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=best_params['batch_size'])

# Define a function to evaluate the storm surge model
def evaluate_storm_surge_model(model, data_loader, criterion):
    model.eval()
    losses = []
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            outputs = model(inputs.to(device))
            loss = criterion(outputs, targets.to(device))
            losses.append(loss.item())
            
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(targets.cpu().numpy())
    
    return np.mean(losses), predictions, actuals

# Test storm surge model after training
def test_storm_surge_model(model, test_loader, criterion, dataset_name="test"):
    # Load the best storm surge model from saved checkpoint
    best_model_path = f'best_model_stormsurges_resnet_paper.pth'
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    print("Loaded best storm surge model!")
    
    model.eval()
    test_loss, test_predictions, test_actuals = evaluate_storm_surge_model(model, test_loader, criterion)

    # Plot predicted vs actual storm surge values using actual test indices
    test_indices = list(range(len(test_actuals)))  # Use the length of test_actuals to create indices
    plt.figure(figsize=(10, 5))
    plt.plot(test_indices, test_actuals, label='Actual', marker='o', linestyle='--', color='b', alpha=0.7)
    plt.plot(test_indices, test_predictions, label='Predicted', marker='x', linestyle='-', color='r', alpha=0.7)
    plt.title(f'Predicted vs Actual Storm Surge Values by {dataset_name.capitalize()} Index (ResNet)')
    plt.xlabel(f'{dataset_name.capitalize()} Sample Index')
    plt.ylabel('Storm Surge (m)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_folder, f'predicted_vs_actual_timeseries_{dataset_name}_resnet_paper.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot predictions vs actual storm surge values
    plt.figure(figsize=(10, 5))
    plt.scatter(test_actuals, test_predictions, alpha=0.5)
    plt.plot([min(test_actuals), max(test_actuals)], [min(test_actuals), max(test_actuals)], 'r--')
    plt.title(f'Predictions vs Actuals (Best Storm Surge Model - ResNet) - {dataset_name.capitalize()}')
    plt.xlabel('Actual Storm Surge Values')
    plt.ylabel('Predicted Storm Surge Values')
    plt.grid(True)
    plt.savefig(os.path.join(plots_folder, f'predictions_vs_actuals_scatter_{dataset_name}_resnet_paper.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot the selected target metric (actual vs predicted storm surge values)
    plt.figure(figsize=(10, 5))
    plt.scatter(range(len(test_actuals)), test_actuals, 
                alpha=0.7, s=20, c='blue', edgecolor='k', linewidth=0.5, label='Actual')
    plt.scatter(range(len(test_predictions)), test_predictions, 
                alpha=0.7, s=20, c='red', edgecolor='k', linewidth=0.5, label='Predicted')
    plt.title(f'Actual vs Predicted Storm Surge (ResNet) - {dataset_name.capitalize()}')
    plt.xlabel('Sample Index')
    plt.ylabel('Storm Surge (m)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_folder, f'actual_vs_predicted_storm_surge_{dataset_name}_resnet_paper.png'), dpi=300, bbox_inches='tight')
    plt.show()

    print(f'{dataset_name.capitalize()} Loss: {test_loss:.6f}')

    # Calculate storm surge metrics
    mse = mean_squared_error(test_actuals, test_predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(test_actuals, test_predictions)
    r2 = r2_score(test_actuals, test_predictions)

    print(f"MSE: {mse:.6f}", f"RMSE: {rmse:.6f}", f"MAE: {mae:.6f}", f"R²: {r2:.6f}")

# Call the test function for storm surge model
test_storm_surge_model(model, test_loader, criterion, "test")

# Load the saved features and targets from the .pt files
atlantic_features = torch.load('storm_features_wh_atlantic.pt')
atlantic_targets = torch.load('storm_targets_wh_atlantic.pt')
# Print the shapes of the loaded tensors to verify
print(f"Loaded features tensor shape: {atlantic_features.shape}")
print(f"Loaded targets tensor shape: {atlantic_targets.shape}")

# Find indices where atlantic_targets has 0 values
zero_indices = torch.where(atlantic_targets == 0)[0]
print(f"Found {len(zero_indices)} samples with zero targets")
print(f"Zero indices: {zero_indices}")

# Delete those indices from atlantic_features and atlantic_targets
atlantic_features = torch.index_select(atlantic_features, 0, torch.tensor([i for i in range(atlantic_features.shape[0]) if i not in zero_indices]))
atlantic_targets = torch.index_select(atlantic_targets, 0, torch.tensor([i for i in range(atlantic_targets.shape[0]) if i not in zero_indices]))

print(f"After removing zero targets: atlantic_features {atlantic_features.shape}, atlantic_targets {atlantic_targets.shape}")

# Check for NaN values in atlantic_features and atlantic_targets
nan_in_atlantic_features = torch.isnan(atlantic_features).any()
nan_in_atlantic_targets = torch.isnan(atlantic_targets).any()

print(f"NaN values in atlantic_features: {nan_in_atlantic_features}")
print(f"NaN values in atlantic_targets: {nan_in_atlantic_targets}")

# Get more detailed information about NaNs if they exist
if nan_in_atlantic_features:
    # Count NaNs per channel
    nan_count_per_channel_atlantic = torch.isnan(atlantic_features).sum(dim=(0, 2, 3))
    print(f"NaN count per channel in atlantic_features: {nan_count_per_channel_atlantic}")
    
    # Percentage of NaNs in the dataset
    total_elements_atlantic = atlantic_features.numel()
    nan_percentage_atlantic = torch.isnan(atlantic_features).sum().item() / total_elements_atlantic * 100
    print(f"Percentage of NaNs in atlantic_features: {nan_percentage_atlantic:.4f}%")
    
    # Replace NaN values with 0 in atlantic_features
    atlantic_features = torch.nan_to_num(atlantic_features, nan=0.0)
    print("Replaced NaN values with 0 in atlantic_features")

if nan_in_atlantic_targets:
    # Count NaNs in targets
    nan_count_atlantic_targets = torch.isnan(atlantic_targets).sum().item()
    print(f"Number of NaN values in atlantic_targets: {nan_count_atlantic_targets}")
    print(f"Percentage of NaNs in atlantic_targets: {nan_count_atlantic_targets / atlantic_targets.numel() * 100:.4f}%")
    
    # Replace NaN values with 0 in atlantic_targets
    atlantic_targets = torch.nan_to_num(atlantic_targets, nan=0.0)
    print("Replaced NaN values with 0 in atlantic_targets")

# Verify NaNs have been removed
print(f"NaN values remaining in atlantic_features: {torch.isnan(atlantic_features).any()}")
print(f"NaN values remaining in atlantic_targets: {torch.isnan(atlantic_targets).any()}")

# Apply the same normalization to Atlantic data as was done for test data
# First 11 channels
atlantic_norm_first = atlantic_features[:, :11, :, :]
atlantic_norm_first = (atlantic_norm_first - xavg_first) / (xmax_first - xmin_first)

# Channel 13 (index 12)
atlantic_norm_ch13 = atlantic_features[:, 12:13, :, :]
atlantic_norm_ch13 = (atlantic_norm_ch13 - xavg_ch13) / (xmax_ch13 - xmin_ch13)

# Combine all parts (keeping channel 12 and channels 14+ unchanged)
atlantic_features_normalized = torch.cat((
    atlantic_norm_first,
    atlantic_features[:, 11:12, :, :],  # Channel 12 unchanged
    atlantic_norm_ch13,
    atlantic_features[:, 13:, :, :]     # Channels 14+ unchanged
), dim=1)

# Create dataset and dataloader
atlantic_dataset = TensorDataset(atlantic_features_normalized, atlantic_targets)
atlantic_loader = DataLoader(atlantic_dataset, batch_size=best_params['batch_size'])

# Call the test function for storm surge model
test_storm_surge_model(model, atlantic_loader, criterion, "atlantic")