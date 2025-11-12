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

# === Choose target metric ===
target_metrics = ['sum', 'max', 'min', 'mean', 'median', 'p25', 'p75', 'p90', 'p95', 'p99']
TARGET_METRIC = 0

# === Model Hyperparameters (will be tuned by Optuna) ===
BATCH_SIZE = 16
LR = 0.0001
NUM_EPOCHS = 500

# load the datasets from different years and concatenate them
# Load features
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

# Create plots directory
plots_dir = 'senet_precipitation'
os.makedirs(plots_dir, exist_ok=True)

# Plot the selected target metric (storm rainfall measurements)
plt.figure(figsize=(10, 5))
# Plot the data points with connecting lines (very weak/transparent)
x_indices = range(len(y[:, target_metric_index]))
y_values = y[:, target_metric_index].numpy()
plt.plot(x_indices, y_values, '-', color='blue', alpha=0.6, linewidth=0.8)
# Plot the scatter points on top
plt.scatter(x_indices, y_values, 
            alpha=0.7, s=20, c='blue', edgecolor='k', linewidth=0.5)
plt.title(f'3-H Storm Induced Precipitation')
plt.xlabel('Sample Index')
plt.ylabel(f'Precipitation (m)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{plots_dir}/storm_precipitation_timeseries.png')
plt.show()

# Plot histogram to show distribution of rainfall values
plt.figure(figsize=(10, 5))
plt.hist(y[:, target_metric_index].numpy(), bins=30, alpha=0.7, color='skyblue', edgecolor='black')
plt.title(f'Distribution of Storm Rainfall Values ({selected_metric})')
plt.xlabel(f'Rainfall ({selected_metric})')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{plots_dir}/storm_precipitation_distribution.png')
plt.show()

# Print some statistics about the target values
target_values = y[:, target_metric_index].numpy()
print(f"Statistics for {selected_metric} rainfall:")
print(f"Number of samples: {len(target_values)}")
print(f"Min: {target_values.min():.4f}")
print(f"Max: {target_values.max():.4f}")
print(f"Mean: {target_values.mean():.4f}")
print(f"Median: {np.median(target_values):.4f}")
print(f"Standard deviation: {target_values.std():.4f}")


# Split dataset
n_samples = len(X)

# Use explicit indices for splits
train_indices = list(range(0, 2660))  # Indices from 0 to 2659
val_indices = list(range(2660, 2990))  # Indices from 2660 to 2989
test_indices = list(range(2990, n_samples))  # Indices from 2990 to end

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


# Create datasets
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

"""
ResNet and SENet convolutional neural network architectures for storm surge
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
        
        # Configurable activation function
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


class SELayer(nn.Module):
    """Squeeze-and-Excitation layer."""
    def __init__(self, channel, reduction=16, activation='silu'):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Configurable activation function
        if activation == 'silu':
            act_func = nn.SiLU(inplace=True)
        elif activation == 'relu':
            act_func = nn.ReLU(inplace=True)
        
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            act_func,
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

    def __init__(self, in_channels, mid_channels, identity_downsample=None, stride=1, reduction=16, activation='silu'):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, mid_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(mid_channels * self.expansion)
        
        # Configurable activation function
        if activation == 'silu':
            self.activation = nn.SiLU(inplace=True)
        elif activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        
        self.se = SELayer(mid_channels * self.expansion, reduction, activation)
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

        out = self.se(out)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        out += identity
        out = self.activation(out)
        return out


class ResNetBase(nn.Module):
    """Generic ResNet/SENet backbone for storm surge regression."""
    def __init__(self, block, layers, image_channels, dropout_rate=0.5, activation='silu'):
        super().__init__()
        self.in_channels = 64
        self.activation_type = activation
        
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Configurable activation function
        if activation == 'silu':
            self.activation = nn.SiLU(inplace=True)
        elif activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        
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
        layers.append(block(self.in_channels, mid_channels, identity_downsample, stride, activation=self.activation_type))
        self.in_channels = mid_channels * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, mid_channels, activation=self.activation_type))
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


def SDS_SENet(img_channel, dropout_rate=0.5, activation='silu'):
    """SENet-50-style architecture for storm surge regression."""
    return ResNetBase(SEBottleneck, [3, 4, 6, 3], img_channel, dropout_rate, activation)

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
        self.best_model_path = 'best_model_precipitation_senet.pth'

    def train_ep(self):
        self.model.train()
        losses = []
        for inputs, targets in self.train_loader:
            self.optimizer.zero_grad()
            
            outputs = self.model(inputs.to(device))
            loss = self.criterion(outputs, targets.to(device).unsqueeze(1))

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
            loss = self.criterion(outputs, targets.to(device).unsqueeze(1))
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
            print(f'Epoch {epoch}/{num_epochs} -> Training Loss: {train_loss:.6f}, Validation Loss: {val_loss:.6f}')

            # Save model if best validation loss
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.best_model_path)
                
        # Save final validation plot
        self.save_validation_plot(val_predictions, val_actuals)
        return train_losses, eval_losses, (val_predictions, val_actuals)
    
    def save_validation_plot(self, predictions, actuals):
        # Plot predicted vs actual values
        eval_indices = list(range(len(actuals)))
        plt.figure(figsize=(10, 5))
        plt.plot(eval_indices, actuals, label='Actual', marker='o', linestyle='--', color='b')
        plt.plot(eval_indices, predictions, label='Predicted', marker='x', linestyle='-', color='r')
        plt.title('Predicted vs Actual Values by Validation Index')
        plt.xlabel('Validation Index')
        plt.ylabel('Values')
        plt.legend()
        plt.grid(True)
        
        # Save the plot to the plots directory
        plot_filename = f'{plots_dir}/final_validation_plot.png'
        plt.savefig(plot_filename)
        plt.close()


# Optuna objective function
def objective(trial):
    # Suggest hyperparameters according to the specified search space
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.6)
    activation = trial.suggest_categorical('activation', ['relu', 'silu'])
    
    # Create data loaders with the suggested batch size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Instantiate model with suggested hyperparameters
    model = SDS_SENet(
        img_channel=X_train.shape[1], 
        dropout_rate=dropout_rate, 
        activation=activation
    ).to(device)
    
    # Define Loss and Optimizer (fixed to Adam with tuned weight_decay)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Create trainer
    trainer = Trainer(model, optimizer, criterion, train_loader, val_loader)
    
    # Train for a smaller number of epochs for hyperparameter tuning
    train_losses, eval_losses, _ = trainer.train(num_epochs=10)
    
    # Return the best validation loss
    return min(eval_losses)


# Run Optuna study
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

print("Best trial:")
print(f"  Value: {study.best_value}")
print("  Params: ")
for key, value in study.best_params.items():
    print(f"    {key}: {value}")

# Train final model with best hyperparameters
best_params = study.best_params
print(f"\nTraining final model with best hyperparameters...")

# Create data loaders with best batch size
train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=best_params['batch_size'])

# Instantiate model with best hyperparameters
model = SDS_SENet(
    img_channel=X_train.shape[1],
    dropout_rate=best_params['dropout_rate'],
    activation=best_params['activation']
).to(device)

# Define Loss and Optimizer with best hyperparameters
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])

trainer = Trainer(model, optimizer, criterion, train_loader, val_loader)

# Train :)
print(f"Starting training for target metric: {selected_metric}.")
train_losses, eval_losses, (val_predictions, val_actuals) = trainer.train(num_epochs=NUM_EPOCHS)

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
    
    # Save the plot
    plt.savefig(f'{plots_dir}/training_validation_loss.png')
    plt.close()

    print(f"Model from epoch {best_epoch} with validation loss: {val_losses[best_epoch_idx]:.6f}")


evaluate_model(
    num_epochs=NUM_EPOCHS,
    train_losses=train_losses,
    val_losses=eval_losses,
)

X_test = X[test_indices]
y_test = y[test_indices, target_metric_index]  # Select only the target metric

# Apply the same normalisation to test data
X_test_normalized = X_test[:, :11, :, :]   
X_test_normalized = (X_test_normalized - xavg_train) / (xmax_train - xmin_train)

# Combine normalised first 11 features with the remaining unnormalised features
X_test = torch.cat((X_test_normalized, X_test[:, 11:, :, :]), dim=1)

test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=best_params['batch_size'])


# Define a function to evaluate the model
def evaluate_model_loss(model, data_loader, criterion):
    model.eval()
    losses = []
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            outputs = model(inputs.to(device))
            loss = criterion(outputs, targets.to(device).unsqueeze(1))
            losses.append(loss.item())
            
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(targets.cpu().numpy())
    
    return np.mean(losses), predictions, actuals

# Test model after training
def test_model(model, test_loader, criterion):
    # Load the best model from saved checkpoint
    best_model_path = f'best_model_precipitation_senet.pth'
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    print("Loaded best model!")
    
    model.eval()
    test_loss, test_predictions, test_actuals = evaluate_model_loss(model, test_loader, criterion)

    # Plot predicted vs actual values using actual test indices
    test_indices_plot = list(range(len(test_actuals)))  # Use the length of test_actuals to create indices
    plt.figure(figsize=(10, 5))
    plt.plot(test_indices_plot, test_actuals, label='Actual', marker='o', linestyle='--', color='b', alpha=0.7)
    plt.plot(test_indices_plot, test_predictions, label='Predicted', marker='x', linestyle='-', color='r', alpha=0.7)
    plt.title('Predicted vs Actual Values by Test Index')
    plt.xlabel('Test Sample Index')
    plt.ylabel('Precipitation (m)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{plots_dir}/test_predictions_by_index.png')
    plt.close()
    
    # plot predictions vs actuals
    plt.figure(figsize=(10, 5))
    plt.scatter(test_actuals, test_predictions, alpha=0.5)
    plt.plot([min(test_actuals), max(test_actuals)], [min(test_actuals), max(test_actuals)], 'r--')
    plt.title(f'Predictions vs Actuals (Best Model)')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.grid(True)
    plt.savefig(f'{plots_dir}/test_predictions_vs_actuals.png')
    plt.close()
    
    # Plot the selected target metric (actual vs predicted values)
    plt.figure(figsize=(10, 5))
    plt.scatter(range(len(test_actuals)), test_actuals, 
                alpha=0.7, s=20, c='blue', edgecolor='k', linewidth=0.5, label='Actual')
    plt.scatter(range(len(test_predictions)), test_predictions, 
                alpha=0.7, s=20, c='red', edgecolor='k', linewidth=0.5, label='Predicted')
    plt.title(f'Actual vs Predicted Precipitation')
    plt.xlabel('Sample Index')
    plt.ylabel('Precipitation')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{plots_dir}/test_actual_vs_predicted.png')
    plt.close()
  
    print(f'Test Loss: {test_loss:.6f}')

    # calculate metrics
    mse = mean_squared_error(test_actuals, test_predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(test_actuals, test_predictions)
    r2 = r2_score(test_actuals, test_predictions)

    print(f"MSE: {mse:.6f}", f"RMSE: {rmse:.6f}", f"MAE: {mae:.6f}", f"R²: {r2:.6f}")

# Call the test function
test_model(model, test_loader, criterion)

# Load the saved features and targets from the .pt files
atlantic_features = torch.load('storm_features_precipitation_atlantic.pt', weights_only=True)
atlantic_targets = torch.load('storm_targets_precipitation_atlantic.pt', weights_only=True)

# Select the target metric index
selected_target = atlantic_targets[:, target_metric_index]

# Print the shapes of the loaded tensors to verify
print(f"Loaded features tensor shape: {atlantic_features.shape}")
print(f"Loaded targets tensor shape: {atlantic_targets.shape}")
print(f"Selected target metric shape: {selected_target.shape}")


# Apply the same normalisation to atlantic_features
atlantic_features_normalized = atlantic_features[:, :11, :, :]   
atlantic_features_normalized = (atlantic_features_normalized - xavg_train) / (xmax_train - xmin_train)

# Combine normalised first 11 features with the remaining unnormalised features
atlantic_features = torch.cat((atlantic_features_normalized, atlantic_features[:, 11:, :, :]), dim=1)

atlantic_test_dataset = TensorDataset(atlantic_features, selected_target)
atlantic_test_loader = DataLoader(atlantic_test_dataset, batch_size=best_params['batch_size'])

# Call the test function
test_model(model, atlantic_test_loader, criterion)
