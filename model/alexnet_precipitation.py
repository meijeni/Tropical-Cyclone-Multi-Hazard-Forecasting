import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import optuna

# Create folder for plots
plots_folder = 'validation_plots_alexnet_precipitation_pacific'
os.makedirs(plots_folder, exist_ok=True)

# === Choose target metric ===
target_metrics = ['sum', 'max', 'min', 'mean', 'median', 'p25', 'p75', 'p90', 'p95', 'p99']
TARGET_METRIC = 0

# === Model Hyperparameters (will be tuned by Optuna) ===
BATCH_SIZE = 16
LR = 0.0001
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 300

# Load the datasets from different years
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

# Concatenate all years
X = torch.cat([X_2019_2020, X_2021, X_2022, X_2023, X_2024], dim=0)
y = torch.cat([y_2019_2020, y_2021, y_2022, y_2023, y_2024], dim=0)

print(X.shape, y.shape)

# Remove first sample
X = X[1:]
y = y[1:]
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

# Plot the selected target metric (storm rainfall measurements)
plt.figure(figsize=(10, 5))
x_indices = range(len(y[:, target_metric_index]))
y_values = y[:, target_metric_index].numpy()
plt.plot(x_indices, y_values, '-', color='blue', alpha=0.6, linewidth=0.8)
plt.scatter(x_indices, y_values, 
            alpha=0.7, s=20, c='blue', edgecolor='k', linewidth=0.5)
plt.title(f'3-H Storm Induced Precipitation')
plt.xlabel('Sample Index')
plt.ylabel(f'Precipitation (m)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(plots_folder, 'precipitation_timeseries.png'), dpi=300, bbox_inches='tight')
plt.show()

# Plot histogram to show distribution of rainfall values
plt.figure(figsize=(10, 5))
plt.hist(y[:, target_metric_index].numpy(), bins=30, alpha=0.7, color='skyblue', edgecolor='black')
plt.title(f'Distribution of Storm Rainfall Values ({selected_metric})')
plt.xlabel(f'Rainfall ({selected_metric})')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(plots_folder, 'precipitation_distribution.png'), dpi=300, bbox_inches='tight')
plt.show()

# Print statistics about the target values
target_values = y[:, target_metric_index].numpy()
print(f"Statistics for {selected_metric} rainfall:")
print(f"Number of samples: {len(target_values)}")
print(f"Min: {target_values.min():.4f}")
print(f"Max: {target_values.max():.4f}")
print(f"Mean: {target_values.mean():.4f}")
print(f"Median: {np.median(target_values):.4f}")
print(f"Standard deviation: {target_values.std():.4f}")


# Split dataset into train/val/test
n_samples = len(X)
train_size, val_size = int(n_samples * 0.8), int(n_samples * 0.1)

# Define specific indices for each split
train_indices = list(range(0, 2660))
val_indices = list(range(2660, 2990))
test_indices = list(range(2990, 3313))

# Split data
X_train, y_train = X[train_indices], y[train_indices, target_metric_index]
X_val, y_val = X[val_indices], y[val_indices, target_metric_index]

# Normalize only the first 11 channels, keep the rest unchanged
X_train_normalized = X_train[:, :11, :, :]          

# Compute per-feature min, max and mean for the first 11 channels
xmin_train = torch.amin(X_train_normalized, dim=(0, 2, 3))
xmax_train = torch.amax(X_train_normalized, dim=(0, 2, 3))
xavg_train = torch.mean(X_train_normalized, dim=(0, 2, 3))

# Expand them to [1, 11, 1, 1] for broadcasting
xmin_train = xmin_train.view(1, -1, 1, 1)
xmax_train = xmax_train.view(1, -1, 1, 1)
xavg_train = xavg_train.view(1, -1, 1, 1)

# Normalize the first 11 channels
X_train_normalized = (X_train_normalized - xavg_train) / (xmax_train - xmin_train)

print("Per-feature minima:", xmin_train.flatten())
print("Per-feature maxima:", xmax_train.flatten())
print("Per-feature means: ", xavg_train.flatten())

# Combine normalized first 11 features with the remaining unnormalized features
X_train = torch.cat((X_train_normalized, X_train[:, 11:, :, :]), dim=1)

# Apply the same normalization to validation data
X_val_normalized = X_val[:, :11, :, :]   
X_val_normalized = (X_val_normalized - xavg_train) / (xmax_train - xmin_train)
X_val = torch.cat((X_val_normalized, X_val[:, 11:, :, :]), dim=1)


# Create datasets
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

# AlexNet-inspired CNN
class AlexNetPrecipitation(nn.Module):
    def __init__(self, num_features, dropout_rate=0.5, activation='relu'):
        super(AlexNetPrecipitation, self).__init__()
        
        # Choose activation function
        if activation == 'relu':
            act_fn = nn.ReLU(inplace=True)
        elif activation == 'silu':
            act_fn = nn.SiLU(inplace=True)
        
        self.features = nn.Sequential(
            nn.Conv2d(num_features, 64, kernel_size=3, stride=1, padding=1),  
            act_fn,
            nn.MaxPool2d(kernel_size=2, stride=2),  
            
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            act_fn,
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            act_fn,
            
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            act_fn,
         
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            act_fn,
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(256 * 6 * 6, 4096),
            act_fn,
            nn.Dropout(dropout_rate),
            nn.Linear(4096, 1024),
            act_fn,
            nn.Dropout(dropout_rate * 0.6),
            nn.Linear(1024, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

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
        self.best_model_path = 'best_model_precipitation_alexnet.pth'

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
        eval_indices = list(range(len(actuals)))
        plt.figure(figsize=(10, 5))
        plt.plot(eval_indices, actuals, label='Actual', marker='o', linestyle='--', color='b')
        plt.plot(eval_indices, predictions, label='Predicted', marker='x', linestyle='-', color='r')
        plt.title('Predicted vs Actual Values by Validation Index')
        plt.xlabel('Validation Index')
        plt.ylabel('Values')
        plt.legend()
        plt.grid(True)
        
        plot_filename = os.path.join(plots_folder, 'final_validation_plot.png')
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()


# Optuna objective function for hyperparameter tuning
def objective(trial):
    # Suggest hyperparameters
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.6)
    activation = trial.suggest_categorical('activation', ['relu', 'silu'])
    
    # Create data loaders with the suggested batch size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Instantiate model with suggested hyperparameters
    model = AlexNetPrecipitation(
        num_features=X_train.shape[1],
        dropout_rate=dropout_rate,
        activation=activation
    ).to(device)
    
    # Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Create trainer
    trainer = Trainer(model, optimizer, criterion, train_loader, val_loader)
    
    # Train for fewer epochs during hyperparameter tuning
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
model = AlexNetPrecipitation(
    num_features=X_train.shape[1],
    dropout_rate=best_params['dropout_rate'],
    activation=best_params['activation']
).to(device)

# Define loss and optimizer with best hyperparameters
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])

trainer = Trainer(model, optimizer, criterion, train_loader, val_loader)

# Train the model
print(f"Starting training for target metric: {selected_metric}.")
train_losses, eval_losses, (val_predictions, val_actuals) = trainer.train(num_epochs=NUM_EPOCHS)

def evaluate_model(num_epochs, train_losses, val_losses):
    # Find the best epoch (lowest validation loss)
    best_epoch_idx = np.argmin(val_losses)
    best_epoch = best_epoch_idx + 1
    
    # Plot training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs+1), train_losses, marker='o', label='Training Loss')
    plt.plot(range(1, num_epochs+1), val_losses, marker='x', label='Validation Loss')
    plt.axvline(x=best_epoch, color='g', linestyle='--', label=f'Best Epoch ({best_epoch})')
    plt.title('Training vs Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(os.path.join(plots_folder, 'training_validation_loss.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Model from epoch {best_epoch} with validation loss: {val_losses[best_epoch_idx]:.6f}")


evaluate_model(
    num_epochs=NUM_EPOCHS,
    train_losses=train_losses,
    val_losses=eval_losses,
)

# Prepare test data
X_test = X[test_indices]
y_test = y[test_indices, target_metric_index]

# Apply the same normalization to test data
X_test_normalized = X_test[:, :11, :, :]   
X_test_normalized = (X_test_normalized - xavg_train) / (xmax_train - xmin_train)
X_test = torch.cat((X_test_normalized, X_test[:, 11:, :, :]), dim=1)

test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=best_params['batch_size'])


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
def test_model(model, test_loader, criterion, dataset_name="test"):
    # Load the best model from saved checkpoint
    best_model_path = f'best_model_precipitation_alexnet.pth'
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    print("Loaded best model!")
    
    model.eval()
    test_loss, test_predictions, test_actuals = evaluate_model_loss(model, test_loader, criterion)

    # Plot predicted vs actual values by index
    test_indices = list(range(len(test_actuals)))
    plt.figure(figsize=(10, 5))
    plt.plot(test_indices, test_actuals, label='Actual', marker='o', linestyle='--', color='b', alpha=0.7)
    plt.plot(test_indices, test_predictions, label='Predicted', marker='x', linestyle='-', color='r', alpha=0.7)
    plt.title(f'Predicted vs Actual Values by {dataset_name.capitalize()} Index')
    plt.xlabel(f'{dataset_name.capitalize()} Sample Index')
    plt.ylabel('Precipitation (m)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_folder, f'test_predictions_by_index_{dataset_name}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot predictions vs actuals scatter
    plt.figure(figsize=(10, 5))
    plt.scatter(test_actuals, test_predictions, alpha=0.5)
    plt.plot([min(test_actuals), max(test_actuals)], [min(test_actuals), max(test_actuals)], 'r--')
    plt.title(f'Predictions vs Actuals (Best Model) - {dataset_name.capitalize()}')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.grid(True)
    plt.savefig(os.path.join(plots_folder, f'test_predictions_vs_actuals_{dataset_name}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot actual vs predicted precipitation
    plt.figure(figsize=(10, 5))
    plt.scatter(range(len(test_actuals)), test_actuals, 
                alpha=0.7, s=20, c='blue', edgecolor='k', linewidth=0.5, label='Actual')
    plt.scatter(range(len(test_predictions)), test_predictions, 
                alpha=0.7, s=20, c='red', edgecolor='k', linewidth=0.5, label='Predicted')
    plt.title(f'Actual vs Predicted Precipitation - {dataset_name.capitalize()}')
    plt.xlabel('Sample Index')
    plt.ylabel('Precipitation')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_folder, f'test_actual_vs_predicted_{dataset_name}.png'), dpi=300, bbox_inches='tight')
    plt.close()
  
    print(f'{dataset_name.capitalize()} Loss: {test_loss:.6f}')

    # Calculate evaluation metrics
    mse = mean_squared_error(test_actuals, test_predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(test_actuals, test_predictions)
    r2 = r2_score(test_actuals, test_predictions)

    print(f"MSE: {mse:.6f}", f"RMSE: {rmse:.6f}", f"MAE: {mae:.6f}", f"R²: {r2:.6f}")

# Test on test set
test_model(model, test_loader, criterion, "test")

# Load the Atlantic dataset
atlantic_features = torch.load('storm_features_precipitation_atlantic.pt')
atlantic_targets = torch.load('storm_targets_precipitation_atlantic.pt')

# Select the target metric
selected_target = atlantic_targets[:, target_metric_index]

print(f"Loaded features tensor shape: {atlantic_features.shape}")
print(f"Loaded targets tensor shape: {atlantic_targets.shape}")
print(f"Selected target metric shape: {selected_target.shape}")

# Check for NaN values in Atlantic data
nan_in_atlantic_features = torch.isnan(atlantic_features).any()
nan_in_atlantic_targets = torch.isnan(selected_target).any()

print(f"NaN values in atlantic_features: {nan_in_atlantic_features}")
print(f"NaN values in atlantic_targets: {nan_in_atlantic_targets}")

if nan_in_atlantic_features:
    # Count NaNs per channel
    nan_count_per_channel_atlantic = torch.isnan(atlantic_features).sum(dim=(0, 2, 3))
    print(f"NaN count per channel in atlantic_features: {nan_count_per_channel_atlantic}")
    
    # Calculate percentage of NaNs
    total_elements_atlantic = atlantic_features.numel()
    nan_percentage_atlantic = torch.isnan(atlantic_features).sum().item() / total_elements_atlantic * 100
    print(f"Percentage of NaNs in atlantic_features: {nan_percentage_atlantic:.4f}%")
    
    # Replace NaN values with 0
    atlantic_features = torch.nan_to_num(atlantic_features, nan=0.0)
    print("Replaced NaN values with 0 in atlantic_features")

if nan_in_atlantic_targets:
    # Count NaNs in targets
    nan_count_atlantic_targets = torch.isnan(selected_target).sum().item()
    print(f"Number of NaN values in atlantic_targets: {nan_count_atlantic_targets}")
    print(f"Percentage of NaNs in atlantic_targets: {nan_count_atlantic_targets / selected_target.numel() * 100:.4f}%")
    
    # Replace NaN values with 0
    selected_target = torch.nan_to_num(selected_target, nan=0.0)
    print("Replaced NaN values with 0 in atlantic_targets")

# Verify NaNs have been removed
print(f"NaN values remaining in atlantic_features: {torch.isnan(atlantic_features).any()}")
print(f"NaN values remaining in atlantic_targets: {torch.isnan(selected_target).any()}")

# Apply the same normalization to Atlantic features
atlantic_features_normalized = atlantic_features[:, :11, :, :]   
atlantic_features_normalized = (atlantic_features_normalized - xavg_train) / (xmax_train - xmin_train)
atlantic_features = torch.cat((atlantic_features_normalized, atlantic_features[:, 11:, :, :]), dim=1)

atlantic_test_dataset = TensorDataset(atlantic_features, selected_target)
atlantic_test_loader = DataLoader(atlantic_test_dataset, batch_size=best_params['batch_size'])

# Test on Atlantic dataset
test_model(model, atlantic_test_loader, criterion, "atlantic")