"""
Script for training deep learning models and evaluating them on validation data.
"""

import torch
import numpy as np
import os
from torch.utils.data import TensorDataset, DataLoader
import optuna
import config
import model
import visualisation


def normalize_features(X_train, X_val, normalize_channels=None):
    """
    Normalise features using training statistics. (for precipitation)
    
    Args:
        X_train: Training features tensor
        X_val: Validation features tensor
        normalize_channels: List of channel indices to normalise (None = normalise first 11)
    
    Returns:
        Normalised X_train, X_val, normalisation statistics
    """
    if normalize_channels is None:
        normalize_channels = config.NORMALIZE_CHANNELS_FIRST_11
    
    # Extract channels to normalise
    X_train_normalized = X_train[:, normalize_channels, :, :]
    
    # Compute per-feature min, max and mean
    xmin_train = torch.amin(X_train_normalized, dim=(0, 2, 3)).view(1, -1, 1, 1)
    xmax_train = torch.amax(X_train_normalized, dim=(0, 2, 3)).view(1, -1, 1, 1)
    xavg_train = torch.mean(X_train_normalized, dim=(0, 2, 3)).view(1, -1, 1, 1)
    
    # Normalise training data
    X_train_normalized = (X_train_normalized - xavg_train) / (xmax_train - xmin_train)
    
    # Combine normalised channels with unnormalised channels
    if len(normalize_channels) == X_train.shape[1]:
        X_train = X_train_normalized
    else:
        # Keep other channels unchanged
        other_channels = [i for i in range(X_train.shape[1]) if i not in normalize_channels]
        X_train = torch.cat((X_train_normalized, X_train[:, other_channels, :, :]), dim=1)
    
    # Apply same normalisation to validation data
    X_val_normalized = X_val[:, normalize_channels, :, :]
    X_val_normalized = (X_val_normalized - xavg_train) / (xmax_train - xmin_train)
    
    if len(normalize_channels) == X_val.shape[1]:
        X_val = X_val_normalized
    else:
        X_val = torch.cat((X_val_normalized, X_val[:, other_channels, :, :]), dim=1)
    
    norm_stats = {
        'xmin': xmin_train,
        'xmax': xmax_train,
        'xavg': xavg_train,
        'normalize_channels': normalize_channels
    }
    
    return X_train, X_val, norm_stats


def normalize_features_surge(X_train, X_val):
    """
    Normalise features for storm surge model (first 11 channels and channel 13).
    
    Args:
        X_train: Training features tensor
        X_val: Validation features tensor
    
    Returns:
        Normalised X_train, X_val, normalisation statistics
    """
    # Normalise first 11 channels
    X_train_norm_first = X_train[:, :11, :, :]
    xmin_first = torch.amin(X_train_norm_first, dim=(0, 2, 3)).view(1, -1, 1, 1)
    xmax_first = torch.amax(X_train_norm_first, dim=(0, 2, 3)).view(1, -1, 1, 1)
    xavg_first = torch.mean(X_train_norm_first, dim=(0, 2, 3)).view(1, -1, 1, 1)
    X_train_norm_first = (X_train_norm_first - xavg_first) / (xmax_first - xmin_first)
    
    # Normalise channel 13
    X_train_norm_ch13 = X_train[:, 12:13, :, :]
    xmin_ch13 = torch.amin(X_train_norm_ch13, dim=(0, 2, 3)).view(1, -1, 1, 1)
    xmax_ch13 = torch.amax(X_train_norm_ch13, dim=(0, 2, 3)).view(1, -1, 1, 1)
    xavg_ch13 = torch.mean(X_train_norm_ch13, dim=(0, 2, 3)).view(1, -1, 1, 1)
    X_train_norm_ch13 = (X_train_norm_ch13 - xavg_ch13) / (xmax_ch13 - xmin_ch13)
    
    # Combine all parts
    X_train = torch.cat((
        X_train_norm_first,
        X_train[:, 11:12, :, :],  # Channel 12 unchanged
        X_train_norm_ch13,
        X_train[:, 13:, :, :]     # Channels 14+ unchanged
    ), dim=1)
    
    # Apply same normalisation to validation data
    X_val_norm_first = (X_val[:, :11, :, :] - xavg_first) / (xmax_first - xmin_first)
    X_val_norm_ch13 = (X_val[:, 12:13, :, :] - xavg_ch13) / (xmax_ch13 - xmin_ch13)
    
    X_val = torch.cat((
        X_val_norm_first,
        X_val[:, 11:12, :, :],
        X_val_norm_ch13,
        X_val[:, 13:, :, :]
    ), dim=1)
    
    norm_stats = {
        'xmin_first': xmin_first,
        'xmax_first': xmax_first,
        'xavg_first': xavg_first,
        'xmin_ch13': xmin_ch13,
        'xmax_ch13': xmax_ch13,
        'xavg_ch13': xavg_ch13
    }
    
    return X_train, X_val, norm_stats


def load_data(task_type='precipitation', data_suffixes=None):
    """
    Load processed data files.
    
    Args:
        task_type: 'precipitation' or 'stormsurge'
        data_suffixes: List of suffixes for data files 
                      For precipitation: ['_2019_2020', '_2021', '_2022', '_2023', '_2024']
                      For stormsurge: ['_firsthalf', '_secondhalf']
    
    Returns:
        X, y tensors
    """
    if data_suffixes is None:
        if task_type == 'precipitation':
            data_suffixes = ['_2019_2020', '_2021', '_2022', '_2023', '_2024']
        else:  # stormsurge
            data_suffixes = ['_firsthalf', '_secondhalf']
    
    X_list, y_list = [], []
    
    for suffix in data_suffixes:
        if task_type == 'precipitation':
            features_file = config.FEATURES_PATTERN.format(f"precipitation{suffix}")
            targets_file = config.TARGETS_PATTERN.format(f"precipitation{suffix}")
        else:  # stormsurge
            features_file = config.FEATURES_PATTERN.format(f"wh{suffix}")
            targets_file = config.TARGETS_PATTERN.format(f"swh{suffix}")
        
        features_path = os.path.join(config.PROCESSED_DATA_DIR, features_file)
        targets_path = os.path.join(config.PROCESSED_DATA_DIR, targets_file)
        
        try:
            X_part = torch.load(features_path, weights_only=True)
            y_part = torch.load(targets_path, weights_only=True)
            X_list.append(X_part)
            y_list.append(y_part)
        except FileNotFoundError as e:
            print(f"Warning: Could not load {features_path} or {targets_path}: {e}")
            continue
    
    if not X_list:
        raise ValueError("No data files found!")
    
    X = torch.cat(X_list, dim=0)
    y = torch.cat(y_list, dim=0)
    
    # Remove first sample (common in original code)
    X = X[1:]
    y = y[1:]
    
    return X, y


def prepare_data(X, y, task_type='precipitation', target_metric_index=0, 
                 train_indices=None, val_indices=None):
    """
    Prepare data for training: split and normalise.
    
    Args:
        X: Features tensor
        y: Targets tensor
        task_type: 'precipitation' or 'stormsurge'
        target_metric_index: Index of target metric (for precipitation)
        train_indices: Training indices (if None, use config defaults)
        val_indices: Validation indices (if None, use config defaults)
    
    Returns:
        X_train, y_train, X_val, y_val, normalisation statistics
    """
    # Select target metric for precipitation
    if task_type == 'precipitation':
        if y.dim() > 1:
            y = y[:, target_metric_index]
    else:  # stormsurge - already single value
        if y.dim() > 1:
            y = y.squeeze()
    
    # Use default indices if not provided
    if train_indices is None:
        train_indices = config.TRAIN_INDICES if task_type == 'precipitation' else config.TRAIN_INDICES_SURGE
    if val_indices is None:
        val_indices = config.VAL_INDICES if task_type == 'precipitation' else config.VAL_INDICES_SURGE
    
    # Split data
    X_train, y_train = X[train_indices], y[train_indices]
    X_val, y_val = X[val_indices], y[val_indices]
    
    # Normalise features
    if task_type == 'precipitation':
        X_train, X_val, norm_stats = normalize_features(X_train, X_val)
    else:  # stormsurge
        X_train, X_val, norm_stats = normalize_features_surge(X_train, X_val)
    
    return X_train, y_train, X_val, y_val, norm_stats


def create_data_loaders(X_train, y_train, X_val, y_val, batch_size):
    """
    Create data loaders for training and validation.
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        batch_size: Batch size
    
    Returns:
        train_loader, val_loader
    """
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    return train_loader, val_loader


def hyperparameter_tuning(X_train, y_train, X_val, y_val, model_type, task_type):
    """
    Perform hyperparameter tuning using Optuna.
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        model_type: 'alexnet', 'resnet', or 'senet'
        task_type: 'precipitation' or 'stormsurge'
    
    Returns:
        Best hyperparameters dictionary
    """
    def objective(trial):
        # Suggest hyperparameters
        lr = trial.suggest_float('lr', *config.HYPERPARAMETER_RANGES['lr'], log=True)
        batch_size = trial.suggest_categorical('batch_size', config.HYPERPARAMETER_RANGES['batch_size'])
        weight_decay = trial.suggest_float('weight_decay', *config.HYPERPARAMETER_RANGES['weight_decay'], log=True)
        dropout_rate = trial.suggest_float('dropout_rate', *config.HYPERPARAMETER_RANGES['dropout_rate'])
        activation = trial.suggest_categorical('activation', config.HYPERPARAMETER_RANGES['activation'])
        
        # Create data loaders
        train_loader, val_loader = create_data_loaders(X_train, y_train, X_val, y_val, batch_size)
        
        # Create model
        model_instance = model.create_model(
            model_type, 
            X_train.shape[1], 
            dropout_rate=dropout_rate, 
            activation=activation,
            task_type=task_type
        )
        
        # Create optimiser
        criterion = torch.nn.MSELoss()
        optimizer = model.create_optimizer(model_instance, 'adam', lr, weight_decay)
        
        # Create trainer
        best_model_path = os.path.join(config.MODEL_DIR, 'temp_best_model.pth')
        trainer = model.Trainer(model_instance, optimizer, criterion, train_loader, val_loader, best_model_path)
        
        # Train for reduced epochs
        train_losses, eval_losses, _ = trainer.train(num_epochs=config.OPTUNA_TUNING_EPOCHS)
        
        # Return best validation loss
        return min(eval_losses)
    
    # Run Optuna study
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=config.OPTUNA_N_TRIALS)
    
    print("Best trial:")
    print(f"  Value: {study.best_value}")
    print("  Params: ")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")
    
    return study.best_params


def train_model(X_train, y_train, X_val, y_val, model_type, task_type, 
                hyperparameters=None, num_epochs=None, plot_dir=None):
    """
    Train a model with given hyperparameters.
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        model_type: 'alexnet', 'resnet', or 'senet'
        task_type: 'precipitation' or 'stormsurge'
        hyperparameters: Dictionary of hyperparameters (if None, use defaults)
        num_epochs: Number of training epochs (if None, use config default)
        plot_dir: Directory for saving plots
    
    Returns:
        Trained model, training history, normalisation statistics
    """
    if num_epochs is None:
        num_epochs = config.NUM_EPOCHS
    
    if hyperparameters is None:
        hyperparameters = {
            'lr': config.LEARNING_RATE,
            'batch_size': config.BATCH_SIZE,
            'weight_decay': config.WEIGHT_DECAY,
            'dropout_rate': config.DROPOUT_RATE,
            'activation': 'silu'
        }
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        X_train, y_train, X_val, y_val, hyperparameters['batch_size']
    )
    
    # Create model
    model_instance = model.create_model(
        model_type,
        X_train.shape[1],
        dropout_rate=hyperparameters['dropout_rate'],
        activation=hyperparameters['activation'],
        task_type=task_type
    )
    
    # Create optimiser
    criterion = torch.nn.MSELoss()
    optimizer = model.create_optimizer(
        model_instance, 'adam', hyperparameters['lr'], hyperparameters['weight_decay']
    )
    
    # Determine model path
    model_name_key = f'{task_type}_{model_type}'
    model_filename = config.MODEL_NAMES.get(model_name_key, f'best_model_{task_type}_{model_type}.pth')
    best_model_path = os.path.join(config.MODEL_DIR, model_filename)
    
    # Create trainer
    trainer = model.Trainer(model_instance, optimizer, criterion, train_loader, val_loader, best_model_path)
    
    # Train model
    print(f"Starting training for {task_type} with {model_type}...")
    train_losses, eval_losses, (val_predictions, val_actuals) = trainer.train(num_epochs=num_epochs)
    
    # Create plots
    if plot_dir is None:
        plot_dir = config.PLOT_DIRS.get(model_name_key, 'plots')
    
    visualisation.plot_training_validation_loss(
        train_losses, eval_losses, plot_dir, 'training_validation_loss.png', num_epochs
    )
    visualisation.plot_validation_results(val_predictions, val_actuals, plot_dir, 'final_validation_plot.png')
    
    return model_instance, {'train_losses': train_losses, 'eval_losses': eval_losses}, None


def main(task_type='precipitation', model_type='resnet', use_hyperparameter_tuning=True):
    """
    Main training function.
    
    Args:
        task_type: 'precipitation' or 'stormsurge'
        model_type: 'alexnet', 'resnet', or 'senet'
        use_hyperparameter_tuning: Whether to use Optuna for hyperparameter tuning
    """
    print(f"Training {model_type} model for {task_type} prediction")
    print(f"Using device: {config.DEVICE}")
    
    # Load data
    X, y = load_data(task_type)
    print(f"Loaded data: X shape {X.shape}, y shape {y.shape}")
    
    # Check for NaN values
    if torch.isnan(X).any():
        print("Warning: NaN values found in X, replacing with 0")
        X = torch.nan_to_num(X, nan=0.0)
    if torch.isnan(y).any():
        print("Warning: NaN values found in y, replacing with 0")
        y = torch.nan_to_num(y, nan=0.0)
    
    # For storm surge, remove zero targets
    if task_type == 'stormsurge':
        zero_indices = torch.where(y == 0)[0]
        if len(zero_indices) > 0:
            print(f"Removing {len(zero_indices)} samples with zero targets")
            keep_indices = torch.tensor([i for i in range(X.shape[0]) if i not in zero_indices])
            X = torch.index_select(X, 0, keep_indices)
            y = torch.index_select(y, 0, keep_indices)
    
    # Prepare data
    target_metric_index = config.TARGET_METRIC_INDEX if task_type == 'precipitation' else 0
    X_train, y_train, X_val, y_val, norm_stats = prepare_data(
        X, y, task_type, target_metric_index
    )
    
    # Create data loaders for analysis
    train_loader, val_loader = create_data_loaders(X_train, y_train, X_val, y_val, config.BATCH_SIZE)
    
    # Plot data analysis (for precipitation)
    if task_type == 'precipitation':
        plot_dir = config.PLOT_DIRS.get(f'{task_type}_{model_type}', 'plots')
        visualisation.plot_precipitation_analysis(
            y_train.numpy(), 
            config.TARGET_METRICS[target_metric_index],
            plot_dir
        )
    
    # Hyperparameter tuning
    best_params = None
    if use_hyperparameter_tuning:
        print("Starting hyperparameter tuning...")
        best_params = hyperparameter_tuning(X_train, y_train, X_val, y_val, model_type, task_type)
    else:
        best_params = {
            'lr': config.LEARNING_RATE,
            'batch_size': config.BATCH_SIZE,
            'weight_decay': config.WEIGHT_DECAY,
            'dropout_rate': config.DROPOUT_RATE,
            'activation': 'silu'
        }
    
    # Train final model
    plot_dir = config.PLOT_DIRS.get(f'{task_type}_{model_type}', 'plots')
    trained_model, history, _ = train_model(
        X_train, y_train, X_val, y_val, model_type, task_type,
        hyperparameters=best_params, plot_dir=plot_dir
    )
    
    print("Training complete!")
    return trained_model, history, norm_stats


if __name__ == "__main__":
    # Example usage:
    main('precipitation', 'resnet', use_hyperparameter_tuning=True)
    # main('stormsurge', 'resnet', use_hyperparameter_tuning=True)
    pass

