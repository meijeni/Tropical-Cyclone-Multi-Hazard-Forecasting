"""
Prediction script for generating outputs on datasets excluded from training.
Handles inference for test, operational, and unseen future datasets using trained models.
"""

import torch
import numpy as np
import os
from torch.utils.data import TensorDataset, DataLoader
import config
import model
import visualisation
import training


def load_trained_model(model_type, task_type, num_features):
    """
    Load a trained model from checkpoint.
    
    Args:
        model_type: 'alexnet', 'resnet', or 'senet'
        task_type: 'precipitation' or 'stormsurge'
        num_features: Number of input features
    
    Returns:
        Loaded model instance
    """
    # Determine model path
    model_name_key = f'{task_type}_{model_type}'
    model_filename = config.MODEL_NAMES.get(model_name_key, f'best_model_{task_type}_{model_type}.pth')
    model_path = os.path.join(config.MODEL_DIR, model_filename)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    # Create model architecture
    # We need to know the hyperparameters - for now use defaults
    # In production, these should be saved with the model
    model_instance = model.create_model(
        model_type,
        num_features,
        dropout_rate=config.DROPOUT_RATE,
        activation='silu',  # Default activation
        task_type=task_type
    )
    
    # Load weights
    model_instance.load_state_dict(torch.load(model_path, weights_only=True))
    model_instance.eval()
    
    print(f"Loaded model from {model_path}")
    return model_instance


def normalize_test_features(X_test, norm_stats, task_type='precipitation'):
    """
    Normalise test features using training statistics.
    
    Args:
        X_test: Test features tensor
        norm_stats: Normalisation statistics from training
        task_type: 'precipitation' or 'stormsurge'
    
    Returns:
        Normalised X_test
    """
    if task_type == 'precipitation':
        normalize_channels = norm_stats.get('normalize_channels', config.NORMALIZE_CHANNELS_FIRST_11)
        xmin = norm_stats['xmin']
        xmax = norm_stats['xmax']
        xavg = norm_stats['xavg']
        
        X_test_normalized = X_test[:, normalize_channels, :, :]
        X_test_normalized = (X_test_normalized - xavg) / (xmax - xmin)
        
        if len(normalize_channels) == X_test.shape[1]:
            X_test = X_test_normalized
        else:
            other_channels = [i for i in range(X_test.shape[1]) if i not in normalize_channels]
            X_test = torch.cat((X_test_normalized, X_test[:, other_channels, :, :]), dim=1)
    
    else:  # stormsurge
        xmin_first = norm_stats['xmin_first']
        xmax_first = norm_stats['xmax_first']
        xavg_first = norm_stats['xavg_first']
        xmin_ch13 = norm_stats['xmin_ch13']
        xmax_ch13 = norm_stats['xmax_ch13']
        xavg_ch13 = norm_stats['xavg_ch13']
        
        X_test_norm_first = (X_test[:, :11, :, :] - xavg_first) / (xmax_first - xmin_first)
        X_test_norm_ch13 = (X_test[:, 12:13, :, :] - xavg_ch13) / (xmax_ch13 - xmin_ch13)
        
        X_test = torch.cat((
            X_test_norm_first,
            X_test[:, 11:12, :, :],
            X_test_norm_ch13,
            X_test[:, 13:, :, :]
        ), dim=1)
    
    return X_test


def predict(model_instance, data_loader):
    """
    Make predictions using a trained model.
    
    Args:
        model_instance: Trained model
        data_loader: DataLoader for test data
    
    Returns:
        predictions, actuals arrays
    """
    model_instance.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            outputs = model_instance(inputs.to(config.DEVICE))
            predictions.extend(outputs.cpu().numpy().flatten())
            actuals.extend(targets.cpu().numpy().flatten())
    
    return np.array(predictions), np.array(actuals)


def evaluate_test_set(task_type='precipitation', model_type='resnet', 
                     test_indices=None, norm_stats=None):
    """
    Evaluate model on test set.
    
    Args:
        task_type: 'precipitation' or 'stormsurge'
        model_type: 'alexnet', 'resnet', or 'senet'
        test_indices: Test indices (if None, use config defaults)
        norm_stats: Normalisation statistics (if None, will need to recompute)
    
    Returns:
        predictions, actuals, metrics dictionary
    """
    print(f"Evaluating {model_type} model on {task_type} test set")
    
    # Load data
    X, y = training.load_data(task_type)
    
    # Check for NaN values
    if torch.isnan(X).any():
        X = torch.nan_to_num(X, nan=0.0)
    if torch.isnan(y).any():
        y = torch.nan_to_num(y, nan=0.0)
    
    # For storm surge, remove zero targets
    if task_type == 'stormsurge':
        zero_indices = torch.where(y == 0)[0]
        if len(zero_indices) > 0:
            keep_indices = torch.tensor([i for i in range(X.shape[0]) if i not in zero_indices])
            X = torch.index_select(X, 0, keep_indices)
            y = torch.index_select(y, 0, keep_indices)
    
    # Select target metric
    target_metric_index = config.TARGET_METRIC_INDEX if task_type == 'precipitation' else 0
    if y.dim() > 1:
        y = y[:, target_metric_index] if task_type == 'precipitation' else y.squeeze()
    
    # Get test indices
    if test_indices is None:
        test_indices = config.TEST_INDICES if task_type == 'precipitation' else config.TEST_INDICES_SURGE
    
    # Split test data
    X_test = X[test_indices]
    y_test = y[test_indices]
    
    # Normalise test data
    # If norm_stats not provided, we need to compute from training data
    if norm_stats is None:
        print("Warning: Normalisation statistics not provided. Computing from training data...")
        X_train, y_train, X_val, y_val, norm_stats = training.prepare_data(
            X, y, task_type, target_metric_index
        )
    
    X_test = normalize_test_features(X_test, norm_stats, task_type)
    
    # Create data loader
    test_dataset = TensorDataset(X_test, y_test)
    batch_size = config.BATCH_SIZE  # Could be loaded from saved hyperparameters
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Load model
    model_instance = load_trained_model(model_type, task_type, X_test.shape[1])
    
    # Make predictions
    predictions, actuals = predict(model_instance, test_loader)
    
    # Create plots
    plot_dir = config.PLOT_DIRS.get(f'{task_type}_{model_type}', 'plots')
    value_type = "Precipitation" if task_type == 'precipitation' else "Storm Surge"
    unit = "m"
    
    visualisation.create_test_plots(
        predictions, actuals, plot_dir, "test", value_type, unit
    )
    
    # Print metrics
    metrics = visualisation.print_evaluation_metrics(predictions, actuals, "test")
    
    return predictions, actuals, metrics


def predict_atlantic_data(task_type='precipitation', model_type='resnet', norm_stats=None):
    """
    Make predictions on Atlantic dataset (transfer learning evaluation).
    
    Args:
        task_type: 'precipitation' or 'stormsurge'
        model_type: 'alexnet', 'resnet', or 'senet'
        norm_stats: Normalisation statistics from training
    
    Returns:
        predictions, actuals, metrics dictionary
    """
    print(f"Evaluating {model_type} model on Atlantic {task_type} data")
    
    # Load Atlantic data
    if task_type == 'precipitation':
        features_file = config.FEATURES_PATTERN.format("precipitation_atlantic")
        targets_file = config.TARGETS_PATTERN.format("precipitation_atlantic")
    else:  # stormsurge
        features_file = config.FEATURES_PATTERN.format("wh_atlantic")
        targets_file = config.TARGETS_PATTERN.format("wh_atlantic")
    
    features_path = os.path.join(config.PROCESSED_DATA_DIR, features_file)
    targets_path = os.path.join(config.PROCESSED_DATA_DIR, targets_file)
    
    if not os.path.exists(features_path) or not os.path.exists(targets_path):
        raise FileNotFoundError(f"Atlantic data not found: {features_path} or {targets_path}")
    
    atlantic_features = torch.load(features_path, weights_only=True)
    atlantic_targets = torch.load(targets_path, weights_only=True)
    
    # Select target metric
    target_metric_index = config.TARGET_METRIC_INDEX if task_type == 'precipitation' else 0
    if atlantic_targets.dim() > 1:
        atlantic_targets = atlantic_targets[:, target_metric_index] if task_type == 'precipitation' else atlantic_targets.squeeze()
    
    # Check for NaN values
    if torch.isnan(atlantic_features).any():
        atlantic_features = torch.nan_to_num(atlantic_features, nan=0.0)
    if torch.isnan(atlantic_targets).any():
        atlantic_targets = torch.nan_to_num(atlantic_targets, nan=0.0)
    
    # For storm surge, remove zero targets
    if task_type == 'stormsurge':
        zero_indices = torch.where(atlantic_targets == 0)[0]
        if len(zero_indices) > 0:
            keep_indices = torch.tensor([i for i in range(atlantic_features.shape[0]) if i not in zero_indices])
            atlantic_features = torch.index_select(atlantic_features, 0, keep_indices)
            atlantic_targets = torch.index_select(atlantic_targets, 0, keep_indices)
    
    # Normalise Atlantic data
    if norm_stats is None:
        print("Warning: Normalisation statistics not provided. Loading training data to compute...")
        X, y = training.load_data(task_type)
        X_train, y_train, X_val, y_val, norm_stats = training.prepare_data(
            X, y, task_type, target_metric_index
        )
    
    atlantic_features = normalize_test_features(atlantic_features, norm_stats, task_type)
    
    # Create data loader
    atlantic_dataset = TensorDataset(atlantic_features, atlantic_targets)
    batch_size = config.BATCH_SIZE
    atlantic_loader = DataLoader(atlantic_dataset, batch_size=batch_size)
    
    # Load model
    model_instance = load_trained_model(model_type, task_type, atlantic_features.shape[1])
    
    # Make predictions
    predictions, actuals = predict(model_instance, atlantic_loader)
    
    # Create plots
    plot_dir = config.PLOT_DIRS.get(f'{task_type}_{model_type}', 'plots')
    value_type = "Precipitation" if task_type == 'precipitation' else "Storm Surge"
    unit = "m"
    
    visualisation.create_test_plots(
        predictions, actuals, plot_dir, "atlantic", value_type, unit
    )
    
    # Print metrics
    metrics = visualisation.print_evaluation_metrics(predictions, actuals, "atlantic")
    
    return predictions, actuals, metrics


def predict_future_data(X_future, model_type, task_type, norm_stats=None):
    """
    Make predictions on future/operational data.
    
    Args:
        X_future: Future features tensor (already processed)
        model_type: 'alexnet', 'resnet', or 'senet'
        task_type: 'precipitation' or 'stormsurge'
        norm_stats: Normalisation statistics from training
    
    Returns:
        predictions array
    """
    print(f"Making predictions on future data using {model_type} model")
    
    # Normalise future data
    if norm_stats is None:
        raise ValueError("Normalisation statistics must be provided for future predictions")
    
    X_future = normalize_test_features(X_future, norm_stats, task_type)
    
    # Create data loader
    # For future data, we don't have targets, so create dummy targets
    dummy_targets = torch.zeros(X_future.shape[0])
    future_dataset = TensorDataset(X_future, dummy_targets)
    batch_size = config.BATCH_SIZE
    future_loader = DataLoader(future_dataset, batch_size=batch_size)
    
    # Load model
    model_instance = load_trained_model(model_type, task_type, X_future.shape[1])
    
    # Make predictions
    predictions, _ = predict(model_instance, future_loader)
    
    return predictions


def main(task_type='precipitation', model_type='resnet', evaluate_test=True, evaluate_atlantic=True):
    """
    Main prediction function.
    
    Args:
        task_type: 'precipitation' or 'stormsurge'
        model_type: 'alexnet', 'resnet', or 'senet'
        evaluate_test: Whether to evaluate on test set
        evaluate_atlantic: Whether to evaluate on Atlantic data
    """
    print(f"Running predictions for {task_type} using {model_type} model")
    
    # Load normalisation statistics from training
    # In practise, these should be saved during training
    X, y = training.load_data(task_type)
    target_metric_index = config.TARGET_METRIC_INDEX if task_type == 'precipitation' else 0
    X_train, y_train, X_val, y_val, norm_stats = training.prepare_data(
        X, y, task_type, target_metric_index
    )
    
    results = {}
    
    # Evaluate on test set
    if evaluate_test:
        print("\n" + "="*50)
        print("Evaluating on test set...")
        print("="*50)
        test_predictions, test_actuals, test_metrics = evaluate_test_set(
            task_type, model_type, norm_stats=norm_stats
        )
        results['test'] = {
            'predictions': test_predictions,
            'actuals': test_actuals,
            'metrics': test_metrics
        }
    
    # Evaluate on Atlantic data
    if evaluate_atlantic:
        print("\n" + "="*50)
        print("Evaluating on Atlantic data...")
        print("="*50)
        try:
            atlantic_predictions, atlantic_actuals, atlantic_metrics = predict_atlantic_data(
                task_type, model_type, norm_stats=norm_stats
            )
            results['atlantic'] = {
                'predictions': atlantic_predictions,
                'actuals': atlantic_actuals,
                'metrics': atlantic_metrics
            }
        except FileNotFoundError as e:
            print(f"Warning: Could not evaluate on Atlantic data: {e}")
    
    return results


if __name__ == "__main__":
    # Example usage:
    results = main('precipitation', 'resnet', evaluate_test=True, evaluate_atlantic=True)
    # results = main('stormsurge', 'resnet', evaluate_test=True, evaluate_atlantic=True)
    pass

