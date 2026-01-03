"""
Visualisation functions for generating figures and plots.
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import config
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def ensure_plot_dir(plot_dir):
    """Ensure plot directory exists."""
    os.makedirs(plot_dir, exist_ok=True)


def plot_timeseries(y_values, title, xlabel, ylabel, plot_dir, filename, indices=None):
    """
    Plot a timeseries of values.
    
    Args:
        y_values: Array of y values
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        plot_dir: Directory to save plot
        filename: Filename for saved plot
        indices: Optional x-axis indices
    """
    ensure_plot_dir(plot_dir)
    
    if indices is None:
        indices = range(len(y_values))
    
    plt.figure(figsize=(10, 5))
    plt.plot(indices, y_values, '-', color='blue', alpha=0.6, linewidth=0.8)
    plt.scatter(indices, y_values, alpha=0.7, s=20, c='blue', edgecolor='k', linewidth=0.5)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filepath = os.path.join(plot_dir, filename)
    plt.savefig(filepath, dpi=config.PLOT_DPI, bbox_inches='tight')
    plt.close()


def plot_distribution(y_values, title, xlabel, ylabel, plot_dir, filename, bins=30):
    """
    Plot a histogram distribution.
    
    Args:
        y_values: Array of values
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        plot_dir: Directory to save plot
        filename: Filename for saved plot
        bins: Number of bins for histogram
    """
    ensure_plot_dir(plot_dir)
    
    plt.figure(figsize=(10, 5))
    plt.hist(y_values, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filepath = os.path.join(plot_dir, filename)
    plt.savefig(filepath, dpi=config.PLOT_DPI, bbox_inches='tight')
    plt.close()


def plot_training_validation_loss(train_losses, val_losses, plot_dir, filename, num_epochs=None):
    """
    Plot training and validation loss curves.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        plot_dir: Directory to save plot
        filename: Filename for saved plot
        num_epochs: Number of epochs (if None, inferred from lengths)
    """
    ensure_plot_dir(plot_dir)
    
    if num_epochs is None:
        num_epochs = len(train_losses)
    
    # Find the best epoch (lowest validation loss)
    best_epoch_idx = np.argmin(val_losses)
    best_epoch = best_epoch_idx + 1
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs+1), train_losses, marker='o', label='Training Loss')
    plt.plot(range(1, num_epochs+1), val_losses, marker='x', label='Validation Loss')
    plt.axvline(x=best_epoch, color='g', linestyle='--', label=f'Best Epoch ({best_epoch})')
    plt.title('Training vs Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    
    filepath = os.path.join(plot_dir, filename)
    plt.savefig(filepath, dpi=config.PLOT_DPI, bbox_inches='tight')
    plt.close()
    
    print(f"Model from epoch {best_epoch} with validation loss: {val_losses[best_epoch_idx]:.6f}")


def plot_predictions_vs_actuals(predictions, actuals, plot_dir, filename, dataset_name="test", 
                                 xlabel="Sample Index", ylabel="Values"):
    """
    Plot predictions vs actuals as a timeseries.
    
    Args:
        predictions: Array of predictions
        actuals: Array of actual values
        plot_dir: Directory to save plot
        filename: Filename for saved plot
        dataset_name: Name of dataset (for title)
        xlabel: X-axis label
        ylabel: Y-axis label
    """
    ensure_plot_dir(plot_dir)
    
    indices = list(range(len(actuals)))
    plt.figure(figsize=(10, 5))
    plt.plot(indices, actuals, label='Actual', marker='o', linestyle='--', color='b', alpha=0.7)
    plt.plot(indices, predictions, label='Predicted', marker='x', linestyle='-', color='r', alpha=0.7)
    plt.title(f'Predicted vs Actual Values by {dataset_name.capitalize()} Index')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    
    filepath = os.path.join(plot_dir, filename)
    plt.savefig(filepath, dpi=config.PLOT_DPI, bbox_inches='tight')
    plt.close()


def plot_scatter_predictions_vs_actuals(predictions, actuals, plot_dir, filename, dataset_name="test",
                                        xlabel="Actual Values", ylabel="Predicted Values"):
    """
    Plot predictions vs actuals as a scatter plot.
    
    Args:
        predictions: Array of predictions
        actuals: Array of actual values
        plot_dir: Directory to save plot
        filename: Filename for saved plot
        dataset_name: Name of dataset (for title)
        xlabel: X-axis label
        ylabel: Y-axis label
    """
    ensure_plot_dir(plot_dir)
    
    plt.figure(figsize=(10, 5))
    plt.scatter(actuals, predictions, alpha=0.5)
    plt.plot([min(actuals), max(actuals)], [min(actuals), max(actuals)], 'r--', label='Perfect Prediction')
    plt.title(f'Predictions vs Actuals (Best Model) - {dataset_name.capitalize()}')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    
    filepath = os.path.join(plot_dir, filename)
    plt.savefig(filepath, dpi=config.PLOT_DPI, bbox_inches='tight')
    plt.close()


def plot_actual_vs_predicted_scatter(predictions, actuals, plot_dir, filename, dataset_name="test",
                                     xlabel="Sample Index", ylabel="Values"):
    """
    Plot actual and predicted values as scatter plots.
    
    Args:
        predictions: Array of predictions
        actuals: Array of actual values
        plot_dir: Directory to save plot
        filename: Filename for saved plot
        dataset_name: Name of dataset (for title)
        xlabel: X-axis label
        ylabel: Y-axis label
    """
    ensure_plot_dir(plot_dir)
    
    plt.figure(figsize=(10, 5))
    plt.scatter(range(len(actuals)), actuals, 
                alpha=0.7, s=20, c='blue', edgecolor='k', linewidth=0.5, label='Actual')
    plt.scatter(range(len(predictions)), predictions, 
                alpha=0.7, s=20, c='red', edgecolor='k', linewidth=0.5, label='Predicted')
    plt.title(f'Actual vs Predicted - {dataset_name.capitalize()}')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    filepath = os.path.join(plot_dir, filename)
    plt.savefig(filepath, dpi=config.PLOT_DPI, bbox_inches='tight')
    plt.close()


def plot_validation_results(predictions, actuals, plot_dir, filename):
    """
    Plot validation results.
    
    Args:
        predictions: Array of predictions
        actuals: Array of actual values
        plot_dir: Directory to save plot
        filename: Filename for saved plot
    """
    ensure_plot_dir(plot_dir)
    
    eval_indices = list(range(len(actuals)))
    plt.figure(figsize=(10, 5))
    plt.plot(eval_indices, actuals, label='Actual', marker='o', linestyle='--', color='b')
    plt.plot(eval_indices, predictions, label='Predicted', marker='x', linestyle='-', color='r')
    plt.title('Predicted vs Actual Values by Validation Index')
    plt.xlabel('Validation Index')
    plt.ylabel('Values')
    plt.legend()
    plt.grid(True)
    
    filepath = os.path.join(plot_dir, filename)
    plt.savefig(filepath, dpi=config.PLOT_DPI, bbox_inches='tight')
    plt.close()


def print_evaluation_metrics(predictions, actuals, dataset_name="test"):
    """
    Print evaluation metrics.
    
    Args:
        predictions: Array of predictions
        actuals: Array of actual values
        dataset_name: Name of dataset
    """
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)
    
    print(f'\n{dataset_name.capitalize()} Set Metrics:')
    print(f"MSE: {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"R²: {r2:.6f}")
    
    return {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}


def plot_precipitation_analysis(y_values, target_metric, plot_dir):
    """
    Create comprehensive precipitation analysis plots.
    
    Args:
        y_values: Array of precipitation values
        target_metric: Name of target metric
        plot_dir: Directory to save plots
    """
    ensure_plot_dir(plot_dir)
    
    # Timeseries plot
    plot_timeseries(
        y_values, 
        '3-H Storm Induced Precipitation',
        'Sample Index',
        'Precipitation (m)',
        plot_dir,
        'precipitation_timeseries.png'
    )
    
    # Distribution plot
    plot_distribution(
        y_values,
        f'Distribution of Storm Rainfall Values ({target_metric})',
        f'Rainfall ({target_metric})',
        'Frequency',
        plot_dir,
        'precipitation_distribution.png'
    )
    
    # Print statistics
    print(f"Statistics for {target_metric} rainfall:")
    print(f"Number of samples: {len(y_values)}")
    print(f"Min: {y_values.min():.4f}")
    print(f"Max: {y_values.max():.4f}")
    print(f"Mean: {y_values.mean():.4f}")
    print(f"Median: {np.median(y_values):.4f}")
    print(f"Standard deviation: {y_values.std():.4f}")


def create_test_plots(predictions, actuals, plot_dir, dataset_name="test", 
                      value_type="Precipitation", unit="m"):
    """
    Create all test evaluation plots.
    
    Args:
        predictions: Array of predictions
        actuals: Array of actual values
        plot_dir: Directory to save plots
        dataset_name: Name of dataset
        value_type: Type of value being predicted (e.g., "Precipitation", "Storm Surge")
        unit: Unit of measurement
    """
    ensure_plot_dir(plot_dir)
    
    # Timeseries plot
    plot_predictions_vs_actuals(
        predictions, actuals, plot_dir,
        f'test_predictions_by_index_{dataset_name}.png',
        dataset_name,
        f'{dataset_name.capitalize()} Sample Index',
        f'{value_type} ({unit})'
    )
    
    # Scatter plot
    plot_scatter_predictions_vs_actuals(
        predictions, actuals, plot_dir,
        f'test_predictions_vs_actuals_{dataset_name}.png',
        dataset_name,
        f'Actual {value_type} ({unit})',
        f'Predicted {value_type} ({unit})'
    )
    
    # Actual vs predicted scatter
    plot_actual_vs_predicted_scatter(
        predictions, actuals, plot_dir,
        f'test_actual_vs_predicted_{dataset_name}.png',
        dataset_name,
        'Sample Index',
        f'{value_type} ({unit})'
    )
    
    # Print metrics
    print_evaluation_metrics(predictions, actuals, dataset_name)


