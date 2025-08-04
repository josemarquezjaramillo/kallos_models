"""
Model Evaluation Module
======================

This module provides functionality for evaluating trained time series forecasting models
on hold-out test data. It handles loading saved models and scalers, generating forecasts,
calculating performance metrics, and creating visualization charts.

Example:
    from kallos_models import evaluation
    
    # Evaluate a trained model on hold-out test data
    evaluation.run_evaluation(
        model_path="./trained_models/gru_BTC.pt",
        scaler_path="./trained_models/gru_BTC_scaler.pkl",
        asset_id="BTC",
        test_start_date="2023-01-01",
        test_end_date="2023-03-31",
        db_url="postgresql://user:pass@localhost:5432/crypto_db",
        target_col="close",
        feature_groups={
            "volume_features": ["volume"],
            "bounded_features": ["rsi"]
        },
        output_path="./evaluation_results"
    )
"""

import json
import logging
import os
import pickle
from typing import Dict, List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.metrics import mae, wmape, rmse
from scipy.stats import spearmanr, kendalltau

from . import datasets
from .architectures import load_model_with_custom_loss
from .preprocessing import transform_features_to_dataframe

# Set up a basic logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def calculate_directional_accuracy(true_series: TimeSeries, forecast_series: TimeSeries) -> float:
    """
    Calculate the directional accuracy - the percentage of times the model correctly
    predicts the direction of movement (up/down).
    
    Parameters:
        true_series (TimeSeries): The ground truth time series
        forecast_series (TimeSeries): The forecasted time series
    
    Returns:
        float: The directional accuracy as a percentage (0-100)
    """
    # Convert to pandas for easier manipulation
    true_df = true_series.to_series()
    forecast_df = forecast_series.to_series()

    # Calculate daily returns (change direction)
    true_direction = true_df.diff().dropna()
    forecast_direction = forecast_df.diff().dropna()
    
    # Align the two series
    true_direction, forecast_direction = true_direction.align(forecast_direction, join='inner')
    
    # Calculate if the directions match (both positive or both negative)
    # Note: zeros are considered correct if both forecast and true are zero
    correct_direction = true_direction * forecast_direction > 0
    correct_direction = correct_direction.astype(float)  # Convert boolean to float for mean calculation
    
    # Calculate percentage of correct directions
    accuracy = correct_direction.mean() * 100.0
    return accuracy


def calculate_rank_correlation(true_series: TimeSeries, forecast_series: TimeSeries) -> Dict[str, float]:
    """
    Calculate Spearman and Kendall rank correlations between predicted and actual values.
    These metrics measure how well the model ranks the relative magnitude of returns.
    
    Parameters:
        true_series (TimeSeries): The ground truth time series
        forecast_series (TimeSeries): The forecasted time series
    
    Returns:
        Dict[str, float]: Dictionary with Spearman and Kendall correlation coefficients
    """
    # Convert to pandas for correlation calculation
    true_values = true_series.to_series().values
    forecast_values = forecast_series.to_series().values

    # Calculate Spearman rank correlation
    spearman_corr, _ = spearmanr(true_values, forecast_values)
    
    # Calculate Kendall tau rank correlation
    kendall_corr, _ = kendalltau(true_values, forecast_values)
    
    return {
        "spearman_correlation": spearman_corr,
        "kendall_correlation": kendall_corr
    }


def calculate_directional_weighted_mse(y_true: TimeSeries, y_hat: TimeSeries, direction_penalty: float = 5.0) -> float:
    """
    Calculate Mean Squared Error with a higher penalty for predictions that get the direction wrong.
    
    This metric penalizes errors differently based on whether they got the direction right:
    - If sign(prediction) == sign(actual): Normal squared error
    - If sign(prediction) != sign(actual): Squared error multiplied by penalty factor
    
    Parameters:
        true_series (TimeSeries): The ground truth time series
        forecast_series (TimeSeries): The forecasted time series
        wrong_direction_penalty (float): Multiplier for errors with wrong direction (default: 3.0)
    
    Returns:
        float: The directional-weighted MSE
    """
    # Convert to pandas for calculation
    y_true = y_true.to_series()
    y_hat = y_hat.to_series()

    # Check if directions match
    direction_indicator = (y_hat * y_true > 0).astype(float)
    
    # Calculate errors
    mse = (y_hat - y_true) ** 2
    
    # Apply penalties based on direction correctness
    adjusted_mse = mse * (direction_indicator + direction_penalty * (1 - direction_indicator))
    
    
    # Calculate mean
    return np.mean(adjusted_mse)


def calculate_expectancy_score(true_series: TimeSeries, forecast_series: TimeSeries) -> Dict[str, float]:
    """
    Calculate trading expectancy score based on the model's predictions.
    
    The expectancy score indicates the expected profit per trade and is calculated as:
    Expectancy = (Win% * Average Win) - (Loss% * Average Loss)
    
    Parameters:
        true_series (TimeSeries): The ground truth time series
        forecast_series (TimeSeries): The forecasted time series
    
    Returns:
        Dict[str, float]: Dictionary with expectancy metrics:
            - win_rate: Percentage of predictions with correct direction
            - avg_win: Average return when direction is correct
            - avg_loss: Average return when direction is wrong
            - expectancy: Expected return per trade
    """
    # Convert to pandas
    true_df = true_series.to_series()
    forecast_df = forecast_series.to_series()

    # Calculate pct_change for true values (these would be the actual returns)
    true_returns = true_df.pct_change().dropna()
    
    # Get the predicted directions (1 for positive, -1 for negative)
    forecast_aligned = forecast_df.reindex(true_returns.index)
    predicted_directions = np.sign(forecast_aligned)
    
    # Actual directions
    actual_directions = np.sign(true_returns)
    
    # Identify wins (correct direction predictions)
    wins = (predicted_directions * actual_directions) > 0
    
    # Win rate
    win_rate = np.mean(wins)
    
    # Calculate average win and average loss
    if np.any(wins):
        avg_win = np.mean(np.abs(true_returns.values[wins]))
    else:
        avg_win = 0
    
    if np.any(~wins):
        avg_loss = np.mean(np.abs(true_returns.values[~wins]))
    else:
        avg_loss = 0
    
    # Calculate expectancy
    expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
    
    return {
        "win_rate": float(win_rate * 100),  # Convert to percentage
        "avg_win": float(avg_win),
        "avg_loss": float(avg_loss),
        "expectancy": float(expectancy)
    }


def calculate_confusion_metrics(true_series: TimeSeries, forecast_series: TimeSeries) -> Dict[str, float]:
    """
    Calculate classification metrics for directional predictions.
    
    Parameters:
        true_series (TimeSeries): The ground truth time series
        forecast_series (TimeSeries): The forecasted time series
    
    Returns:
        Dict[str, float]: Dictionary with classification metrics:
            - precision: Precision for positive return predictions
            - recall: Recall for positive return predictions
            - f1_score: F1 score (harmonic mean of precision and recall)
    """
    # Convert to pandas
    true_df = true_series.to_series()
    forecast_df = forecast_series.to_series()

    # Convert to binary classifications (positive vs non-positive)
    true_positive = (true_df > 0).astype(int)
    pred_positive = (forecast_df > 0).astype(int)
    
    # Align the series
    true_aligned, pred_aligned = true_positive.align(pred_positive, join='inner')
    
    # True positives, false positives, false negatives
    tp = np.sum((true_aligned == 1) & (pred_aligned == 1))
    fp = np.sum((true_aligned == 0) & (pred_aligned == 1))
    fn = np.sum((true_aligned == 1) & (pred_aligned == 0))
    
    # Calculate precision, recall, F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1_score)
    }


def generate_evaluation_report(true_series: TimeSeries, forecast_series: TimeSeries) -> Dict[str, float]:
    """
    Calculate performance metrics comparing forecasted values against actual values.
    
    Parameters:
        true_series (TimeSeries): The ground truth time series with actual values
        forecast_series (TimeSeries): The forecasted time series from the model
    
    Returns:
        Dict[str, float]: A dictionary containing performance metrics:
            - rmse: Root Mean Squared Error
            - mae: Mean Absolute Error
            - wmape: Weighted Mean Absolute Percentage Error
            - directional_accuracy: Percentage of correct direction predictions
            - directional_weighted_mse: MSE with higher penalty for wrong direction
            - spearman_correlation: Spearman rank correlation coefficient
            - kendall_correlation: Kendall rank correlation coefficient
            - win_rate: Percentage of predictions with correct direction
            - avg_win: Average return when direction is correct
            - avg_loss: Average return when direction is wrong
            - expectancy: Expected return per trade
            - precision: Precision for positive return predictions
            - recall: Recall for positive return predictions
            - f1_score: F1 score for directional predictions
    
    Notes:
        - Lower values indicate better performance for error metrics (RMSE, MAE, WMAPE)
        - Higher values are better for directional accuracy and correlation metrics
        - The function logs the metrics for immediate feedback
    
    Example:
        >>> from darts import TimeSeries
        >>> import numpy as np
        >>> 
        >>> # Create sample true and forecasted series
        >>> times = pd.date_range('2023-01-01', periods=10, freq='D')
        >>> true_values = np.array([100, 101, 103, 106, 110, 115, 121, 127, 135, 142])
        >>> forecast_values = np.array([99, 103, 105, 108, 112, 118, 123, 130, 136, 145])
        >>> 
        >>> true = TimeSeries.from_times_and_values(times, true_values)
        >>> forecast = TimeSeries.from_times_and_values(times, forecast_values)
        >>> 
        >>> # Calculate evaluation metrics
        >>> metrics = generate_evaluation_report(true, forecast)
        >>> print(metrics)
        {'rmse': 2.0, 'mae': 1.8, 'wmape': 1.52, 'directional_accuracy': 90.0, 'spearman_correlation': 0.9642857142857143, 'kendall_correlation': 0.8571428571428571, 'information_ratio': 2.5}
    """
    # Calculate traditional error metrics
    report = {
        "rmse": rmse(true_series, forecast_series),
        "mae": mae(true_series, forecast_series),
        "wmape": wmape(true_series, forecast_series),
    }
    
    # Calculate directional accuracy
    report["directional_accuracy"] = calculate_directional_accuracy(true_series, forecast_series)
    
    # Calculate directional-weighted MSE
    report["directional_weighted_mse"] = calculate_directional_weighted_mse(true_series, forecast_series)
    
    # Calculate rank correlations
    rank_correlations = calculate_rank_correlation(true_series, forecast_series)
    report.update(rank_correlations)
    
    # Calculate expectancy score metrics
    expectancy_metrics = calculate_expectancy_score(true_series, forecast_series)
    report.update(expectancy_metrics)
    
    # Calculate confusion matrix metrics
    confusion_metrics = calculate_confusion_metrics(true_series, forecast_series)
    report.update(confusion_metrics)
    
    logging.info(f"Evaluation Metrics: {report}")
    return report


def plot_forecast(
    true_series: TimeSeries,
    forecast_series: TimeSeries,
    title: str,
    output_path: str
) -> None:
    """
    Create and save a visualization of the forecasted vs. actual values.
    
    Parameters:
        true_series (TimeSeries): The ground truth time series with actual values
        forecast_series (TimeSeries): The forecasted time series from the model
        title (str): The title for the plot
        output_path (str): The file path where the plot image will be saved
    
    Returns:
        None
    
    Notes:
        - Uses matplotlib for creating the visualization
        - The plot includes both actual and forecasted lines with a legend
        - The x-axis represents the datetime index
        - The y-axis represents the target variable values
        - The saved image is in PNG format
    
    Example:
        >>> from darts import TimeSeries
        >>> import pandas as pd
        >>> import numpy as np
        >>> 
        >>> # Create sample data
        >>> dates = pd.date_range('2023-01-01', '2023-01-31', freq='D')
        >>> actual = np.sin(np.linspace(0, 4*np.pi, len(dates))) * 10 + 50
        >>> forecast = actual + np.random.normal(0, 1, len(dates))
        >>> 
        >>> # Convert to TimeSeries
        >>> actual_ts = TimeSeries.from_times_and_values(dates, actual)
        >>> forecast_ts = TimeSeries.from_times_and_values(dates, forecast)
        >>> 
        >>> # Create and save plot
        >>> plot_forecast(
        ...     actual_ts, 
        ...     forecast_ts, 
        ...     "Bitcoin Price Forecast - January 2023",
        ...     "./bitcoin_forecast_jan2023.png"
        ... )
    """
    plt.figure(figsize=(12, 6))
    true_series.plot(label="Actual")
    forecast_series.plot(label="Forecast")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()
    logging.info(f"Evaluation chart saved to {output_path}")


def run_evaluation(
    model_path: str,
    scaler_path: str,
    asset_id: str,
    test_start_date: str,
    test_end_date: str,
    db_kwargs: Dict[str, Union[int, str]],
    target_col: str,
    feature_groups: Dict[str, List[str]],
    output_path: str,
    study_name: str = None  # Add study_name parameter with default=None for backward compatibility
) -> None:
    """
    Evaluate a trained model on hold-out test data and save performance metrics and charts.
    
    This function orchestrates the complete evaluation workflow:
    1. Load the saved model and scaler
    2. Fetch test data from the database
    3. Preprocess the test data using the saved scaler
    4. Generate forecasts using the model
    5. Calculate performance metrics
    6. Create and save visualization charts
    
    Parameters:
        model_path (str): Path to the saved Darts model file (.pt)
        scaler_path (str): Path to the saved scikit-learn scaler file (.pkl)
        asset_id (str): The asset identifier (e.g., "BTC", "ETH")
        test_start_date (str): Start date of the test period in ISO format (YYYY-MM-DD)
        test_end_date (str): End date of the test period in ISO format (YYYY-MM-DD)
        db_kwargs (Dict[str, Union[int, str]]): Database connection parameters containing:
            - postgres_user (str): Database username
            - postgres_password (str): Database password
            - postgres_host (str): Database host address
            - postgres_port (int): Database port number
            - postgres_db (str): Database name
        target_col (str): The name of the target column to predict
        feature_groups (Dict[str, List[str]]): Dictionary mapping feature group names to lists of column names
        output_path (str): Directory where evaluation results will be saved
        study_name (str, optional): The name of the study that produced this model.
            Used for naming output files. If None, will use asset_id. Default: None.
    
    Returns:
        None
    """
    logging.info(f"Starting evaluation for model at '{model_path}'")
    os.makedirs(output_path, exist_ok=True)

    # 1. Load model and scaler with custom loss function
    model = load_model_with_custom_loss(model_path)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    # 2. Load data for the required period (test set + lookback window)
    # The model needs `input_chunk_length` of data prior to the test start date
    required_start_date = pd.to_datetime(test_start_date) - pd.DateOffset(days=model.input_chunk_length)
    
    # Use the same database loading approach as other modules
    full_df = datasets.load_features_from_db(asset_id, test_end_date, db_kwargs)
    eval_df = full_df.loc[required_start_date:]

    # 3. Prepare data
    all_feature_cols = [col for group in feature_groups.values() for col in group]
    target_series = TimeSeries.from_dataframe(eval_df[[target_col]], freq=eval_df.index.freq)
    
    features_df = eval_df[all_feature_cols]
    
    # Use the new function to handle transformation and DataFrame creation
    features_norm_df = transform_features_to_dataframe(
        scaler, features_df, feature_groups
    )
    
    covariates_series = TimeSeries.from_dataframe(features_norm_df, freq=eval_df.index.freq)

    # 4. Generate forecast - use fixed 90 days for test period
    logging.info(f"Generating forecast for 90 days from {test_start_date}")
    forecast_series = model.predict(
        n=90,  # Fixed 90-day prediction window
        past_covariates=covariates_series  # Match tuner.py approach (no target series)
    )

    # 5. Generate and save evaluation artifacts
    true_series_test = target_series.slice_intersect(forecast_series)
    
    # Use study_name for file naming if provided, otherwise fall back to asset_id
    file_prefix = study_name if study_name else asset_id
    
    report = generate_evaluation_report(true_series_test, forecast_series)
    report_path = os.path.join(output_path, f"{file_prefix}_evaluation_metrics.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
    
    chart_title = f"Forecast vs. Actuals for {asset_id} ({model.__class__.__name__})"
    chart_path = os.path.join(output_path, f"{file_prefix}_evaluation_plot.png")
    plot_forecast(true_series_test, forecast_series, chart_title, chart_path)

    logging.info("Evaluation complete.")
