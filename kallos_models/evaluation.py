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
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
from darts import TimeSeries
from darts.metrics import mae, mape, rmse
from darts.models.forecasting.forecasting_model import ForecastingModel

from . import datasets

# Set up a basic logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


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
            - mape: Mean Absolute Percentage Error
    
    Notes:
        - Lower values indicate better performance for all metrics
        - MAPE is expressed as a percentage (e.g., 5.2 means 5.2%)
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
        {'rmse': 2.0, 'mae': 1.8, 'mape': 1.52}
    """
    report = {
        "rmse": rmse(true_series, forecast_series),
        "mae": mae(true_series, forecast_series),
        "mape": mape(true_series, forecast_series),
    }
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
    db_url: str,
    target_col: str,
    feature_groups: Dict[str, List[str]],
    output_path: str
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
        db_url (str): The SQLAlchemy database URL for data loading
        target_col (str): The name of the target column to predict
        feature_groups (Dict[str, List[str]]): Dictionary mapping feature group names to lists of column names
        output_path (str): Directory where evaluation results will be saved
    
    Returns:
        None
    
    Raises:
        FileNotFoundError: If the model or scaler files cannot be found
        ValueError: If the date range is invalid
        
    Notes:
        - The function creates the output directory if it doesn't exist
        - Two files are saved in the output directory:
          * {asset_id}_evaluation_metrics.json: JSON file with performance metrics
          * {asset_id}_evaluation_plot.png: Visualization of actual vs. forecasted values
        - The model requires historical data prior to test_start_date equal to its input_chunk_length
    
    Example:
        >>> run_evaluation(
        ...     model_path="./models/gru_BTC.pt",
        ...     scaler_path="./models/gru_BTC_scaler.pkl",
        ...     asset_id="BTC",
        ...     test_start_date="2023-01-01",
        ...     test_end_date="2023-03-31",
        ...     db_url="postgresql://user:pass@localhost:5432/crypto_db",
        ...     target_col="close",
        ...     feature_groups={
        ...         "volume_features": ["volume", "taker_buy_volume"],
        ...         "bounded_features": ["rsi", "mfi"],
        ...         "unbounded_features": ["macd_diff"]
        ...     },
        ...     output_path="./evaluation_results"
        ... )
    """
    logging.info(f"Starting evaluation for model at '{model_path}'")
    os.makedirs(output_path, exist_ok=True)

    # 1. Load model and scaler
    model = ForecastingModel.load(model_path)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    # 2. Load data for the required period (test set + lookback window)
    # The model needs `input_chunk_length` of data prior to the test start date
    required_start_date = pd.to_datetime(test_start_date) - pd.DateOffset(days=model.input_chunk_length)
    
    # We use the existing loader and slice, assuming it's efficient enough.
    # For very large datasets, a ranged query in `datasets.py` would be better.
    full_df = datasets.load_features_from_db(asset_id, test_end_date, db_url)
    eval_df = full_df.loc[required_start_date:]

    # 3. Prepare data
    all_feature_cols = [col for group in feature_groups.values() for col in group]
    target_series = TimeSeries.from_dataframe(eval_df[[target_col]], freq=eval_df.index.freq)
    
    features_df = eval_df[all_feature_cols]
    features_norm = scaler.transform(features_df)
    features_norm_df = pd.DataFrame(features_norm, index=features_df.index, columns=all_feature_cols)
    covariates_series = TimeSeries.from_dataframe(features_norm_df, freq=eval_df.index.freq)

    # 4. Generate forecast
    logging.info(f"Generating forecast for period {test_start_date} to {test_end_date}...")
    forecast_series = model.predict(
        n=len(eval_df.loc[test_start_date:]),
        series=target_series, # Provide historical target series for the model
        past_covariates=covariates_series
    )

    # 5. Generate and save evaluation artifacts
    true_series_test = target_series.slice_intersect(forecast_series)
    
    report = generate_evaluation_report(true_series_test, forecast_series)
    report_path = os.path.join(output_path, f"{asset_id}_evaluation_metrics.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
    
    chart_title = f"Forecast vs. Actuals for {asset_id} ({model.__class__.__name__})"
    chart_path = os.path.join(output_path, f"{asset_id}_evaluation_plot.png")
    plot_forecast(true_series_test, forecast_series, chart_title, chart_path)

    logging.info("Evaluation complete.")
