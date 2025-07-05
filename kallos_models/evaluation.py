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
    """Calculates a dictionary of key performance indicators.

    Args:
        true_series (TimeSeries): The ground truth time series.
        forecast_series (TimeSeries): The forecasted time series.

    Returns:
        Dict[str, float]: A dictionary containing RMSE, MAE, and MAPE scores.
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
    """Plots the true vs. forecasted series and saves the chart.

    Args:
        true_series (TimeSeries): The ground truth time series.
        forecast_series (TimeSeries): The forecasted time series.
        title (str): The title for the chart.
        output_path (str): The file path to save the plot image.
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
    """Orchestrates the model evaluation process on a hold-out test set.

    Args:
        model_path (str): Path to the saved Darts model file.
        scaler_path (str): Path to the saved scikit-learn scaler file.
        asset_id (str): The asset identifier.
        test_start_date (str): The start date of the test period (YYYY-MM-DD).
        test_end_date (str): The end date of the test period (YYYY-MM-DD).
        db_url (str): The database connection URL.
        target_col (str): The name of the target column.
        feature_groups (Dict[str, List[str]]): The feature group dictionary.
        output_path (str): The directory to save evaluation artifacts.
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
