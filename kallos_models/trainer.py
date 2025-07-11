"""
Model Training Module
====================

This module provides functionality for training the final model using the optimal
hyperparameters found during the tuning phase. It handles loading the complete dataset,
preprocessing features, training the model, and saving both the model and scaler for
later use in inference.

Example:
    from kallos_models import trainer
    
    # Train a final GRU model with optimal hyperparameters
    trainer.train_and_save_model(
        asset_id="BTC",
        end_date="2023-01-01",
        db_url="postgresql://user:pass@localhost:5432/crypto_db",
        model_name="gru",
        target_col="close",
        feature_groups={
            "volume_features": ["volume"],
            "bounded_features": ["rsi"]
        },
        optimal_hparams={
            "hidden_dim": 128,
            "n_rnn_layers": 2,
            "dropout": 0.2,
            "input_chunk_length": 60,
            "batch_size": 64,
            "optimizer_kwargs": {"lr": 0.001}
        },
        output_path="./models"
    )
"""

import logging
import os
import pickle
from typing import Dict, List, Tuple

import pandas as pd
from darts import TimeSeries
from darts.models.forecasting.forecasting_model import ForecastingModel
from sklearn.compose import ColumnTransformer

from . import architectures, datasets

# Set up a basic logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def train_and_save_model(
    asset_id: str,
    end_date: str,
    db_url: str,
    model_name: str,
    target_col: str,
    feature_groups: Dict[str, List[str]],
    optimal_hparams: Dict,
    output_path: str
) -> Tuple[ForecastingModel, ColumnTransformer]:
    """
    Train a final model on all available data and save both model and scaler to disk.
    
    This function represents the final step in the model development pipeline. It:
    1. Loads all available data up to the specified end date
    2. Creates and fits a feature transformer on the entire dataset
    3. Transforms features using the fitted transformer
    4. Instantiates a model with the optimal hyperparameters
    5. Trains the model on the full dataset
    6. Saves both the trained model and fitted scaler to disk for later inference
    
    Parameters:
        asset_id (str): The asset identifier (e.g., "BTC", "ETH")
        end_date (str): The end date for the data in ISO format (YYYY-MM-DD)
        db_url (str): The SQLAlchemy database URL for data loading
        model_name (str): The type of model to create (e.g., "gru", "lstm", "transformer")
        target_col (str): The name of the target column in the dataset
        feature_groups (Dict[str, List[str]]): Dictionary mapping feature group names to lists of column names
        optimal_hparams (Dict): Dictionary containing the optimal hyperparameters found during tuning
        output_path (str): Directory path where the model and scaler will be saved
    
    Returns:
        Tuple[ForecastingModel, ColumnTransformer]: The trained model and fitted scaler
    
    Raises:
        ValueError: If the specified output directory cannot be created
        IOError: If the model or scaler cannot be saved to disk
    
    Notes:
        - Unlike the tuning phase, the scaler is fitted on the entire dataset
        - Both the model (.pt file) and scaler (.pkl file) must be saved together
          as they need to be used in tandem for inference
    
    Example:
        model, scaler = train_and_save_model(
            asset_id="BTC",
            end_date="2023-01-01",
            db_url="postgresql://user:pass@localhost:5432/crypto_db",
            model_name="gru",
            target_col="close",
            feature_groups={
                "volume_features": ["volume", "taker_buy_base_asset_volume"],
                "bounded_features": ["rsi", "mfi"]
            },
            optimal_hparams={
                "hidden_dim": 128,
                "n_rnn_layers": 2,
                "dropout": 0.2,
                "batch_size": 64,
                "input_chunk_length": 60,
                "optimizer_kwargs": {"lr": 0.001}
            },
            output_path="./trained_models"
        )
    """
    logging.info(f"Starting final training for {model_name} on asset {asset_id}.")
    
    # 1. Load data
    full_df = datasets.load_features_from_db(asset_id, end_date, db_url)
    
    # 2. Separate target and features
    all_feature_cols = [col for group in feature_groups.values() for col in group]
    target_df = full_df[[target_col]]
    features_df = full_df[all_feature_cols]
    
    # 3. Create and fit scaler on the entire feature set
    scaler = datasets.create_feature_transformer(feature_groups)
    logging.info("Fitting scaler on the entire dataset...")
    scaler.fit(features_df)
    
    # 5. Transform features
    features_norm = scaler.transform(features_df)
    features_norm_df = pd.DataFrame(features_norm, index=features_df.index, columns=all_feature_cols)
    
    # 6. Convert to Darts TimeSeries
    target_series = TimeSeries.from_dataframe(target_df, freq=full_df.index.freq)
    covariates_series = TimeSeries.from_dataframe(features_norm_df, freq=full_df.index.freq)
    
    # 7. Instantiate model with optimal hyperparameters
    model = architectures.create_model(model_name, **optimal_hparams)
    
    # 8. Train the model on the full TimeSeries
    logging.info("Training final model on the full dataset...")
    model.fit(target_series, past_covariates=covariates_series, verbose=True)
    
    # 9. Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # 10. Define file paths
    model_filename = f"{model_name}_{asset_id}.pt" # Darts saves PyTorch models with .pt
    scaler_filename = f"{model_name}_{asset_id}_scaler.pkl"
    model_filepath = os.path.join(output_path, model_filename)
    scaler_filepath = os.path.join(output_path, scaler_filename)
    
    # 11. Save the Darts model
    model.save(model_filepath)
    
    # 12. Save the fitted scaler
    with open(scaler_filepath, 'wb') as f:
        pickle.dump(scaler, f)
        
    logging.info(f"Model saved to: {model_filepath}")
    logging.info(f"Scaler saved to: {scaler_filepath}")
    
    return model, scaler
