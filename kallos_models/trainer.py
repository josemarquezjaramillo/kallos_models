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
    """Trains a final model on all data and saves the model and scaler.

    Args:
        asset_id (str): The asset identifier.
        end_date (str): The final date for the data.
        db_url (str): The database connection URL.
        model_name (str): The name of the model to train.
        target_col (str): The name of the target column.
        feature_groups (Dict[str, List[str]]): The feature group dictionary.
        optimal_hparams (Dict): The dictionary of best hyperparameters.
        output_path (str): The directory path to save the artifacts.

    Returns:
        Tuple[ForecastingModel, ColumnTransformer]: The trained model and fitted scaler.
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
